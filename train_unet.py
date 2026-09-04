"""Treino do Estágio A. Determinístico: seed fixa, sem RNG global."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.generator import load_sample
from identify.extract import UNet, dice_bce_loss, letterbox


class MaskDataset(Dataset):
    # HISTÓRICO (ver HANDOFF_P2_3.md, Rulings 7/8): o alvo binário depois do
    # `letterbox` teve três versões — limiar 127 (perdia a curva quase toda
    # em imagens grandes, cobertura de colunas caía a 0,4%), limiar 0
    # (recuperava a presença mas inflava a ÁREA do alvo até 3,03x, piorando
    # o IoU de teste de 0,572 para 0,495 apesar do IoU de validação subir
    # pra 0,91) e limiar 32 (equilíbrio medido por varredura: cobertura
    # 85,1%, inflação 2,14x — melhor resultado das quatro rodadas de treino,
    # IoU de teste 0,560, mas ζ em 2.6 ainda ficava 0,64 p.p. acima do alvo).
    #
    # Em vez de continuar caçando um QUARTO limiar mágico, este alvo é
    # CONTÍNUO: usa o valor real que sai do `cv2.INTER_AREA` (0 a 255,
    # normalizado para 0,0-1,0), sem nenhuma binarização. `dice_bce_loss`
    # (BCE + Dice) aceita alvo contínuo nativamente — é a definição
    # matemática usual dos dois, sem mudança de código lá. A vantagem sobre
    # qualquer limiar fixo: uma caixa do downscale com pouca cobertura de
    # curva vira um alvo baixo mas não-zero (preserva o gradiente de BCE,
    # resolvendo o sumiço do limiar 127 sem escolher limiar nenhum), e uma
    # caixa com cobertura parcial não empurra a rede a prever confiança
    # alta ali (limitando a inflação de área do limiar 0/32 pela raiz, não
    # por um corte arbitrário).
    def __init__(self, root: str | list[str], size: int = 512, in_ch: int = 1):
        roots = [root] if isinstance(root, str) else list(root)
        self.dirs = [d for r in roots for d in sorted(Path(r).glob("sample_*"))]
        self.size = size
        self.in_ch = int(in_ch)

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, i: int):
        # `in_ch=3` entrega RGB. A conversao para cinza e DESTRUTIVA: projeta
        # R^3 em R^1, e dois objetos de luminancia igual viram o MESMO byte —
        # medido nas duas imagens reais do Ruling 55, curva (44,160,44) e reta
        # de referencia (230,61,61) dao ambas 112. Com 1 canal a tarefa
        # "separe a curva da reta" nao e dificil, e impossivel. O caminho de
        # 1 canal fica identico ao anterior.
        m = load_sample(self.dirs[i])
        if self.in_ch == 3:
            ent = np.ascontiguousarray(m["image"][..., :3])
            x, _ = letterbox(ent, self.size)
            xt = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1)
        else:
            w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            gray = (m["image"].astype(np.float32) @ w).round().astype(np.uint8)
            x, _ = letterbox(gray, self.size)
            xt = torch.from_numpy(x.astype(np.float32) / 255.0)[None]
        y, _ = letterbox(m["mask"], self.size)
        return xt, torch.from_numpy(y.astype(np.float32) / 255.0)[None]


def iou(logits, target, thr: float = 0.5) -> float:
    # `target` agora pode ser contínuo (MaskDataset) — binariza aqui dentro
    # pra manter o IoU de validação interpretável como métrica de progresso
    # (mesmo alvo binário que as rodadas anteriores usavam para reportar,
    # só que agora o TREINO em si não vê essa binarização).
    p = (torch.sigmoid(logits) >= thr).float()
    t = (target >= thr).float()
    inter = (p * t).sum(dim=(1, 2, 3))
    union = ((p + t) >= 1).float().sum(dim=(1, 2, 3))
    return float((inter / union.clamp(min=1.0)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="models/unet_stageA.pt")
    # Hipotese (a) do Ruling 10 (HANDOFF_P2_3): capacidade da rede. `base=16`
    # e o que as cinco rodadas usaram; 24 e 32 sao as alternativas medidas la.
    ap.add_argument("--val-dir", action="append", default=None,
                    help="repeticoes acumulam. Default: data/val. Incluir o "
                         "estrato aqui faz a selecao do melhor checkpoint "
                         "levar em conta o fenomeno que ele reproduz.")
    ap.add_argument("--in-ch", type=int, default=1, choices=(1, 3),
                    help="1 = cinza (comportamento anterior); 3 = RGB. Ver "
                         "MaskDataset.__getitem__ para o porque.")
    ap.add_argument("--base", type=int, default=16,
                    help="canais da primeira camada da UNet (16, 24, 32)")
    # Hipotese (b) do Ruling 10: tamanho do dataset. Aceita mais de um
    # diretorio para somar um split extra (seed-base >= 4) ao data/train atual
    # sem tocar em data/val e data/test, que ficam fixos para os numeros
    # continuarem comparaveis com as rodadas 1-5.
    ap.add_argument("--train-dir", action="append", default=None,
                    help="diretorio de treino (repetivel; padrao: data/train)")
    # Sem isto, comparar 4.200 com 8.400 amostras mudaria DUAS variaveis de uma
    # vez (diversidade de dados E numero de passos de gradiente por epoca).
    # Fixando os passos, a unica diferenca entre as duas curvas de IoU_val e a
    # diversidade -- que e exatamente a hipotese (b) do Ruling 10.
    ap.add_argument("--batches-per-epoch", type=int, default=0,
                    help="limita os passos de gradiente por epoca (0 = split inteiro)")
    # Scheduler de LR — ver Ruling no HANDOFF_P2_3.md: sem ele, o treino sobe
    # rápido nas 3 primeiras épocas e depois oscila em torno de um platô
    # (IoU_val 0,65-0,67 por 10+ épocas, sem tendência de melhora) porque o
    # passo de otimização fica grande demais para refinar perto do mínimo.
    # ReduceLROnPlateau reage à métrica que de fato importa (IoU de
    # validação), reduzindo o LR só quando ela para de melhorar — não segue
    # um cronograma fixo que teria de ser adivinhado de antemão.
    ap.add_argument("--lr-patience", type=int, default=1,
                    help="épocas sem melhora de IoU_val antes de reduzir o LR")
    ap.add_argument("--lr-factor", type=float, default=0.5,
                    help="fator de redução do LR quando o platô dispara")
    ap.add_argument("--lr-threshold", type=float, default=0.01,
                    help="ganho mínimo de IoU_val para contar como melhora real")
    # O `IoU_val` e quase CEGO ao defeito do plato (§40.9): o estrato que a rede
    # nao sabe segmentar custa 5 pontos de IoU (0,7814 -> 0,7304) enquanto a
    # cobertura do plato nele desaba 42 (0,944 -> 0,527). O plato e uma linha
    # FINA, poucos pixels contra o corpo da curva, e o IoU e dominado pelo corpo.
    # Como a selecao do melhor checkpoint e por `m > melhor` sobre o `IoU_val`,
    # a epoca que aprender o plato pode nao ser a selecionada.
    #
    # Guardar TODAS e escolher depois resolve isso sem tocar em metrica nenhuma:
    # o `IoU_val` continua sendo calculado e reportado igual, para os numeros
    # seguirem comparaveis com as rodadas historicas. Opt-in: sem o flag o
    # comportamento e byte a byte o de antes.
    ap.add_argument("--save-epoch-dir", default=None,
                    help="se dado, salva `epoca_NN.pt` neste diretorio a cada "
                         "epoca, ALEM do melhor por IoU_val em --out. Para "
                         "selecionar por metrica que o IoU_val nao enxerga.")
    a = ap.parse_args()

    torch.manual_seed(20260817)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dir_epocas = Path(a.save_epoch_dir) if a.save_epoch_dir else None
    if dir_epocas is not None:
        dir_epocas.mkdir(parents=True, exist_ok=True)
    train_dirs = a.train_dir or ["data/train"]
    ds_tr = MaskDataset(train_dirs, a.size, in_ch=a.in_ch)
    passos = a.batches_per_epoch or (len(ds_tr) // a.batch)
    print(f"treino: {len(ds_tr)} amostras de {train_dirs}  base={a.base}  "
          f"{passos} passos/epoca", flush=True)
    tr = DataLoader(ds_tr, batch_size=a.batch,
                    shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(MaskDataset(a.val_dir or ["data/val"], a.size, in_ch=a.in_ch), batch_size=a.batch,
                    num_workers=2)
    model = UNet(base=a.base, in_ch=a.in_ch).to(a.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parametros: {n_par}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=a.lr_factor, patience=a.lr_patience,
        threshold=a.lr_threshold, threshold_mode="abs")
    melhor = -1.0
    for ep in range(a.epochs):
        t0 = time.perf_counter()
        model.train()
        for nb, (x, y) in enumerate(tr):
            if a.batches_per_epoch and nb >= a.batches_per_epoch:
                break
            x, y = x.to(a.device), y.to(a.device)
            opt.zero_grad()
            loss = dice_bce_loss(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ious = [iou(model(x.to(a.device)), y.to(a.device)) for x, y in va]
        m = float(np.mean(ious))
        lr_antes = opt.param_groups[0]["lr"]
        sched.step(m)
        lr_depois = opt.param_groups[0]["lr"]
        marca = " (LR reduzido)" if lr_depois < lr_antes else ""
        print(f"epoca {ep:02d}  IoU_val={m:.4f}  lr={lr_depois:.2e}{marca}  "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        if dir_epocas is not None:
            torch.save(model.state_dict(), dir_epocas / f"epoca_{ep:02d}.pt")
        if m > melhor:
            melhor = m
            torch.save(model.state_dict(), a.out)
    print(f"melhor IoU_val={melhor:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
