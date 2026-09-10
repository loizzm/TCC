"""SONDAGEM: o encoder JA PROMOVIDO carrega o sinal de multi-degrau?

Esta e a medicao que decide se o retreino conjunto e necessario, e ela custa
minutos em vez das ~8 h de um treino completo.

O metodo: congelar `models/unet_stageA.pt` inteiro, extrair a ativacao do
GARGALO de cada amostra UMA vez, e treinar so a cabeca de contagem sobre essas
features cacheadas. Duas propriedades tornam isso barato e seguro:

  - a segmentacao NAO PODE regredir, porque os pesos dela nao se movem;
  - o forward roda uma vez por amostra, nao uma vez por epoca. A cabeca
    consome `gargalo.mean(dim=2)`, de forma (C, W'), que e ~64 KB por amostra
    contra 2 MB do gargalo inteiro — o cache cabe na memoria e o treino da
    cabeca leva segundos.

MEDE-SE POR AUC, NAO POR ACERTO. A primeira versao deste arquivo media acerto
e concluiu "as features nao carregam o sinal" — CONCLUSAO ERRADA, e o defeito
era do experimento. O estrato tem 81 % de multi-degrau, entao a cabeca colapsa
na classe maioritaria (revocacao 100 %, precisao 79,5 %) e o acerto fica colado
na taxa-base sem que isso diga nada sobre o sinal. Corrigido com `pos_weight` e
AUC, a mesma cabeca sobre as MESMAS features da AUC 0,88. Acerto e uma metrica
cega em populacao desbalanceada; AUC nao e.

Se a AUC ficar bem acima de 0,5, as features atuais ja servem e o retreino
conjunto pode ser dispensavel para esta tarefa. Se ficar perto de 0,5, o sinal
nao esta nas features e o encoder precisa ser retreinado — e ai a sondagem
pagou por si, porque evitou descobrir isso depois de 8 h de GPU.

Uso:
    .venv/bin/python sonda_cabeca_contagem.py [--epocas 60] [--lr 1e-3]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_auc_score

from identify.extract import UNet, letterbox, load_model

RAIZ = Path(__file__).resolve().parent


@torch.no_grad()
def extrai(modelo, dirs, device, size=512):
    """Gargalo com o eixo Y ja colapsado — exatamente o que a cabeca consome."""
    X, Y = [], []
    amostras = [d for r in dirs for d in sorted(Path(r).glob("sample_*"))]
    for i, d in enumerate(amostras):
        meta = json.loads((d / "meta.json").read_text())
        img = np.asarray(Image.open(d / "image.png").convert("RGB"))
        small, _ = letterbox(np.ascontiguousarray(img[..., :3]), size)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(device)
        g = modelo.gargalo(x).mean(dim=2)          # (1, C, W')
        X.append(g[0].cpu().numpy().astype(np.float32))
        Y.append(1.0 if int(meta.get("n_degraus", 1)) > 1 else 0.0)
        if i % 200 == 0:
            print(f"    {i}/{len(amostras)}", flush=True)
    return np.stack(X), np.array(Y, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modelo", default="models/unet_stageA.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / a.modelo), dev)
    for p in modelo.parameters():
        p.requires_grad = False
    modelo.eval()

    cache = RAIZ / "reports" / "sonda_contagem_features.npz"
    if cache.exists():
        z = np.load(cache)
        Xtr, ytr, Xva, yva = z["Xtr"], z["ytr"], z["Xva"], z["yva"]
        print(f"features do cache: treino {Xtr.shape}, val {Xva.shape}")
    else:
        t0 = time.time()
        print("  extraindo features de treino...")
        Xtr, ytr = extrai(modelo, ["data/train_multi"], dev)
        print("  extraindo features de validacao...")
        Xva, yva = extrai(modelo, ["data/val_multi"], dev)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
        print(f"  {time.time()-t0:.0f}s · cache em {cache}")

    base = float(max(yva.mean(), 1 - yva.mean()))
    print(f"\n  treino n={len(ytr)} ({ytr.mean():.1%} multi) · "
          f"val n={len(yva)} ({yva.mean():.1%} multi)")
    print(f"  TAXA-BASE (chutar sempre a classe maioritaria): {base:.1%}\n")

    # A cabeca vem do PROPRIO modelo, nao reimplementada aqui: uma copia
    # divergiria em silencio da que vai para producao, e a sondagem passaria a
    # medir outra coisa. `UNet(com_contagem=True)` so para pegar o modulo.
    torch.manual_seed(0)
    C = Xtr.shape[1]
    cabeca = UNet(base=C // 16, levels=4, in_ch=3,
                  com_contagem=True).cabeca_conta.to(dev)
    opt = torch.optim.AdamW(cabeca.parameters(), lr=a.lr)
    # Sem `pos_weight` a cabeca colapsa na classe maioritaria — ver a docstring.
    pw = torch.tensor([float((1.0 - ytr.mean()) / max(ytr.mean(), 1e-6))], device=dev)
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr)[:, None].to(dev)
    Xva_t = torch.from_numpy(Xva).to(dev)
    yva_t = torch.from_numpy(yva)[:, None].to(dev)

    melhor = {"auc": -1.0}
    for ep in range(a.epocas):
        cabeca.train()
        perm = torch.randperm(len(ytr_t), device=dev)
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                cabeca(Xtr_t[idx]), ytr_t[idx], pos_weight=pw)
            loss.backward()
            opt.step()
        cabeca.eval()
        with torch.no_grad():
            lo = cabeca(Xva_t).cpu().numpy().ravel()
        auc = float(roc_auc_score(yva, lo))
        if auc > melhor["auc"]:
            melhor = {"auc": auc, "ep": ep, "lo": lo.copy()}
        if ep % 10 == 0 or ep == a.epocas - 1:
            print(f"  epoca {ep:03d}  AUC_val={auc:.4f}  perda={float(loss):.4f}")

    pr, rc, _ = precision_recall_curve(yva, melhor["lo"])
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    i = int(np.nanargmax(f1))
    print(f"\n  MELHOR AUC = {melhor['auc']:.4f} na epoca {melhor['ep']}")
    print(f"  {'criterio':<28}{'precisao':>10}{'revocacao':>11}{'F1':>8}")
    print(f"  {'melhor F1':<28}{pr[i]:>10.1%}{rc[i]:>11.1%}{f1[i]:>8.3f}")
    for alvo in (0.90, 0.95, 0.98):
        k = np.flatnonzero(pr >= alvo)
        if k.size:
            j = k[int(np.argmax(rc[k]))]
            print(f"  {'precisao >= ' + f'{alvo:.0%}':<28}{pr[j]:>10.1%}{rc[j]:>11.1%}"
                  f"{2*pr[j]*rc[j]/max(pr[j]+rc[j],1e-9):>8.3f}")
    print()
    print("  Referencias medidas neste projeto:")
    print("    gate de ganho (heuristica de residuo)   F1 0,740")
    print("    detector de derivada na curva IDEAL     F1 0,923")
    print("    contagem perfeita como portao           F1 0,825 no fim da pipeline")
    print()
    if melhor["auc"] > 0.80:
        print("  VEREDITO: as features do encoder JA PROMOVIDO carregam o sinal,")
        print("  mesmo nunca tendo visto uma figura multi-degrau no treino. A")
        print("  cabeca pode ser treinada com o encoder CONGELADO — sem retreino")
        print("  conjunto, e sem risco NENHUM para a segmentacao.")
    elif melhor["auc"] > 0.65:
        print("  VEREDITO: sinal parcial. A cabeca congelada ajuda mas nao basta;")
        print("  o treino conjunto provavelmente vale as ~8 h.")
    else:
        print("  VEREDITO: as features atuais NAO carregam o sinal. O encoder")
        print("  precisa ser retreinado com o estrato multi-degrau.")


if __name__ == "__main__":
    main()
