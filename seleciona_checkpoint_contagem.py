#!/usr/bin/env python3
"""Escolhe o checkpoint do retreino multi-degrau pelas DUAS metricas, e a de
contagem medida no CORPUS REAL — nao no sintetico.

Por que este script existe, e por que ele nao e o `seleciona_checkpoint.py`.
Aquele mede cobertura do PLATO, o defeito que o retreino de ganho negativo
existia para consertar. Aqui o alvo e outro e as armadilhas sao duas:

  1. `train_unet.py` guarda o melhor por `IoU_val`, que nao mede contagem de
     jeito nenhum. Selecionar por ele e selecionar pelo instrumento errado —
     a mesma licao do §40.9.

  2. O `acerto_contagem` que o log imprime e IN-DISTRIBUTION, em
     `data/val_multi`. Medido nesta sessao, ele MENTE: a cabeca de encoder
     congelado deu AUC 0,97 no sintetico e 0,59 no corpus real, em tres
     variantes de corpus seguidas. Um checkpoint escolhido pelo numero
     sintetico tem chance real de ser o pior no real.

Por isso a metrica de contagem aqui e a AUC nas 299 figuras de
`reports/amostras_aleatorias` — figuras de render REAL, fora da distribuicao
de treino. E a segmentacao entra como GUARDA: um checkpoint que sobe a
contagem derrubando o IoU nao aprendeu a tarefa nova, trocou uma pela outra.

Uso:
    .venv/bin/python seleciona_checkpoint_contagem.py models/epocas_multi \\
        [--referencia models/unet_stageA.pt] [--limite-iou 0.02]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from identify.extract import UNet, letterbox, load_model
from train_unet import MaskDataset, iou
from torch.utils.data import DataLoader

RAIZ = Path(__file__).resolve().parent
REAL = RAIZ / "reports" / "amostras_aleatorias"
LOTES_REAIS = (REAL, REAL / "balanceado", REAL / "balanceado2")


def figuras_reais():
    itens = []
    for d in LOTES_REAIS:
        f = d / "verdade.json"
        if not f.exists():
            continue
        for v in json.loads(f.read_text()):
            p = d / v["arquivo"]
            if p.exists():
                itens.append((p, 1.0 if v["n_degraus"] > 1 else 0.0))
    return itens


@torch.no_grad()
def auc_contagem_real(modelo, itens, dev):
    """AUC de multi-degrau nas figuras reais. `nan` se o modelo nao tem cabeca."""
    if getattr(modelo, "cabeca_conta", None) is None:
        return float("nan")
    esc, y = [], []
    for p, rot in itens:
        img = np.asarray(Image.open(p).convert("RGB"))
        small, _ = letterbox(np.ascontiguousarray(img[..., :3]), 512)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(dev)
        m, c = modelo(x, com_contagem=True)
        esc.append(float(c[0, 0]))
        y.append(rot)
    return float(roc_auc_score(y, esc))


@torch.no_grad()
def iou_val(modelo, dirs, dev, batch=6, limite=None):
    dl = DataLoader(MaskDataset(dirs, 512, in_ch=3), batch_size=batch, num_workers=2)
    vs = []
    for i, (x, y, _n, _w) in enumerate(dl):
        if limite and i >= limite:
            break
        vs.append(iou(modelo(x.to(dev)), y.to(dev)))
    return float(np.mean(vs)) if vs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--referencia", type=Path,
                    default=RAIZ / "models" / "unet_stageA.pt",
                    help="checkpoint promovido, para a guarda de regressao")
    ap.add_argument("--val-dir", action="append",
                    default=None, help="default: data/val + data/val_multi")
    ap.add_argument("--limite-iou", type=float, default=0.02,
                    help="queda de IoU tolerada contra a referencia")
    ap.add_argument("--batches-iou", type=int, default=40,
                    help="batches de validacao por checkpoint (custo x ruido)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vdirs = a.val_dir or ["data/val", "data/val_multi"]
    itens = figuras_reais()
    print(f"  {len(itens)} figuras reais para a AUC de contagem")

    ref = load_model(str(a.referencia), dev)
    iou_ref = iou_val(ref, vdirs, dev, limite=a.batches_iou)
    print(f"  referencia {a.referencia.name}: IoU_val={iou_ref:.4f}"
          f"  (sem cabeca de contagem)\n")

    linhas = []
    for ck in sorted(a.ckpt_dir.glob("epoca_*.pt")):
        m = load_model(str(ck), dev)
        i_ = iou_val(m, vdirs, dev, limite=a.batches_iou)
        au = auc_contagem_real(m, itens, dev)
        ok = (i_ >= iou_ref - a.limite_iou)
        linhas.append({"ckpt": ck.name, "iou": i_, "auc_real": au, "guarda_ok": ok})
        print(f"  {ck.name}  IoU_val={i_:.4f}{'' if ok else '  <-- REGREDIU'}"
              f"  AUC_contagem_REAL={au:.4f}", flush=True)

    # Nome derivado do diretorio de checkpoints. Era fixo em
    # `selecao_multi.json`, e rodar este script num segundo retreino
    # sobrescrevia em silencio o registro do primeiro.
    (RAIZ / "reports" / f"selecao_{a.ckpt_dir.name}.json").write_text(
        json.dumps({"iou_referencia": iou_ref, "checkpoints": linhas}, indent=2))

    aptos = [x for x in linhas if x["guarda_ok"] and np.isfinite(x["auc_real"])]
    print()
    if not aptos:
        print("  NENHUM checkpoint passa a guarda de segmentacao. Nao promover.")
        return
    melhor = max(aptos, key=lambda x: x["auc_real"])
    print(f"  VENCEDOR: {melhor['ckpt']}  "
          f"AUC_contagem_REAL={melhor['auc_real']:.4f}  IoU_val={melhor['iou']:.4f}")
    print(f"  ({len(aptos)}/{len(linhas)} passaram a guarda de IoU)")
    print()
    print("  Referencias medidas nas MESMAS 299 figuras:")
    print("    gate de residuo (producao)      F1 0,571")
    print("    gate de residuo (sem piso)      F1 0,740")
    print("    cabeca com encoder CONGELADO    AUC real 0,59 a 0,65")
    print()
    if melhor["auc_real"] < 0.75:
        print("  ATENCAO: AUC real abaixo de 0,75. O treino conjunto NAO superou")
        print("  o patamar do encoder congelado de forma material — promover a")
        print("  cabeca nao se justifica, e a heuristica de ganho continua melhor.")
    else:
        print("  A AUC real supera o patamar do encoder congelado. Vale avaliar a")
        print("  promocao, medindo o efeito PONTA A PONTA na pipeline antes.")


if __name__ == "__main__":
    main()
