#!/usr/bin/env python3
"""Escolhe o checkpoint do retreino FORA DA FAMILIA (§65) pelo objetivo certo.

POR QUE NAO REUSAR OS OUTROS DOIS SELETORES. `helpers/seleciona_checkpoint.py` mede
cobertura do plato em `data/val` — a metrica do retreino de ganho negativo.
e o seletor da frente de multi-degrau media AUC de contagem no corpus real.
Nenhum dos dois olha para o defeito que ESTE retreino existe para consertar, e
promover pelo instrumento do retreino anterior e' o erro que o §40.9 ja
registrou uma vez. O de contagem saiu do repositorio com a frente inteira
(§68, MULTI_DEGRAU.md); o que sobrou aqui e a guarda de segmentacao.

O OBJETIVO e' a cobertura do plato de repouso em `data/val_nmp`: das colunas em
que a figura desenha a curva em repouso, quantas a mascara acende. Baseline
medido com `models/unet_stageA.pt`: 0,286 (contra 0,884 em `data/val`).

AS GUARDAS sao duas, e cada uma protege um ganho que ja foi pago:
  * cobertura do plato em `data/val` — o retreino de ganho negativo;
  * IoU em `data/val` + `data/val_multi` — a segmentacao em geral.
Um checkpoint que sobe o objetivo derrubando qualquer uma delas nao aprendeu a
tarefa nova: trocou uma pela outra. Nao promover.

NAO SELECIONE PELO `IoU_val` DO LOG. Ele mistura seis populacoes e ja
anticorrelacionou com a metrica real neste projeto (Spearman -0,401 depois da
epoca 08 do retreino multi): `train_unet.py` teria escolhido o checkpoint pior.

Uso:
    .venv/bin/python helpers/seleciona_checkpoint_nmp.py models/epocas_nmp \\
        [--referencia models/unet_stageA.pt]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import torch as _torch
from torch.utils.data import DataLoader

from identify.extract import load_model
from mede_plato_repouso import cobertura_do_plato
from train_unet import MaskDataset, iou

RAIZ = Path(__file__).resolve().parent.parent


@_torch.no_grad()
def iou_val(modelo, dirs, dev, batch: int = 6, limite=None) -> float:
    """IoU medio em `dirs`. Vivia em `seleciona_checkpoint_contagem.py`, que
    saiu com a frente de multi-degrau (§68)."""
    dl = DataLoader(MaskDataset(dirs, 512, in_ch=3), batch_size=batch, num_workers=2)
    vs = []
    for i, (x, y) in enumerate(dl):
        if limite and i >= limite:
            break
        vs.append(iou(modelo(x.to(dev)), y.to(dev)))
    return float(np.mean(vs)) if vs else float("nan")


def avalia(modelo, dev, a) -> dict:
    """Objetivo + as tres guardas, num checkpoint."""
    modelo.eval()
    alvo = cobertura_do_plato(RAIZ / "data" / "val_nmp", modelo, dev, lim=a.lim_plato)
    base = cobertura_do_plato(RAIZ / "data" / "val", modelo, dev, lim=a.lim_plato)
    return {
        "plato_nmp": float(np.median(alvo)) if alvo.size else float("nan"),
        "plato_val": float(np.median(base)) if base.size else float("nan"),
        "iou": iou_val(modelo, a.val_dir, dev, limite=a.batches_iou),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--referencia", type=Path,
                    default=RAIZ / "models" / "unet_stageA.pt")
    ap.add_argument("--val-dir", action="append", default=None,
                    help="para o IoU. default: data/val + data/val_multi")
    ap.add_argument("--lim-plato", type=int, default=120,
                    help="amostras por corpus na cobertura do plato")
    ap.add_argument("--batches-iou", type=int, default=40)
    ap.add_argument("--limite-iou", type=float, default=0.02,
                    help="queda de IoU tolerada contra a referencia")
    ap.add_argument("--limite-plato", type=float, default=0.03,
                    help="queda tolerada na cobertura do plato de data/val")
    a = ap.parse_args()
    a.val_dir = a.val_dir or ["data/val", "data/val_multi"]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = avalia(load_model(str(a.referencia), dev), dev, a)
    print(f"  referencia {a.referencia.name}: "
          f"plato_nmp={ref['plato_nmp']:.3f}  plato_val={ref['plato_val']:.3f}  "
          f"IoU={ref['iou']:.4f}")
    print()

    linhas = []
    for ck in sorted(a.ckpt_dir.glob("epoca_*.pt")):
        m = avalia(load_model(str(ck), dev), dev, a)
        g_iou = m["iou"] >= ref["iou"] - a.limite_iou
        g_plato = m["plato_val"] >= ref["plato_val"] - a.limite_plato
        m.update(ckpt=ck.name, guarda_iou=g_iou, guarda_plato=g_plato,
                 apto=bool(g_iou and g_plato))
        linhas.append(m)
        falhou = [n for n, ok in (("IoU", g_iou), ("plato_val", g_plato))
                  if not ok]
        print(f"  {ck.name}  plato_nmp={m['plato_nmp']:.3f}  "
              f"plato_val={m['plato_val']:.3f}  IoU={m['iou']:.4f}"
              + (f"  <-- REGREDIU: {', '.join(falhou)}" if falhou else ""),
              flush=True)

    (RAIZ / "reports" / f"selecao_{a.ckpt_dir.name}.json").write_text(
        json.dumps({"referencia": ref, "checkpoints": linhas}, indent=2))

    aptos = [x for x in linhas if x["apto"]]
    print()
    if not aptos:
        print("  NENHUM checkpoint passa as guardas. NAO PROMOVER — o treino")
        print("  trocou um ganho por outro em vez de somar.")
        return
    melhor = max(aptos, key=lambda x: x["plato_nmp"])
    print(f"  VENCEDOR: {melhor['ckpt']}  plato_nmp={melhor['plato_nmp']:.3f} "
          f"(referencia {ref['plato_nmp']:.3f})  "
          f"plato_val={melhor['plato_val']:.3f}  IoU={melhor['iou']:.4f}")
    print(f"  ({len(aptos)}/{len(linhas)} passaram as guardas)")
    if melhor["plato_nmp"] <= ref["plato_nmp"] + 0.05:
        print()
        print("  ATENCAO: o objetivo mal se moveu. O estrato entrou no treino e")
        print("  a mascara continua largando a curva fora da familia — antes de")
        print("  promover, e' o caso de duvidar do estrato, nao do checkpoint.")


if __name__ == "__main__":
    main()
