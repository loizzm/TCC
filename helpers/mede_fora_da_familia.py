"""Mede o defeito que o estrato `fase_nao_minima` existe para curar.

Duas medidas, e elas respondem a perguntas diferentes:

  RECALL DA MASCARA — dos pixels que a figura de fato pinta de curva
  (`mask.png`, que o gerador escreve), quantos a rede acende. Esta e' a
  medida direta do defeito: a rede aprendeu "aproximacao monotona a um
  patamar" como se fosse a definicao de curva de dados, e numa curva fora da
  familia ela segmenta so a sub-parte que se parece com isso.

  VEREDITO DA PIPELINE — o que sai do outro lado. O certo aqui e' RECUSAR com
  `resposta_inversa`. Recusar por qualquer outro motivo, ou nao recusar, e'
  errado; e recusar com a mascara vazia e' acertar pelo motivo errado, que e'
  por que as duas medidas tem de ser lidas juntas.

Rodar ANTES e DEPOIS do retreino. Em `data/val` o recall tem de NAO cair —
subir num as custas do outro e' trocar de vies, nao aprender.

Uso:
    .venv/bin/python helpers/mede_fora_da_familia.py [--modelo models/unet_stageA.pt]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from identify.extract import load_model, predict_mask
from identify.pipeline import identify_from_image

RAIZ = Path(__file__).resolve().parent.parent


def mede(d: Path, modelo, dev, lim: int) -> dict:
    recalls, npts = [], []
    motivos: Counter[str] = Counter()
    for s in sorted(d.glob("sample_*"))[:lim]:
        img = np.asarray(Image.open(s / "image.png").convert("RGB"))
        alvo = np.asarray(Image.open(s / "mask.png").convert("L")) > 127
        pred = predict_mask(modelo, img, dev) > 127
        if alvo.sum():
            recalls.append(float((pred & alvo).sum()) / float(alvo.sum()))
        r = identify_from_image(img, modelo, dev)
        npts.append(int(r["n_points"]))
        motivos[("ok" if r["ok"] else (r["reason"] or "?"))] += 1
    return {"n": len(npts), "recall": np.array(recalls),
            "n_points": np.array(npts), "motivos": motivos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="models/unet_stageA.pt")
    ap.add_argument("--dir", action="append", default=None)
    ap.add_argument("--lim", type=int, default=150)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / a.modelo), dev)
    modelo.eval()
    dirs = a.dir or ["data/val", "data/val_nmp"]

    print(f"FORA DA FAMILIA — {Path(a.modelo).name}\n")
    print(f"  {'corpus':<18}{'n':>5}{'recall p10':>12}{'p50':>8}{'p90':>8}"
          f"{'pts p50':>9}")
    resultados = {}
    for d in dirs:
        r = mede(RAIZ / d, modelo, dev, a.lim)
        resultados[d] = r
        rc = r["recall"]
        print(f"  {d:<18}{r['n']:>5}{np.percentile(rc,10):>12.3f}"
              f"{np.median(rc):>8.3f}{np.percentile(rc,90):>8.3f}"
              f"{np.median(r['n_points']):>9.0f}")
    print("\n  veredito da pipeline")
    for d, r in resultados.items():
        tot = max(r["n"], 1)
        itens = ", ".join(f"{k}={v} ({v/tot:.0%})"
                          for k, v in r["motivos"].most_common())
        print(f"    {d:<18}{itens}")


if __name__ == "__main__":
    main()
