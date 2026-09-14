"""Mede DIRETAMENTE o defeito que o estrato `plato_no_meio` existe para curar.

A metrica e a cobertura do PLATO DE REPOUSO: das colunas em que a figura
desenha a curva em repouso (antes de a resposta arrancar), quantas a mascara
acende. Isto e mais direto que medir pelo `_undershoot`, que so ve o defeito
de lado — e foi assim que ele apareceu (deteccao de fase nao-minima em 3 de 40,
correlacao -0,11 com o undershoot real).

Rodar ANTES e DEPOIS do retreino, nos dois conjuntos:
  `data/val_plato`   — repouso no MEIO do quadro (o estrato novo)
  `data/val`         — repouso na BORDA (o caso que ja funciona; guarda de
                       regressao, tem de continuar alto)

Uso:
    .venv/bin/python helpers/mede_plato_repouso.py [--modelo models/unet_stageA.pt]
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
from PIL import Image

from identify.extract import load_model, predict_mask

RAIZ = Path(__file__).resolve().parent.parent


def cobertura_do_plato(d: Path, modelo, dev, lim=200):
    """Fracao das colunas do PLATO que a mascara acende, por amostra."""
    out = []
    for s in sorted(d.glob("sample_*"))[:lim]:
        meta = json.loads((s / "meta.json").read_text())
        t = np.asarray(meta["series"]["t"], float)
        y = np.asarray(meta["series"]["y"], float)
        theta = float(meta["params"]["theta"])
        # colunas de repouso: antes de a resposta arrancar
        if theta <= t[0]:
            continue
        img = np.asarray(Image.open(s / "image.png").convert("RGB"))
        mk = predict_mask(modelo, img, dev) > 127
        x0, y0, x1, y1 = meta["plot_bbox_px"]
        aa = meta["axis_affine"]
        # dado -> pixel, invertendo a afim gravada no meta
        col = lambda tv: (tv - aa["ox"]) / aa["sx"]
        lin = lambda yv: (yv - aa["oy"]) / aa["sy"]
        c0, c1 = int(round(col(t[0]))), int(round(col(theta)))
        if c1 - c0 < 5:
            continue
        rep = float(np.median(y[:5]))
        r = int(round(lin(rep)))
        H, W = mk.shape
        cs = np.arange(max(c0, 0), min(c1, W - 1) + 1)
        acesa = np.zeros(cs.size, bool)
        for dy in range(-4, 5):                 # banda de +-4 px
            acesa |= mk[np.clip(r + dy, 0, H - 1), cs]
        out.append(float(acesa.mean()))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="models/unet_stageA.pt")
    ap.add_argument("--dir", action="append",
                    default=None, help="default: data/val e data/val_plato")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / a.modelo), dev)
    modelo.eval()
    dirs = a.dir or ["data/val", "data/val_plato"]

    print(f"COBERTURA DO PLATO DE REPOUSO — {Path(a.modelo).name}\n")
    print(f"  {'corpus':<20}{'n':>5}{'p10':>9}{'p50':>9}{'p90':>9}{'< 50 %':>10}")
    for d in dirs:
        c = cobertura_do_plato(RAIZ / d, modelo, dev)
        if not c.size:
            print(f"  {d:<20} (sem amostras com plato mensuravel)")
            continue
        print(f"  {d:<20}{c.size:>5}{np.percentile(c,10):>9.3f}"
              f"{np.median(c):>9.3f}{np.percentile(c,90):>9.3f}"
              f"{(c < 0.5).mean():>10.1%}")
    print()
    print("  ALVO do retreino: `val_plato` subir para perto de `val`, e `val`")
    print("  NAO cair. Subir um as custas do outro e trocar de vies, nao")
    print("  aprender posicao — o mesmo criterio do `helpers/seleciona_checkpoint.py`.")


if __name__ == "__main__":
    main()
