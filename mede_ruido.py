"""Mede o defeito que o estrato `ruido_alto` existe para curar.

O corpus base sorteia `snr_db` em U(20, 60) e NUNCA desce de 20 — medido nos
quatro corpora, minimo 20,0 dB. A rede aprendeu a segmentar curva NITIDA;
abaixo de 20 dB ela extrapola, e o joelho da assertividade cai exatamente ali.

Duas medidas, e elas respondem a coisas diferentes:

  RECALL DA MASCARA — dos pixels que a figura pinta de curva (`mask.png`, que o
  gerador escreve), quantos a rede acende. E a medida direta: com ruido a curva
  vira uma FAIXA, e a rede precisa decidir onde dentro dela esta "a curva".

  VEREDITO DA PIPELINE — o que sai do outro lado. Aqui toda recusa e falso
  positivo: as plantas do estrato sao de fase minima e dentro da familia.

Rodar ANTES e DEPOIS do retreino. Em `data/val` o recall tem de NAO cair —
subir num as custas do outro e trocar de vies, nao aprender.

Uso:
    .venv/bin/python mede_ruido.py [--modelo models/unet_stageA.pt]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from identify.extract import load_model, predict_mask
from identify.pipeline import identify_from_image

RAIZ = Path(__file__).resolve().parent
FAIXAS = ((5, 10), (10, 15), (15, 20), (20, 60))


def mede(d: Path, modelo, dev, lim: int) -> list[dict]:
    out = []
    for s in sorted(d.glob("sample_*"))[:lim]:
        meta = json.loads((s / "meta.json").read_text())
        img = np.asarray(Image.open(s / "image.png").convert("RGB"))
        alvo = np.asarray(Image.open(s / "mask.png").convert("L")) > 127
        pred = predict_mask(modelo, img, dev) > 127
        r = identify_from_image(img, modelo, dev)
        out.append({
            "snr": float(meta["render"]["snr_db"]),
            "recall": (float((pred & alvo).sum()) / float(alvo.sum())
                       if alvo.sum() else float("nan")),
            "ok": bool(r["ok"]),
            "reason": r["reason"] or "",
            "n_points": int(r["n_points"]),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="models/unet_stageA.pt")
    ap.add_argument("--dir", action="append", default=None)
    ap.add_argument("--lim", type=int, default=150)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / a.modelo), dev)
    modelo.eval()
    dirs = a.dir or ["data/val", "data/val_ruido"]

    print(f"RUIDO — {Path(a.modelo).name}\n")
    tudo = {}
    for d in dirs:
        tudo[d] = mede(RAIZ / d, modelo, dev, a.lim)
        r = np.array([x["recall"] for x in tudo[d]])
        ok = sum(1 for x in tudo[d] if x["ok"])
        print(f"  {d:<18} n={len(tudo[d]):<4} recall p10={np.nanpercentile(r,10):.3f} "
              f"p50={np.nanmedian(r):.3f}   entrega {ok/len(tudo[d]):.0%}")

    print(f"\n  POR FAIXA DE SNR (as duas populacoes juntas)")
    linhas = [x for v in tudo.values() for x in v]
    print(f"  {'SNR':>12}{'n':>5}{'recall p50':>12}{'entrega':>9}{'  motivos de recusa'}")
    for lo, hi in FAIXAS:
        g = [x for x in linhas if lo <= x["snr"] < hi]
        if not g:
            continue
        r = np.array([x["recall"] for x in g])
        ok = sum(1 for x in g if x["ok"])
        mot = Counter(x["reason"] for x in g if not x["ok"])
        txt = ", ".join(f"{k}={v}" for k, v in mot.most_common(3)) or "—"
        print(f"  {f'{lo}-{hi} dB':>12}{len(g):>5}{np.nanmedian(r):>12.3f}"
              f"{ok/len(g):>9.0%}   {txt}")
    print()
    print("  ALVO do retreino: a faixa 5-20 dB subir, e a 20-60 NAO cair.")
    print("  Toda recusa aqui e FALSO POSITIVO: as plantas sao de fase minima")
    print("  e dentro da familia, por construcao do estrato.")


if __name__ == "__main__":
    main()
