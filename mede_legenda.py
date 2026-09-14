"""Mede o defeito que o estrato `legenda_oclusora` existe para curar.

O DEFEITO, isolado num par controlado de imagem REAL. `caso_real_neg_super.png`
e `caso_real_neg_super_legenda_movida.png` sao a mesma figura; a unica
diferenca e' a caixa da legenda estar em 'lower left', atravessando a faixa de
acomodacao, ou em 'upper right'. Com ela em cima da curva, `wn` e `zeta` saem
com 26 % e 30 % de erro; movida, com 2,97 % e 2,23 %. A rede segue a BORDA
HORIZONTAL da caixa como se fosse patamar, isso antecipa a acomodacao, e o
ajuste compensa com polo dominante mais lento e menos amortecido.

TRES MEDIDAS, e elas respondem a coisas diferentes:

  RECALL DENTRO x FORA DA CAIXA (`data/val_legenda`). Dos pixels que a figura
  pinta de curva, quantos a rede acende — separados por estarem sob a caixa ou
  nao. O PAR e' o numero: recall fora alto e dentro baixo e' exatamente o
  defeito, e a razao entre os dois e' imune a amostras dificeis por outro
  motivo. A pegada da caixa e' obtida por DIFERENCA DE RENDER (a amostra e'
  re-renderizada sem legenda a partir do seed): `add_axes` usa retangulo fixo e
  `savefig` nao usa `bbox_inches`, entao a legenda nao move os eixos e tudo que
  difere entre os dois PNG E' a caixa. Exato, nao proxy — a primeira tentativa
  mediu por COR e contou anti-aliasing como oclusao.

  GUARDA (`data/val`). Recall global no corpus base. Tem de NAO cair: subir num
  as custas do outro e trocar de vies, nao aprender.

  O PAR REAL. Os erros de `K`, `wn`, `zeta` e `theta` ponta a ponta nas duas
  fixtures. E' o alvo de verdade — o resto e' o mecanismo.

Uso:
    .venv/bin/python mede_legenda.py [--modelo models/unet_stageA.pt]
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset.generator import render_sample, sample_style, sample_system
from identify.extract import load_model, predict_mask
from identify.pipeline import identify_from_image

RAIZ = Path(__file__).resolve().parent
FIX = RAIZ / "tests" / "fixtures"

# Verdade das duas fixtures — a MESMA de `tests/part2/test_caso_real_negativo.py`.
NEG_SUPER = {"K": -3.0, "wn": 4.0, "zeta": 1.25, "theta": 3.5}


def pegada_da_caixa(meta: dict, tmp: Path) -> np.ndarray | None:
    """Mascara booleana da caixa da legenda, por diferenca de render."""
    seed = meta.get("seed")
    if seed is None:
        return None
    f = np.random.SeedSequence(int(seed)).spawn(3)
    spec = sample_system(np.random.default_rng(f[0]))
    style = sample_style(np.random.default_rng(f[1]))
    render_sample(spec, replace(style, has_legend=False), tmp,
                  add_noise=True, rng=np.random.default_rng(f[2]), seed=int(seed))
    return np.asarray(Image.open(tmp / "image.png").convert("RGB"), dtype=np.int16)


def mede_recall(d: Path, modelo, dev, lim: int, com_caixa: bool) -> list[dict]:
    out = []
    tmp = Path(tempfile.mkdtemp(prefix="legenda_"))
    try:
        for s in sorted(d.glob("sample_*"))[:lim]:
            meta = json.loads((s / "meta.json").read_text())
            img = np.asarray(Image.open(s / "image.png").convert("RGB"))
            alvo = np.asarray(Image.open(s / "mask.png").convert("L")) > 127
            if not alvo.any():
                continue
            pred = predict_mask(modelo, img, dev) > 127
            r = {"recall": float((pred & alvo).sum()) / float(alvo.sum())}
            if com_caixa:
                sem = pegada_da_caixa(meta, tmp)
                if sem is None or sem.shape != img.shape:
                    continue
                caixa = np.abs(img.astype(np.int16) - sem).sum(axis=2) > 0
                dentro, fora = alvo & caixa, alvo & ~caixa
                if dentro.sum() < 20 or fora.sum() < 20:
                    continue
                r["dentro"] = float((pred & dentro).sum()) / float(dentro.sum())
                r["fora"] = float((pred & fora).sum()) / float(fora.sum())
                r["frac"] = float(dentro.sum()) / float(alvo.sum())
            out.append(r)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def mede_par_real(modelo, dev) -> None:
    print("  PAR REAL — erro relativo ponta a ponta")
    print(f"    {'figura':<24}{'K':>9}{'wn':>9}{'zeta':>9}{'theta':>9}")
    for nome, png in (("legenda OCLUINDO", "caso_real_neg_super.png"),
                      ("legenda movida", "caso_real_neg_super_legenda_movida.png")):
        img = np.asarray(Image.open(FIX / png).convert("RGB"))
        r = identify_from_image(img, modelo, dev)
        if not r["ok"]:
            print(f"    {nome:<24} SEM FISICO: {r['reason']!r}")
            continue
        if r["order"] != "second":
            print(f"    {nome:<24} ordem {r['order']!r} (esperada 'second')")
            continue
        e = {k: abs(r["params"][k] - v) / abs(v) for k, v in NEG_SUPER.items()}
        print(f"    {nome:<24}" + "".join(f"{e[k]:>8.1%} " for k in
                                          ("K", "wn", "zeta", "theta")))
    print("    alvo: a linha de cima chegar perto da de baixo. Hoje wn e zeta")
    print("    saem 26 % e 30 % com a caixa e ~3 % sem ela.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="models/unet_stageA.pt")
    ap.add_argument("--lim", type=int, default=300)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / a.modelo), dev)
    modelo.eval()

    print(f"ESTRATO LEGENDA OCLUSORA — {Path(a.modelo).name}\n")

    v = mede_recall(RAIZ / "data" / "val_legenda", modelo, dev, a.lim, True)
    if v:
        de = np.array([x["dentro"] for x in v])
        fo = np.array([x["fora"] for x in v])
        fr = np.array([x["frac"] for x in v])
        print(f"  val_legenda (n={len(v)}) — recall nos pixels da curva")
        print(f"    {'':<10}{'p10':>9}{'p50':>9}{'p90':>9}")
        for nome, arr in (("DENTRO", de), ("FORA", fo)):
            print(f"    {nome:<10}{np.percentile(arr,10):>9.3f}"
                  f"{np.median(arr):>9.3f}{np.percentile(arr,90):>9.3f}")
        print(f"    razao dentro/fora  p50 = {np.median(de / np.maximum(fo,1e-9)):.3f}"
              f"   (1,000 = a caixa nao atrapalha)")
        print(f"    fracao da curva sob a caixa  p50 = {np.median(fr):.3f}")
    else:
        print("  val_legenda: sem amostras mensuraveis (o corpus foi gerado?)")

    g = mede_recall(RAIZ / "data" / "val", modelo, dev, a.lim, False)
    if g:
        r = np.array([x["recall"] for x in g])
        print(f"\n  GUARDA val (n={len(g)}) — recall global"
              f"  p10={np.percentile(r,10):.3f} p50={np.median(r):.3f}"
              f" p90={np.percentile(r,90):.3f}")
        print("    tem de NAO cair.")
    print()
    mede_par_real(modelo, dev)


if __name__ == "__main__":
    main()
