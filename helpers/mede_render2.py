"""Mede o que o estrato `familia_alt` (§73) existe para curar, e as guardas.

O DEFEITO. A rede treina no render de `dataset/generator.py` e e' avaliada no
de `rg_aleatorio.py`. Medido com o mesmo instrumento dos dois lados — a fracao
das colunas em que a mascara acende a +-5 px da curva VERDADEIRA, projetada em
pixels pela afim que o `calibrate()` estima:

    rg_aleatorio (avaliacao)   p50 = 0,713   79 % abaixo de 80 %
    dataset/generator (treino) p50 = 0,925   29 % abaixo de 80 %
    Mann-Whitney p = 1,05e-12

Isso e' o dobro do custo de qualquer outro fator medido no estrato |K| < 1, e
sustenta o diagnostico ponta a ponta daquele lote: o Estagio D acerta 100/100
com a serie verdadeira, a calibracao esta quase exata (0,24 % de erro de escala
em X) e a melhor afim possivel PIORA o resultado — o que sobra e' a forma da
serie, e ela e' 30x pior nas figuras reprovadas (p = 1,1e-04).

O ALVO: `lote_selecao` subir. As GUARDAS: `data/val` NAO cair, e o veredito nos
lotes de controle nao piorar. Subir um as custas do outro e' trocar de vies,
nao aprender — mesmo criterio de todos os retreinos anteriores.

POR QUE UM LOTE DE SELECAO SEPARADO. `lote_k_menor1`, `lote_k_maior1` e
`lote_ruido` sao os lotes de CONTROLE desta parte do trabalho. Escolher a epoca
por eles e' escolher olhando o teste: o numero final deixaria de ser estimativa
de generalizacao. `lote_selecao` e' da MESMA familia de render (que e' o que se
quer medir) com semente nova, e existe so para escolher a epoca.

Uso:
    .venv/bin/python helpers/mede_render2.py [--modelo models/unet_stageA.pt]
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
from scipy import signal

from identify.calibrate import calibrate
from identify.extract import load_model, predict_mask

RAIZ = Path(__file__).resolve().parent.parent
TOL_PX = 5.0


def _resposta(order, K, tau, wn, zeta, theta, t):
    """Grade UNIFORME + interpolacao: `scipy.signal.step` exige passo constante
    e a grade de colunas de uma imagem nao tem."""
    tu = np.linspace(float(t[0]), float(t[-1]), max(int(t.size), 512))
    y = np.zeros_like(tu)
    m = tu >= theta
    ta = tu[m] - theta
    if ta.size:
        if order == "fopdt":
            y[m] = float(K) * (1.0 - np.exp(-ta / float(tau)))
        else:
            wn = float(wn)
            _, ya = signal.step(signal.TransferFunction(
                [float(K) * wn * wn], [1.0, 2.0 * float(zeta) * wn, wn * wn]), T=ta)
            y[m] = ya
    return np.interp(t, tu, y)


def _cobertura(img, modelo, dev, order, K, tau, wn, zeta, theta, t_ini, t_fim):
    cal = calibrate(img)
    if not cal.ok:
        return None
    mk = predict_mask(modelo, img, dev) > 127
    H, W = mk.shape
    x0 = int(np.clip((t_ini - cal.ox) / cal.sx, 0, W - 1))
    x1 = int(np.clip((t_fim - cal.ox) / cal.sx, 0, W - 1))
    if abs(x1 - x0) < 20:
        return None
    xs = np.arange(min(x0, x1), max(x0, x1) + 1)
    t = cal.sx * xs.astype(float) + cal.ox
    l = (_resposta(order, K, tau, wn, zeta, theta, t) - cal.oy) / cal.sy
    ac = 0
    for xi, li in zip(xs, l):
        if not np.isfinite(li):
            continue
        lo, hi = int(max(li - TOL_PX, 0)), int(min(li + TOL_PX, H - 1))
        ac += bool(mk[lo:hi + 1, xi].any())
    return ac / xs.size


def lote_rg(d: Path, modelo, dev, lim: int) -> np.ndarray:
    out = []
    for v in json.loads((d / "verdade.json").read_text())[:lim]:
        img = np.asarray(Image.open(d / v["arquivo"]).convert("RGB"))
        c = _cobertura(img, modelo, dev, v["order"], float(v["K"]), v.get("tau"),
                       v.get("wn"), v.get("zeta"), float(v["theta"]),
                       0.0, float(v["t_fim"]))
        if c is not None:
            out.append(c)
    return np.array(out)


def corpus(d: Path, modelo, dev, lim: int) -> np.ndarray:
    out = []
    for s in sorted(d.glob("sample_*")):
        if len(out) >= lim:
            break
        m = json.loads((s / "meta.json").read_text())
        img = np.asarray(Image.open(s / "image.png").convert("RGB"))
        p = m["params"]
        c = _cobertura(img, modelo, dev, m["order"],
                       float(p["K"]) * float(m["step_amplitude"]),
                       p.get("tau"), p.get("wn"), p.get("zeta"), float(p["theta"]),
                       float(m["t_window"][0]), float(m["t_window"][1]))
        if c is not None:
            out.append(c)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="models/unet_stageA.pt")
    ap.add_argument("--lim", type=int, default=150)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / a.modelo), dev)
    modelo.eval()

    print(f"COBERTURA DA MASCARA SOBRE A CURVA VERDADEIRA — {Path(a.modelo).name}\n")
    print(f"  {'corpus':<38}{'n':>5}{'p10':>9}{'p50':>9}{'p90':>9}{'< 80 %':>9}")
    linhas = (
        ("ALVO  lote_selecao (familia rg)",
         lote_rg(RAIZ / "reports/amostras_aleatorias/lote_selecao", modelo, dev, a.lim)),
        ("      val_render2 (estrato novo)",
         corpus(RAIZ / "data/val_render2", modelo, dev, a.lim)),
        ("GUARDA data/val (familia de treino)",
         corpus(RAIZ / "data/val", modelo, dev, a.lim)),
    )
    for nome, v in linhas:
        if not v.size:
            print(f"  {nome:<38} (sem amostras mensuraveis)")
            continue
        print(f"  {nome:<38}{v.size:>5}{np.percentile(v,10):>9.3f}{np.median(v):>9.3f}"
              f"{np.percentile(v,90):>9.3f}{(v<0.80).mean():>9.0%}")
    print()
    print("  ALVO: `lote_selecao` subir. GUARDA: `data/val` NAO cair.")
    print("  `val_render2` diz se a rede APRENDEU o estrato — se ele nao subir,")
    print("  o treino nao pegou, e nao adianta olhar o alvo.")
    print()
    print("  Depois de escolher a epoca, medir o veredito ponta a ponta nos")
    print("  lotes de CONTROLE (k_maior1, k_menor1, ruido) — que NAO entram na")
    print("  selecao, para o numero final continuar sendo generalizacao.")


if __name__ == "__main__":
    main()
