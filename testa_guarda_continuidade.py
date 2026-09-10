"""Testa uma GUARDA DE CONTINUIDADE em `mask_to_polyline`, sem retreino.

O bug, isolado em `identify/polyline.py:114`:

    if not multi_ramo or anterior is None:
        v = float(np.median(linhas))     # aceita o bloco, esteja onde estiver

Numa coluna de RAMO UNICO o codigo aceita o bloco incondicionalmente. Isso
quebra quando a curva de saida e tracejada: no VAO do tracejado a unica tinta
da coluna e a LINHA DE ENTRADA, entao o bloco unico e o distrator, ele vira o
novo `anterior`, e da coluna seguinte em diante a logica de "siga o bloco mais
proximo" segue o objeto errado com confianca. A polilinha fica AGARRADA na
entrada.

Evidencia de que e esse o mecanismo (n=139 figuras de um degrau):
    curva SOLIDA (sem vaos)      8/43  ruins (18,6 %)
    tracejada/pontilhada        46/96  ruins (47,9 %)
    Fisher OR=0,248  p=0,0012

A GUARDA: numa coluna de ramo unico, se o bloco esta longe demais de
`anterior`, nao aceite — pule a coluna. `anterior` sobrevive ao vao, a
interpolacao final ja preenche o buraco, e quando a tinta da curva volta a
referencia ainda esta correta. O limiar e em multiplos da espessura mediana do
traco, a mesma escala que o resto da funcao ja usa.

Mede a serie extraida contra a verdade analitica. Nao toca no repositorio:
importa `mask_to_polyline` e reimplementa o laco com a guarda ao lado.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import signal as sg
from skimage.morphology import skeletonize

from identify.calibrate import calibrate
from identify.extract import load_model, predict_mask
from identify.polyline import (MAX_GAP_FRAC, MIN_COMPONENT_PX, _blocos,
                               mask_to_polyline, polyline_to_series)

RAIZ = Path(__file__).resolve().parent / "reports" / "amostras_aleatorias"
LOTES = (RAIZ, RAIZ / "balanceado", RAIZ / "balanceado2")
VAO_MIN_FRAC = 3.0


def polilinha_com_guarda(mask, bbox=None, salto_max=None):
    """`mask_to_polyline` + guarda de continuidade no ramo unico.

    `salto_max` em multiplos da espessura mediana; None desliga a guarda e o
    caminho volta a ser byte a byte o original.
    """
    if bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        fora = np.ones(mask.shape, dtype=bool)
        fora[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = False
        mask = mask.copy()
        mask[fora] = 0
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        return np.empty(0), np.empty(0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.empty(0), np.empty(0)
    uniao = np.zeros(binary.shape, dtype=bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX:
            uniao |= (lab == k)
    if not uniao.any():
        return np.empty(0), np.empty(0)
    skel = skeletonize(uniao)
    esp = uniao.sum(axis=0)
    esp = esp[esp > 0]
    espessura = float(np.median(esp)) if esp.size else 1.0

    xs, ys = [], []
    anterior = ultimo_x = None
    for x in range(skel.shape[1]):
        coluna = skel[:, x]
        linhas = np.flatnonzero(coluna)
        if not linhas.size:
            continue
        if (anterior is not None and ultimo_x is not None
                and (x - ultimo_x) > VAO_MIN_FRAC * espessura):
            anterior = None
        blocos = _blocos(coluna)
        multi = False
        if len(blocos) > 1:
            b = sorted(blocos)
            multi = max(q[0] - p[1] for p, q in zip(b, b[1:])) > VAO_MIN_FRAC * espessura
        if not multi or anterior is None:
            v = float(np.median(linhas))
            # ---- A GUARDA ----
            if (salto_max is not None and anterior is not None
                    and abs(v - anterior) > salto_max * espessura):
                # Salto grande demais para ser a mesma curva numa coluna
                # vizinha: provavelmente e outro objeto. Pula a coluna e
                # PRESERVA `anterior` — a interpolacao final cobre o buraco.
                continue
        else:
            a, b = min(blocos,
                       key=lambda t: 0.0 if t[0] <= anterior <= t[1]
                       else min(abs(t[0] - anterior), abs(t[1] - anterior)))
            dentro = linhas[(linhas >= a) & (linhas <= b)]
            v = float(np.median(dentro)) if dentro.size else float(np.median(linhas))
        xs.append(float(x)); ys.append(v)
        anterior = v; ultimo_x = x
    if len(xs) < 2:
        return np.empty(0), np.empty(0)
    xa, ya = np.asarray(xs), np.asarray(ys)
    xf = np.arange(int(xa[0]), int(xa[-1]) + 1, dtype=float)
    yf = np.interp(xf, xa, ya)
    larg = xa[-1] - xa[0]
    if larg > 0:
        for i in np.flatnonzero(np.diff(xa) > MAX_GAP_FRAC * larg):
            yf[(xf > xa[i]) & (xf < xa[i + 1])] = np.nan
    ok = ~np.isnan(yf)
    return xf[ok], yf[ok]


def verdade(v, t):
    y = np.zeros_like(t)
    for U, inst in v["degraus"]:
        K = v["K_planta"] * U
        th = inst + v["theta_sistema"]
        m = t >= th
        if not m.any():
            continue
        ta = t[m] - th
        if v["order"] == "fopdt":
            y[m] += K * (1 - np.exp(-ta / v["tau"]))
        else:
            wn, z = v["wn"], v["zeta"]
            tu = np.linspace(0, float(ta.max()) + 1e-12, 3000)
            _, yu = sg.step(sg.TransferFunction([K*wn*wn], [1, 2*z*wn, wn*wn]), T=tu)
            y[m] += np.interp(ta, tu, yu)
    return y


def main() -> None:
    modelo = load_model(str(Path(__file__).resolve().parent / "models" /
                            "unet_stageA.pt"), "cpu")
    LIMIARES = [None, 12.0, 8.0, 5.0, 3.0]
    res = {k: [] for k in LIMIARES}
    for d in LOTES:
        ver = {v["arquivo"]: v for v in json.loads((d / "verdade.json").read_text())}
        for nome, v in ver.items():
            p = d / nome
            if not p.exists():
                continue
            img = np.asarray(Image.open(p).convert("RGB"))
            cal = calibrate(img)
            if not cal.ok:
                continue
            mk = predict_mask(modelo, img, "cpu")
            bb = cal.bbox_px if any(cal.bbox_px) else None
            for lim in LIMIARES:
                if lim is None:
                    xp, yp = mask_to_polyline(mk, bbox=bb)   # o de producao
                else:
                    xp, yp = polilinha_com_guarda(mk, bbox=bb, salto_max=lim)
                if xp.size < 10:
                    res[lim].append(dict(arq=f"{d.name}/{nome}", nd=v["n_degraus"],
                                         ls=v["linestyle"], nrmse=float("inf")))
                    continue
                t, y = polyline_to_series(xp, yp, cal)
                o = np.argsort(t); t, y = t[o], y[o]
                yt = verdade(v, t)
                fx = float(np.ptp(yt)) or 1.0
                res[lim].append(dict(arq=f"{d.name}/{nome}", nd=v["n_degraus"],
                                     ls=v["linestyle"],
                                     nrmse=float(np.sqrt(np.mean((y - yt) ** 2)) / fx)))
            print(f"  {d.name}/{nome}", flush=True)

    json.dump({str(k): v for k, v in res.items()},
              open(RAIZ / "guarda_continuidade.json", "w"))

    print("\nQUALIDADE DA SERIE EXTRAIDA — guarda de continuidade no ramo unico")
    print("=" * 74)
    print(f"  {'limiar (x espessura)':<24}{'n':>5}{'NRMSE p50':>12}{'sujas (>=0.05)':>17}")
    for lim in LIMIARES:
        g = [x for x in res[lim] if np.isfinite(x["nrmse"])]
        sujas = sum(1 for x in g if x["nrmse"] >= 0.05)
        rot = "sem guarda (producao)" if lim is None else f"{lim:g}"
        print(f"  {rot:<24}{len(g):>5}{np.median([x['nrmse'] for x in g]):>12.4f}"
              f"{sujas:>10}/{len(g):<6} ({sujas/len(g):5.1%})")
    print()
    print("  so figuras de UM degrau:")
    for lim in LIMIARES:
        g = [x for x in res[lim] if np.isfinite(x["nrmse"]) and x["nd"] == 1]
        sujas = sum(1 for x in g if x["nrmse"] >= 0.05)
        rot = "sem guarda (producao)" if lim is None else f"{lim:g}"
        print(f"    {rot:<24}{len(g):>5}{np.median([x['nrmse'] for x in g]):>12.4f}"
              f"{sujas:>10}/{len(g):<6} ({sujas/len(g):5.1%})")
    print()
    print("  por tipo de linha (sujas), producao -> melhor limiar:")
    melhor = min([l for l in LIMIARES if l is not None],
                 key=lambda l: sum(1 for x in res[l]
                                   if np.isfinite(x["nrmse"]) and x["nrmse"] >= 0.05))
    for ls in ("-", "--", "-.", ":"):
        a = [x for x in res[None] if x["ls"] == ls and np.isfinite(x["nrmse"])]
        b = [x for x in res[melhor] if x["ls"] == ls and np.isfinite(x["nrmse"])]
        sa = sum(1 for x in a if x["nrmse"] >= 0.05)
        sb = sum(1 for x in b if x["nrmse"] >= 0.05)
        print(f"    {ls:<5} {sa}/{len(a)} -> {sb}/{len(b)}")
    print(f"\n  melhor limiar por sujas: {melhor}")


if __name__ == "__main__":
    main()
