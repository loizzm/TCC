"""TETO da separacao por cor: existe uma cor que ISOLA a linha de entrada?

A sondagem heuristica (`sonda_linha_entrada.py`) chega a F1 0,33-0,49, abaixo
do gate de ganho atual (0,740). Isso pode significar duas coisas muito
diferentes, e a decisao de retreinar depende de qual e:

  (a) o sinal esta na imagem e a heuristica e que e fraca  -> vale engenharia
  (b) a cor NAO isola a linha de entrada                   -> so retreino

Este arquivo mede (a) contra (b) com um ORACULO: a trajetoria verdadeira da
linha de entrada vem de `verdade.json` + calibracao, e para cada cor presente
na figura se mede quanto dela cai sobre essa trajetoria (precisao) e quanto da
trajetoria ela cobre (revocacao). O melhor valor por figura e o TETO de
qualquer detector que separe por cor — nenhuma heuristica supera o oraculo.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from identify.calibrate import calibrate

RAIZ = Path(__file__).resolve().parent / "reports" / "amostras_aleatorias"
LOTES = (RAIZ, RAIZ / "balanceado", RAIZ / "balanceado2")
TOL_PX = 3


def trajetoria(cal, v, xs):
    """Linha de entrada verdadeira, em linhas de pixel, para as colunas `xs`."""
    tv = cal.sx * xs + cal.ox
    u = np.zeros_like(tv)
    acc = 0.0
    for U, inst in v["degraus"]:
        u[tv >= inst] = acc + U
        acc += U
    return (u - cal.oy) / cal.sy


def main() -> None:
    linhas = []
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
            H, W = img.shape[:2]
            x0, y0, x1, y1 = (int(z) for z in cal.bbox_px)
            xs = np.arange(max(x0 + 2, 0), min(x1 - 2, W - 1) + 1)
            if xs.size < 10:
                continue
            yl = np.clip(np.rint(trajetoria(cal, v, xs)).astype(int), 0, H - 1)

            alvo = np.zeros((H, W), bool)
            for dy in range(-TOL_PX, TOL_PX + 1):
                alvo[np.clip(yl + dy, 0, H - 1), xs] = True

            sub = img[y0 + 2:y1 - 1, x0 + 2:x1 - 1]
            q = sub.astype(np.int32) // 32
            chave_sub = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
            cont = Counter(chave_sub.ravel())
            fundo = cont.most_common(1)[0][0]

            qi = img.astype(np.int32) // 32
            chave = qi[..., 0] * 64 + qi[..., 1] * 8 + qi[..., 2]
            dentro = np.zeros((H, W), bool)
            dentro[y0 + 2:y1 - 1, x0 + 2:x1 - 1] = True

            melhor = {"f1": -1.0}
            for cor, n in cont.most_common(12):
                if cor == fundo or n < 0.2 * xs.size:
                    continue
                m = (chave == cor) & dentro
                if not m.any():
                    continue
                tp = int((m & alvo).sum())
                prec = tp / max(int(m.sum()), 1)
                # revocacao medida por COLUNA: a linha existe em toda coluna
                col = np.zeros(xs.size, bool)
                for dy in range(-TOL_PX, TOL_PX + 1):
                    col |= m[np.clip(yl + dy, 0, H - 1), xs]
                rec = float(col.mean())
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                if f1 > melhor["f1"]:
                    melhor = {"f1": f1, "prec": prec, "rec": rec, "cor": int(cor)}
            if melhor["f1"] < 0:
                continue
            linhas.append({"arquivo": f"{d.name}/{nome}", "familia": v["familia"],
                           "n_degraus": v["n_degraus"],
                           "fundo_escuro": v["fundo_escuro"], **melhor})

    (RAIZ / "sondagem_teto.json").write_text(json.dumps(linhas, indent=2),
                                             encoding="utf-8")
    L = []
    def P(s=""):
        L.append(s)
    n = len(linhas)
    f1 = np.array([x["f1"] for x in linhas])
    pr = np.array([x["prec"] for x in linhas])
    rc = np.array([x["rec"] for x in linhas])
    P(f"TETO DA SEPARACAO POR COR — melhor cor por figura contra a")
    P(f"trajetoria VERDADEIRA da linha de entrada.  n={n}")
    P("=" * 72)
    P(f"  {'metrica':<28}{'p10':>9}{'p50':>9}{'p90':>9}")
    for rot, a in (("F1 (cor x linha real)", f1),
                   ("precisao (tinta no alvo)", pr),
                   ("revocacao (colunas cobertas)", rc)):
        P(f"  {rot:<28}{np.percentile(a,10):>9.3f}{np.median(a):>9.3f}"
          f"{np.percentile(a,90):>9.3f}")
    P()
    for lim in (0.95, 0.90, 0.80, 0.60):
        P(f"  figuras com uma cor que isola a entrada a F1 >= {lim:.2f}: "
          f"{int((f1>=lim).sum())}/{n} ({(f1>=lim).mean():.1%})")
    P()
    P("  por familia:")
    for fam in sorted({x["familia"] for x in linhas}):
        g = [x for x in linhas if x["familia"] == fam]
        a = np.array([x["f1"] for x in g])
        P(f"    {fam:<7} n={len(g):<4d} F1 med={np.median(a):.3f}  "
          f">=0,90: {int((a>=0.90).sum())}/{len(g)} ({(a>=0.90).mean():5.1%})")
    P("  por tema:")
    for esc in (False, True):
        g = [x for x in linhas if x["fundo_escuro"] == esc]
        if not g:
            continue
        a = np.array([x["f1"] for x in g])
        P(f"    escuro={str(esc):<6} n={len(g):<4d} F1 med={np.median(a):.3f}  "
          f">=0,90: {int((a>=0.90).sum())}/{len(g)} ({(a>=0.90).mean():5.1%})")
    txt = "\n".join(L)
    (RAIZ / "sondagem_teto.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
