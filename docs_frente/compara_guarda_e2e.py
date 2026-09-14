"""Efeito PONTA A PONTA da guarda de continuidade, contra a producao.

A guarda vive em `identify/polyline.py:114`, no caminho de RAMO UNICO: hoje o
codigo aceita o bloco da coluna incondicionalmente, por mais longe que ele
esteja do ponto anterior. Num vao de curva tracejada a unica tinta da coluna e
a LINHA DE ENTRADA, e a polilinha se agarra nela. A guarda rejeita o salto e
preserva a referencia; a interpolacao final, que ja existe, cobre o buraco.

Este arquivo compara a saida da pipeline COM a guarda (`e2e_guarda.json`,
limiar 8x a espessura mediana) contra a de producao (`analise.json` dos tres
lotes). Mede o que interessa: recusas, truncagem espuria, erro de K e a
deteccao de multi-degrau.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "reports" / "amostras_aleatorias"
LOTES = (R, R / "balanceado", R / "balanceado2")


def tdom(o, tau, wn, z):
    if o == "fopdt":
        return float(tau)
    return float(1 / (z * wn)) if z <= 1 else float((z + np.sqrt(z * z - 1)) / wn)


prod = {}
for d in LOTES:
    for x in json.loads((d / "analise.json").read_text()):
        prod[f"{d.name}/{x['arquivo']}"] = x
guarda = {x["arquivo"]: x for x in json.loads((R / "e2e_guarda.json").read_text())}
chaves = sorted(set(prod) & set(guarda))

def erroK(x, hat):
    p = hat.get("params") or {}
    if not hat.get("ok") or p.get("K") is None:
        return None
    return abs((float(p["K"]) - x["K_true"]) / abs(x["K_true"]))

L = []
def P(s=""):
    L.append(s)

P("GUARDA DE CONTINUIDADE — efeito ponta a ponta (limiar 8x espessura)")
P("=" * 78)
P(f"  n = {len(chaves)} figuras, as mesmas nos dois lados")
P()

for rot_pop, filtro in (("TODAS", lambda k: True),
                        ("SO 1 DEGRAU", lambda k: prod[k]["n_degraus"] == 1),
                        ("SO 2+ DEGRAUS", lambda k: prod[k]["n_degraus"] > 1)):
    ks = [k for k in chaves if filtro(k)]
    P(f"  {rot_pop}  (n={len(ks)})")
    P(f"    {'':<14}{'fisico':>10}{'recusas':>10}{'|errK| p50':>13}"
      f"{'trunc espuria':>16}")
    for rot, src in (("producao", prod), ("com guarda", guarda)):
        ok = [k for k in ks if src[k]["ok"]]
        ek = [e for k in ok
              for e in [abs(src[k]["err_K"]) if src is prod
                        else erroK(prod[k], guarda[k])] if e is not None]
        esp = sum(1 for k in ok if prod[k]["n_degraus"] == 1
                  and src[k]["truncado_em"] is not None)
        n1 = sum(1 for k in ks if prod[k]["n_degraus"] == 1)
        P(f"    {rot:<14}{len(ok)/len(ks):>10.1%}{len(ks)-len(ok):>10}"
          f"{(np.median(ek) if ek else float('nan')):>13.4f}"
          f"{esp:>10}/{n1:<5}")
    P()

P("  DETECCAO DE MULTI-DEGRAU:")
for rot, src in (("producao  ", prod), ("com guarda", guarda)):
    ok = [k for k in chaves if src[k]["ok"]]
    TP = sum(1 for k in ok if prod[k]["n_degraus"] > 1 and src[k]["truncado_em"] is not None)
    FN = sum(1 for k in ok if prod[k]["n_degraus"] > 1 and src[k]["truncado_em"] is None)
    FP = sum(1 for k in ok if prod[k]["n_degraus"] == 1 and src[k]["truncado_em"] is not None)
    p_ = TP / max(1, TP + FP); r_ = TP / max(1, TP + FN)
    P(f"    {rot} TP={TP:>3} FN={FN:>3} FP={FP:>3}  precisao={p_:6.1%} "
      f"revocacao={r_:6.1%}  F1={2*p_*r_/max(1e-9,p_+r_):.3f}")
P()

P("  MOTIVOS DE RECUSA:")
for rot, src in (("producao  ", prod), ("com guarda", guarda)):
    c = Counter(src[k]["reason"] for k in chaves if not src[k]["ok"])
    P(f"    {rot} " + (", ".join(f"{a}={b}" for a, b in c.most_common()) or "nenhuma"))
P()

virou_ok = [k for k in chaves if not prod[k]["ok"] and guarda[k]["ok"]]
virou_ruim = [k for k in chaves if prod[k]["ok"] and not guarda[k]["ok"]]
sumiu = [k for k in chaves if prod[k]["n_degraus"] == 1 and prod[k]["ok"]
         and guarda[k]["ok"] and prod[k]["truncado_em"] is not None
         and guarda[k]["truncado_em"] is None]
surgiu = [k for k in chaves if prod[k]["n_degraus"] == 1 and prod[k]["ok"]
          and guarda[k]["ok"] and prod[k]["truncado_em"] is None
          and guarda[k]["truncado_em"] is not None]
P("  MUDANCAS, figura a figura:")
P(f"    recusada -> aceita:              {len(virou_ok)}")
P(f"    aceita -> recusada:              {len(virou_ruim)}")
P(f"    truncagem espuria sumiu:         {len(sumiu)}")
P(f"    truncagem espuria nova:          {len(surgiu)}")
if virou_ruim:
    P(f"    (as que pioraram: {', '.join(k.split('/')[-1] for k in virou_ruim)})")
P()

P("  EXATIDAO no nucleo de 1 degrau sem truncagem espuria:")
for rot, src in (("producao  ", prod), ("com guarda", guarda)):
    g = [k for k in chaves if prod[k]["n_degraus"] == 1 and src[k]["ok"]
         and src[k]["truncado_em"] is None]
    ek = [e for k in g
          for e in [abs(prod[k]["err_K"]) if src is prod else erroK(prod[k], guarda[k])]
          if e is not None]
    P(f"    {rot} n={len(g):<4d} |errK| p50={np.median(ek):.4f}")

txt = "\n".join(L)
(R / "guarda_e2e.txt").write_text(txt, encoding="utf-8")
print(txt)
