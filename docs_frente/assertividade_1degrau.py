"""Assertividade na populacao de UM DEGRAU — o caso sem multi-degrau.

Todo o resto da analise mistura as duas populacoes, e a de multi-degrau domina
os erros (|erro de K| mediano 20,7 % contra 0,35 %). Este recorte responde
"quanto o sistema acerta quando o problema e o que ele foi projetado para
resolver": uma imagem, um degrau, uma planta.

Le os tres lotes com a linha de entrada desenhada (`amostras_aleatorias`,
`balanceado`, `balanceado2`) e, como contraste, os dois lotes sem ela
(`sem_omite`, `sem_omite_fit`), que sao a mesma populacao com o distrator
removido.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent / "reports" / "amostras_aleatorias"
COM = (RAIZ, RAIZ / "balanceado", RAIZ / "balanceado2")
SEM = (RAIZ / "sem_omite", RAIZ / "sem_omite_fit")


def carrega(dirs):
    out = []
    for d in dirs:
        f = d / "analise.json"
        if not f.exists():
            continue
        for x in json.loads(f.read_text()):
            x["lote"] = d.name
            out.append(x)
    return [x for x in out if x["n_degraus"] == 1]


def med(g, k):
    v = [abs(x[k]) for x in g if x.get(k) is not None]
    return float(np.median(v)) if v else float("nan")


def p90(g, k):
    v = [abs(x[k]) for x in g if x.get(k) is not None]
    return float(np.percentile(v, 90)) if v else float("nan")


L = []
def P(s=""):
    L.append(s)


um = carrega(COM)
sem = carrega(SEM)
ok = [x for x in um if x["ok"]]
resp = [x for x in um if x["respondeu"]]

P("ASSERTIVIDADE — SO FIGURAS DE UM DEGRAU (sem multi-degrau)")
P("=" * 76)
P(f"  populacao: {len(um)} figuras dos tres lotes com a entrada desenhada")
P()
P("1. ENTREGA")
P("-" * 76)
est = sum(1 for x in resp if x["order_hat"] == x["order_true"])
P(f"  calibrou os dois eixos       {sum(x['cal_ok'] for x in um)}/{len(um)}"
  f"  ({sum(x['cal_ok'] for x in um)/len(um):.1%})")
P(f"  entregou parametro fisico    {len(ok)}/{len(um)}  ({len(ok)/len(um):.1%})")
P(f"  rotulo de estrutura correto  {est}/{len(resp)}  ({est/len(resp):.1%})")
P(f"  sinal de K correto           {sum(1 for x in ok if x.get('sinal_K_ok'))}/{len(ok)}"
  f"  ({sum(1 for x in ok if x.get('sinal_K_ok'))/len(ok):.1%})")
P()
c = Counter(x["reason"] for x in um if not x["ok"])
P(f"  recusas: {len(um)-len(ok)}  " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
P()

P("2. EXATIDAO DOS PARAMETROS (erro relativo, so as que entregaram)")
P("-" * 76)
P(f"  {'grandeza':<34}{'n':>5}{'mediana':>11}{'p90':>11}")
for k, rot in (("err_K", "K"),
               ("err_t_dom", "t_dom (unifica as ordens)"),
               ("err_tau", "tau (quando FOPDT dos dois lados)"),
               ("err_wn", "wn (quando 2a ordem dos dois lados)"),
               ("err_zeta", "zeta (idem)"),
               ("err_theta_rel", "theta (relativo)"),
               ("err_theta_T", "theta (|dtheta| / janela)")):
    n = len([x for x in ok if x.get(k) is not None])
    P(f"  {rot:<34}{n:>5}{med(ok,k):>11.4f}{p90(ok,k):>11.4f}")
P()
P("  erro de CURVA reconstruida (NRMSE contra a verdade analitica):")
for lim in (0.01, 0.02, 0.05, 0.10):
    c2 = sum(1 for x in ok if x.get("nrmse_curva") is not None and x["nrmse_curva"] <= lim)
    P(f"    NRMSE <= {lim:.2f}:  {c2}/{len(ok)} das entregues ({c2/len(ok):5.1%})"
      f"   ·   {c2}/{len(um)} do total ({c2/len(um):5.1%})")
P()

P("3. AS DUAS SUBPOPULACOES (a truncagem espuria parte a amostra em duas)")
P("-" * 76)
lim = [x for x in ok if x["truncado_em"] is None]
tru = [x for x in ok if x["truncado_em"] is not None]
P(f"  {'':<30}{'n':>5}{'|errK| p50':>12}{'NRMSE p50':>12}{'estrutura':>12}")
for rot, g in (("sem truncagem espuria", lim), ("COM truncagem espuria", tru)):
    e = sum(1 for x in g if x["order_hat"] == x["order_true"])
    P(f"  {rot:<30}{len(g):>5}{med(g,'err_K'):>12.4f}"
      f"{med(g,'nrmse_curva'):>12.4f}{e}/{len(g):<12}")
P()
P(f"  a truncagem espuria multiplica o erro de K por "
  f"{med(tru,'err_K')/max(med(lim,'err_K'),1e-9):.0f}x")
P()

P("4. O NUCLEO LIMPO — um degrau, sem truncagem espuria")
P("-" * 76)
P(f"  n = {len(lim)} de {len(um)} ({len(lim)/len(um):.1%} da populacao de um degrau)")
e = sum(1 for x in lim if x["order_hat"] == x["order_true"])
P(f"  estrutura correta      {e}/{len(lim)} ({e/len(lim):.1%})")
for k, rot in (("err_K", "K"), ("err_t_dom", "t_dom"), ("err_theta_T", "theta/janela")):
    P(f"  {rot:<22} mediana {med(lim,k):.4f}   p90 {p90(lim,k):.4f}")
for l2 in (0.01, 0.02, 0.05):
    c2 = sum(1 for x in lim if x["nrmse_curva"] <= l2)
    P(f"  NRMSE <= {l2:.2f}          {c2}/{len(lim)} ({c2/len(lim):.1%})")
P()

P("5. ROBUSTEZ AO RENDER (a variacao de estilo move a agulha?)")
P("-" * 76)
def estrato(chave, rot):
    P(f"  por {rot}:")
    gr = {}
    for x in um:
        gr.setdefault(x[chave], []).append(x)
    for k in sorted(gr, key=str):
        g = gr[k]; g_ok = [x for x in g if x["ok"]]
        if not g_ok:
            P(f"    {str(k):<16} n={len(g):<4d} fisico=0%")
            continue
        P(f"    {str(k):<16} n={len(g):<4d} fisico={len(g_ok)/len(g):6.1%}"
          f"  |errK| p50={med(g_ok,'err_K'):7.4f}  NRMSE p50={med(g_ok,'nrmse_curva'):7.4f}")
    P()
estrato("linestyle", "tipo de linha")
estrato("fundo_escuro", "tema")
estrato("preenchimento", "preenchimento")
estrato("order_true", "ordem verdadeira")

P("6. CONTRASTE — a mesma populacao SEM a linha de entrada desenhada")
P("-" * 76)
sok = [x for x in sem if x["ok"]]
sest = sum(1 for x in sem if x["respondeu"] and x["order_hat"] == x["order_true"])
srsp = sum(1 for x in sem if x["respondeu"])
stru = sum(1 for x in sok if x["truncado_em"] is not None)
ctru = sum(1 for x in ok if x["truncado_em"] is not None)
P(f"  {'':<26}{'n':>5}{'fisico':>9}{'estrut':>9}{'|errK| p50':>12}{'trunc espuria':>15}")
P(f"  {'COM entrada (hoje)':<26}{len(um):>5}{len(ok)/len(um):>9.1%}"
  f"{est/len(resp):>9.1%}{med(ok,'err_K'):>12.4f}{ctru:>9}/{len(ok):<5}")
P(f"  {'SEM entrada':<26}{len(sem):>5}{len(sok)/len(sem):>9.1%}"
  f"{sest/max(1,srsp):>9.1%}{med(sok,'err_K'):>12.4f}{stru:>9}/{len(sok):<5}")
P()
P("  A diferenca entre as duas linhas e o CUSTO do distrator, nao um defeito")
P("  da fisica: a mesma planta, o mesmo estilo, o mesmo ajuste.")

txt = "\n".join(L)
(RAIZ / "assertividade_1degrau.txt").write_text(txt, encoding="utf-8")
print(txt)
