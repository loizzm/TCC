"""Cruza o erro por SINAL DE K e por NUMERO DE DEGRAUS.

Le os dois lotes de `rg_aleatorio.py`:
  - `reports/amostras_aleatorias`            (33 por familia de gerador)
  - `reports/amostras_aleatorias/balanceado` (fatorial 2x2, render sorteado)

O lote por familia responde "como cada gerador se sai", mas confunde as duas
variaveis: `neg` e 1 degrau negativo por construcao e `rg` e 2 degraus, entao a
celula (K>0, 1 degrau) fica com n=2 nele. O lote balanceado existe para
desfazer esse confundimento. As tabelas saem para os dois lotes e para a uniao.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, fisher_exact, mannwhitneyu

RAIZ = Path(__file__).resolve().parent / "reports" / "amostras_aleatorias"
LOTES = {"familia": RAIZ, "balanceado": RAIZ / "balanceado",
         "balanceado2": RAIZ / "balanceado2"}


def carrega(d: Path) -> list[dict]:
    an = json.loads((d / "analise.json").read_text())
    ver = {v["arquivo"]: v for v in json.loads((d / "verdade.json").read_text())}
    for x in an:
        v = ver[x["arquivo"]]
        x["sinal"] = "K>0" if v["K"] > 0 else "K<0"
        x["degraus_rot"] = "1 degrau" if v["n_degraus"] == 1 else "2+ degraus"
        x["K_total"] = float(v["K_planta"] * sum(u for u, _ in v["degraus"]))
        x["K_planta"] = float(v["K_planta"])
        x["lote"] = d.name
    return an


def med(g, chave, absoluto=True):
    v = [abs(x[chave]) if absoluto else x[chave]
         for x in g if x.get(chave) is not None]
    return float(np.median(v)) if v else float("nan")


def linha(rot, g):
    if not g:
        return f"  {rot:<26} n=0"
    ok = [x for x in g if x["ok"]]
    resp = [x for x in g if x["respondeu"]]
    est = sum(1 for x in resp if x["order_hat"] == x["order_true"])
    n5 = sum(1 for x in ok if x["nrmse_curva"] <= 0.05)
    trunc = sum(1 for x in ok if x["truncado_em"] is not None)
    return (f"  {rot:<26} n={len(g):<4d}"
            f" fis={len(ok)/len(g):6.1%}"
            f" estr={est}/{len(resp):<3d}"
            f" NRMSE med={med(ok,'nrmse_curva'):7.4f}"
            f" |errK| med={med(ok,'err_K'):7.4f}"
            f" |errK|>50%={sum(1 for x in ok if abs(x['err_K'])>0.5):>2d}/{len(ok):<3d}"
            f" |dth|/T={med(ok,'err_theta_T'):6.4f}"
            f" trunc={trunc}/{len(ok)}")


def tabela(dados, titulo):
    print("=" * 118)
    print(titulo)
    print("=" * 118)
    print(linha("TODAS", dados))
    print()
    print("  -- so pelo SINAL DE K")
    for s in ("K>0", "K<0"):
        print(linha(s, [x for x in dados if x["sinal"] == s]))
    print()
    print("  -- so pelo NUMERO DE DEGRAUS")
    for d in ("1 degrau", "2+ degraus"):
        print(linha(d, [x for x in dados if x["degraus_rot"] == d]))
    print()
    print("  -- CRUZADO (e aqui que o confundimento se desfaz)")
    for s in ("K>0", "K<0"):
        for d in ("1 degrau", "2+ degraus"):
            print(linha(f"{s} · {d}",
                        [x for x in dados if x["sinal"] == s and x["degraus_rot"] == d]))
    print()


def _cmh(tabelas):
    """Cochran-Mantel-Haenszel: razao de chances comum aos estratos, e p.

    Serve para responder "o sinal de K importa DEPOIS de controlar o resto?".
    Sem estratificar, a resposta sai errada: o sorteio deixou `K_planta < 1`
    mais frequente entre as amostras de K > 0, e e `K_planta < 1` — nao o
    sinal — que move a recusa.
    """
    num = den = a_s = e_s = v_s = 0.0
    for a, b, c, d in tabelas:
        n = a + b + c + d
        if n <= 1:
            continue
        num += a * d / n
        den += b * c / n
        a_s += a
        e_s += (a + b) * (a + c) / n
        v_s += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    stat = (abs(a_s - e_s) - 0.5) ** 2 / v_s if v_s > 0 else 0.0
    return (num / den if den else float("nan")), float(chi2.sf(stat, 1))


def _testes(dados) -> str:
    L = []
    for x in dados:
        x["kpos"] = x["K_true"] > 0
    um = [x for x in dados if x["n_degraus"] == 1]

    # 1. resposta menor que o degrau desenhado (K_planta < 1) -> recusa
    tabs = []
    L.append("  A) 'resposta MENOR que o degrau desenhado' (K_planta<1) -> RECUSA")
    for prim, rot1 in ((True, "K>0"), (False, "K<0")):
        for umd, rot2 in ((True, "1 degrau"), (False, "2+ degraus")):
            g = [x for x in dados if x["kpos"] == prim and (x["n_degraus"] == 1) == umd]
            lo = [x for x in g if x["K_planta"] < 1]
            hi = [x for x in g if x["K_planta"] >= 1]
            a, c = sum(1 for x in lo if not x["ok"]), sum(1 for x in hi if not x["ok"])
            tabs.append((a, len(lo) - a, c, len(hi) - c))
            L.append(f"     {rot1} · {rot2:<11} K_planta<1: {a:>2}/{len(lo):<3}"
                     f" ({a/max(1,len(lo)):5.1%})   >=1: {c:>2}/{len(hi):<3}"
                     f" ({c/max(1,len(hi)):5.1%})")
    orc, p = _cmh(tabs)
    L.append(f"     CMH sobre os 4 estratos: OR={orc:.2f}  p={p:.5f}"
             + ("   <- EFEITO REAL" if p < 0.05 else "   (nao significativo)"))
    L.append("")

    # 2. sinal de K -> recusa, controlando K_planta<1
    tabs = []
    L.append("  B) SINAL DE K -> RECUSA, controlando K_planta<1")
    for kp, rot1 in ((True, "K_planta<1 "), (False, "K_planta>=1")):
        for umd, rot2 in ((True, "1 degrau"), (False, "2+ degraus")):
            g = [x for x in dados if (x["K_planta"] < 1) == kp
                 and (x["n_degraus"] == 1) == umd]
            pos = [x for x in g if x["kpos"]]
            neg = [x for x in g if not x["kpos"]]
            a, c = sum(1 for x in pos if not x["ok"]), sum(1 for x in neg if not x["ok"])
            tabs.append((a, len(pos) - a, c, len(neg) - c))
            L.append(f"     {rot1} · {rot2:<11} K>0: {a:>2}/{len(pos):<3}"
                     f"   K<0: {c:>2}/{len(neg):<3}")
    orc, p = _cmh(tabs)
    L.append(f"     CMH: OR={orc:.2f}  p={p:.5f}"
             + ("   <- EFEITO REAL" if p < 0.05 else "   (NAO significativo)"))
    L.append("")

    # 3. sinal -> exatidao
    L.append("  C) SINAL DE K -> EXATIDAO (Mann-Whitney, so as que entregaram fisica)")
    for umd, rot in ((True, "1 degrau"), (False, "2+ degraus")):
        g = [x for x in dados if x["ok"] and (x["n_degraus"] == 1) == umd]
        for chave, nome in (("err_K", "|errK|"), ("nrmse_curva", "NRMSE ")):
            a = [abs(x[chave]) for x in g if x["kpos"]]
            b = [abs(x[chave]) for x in g if not x["kpos"]]
            L.append(f"     {rot:<11} {nome} K>0={np.median(a):.4f} (n={len(a)})"
                     f"  K<0={np.median(b):.4f} (n={len(b)})"
                     f"  p={mannwhitneyu(a, b).pvalue:.3f}")
    L.append("")

    # 4. truncagem espuria por sinal
    L.append("  D) TRUNCAGEM ESPURIA (1 degrau) -> SINAL DE K")
    ok = [x for x in um if x["ok"]]
    pos = [x for x in ok if x["kpos"]]
    neg = [x for x in ok if not x["kpos"]]
    a = sum(1 for x in pos if x["truncado_em"] is not None)
    c = sum(1 for x in neg if x["truncado_em"] is not None)
    _, p = fisher_exact([[a, len(pos) - a], [c, len(neg) - c]])
    L.append(f"     K>0: {a}/{len(pos)} ({a/len(pos):.1%})   "
             f"K<0: {c}/{len(neg)} ({c/len(neg):.1%})   Fisher p={p:.4f}"
             + ("   <- EFEITO REAL" if p < 0.05 else "   (NAO significativo)"))
    L.append("     confundidor: o render escuro so aparece na familia `neg`:")
    for rot, g in (("K>0", pos), ("K<0", neg)):
        for esc in (True, False):
            s2 = [x for x in g if x["fundo_escuro"] == esc]
            if not s2:
                continue
            t = sum(1 for x in s2 if x["truncado_em"] is not None)
            L.append(f"       {rot} fundo_escuro={str(esc):<5} {t}/{len(s2)} ({t/len(s2):5.1%})")
    L.append("")

    # 5. revocacao por sinal
    L.append("  E) REVOCACAO DA TRUNCAGEM (2+ degraus) -> SINAL DE K")
    mu = [x for x in dados if x["n_degraus"] > 1 and x["ok"]]
    pos = [x for x in mu if x["kpos"]]
    neg = [x for x in mu if not x["kpos"]]
    a = sum(1 for x in pos if x["truncado_em"] is not None)
    c = sum(1 for x in neg if x["truncado_em"] is not None)
    _, p = fisher_exact([[a, len(pos) - a], [c, len(neg) - c]])
    L.append(f"     K>0: {a}/{len(pos)} ({a/len(pos):.1%})   "
             f"K<0: {c}/{len(neg)} ({c/len(neg):.1%})   Fisher p={p:.4f}"
             + ("   <- EFEITO REAL" if p < 0.05 else "   (NAO significativo)"))
    return "\n".join(L)


def main() -> None:
    lotes = {k: carrega(v) for k, v in LOTES.items()}
    for k, v in lotes.items():
        tabela(v, f"LOTE {k.upper()}  (n={len(v)})")
    uniao = [x for v in lotes.values() for x in v]
    tabela(uniao, f"UNIAO DOS DOIS LOTES  (n={len(uniao)})")

    print("=" * 118)
    print("RECUSAS por celula (a pipeline nomeou a causa em vez de errar calado)")
    print("=" * 118)
    from collections import Counter
    for s in ("K>0", "K<0"):
        for d in ("1 degrau", "2+ degraus"):
            g = [x for x in uniao if x["sinal"] == s and x["degraus_rot"] == d]
            r = Counter(x["reason"] for x in g if not x["ok"])
            tot = sum(r.values())
            print(f"  {s} · {d:<11} {tot:>2d}/{len(g):<3d} "
                  + (", ".join(f"{k}={c}" for k, c in r.most_common()) or "—"))
    print()

    print("=" * 118)
    print("MULTI-DEGRAU: o K reportado descreve o 1o degrau ou a EXCURSAO TOTAL?")
    print("=" * 118)
    for s in ("K>0", "K<0"):
        for rot, cond in (("truncou", lambda x: x["truncado_em"] is not None),
                          ("nao truncou", lambda x: x["truncado_em"] is None)):
            g = [x for x in uniao if x["ok"] and x["degraus_rot"] == "2+ degraus"
                 and x["sinal"] == s and cond(x)]
            if not g:
                continue
            e1 = [abs(x["K_hat"] - x["K_true"]) / abs(x["K_true"]) for x in g]
            eT = [abs(x["K_hat"] - x["K_total"]) / abs(x["K_total"]) for x in g]
            print(f"  {s} · {rot:<12} n={len(g):<3d} "
                  f"erro vs K do 1o degrau={np.median(e1):6.3f}   "
                  f"erro vs K TOTAL={np.median(eT):6.3f}   "
                  f"mais perto do total em {sum(1 for a,b in zip(e1,eT) if b<a)}/{len(g)}")
    print()

    print("=" * 118)
    print("TESTES: o sinal de K importa, depois de controlar o resto?")
    print("=" * 118)
    print(_testes(uniao))
    print()

    print("=" * 118)
    print("TRUNCAGEM como detector de multi-degrau, por sinal de K")
    print("=" * 118)
    for s in ("K>0", "K<0"):
        g = [x for x in uniao if x["ok"] and x["sinal"] == s]
        TP = sum(1 for x in g if x["degraus_rot"] == "2+ degraus" and x["truncado_em"] is not None)
        FN = sum(1 for x in g if x["degraus_rot"] == "2+ degraus" and x["truncado_em"] is None)
        FP = sum(1 for x in g if x["degraus_rot"] == "1 degrau" and x["truncado_em"] is not None)
        TN = sum(1 for x in g if x["degraus_rot"] == "1 degrau" and x["truncado_em"] is None)
        print(f"  {s}  TP={TP:<3d} FN={FN:<3d} FP={FP:<3d} TN={TN:<3d}  "
              f"precisao={TP/max(1,TP+FP):6.1%}  revocacao={TP/max(1,TP+FN):6.1%}")
    print()


if __name__ == "__main__":
    main()
