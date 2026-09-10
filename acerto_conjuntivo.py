"""Taxa de acerto CONJUNTIVA: quantas figuras acertam TUDO ao mesmo tempo.

Todas as metricas do relatorio sao marginais — entrega, estrutura, erro de K,
erro de theta, NRMSE — e cada uma isolada parece boa. A pergunta que este
arquivo responde e outra: em quantas figuras o sistema acerta TODAS de uma vez.
E o numero que importa para quem vai USAR a saida, porque um `K` certo com a
estrutura errada nao serve.

Reporta tambem, entre as que reprovam, QUAL criterio foi o binding — sem isso
o numero conjuntivo diz que falhou, mas nao onde investir.

`--dir` aceita qualquer lote com `analise.json`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# Niveis de exigencia. `t_dom` entra no lugar da estrutura no nivel PRATICO
# porque trocar o rotulo sem mudar a dinamica nao e erro de uso — um polo lento
# de 2a ordem superamortecida E um `tau` de FOPDT (ver ARQUITETURA.md).
NIVEIS = {
    "ESTRITO":  dict(K=0.05, theta_T=0.02, nrmse=0.02, estrutura=True,  t_dom=None),
    "PRATICO":  dict(K=0.10, theta_T=0.05, nrmse=0.05, estrutura=False, t_dom=0.15),
    "TOLERANTE":dict(K=0.20, theta_T=0.10, nrmse=0.10, estrutura=False, t_dom=0.30),
}


def avalia(x, c):
    """Lista dos criterios REPROVADOS por esta figura."""
    f = []
    if not x["ok"]:
        return ["nao entregou (" + str(x["reason"]) + ")"]
    if c["estrutura"] and x["order_hat"] != x["order_true"]:
        f.append("estrutura")
    if c["t_dom"] is not None:
        e = x.get("err_t_dom")
        if e is None or abs(e) > c["t_dom"]:
            f.append("t_dom")
    if not x.get("sinal_K_ok"):
        f.append("sinal de K")
    e = x.get("err_K")
    if e is None or abs(e) > c["K"]:
        f.append("K")
    e = x.get("err_theta_T")
    if e is None or abs(e) > c["theta_T"]:
        f.append("theta")
    e = x.get("nrmse_curva")
    if e is None or e > c["nrmse"]:
        f.append("curva")
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, action="append", required=True)
    a = ap.parse_args()
    A = []
    for d in a.dir:
        for x in json.loads((d / "analise.json").read_text()):
            x["_lote"] = d.name
            A.append(x)

    print(f"ACERTO CONJUNTIVO — todas as metricas na mesma figura   n={len(A)}\n")
    for nome, c in NIVEIS.items():
        crit = (f"K<={c['K']:.0%}, theta/T<={c['theta_T']:.0%}, "
                f"NRMSE<={c['nrmse']:.0%}, "
                + ("estrutura exata" if c["estrutura"] else f"t_dom<={c['t_dom']:.0%}"))
        print(f"  {nome}  ({crit})")
        for rot, g in (("TODAS", A),
                       ("1 degrau", [x for x in A if x["n_degraus"] == 1]),
                       ("2+ degraus", [x for x in A if x["n_degraus"] > 1])):
            if not g:          # lote de um degrau so: o grupo multi fica vazio
                continue
            bons = sum(1 for x in g if not avalia(x, c))
            print(f"    {rot:<12} {bons:>3}/{len(g):<4} = {bons/len(g):6.1%}")
        print()

    print("  QUAL CRITERIO REPROVA (nivel PRATICO, contando cada reprovacao):")
    c = NIVEIS["PRATICO"]
    cont = Counter()
    sozinho = Counter()
    for x in A:
        f = avalia(x, c)
        for k in f:
            cont[k.split(" (")[0]] += 1
        if len(f) == 1:
            sozinho[f[0].split(" (")[0]] += 1
    n_ruins = sum(1 for x in A if avalia(x, c))
    print(f"    figuras que reprovam: {n_ruins}/{len(A)}")
    print(f"    {'criterio':<22}{'reprova':>9}{'sozinho':>10}")
    for k, v in cont.most_common():
        print(f"    {k:<22}{v:>9}{sozinho.get(k, 0):>10}")
    print()
    print("    'sozinho' = era o UNICO criterio reprovado; corrigi-lo salvaria a figura.")
    print()
    print("  E se o multi-degrau fosse resolvido? (so as de 1 degrau, nivel PRATICO)")
    g = [x for x in A if x["n_degraus"] == 1]
    if g:
        bons = sum(1 for x in g if not avalia(x, c))
        print(f"    {bons}/{len(g)} = {bons/len(g):.1%}")


if __name__ == "__main__":
    main()
