"""Taxa de acerto CONJUNTIVA: quantas figuras acertam TUDO ao mesmo tempo.

Todas as metricas do relatorio sao marginais — entrega, estrutura, erro de K,
erro de theta, NRMSE — e cada uma isolada parece boa. A pergunta que este
arquivo responde e outra: em quantas figuras o sistema acerta TODAS de uma vez.
E o numero que importa para quem vai USAR a saida, porque um `K` certo com a
estrutura errada nao serve.

Reporta tambem, entre as que reprovam, QUAL criterio foi o binding — sem isso
o numero conjuntivo diz que falhou, mas nao onde investir.

`--dir` aceita qualquer lote com `analise.json`.

NIVEL UNICO: ESTRITO
--------------------
Havia tres niveis (ESTRITO, PRATICO, TOLERANTE). Os dois mais frouxos foram
REMOVIDOS, e com eles a metrica `t_dom` (constante de tempo dominante), que
substituia a exigencia de estrutura exata nos dois.

POR QUE, medido em 700 figuras dos quatro lotes de controle:

  1. `t_dom` era o criterio MAIS SEVERO de todos em 24 figuras, nas quais ele
     reprovava sozinho — com tudo o mais aprovado e o NRMSE da curva
     reconstruida em 0,0022 a 0,0045. Reprovar uma reconstrucao de 0,2 % pela
     constante de tempo e medir outra coisa que nao a identificacao.
  2. A causa e' de PARAMETRIZACAO, nao do estimador. `t_dom` para 2a ordem e'
     `(zeta + sqrt(zeta^2-1))/wn`, cuja derivada em zeta e'

         d(t_dom)/d(zeta) = (1 + zeta/sqrt(zeta^2-1)) / wn

     SINGULAR em zeta = 1. Um erro de 0,05 em zeta custa 35,3 % de `t_dom` em
     zeta = 1,01 e 1,0 % em zeta = 5. Perto do amortecimento critico os dois
     polos quase coincidem e separa-los em "lento" e "rapido" e' mal-posto.
  3. Os dados acompanham: entre as de 2a ordem, 50 % das que reprovavam so por
     `t_dom` tinham 0,8 <= zeta <= 1,5, contra 27 % das que passavam (zeta
     mediano 1,08 contra 1,34). Fisher OR = 2,74, p = 0,070 — indicio com
     n = 14, nao prova.

  Hipotese testada e REFUTADA: janela curta. `janela_ef` mediana e' 2,78 nas
  que reprovam e 2,79 nas que passam.

O QUE SE PERDE COM A REMOCAO, dito por inteiro. `t_dom` existia porque o
rotulo de estrutura NAO e' observavel da saida quando as duas estruturas sao
equivalentes: um polo lento de 2a ordem sobreamortecida e' um `tau` de FOPDT, e
o tempo morto absorve o polo rapido (medido: com zeta = 2 o melhor FOPDT tem
theta = 0,250 contra tau_rapido = 0,268, e o residuo cai para 0,0016). Nas 700
figuras, 37 reprovam o ESTRITO SO por estrutura, todas com `t_dom` a menos de
15 % e NRMSE de curva mediano 0,0037 — sao 5,3 % do corpus reprovadas por um
NOME, com a dinamica reproduzida a 0,4 %.

Esse custo fica, e e' deliberado: o ESTRITO e' o nivel que exige o rotulo
certo, e essa exigencia e' o enunciado dele, nao um defeito. Quem quiser medir
a ambiguidade estrutural deve olhar a matriz de confusao e o NRMSE de curva em
`analisa_aleatorias.py`, que continuam reportando `err_t_dom` como
DIAGNOSTICO. O que saiu foi o uso dele como CRITERIO DE APROVACAO.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Nivel unico de exigencia. `estrutura=True` significa rotulo exato: o ESTRITO
# nao aceita troca de estrutura, nem quando ela e' dinamicamente inocua.
ESTRITO = dict(K=0.05, theta_T=0.02, nrmse=0.02, estrutura=True)


def avalia(x, c=ESTRITO):
    """Lista dos criterios REPROVADOS por esta figura. Vazia = aprovada."""
    f = []
    if not x["ok"]:
        return ["nao entregou (" + str(x["reason"]) + ")"]
    if c["estrutura"] and x["order_hat"] != x["order_true"]:
        f.append("estrutura")
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

    c = ESTRITO
    print(f"ACERTO CONJUNTIVO — todas as metricas na mesma figura   n={len(A)}\n")
    print(f"  ESTRITO  (entregou, estrutura exata, sinal de K, "
          f"K<={c['K']:.0%}, theta/T<={c['theta_T']:.0%}, NRMSE<={c['nrmse']:.0%})")
    bons = sum(1 for x in A if not avalia(x, c))
    print(f"    {'TODAS':<12} {bons:>3}/{len(A):<4} = {bons/len(A):6.1%}")
    print()

    print("  QUAL CRITERIO REPROVA (contando cada reprovacao):")
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


if __name__ == "__main__":
    main()
