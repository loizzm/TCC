"""Taxa de acerto CONJUNTIVA: quantas figuras acertam TUDO ao mesmo tempo.

Todas as metricas do relatorio sao marginais — entrega, estrutura, erro de K,
erro de theta, NRMSE — e cada uma isolada parece boa. A pergunta que este
arquivo responde e outra: em quantas figuras o sistema acerta TODAS de uma vez.
E o numero que importa para quem vai USAR a saida, porque um `K` certo com a
estrutura errada nao serve.

Reporta tambem, entre as que reprovam, QUAL criterio foi o binding — sem isso
o numero conjuntivo diz que falhou, mas nao onde investir.

`--dir` aceita qualquer lote com `analise.json`.

NIVEL ESTRITO
-------------
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

NIVEL ESTENDIDO
---------------
Reportado AO LADO do ESTRITO, nao no lugar dele: entregou, estrutura exata e
NRMSE da curva < 2 % num horizonte de theta + 6 t_dom (`avalia_estendido`).
Nao tem limiar proprio de K nem de theta.

POR QUE, medido no `lote_misto2` (400 figuras, modelo `unet_stageA.pt`):

  1. O NRMSE na JANELA e cego a erro de extrapolacao. Com a janela curta,
     (K, zeta, theta) deslizam juntos e desenham a mesma subida com patamares
     diferentes: `ruido_Kmaior1_15dB_17` tem K errado em 52 % com NRMSE de
     janela 0,6 % (estendido: 25 %). Um nivel "estrutura + NRMSE de janela"
     aprovaria 18 figuras com |erro de K| > 5 %, ate 52 %.
  2. O NRMSE ESTENDIDO absorve K: entre as aprovadas, nenhuma tem |erro de K|
     acima de 5 % (max 3,9 %). Ele e MAIS exigente que o K <= 5 % do ESTRITO —
     um erro puro de 5 % em K da NRMSE 4,3 % em 6 t_dom — e por isso reprova 9
     figuras que o ESTRITO aprova, todas com K errado entre 3 % e 5 %.
  3. theta NAO e absorvido por inteiro: erro de atraso so existe no
     transitorio e se dilui no regime permanente. Passam 4 figuras com
     |dtheta|/T entre 1,9 % e 2,8 %.
  4. Mesmo veredito que o ESTRITO em 96,8 % das figuras (75,0 % contra 76,2 %),
     com DOIS criterios em vez de quatro.

POR QUE m = 6, pela varredura que `main` imprime (SENSIBILIDADE AO HORIZONTE):
e o MENOR horizonte em que nenhuma aprovada erra K acima de 5 % — em 4,6 t_dom
(o assentamento a 1 %) passam 3, ate 9,6 %; em 5, passa 1. Acima de 6 o nivel
so fica mais severo com K (max aprovado 2,5 % em 12 t_dom) sem ganho em theta.

NIVEL PRATICO
-------------
O ESTENDIDO sem exigir o ROTULO: a troca FOPDT <-> 2a ordem e aceita nos dois
sentidos, e a curva estendida continua sendo a guarda da dinamica.

NAO e o PRATICO removido acima. Aquele trocava a estrutura por `t_dom`, que e
singular em zeta = 1; este nao poe metrica de parametro nenhuma no lugar — quem
julga a troca e a curva ate theta + 6 t_dom.

Nem theta entra, e de proposito: na troca 2a ordem -> FOPDT o theta ajustado
e ATRASO APARENTE, que absorve o polo rapido (a "regra da metade" do SIMC faz
essa aproximacao deliberadamente). Cobrar theta contra o atraso verdadeiro
reprovaria a troca pela propria definicao dela.

MEDIDO no `lote_misto2` (400 figuras, modelo `unet_stageA.pt`):

  1. 337/400 = 84,2 %, contra 300 (ESTENDIDO) e 305 (ESTRITO). As 37 a mais
     sao todas trocas de rotulo, com NRMSE estendido mediano 0,96 %.
  2. As trocas 2a ordem -> FOPDT aprovadas tem zeta verdadeiro entre 1,18 e
     2,44 (mediana 2,01): TODAS sobreamortecidas, a regiao em que as duas
     estruturas sao observacionalmente equivalentes. Nenhuma subamortecida
     passa — o sobressinal que o FOPDT nao tem estoura a curva.
  3. Entre as aprovadas, nenhuma com |erro de K| > 5 % (max 3,9 %). Passam 19
     com |dtheta|/T > 2 % (max 9,1 %), 15 delas trocas: e o atraso aparente.
  4. As 14 trocas que ainda reprovam tem NRMSE estendido mediano 5,5 % e estao
     todas em SNR <= 15 dB.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from analisa_aleatorias import HORIZONTE_T_DOM, HORIZONTES_SENSIBILIDADE

# Nivel ESTRITO. `estrutura=True` significa rotulo exato: o ESTRITO
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


# Nivel ESTENDIDO: estrutura exata + NRMSE da curva ate theta + 6 t_dom.
# Limiar ESTRITO (`<`), como foi pedido — a fronteira de 2 % reprova.
ESTENDIDO = dict(nrmse=0.02, estrutura=True)

# Nivel PRATICO: o ESTENDIDO sem exigir o ROTULO de estrutura. A troca FOPDT <->
# 2a ordem e aceita nos dois sentidos; o que continua valendo e a curva
# estendida, que reprova a troca quando ela muda a dinamica (sobressinal que o
# FOPDT nao tem, patamar diferente, atraso que o polo rapido nao explica).
PRATICO = dict(nrmse=0.02, estrutura=False)


def avalia_estendido(x, c=ESTENDIDO, horizonte=None):
    """Criterios reprovados no nivel ESTENDIDO (ou PRATICO, com `c=PRATICO`).
    Vazia = aprovada.

    `horizonte=None` le `nrmse_estendido` (o horizonte padrao de
    `analisa_aleatorias.HORIZONTE_T_DOM`); um numero le a entrada
    correspondente de `nrmse_horizontes`, para a varredura de sensibilidade.
    """
    if not x["ok"]:
        return ["nao entregou (" + str(x["reason"]) + ")"]
    f = []
    if c["estrutura"] and x["order_hat"] != x["order_true"]:
        f.append("estrutura")
    e = (x.get("nrmse_estendido") if horizonte is None
         else (x.get("nrmse_horizontes") or {}).get(str(float(horizonte))))
    if e is None or not e < c["nrmse"]:
        f.append("curva estendida")
    return f


def _quem_reprova(A, fn) -> None:
    cont = Counter()
    sozinho = Counter()
    for x in A:
        f = fn(x)
        for k in f:
            cont[k.split(" (")[0]] += 1
        if len(f) == 1:
            sozinho[f[0].split(" (")[0]] += 1
    n_ruins = sum(1 for x in A if fn(x))
    print(f"    figuras que reprovam: {n_ruins}/{len(A)}")
    print(f"    {'criterio':<22}{'reprova':>9}{'sozinho':>10}")
    for k, v in cont.most_common():
        print(f"    {k:<22}{v:>9}{sozinho.get(k, 0):>10}")


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
    _quem_reprova(A, lambda x: avalia(x, c))
    print()
    print("    'sozinho' = era o UNICO criterio reprovado; corrigi-lo salvaria a figura.")
    print()

    if not any("nrmse_estendido" in x for x in A):
        print("  ESTENDIDO: analise.json sem `nrmse_estendido` — rode "
              "`analisa_aleatorias.py --dir` de novo neste lote.")
        return

    for nome, e, desc in (
            ("ESTENDIDO", ESTENDIDO, "estrutura exata"),
            ("PRATICO", PRATICO, "troca FOPDT <-> 2a ordem aceita")):
        print(f"  {nome}  (entregou, {desc}, NRMSE ate theta + "
              f"{HORIZONTE_T_DOM:g} t_dom < {e['nrmse']:.0%})")
        bons_e = sum(1 for x in A if not avalia_estendido(x, e))
        print(f"    {'TODAS':<12} {bons_e:>3}/{len(A):<4} = {bons_e/len(A):6.1%}")
        print()
        print("  QUAL CRITERIO REPROVA (contando cada reprovacao):")
        _quem_reprova(A, lambda x, e=e: avalia_estendido(x, e))
        print()
        _sensibilidade(A, e)
        print()

    _trocas_no_pratico(A)


def _sensibilidade(A, e) -> None:
    # K e theta nao sao criterio nos niveis estendidos: o que mostra se o NRMSE
    # estendido os cobre e o erro que sobra ENTRE AS APROVADAS, em cada horizonte.
    c = ESTRITO
    print("  SENSIBILIDADE AO HORIZONTE (m = horizonte em t_dom a partir de theta)")
    print(f"    {'m':>5} {'aprova':>13} {'|eK|>5%':>8} {'max|eK|':>8} "
          f"{'|dth/T|>2%':>11} {'max|dth/T|':>11} {'=ESTRITO':>9}")
    for m in HORIZONTES_SENSIBILIDADE:
        ap = [x for x in A if not avalia_estendido(x, e, horizonte=m)]
        ek = [abs(x["err_K"]) for x in ap if x.get("err_K") is not None]
        et = [abs(x["err_theta_T"]) for x in ap if x.get("err_theta_T") is not None]
        igual = sum(1 for x in A
                    if bool(avalia(x, c)) == bool(avalia_estendido(x, e, horizonte=m)))
        marca = "  <- padrao" if m == HORIZONTE_T_DOM else ""
        print(f"    {m:>5g} {len(ap):>4}/{len(A):<3} {len(ap)/len(A):5.1%} "
              f"{sum(v > c['K'] for v in ek):>8} {max(ek, default=0):>8.1%} "
              f"{sum(v > c['theta_T'] for v in et):>11} {max(et, default=0):>11.1%} "
              f"{igual/len(A):>9.1%}{marca}")
    print()
    print("    |eK|>5% e |dth/T|>2% contam APROVADAS que o ESTRITO reprovaria por K/theta;")
    print("    '=ESTRITO' e a fracao de figuras com o mesmo veredito nos dois niveis.")


def _trocas_no_pratico(A) -> None:
    # O que o PRATICO aceita a mais que o ESTENDIDO sao exatamente as trocas de
    # rotulo; aqui, por sentido, quantas passam e com que erro de K e theta.
    # Na troca 2a ordem -> FOPDT o theta ajustado e ATRASO APARENTE (absorve o
    # polo rapido), entao |dth/T| e reportado, nao cobrado.
    print("  TROCAS DE ESTRUTURA NO PRATICO (entregues, por sentido)")
    print(f"    {'verdade -> predito':<20} {'n':>4} {'aprova':>7} "
          f"{'max|eK| ap.':>12} {'max|dth/T| ap.':>15}")
    for vv, pp in (("second", "fopdt"), ("fopdt", "second")):
        g = [x for x in A if x["ok"] and x["order_true"] == vv and x["order_hat"] == pp]
        ap = [x for x in g if not avalia_estendido(x, PRATICO)]
        ek = [abs(x["err_K"]) for x in ap if x.get("err_K") is not None]
        et = [abs(x["err_theta_T"]) for x in ap if x.get("err_theta_T") is not None]
        print(f"    {vv + ' -> ' + pp:<20} {len(g):>4} {len(ap):>7} "
              f"{max(ek, default=0):>12.1%} {max(et, default=0):>15.1%}")

if __name__ == "__main__":
    main()
