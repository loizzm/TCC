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
Entregou, estrutura exata e NRMSE da curva < 2 % num horizonte de theta + 6
t_dom (`avalia`). NAO tem limiar proprio de K nem de theta: o NRMSE estendido
os cobre.

Este nivel SUBSTITUI a versao antiga do ESTRITO, que cobrava K <= 5 %,
theta/T <= 2 %, sinal de K e NRMSE medido so na JANELA desenhada. Tres niveis
chegaram a existir no caminho (o proprio ESTRITO antigo, um ESTENDIDO e um
PRATICO); todos foram descartados em favor deste. POR QUE, medido no
`lote_misto2` (400 figuras):

  1. O NRMSE na JANELA e cego a erro de extrapolacao. Com a janela curta,
     (K, zeta, theta) deslizam juntos e desenham a mesma subida com patamares
     diferentes: `ruido_Kmaior1_15dB_17` tem K errado em 52 % com NRMSE de
     janela 0,6 % (estendido: 25 %). A versao antiga, que media o NRMSE so na
     janela, aprovava 18 figuras com |erro de K| > 5 %, ate 52 %.
  2. O NRMSE ESTENDIDO absorve K: entre as aprovadas, nenhuma tem |erro de K|
     acima de 5 % (max 3,9 %). Ele e MAIS exigente que o K <= 5 % antigo —
     um erro puro de 5 % em K da NRMSE 4,3 % em 6 t_dom.
  3. theta NAO e absorvido por inteiro: erro de atraso so existe no
     transitorio e se dilui no regime permanente. Passam 4 figuras com
     |dtheta|/T entre 1,9 % e 2,8 % — custo aceito, DOIS criterios em vez de
     quatro, mesmo veredito que a versao antiga em 96,8 % das figuras.

POR QUE m = 6, pela varredura que `main` imprime (SENSIBILIDADE AO HORIZONTE):
e o MENOR horizonte em que nenhuma aprovada erra K acima de 5 % — em 4,6 t_dom
(o assentamento a 1 %) passam 3, ate 9,6 %; em 5, passa 1. Acima de 6 o nivel
so fica mais severo com K (max aprovado 2,5 % em 12 t_dom) sem ganho em theta.

A ambiguidade estrutural (um polo lento de 2a ordem sobreamortecida e' um `tau`
de FOPDT) continua reportada como DIAGNOSTICO na matriz de confusao e no
`err_t_dom` de `analisa_aleatorias.py`. Aqui a estrutura exata e EXIGIDA: o
nivel conjuntivo e o que vale para quem usa a saida, e o rotulo faz parte dela.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from analisa_aleatorias import HORIZONTE_T_DOM, HORIZONTES_SENSIBILIDADE

# Nivel ESTRITO: estrutura exata + NRMSE da curva ate theta + 6 t_dom.
# Limiar com `<` — a fronteira de 2 % reprova.
ESTRITO = dict(nrmse=0.02, estrutura=True)

# DOMINIO DE APLICABILIDADE. Abaixo de 15 dB a curva some no ruido: a 5 dB a
# dispersao do patamar e da ordem da propria resposta (ver as figuras
# `ruido_*_05dB_*`), e nem o valor final nem o rotulo de estrutura sao
# observaveis — a limitacao e de INFORMACAO, nao do metodo. Um grafico de
# livro, artigo ou bancada nao chega perto desse ruido. O `lote_misto2` cobre
# 5 a 60 dB de proposito, como teste de estresse; o numero de uso realista e o
# do subconjunto SNR >= 15 dB, reportado AO LADO do global (nunca no lugar).
SNR_MIN_DB = 15


def snr_db(x):
    """SNR em dB lido do nome do arquivo (`..._NNdB_...`). `None` se nao houver
    — lote sem ruido (balanceado, etc.), que entao nao sofre o corte."""
    m = re.search(r"_(\d+)dB", x.get("arquivo", ""))
    return int(m.group(1)) if m else None


def faixa_ganho(x):
    """Grupo de ganho pelo nome: 'K>1' (Kmaior1), 'K<1' (Kmenor1) ou `None`."""
    nome = x.get("arquivo", "")
    if "Kmaior1" in nome:
        return "K>1"
    if "Kmenor1" in nome:
        return "K<1"
    return None


def avalia(x, c=ESTRITO, horizonte=None):
    """Lista dos criterios REPROVADOS por esta figura. Vazia = aprovada.

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
        f.append("curva")
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


def _sensibilidade(A) -> None:
    # K e theta nao sao criterio: o que a varredura mostra e se o NRMSE
    # estendido os cobre e o erro que sobra ENTRE AS APROVADAS, por horizonte.
    print("  SENSIBILIDADE AO HORIZONTE (m = horizonte em t_dom a partir de theta)")
    print(f"    {'m':>5} {'aprova':>13} {'|eK|>5%':>8} {'max|eK|':>8} "
          f"{'|dth/T|>2%':>11} {'max|dth/T|':>11}")
    for m in HORIZONTES_SENSIBILIDADE:
        ap = [x for x in A if not avalia(x, horizonte=m)]
        ek = [abs(x["err_K"]) for x in ap if x.get("err_K") is not None]
        et = [abs(x["err_theta_T"]) for x in ap if x.get("err_theta_T") is not None]
        marca = "  <- padrao" if m == HORIZONTE_T_DOM else ""
        print(f"    {m:>5g} {len(ap):>4}/{len(A):<3} {len(ap)/len(A):5.1%} "
              f"{sum(v > 0.05 for v in ek):>8} {max(ek, default=0):>8.1%} "
              f"{sum(v > 0.02 for v in et):>11} {max(et, default=0):>11.1%}{marca}")
    print()
    print("    |eK|>5% e |dth/T|>2% contam APROVADAS: o erro de K e theta que o")
    print("    NRMSE estendido deixa passar em cada horizonte.")


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

    if not any("nrmse_estendido" in x for x in A):
        print("  analise.json sem `nrmse_estendido` — rode "
              "`analisa_aleatorias.py --dir` de novo neste lote.")
        return

    print(f"  ESTRITO  (entregou, estrutura exata, NRMSE ate theta + "
          f"{HORIZONTE_T_DOM:g} t_dom < {ESTRITO['nrmse']:.0%})")
    bons = sum(1 for x in A if not avalia(x))
    print(f"    {'TODAS':<12} {bons:>3}/{len(A):<4} = {bons/len(A):6.1%}")
    print()
    print("  QUAL CRITERIO REPROVA (contando cada reprovacao):")
    _quem_reprova(A, avalia)
    print()
    print("    'sozinho' = era o UNICO criterio reprovado; corrigi-lo salvaria a figura.")
    print()
    _dominio_aplicabilidade(A)
    _sensibilidade(A)


def _dominio_aplicabilidade(A) -> None:
    # Reportado AO LADO do global, nao no lugar: o corte e' hipotese de operacao
    # declarada, nao filtro escondido. So aparece quando ha SNR no nome; lotes
    # sem ruido nao tem dominio a recortar.
    com_snr = [x for x in A if snr_db(x) is not None]
    if not com_snr:
        return
    sub = [x for x in com_snr if snr_db(x) >= SNR_MIN_DB]
    print(f"  DOMINIO DE APLICABILIDADE (SNR >= {SNR_MIN_DB} dB, "
          f"{len(sub)}/{len(com_snr)} figuras com ruido)")

    def taxa(g):
        b = sum(1 for x in g if not avalia(x))
        return f"{b}/{len(g)} = {b/len(g):6.1%}" if g else "       —"

    for grp in ("K>1", "K<1"):
        print(f"    {grp:<12} {taxa([x for x in sub if faixa_ganho(x) == grp])}")
    print(f"    {'TODAS':<12} {taxa(sub)}")
    print()


if __name__ == "__main__":
    main()
