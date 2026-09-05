"""Gera o par controlado de multi-degrau usado por
`tests/part2/test_caso_real_multidegrau.py`.

O PAR CONTROLADO e a razao deste arquivo existir. `caso_real_multi_fopdt.png` e
`caso_real_multi_fopdt_1degrau.png` sao a MESMA planta, o MESMO render, os
MESMOS limites de eixo — muda uma variavel so, o segundo degrau. Enquanto as
duas coexistirem, nenhuma explicacao alternativa sobrevive a discordancia entre
elas (Ruling 66).

Este gerador reproduz a geometria de `rg.py` no que importa para a mascara:
`figsize=(9, 5)`, `dpi=110`, estilo `seaborn-v0_8-darkgrid`, as mesmas cores,
os mesmos limites de eixo, a mesma posicao de legenda, e a linha tracejada do
sinal de entrada acumulado (o "degrau" desenhado em preto,
`drawstyle="steps-post"`). O UNICO respeito em que o render NAO e uma copia
sao os titulos: aqui sao mais curtos que os de `rg.py` (ex.: "FOPDT: dois
degraus positivos" contra "FOPDT: Dois Degraus Positivos Consecutivos" em
`rg.py:80`) — decisao deliberada da spec deste arquivo, nao um descuido. Os
titulos sao texto que a mascara tem que ignorar; o que o par controlado exige
e que as duas metades compartilhem o MESMO titulo, o que elas fazem.

A linha tracejada nao e decoracao: toda medida da spec (§39.3, §41) foi feita
sobre a geometria de `rg.py` COM essa linha presente. Uma primeira versao
deste arquivo omitiu a linha tracejada por engano — o efeito nao foi neutro:
sem ela a curva de saida encosta no rodape do quadro e o trecho ja assentado
fica um segmento perfeitamente reto, os dois defeitos de Estagio A
documentados em §39.3 (A e B) que o retreino de §41 especificamente ensinou a
mascara a separar da linha tracejada. Medido com uma variavel isolada (mesmo
dpi, mesmos eixos, so a linha tracejada entrando e saindo): no controle
negativo a cobertura da mascara cai de 96.2% (com a tracejada) para 53.2%
(sem ela), e a estrutura selecionada muda de FOPDT para 2a ordem
SUPERAMORTECIDA (wn=8.86, zeta=2.33) — so o ROTULO da estrutura muda; um polo
lento `1/1.996` de uma 2a ordem superamortecida e numericamente o mesmo
`tau = 0.501` do FOPDT verdadeiro, a dinamica nao muda. Na figura FOPDT de
dois degraus a cobertura cai de 97.0% (com a tracejada) para 92.3% (sem ela),
com a mesma troca de rotulo de estrutura. Por isso a linha tracejada fica;
ela e contexto que a mascara usa, nao um distrator.

Verdade declarada na propria funcao de transferencia, sem estimativa:

    Sistema A (`caso_real_multi_sub.png`)
        TransferFunction([10], [1, 2, 10]), theta = 0
        degraus (-1.0 em t=1.0) e (-2.0 em t=6.0)
        -> 1o degrau: 2a ordem, K = 1 x (-1) = -1, wn = sqrt(10) = 3.1623,
           zeta = 2 / (2*sqrt(10)) = 0.3162, partida = 1.0 s

    Sistema B (`caso_real_multi_fopdt.png`)
        TransferFunction([2], [1, 2]), theta = 0.5
        degraus (+2.0 em t=1.0) e (+1.5 em t=4.0)
        -> 1o degrau: FOPDT, K = 1 x 2 = 2, tau = 0.5, partida = 1.0 + 0.5 = 1.5

    Controle (`caso_real_multi_fopdt_1degrau.png`)
        Sistema B com o SEGUNDO degrau removido. Mesma verdade do 1o degrau.
        A linha tracejada do controle mostra so um degrau (nao um patamar
        duplo) — essa diferenca na entrada acumulada E a variavel sob teste,
        nao uma segunda diferenca introduzida por acidente.

`K` segue a convencao do projeto: e `K_planta x U`, nao o ganho DC da planta.
O sinal de entrada acumulado ser desenhado no grafico NAO significa que a
pipeline le essa curva — ela nunca separa entrada de saida; le so `c(t)`.
Ver ARQUITETURA.md, secao "O que K significa".
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # sem display: roda em qualquer lugar
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

SAIDA = Path(__file__).resolve().parent / "tests" / "fixtures"


def multi_step_response(sistema, theta, lista_degraus, vetor_tempo):
    y_total = np.zeros_like(vetor_tempo)
    ref_entrada = np.zeros_like(vetor_tempo)
    valor_acumulado = 0
    for amplitude, instante in lista_degraus:
        mascara_entrada = vetor_tempo >= instante
        ref_entrada[mascara_entrada] = valor_acumulado + amplitude
        valor_acumulado += amplitude

        tempo_reacao = instante + theta
        mascara = vetor_tempo >= tempo_reacao
        t_ativo = vetor_tempo[mascara] - tempo_reacao
        if len(t_ativo) > 0:
            _, y_ativo = signal.step(sistema, T=t_ativo)
            y_total[mascara] += y_ativo * amplitude
    return y_total, ref_entrada


def _figura(nome, t, y, u, cor, titulo, xlim, ylim, loc_legenda):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(9, 5))
    plt.plot(t, y, color=cor, linewidth=2.5, label="Saida do Sistema $c(t)$")
    plt.plot(t, u, color="black", linestyle="--", linewidth=1.5,
              drawstyle="steps-post", label="Sinal de Entrada Acumulado")
    plt.title(titulo, fontsize=13, fontweight="bold")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc=loc_legenda)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA / nome, dpi=110)
    plt.close(fig)
    print(f"escrito: {SAIDA / nome}")


def main() -> None:
    t = np.linspace(0, 12, 1000)

    sys_a = signal.TransferFunction([10], [1, 2, 10])
    y_a, u_a = multi_step_response(sys_a, 0.0, [(-1.0, 1.0), (-2.0, 6.0)], t)
    _figura("caso_real_multi_sub.png", t, y_a, u_a,
            "#e74c3c", "Subamortecido: dois degraus negativos", (0, 12), (-4, 0.5),
            "upper right")

    sys_b = signal.TransferFunction([2], [1, 2])
    # O PAR CONTROLADO: tudo identico, menos a lista de degraus (e portanto
    # a linha tracejada da entrada acumulada, que reflete essa lista).
    y_b2, u_b2 = multi_step_response(sys_b, 0.5, [(2.0, 1.0), (1.5, 4.0)], t)
    _figura("caso_real_multi_fopdt.png", t, y_b2, u_b2,
            "#2ecc71", "FOPDT: dois degraus positivos", (0, 8), (0, 4),
            "lower right")
    y_b1, u_b1 = multi_step_response(sys_b, 0.5, [(2.0, 1.0)], t)
    _figura("caso_real_multi_fopdt_1degrau.png", t, y_b1, u_b1,
            "#2ecc71", "FOPDT: dois degraus positivos", (0, 8), (0, 4),
            "lower right")


if __name__ == "__main__":
    main()
