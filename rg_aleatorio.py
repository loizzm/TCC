"""Gera 33 figuras ALEATORIAS por familia de gerador (rg / rg_negativo /
rg_multidegrau) e grava a verdade de cada uma em `verdade.json`.

Por que existe. `rg.py`, `rg_negativo.py` e `rg_multidegrau.py` sao geradores
de fixture: cada um produz um punhado de figuras FIXAS, escolhidas a mao. Isso
serve para regressao, nao para medir assertividade — n=3 nao distingue acerto
de sorte. Este arquivo mantem a ASSINATURA DE RENDER de cada um dos tres e
sorteia tudo o mais: planta, degraus, cores, legendas, tipo de linha, largura
de linha, preenchimento, grade, dpi, tamanho da figura e limites de eixo.

As tres familias, e o que cada uma preserva do gerador que imita:

  familia "rg"    -> `rg.py`: estilo seaborn-v0_8-darkgrid, DOIS degraus de
                     mesmo sinal, e a linha tracejada do sinal de entrada
                     acumulado (`drawstyle="steps-post"`) no mesmo quadro.
  familia "neg"   -> `rg_negativo.py`: estilo dark_background, UM degrau
                     NEGATIVO, o degrau de entrada desenhado com `plt.step`,
                     `axvspan` cobrindo a faixa do atraso e grade pontilhada.
  familia "multi" -> `rg_multidegrau.py`: mesmo render do "rg", mas gravado
                     via Agg com dpi explicito, titulos curtos, e de 1 a 3
                     degraus — o de 1 degrau e o CONTROLE do par (a mesma
                     planta sem o segundo degrau).

VERDADE. A verdade de cada figura vem da funcao de transferencia sorteada, nao
de leitura do grafico. Para as familias com mais de um degrau a verdade
declarada e a do PRIMEIRO degrau, que e o que a pipeline pode recuperar de uma
janela truncada. `K` segue a convencao do projeto (ARQUITETURA.md, "O que K
significa"): e `K_planta x U`, a excursao por unidade de entrada, nao o ganho
DC da planta. `theta` e o INSTANTE DE PARTIDA — `instante_do_degrau +
theta_do_sistema` —, porque os dois nao se separam sem ler a entrada.

DOMINIO DOS SORTEIOS. As faixas de planta ficam dentro do dominio de treino
(`dataset/generator.py:sample_system`: K loguniform(0,2..20), tau
loguniform(0,05..50), wn loguniform(0,02..20), zeta U(0,10..3,00), theta
loguniform(0,05..1,0) x t_dom). O que sai do dominio de proposito e o RENDER:
tema escuro, entrada plotada junto, multiplos degraus, axvspan sobre a
resposta. Assim a medida isola o efeito do render, e nao o de extrapolar
fisica.

Uso:
    python rg_aleatorio.py [--n 33] [--seed 20260908] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # sem display: roda em qualquer lugar
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# ---------------------------------------------------------------- paletas

# Cores de linha para FUNDO CLARO (seaborn-v0_8-darkgrid): luminancia baixa.
CORES_CLARO = [
    "#e74c3c", "#2ecc71", "#2980b9", "#8e44ad", "#d35400", "#16a085",
    "#c0392b", "#27ae60", "#2c3e50", "#7f8c8d", "#b7950b", "#1f77b4",
    "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#005f73",
]
# Cores de linha para FUNDO ESCURO (dark_background): luminancia alta.
CORES_ESCURO = [
    "#00ffcc", "#ff66cc", "#ffcc00", "#66ff66", "#66ccff", "#ff9966",
    "#cc99ff", "#ffffff", "#7fffd4", "#ffa07a", "#adff2f", "#f0e68c",
]
# Cor da tracejada de ENTRADA. Contrasta com o fundo, nao com a curva.
CORES_ENTRADA_CLARO = ["black", "#222222", "#444444", "#333366", "dimgray"]
CORES_ENTRADA_ESCURO = ["white", "#dddddd", "#eeeeee", "#cccccc", "lightgray"]

ESTILOS_LINHA = ["-", "--", "-.", ":"]
ESTILOS_GRADE = [":", "--", "-.", "-"]
LOCS_LEGENDA = ["upper right", "upper left", "lower left", "lower right",
                "center right", "center left", "best", "upper center",
                "lower center"]

ROTULOS_SAIDA = [
    "Saida do Sistema $c(t)$", "Resposta $y(t)$", "Saida medida",
    "$c(t)$", "Resposta do processo", "Variavel de processo (PV)",
    "Saida do Sistema", "Resposta ao degrau", "$y(t)$ simulado",
]
ROTULOS_ENTRADA = [
    "Sinal de Entrada Acumulado", "Entrada $u(t)$", "Degrau de Entrada",
    "Referencia (SP)", "$u(t)$", "Entrada acumulada", "Setpoint",
]
ROTULOS_FAIXA = [
    "Tempo Morto", "Atraso de Transporte", "Faixa do atraso",
    "$\\theta$", "Tempo morto do processo", "Latencia",
]
TITULOS = [
    "Resposta ao degrau", "Identificacao da planta", "Ensaio em malha aberta",
    "Resposta temporal do processo", "Curva de reacao do processo",
    "Ensaio degrau: saida do sistema", "Dinamica do processo",
    "Teste de degrau", "Resposta do sistema no tempo",
]
ROTULOS_X = ["Tempo (s)", "t (s)", "Tempo [s]", "Tempo decorrido (s)"]
ROTULOS_Y = ["Amplitude", "Saida", "y", "Amplitude (u.a.)", "Variavel de processo"]

PREENCHIMENTOS = ["nenhum", "nenhum", "axvspan_atraso", "fill_between",
                  "axhspan_ref", "axvspan_e_fill"]

# MODO DA LINHA DE ENTRADA. Existe para montar o PAR CONTROLADO que responde
# "o que a pipeline entrega numa figura que NAO desenha o degrau de entrada?":
#
#   "desenha"   -> como os tres geradores reais fazem (rg.py:64,
#                  rg_negativo.py:104, rg_multidegrau.py:95: todos tracejados).
#   "omite"     -> nao desenha a linha, mas MANTEM os limites de eixo que ela
#                  produziria. Isola UMA variavel — a tinta no quadro — e por
#                  isso e o corte que atribui o defeito de extracao a ela.
#   "omite_fit" -> nao desenha e RECALCULA os limites so com a resposta. E a
#                  figura realista de quem nunca plotou a entrada, e o que
#                  responde a pergunta de verdade. Muda duas coisas de uma vez
#                  (tinta e enquadramento), entao nao serve para atribuicao.
#
# O `rg_multidegrau.py` ja tinha medido que omitir a tracejada NAO e neutro:
# sem ela a curva encosta no rodape do quadro e a cobertura da mascara cai de
# 96,2 % para 53,2 %. Por isso os dois modos, e nao um so.
MODO_ENTRADA = "desenha"


# ---------------------------------------------------------------- fisica

def t_dominante(ordem: str, tau, wn, zeta) -> float:
    """Constante de tempo dominante — mesma definicao de `dataset/generator.py`."""
    if ordem == "fopdt":
        return float(tau)
    if zeta <= 1.0:
        return float(1.0 / (zeta * wn))
    return float((zeta + np.sqrt(zeta * zeta - 1.0)) / wn)


def planta(ordem: str, K: float, tau, wn, zeta) -> signal.TransferFunction:
    """Funcao de transferencia com ganho DC = K (sem o atraso, que e aplicado
    por deslocamento no tempo, como nos tres geradores originais)."""
    if ordem == "fopdt":
        return signal.TransferFunction([K], [tau, 1.0])
    return signal.TransferFunction([K * wn * wn], [1.0, 2.0 * zeta * wn, wn * wn])


def resposta_multi_degrau(sistema, theta_sis, degraus, t):
    """Superposicao de degraus — copia literal da logica de `rg.py`/
    `rg_multidegrau.py`, para que a geometria da curva seja a mesma."""
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    acumulado = 0.0
    for amplitude, instante in degraus:
        u[t >= instante] = acumulado + amplitude
        acumulado += amplitude
        t_reacao = instante + theta_sis
        mascara = t >= t_reacao
        t_ativo = t[mascara] - t_reacao
        if t_ativo.size:
            _, y_ativo = signal.step(sistema, T=t_ativo)
            y[mascara] += y_ativo * amplitude
    return y, u


def _loguniforme(rng, lo, hi) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sorteia_planta(rng) -> dict:
    """Planta dentro do dominio de treino de `dataset/generator.py`."""
    ordem = "fopdt" if rng.random() < 0.5 else "second"
    K_planta = _loguniforme(rng, 0.3, 10.0)
    if ordem == "fopdt":
        tau, wn, zeta = _loguniforme(rng, 0.1, 20.0), None, None
    else:
        tau, wn, zeta = None, _loguniforme(rng, 0.05, 10.0), float(rng.uniform(0.15, 2.5))
    t_dom = t_dominante(ordem, tau, wn, zeta)
    # theta = 0 em 1 de cada 4 (o `sys_1` do `rg.py` tem theta = 0).
    theta_sis = 0.0 if rng.random() < 0.25 else _loguniforme(rng, 0.05, 1.0) * t_dom
    return {"ordem": ordem, "K_planta": K_planta, "tau": tau, "wn": wn,
            "zeta": zeta, "t_dom": t_dom, "theta_sistema": float(theta_sis)}


def sorteia_degraus(rng, p: dict, n_degraus: int, sinal: int) -> tuple[list, float]:
    # `n_degraus` continua no parametro e vale sempre 1 desde §68 — mantido
    # porque a assinatura entra no stream de RNG e mexer nela mudaria TODA
    # figura ja gerada. Ver MULTI_DEGRAU.md.
    """Lista [(amplitude, instante)] e o fim da janela.

    O primeiro degrau cai cedo na janela; os seguintes ficam separados por pelo
    menos 0,8 t_dom, para que o transitorio do PRIMEIRO — a verdade declarada —
    esteja de fato visivel antes de o proximo entrar.
    """
    t_dom = p["t_dom"]
    t1 = float(rng.uniform(0.0, 0.6) * t_dom)
    degraus = [(sinal * _loguniforme(rng, 0.5, 4.0), t1)]
    t_ant = t1
    for _ in range(n_degraus - 1):
        t_ant = t_ant + float(_loguniforme(rng, 0.8, 3.0) * t_dom)
        degraus.append((sinal * _loguniforme(rng, 0.4, 3.0), t_ant))
    t_fim = t_ant + p["theta_sistema"] + float(_loguniforme(rng, 1.5, 5.0) * t_dom)
    return degraus, t_fim


def sorteia_estilo(rng, escuro: bool) -> dict:
    cores = CORES_ESCURO if escuro else CORES_CLARO
    entrada = CORES_ENTRADA_ESCURO if escuro else CORES_ENTRADA_CLARO
    return {
        "cor": cores[int(rng.integers(len(cores)))],
        "cor_entrada": entrada[int(rng.integers(len(entrada)))],
        "linestyle": ESTILOS_LINHA[int(rng.integers(len(ESTILOS_LINHA)))],
        "linestyle_entrada": ESTILOS_LINHA[int(rng.integers(len(ESTILOS_LINHA)))],
        "linewidth": float(rng.uniform(1.0, 3.5)),
        "linewidth_entrada": float(rng.uniform(0.8, 2.2)),
        "legenda_loc": LOCS_LEGENDA[int(rng.integers(len(LOCS_LEGENDA)))],
        "rotulo_saida": ROTULOS_SAIDA[int(rng.integers(len(ROTULOS_SAIDA)))],
        "rotulo_entrada": ROTULOS_ENTRADA[int(rng.integers(len(ROTULOS_ENTRADA)))],
        "rotulo_faixa": ROTULOS_FAIXA[int(rng.integers(len(ROTULOS_FAIXA)))],
        "titulo": TITULOS[int(rng.integers(len(TITULOS)))],
        "rotulo_x": ROTULOS_X[int(rng.integers(len(ROTULOS_X)))],
        "rotulo_y": ROTULOS_Y[int(rng.integers(len(ROTULOS_Y)))],
        "preenchimento": PREENCHIMENTOS[int(rng.integers(len(PREENCHIMENTOS)))],
        "alpha_preench": float(rng.uniform(0.10, 0.45)),
        "grade_style": ESTILOS_GRADE[int(rng.integers(len(ESTILOS_GRADE)))],
        "figsize": (float(rng.uniform(6.5, 10.0)), float(rng.uniform(4.0, 6.0))),
        "dpi": int(rng.integers(90, 145)),
        "fontsize_titulo": int(rng.integers(11, 15)),
        "margem_y": float(rng.uniform(0.05, 0.20)),
    }


def _limites(t_fim, y, u, margem, com_entrada):
    # `omite` mantem os limites da entrada de proposito (par controlado);
    # so `omite_fit` reenquadra na resposta.
    if MODO_ENTRADA == "omite_fit":
        com_entrada = False
    dados = [y, u] if com_entrada else [y]
    lo = min(float(np.min(d)) for d in dados)
    hi = max(float(np.max(d)) for d in dados)
    faixa = max(hi - lo, 1e-9)
    return (0.0, float(t_fim)), (lo - margem * faixa, hi + margem * faixa)


def _preenche(st, degraus, theta_sis, t, y, xlim, ylim):
    """Aplica o preenchimento sorteado. Devolve o rotulo usado na legenda, ou
    None quando o preenchimento nao entra na legenda."""
    modo = st["preenchimento"]
    if modo in ("axvspan_atraso", "axvspan_e_fill") and theta_sis > 0:
        for amplitude, instante in degraus:
            plt.axvspan(instante, instante + theta_sis, color="gray",
                        alpha=st["alpha_preench"],
                        label=st["rotulo_faixa"] if instante == degraus[0][1] else None)
    if modo in ("fill_between", "axvspan_e_fill"):
        plt.fill_between(t, y, y[0], color=st["cor"], alpha=st["alpha_preench"] * 0.6)
    if modo == "axhspan_ref":
        alvo = float(y[-1])
        meia = 0.05 * max(abs(ylim[1] - ylim[0]), 1e-9)
        plt.axhspan(alvo - meia, alvo + meia, color="gray",
                    alpha=st["alpha_preench"], label="Faixa de acomodacao")


# ---------------------------------------------------------------- familias

def _estilo(nome: str) -> None:
    """Aplica um estilo do matplotlib SEM herdar o anterior.

    `plt.style.use` so sobrescreve as chaves que o estilo declara — o resto
    permanece como estava. Isso vaza entre familias: `seaborn-v0_8-darkgrid`
    zera `axes.linewidth` e `dark_background` NAO a restaura, entao uma figura
    "neg" gerada depois de uma "rg" sai SEM AS MOLDURAS DOS EIXOS. E um defeito
    do gerador que se disfarca de defeito da pipeline: `detect_plot_bbox`
    (identify/calibrate.py) procura exatamente o spine inferior e o esquerdo, e
    sem eles devolve `bbox_not_found`. Medido: 29 das 33 figuras "neg" da
    primeira geracao falharam a calibracao por isto. `rcdefaults()` antes de
    cada estilo torna a geracao independente da ordem das familias.
    """
    plt.rcdefaults()
    plt.style.use(nome)



def figura_rg(caminho: Path, p, degraus, t, y, u, st) -> None:
    """Assinatura do `rg.py`: seaborn darkgrid, entrada acumulada tracejada."""
    _estilo("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=st["figsize"])
    xlim, ylim = _limites(t[-1], y, u, st["margem_y"], True)
    plt.plot(t, y, color=st["cor"], linewidth=st["linewidth"],
             linestyle=st["linestyle"], label=st["rotulo_saida"])
    if MODO_ENTRADA == "desenha":
        plt.plot(t, u, color=st["cor_entrada"], linestyle=st["linestyle_entrada"],
                 linewidth=st["linewidth_entrada"], drawstyle="steps-post",
                 label=st["rotulo_entrada"])
    _preenche(st, degraus, p["theta_sistema"], t, y, xlim, ylim)
    plt.title(st["titulo"], fontsize=st["fontsize_titulo"], fontweight="bold")
    plt.xlabel(st["rotulo_x"]); plt.ylabel(st["rotulo_y"])
    plt.legend(loc=st["legenda_loc"])
    plt.xlim(*xlim); plt.ylim(*ylim)
    fig.savefig(caminho, dpi=st["dpi"])
    plt.close(fig)


def figura_neg(caminho: Path, p, degraus, t, y, u, st) -> None:
    """Assinatura do `rg_negativo.py`: fundo escuro, um degrau negativo
    desenhado com `plt.step`, `axvspan` no atraso e grade pontilhada."""
    _estilo("dark_background")
    fig = plt.figure(figsize=st["figsize"])
    amplitude, instante = degraus[0]
    xlim, ylim = _limites(t[-1], y, u, st["margem_y"], True)
    plt.plot(t, y, color=st["cor"], linewidth=st["linewidth"],
             linestyle=st["linestyle"], label=st["rotulo_saida"])
    # A entrada como escada acumulada. Com UM degrau isto e literalmente o
    # `plt.step([0, t_deg, t_fim], [0, U, U])` do `rg_negativo.py`; com mais de
    # um, a mesma escada com os degraus somados.
    if MODO_ENTRADA == "desenha":
        xs, ys, acc = [0.0], [0.0], 0.0
        for amp_i, inst_i in degraus:
            xs.append(inst_i); ys.append(acc + amp_i); acc += amp_i
        xs.append(float(t[-1])); ys.append(acc)
        plt.step(xs, ys, color=st["cor_entrada"], linestyle=st["linestyle_entrada"],
                 linewidth=st["linewidth_entrada"], where="post",
                 label=f"{st['rotulo_entrada']} (t={instante:.3g}s)")
    if p["theta_sistema"] > 0:
        for j, (_a, inst_i) in enumerate(degraus):
            plt.axvspan(inst_i, inst_i + p["theta_sistema"], color="#333333",
                        alpha=float(min(0.9, 0.4 + st["alpha_preench"])),
                        label=(f"{st['rotulo_faixa']} "
                               f"($\\theta$={p['theta_sistema']:.3g}s)")
                              if j == 0 else None)
    if st["preenchimento"] in ("fill_between", "axvspan_e_fill"):
        plt.fill_between(t, y, y[0], color=st["cor"], alpha=st["alpha_preench"] * 0.6)
    plt.title(st["titulo"], fontsize=st["fontsize_titulo"])
    plt.xlabel(st["rotulo_x"]); plt.ylabel(st["rotulo_y"])
    plt.legend(loc=st["legenda_loc"])
    plt.grid(color="#444444", linestyle=st["grade_style"])
    plt.xlim(*xlim); plt.ylim(*ylim)
    fig.savefig(caminho, dpi=st["dpi"])
    plt.close(fig)


def figura_multi(caminho: Path, p, degraus, t, y, u, st) -> None:
    """Assinatura do `rg_multidegrau.py`: mesmo render do `rg.py`, titulo curto,
    gravado via Agg com dpi explicito."""
    _estilo("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=st["figsize"])
    xlim, ylim = _limites(t[-1], y, u, st["margem_y"], True)
    plt.plot(t, y, color=st["cor"], linewidth=st["linewidth"],
             linestyle=st["linestyle"], label=st["rotulo_saida"])
    if MODO_ENTRADA == "desenha":
        plt.plot(t, u, color=st["cor_entrada"], linestyle=st["linestyle_entrada"],
                 linewidth=st["linewidth_entrada"], drawstyle="steps-post",
                 label=st["rotulo_entrada"])
    _preenche(st, degraus, p["theta_sistema"], t, y, xlim, ylim)
    n = len(degraus)
    titulo = ("FOPDT" if p["ordem"] == "fopdt" else "2a ordem")
    titulo += f": {n} degrau" + ("s" if n > 1 else "")
    plt.title(titulo, fontsize=st["fontsize_titulo"], fontweight="bold")
    plt.xlabel(st["rotulo_x"]); plt.ylabel(st["rotulo_y"])
    plt.legend(loc=st["legenda_loc"])
    plt.xlim(*xlim); plt.ylim(*ylim)
    fig.savefig(caminho, dpi=st["dpi"])
    plt.close(fig)


# Indice fixo por familia: entra na semente. `hash()` de str NAO serve — muda
# a cada processo (PYTHONHASHSEED), e a geracao deixaria de ser reproduzivel.
IDX_FAMILIA = {"rg": 0, "neg": 1, "multi": 2}

# UM DEGRAU SEMPRE (§68). As familias eram definidas tambem pelo numero de
# degraus — "rg" dava 2, "neg" dava 1, "multi" sorteava 1/2/3. Com a frente de
# multi-degrau fora do codigo, o que distingue as familias e so o RENDER, que
# e o papel que elas devem ter nesta avaliacao. Ver MULTI_DEGRAU.md.
FAMILIAS = {
    "rg": {"render": figura_rg, "escuro": False,
           "sinal": lambda rng: 1 if rng.random() < 0.6 else -1},
    "neg": {"render": figura_neg, "escuro": True,
            "sinal": lambda rng: -1},
    "multi": {"render": figura_multi, "escuro": False,
              "sinal": lambda rng: 1 if rng.random() < 0.5 else -1},
}


# ---------------------------------------------------------------- lotes

# LOTE BALANCEADO. O lote por familia responde "como cada gerador se sai", mas
# NAO separa o sinal de K do numero de degraus: `neg` e 1 degrau negativo por
# construcao e `rg` e 2 degraus, entao a celula (K>0, 1 degrau) fica com n=2 e
# qualquer conclusao sobre "o problema e o ganho negativo" estaria confundida
# com "o problema e multi-degrau". Aqui as duas variaveis viram um fatorial
# 2x2 e a ASSINATURA DE RENDER passa a ser ruido sorteado uniformemente entre
# as tres — que e o papel certo dela nesta pergunta.
CELULAS = [("K+", 1, +1, 1), ("K-", 1, -1, 1)]


def _uma_amostra(rng, render, escuro, sinal):
    p = sorteia_planta(rng)
    degraus, t_fim = sorteia_degraus(rng, p, 1, sinal)
    sistema = planta(p["ordem"], p["K_planta"], p["tau"], p["wn"], p["zeta"])
    t = np.linspace(0.0, t_fim, int(rng.integers(800, 2001)))
    y, u = resposta_multi_degrau(sistema, p["theta_sistema"], degraus, t)
    st = sorteia_estilo(rng, escuro)
    return p, degraus, t_fim, t, y, u, st


def _registro(nome, familia, p, degraus, t_fim, st, escuro, celula=None):
    U1, t1 = degraus[0]
    return {
        "arquivo": nome, "familia": familia, "celula": celula,
        "order": p["ordem"],
        "K": float(p["K_planta"] * U1),
        "tau": None if p["tau"] is None else float(p["tau"]),
        "wn": None if p["wn"] is None else float(p["wn"]),
        "zeta": None if p["zeta"] is None else float(p["zeta"]),
        "theta": float(t1 + p["theta_sistema"]),
        "K_planta": float(p["K_planta"]), "U1": float(U1),
        "theta_sistema": float(p["theta_sistema"]), "t_degrau1": float(t1),
        "t_dom": float(p["t_dom"]), "n_degraus": len(degraus),
        "degraus": [[float(a_), float(t_)] for a_, t_ in degraus],
        "t_fim": float(t_fim), "janela_em_t_dom": float(t_fim / p["t_dom"]),
        "linestyle": st["linestyle"], "linewidth": st["linewidth"],
        "cor": st["cor"], "legenda_loc": st["legenda_loc"],
        "preenchimento": st["preenchimento"], "dpi": st["dpi"],
        "figsize": list(st["figsize"]), "fundo_escuro": bool(escuro),
    }


def lote_por_familia(n, seed, out):
    verdades = []
    for familia, cfg in FAMILIAS.items():
        for i in range(n):
            # Uma semente por (familia, indice): reproduzivel amostra a amostra,
            # e mexer no n de uma familia nao muda as outras.
            rng = np.random.default_rng([seed, IDX_FAMILIA[familia], i])
            p, degraus, t_fim, t, y, u, st = _uma_amostra(
                rng, cfg["render"], cfg["escuro"],
                cfg["sinal"](rng))
            nome = f"{familia}_{i:02d}.png"
            cfg["render"](out / nome, p, degraus, t, y, u, st)
            verdades.append(_registro(nome, familia, p, degraus, t_fim, st,
                                      cfg["escuro"]))
            print(f"escrito: {out / nome}")
    return verdades


def lote_balanceado(n, seed, out):
    """Fatorial 2x2 (sinal de K) x (1 ou 2 degraus), render sorteado.

    `so_1_degrau` restringe as celulas ao caso de UM degrau, mantendo o
    balanceamento por sinal. Serve para medir a populacao em que o sistema foi
    projetado para operar, isolada do multi-degrau — que domina o erro
    (conjuntivo 94 % contra 46 %, medido no lote100).
    """
    nomes_fam = list(FAMILIAS)
    verdades = []
    celulas = CELULAS
    for c, (rot_sinal, rot_deg, sinal, n_deg) in enumerate(celulas):
        for i in range(n):
            rng = np.random.default_rng([seed, 100 + c, i])
            fam = nomes_fam[int(rng.integers(len(nomes_fam)))]
            cfg = FAMILIAS[fam]
            p, degraus, t_fim, t, y, u, st = _uma_amostra(
                rng, cfg["render"], cfg["escuro"], sinal)
            celula = f"{rot_sinal}_{rot_deg}deg"
            nome = f"bal_{celula}_{i:02d}.png"
            cfg["render"](out / nome, p, degraus, t, y, u, st)
            verdades.append(_registro(nome, fam, p, degraus, t_fim, st,
                                      cfg["escuro"], celula=celula))
            print(f"escrito: {out / nome}")
    return verdades


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=33,
                    help="figuras por familia (ou por celula, no modo balanceado)")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--modo", choices=("familia", "balanceado"), default="familia")
    ap.add_argument("--entrada", choices=("desenha", "omite", "omite_fit"),
                    default="desenha",
                    help="se a linha do degrau de entrada e desenhada; "
                         "'omite' mantem os limites de eixo (par controlado), "
                         "'omite_fit' reenquadra na resposta")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    global MODO_ENTRADA
    MODO_ENTRADA = a.entrada
    raiz = Path(__file__).resolve().parent / "reports" / "amostras_aleatorias"
    out = a.out or (raiz if a.modo == "familia" else raiz / "balanceado")
    out.mkdir(parents=True, exist_ok=True)

    if a.modo == "familia":
        verdades = lote_por_familia(a.n, a.seed, out)
    else:
        verdades = lote_balanceado(a.n, a.seed, out)

    (out / "verdade.json").write_text(
        json.dumps(verdades, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(verdades)} figuras · verdade em {out / 'verdade.json'}")


if __name__ == "__main__":
    main()
