"""Gerador saneado: sorteia sistema + estilo, renderiza imagem, mascara e meta.

Sem vazamento de rotulo: o estilo visual vem de um stream de RNG independente
do stream que sorteia o sistema dinamico (ver `randomize.sample_style`).
Determinismo bit-a-bit: mesma seed => mesmos bytes de image.png e mask.png.
A igualdade de BYTES so e garantida dentro de um ambiente pinado (ver
`requirements.txt`): versoes diferentes de matplotlib/pillow/libpng podem mudar
o rasterizador ou o encoder PNG. A igualdade dos VALORES (series, params, meta)
independe do ambiente.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, LinearLocator, MaxNLocator  # noqa: E402
from PIL import Image  # noqa: E402

from dataset.randomize import RenderStyle, sample_style  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

# 1 -> 2: entra o estrato MULTI-DEGRAU. As chaves novas (`degraus`,
# `n_degraus`, `u_final`) sao ADITIVAS e saem em toda amostra; numa amostra de
# um degrau elas valem `[[step_amplitude, 0.0]]`, `1` e `step_amplitude`, que
# descreve exatamente o que o corpus base sempre fez. Nenhuma chave antiga
# mudou de nome ou de significado, entao um leitor da v1 continua funcionando —
# `params` segue descrevendo o PRIMEIRO degrau, que numa amostra de um degrau
# e o unico.
SCHEMA_VERSION: int = 2

# 2 -> 3: entra o estrato FORA DA FAMILIA (fase nao-minima). As chaves novas
# (`fora_da_familia` e `params.a`) sao CONDICIONAIS: so aparecem na amostra que
# de fato esta fora da familia. Por isso a versao tambem e' condicional — a
# amostra declara o schema a que ELA obedece, e o corpus base continua v2 byte
# a byte. Bumpar a versao global reescreveria todo meta.json do corpus base
# para nao dizer nada de novo sobre ele, e quebraria a igualdade de bytes que
# `test_o_padrao_nao_muda_um_byte` protege.
#
# Um leitor v2 que caia numa amostra v3 ve `schema_version == 3` e chaves que
# nao conhece. O contrato e': se `meta.get("fora_da_familia")` for verdadeiro,
# `params` NAO descreve a curva inteira — descreve so a parte de POLOS do
# sistema, e ha um zero de fase nao-minima em `params["a"]` por cima.
SCHEMA_VERSION_FORA_DA_FAMILIA: int = 3

ORDERS: tuple[str, str] = ("fopdt", "second")
N_SERIES: int = 512
STEP_AMPLITUDE: float = 1.0


# --------------------------------------------------------------------------
# Sistema dinamico
# --------------------------------------------------------------------------


def dominant_time_constant(
    order: str, tau: float | None, wn: float | None, zeta: float | None
) -> float:
    """Constante de tempo dominante (contrato §1, Ruling K). Fonte unica.

    fopdt              -> tau
    second, zeta <= 1  -> 1/(zeta*wn)
    second, zeta > 1   -> 1/(wn*(zeta - sqrt(zeta^2-1))), na forma racionalizada
                          (zeta + sqrt(zeta^2-1))/wn, que evita o cancelamento
                          catastrofico para zeta grande. Continua em zeta=1.
    """
    if order == "fopdt":
        return float(tau)
    zeta = float(zeta)
    wn = float(wn)
    if zeta <= 1.0:
        return float(1.0 / (zeta * wn))
    return float((zeta + np.sqrt(zeta * zeta - 1.0)) / wn)


@dataclass(frozen=True)
class SystemSpec:
    """Sistema sorteado e janela de observacao.

    `degraus` e o estrato MULTI-DEGRAU (opt-in). `None` significa UM degrau em
    `t=0` com amplitude `step_amplitude` — o comportamento historico, byte a
    byte. Quando preenchido, e uma tupla de `(amplitude, instante)` e a saida e
    a SUPERPOSICAO das respostas: mesma planta, varias entradas.

    A convencao de verdade com mais de um degrau e a do `rg_multidegrau.py`:
    `params` no meta descreve o PRIMEIRO degrau, porque e o que a pipeline pode
    recuperar de um prefixo. `K` segue sendo `K_planta x U1`.

    `a_zero` e o estrato FORA DA FAMILIA (opt-in). `None` e' o comportamento
    historico. Preenchido, multiplica a planta por `(1 - a*s)` — um zero no
    SEMIPLANO DIREITO, que produz resposta inversa. Os polos nao mudam: `order`,
    `tau`/`wn`/`zeta` e portanto `t_dom` continuam descrevendo a mesma dinamica
    de polos, e e' por isso que `order` continua sendo `fopdt` ou `second` em
    vez de ganhar um terceiro valor. O que muda e' que a curva inteira ja nao
    e' membro da familia de modelos do Estagio D, e o meta diz isso na chave
    `fora_da_familia`.
    """

    order: str
    K: float
    tau: float | None
    theta: float
    wn: float | None
    zeta: float | None
    t_start: float
    t_end: float
    step_amplitude: float
    degraus: tuple[tuple[float, float], ...] | None = None
    a_zero: float | None = None

    @property
    def t_dom(self) -> float:
        """Constante de tempo dominante (delega para `dominant_time_constant`)."""
        return dominant_time_constant(self.order, self.tau, self.wn, self.zeta)


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


# Quantas constantes de tempo dominantes o estrato `reta_no_patamar` garante
# DEPOIS do tempo morto. `sample_system` sorteia a janela em
# `loguniform(0.5, 6.0) * t_dom`, e assentar a 1% leva ~4.6 t_dom — por isso o
# corpus praticamente nunca mostra o patamar: medido, a fracao final ja
# assentada tem MEDIANA de 0,98% da janela, e ZERO de 60 amostras chegam a 30%.
# As duas imagens reais do Ruling 55 usam 10 e ~21 t_dom, com ~50% da janela
# assentada. Sem patamar visivel a reta de referencia toca a curva so na ultima
# coluna, e nao ha trecho colinear para ocluir — foi o que a primeira tentativa
# deste estrato mediu (cobertura mediana 0,96, criterio pede < 0,75).
# Valor escolhido por VARREDURA medida (n=40 por ponto), tendo como alvo o
# maior vao de colunas sem tinta predita das duas imagens reais: 0,1669
# (Figure_1) e 0,3802 (resposta_degrau).
#   T_DOM   mediana     p90      max    >=0.15   >=0.20
#     9.2    0.0434   0.2710   0.4319    10/40     6/40
#    14.0    0.0479   0.4902   0.5744    11/40     9/40
#    20.0    0.0501   0.6007   0.7095    12/40     9/40   <- escolhido
#    30.0    0.0543   0.6724   0.8270    12/40    10/40
#    45.0    0.0516   0.6469   0.8907    14/40    12/40
# 20 casa com a janela das imagens reais (10 e ~21 t_dom). Acima disso o maximo
# vai a 0,83-0,89, FORA da faixa real, e a mediana nao sobe: os bloqueadores
# medidos sao ruido (Spearman SNR x vao +0,423, p=7e-4) e marcador (-0,346,
# p=7e-3), que fazem a curva escapar de tras da reta. Sao variacao legitima de
# estilo, entao o estrato exibe o fenomeno com forca em ~30% das amostras e
# isso e o correto — suprimi-los deixaria o estrato limpo demais para ser real.
_T_DOM_ESTRATO = 20.0


# Estrato FORA DA FAMILIA (fase nao-minima). O que se SORTEIA e' a
# PROFUNDIDADE DO MERGULHO, nao o zero `a` — e `a` sai resolvido por bissecao
# para produzi-la. A primeira versao sorteava `a` em constantes de tempo
# dominantes e falhou na medicao: com o mesmo `a/t_dom`, `fopdt` mergulha
# `K*a/tau` (grau relativo ZERO, ha salto) e `second` mergulha muito menos
# (grau relativo 1, a curva sai de zero com derivada finita). Em 60 amostras,
# TODAS as 9 abaixo do limiar da guarda eram de 2a ordem. Sortear o mergulho
# poe as duas ordens na MESMA escala de severidade, que e' o que o estrato
# precisa declarar.
#
# O mergulho e' medido como `-min(y) / ptp(y)` na serie limpa, com o repouso em
# zero por construcao (antes de `theta` a resposta e' identicamente nula). E' a
# mesma razao que `pipeline._undershoot` calcula, mas com aritmetica propria:
# o gerador nao importa a pipeline, senao o corpus passaria a ser definido pelo
# codigo que ele existe para testar.
#
# O PISO E' 1,5x O LIMIAR DA GUARDA (`_UNDERSHOOT_MAX = 0,08`). O estrato so
# serve como conjunto positivo se toda amostra dele DEVE mesmo ser recusada;
# uma amostra em cima do limiar mediria o arredondamento da guarda, nao a
# guarda. O teto de 0,45 e' onde o mergulho para de ser um detalhe da curva e
# passa a ser metade do desenho.
_NMP_MERGULHO = (0.12, 0.45)

# Janela MINIMA do estrato fora-da-familia, em t_dom depois do tempo morto.
# Nao e' folga estetica: sem ela o estrato produz amostras que NAO exibem
# resposta inversa. Medido em 24 amostras com a janela padrao
# (`loguniform(0.5, 6) * t_dom`), 2 saem com a curva ainda MERGULHADA na ultima
# coluna — ela desce, cruza o zero e o quadro acaba antes de ela assentar acima
# do repouso. Nessas, `_undershoot` da pipeline le 0,000 e ACERTA: o que esta
# desenhado e' um decaimento monotono, nao uma resposta inversa, e recusar por
# `resposta_inversa` seria acertar pelo motivo errado.
# O cruzamento de volta pelo zero acontece em `tau * ln(1 + a/tau)`, que na
# faixa util vale ~0,6 t_dom; assentar a 1% depois disso leva ~4,6 t_dom (mesmo
# numero que `_T_DOM_ESTRATO` cita). 5,5 = 0,6 + 4,6, arredondado para cima.
# Amostra que ja tem janela maior fica como esta — o `max` so levanta o piso,
# como em `janela_assentada`.
_NMP_T_DOM_MIN = 5.5


def _mergulho_relativo(spec: SystemSpec, t: np.ndarray) -> float:
    """`-min(y)/ptp(y)` da serie limpa, no sentido do degrau. 0 se nao mergulha."""
    y = step_response(spec, t)
    faixa = float(np.ptp(y))
    if faixa < 1e-15:
        return 0.0
    d = 1.0 if float(spec.K) * float(spec.step_amplitude) >= 0.0 else -1.0
    return max(0.0, -float(np.min(y * d)) / faixa)


def _resolve_a_zero(spec: SystemSpec, t: np.ndarray, alvo: float) -> float:
    """Menor `a > 0` cuja resposta mergulha `alvo` da faixa. Bissecao.

    Monotona em `a`: o mergulho tende a zero com `a -> 0` e a 1 com `a` grande
    (a parcela `-a*y0'` domina numerador e denominador). 60 iteracoes levam o
    intervalo relativo a 1e-18, muito abaixo de qualquer coisa que a
    rasterizacao distinga.
    """
    lo, hi = 0.0, 1.0 * spec.t_dom
    for _ in range(80):                     # expande ate cobrir o alvo
        if _mergulho_relativo(replace(spec, a_zero=hi), t) >= alvo:
            break
        hi *= 2.0
    for _ in range(60):
        meio = 0.5 * (lo + hi)
        if _mergulho_relativo(replace(spec, a_zero=meio), t) < alvo:
            lo = meio
        else:
            hi = meio
    return float(hi)


# Estrato MULTI-DEGRAU. Numeros escolhidos para que o transitorio do PRIMEIRO
# degrau — a verdade declarada — esteja de fato visivel antes do proximo
# entrar, e para que o corpus cubra a faixa em que a deteccao e dificil.
#
# `_MULTI_SEP` e a separacao entre degraus em constantes de tempo dominantes.
# Medido no corpus aleatorio de 297 figuras: os multi-degrau que a heuristica
# NAO detecta tem separacao mediana de 1,41 t_dom contra 1,98 dos detectados, e
# razao |U2/U1| mediana de 0,535 contra 0,771. O estrato tem de cobrir os dois
# regimes, senao ensina so o caso facil — por isso a faixa comeca em 0,8.
# 0,4 e nao 0,8 no piso. A primeira versao usava (0.8, 3.0), copiado do
# gerador aleatorio, e o corpus saia SISTEMATICAMENTE MAIS FACIL que o real:
# separacao p10/p50 de 1,12/2,22 t_dom contra 0,89/1,72 do real. Medido na
# cabeca treinada nesse corpus e avaliada no real, a AUC por faixa de separacao
# e monotona e denuncia o buraco:
#     0,0 - 1,2 t_dom -> AUC 0,518  (acaso; e a faixa que o corpus nao cobre)
#     1,2 - 2,0 t_dom -> AUC 0,666
#     2,0 - 3,5 t_dom -> AUC 0,739
# Separacao apertada e o caso DIFICIL — o segundo degrau entra antes de o
# primeiro transitorio terminar — e e justamente onde o corpus tinha menos
# amostras. NECESSARIO, NAO SUFICIENTE: mesmo na faixa bem coberta a AUC real
# fica em 0,739 contra 0,99 in-distribution, entao sobra um componente de
# dominio nao identificado.
_MULTI_SEP = (0.4, 3.0)
_MULTI_RAZAO = (0.25, 1.5)      # |U_i / U_1|
_MULTI_N = (1, 2, 3)
_MULTI_P = (0.20, 0.55, 0.25)


# Janela do estrato, em constantes de tempo dominantes DEPOIS do tempo morto.
# Sorteada IGUAL para 1, 2 e 3 degraus — ver o aviso de vazamento abaixo.
_MULTI_JANELA = (2.5, 9.0)


def sorteia_degraus(rng: np.random.Generator, spec: SystemSpec) -> tuple:
    """`((amplitude, instante), ...)` e a janela, para o estrato multi-degrau.

    Todos os degraus tem o MESMO SINAL, que e o caso dificil: a soma continua
    monotona e em forma de S, e por isso o residuo de um ajuste de degrau unico
    quase nao denuncia o segundo degrau (medido: o piso de residuo barra 80,7 %
    dos nao detectados). Degraus de sinais opostos criariam uma inversao
    visivel e tornariam o estrato facil demais para ser util.

    VAZAMENTO QUE ESTA FUNCAO EXISTE PARA EVITAR. A primeira versao sorteava as
    separacoes e depois ESTICAVA `t_end` para caber o ultimo degrau. Efeito
    medido em 1900 amostras: `janela/t_dom` passava a separar 1 degrau de 2+
    sozinha, com AUC 0,822 — um atalho. Uma cabeca treinada assim aprende a ler
    "quantas constantes de tempo cabem no quadro" em vez de "ha uma
    re-aceleracao", chega a AUC 0,98 na validacao sintetica e desaba para 0,60
    no corpus real. E o mesmo tipo de defeito que a regra anti-vazamento de
    `randomize.py` existe para impedir, so que pelo eixo do tempo em vez do
    estilo.

    A correcao: a JANELA e sorteada PRIMEIRO, da mesma distribuicao para
    qualquer numero de degraus, e os degraus sao colocados DENTRO dela. Com
    isso `janela/t_dom` fica identica nas duas classes e deixa de informar.
    """
    t_dom = dominant_time_constant(spec.order, spec.tau, spec.wn, spec.zeta)
    janela = float(_loguniform(rng, *_MULTI_JANELA))          # em t_dom
    n = int(rng.choice(_MULTI_N, p=_MULTI_P))
    U1 = float(spec.step_amplitude)
    degraus = [(U1, 0.0)]
    # O PISO E EM t_dom, NAO EM FRACAO DA JANELA. A versao anterior usava
    # `lo = 0.25 * janela`, e era ELE — nao `_MULTI_SEP[0]` — que amarrava a
    # separacao minima: com `janela` mediana de 4,85 t_dom, `0,25*janela` da
    # 1,2 t_dom, exatamente o p10 medido. Baixar `_MULTI_SEP[0]` de 0,8 para
    # 0,4 nao mudou uma virgula da distribuicao, porque ele so agia como piso
    # de desempate entre posicoes sorteadas. Ancorar `lo` em `_MULTI_SEP[0]`
    # faz a faixa dificil (segundo degrau entrando antes de o primeiro
    # transitorio terminar) aparecer de fato no corpus.
    lo, hi = _MULTI_SEP[0], 0.80 * janela
    if hi <= lo:
        hi = lo + 0.1
    if n > 1 and hi - lo > 0:
        pos = np.sort(rng.uniform(lo, hi, size=n - 1))
        anterior = 0.0
        for x in pos:
            inst = float(max(x, anterior + _MULTI_SEP[0]))
            if inst > janela * 0.90:      # nao cabe mais: para de acrescentar
                break
            degraus.append((U1 * float(_loguniform(rng, *_MULTI_RAZAO)),
                            inst * t_dom))
            anterior = inst
    return tuple(degraus), float(janela * t_dom)


def sample_system(rng: np.random.Generator) -> SystemSpec:
    """Sorteia (order, K, tau|wn/zeta, theta) e a janela temporal."""
    order = ORDERS[int(rng.integers(0, 2))]
    K = _loguniform(rng, 0.2, 20.0)
    if order == "fopdt":
        tau: float | None = _loguniform(rng, 0.05, 50.0)
        wn: float | None = None
        zeta: float | None = None
    else:
        tau = None
        wn = _loguniform(rng, 0.02, 20.0)
        zeta = float(rng.uniform(0.10, 3.00))
    t_dom = dominant_time_constant(order, tau, wn, zeta)
    theta = _loguniform(rng, 0.05, 1.0) * t_dom
    t_end = theta + _loguniform(rng, 0.5, 6.0) * t_dom
    return SystemSpec(
        order=order,
        K=K,
        tau=tau,
        theta=float(theta),
        wn=wn,
        zeta=zeta,
        t_start=0.0,
        t_end=float(t_end),
        step_amplitude=STEP_AMPLITUDE,
    )


def _derivada_degrau(spec: SystemSpec, t: np.ndarray) -> np.ndarray:
    """d/dt da resposta ao degrau dos POLOS (= resposta ao impulso, escalada).

    Analitica, nao numerica: a diferenca finita erraria justamente no salto em
    `t = theta`, que e' a geometria que o estrato fora-da-familia existe para
    exibir. Fonte da forma fechada: derivar termo a termo a expressao de
    `step_response`; para `zeta > 1` usa-se `r1*r2 = wn^2`.

    Superpoe `degraus` pelo mesmo argumento de `step_response`: derivar e'
    linear, entao a derivada da soma e' a soma das derivadas.
    """
    if spec.degraus:
        t = np.asarray(t, dtype=float)
        d = np.zeros_like(t)
        for amplitude, instante in spec.degraus:
            parcela = replace(spec, degraus=None, step_amplitude=float(amplitude),
                              theta=float(spec.theta) + float(instante))
            d = d + _derivada_degrau(parcela, t)
        return d
    t = np.asarray(t, dtype=float)
    u = t - spec.theta
    active = u >= 0.0
    uu = np.where(active, u, 0.0)
    s = spec.step_amplitude * spec.K

    if spec.order == "fopdt":
        tau = float(spec.tau)
        d = (s / tau) * np.exp(-uu / tau)
    else:
        wn = float(spec.wn)
        zeta = float(spec.zeta)
        if abs(zeta - 1.0) <= 1e-6:
            d = s * wn * wn * uu * np.exp(-wn * uu)
        elif zeta < 1.0:
            wd = wn * np.sqrt(1.0 - zeta * zeta)
            d = s * (wn * wn / wd) * np.exp(-zeta * wn * uu) * np.sin(wd * uu)
        else:
            rad = np.sqrt(zeta * zeta - 1.0)
            r1 = wn * (-zeta + rad)
            r2 = wn * (-zeta - rad)
            d = s * (wn * wn) * (np.exp(r1 * uu) - np.exp(r2 * uu)) / (r1 - r2)
    return np.where(active, d, 0.0)


def step_response(spec: SystemSpec, t: np.ndarray) -> np.ndarray:
    """Resposta analitica limpa ao degrau (contrato §1). Vetorizada.

    Com `spec.a_zero` preenchido a planta vira `(1 - a*s) * G_polos(s)`, e a
    resposta ao degrau e' `y0(t) - a * y0\'(t)` — exata, porque derivar e'
    linear. Para `fopdt` o resultado tem grau relativo ZERO: ha um SALTO em
    `t = theta` para `-K*a/tau`, e depois a subida. Para `second` o grau
    relativo e' 1: nao ha salto, a curva mergulha e volta. Os dois sao fase
    nao-minima; o primeiro e' o que quebra o prior "aproximacao monotona a um
    patamar" com mais forca, porque nenhuma curva DENTRO da familia e'
    descontinua em `t = theta` (ali o valor e' 0, igual ao repouso).

    Com `spec.degraus` preenchido devolve a SUPERPOSICAO — mesma planta, uma
    parcela por degrau, cada uma deslocada pelo instante do seu degrau. O
    sistema e linear, entao superpor e exato, nao aproximacao. Com `degraus`
    None o caminho e byte a byte o historico.
    """
    if spec.a_zero:
        # Antes da superposicao de proposito: o zero e' da PLANTA, entao ele
        # multiplica cada parcela igualmente e `y0 - a*y0\'` da o mesmo
        # resultado somando antes ou depois. Fazer aqui evita recursao dupla.
        base = replace(spec, a_zero=None)
        return (step_response(base, t)
                - float(spec.a_zero) * _derivada_degrau(base, t))
    if spec.degraus:
        t = np.asarray(t, dtype=float)
        y = np.zeros_like(t)
        for amplitude, instante in spec.degraus:
            parcela = replace(spec, degraus=None, step_amplitude=float(amplitude),
                              theta=float(spec.theta) + float(instante))
            y = y + step_response(parcela, t)
        return y
    t = np.asarray(t, dtype=float)
    u = t - spec.theta
    active = u >= 0.0
    uu = np.where(active, u, 0.0)
    s = spec.step_amplitude * spec.K

    if spec.order == "fopdt":
        y = s * (1.0 - np.exp(-uu / float(spec.tau)))
    else:
        wn = float(spec.wn)
        zeta = float(spec.zeta)
        if abs(zeta - 1.0) <= 1e-6:
            y = s * (1.0 - np.exp(-wn * uu) * (1.0 + wn * uu))
        elif zeta < 1.0:
            wd = wn * np.sqrt(1.0 - zeta * zeta)
            y = s * (
                1.0
                - np.exp(-zeta * wn * uu)
                * (np.cos(wd * uu) + (zeta * wn / wd) * np.sin(wd * uu))
            )
        else:
            rad = np.sqrt(zeta * zeta - 1.0)
            r1 = wn * (-zeta + rad)
            r2 = wn * (-zeta - rad)
            y = s * (1.0 + (r2 * np.exp(r1 * uu) - r1 * np.exp(r2 * uu)) / (r1 - r2))

    return np.where(active, y, 0.0)


def entrada_acumulada(spec: SystemSpec, t: np.ndarray) -> np.ndarray:
    """Sinal de entrada `u(t)` — a escada acumulada, sem o tempo morto.

    Nao entra em modelo nenhum: serve para o meta (verdade auditavel) e, na
    Etapa 2 do plano, para a segunda mascara. Com `degraus` None e uma
    constante em `step_amplitude`, que e o degrau em `t=0` do corpus base.
    """
    t = np.asarray(t, dtype=float)
    if not spec.degraus:
        return np.full_like(t, float(spec.step_amplitude))
    u = np.zeros_like(t)
    acc = 0.0
    for amplitude, instante in spec.degraus:
        u[t >= float(instante)] = acc + float(amplitude)
        acc += float(amplitude)
    return u


# --------------------------------------------------------------------------
# Ruido e quantizacao
# --------------------------------------------------------------------------


def _apply_noise(y: np.ndarray, style: RenderStyle, rng: np.random.Generator) -> np.ndarray:
    """Ruido gaussiano aditivo (SNR em dB sobre a variancia do sinal) + quantizacao."""
    y_out = np.asarray(y, dtype=float).copy()
    var = float(np.var(y_out))
    if var > 0.0 and np.isfinite(var):
        sigma = float(np.sqrt(var / (10.0 ** (style.snr_db / 10.0))))
        y_out = y_out + sigma * rng.standard_normal(y_out.size)

    levels = int(style.quantization_levels)
    if levels > 1:
        lo = float(np.min(y_out))
        hi = float(np.max(y_out))
        span = hi - lo
        if span > 1e-15:
            q = np.round((y_out - lo) / span * (levels - 1))
            y_out = lo + q * span / (levels - 1)
    return y_out


# --------------------------------------------------------------------------
# Renderizacao
# --------------------------------------------------------------------------


def _axis_limits(
    t: np.ndarray, y: np.ndarray, style: RenderStyle
) -> tuple[tuple[float, float], tuple[float, float]]:
    t0, t1 = float(np.min(t)), float(np.max(t))
    span_t = t1 - t0
    if span_t <= 0.0:
        span_t = max(abs(t1), 1.0)
    xlim = (t0 - style.x_margin_lo * span_t, t1 + style.x_margin_hi * span_t)

    y0, y1 = float(np.min(y)), float(np.max(y))
    span_y = y1 - y0
    if span_y <= 1e-12:
        span_y = max(abs(y1), 1.0) * 0.1
    ylim = (y0 - style.y_margin_lo * span_y, y1 + style.y_margin_hi * span_y)
    return xlim, ylim


def _ylim_plato_no_meio(y: np.ndarray, ylim: tuple[float, float],
                        rng: np.random.Generator) -> tuple[float, float]:
    """Estica o quadro para que o NIVEL DE REPOUSO caia longe da borda.

    Estrato OOD opt-in que ataca um PRIOR DE POSICAO medido. No corpus atual o
    repouso esta sempre colado numa borda: em `data/train` (K > 0) ele ocupa a
    fracao 0,038 a 0,137 da altura do quadro; em `data/train_kneg` (K < 0),
    0,864 a 0,963. Na faixa do meio (0,25 a 0,75) o corpus tem **0,0 %** das
    amostras, nos dois. A rede aprende "o repouso fica na borda" e perde o
    plato quando ele nao esta la — medido em 40 figuras de fase nao-minima,
    onde o repouso cai no MEIO por construcao: zero de 60 colunas iniciais com
    tinta, `_nivel_de_repouso` mede o pico, e a guarda `_UNDERSHOOT_MAX` deixa
    de funcionar (deteccao de 3 em 40, correlacao -0,11 com o undershoot real).

    So o QUADRO muda. A serie, o spec e o estilo ficam identicos aos da mesma
    seed sem o flag, entao o estrato e comparavel AMOSTRA A AMOSTRA com o base
    — a mesma disciplina do `ganho_negativo` (§40.5). Nao se acrescenta
    fenomeno fisico nenhum: o modelo continua na familia FOPDT/2a ordem e o
    `meta.json` nao muda de contrato.
    """
    lo, hi = float(ylim[0]), float(ylim[1])
    repouso = float(np.median(y[:max(1, min(5, y.size))]))
    alvo = float(rng.uniform(0.25, 0.75))
    if not (lo < repouso < hi):
        return ylim
    frac = (repouso - lo) / max(hi - lo, 1e-12)
    if frac < 0.5:
        # repouso perto do rodape: estica PARA BAIXO ate ele subir ate `alvo`
        altura = (hi - repouso) / max(1.0 - alvo, 1e-6)
        return (repouso - alvo * altura, hi)
    # repouso perto do topo: estica PARA CIMA
    altura = (repouso - lo) / max(alvo, 1e-6)
    return (lo, lo + altura)


def _new_figure(style: RenderStyle, facecolor: str) -> tuple[Figure, Axes]:
    fig = Figure(figsize=style.figsize, dpi=style.dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(facecolor)
    ax = fig.add_axes(list(style.axes_rect))
    ax.set_facecolor(facecolor)
    return fig, ax


# Luminancia ITU-R BT.601. E a mesma formula do `identify/calibrate.py`, que
# roda o Estagio B INTEIRO em cinza (moldura, mascara de tinta, blobs de texto,
# recortes do OCR), e a mesma do ramo `in_ch == 1` de
# `identify/extract.py::predict_mask`.
#
# ATENCAO ao que este arquivo usa a luminancia PARA (ver `_cor_colidente`):
# NAO e mais o que a U-Net enxerga. O checkpoint promovido e `in_ch=3` e recebe
# RGB; o ramo de cinza do `predict_mask` so roda para checkpoints antigos.
_PESO_CINZA = (0.299, 0.587, 0.114)


def _cinza(rgb) -> float:
    """Luminancia ITU-R BT.601 de uma cor RGB (0-255), em 0-255.

    Foi escrita como "o que a U-Net enxerga" e isso valia quando o Estagio A
    era de 1 canal. Hoje o checkpoint promovido e `in_ch=3` e nao converte —
    ver `_cor_colidente` para o que isso fez com o estrato construido sobre
    esta funcao.
    """
    return sum(p * c for p, c in zip(_PESO_CINZA, rgb))


def _cor_colidente(hex_curva: str) -> str:
    """Cor com a MESMA luminancia da curva e matiz oposta.

    PREMISSA ORIGINAL, HOJE FALSA PARA O CHECKPOINT PROMOVIDO. Isto foi escrito
    quando o Estagio A convertia a imagem para cinza antes da U-Net: dois
    objetos de luminancia igual chegavam a rede como o MESMO byte — medido nas
    duas imagens reais do Ruling 55, curva (44,160,44) e reta de referencia
    (230,61,61) viram ambas 112. Separar um do outro nao era dificil, era
    IMPOSSIVEL: nenhuma funcao de uma entrada distingue pontos onde a entrada e
    identica. O estrato existia para reproduzir exatamente essa
    impossibilidade.

    O checkpoint promovido e `in_ch=3` e recebe R, G e B separados
    (`identify/extract.py::predict_mask`, ramo RGB). As duas cores que
    colidiam em 112 chegam a ele perfeitamente distinguiveis, e a
    impossibilidade deixou de existir.

    O QUE ISSO SIGNIFICA, e por que a funcao FICA: o estrato continua valido —
    um distrator de luminancia igual sobreposto ao patamar e um caso legitimo,
    e agora serve de CONTROLE do ganho que o RGB trouxe, porque e o caso em que
    1 canal nao tinha chance e 3 canais tem. O que ele deixou de ser e o pior
    caso adversarial: para um modelo de 3 canais, a colisao de luminancia nao e
    mais especial. Quem for medir dificuldade de estrato contra o checkpoint
    atual precisa saber disso, senao le como "a rede ficou boa" o que e "o
    estrato deixou de ser dificil".

    NAO MEDIDO: qual e o IoU deste estrato com o checkpoint RGB, nem se algum
    estrato adversarial equivalente EM RGB (colisao nos tres canais, nao so na
    luminancia) faria sentido construir.

    Sobre a escolha da cor, e independente do acima: uma cor sorteada ao acaso
    colide raramente (medido no gerador: 2,05% dos distratores ficam a menos de
    2 bytes da curva) e, pior, colide sem se SOBREPOR — e a medicao mostrou que
    colisao sem sobreposicao e inofensiva (Spearman colisao x IoU = -0,006,
    p=0,86, n=900). Aqui a colisao e construida de proposito e combinada com
    sobreposicao no patamar.

    Depende so de `style.line_color`, que `sample_style` sorteia cego ao spec,
    entao nao abre caminho de vazamento (tests/test_part1.py:1115).
    """
    import colorsys
    rgb = tuple(int(hex_curva[i:i + 2], 16) for i in (1, 3, 5))
    alvo = _cinza(rgb)
    h, ll, s = colorsys.rgb_to_hls(*[c / 255.0 for c in rgb])
    girado = colorsys.hls_to_rgb((h + 0.5) % 1.0, ll, s)
    lg = _cinza([c * 255.0 for c in girado])
    if lg <= 1e-6:
        return hex_curva
    # reescala para casar a luminancia; se estourar, dessatura ate caber.
    for _ in range(24):
        k = alvo / lg
        cand = [c * 255.0 * k for c in girado]
        if max(cand) <= 255.0:
            return "#%02x%02x%02x" % tuple(int(round(min(255.0, max(0.0, c)))) for c in cand)
        s *= 0.85
        girado = colorsys.hls_to_rgb((h + 0.5) % 1.0, ll, s)
        lg = _cinza([c * 255.0 for c in girado])
        if lg <= 1e-6:
            break
    return hex_curva

def _plot_curve(ax, t: np.ndarray, y: np.ndarray, style: RenderStyle, color: str, label=None):
    kwargs = dict(
        color=color,
        linewidth=style.line_width,
        linestyle=style.line_style,
        solid_capstyle="round",
        antialiased=True,
    )
    if style.marker is not None:
        kwargs.update(
            marker=style.marker,
            markevery=style.markevery,
            markersize=style.marker_size,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=max(0.5, style.line_width * 0.6),
        )
    if label is not None:
        kwargs["label"] = label
    return ax.plot(t, y, **kwargs)


def _apply_locators(ax, style: RenderStyle, xlim, ylim) -> None:
    # limita o numero de ticks ao tamanho fisico do eixo (evita rotulos
    # sobrepostos em figuras pequenas). Depende so do estilo, nunca do rotulo.
    w_in = style.axes_rect[2] * style.figsize[0]
    h_in = style.axes_rect[3] * style.figsize[1]
    nx = int(np.clip(min(style.n_xbins, int(2.2 * w_in)), 3, 8))
    ny = int(np.clip(min(style.n_ybins, int(3.0 * h_in)), 3, 8))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nx, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ny, min_n_ticks=3))
    # hipotese §1.6.2: pelo menos 2 ticks maiores rotulados por eixo
    for axis, lim in ((ax.xaxis, xlim), (ax.yaxis, ylim)):
        vals = np.asarray(axis.get_major_locator()())
        inside = vals[(vals >= min(lim) - 1e-12) & (vals <= max(lim) + 1e-12)]
        if inside.size < 2:
            axis.set_major_locator(LinearLocator(3))


def _ticks_in_view(ax, axis_name: str, lim, height_px: int) -> list[list[float]]:
    """Ticks maiores visiveis dentro dos limites, em indices de pixel da imagem.

    Devolve [[coord_px, valor], ...]: coordenada horizontal para o eixo x e
    vertical (origem no topo) para o eixo y.
    """
    lo, hi = min(lim), max(lim)
    vals = ax.get_xticks() if axis_name == "x" else ax.get_yticks()
    out: list[list[float]] = []
    for v in np.asarray(vals, dtype=float):
        if not (lo - 1e-12 <= v <= hi + 1e-12):
            continue
        if axis_name == "x":
            coord = float(ax.transData.transform((v, lo))[0]) - 0.5
        else:
            py_mpl = float(ax.transData.transform((lo, v))[1])
            coord = float(height_px - py_mpl) - 0.5
        out.append([coord, float(v)])
    return out


def render_sample(
    spec: SystemSpec,
    style: RenderStyle,
    out_dir: str | Path,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
    *,
    seed: int | None = None,
    reta_no_patamar: bool = False,
    anotacao_com_seta: bool = False,
    banda_de_acomodacao: bool = False,
    plato_no_meio: bool = False,
) -> dict:
    """Escreve image.png, mask.png e meta.json em out_dir. Devolve o dict do meta."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rng is None:
        rng = np.random.default_rng(0)

    # Marca o estrato OOD numa COPIA do estilo (via `dataclasses.replace`),
    # nunca mutando o objeto `style` recebido do chamador: `render_sample` nao
    # e o dono desse objeto, e mutar um argumento e efeito colateral
    # observavel para quem chamou. `has_reference_line` e campo de RENDER
    # (dataset/randomize.py), nao sorteado por `sample_style`.
    style = replace(style, has_reference_line=bool(reta_no_patamar),
                    has_annotation_arrow=bool(anotacao_com_seta),
                    has_settling_band=bool(banda_de_acomodacao))

    t = np.linspace(spec.t_start, spec.t_end, N_SERIES)
    y_clean = step_response(spec, t)
    u_serie = entrada_acumulada(spec, t)
    y_draw = _apply_noise(y_clean, style, rng) if add_noise else y_clean.copy()

    xlim, ylim = _axis_limits(t, y_draw, style)
    if plato_no_meio:
        # Depois de `_axis_limits` e antes de qualquer plotagem: a figura da
        # imagem e a da mascara usam os MESMOS limites (linhas ~413 e ~601),
        # entao a verdade fica consistente.
        ylim = _ylim_plato_no_meio(y_draw, ylim, rng)

    # ---------------- figura de verdade ----------------
    fig, ax = _new_figure(style, style.bg_color)
    _plot_curve(ax, t, y_draw, style, style.line_color,
                label=style.legend_text if style.has_legend else None)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _apply_locators(ax, style, xlim, ylim)

    # distratores: posicoes uniformes nos limites dos eixos, sem relacao com o rotulo
    distractors = list(style.distractors)
    if reta_no_patamar:
        # Estrato OOD, opt-in (HANDOFF_P2_7 §34.5). O laco acima sorteia
        # posicao uniforme; aqui se acrescenta uma reta horizontal FIXADA no
        # patamar da curva, que e o caso real do setpoint marcado e onde a
        # U-Net perde a curva. Entra no RENDER e nao em `sample_style`, porque
        # a posicao depende do sistema e `sample_style` tem de continuar cega
        # ao spec (tests/test_part1.py:1115). Sob `if`, para que o caminho
        # padrao nao mude um byte.
        # `line_style` tracejado ("--"), de proposito: e o traco real de um
        # setpoint marcado (a linha do caso real que motivou este estrato), e
        # e o caso DIFICIL: um traco tracejado fragmenta em varios blocos
        # separados por coluna, exatamente a condicao multi-bloco que o
        # extrator de polilinha precisa resolver. Solido seria mais facil de
        # extrair e menos fiel ao problema real — nao serviria de estrato de
        # teste para o defeito do §34.5.
        distractors = distractors + [
            {"orient": "h", "frac": None, "no_patamar": True,
             # Cor de LUMINANCIA IGUAL a da curva (ver `_cor_colidente`, e
             # LEIA a ressalva la: o checkpoint promovido e RGB e nao colapsa
             # mais as duas cores). Sem isso o estrato so testa "existe uma
             # reta", e a medicao mostrou que reta de cor qualquer nao degrada
             # nada (Spearman colisao x IoU = -0,006, p=0,86, n=900).
             "color": _cor_colidente(style.line_color), "line_style": "--",
             # ACIMA da curva: os distratores comuns ficam em `zorder=1`, sob
             # ela, e por isso nao ocluem. O fenomeno real do Ruling 55 e a
             # reta passar POR CIMA do trecho assentado — e a mistura
             # antialiasada dos dois que apaga o contraste (medido: cai a 32%
             # do original) e faz a mascara perder a curva.
             "zorder": 3,
             # Espessura acompanhando a da curva: uma reta mais fina deixaria
             # borda de curva visivel dos dois lados e nao ocluiria de fato.
             "line_width": max(2.0, style.line_width * 1.8), "alpha": 1.0}
        ]
    for d in distractors:
        if d["orient"] == "h":
            if d.get("no_patamar"):
                # SETPOINT comandado (`K * degrau`), nao `y_draw[-1]` nem
                # `y_clean[-1]`. Tres razoes, nesta ordem:
                #  - e o que um grafico real marca: nas duas imagens do
                #    Ruling 55 a reta esta exatamente em 1,0, o valor de
                #    referencia, nao "onde a curva calhou de terminar";
                #  - `y_draw[-1]` e uma amostra RUIDOSA, e a reta ancorada
                #    nela cobre a curva so em parte (medido: 12 px de desvio
                #    numa seed, e a oclusao vira parcial);
                #  - `y_clean[-1]` ainda oscila quando a janela nao basta para
                #    assentar (subamortecido), e ai a reta sai do patamar.
                # E derivavel do meta (`params.K` x `step_amplitude`), entao o
                # estrato fica localizavel sem chave nova no contrato.
                val = float(spec.K * spec.step_amplitude)
            else:
                val = ylim[0] + d["frac"] * (ylim[1] - ylim[0])
            ax.axhline(
                val, color=d["color"], linestyle=d["line_style"],
                linewidth=d["line_width"], alpha=d["alpha"],
                zorder=d.get("zorder", 1),
            )
        else:
            val = xlim[0] + d["frac"] * (xlim[1] - xlim[0])
            ax.axvline(
                val, color=d["color"], linestyle=d["line_style"],
                linewidth=d["line_width"], alpha=d["alpha"],
                zorder=d.get("zorder", 1),
            )

    if style.has_grid:
        ax.grid(True, color=style.axes_color, alpha=style.grid_alpha,
                linestyle=style.grid_style, linewidth=0.6, zorder=0)
    if style.has_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="minor", direction=style.tick_direction,
                       length=style.tick_length * 0.5, colors=style.axes_color)
    ax.tick_params(
        which="major",
        direction=style.tick_direction,
        length=style.tick_length if style.has_major_ticks else 0.0,
        colors=style.axes_color,
        labelsize=style.font_size * 0.85,
    )
    for name, visible in zip(("left", "bottom", "right", "top"), style.spines):
        ax.spines[name].set_visible(bool(visible))
        ax.spines[name].set_color(style.axes_color)

    if style.has_title:
        ax.set_title(style.title_text, color=style.axes_color, fontsize=style.font_size)
    if style.has_xlabel:
        ax.set_xlabel(style.xlabel_text, color=style.axes_color, fontsize=style.font_size)
    if style.has_ylabel:
        ax.set_ylabel(style.ylabel_text, color=style.axes_color, fontsize=style.font_size)
    if style.has_legend:
        leg = ax.legend(loc=style.legend_loc, fontsize=style.font_size * 0.8,
                        framealpha=style.legend_alpha)
        leg.get_frame().set_facecolor(style.bg_color)
        leg.get_frame().set_edgecolor(style.axes_color)
        for txt in leg.get_texts():
            txt.set_color(style.axes_color)
    for text, fx, fy in style.annotations:
        ax.text(fx, fy, text, transform=ax.transAxes, color=style.axes_color,
                fontsize=style.font_size * 0.8, zorder=5)

    if banda_de_acomodacao:
        # Banda de +-5 % em torno do SETPOINT comandado, atravessando a
        # figura inteira — o "criterio de acomodacao" que graficos de controle
        # marcam. Ancorada em `K * degrau` pelas MESMAS tres razoes do
        # `no_patamar` acima (e o que o grafico real marca; `y_draw[-1]` e
        # ruidoso; `y_clean[-1]` ainda oscila quando a janela nao assenta).
        # Cor de LUMINANCIA COLIDENTE com a da curva e `alpha` alto: uma banda
        # de cor qualquer nao muda o byte que a U-Net recebe, e o §34.5 ja
        # mediu que distrator sem colisao nao degrada nada (Spearman colisao x
        # IoU = -0,006, p = 0,86).
        alvo = float(spec.K * spec.step_amplitude)
        meia = 0.05 * abs(alvo) if abs(alvo) > 1e-9 else 0.05 * (ylim[1] - ylim[0])
        ax.axhspan(alvo - meia, alvo + meia,
                   color=_cor_colidente(style.line_color), alpha=0.55, zorder=2,
                   linewidth=0.0)

    if anotacao_com_seta:
        # Caixa de texto ligada por SETA ao pico da curva. A seta e' o objeto
        # que importa: e um segmento de reta espesso saindo da curva para
        # fora dela, e e por ele que a mascara foge (medido na imagem externa
        # que motivou este estrato — a polilinha subia pela seta e depois
        # pulava para a legenda). O texto sozinho ja existia em
        # `style.annotations` e nunca degradou nada.
        k = int(np.argmax(np.abs(y_draw - float(y_draw[0]))))
        tp, yp = float(t[k]), float(y_draw[k])
        # A caixa fica DIAGONALMENTE acima e a direita do pico, dentro da
        # moldura: e a posicao que o matplotlib produz com `xytext` em coords
        # de dados e a que o caso real exibe.
        dx = 0.18 * (xlim[1] - xlim[0])
        dy = 0.20 * (ylim[1] - ylim[0])
        cx = min(tp + dx, xlim[1] - 0.02 * (xlim[1] - xlim[0]))
        cy = min(yp + dy, ylim[1] - 0.05 * (ylim[1] - ylim[0]))
        cor = _cor_colidente(style.line_color)
        ax.annotate(
            f"Pico: {yp:.3f}\nem t={tp:.2f}",
            xy=(tp, yp), xytext=(cx, cy),
            fontsize=style.font_size * 0.8, color=style.axes_color,
            bbox={"boxstyle": "round", "facecolor": style.bg_color,
                  "edgecolor": style.axes_color},
            arrowprops={"arrowstyle": "->", "color": cor,
                        "linewidth": max(2.0, style.line_width * 1.5)},
            zorder=6,
        )

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    h_px, w_px = int(buf.shape[0]), int(buf.shape[1])

    # calibracao a partir da figura de verdade (nunca "chutada").
    # matplotlib usa display coords com origem no canto INFERIOR esquerdo; a
    # imagem usa indices de pixel com origem no canto SUPERIOR esquerdo e o
    # centro do pixel i na coordenada continua i+0.5 -> subtrai-se 0.5.
    p = ax.transData.transform(np.array([[xlim[0], ylim[0]], [xlim[1], ylim[1]]]))
    sx = float((xlim[1] - xlim[0]) / (p[1, 0] - p[0, 0]))
    ox = float(xlim[0] - sx * (p[0, 0] - 0.5))
    sy = float(-(ylim[1] - ylim[0]) / (p[1, 1] - p[0, 1]))
    oy = float(ylim[0] - sy * (h_px - p[0, 1] - 0.5))

    # `plot_bbox_px` delimita a AREA DE DADOS (o retangulo dos eixos), em
    # indices de pixel, [x0, y0, x1, y1] com origem no canto superior esquerdo.
    # Convencao: os limites sao o CENTRO da spine desenhada (mesma convencao de
    # indice usada pela afim), logo o retangulo fica ~0.5 px "para dentro" da
    # borda externa do traco da moldura. Nao alterar: a afim foi validada
    # geometricamente contra a mascara com essa convencao.
    bb = ax.get_window_extent()
    plot_bbox_px = [
        int(round(bb.x0 - 0.5)),
        int(round(h_px - bb.y1 - 0.5)),
        int(round(bb.x1 - 0.5)),
        int(round(h_px - bb.y0 - 0.5)),
    ]
    ticks = {
        "x": _ticks_in_view(ax, "x", xlim, h_px),
        "y": _ticks_in_view(ax, "y", ylim, h_px),
    }

    fig.savefig(out / "image.png", dpi=style.dpi, facecolor=style.bg_color,
                metadata={"Software": None})

    # ---------------- figura-mascara (mesma geometria) ----------------
    mfig, max_ = _new_figure(style, "#000000")
    _plot_curve(max_, t, y_draw, style, "#ffffff")
    max_.set_xlim(*xlim)
    max_.set_ylim(*ylim)
    max_.set_axis_off()
    mfig.canvas.draw()
    mbuf = np.asarray(mfig.canvas.buffer_rgba())
    if mbuf.shape[:2] != (h_px, w_px):  # pragma: no cover
        raise RuntimeError("geometria da mascara difere da imagem")
    mask = np.where(mbuf[:, :, 0] > 127, 255, 0).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(out / "mask.png", optimize=False)

    fig.clf()
    mfig.clf()

    with Image.open(out / "image.png") as im:
        img_size = im.size  # (W, H)
    if img_size != (w_px, h_px):  # pragma: no cover
        raise RuntimeError(f"image.png {img_size} != canvas {(w_px, h_px)}")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": out.name,
        "seed": seed,
        "order": spec.order,
        "params": {
            "K": float(spec.K),
            "tau": None if spec.tau is None else float(spec.tau),
            "theta": float(spec.theta),
            "wn": None if spec.wn is None else float(spec.wn),
            "zeta": None if spec.zeta is None else float(spec.zeta),
        },
        "step_amplitude": float(spec.step_amplitude),
        # Estrato multi-degrau (schema v2). `params` acima descreve o PRIMEIRO
        # degrau — a convencao do `rg_multidegrau.py`, e o unico que a pipeline
        # pode recuperar de um prefixo. `n_degraus` e o ROTULO da cabeca de
        # contagem do Estagio A.
        "degraus": [[float(a), float(b)] for a, b in
                    (spec.degraus or ((spec.step_amplitude, 0.0),))],
        "n_degraus": len(spec.degraus) if spec.degraus else 1,
        "u_final": float(u_serie[-1]),
        "t_window": [float(spec.t_start), float(spec.t_end)],
        "plot_bbox_px": plot_bbox_px,
        "axis_affine": {"sx": sx, "ox": ox, "sy": sy, "oy": oy},
        "ticks": ticks,
        "series": {"t": [float(v) for v in t], "y": [float(v) for v in y_draw]},
        "noise": {
            "enabled": bool(add_noise),
            "snr_db": float(style.snr_db),
            "quantization_levels": int(style.quantization_levels),
        },
        "render": style.to_meta(),
    }
    if spec.a_zero:
        # CONDICIONAIS de proposito. Se `fora_da_familia` e `params["a"]`
        # saissem em toda amostra, todo meta.json do corpus base mudaria de
        # bytes para dizer `false`/`null` — quebrando a igualdade que
        # `test_o_padrao_nao_muda_um_byte` protege, sem informar nada. Um
        # leitor le `meta.get("fora_da_familia", False)`.
        #
        # `order` continua sendo a estrutura de POLOS e `params` continua
        # descrevendo-a corretamente; o que NAO vale mais e' tratar `params`
        # como um modelo da curva inteira. Quem compara ajuste contra verdade
        # (os oraculos) tem de olhar esta chave ANTES de comparar.
        meta["schema_version"] = SCHEMA_VERSION_FORA_DA_FAMILIA
        meta["fora_da_familia"] = True
        meta["params"]["a"] = float(spec.a_zero)
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    return meta


# --------------------------------------------------------------------------
# API de alto nivel
# --------------------------------------------------------------------------


def generate_sample(out_dir: str | Path, seed: int, add_noise: bool = True,
                    reta_no_patamar: bool = False,
                    janela_assentada: bool = False,
                    anotacao_com_seta: bool = False,
                    banda_de_acomodacao: bool = False,
                    ganho_negativo: bool = False,
                    multi_degrau: bool = False,
                    plato_no_meio: bool = False,
                    fase_nao_minima: bool = False) -> dict:
    """Sorteia sistema+estilo com streams independentes, renderiza e devolve o meta."""
    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(3)
    rng_sys = np.random.default_rng(children[0])
    rng_style = np.random.default_rng(children[1])
    rng_noise = np.random.default_rng(children[2])

    spec = sample_system(rng_sys)
    style = sample_style(rng_style)  # nao ve o spec: anti-vazamento estrutural
    if janela_assentada:
        # Janela longa o bastante para o PATAMAR ficar visivel. Eixo separado de
        # `reta_no_patamar` de proposito: sao dois fenomenos distintos e o
        # estrato OOD e a combinacao dos dois. Separados, dao ablacao (a reta
        # sozinha? a janela sozinha?) e permitem controle pareado no teste —
        # com os dois no mesmo parametro, a extensao muda a geometria da curva
        # e nenhum diferencial pixel a pixel e possivel.
        # Nao mexe no meta: a janela ja e observavel em `t_window`.
        t_dom = dominant_time_constant(spec.order, spec.tau, spec.wn, spec.zeta)
        spec = replace(spec, t_end=float(max(spec.t_end,
                                             spec.theta + _T_DOM_ESTRATO * t_dom)))
    if ganho_negativo:
        # Estrato OOD opt-in (§40.5), molde do `reta_no_patamar` (§34.5). So o
        # SINAL de K muda: `sample_system` continua sorteando K > 0, e |K| fica
        # identico ao do mesmo seed sem o flag. Isso e o que torna o estrato
        # comparavel AMOSTRA A AMOSTRA com o base — a diferenca entre os dois
        # e o sinal e nada mais, entao qualquer diferenca de resultado e
        # atribuivel a ele.
        #
        # Opt-in, e nao um sorteio dentro de `sample_system`, porque mexer no
        # sorteio moveria TODA amostra do corpus base, e com ela todo numero
        # historico da Parte 1 e da Parte 2. O corpus base fica byte a byte
        # identico e o estrato novo entra ao lado. `test_o_padrao_nao_muda_um_byte`
        # assevera isso comparando os PNG e o meta.json.
        #
        # Aplicado ao SPEC e depois de `sample_style`, nunca ao estilo: o
        # estilo nao pode ver o spec (anti-vazamento estrutural de
        # `randomize.py`), e um traco que mudasse de cor ou espessura por causa
        # do sinal do ganho ensinaria a rede a ler o sinal do RENDER em vez da
        # forma da curva.
        spec = replace(spec, K=-spec.K)
    if multi_degrau:
        # Estrato OOD opt-in, molde do `ganho_negativo` (§40.5). Sorteado com
        # `rng_sys`, NUNCA com `rng_style`: o estilo nao pode ver o spec
        # (anti-vazamento de `randomize.py`), e um render que mudasse com o
        # numero de degraus ensinaria a rede a ler o rotulo do render em vez da
        # forma da curva. A janela e esticada para caber o ultimo degrau mais o
        # transitorio dele, senao o segundo degrau cai fora do quadro e a
        # amostra fica rotulada como multi sem mostrar nada.
        # A JANELA vem junto e NAO depende do numero de degraus — ver o
        # vazamento documentado em `sorteia_degraus`.
        degraus, janela = sorteia_degraus(rng_sys, spec)
        spec = replace(spec, degraus=degraus,
                       t_end=float(spec.t_start + spec.theta + janela))
    if fase_nao_minima:
        # Estrato OOD opt-in, molde do `ganho_negativo` (§40.5) — e o unico que
        # sai FORA da familia de modelos do Estagio D. Sorteado com `rng_sys`
        # (nunca `rng_style`) e APLICADO POR ULTIMO, depois de `multi_degrau`,
        # para que a ordem dos consumos de `rng_sys` seja estavel: quem liga so
        # este flag ve exatamente o spec do mesmo seed sem flag, mais o zero.
        # E' o que torna o estrato comparavel amostra a amostra com o base.
        #
        # O zero NAO mexe nos polos. `order`, `tau`/`wn`/`zeta` e `t_dom`
        # continuam validos e continuam no meta com o mesmo significado; por
        # isso `order` nao ganha um terceiro valor. O que o meta ganha e' a
        # chave `fora_da_familia`, que diz ao leitor que `params` ja nao
        # descreve a curva inteira.
        # A janela SOBE ao piso de `_NMP_T_DOM_MIN` ANTES de resolver o zero
        # (ver la o porque): o mergulho e' medido em fracao da faixa de y, e a
        # faixa depende da janela. Resolver primeiro daria um `a` calibrado
        # contra um quadro que nao e' o que sera desenhado.
        # Isso quebra o pareamento pixel a pixel com o mesmo seed sem o flag
        # nas amostras de janela curta — mesmo custo que `janela_assentada`
        # paga, pelo mesmo motivo. Nas que ja tinham janela suficiente o
        # pareamento continua exato.
        spec = replace(spec, t_end=float(max(
            spec.t_end, spec.theta + _NMP_T_DOM_MIN * spec.t_dom)))
        alvo_merg = float(rng_sys.uniform(*_NMP_MERGULHO))
        grade = np.linspace(spec.t_start, spec.t_end, N_SERIES)
        spec = replace(spec, a_zero=_resolve_a_zero(spec, grade, alvo_merg))
    return render_sample(spec, style, out_dir, add_noise=add_noise, rng=rng_noise,
                         seed=int(seed), reta_no_patamar=reta_no_patamar,
                         anotacao_com_seta=anotacao_com_seta,
                         banda_de_acomodacao=banda_de_acomodacao,
                         plato_no_meio=plato_no_meio)


def _generate_one(args: tuple) -> str:
    (out_dir, seed, add_noise, reta, janela, seta, banda, kneg, multi, plato,
     nmp) = args
    generate_sample(out_dir, seed, add_noise=add_noise, reta_no_patamar=reta,
                    janela_assentada=janela, anotacao_com_seta=seta,
                    banda_de_acomodacao=banda, ganho_negativo=kneg,
                    multi_degrau=multi, plato_no_meio=plato,
                    fase_nao_minima=nmp)
    return str(out_dir)


def generate_dataset(
    out_dir: str | Path,
    n: int,
    seed: int = 0,
    workers: int | None = None,
    add_noise: bool = True,
    reta_no_patamar: bool = False,
    janela_assentada: bool = False,
    anotacao_com_seta: bool = False,
    banda_de_acomodacao: bool = False,
    ganho_negativo: bool = False,
    multi_degrau: bool = False,
    plato_no_meio: bool = False,
    fase_nao_minima: bool = False,
) -> list[str]:
    """Gera n amostras em paralelo. Resultado independe do numero de workers."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(root / f"sample_{i:05d}"), int(seed) * 1_000_003 + i, bool(add_noise),
         bool(reta_no_patamar), bool(janela_assentada),
         bool(anotacao_com_seta), bool(banda_de_acomodacao),
         bool(ganho_negativo), bool(multi_degrau), bool(plato_no_meio),
         bool(fase_nao_minima))
        for i in range(int(n))
    ]
    if workers is not None and workers <= 1:
        return [_generate_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_generate_one, jobs, chunksize=4))


def load_sample(sample_dir: str | Path) -> dict:
    """Le meta.json (series como np.ndarray) e anexa 'image' e 'mask' uint8."""
    d = Path(sample_dir)
    with open(d / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["series"] = {
        "t": np.asarray(meta["series"]["t"], dtype=float),
        "y": np.asarray(meta["series"]["y"], dtype=float),
    }
    with Image.open(d / "image.png") as im:
        meta["image"] = np.asarray(im.convert("RGB"), dtype=np.uint8)
    with Image.open(d / "mask.png") as im:
        meta["mask"] = np.asarray(im.convert("L"), dtype=np.uint8)
    return meta


if __name__ == "__main__":  # pragma: no cover
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sd = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    dirs = generate_dataset(out, n, seed=sd, workers=os.cpu_count())
    print(f"{len(dirs)} amostras em {out}")
