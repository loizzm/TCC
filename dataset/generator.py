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
import matplotlib.style  # noqa: E402  (submodulo nao vem no `import matplotlib`)

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, LinearLocator, MaxNLocator  # noqa: E402
from PIL import Image  # noqa: E402

from dataset.randomize import (RenderStyle, _sample_text,  # noqa: E402
                               luminance, sample_style)

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

# 1 -> 2: entrou o estrato MULTI-DEGRAU, com as chaves `degraus`, `n_degraus`
# e `u_final`.
# 2 -> 2 (sem bump): a frente de multi-degrau foi REMOVIDA do codigo (§68) e
# com ela as tres chaves. A versao NAO recua para 1: o corpus em disco gerado
# na v2 continua valido e continua declarando 2, e recuar faria dois conteudos
# diferentes compartilharem o mesmo numero. Um leitor deve usar
# `meta.get("n_degraus", 1)`. Ver MULTI_DEGRAU.md.
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


# Estrato RUIDO ALTO. `dataset/randomize.py` sorteia `snr_db` em U(20, 60) e
# NUNCA desce de 20 — medido nos quatro corpora, minimo 20,0 a 20,3 dB, mediana
# 40. Metade das amostras de treino e praticamente limpa, e a rede aprendeu a
# segmentar curva NITIDA.
#
# O DANO, medido num lote de 100 figuras de avaliacao com ruido em cinco niveis
# de SNR (o gerador de avaliacao tambem nao tinha ruido — foi acrescentado em
# `gera_lote_ruidoso.py`). Entrega fisica, com a guarda adaptativa ja ligada:
#     30 dB  95 %      25 dB  95 %      20 dB  80 %
#     15 dB  80 %      10 dB  85 %
# e o acerto CONJUNTIVO no nivel PRATICO: 90 %, 90 %, 70 %, 60 %, 55 %. O
# joelho cai exatamente na borda da distribuicao de treino, e abaixo dela a
# mascara esta extrapolando.
#
# A FAIXA e' U(5, 20), CONTIGUA com a base: a uniao cobre 5 a 60 dB sem buraco.
# Uniforme em dB, e nao em razao de potencia, porque dB ja e' escala
# logaritmica — uniforme ali distribui o esforco por igual entre "levemente
# ruidoso" e "muito ruidoso". O piso de 5 dB deixa margem abaixo dos 10 dB que
# o lote de avaliacao usa: treinar ate exatamente o limite do teste seria
# treinar PARA o teste.
#
# CONVERSAO para o que se ve na figura (desvio do ruido / faixa de y, medido):
#     30 dB -> 0,011    20 dB -> 0,035    15 dB -> 0,062    10 dB -> 0,114
_SNR_BAIXO = (5.0, 20.0)


# Estrato LEGENDA OCLUSORA. O corpus tem legenda em ~47 % das amostras e ela
# NUNCA cobre a curva: medido em 120 amostras, a fracao da curva coberta pela
# caixa tem mediana 0,0000, p90 0,0000 e MAXIMO 0,0028 — zero amostras acima de
# 1 %. A culpa e' do matplotlib: `loc="best"` procura ativamente o espaco livre,
# e as outras cinco posicoes sao CANTOS, onde uma resposta ao degrau raramente
# passa. A legenda existe no corpus como DISTRATOR ADJACENTE, nunca como
# OCLUSOR.
#
# O DANO, medido numa imagem real (`caso_real_neg_super`, legenda em
# 'lower left' atravessando a faixa da ACOMODACAO): a rede segue a BORDA da
# caixa e cria um patamar falso, o que antecipa a acomodacao e o ajuste
# compensa com polo dominante mais lento e menos amortecimento. O par
# controlado — a MESMA imagem com a legenda movida — recupera tudo com erro
# <= 3 %, entao a causa esta isolada numa unica variavel.
#
# COMO — SEGUNDA VERSAO. A primeira ancorava o CENTRO da caixa no valor da
# curva num ponto qualquer do transitorio, e falhou no unico teste que
# importava: pareando 80 amostras com e sem a caixa, a penalidade ponta a ponta
# foi NULA nos dois modelos (79,1 % contra 78,3 % no promovido; 76,5 % contra
# 79,7 % no `ruido/epoca_17`). A caixa tapava a curva — 100 % das amostras
# acima de 1 % dos pixels — e mesmo assim nao produzia o defeito. Tapar nao
# basta: o corpus tinha OCLUSAO, mas nao ESSA oclusao.
#
# A geometria que importa, medida na figura real: a caixa vai de x 106 a 357
# (0,00 a 0,43 da largura da curva) e de y 333 a 405; o patamar assentado esta
# em y=394, ou seja a ONZE pixels da borda INFERIOR da caixa e paralelo a ela.
# A curva cai dentro da faixa vertical da caixa em so 28,7 % das colunas: ela
# entra pela esquerda por cima, DESCE atravessando a caixa e assenta raspando a
# borda de baixo. O que a caixa esconde e' a CHEGADA ao patamar, e o que a rede
# poe no lugar e' um segmento horizontal no nivel do patamar — acomodacao
# antecipada.
#
# Entao a ancora precisa de tres coisas, e nao de uma:
#   HORIZONTAL — sobre o JOELHO, que e' `theta`: onde a curva troca repouso
#     por transitorio.
#   VERTICAL   — no nivel ASSENTADO, nao no valor da curva no ponto. E' o que
#     poe a borda da caixa PARALELA ao patamar.
#   LADO       — a caixa se estende para o lado LIVRE (para baixo se a resposta
#     assenta em cima, para cima se assenta embaixo), como um plotador real
#     faria, deixando a curva logo para dentro da borda vizinha.
#
# O QUE A SEGUNDA VERSAO ERROU, e e' um ERRO DE UNIDADE. Ela punha o CENTRO da
# caixa em `theta + U(0,6; 3,0) * t_dom` — de 0,6 a 3 constantes de tempo
# DEPOIS do inicio, que e' o patamar assentado e nao o joelho. `t_dom` e' uma
# escala da DINAMICA; a caixa e' um objeto de LAYOUT, medido em fracao do eixo,
# e os dois nao se convertem: na figura real `t_dom` = 0,5 s numa janela de
# 10 s, entao a caixa (0,43 do eixo) tem quase NOVE `t_dom` de largura e
# qualquer deslocamento medido em `t_dom` some diante dela.
#
# MEDIDO nos 200 pares de `val_parleg_canto` x `val_parleg_ocl`:
#   o joelho cai em 0,14 da largura do eixo (p50; p10 0,07, p90 0,36);
#   a ancora MAIS PROXIMA possivel (u = 0,6) cai em 0,42 (p50);
#   em 12 % das amostras ate ela ja passa de 0,88 e era grampeada na borda,
#     para QUALQUER sorteio de `u`.
# Resultado: o joelho ficou DENTRO da caixa em 6 de 193 pares (3,1 %), e a
# 298 px dela (p50) nos outros. O estrato ensinou "caixa sobre uma reta
# assentada" — que a rede ja acertava: a deformacao da serie no joelho entre as
# duas metades do par deu p50 = 0,0 px e p90 entre 1,0 e 2,0 px nos sete
# checkpoints medidos, contra 6,8 px do modelo promovido na figura REAL. Sem
# dano no corpus nao ha o que aprender nem o que medir, e foi por isso que os
# quatro instrumentos sinteticos de oclusao nao ordenaram os checkpoints.
#
# ONDE O JOELHO CAI DENTRO DA CAIXA, medido no par real: a caixa vai de 0,00 a
# 0,43 do eixo, centro em 0,215, e o joelho em 0,35 — a +0,63 MEIA-LARGURA do
# centro, perto da borda DIREITA. A caixa cobre o tempo morto, o joelho e o
# comeco do transitorio, e a curva sai pela direita ainda em movimento.
#
# A CORRECAO: medir a caixa depois de desenhada e deslocar em MEIAS-LARGURAS
# dela. Com `u < 1` o joelho fica dentro da caixa POR CONSTRUCAO, qualquer que
# seja o rotulo, a fonte, o dpi ou a janela — o que a versao anterior nao
# garantia em nenhum desses casos.
# A TERCEIRA VERSAO, e o que a segunda ainda errava. Ancorar no JOELHO e' o
# alvo certo mas a REFERENCIA errada: medida na figura real pela serie extraida
# da metade de controle, ela tem repouso em y=72, patamar em y=317, e o
# transitorio inteiro cabe em x 248..281 — 33 px de uma curva de 577, ou 5,7 %
# da largura. A caixa tem 254 px, quase OITO vezes o transitorio: ela cobre o
# tempo morto (144 px), o transitorio todo e mais 77 px de patamar. Ancorar o
# CENTRO no joelho joga a caixa para tras, sobre o repouso, onde ela nao toca a
# curva — medido na correcao intermediaria: a fracao da curva tapada caiu de
# 0,051 (p50) para 0,000, o oposto do que se queria.
#
# A referencia que funciona nos DOIS regimes e' a CHEGADA ao patamar. No corpus
# o transitorio ocupa boa parte da janela (o joelho cai em 0,14 do eixo e
# `3*t_dom` atravessa quase tudo); na figura real ele ocupa 5,7 %. `theta` e
# `t_dom` nao normalizam entre os dois; a chegada sim, porque e' o ponto a
# partir do qual existe patamar para a borda da caixa imitar.
#
# E' ASSIM que a caixa real faz o estrago, e a leitura esta no proprio
# `_OCLUSAO_FOLGA`: a borda de CIMA dela corre 14 px abaixo do patamar e se
# estende 144 px PARA TRAS, cruzando o joelho. A rede segue essa borda
# horizontal como se fosse patamar e antecipa a acomodacao — `theta` sai tarde,
# o joelho em S some, e o ajuste vira FOPDT. Medido: `theta` >= 3,57 da `fopdt`,
# `theta` <= 3,48 da `second` com zeta 0,75-0,96, e a faixa inteira entre os
# checkpoints e' de 0,18 s, 1,8 % da janela.
_OCLUSAO_CHEGADA = 0.95         # fracao da excursao que define "chegou"
_OCLUSAO_JOELHO = (0.20, 0.80)  # a CHEGADA, em MEIAS-LARGURAS da caixa a
                                # direita do centro dela (o par real: 0,39)
_OCLUSAO_FOLGA = (0.01, 0.06)  # curva para dentro da borda, em fracao do eixo
                               # (o caso real: 11 px de ~464 = 0,024)

# QUANTOS textos do sorteio neutro sao concatenados no rotulo da legenda do
# estrato. A LARGURA da caixa e' a variavel causal, nao so a posicao: o defeito
# e' a rede seguir a BORDA HORIZONTAL da caixa como se fosse patamar, e uma
# borda longa e' um patamar mais convincente. Medido no par real
# `caso_real_neg_super`: a caixa que quebra o ajuste ocupa 43 % da largura da
# curva e tapa 48,1 % das colunas dela; com o rotulo curto que `sample_style`
# sorteia, a caixa do corpus ocupava 17 % (p50) e tapava 16,3 %. Concatenar de
# 1 a 3 textos cobre a faixa real. O rotulo continua semanticamente vazio, e
# continua vindo do MESMO gerador de texto do corpus base — o que muda e' so o
# comprimento.
_OCLUSAO_N_TEXTOS = (2, 5)   # `randint`: 2 a 4. Era (1, 4) — remedido
                             # com a ancora corrigida, a caixa ficava em
                             # 0,245 do eixo (p50) contra 0,43 da real, e
                             # a curva tapada em 0,072 contra 0,1505.


# --------------------------------------------------------------------------
# TERCEIRA FAMILIA DE RENDER (§73)
# --------------------------------------------------------------------------
# POR QUE ELA EXISTE. A rede treina no render deste arquivo e e' avaliada no do
# `rg_aleatorio.py`, e a lacuna entre as duas familias foi MEDIDA: a cobertura
# da mascara sobre a curva verdadeira tem mediana 0,713 na avaliacao contra
# 0,925 no treino (Mann-Whitney p = 1,05e-12; 79 % das figuras de avaliacao
# abaixo de 80 % de cobertura contra 29 % das de treino). Isso e' o dobro do
# custo de qualquer outro fator que se mediu neste estrato.
#
# O QUE A LACUNA NAO E'. Seis decomposicoes vieram NEGATIVAS, e elas estao aqui
# para ninguem refazer:
#   fisica        |K| nao prediz a falha (Mann-Whitney p = 0,68)
#   traco         espessura p = 0,30; dpi p = 0,17; o CONTRASTE ate vai na
#                 direcao contraria (0,594 nas ruins contra 0,570 nas boas)
#   estilo        pontilhado piora de verdade (par controlado, 23 pioram
#                 contra 1, p < 1e-4) mas explica so ~1/3 dos casos
#   posicao       a perda e' uniforme: 33 % no repouso, 33 % no transitorio,
#                 40 % na cauda; e nao ha efeito de moldura
#   elementos     desligar linha de entrada, preenchimentos E o tema de uma vez
#                 nao recupera nada (0,692 contra 0,711 do base)
#   geometria     largura, altura, dpi, aspecto e espessura relativa da
#                 avaliacao caem 100 % dentro de [p1, p99] do treino
#
# A HIPOTESE, E ELA NAO ESTA CONFIRMADA. O que separa as familias nao e' um
# parametro e sim a COMBINACAO: temas do matplotlib produzem conjuntos de
# pixels (grade, fonte, marca de tick, cor de eixo) que o corpus nao gera mesmo
# cobrindo cada parametro isolado. Este estrato testa isso gerando uma TERCEIRA
# familia — nem a deste arquivo, nem a do `rg_aleatorio`.
#
# NAO COPIAR O `rg_aleatorio`. Os temas aqui excluem de proposito os dois que
# ele usa (`seaborn-v0_8-darkgrid` e `dark_background`): treinar no render de
# avaliacao transformaria os lotes de controle em memorizacao de familia e
# invalidaria todo numero da Parte 2.
_TEMAS_ALT: tuple[str, ...] = (
    "ggplot", "bmh", "fivethirtyeight", "Solarize_Light2",
    "tableau-colorblind10", "petroff10", "grayscale", "classic",
)
_ALT_COMPANHEIRO = 0.6   # probabilidade de desenhar o SINAL COMPANHEIRO
_ALT_SOMBRA = 0.5        # probabilidade de sombrear o trecho de tempo morto
_ALT_ALPHA = (0.35, 1.00)        # opacidade da curva
_ALT_FATOR_TRACO = (0.45, 1.00)  # espessura, em fracao da sorteada por `sample_style`
_ALT_GRADE_POR_CIMA = 0.5
_ALT_PAINEL = 0.75              # probabilidade de pintar o PAINEL
_ALT_PAINEL_DELTA = (0.10, 0.30)  # afastamento de luminancia do fundo da figura
# O PAINEL e' a diferenca estrutural, e ela foi encontrada medindo APARENCIA e
# nao desempenho. Estatisticas de imagem das figuras que falham no lote contra
# as duas familias:
#     medida        familia 1   familia 3 (so tinta)   rg que FALHAM
#     tinta            0,021           0,024               0,380
#     entropia         0,288           0,443               1,239
# `_new_figure` pinta a figura E o painel com `style.bg_color` — a MESMA cor —,
# entao o corpus nunca produziu contraste entre painel e moldura. O
# `seaborn-darkgrid` do `rg_aleatorio` pinta o painel de cinza sobre figura
# branca, e sao esses ~40 % de pixels que separam as familias. Mexer em
# opacidade e espessura (a primeira tentativa) deixou a distancia ate o
# centroide das figuras que falham em 3,74 contra 3,93 da familia 1: andou 0,19
# de uma distancia de 1,4, ou seja, nada.
#
# O corpus JA tem fundo escuro em 35 % das amostras (luminancia < 0,3), entao
# fundo escuro nao era a lacuna — medido antes de mexer.
# TINTA POR COLUNA e' o que move a agulha, e isso foi varrido. So os temas
# deixavam a familia em cobertura 0,968 — indistinguivel da familia de treino
# (0,944) e longe da de avaliacao (0,713). Medido em 40 sistemas, um botao de
# cada vez:
#     so temas                     0,968
#     + grade por cima             0,967   <- ZERO. A hipotese de que linha de
#                                             grade cortando o traco apaga
#                                             tinta esta ERRADA.
#     + alpha 0,55                 0,958
#     + traco fino                 0,920
#     alpha + fino                 0,874
#     os tres, alpha 0,40          0,788
# As faixas acima SORTEIAM os dois em vez de fixar, para a familia ABRANGER de
# facil a dificil. O alvo nao e' o minimo: afundar abaixo de 0,713 seria pior,
# porque uma familia que a rede nao ve de jeito nenhum ensina a desistir, nao a
# segmentar. A grade por cima FICA, apesar de medir zero na cobertura, porque o
# objetivo declarado do estrato e' diversidade de render — mas o numero esta
# aqui para ninguem lhe atribuir ganho.
# O SINAL COMPANHEIRO e' um degrau em `steps-post` no mesmo quadro — a forma
# que o `polyline.py` documenta como o pior confusor ("a polilinha fica AGARRADA
# na entrada", Fisher OR = 0,248, p = 0,0012). O corpus base tem distratores,
# mas nenhum com essa forma.


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

    """
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

    """
    if spec.a_zero:
        # Antes da superposicao de proposito: o zero e' da PLANTA, entao ele
        # multiplica cada parcela igualmente e `y0 - a*y0\'` da o mesmo
        # resultado somando antes ou depois. Fazer aqui evita recursao dupla.
        base = replace(spec, a_zero=None)
        return (step_response(base, t)
                - float(spec.a_zero) * _derivada_degrau(base, t))
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

def _plot_curve(ax, t: np.ndarray, y: np.ndarray, style: RenderStyle, color: str,
                label=None, alt: dict | None = None):
    """`alt` so' e' passado pela figura de VERDADE do estrato da terceira
    familia (§73). A figura-MASCARA chama sem ele, de proposito: alpha e
    marcador mudam a aparencia e nao podem mudar a verdade."""
    kwargs = dict(
        color=color,
        linewidth=style.line_width,
        linestyle=style.line_style,
        solid_capstyle="round",
        antialiased=True,
    )
    if alt:
        # ALPHA e ESPESSURA sao os dois eixos que mexem em quanta TINTA a curva
        # deposita por coluna — que e' a grandeza que o instrumento de
        # cobertura mede. Os temas sozinhos nao mexiam neles, e por isso a
        # primeira versao da familia ficou a 4 pontos da familia de treino
        # (0,903 contra 0,944, p = 0,23) em vez de se aproximar dos 0,713 da
        # familia de avaliacao.
        if alt.get("alpha") is not None:
            kwargs["alpha"] = float(alt["alpha"])
        if alt.get("fator_traco") is not None:
            kwargs["linewidth"] = max(style.line_width * float(alt["fator_traco"]), 0.35)
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
    legenda_oclusora: bool = False,
    legenda_no_canto: bool = False,
    alt: dict | None = None,
) -> dict:
    """Escreve image.png, mask.png e meta.json em out_dir. Devolve o dict do meta.

    `alt` traz as escolhas da TERCEIRA FAMILIA DE RENDER (§73) ja sorteadas por
    `generate_sample`: `{"tema", "companheiro", "sombra"}`. Vem pronto de fora
    porque o sorteio precisa de `rng_style`, que so existe la — o mesmo molde do
    `ruido_alto`, e o que garante que `rng_noise` nao se desloca.
    """
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
    if legenda_oclusora or legenda_no_canto:
        # OS DOIS estratos FORCAM a legenda. `sample_style` so a sorteia em
        # metade das amostras; sem forcar, metade do corpus novo sairia byte a
        # byte igual ao base — nao seria material novo, seria o base com peso
        # dobrado. Mesmo molde do `ruido_alto`, que sobrescreve `snr_db` depois
        # de `sample_style`: quem decide e uma flag EXTERNA, entao o estilo
        # continua sem ver o spec (anti-vazamento estrutural).
        #
        # `legenda_no_canto` e' a METADE DE CONTROLE do par (§74): mesma
        # amostra, mesma legenda, mesmo rotulo, so a POSICAO muda. Sem ela o
        # par contrastaria "tem legenda" contra "nao tem", que e' outra
        # variavel — e o defeito medido e' de POSICAO, nao de existencia
        # (`caso_real_neg_super` contra `_legenda_movida`: 32 % contra 2,6 %).
        style = replace(style, has_legend=True)

    t = np.linspace(spec.t_start, spec.t_end, N_SERIES)
    y_clean = step_response(spec, t)
    y_draw = _apply_noise(y_clean, style, rng) if add_noise else y_clean.copy()

    xlim, ylim = _axis_limits(t, y_draw, style)
    if plato_no_meio:
        # Depois de `_axis_limits` e antes de qualquer plotagem: a figura da
        # imagem e a da mascara usam os MESMOS limites (linhas ~413 e ~601),
        # entao a verdade fica consistente.
        ylim = _ylim_plato_no_meio(y_draw, ylim, rng)

    # ---------------- figura de verdade ----------------
    # TEMA DA TERCEIRA FAMILIA. `rcParams` e' GLOBAL e por processo; o
    # `rg_aleatorio._estilo` documenta o estrago de deixar vazar entre figuras
    # (29 de 33 amostras sem moldura porque um estilo zerou `axes.linewidth` e o
    # seguinte nao restaurou). Aqui o estado e' salvo e restaurado no `finally`,
    # e a FIGURA-MASCARA fica FORA do bloco: a geometria dela tem de continuar
    # identica a da imagem, e ela ja e' explicita (`figsize`, `dpi`,
    # `axes_rect`, `xlim`, `ylim`), entao nada do tema pode alcanca-la.
    _rc_salvo = None
    if alt:
        _rc_salvo = matplotlib.rcParams.copy()
        try:
            matplotlib.style.use(alt["tema"])
        except Exception:  # pragma: no cover - tema ausente na versao instalada
            matplotlib.rcParams.update(_rc_salvo)
            _rc_salvo = None
    fig, ax = _new_figure(style, style.bg_color)
    if alt and alt.get("painel"):
        # So o PAINEL (retangulo dos eixos). A figura continua com `bg_color`,
        # e e' o contraste entre os dois que o corpus nunca teve. A
        # figura-MASCARA nao passa por aqui: ela usa `_new_figure(style,
        # "#000000")` e `set_axis_off()`, entao a verdade nao se mexe.
        ax.set_facecolor(alt["painel"])
    _plot_curve(ax, t, y_draw, style, style.line_color,
                label=style.legend_text if style.has_legend else None, alt=alt)
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
        # GRADE POR CIMA (§73). O corpus sempre desenhou a grade ATRAS da curva
        # (`zorder=0`); o tema `seaborn-darkgrid` da familia de avaliacao a
        # desenha clara sobre fundo cinza, cruzando o traco. Linha de grade
        # cortando o traco e' exatamente o que apaga tinta na coluna.
        por_cima = bool(alt and alt.get("grade_por_cima"))
        ax.set_axisbelow(not por_cima)
        ax.grid(True, color=style.axes_color, alpha=style.grid_alpha,
                linestyle=style.grid_style, linewidth=0.6,
                zorder=3 if por_cima else 0)
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
    if alt and alt.get("companheiro"):
        # SINAL COMPANHEIRO: um degrau em `steps-post` no mesmo quadro. E' a
        # forma que o `polyline.py` mede como o pior confusor da extracao, e o
        # corpus base nao tem nenhum distrator com ela.
        alvo = float(spec.K * spec.step_amplitude)
        u = np.where(t >= float(spec.theta), alvo, 0.0)
        ax.plot(t, u, drawstyle="steps-post", color=style.axes_color,
                linewidth=max(style.line_width * 0.7, 0.8), linestyle="--",
                alpha=0.85, zorder=2,
                label="referencia" if style.has_legend else None)
    if alt and alt.get("sombra") and float(spec.theta) > float(spec.t_start):
        ax.axvspan(float(spec.t_start), float(spec.theta),
                   color=style.axes_color, alpha=0.12, zorder=0)
    if style.has_legend:
        if legenda_oclusora:
            # CORPO SOBRE O JOELHO, BORDA NO NIVEL DO PATAMAR, PELO LADO
            # LIVRE — a caixa e' ancorada na CHEGADA e se estende para tras.
            # Ver o bloco de `_OCLUSAO_CHEGADA` para a medicao da figura real
            # que fixa as tres escolhas, e para os dois erros das versoes
            # anteriores (unidade, e referencia).
            r = rng if rng is not None else np.random.default_rng(0)

            # Nivel ASSENTADO como a figura o DESENHA: mediana do ultimo quinto
            # da serie. Nao `K * degrau`, que e' o setpoint comandado e nao o
            # que aparece quando a janela nao assenta; nao `y_draw[-1]`, que e'
            # uma amostra ruidosa so.
            y_fim = float(np.median(y_clean[max(int(0.8 * y_clean.size), 1):]))
            fy = (y_fim - ylim[0]) / max(ylim[1] - ylim[0], 1e-12)
            folga = float(r.uniform(*_OCLUSAO_FOLGA))

            # Lado LIVRE: se a resposta assenta ACIMA do repouso, o espaco esta
            # embaixo e a caixa desce (ancora no topo dela); se assenta abaixo,
            # a caixa sobe. Nos dois casos a ancora fica deslocada de `folga`
            # para o lado de fora, de modo que o patamar caia logo para DENTRO
            # da borda vizinha — os 11 px do caso real.
            sobe = float(spec.K) * float(spec.step_amplitude) >= 0.0
            loc = "upper center" if sobe else "lower center"
            fy_anc = float(np.clip(fy + folga if sobe else fy - folga,
                                   -0.05, 1.05))

            # A CHEGADA ao patamar, em fracao do eixo: o primeiro instante
            # em que a curva LIMPA alcanca `_OCLUSAO_CHEGADA` da excursao.
            # Medida na serie desenhada, nao em `theta + k*t_dom`, que nao
            # normaliza entre um transitorio de 5 % da janela e um de 60 %.
            y_rep = float(np.median(y_clean[:max(int(0.02 * y_clean.size), 3)]))
            sentido = float(np.sign(y_fim - y_rep)) or 1.0
            passou = np.nonzero((y_clean - y_rep) * sentido
                                >= _OCLUSAO_CHEGADA * abs(y_fim - y_rep))[0]
            t_arr = float(t[passou[0]]) if passou.size else float(t[-1])
            fx_arr = (t_arr - xlim[0]) / max(xlim[1] - xlim[0], 1e-12)

            # DESENHA PARA MEDIR, e so entao posiciona. A largura da caixa
            # depende do rotulo, da fonte e do dpi, e so existe depois do
            # layout — e e' ela que da sentido a "deslocar". Medir custa um
            # `draw()` por figura, e so neste estrato: o corpus base nao passa
            # por aqui.
            leg = ax.legend(loc=loc, bbox_to_anchor=(fx_arr, fy_anc),
                            bbox_transform=ax.transAxes,
                            fontsize=style.font_size * 0.8,
                            framealpha=style.legend_alpha)
            rend = fig.canvas.get_renderer()
            meia = 0.5 * float(leg.get_window_extent(rend).transformed(
                ax.transAxes.inverted()).width)

            # A chegada fica `u` MEIAS-LARGURAS a direita do centro, entao
            # o centro recua `u * meia` e o corpo da caixa se estende PARA TRAS
            # sobre o transitorio e o tempo morto — a borda longa da figura
            # real. Com `u < 1` a chegada cai dentro da caixa por construcao,
            # qualquer que sejam rotulo, fonte, dpi ou janela. O grampo e' a
            # caixa nao sair do quadro, agora exato porque a meia-largura e'
            # conhecida (`_OCLUSAO_FX_LIM`, que chutava 0,12 e 0,88, deixou de
            # existir); quando a caixa e' mais larga que o eixo o grampo
            # degenera no centro, que e' o unico lugar possivel.
            u = float(r.uniform(*_OCLUSAO_JOELHO))
            lo, hi = min(meia, 0.5), max(1.0 - meia, 0.5)
            fx = float(np.clip(fx_arr - u * meia, lo, hi))
            leg.set_bbox_to_anchor((fx, fy_anc), transform=ax.transAxes)
        else:
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
    if _rc_salvo is not None:
        matplotlib.rcParams.update(_rc_salvo)
        _rc_salvo = None

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
    if legenda_oclusora:
        # CONDICIONAL pela mesma razao de `fora_da_familia` abaixo: os outros
        # tres flags de render (`has_reference_line`, `has_annotation_arrow`,
        # `has_settling_band`) ja saem SEMPRE em `render`, entao acrescentar
        # mais um ali mudaria os bytes de todo meta.json do corpus base so
        # para dizer `false`. Sem esta chave o estrato seria INVISIVEL no
        # meta: `render.has_legend` vira `true`, mas isso tambem acontece em
        # metade do corpus base, e nada distinguiria uma legenda no canto de
        # uma por cima da curva.
        meta["render"]["legenda_oclusora"] = True
        meta["render"]["par_legenda"] = "oclusora"
    if legenda_no_canto:
        meta["render"]["par_legenda"] = "canto"
    if alt:
        # CONDICIONAL, como as demais: o corpus base nao ganha chave nenhuma.
        # Guarda o NOME DO TEMA, nao um booleano, porque e' a chave que permite
        # separar as familias depois sem reidentificar amostra por amostra —
        # que e' o requisito de conseguir voltar atras.
        meta["render"]["familia_alt"] = str(alt["tema"])
        meta["render"]["companheiro"] = bool(alt.get("companheiro"))
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
                    plato_no_meio: bool = False,
                    fase_nao_minima: bool = False,
                    ruido_alto: bool = False,
                    legenda_oclusora: bool = False,
                    legenda_no_canto: bool = False,
                    familia_alt: bool = False) -> dict:
    """Sorteia sistema+estilo com streams independentes, renderiza e devolve o meta."""
    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(3)
    rng_sys = np.random.default_rng(children[0])
    rng_style = np.random.default_rng(children[1])
    rng_noise = np.random.default_rng(children[2])

    spec = sample_system(rng_sys)
    style = sample_style(rng_style)  # nao ve o spec: anti-vazamento estrutural
    if ruido_alto:
        # Estrato opt-in, molde do `ganho_negativo` (§40.5). Sorteado de
        # `rng_style` DEPOIS de `sample_style` — que nao consome mais nada
        # dali —, entao o corpus base fica byte a byte identico e `rng_noise`
        # nao se desloca: a REALIZACAO do ruido para um dado sigma continua a
        # mesma sequencia.
        #
        # E o unico estrato que mexe no ESTILO e nao no SPEC, e isso nao fere o
        # anti-vazamento: a regra e que o estilo nao pode ver o SISTEMA, e aqui
        # quem decide e uma flag externa. O `snr_db` continua sem saber nada de
        # `K`, `tau` ou ordem.
        style = replace(style, snr_db=float(rng_style.uniform(*_SNR_BAIXO)))
    if legenda_oclusora or legenda_no_canto:
        # Alargamento do ROTULO — vive aqui, e nao em `render_sample`, porque
        # precisa de `rng_style`, que so existe neste escopo. Consumido DEPOIS
        # de `sample_style` e de `ruido_alto`, pelo mesmo motivo deles: o
        # corpus base nao se desloca um byte e `rng_noise` nao se move, entao a
        # realizacao do ruido continua a mesma sequencia.
        #
        # O `has_legend=True` fica em `render_sample` de proposito: e' garantia
        # de render (o flag sem legenda nao desenharia nada e o estrato sairia
        # igual ao base), e vale mesmo para quem chama `render_sample` direto.
        n = int(rng_style.integers(*_OCLUSAO_N_TEXTOS))
        style = replace(style, legend_text=" ".join(
            _sample_text(rng_style) for _ in range(n)))
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
    if fase_nao_minima:
        # Estrato OOD opt-in, molde do `ganho_negativo` (§40.5) — e o unico que
        # sai FORA da familia de modelos do Estagio D. Sorteado com `rng_sys`
        # (nunca `rng_style`) e APLICADO POR ULTIMO,
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
    alt = None
    if familia_alt:
        # Sorteado de `rng_style` e por ULTIMO, pelo mesmo motivo dos outros
        # estratos: o corpus base fica byte a byte identico e `rng_noise` nao
        # se desloca, entao a realizacao do ruido continua a mesma sequencia.
        # Nao ve o `spec` — o tema do grafico nao pode depender da planta, ou a
        # rede aprenderia a ler a fisica do estilo (anti-vazamento estrutural).
        alt = {
            "tema": _TEMAS_ALT[int(rng_style.integers(0, len(_TEMAS_ALT)))],
            "companheiro": bool(rng_style.random() < _ALT_COMPANHEIRO),
            "sombra": bool(rng_style.random() < _ALT_SOMBRA),
            "alpha": float(rng_style.uniform(*_ALT_ALPHA)),
            "fator_traco": float(rng_style.uniform(*_ALT_FATOR_TRACO)),
            "grade_por_cima": bool(rng_style.random() < _ALT_GRADE_POR_CIMA),
        }
        if rng_style.random() < _ALT_PAINEL:
            # Luminancia do painel afastada da do fundo, para QUALQUER lado que
            # caiba em [0, 1]. Cinza puro de proposito: a cor do painel nao pode
            # virar pista da fisica (anti-vazamento), e um cinza nao compete com
            # a cor da curva, que `_sample_palette` ja escolheu com contraste
            # garantido contra `bg_color`.
            lb = luminance(style.bg_color)
            d = float(rng_style.uniform(*_ALT_PAINEL_DELTA))
            alvo = lb - d if lb > 0.5 else lb + d
            v = int(round(float(np.clip(alvo, 0.04, 0.96)) * 255))
            alt["painel"] = f"#{v:02x}{v:02x}{v:02x}"
    return render_sample(spec, style, out_dir, add_noise=add_noise, rng=rng_noise,
                         seed=int(seed), reta_no_patamar=reta_no_patamar,
                         anotacao_com_seta=anotacao_com_seta,
                         banda_de_acomodacao=banda_de_acomodacao,
                         plato_no_meio=plato_no_meio,
                         legenda_oclusora=legenda_oclusora,
                         legenda_no_canto=legenda_no_canto, alt=alt)


def _generate_one(args: tuple) -> str:
    (out_dir, seed, add_noise, reta, janela, seta, banda, kneg, plato,
     nmp, ruido, legenda, canto, alt) = args
    generate_sample(out_dir, seed, add_noise=add_noise, reta_no_patamar=reta,
                    janela_assentada=janela, anotacao_com_seta=seta,
                    banda_de_acomodacao=banda, ganho_negativo=kneg,
                    plato_no_meio=plato,
                    fase_nao_minima=nmp, ruido_alto=ruido,
                    legenda_oclusora=legenda, legenda_no_canto=canto,
                    familia_alt=alt)
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
    plato_no_meio: bool = False,
    fase_nao_minima: bool = False,
    ruido_alto: bool = False,
    legenda_oclusora: bool = False,
    legenda_no_canto: bool = False,
    familia_alt: bool = False,
) -> list[str]:
    """Gera n amostras em paralelo. Resultado independe do numero de workers."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(root / f"sample_{i:05d}"), int(seed) * 1_000_003 + i, bool(add_noise),
         bool(reta_no_patamar), bool(janela_assentada),
         bool(anotacao_com_seta), bool(banda_de_acomodacao),
         bool(ganho_negativo), bool(plato_no_meio),
         bool(fase_nao_minima), bool(ruido_alto), bool(legenda_oclusora),
         bool(legenda_no_canto), bool(familia_alt))
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
