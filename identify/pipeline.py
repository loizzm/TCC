"""Cola dos estágios A, B e D. Única porta de entrada para a Parte 3."""
from __future__ import annotations

import time

import numpy as np

from identify.calibrate import calibrate
from identify.classical import identify
from identify.extract import predict_mask
from identify.polyline import mask_to_polyline, polyline_to_series

# Nº de colunas iniciais usadas para estimar o nível de repouso.
# Antes era uma FRAÇÃO (8 %) da largura, o que supõe prefixo plano por tempo
# morto. Com θ = 0 a curva já subiu ~28 % dentro dessa janela, o patamar sai
# 3,8 % baixo e isso vira 12,6 % de erro em ζ, porque ζ vem da razão de
# overshoot (HANDOFF_P2_7 §34.3). Uma janela pequena e FIXA é correta nos dois
# regimes: com θ grande o prefixo é plano e 5 colunas caem nele; com θ = 0 as 5
# primeiras colunas ainda estão praticamente no repouso. A mediana (e não o
# primeiro valor) dá robustez a um pixel espúrio. Escolhido por varredura
# medida no caso real e no sintético (reports/part2_repouso_varredura.md):
# no subconjunto sem calibração e de 2ª ordem das 300 primeiras de `data/test`
# (n=33 — a população que este estimador serve), o MAPE de ζ cai de 2,92 %
# (fração 8 %, valor antigo) para 1,34 %; no caso real (n=1) o erro de ζ fica
# em 1,1 %.
_N_REPOUSO = 5


def _nivel_de_repouso(y: np.ndarray) -> float:
    """Nível de repouso em pixels, a partir do início da série ordenada por x."""
    n = int(min(_N_REPOUSO, y.size))
    return float(np.median(y[:max(1, n)]))


# Cobertura mínima (extensão observada da polilinha / largura da moldura)
# abaixo da qual `_serie_normalizada` troca a extensão observada pela largura
# da moldura. Todos os números abaixo foram medidos na fix round 2 e vêm com a
# população explicitada (ver reports/part2_repouso_varredura.md).
#
# POPULAÇÃO EM QUE ISTO RODA: `_serie_normalizada` só é chamada quando a
# calibração FALHA. Em `data/test` (900 amostras) são 184 sem calibração, 179
# delas com polilinha ≥ 10 pontos e moldura válida — TODAS as ordens, não só a
# 2ª. Distribuição da cobertura nessas 179: min 0,7713 (`sample_00639`,
# fopdt), p1 0,8512, p5 0,9009, mediana 0,9332, p95 0,9667, max 0,9873;
# 1ª ordem (n=104) min 0,7713, 2ª ordem (n=75) min 0,8937. Abaixo de 0,75: 0
# amostras — o ramo da moldura NUNCA dispara no sintético. A folga do limiar
# até o mínimo observado é de 2,1 p.p. (0,7713 − 0,75), não os ~13 p.p. que a
# fix round 1 documentou olhando só as 75 de 2ª ordem.
#
# No caso real (tests/fixtures/caso_real_2ordem.png, n=1) a cobertura é 0,617:
# a curva assentada se sobrepõe à reta de referência tracejada e a máscara não
# separa as duas ali, ~38 % da largura sem tinta detectada.
#
# POR QUE CONDICIONAR. A moldura NÃO é a janela real do gráfico: ela inclui a
# margem que o matplotlib acrescenta nos dois lados (`x_margin_lo`/
# `x_margin_hi` de `dataset/randomize.py`, U(0,01, 0,06) cada). Medida nas
# mesmas 179: a moldura é de +2,2 % a +11,7 % mais larga que a janela de dados
# (mediana +6,9 %). É uma aproximação POR EXCESSO, e usá-la sempre injeta esse
# viés em toda amostra não truncada. Medido com `_COBERTURA_MIN_MOLDURA = 1.01`
# (moldura sempre) contra o valor atual, no diagnóstico `2.6-adim[wn_T]` de
# tests/part2/test_part2.py (as 300 primeiras de `data/test`):
#   corpus inteiro (n=143):        MAPE de ωₙ·T real 1,76 % -> 2,54 %
#   só sem calibração (n=33):      MAPE de ωₙ·T real 1,35 % -> 7,15 %
# e, nas 184 sem calibração das 900: MAPE de τ/T 0,87 % -> 5,84 % (n=75
# aceitas) e |Δθ/T| mediano 0,0028 -> 0,0197 (n=151). O n=33 é a população que
# esta constante de fato governa; o n=143 está diluído pelas 110 amostras que
# passam pelo caminho físico e nem chegam aqui.
#
# POR QUE 0,75 E NÃO O PONTO DE EMPATE. Com a moldura valendo (1+m)·T (m = soma
# das duas margens) e a cobertura valendo c, o erro de ESCALA de `t` é
# |c(1+m) − 1| usando a extensão observada e m usando a moldura; os dois
# empatam em c* = (1−m)/(1+m). Como m varia por amostra, c* também varia: nas
# 179, c* vai de 0,7912 a 0,9563, mediana 0,8716 — não é uma constante. Subir o
# limiar para essa mediana (0,872) foi MEDIDO na fix round 2: passa a trocar de
# referência em 3 das 179 (0,7713, 0,8505 e 0,8514, todas fopdt) e deixa
# `2.6-adim[zeta]` e `2.6-adim[wn_T]` idênticos dígito a dígito, porque as três
# são de 1ª ordem. Nas grandezas que elas movem o saldo é misto: MAPE de τ/T
# p95 melhora (10,37 % -> 6,42 %) e |Δθ/T| p95 piora (0,0164 -> 0,0181),
# medianas e acerto de ordem (85,9 %) inalterados. A razão é que o empate acima
# só considera a ESCALA: trocar para a moldura move junto a ORIGEM de `t` para
# `bbox_px[0]`, e quando a truncagem é à DIREITA (o regime do caso real) a
# origem observada já estava certa — a troca importa a margem esquerda como
# viés aditivo em θ (medido em `sample_00639`: θ/T de 0,0333 para 0,0484, alvo
# 0,0283). Sem ganho líquido medido, fica 0,75: nenhuma amostra do sintético
# cruza, o comportamento é bit a bit o anterior e só o caso real (0,617) usa a
# moldura.
_COBERTURA_MIN_MOLDURA = 0.75

# Dispersão máxima de `y` nas `_N_REPOUSO` primeiras colunas observadas, como
# fração da faixa total de `y`, para que a troca de referência de tempo pela
# moldura seja aplicável.
#
# POR QUE ESTA CONSTANTE EXISTE. `_COBERTURA_MIN_MOLDURA` olha a cobertura
# TOTAL (`span_observado / span_moldura`) e por isso não distingue de que lado
# falta tinta — mas as duas metades desta correção supõem coisas diferentes:
# `_nivel_de_repouso` lê as 5 PRIMEIRAS colunas observadas e supõe a curva
# PARADA ali. Numa truncagem à ESQUERDA (a U-Net perdendo o trecho plano
# inicial — plausível: um trecho horizontal colado no eixo ou numa linha de
# grade é onde a rede perde tinta, o mesmo modo de falha do patamar no caso
# real) essas colunas já estão NA SUBIDA, e o patamar sai errado: é o defeito
# de 12,6 % em ζ que `_N_REPOUSO` acabou de corrigir, reintroduzido pelo outro
# remédio da mesma task. A docstring dizia "quando a truncagem é à DIREITA (o
# regime do caso real)" e o código não verificava — aqui a suposição vira
# invariante asseverada (revisão final, achado C3).
#
# POR QUE A PLANURA, E NÃO A GEOMETRIA DO LADO. A primeira versão desta guarda
# usava um PROXY: `(x[0] − bbox_px[0]) / largura da moldura ≤ 0,15`, calibrado
# sobre o deslocamento MÁXIMO de margem do matplotlib (0,1227, n=299). O proxy
# foi RETIRADO porque o limiar estava no lugar errado da curva de dano — ver
# HANDOFF_P2_7 §35.9.1. O erro de método vale mais que o número: a população
# que calibrou o 0,15 tinha cobertura mínima 0,8388, ou seja NENHUMA das 299
# amostras entrava neste ramo. O limiar protegia contra um falso positivo
# impossível naquela população enquanto admitia 68 % de erro em ζ logo antes
# dele. A planura não é proxy: é literalmente a condição que
# `_nivel_de_repouso` precisa, é independente da margem, e é medível nas duas
# populações porque a truncagem à direita não mexe nas primeiras colunas.
#
# POR QUE 0,03 — OS DOIS LADOS, MEDIDOS.
# (a) Teto de dano. 82 séries de 2ª ordem determinísticas (ζ ∈ {0,3; 0,5; 0,7}
#     × ωₙ ∈ {1; 2; 4} rad/s × corte à esquerda de 0 a 28 % em passos de 2 %,
#     corte à direita fixo em 35 %, rasterizadas na moldura do caso real).
#     Aceitas com planura ≤ 0,03 (n=21): erro de ζ mediano 0,49 %, p90 6,43 %.
#     Recusadas, planura > 0,03 (n=61): mediano 28,77 %, máximo 99,86 %.
#     A varredura fina em ζ=0,5 / ωₙ=2 é monótona e mostra a curva:
#     planura 0,0064 -> 0,08 %; 0,0423 -> 4,25 %; 0,0613 -> 13,09 %;
#     0,0860 -> 27,34 %; 0,1171 -> 45,44 %; 0,1587 -> 67,82 %.
# (b) Piso de legitimidade. A planura não é zero nem sem truncagem nenhuma:
#     com θ = 0 e ωₙ alto a curva já se move um pouco dentro de 5 colunas.
#     Medida nas 300 primeiras de `data/test` com a cadeia de produção
#     (n=299): mediana 0,0044, p95 0,0316, p99 0,0485, máximo 0,0722; no
#     subconjunto sem calibração (n=60) máximo 0,0483. 0,03 fica em ~p94 do
#     corpus, então NÃO recusa uma série pela movimentação normal de início de
#     curva que o próprio corpus exibe. Caso real (n=1): planura 0,0039 — ~8×
#     de folga, e é a única amostra que de fato usa este ramo.
# Diferente do 0,15, este piso é transferível: a truncagem à DIREITA não altera
# as primeiras colunas, então a distribuição medida no corpus é a mesma que
# vale dentro do ramo.
#
# PONTO CEGO CONHECIDO, medido e registrado (§35.7-9): um corte à esquerda tão
# profundo que a série remanescente RECOMEÇA plana — já no patamar assentado —
# tem planura ~0 e passa. Nas 82 séries isso ocorre 3 vezes, todas com
# cobertura ≤ 0,45 (o caso real é 0,617), com erro de ζ de 0,49 %, 6,43 % e
# 20,33 %. Testei um segundo facet do mesmo invariante (exigir que o nível
# lido seja o EXTREMO de `y`, isto é `desvio` de um sinal só): NÃO fecha o
# buraco — nesses três casos o desvio já é de um sinal só (contra-sinal
# −0,0000). Fica aberto, com o número escrito.
#
# POR QUE RECUSAR, E NÃO CAIR NO COMPORTAMENTO ANTERIOR. Medido em 27 séries
# de 2ª ordem determinísticas (mesmos ζ e ωₙ, corte de 30/40/50 % da janela
# pela esquerda): com o repouso viciado, ajustar mesmo assim dá MAPE de ζ
# mediano de 27,6 % (máx. 80,0 %) usando a moldura e 14,2 % (máx. 51,1 %)
# usando a extensão observada — e o ajuste sequer converge em 19 e 21 dos 27
# casos. Qualquer das duas saídas é pior que o defeito de 12,6 % que a task
# corrigiu e muito além da tolerância de 5 % do caso real. Um número errado que
# ninguém distingue de um certo é pior que nenhum, e o contrato do
# `dimensionless` já sabe sair vazio (`_vazio_adimensional`).
_PLANURA_MAX_FRAC = 0.03


def _serie_normalizada(x_px: np.ndarray, y_px: np.ndarray, bbox_px=None):
    """Polilinha em pixels -> série adimensional, sem depender de calibração.

    `t` em [0, 1] (a janela observada vira a unidade de tempo, ou a moldura —
    ver abaixo) e `y` com o zero no nível de repouso e o sinal invertido,
    porque o pixel cresce para baixo. A escala de `y` é arbitrária: ζ é
    invariante a ela, e `K` sai já dividido pela faixa (é o `K_yrange` do
    PLANO §1.7).

    `bbox_px`: moldura do gráfico `(x0, y0, x1, y1)` detectada por
    `detect_plot_bbox`/`calibrate`, opcional. Achado durante a medição desta
    task (não faz parte do defeito do nível de repouso, mas do MESMO teste):
    quando a polilinha não alcança a borda direita da moldura — no caso real
    isso acontece porque o trecho assentado da curva se sobrepõe à reta de
    referência tracejada e a máscara não separa as duas ali —, normalizar `t`
    pela extensão OBSERVADA da polilinha (`x[-1] - x[0]`) comprime o tempo: a
    janela vira mais curta que a janela real, e ωₙ (que depende de ONDE no
    tempo o pico ocorre) sai inflado na mesma proporção — medido 37,7 % de
    erro no caso real, mesmo com o nível de repouso já corrigido.

    A moldura NÃO é a janela real do gráfico (ela inclui a margem do
    matplotlib nos dois lados; medido nas 179 amostras sem calibração de
    `data/test`: de +2,2 % a +11,7 % mais larga que a janela de dados, mediana
    +6,9 %) — é só uma aproximação POR EXCESSO, preferível à extensão
    observada exclusivamente quando a polilinha está TRUNCADA. Por isso a
    troca só acontece quando a cobertura (extensão observada / largura da
    moldura) cai abaixo de `_COBERTURA_MIN_MOLDURA`; fora disso, usar a
    moldura só importaria o viés da margem — medido, no subconjunto que esta
    função de fato serve (33 amostras sem calibração das 300 primeiras de
    `data/test`), como piora de MAPE de ωₙ·T de 1,35 % para 7,15 %; no corpus
    inteiro daquele diagnóstico (n=143, diluído pelas 110 que passam pelo
    caminho físico) a mesma piora aparece como 1,76 % para 2,54 %.

    Quando a moldura entra, a ORIGEM de `t` também precisa vir da moldura
    (`bbox_px[0]`), não do primeiro ponto da polilinha: os dois têm que vir
    do mesmo referencial, senão o deslocamento entre eles vaza como viés
    aditivo em θ. Medido nas 179 amostras sem calibração de `data/test` (todas
    as ordens, fix round 2): o deslocamento `(x[0] − bbox_px[0]) / largura da
    moldura` tem mediana 3,1 %, p95 5,5 % e máximo 11,4 % — é isso que vazaria
    em θ/T se a origem ficasse presa em `x[0]` com a escala já trocada para a
    moldura (fix round 1, B1). Corrigir isso não custou nada nas outras
    métricas.

    A troca é condicionada ao INVARIANTE DIRETO de que o começo da polilinha
    ainda é o repouso — não a um proxy geométrico de qual lado foi truncado
    (revisão final C3, e a reversão do §35.9.1). O que o código verifica, e
    não apenas espera:

    - cobertura < `_COBERTURA_MIN_MOLDURA` **e** as `_N_REPOUSO` primeiras
      colunas observadas PLANAS (dispersão de `y` ali ≤ `_PLANURA_MAX_FRAC`
      da faixa total de `y`): o nível de repouso é legível, então troca escala
      e origem pela moldura. É o regime do caso real (planura 0,0039).
    - cobertura < `_COBERTURA_MIN_MOLDURA` **e** as primeiras colunas NÃO
      planas: a polilinha começa já na subida — truncagem à esquerda ou
      simétrica —, o nível de repouso é inválido, e a função RECUSA a série
      devolvendo `(None, None)`; o chamador emite o bloco `dimensionless`
      vazio. Ver `_PLANURA_MAX_FRAC` para a curva de dano dos dois lados.
    - cobertura ≥ `_COBERTURA_MIN_MOLDURA`: extensão e origem observadas, como
      antes.

    Sem `bbox_px` (ou moldura inválida), cai no comportamento anterior —
    extensão e origem observadas.

    Devolve `(None, None)` quando não há o que normalizar OU quando o que há
    não é confiável (o caso da truncagem à esquerda, acima).
    """
    if x_px.size < 10:
        return None, None
    k = np.argsort(x_px)
    x = np.asarray(x_px, dtype=float)[k]
    y = np.asarray(y_px, dtype=float)[k]
    span_observado = float(x[-1] - x[0])
    if not np.isfinite(span_observado) or span_observado <= 0:
        return None, None
    origem, span = x[0], span_observado
    if bbox_px is not None and any(bbox_px):
        span_moldura = float(bbox_px[2] - bbox_px[0])
        if np.isfinite(span_moldura) and span_moldura > 0:
            cobertura = span_observado / span_moldura
            if cobertura < _COBERTURA_MIN_MOLDURA:
                # A cobertura total não diz se ainda existe patamar inicial
                # para `_nivel_de_repouso` ler. Quem diz é a PLANURA das
                # primeiras colunas — a condição direta, e não um proxy de
                # qual lado perdeu tinta. Só aqui: com truncagem forte, a
                # suposição de repouso nas 5 primeiras colunas é o que corre
                # risco. Sem truncagem o corpus mostra planura mediana 0,0044,
                # e aplicar a guarda sempre recusaria ~6 % das amostras sem
                # ganho medido.
                n_rep = int(min(_N_REPOUSO, y.size))
                faixa_y = float(np.ptp(y))
                planura = (float(np.ptp(y[:max(1, n_rep)])) / faixa_y
                           if faixa_y > 0 else 0.0)
                if not np.isfinite(planura) or planura > _PLANURA_MAX_FRAC:
                    return None, None
            # A ÂNCORA É SEMPRE A MOLDURA quando ela existe.
            #
            # Antes, a moldura só entrava abaixo de `_COBERTURA_MIN_MOLDURA`,
            # porque uma medição anterior indicava que ela injetava a margem do
            # matplotlib como viés. Essa medição comparava contra o `t_window`
            # do meta, que é a FAIXA DE DADOS — e o eixo x do gráfico mostra o
            # `xlim`, que inclui a margem. Quem lê T do eixo (o consumidor real
            # da saída adimensional, e o que se faz numa imagem de terceiros)
            # obtém o `xlim`, não a faixa de dados: a referência da medição
            # anterior é que estava errada, não o resultado.
            #
            # Remedido com a mascara RGB, recuperando wn FÍSICO via
            # `T = sx * largura_da_moldura` (75 amostras de 2ª ordem sem
            # calibração):
            #     extensão observada -> MAPE 13,51 %, mediana 8,08 %
            #     moldura            -> MAPE  9,81 %, mediana 1,80 %
            # e no caso real `resposta_degrau.png`, 2,8 % -> 0,3 %.
            #
            # O gatilho por cobertura tambem deixou de funcionar como
            # discriminador: com a mascara nova as duas imagens reais cobrem
            # 0,9638 e 0,9775, ACIMA da mediana do corpus (0,9325). Nenhum
            # limiar as separa — o que sobra do `_COBERTURA_MIN_MOLDURA` é o
            # portão da guarda de planura, não a escolha da âncora.
            origem, span = float(bbox_px[0]), span_moldura
    repouso = _nivel_de_repouso(y)
    desvio = repouso - y                       # inverte: pixel cresce p/ baixo
    escala = float(np.max(np.abs(desvio)))
    if not np.isfinite(escala) or escala <= 0:
        return None, None
    return (x - origem) / span, desvio / escala


def _vazio_adimensional() -> dict:
    """Bloco `dimensionless` com as chaves presentes e valores nulos.

    O PLANO §1.7 exige o bloco SEMPRE preenchido (critério 2.11). Quando não há
    polilinha suficiente para ajustar nada, o contrato é honrado pela estrutura:
    as chaves existem, os valores são `None`. Nunca ausente, nunca exceção.
    """
    return {"zeta": None, "wn_T": None, "tau_T": None,
            "theta_T": None, "theta_tau": None, "K_yrange": None}


def _adimensional(params: dict, T: float, y_faixa: float) -> dict:
    """Grandezas adimensionais a partir de um ajuste e das escalas observadas.

    `T` é a duração da janela e `y_faixa` a amplitude do sinal, ambas nas MESMAS
    unidades do ajuste. No quadro normalizado as duas valem 1 por construção, o
    que faz esta função servir aos dois caminhos sem ramificar.
    """
    d = _vazio_adimensional()
    T = float(T) if np.isfinite(T) and T > 0 else float("nan")
    yf = float(y_faixa) if np.isfinite(y_faixa) and y_faixa > 0 else float("nan")
    z, wn, tau = params.get("zeta"), params.get("wn"), params.get("tau")
    th, K = params.get("theta"), params.get("K")
    if z is not None:
        d["zeta"] = float(z)
    if wn is not None and np.isfinite(T):
        d["wn_T"] = float(wn) * T
    if tau is not None and np.isfinite(T) and T > 0:
        d["tau_T"] = float(tau) / T
    if th is not None and np.isfinite(T) and T > 0:
        d["theta_T"] = float(th) / T
    if th is not None and tau not in (None, 0) and tau is not None:
        try:
            d["theta_tau"] = float(th) / float(tau)
        except ZeroDivisionError:
            pass
    if K is not None and np.isfinite(yf) and yf > 0:
        d["K_yrange"] = float(K) / yf
    return d


def _escalas_por_eixo(cal) -> tuple[float | None, float | None]:
    """(T em segundos, faixa de y em unidade fisica) — cada uma se o SEU eixo
    calibrou, independente do outro.

    A calibracao era tudo-ou-nada: um eixo bom era descartado porque o outro
    falhou. Medido em data/test (n=900): exigir os dois da 716 amostras, o eixo
    X sozinho da 789 e o Y sozinho 796. E o eixo X sozinho basta para a JANELA,
    logo para `wn`, `tau` e `theta` em unidade fisica — que era exatamente o que
    faltava nas tres imagens reais do bloco (duas delas tinham X aprovado e Y
    reprovado).
    """
    x0, y0, x1, y1 = cal.bbox_px
    T = None
    if getattr(cal, "ok_x", False) and np.isfinite(cal.sx) and x1 > x0:
        T = abs(float(cal.sx)) * float(x1 - x0)
        T = T if np.isfinite(T) and T > 0 else None
    yf = None
    if getattr(cal, "ok_y", False) and np.isfinite(cal.sy) and y1 > y0:
        yf = abs(float(cal.sy)) * float(y1 - y0)
        yf = yf if np.isfinite(yf) and yf > 0 else None
    return T, yf


def _fisico_parcial(dim: dict, cal) -> dict:
    """Desfaz a adimensionalizacao no que cada eixo permitir.

    `zeta` nunca depende de eixo. `wn`, `tau` e `theta` dependem so do eixo X;
    `K` so do eixo Y. Onde o eixo nao calibrou, a chave sai `None` — nunca um
    numero que finge unidade que nao existe.

    Coerencia da referencia de tempo: neste caminho o bloco adimensional vem de
    `_serie_normalizada`, que ancora `t` na MOLDURA. Por isso `T` aqui e a
    largura da moldura em segundos, e nao a extensao observada da polilinha —
    as duas pontas usam a mesma janela, entao a inversao fecha.
    """
    T, yf = _escalas_por_eixo(cal)
    out = {"K": None, "tau": None, "theta": None, "wn": None, "zeta": None}
    if dim.get("zeta") is not None:
        out["zeta"] = float(dim["zeta"])
    if T is not None:
        if dim.get("wn_T") is not None:
            out["wn"] = float(dim["wn_T"]) / T
        if dim.get("tau_T") is not None:
            out["tau"] = float(dim["tau_T"]) * T
        if dim.get("theta_T") is not None:
            out["theta"] = float(dim["theta_T"]) * T
    if yf is not None and dim.get("K_yrange") is not None:
        out["K"] = float(dim["K_yrange"]) * yf
    return out


def identify_from_image(image_rgb: np.ndarray, model, device: str = "cpu",
                        extractor=None) -> dict:
    """Imagem -> parâmetros. Nunca levanta: falha vira ok=False.

    `extractor`: opcional, `callable(image_rgb) -> mask uint8 0/255` — troca
    `predict_mask` (U-Net, Bloco 3) por outro extrator com a mesma
    assinatura de saída, ex. `identify.extract_classical.extract_mask_classical`
    (Bloco 3b). Quando `None` (padrão, assinatura idêntica à do
    `PLANO_PARTE2.md`), usa a U-Net via `model`/`device` como sempre.

    **Decisão E do PLANO §1.7 (HANDOFF_P2_7 Rulings 34 e 44).** A saída tem dois
    níveis. O `dimensionless` sai SEMPRE (critério 2.11); o `physical` só quando
    a calibração fecha, e é `None` caso contrário. Antes desta mudança
    `cal.ok == False` abortava a amostra inteira — contra o plano, e custando 56
    das ~68 amostras perdidas do pipeline (Ruling 42), justamente onde ζ é
    recuperável sem calibração nenhuma (medido: 53 recuperadas, ζ a 2,93 % de
    MAPE, contra 2,40 % do caminho físico).

    Duas decisões de compatibilidade, deliberadas:

    1. **`params` e `order` no topo continuam existindo** e continuam sendo os do
       nível FÍSICO, porque `tests/part2` e a Parte 3 os consomem. Os campos
       `physical`/`dimensionless`/`calibration` do PLANO §1.7 entram ao lado.
    2. **`ok` continua significando "há saída física"**, não "há resposta". É o
       que o próprio §1.7 pede em "Consequência nos critérios": 2.3, 2.4 e 2.5
       passam a ser medidos sobre o subconjunto em que a calibração declarou
       sucesso. Quem quiser o nível adimensional lê `dimensionless`, que nunca
       é nulo.

    Quando a calibração fecha, o bloco adimensional é DERIVADO do ajuste físico
    em vez de sair de um segundo ajuste. Isso evita um ajuste a mais por imagem e
    faz os dois níveis concordarem por construção — o §21.3 mediu 50,5 % de
    divergência no p95 entre ajustes independentes, e essa divergência deixa de
    existir aqui.
    """
    t0 = time.perf_counter()

    def _saida(order, params, ok, reason, dim, cal, n_pts):
        fis = dict(params) if (ok and params) else None
        # `physical` mantem o contrato antigo: existe se e so se `ok` (o teste
        # 2.11 assevera isso). `physical_parcial` e ADITIVO e entrega o que
        # cada eixo permitir — com a calibracao completa e o proprio `physical`.
        parcial = dict(fis) if fis is not None else _fisico_parcial(dim, cal)
        T_s, y_faixa_s = _escalas_por_eixo(cal)
        return {
            "order": order, "params": params, "ok": bool(ok), "reason": reason,
            "dimensionless": dim, "physical": fis,
            "physical_parcial": parcial,
            "calibration": {"ok": bool(cal.ok), "reason": cal.reason,
                            "ok_x": bool(getattr(cal, "ok_x", False)),
                            "ok_y": bool(getattr(cal, "ok_y", False)),
                            "T_s": T_s, "y_faixa": y_faixa_s,
                            "n_pairs_x": int(cal.n_pairs_x),
                            "n_pairs_y": int(cal.n_pairs_y)},
            "latency_ms": (time.perf_counter() - t0) * 1e3,
            "n_points": int(n_pts),
        }

    cal = calibrate(image_rgb)
    mask = extractor(image_rgb) if extractor is not None else predict_mask(model, image_rgb, device)
    x_px, y_px = mask_to_polyline(mask, bbox=cal.bbox_px if any(cal.bbox_px) else None)

    if x_px.size < 10:
        # Sem polilinha não há nível nenhum: o bloco adimensional existe, vazio.
        return _saida("", {}, False, "polilinha_curta",
                      _vazio_adimensional(), cal, x_px.size)

    if cal.ok:
        t, y = polyline_to_series(x_px, y_px, cal)
        ordem = np.argsort(t)
        t, y = t[ordem], y[ordem]
        fit = identify(t, y)
        dim = (_adimensional(fit.params, float(t[-1] - t[0]), float(np.ptp(y)))
               if fit.success else _vazio_adimensional())
        return _saida(fit.order, fit.params, bool(fit.success),
                      "" if fit.success else "ajuste_falhou", dim, cal, x_px.size)

    # Calibração falhou: só o nível adimensional é possível. No quadro
    # normalizado T = 1 e a faixa de y = 1, então `_adimensional` recebe as duas
    # como 1 e devolve as grandezas já normalizadas. `cal.bbox_px` (moldura)
    # segue disponível mesmo com `cal.ok == False` (falha é no mapeamento de
    # unidades dos eixos, não na detecção da moldura) e serve de referência de
    # tempo SÓ quando a polilinha estiver truncada — ver `_serie_normalizada`.
    tn, yn = _serie_normalizada(x_px, y_px, bbox_px=cal.bbox_px)
    if tn is None:
        return _saida("", {}, False, cal.reason,
                      _vazio_adimensional(), cal, x_px.size)
    g = identify(tn, yn)
    dim = _adimensional(g.params, 1.0, 1.0) if g.success else _vazio_adimensional()
    # `order` É adimensional — o PLANO §1.7 lista a estrutura como não dependente
    # de calibração —, então sai preenchido mesmo sem nível físico. Já `params`
    # fica vazio: ele é, por contrato, o bloco FÍSICO, que aqui não existe.
    # Consumidores antigos não se confundem porque todos passam por `ok`, que
    # continua falso; quem quer a estrutura sem calibração lê `order`.
    return _saida(g.order if g.success else "", {}, False, cal.reason,
                  dim, cal, x_px.size)
