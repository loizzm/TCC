"""Máscara -> polilinha -> série física. Determinístico, sem torch.

O `HANDOFF.md §4` mede o dado de projeto que dimensiona este módulo: o extrator
ingênuo "mediana por coluna" erra 0,19 px em linha sólida contra 0,92 px em
pontilhada, e o estilo `:` deixa 43% das colunas SEM TINTA. Por isso a
interpolação de vãos não é enfeite: sem ela, quase metade do domínio some.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

MAX_GAP_FRAC = 0.15    # vão máximo interpolado, como fração da largura da curva
MIN_COMPONENT_PX = 2   # componente menor que isso é ruído de 1 px, não traço


def _blocos(coluna: np.ndarray) -> list[tuple[int, int]]:
    """Blocos contíguos de tinta numa coluna -> [(linha_inicial, linha_final)]."""
    idx = np.flatnonzero(coluna)
    if idx.size == 0:
        return []
    cortes = np.flatnonzero(np.diff(idx) > 1)
    blocos, ini = [], 0
    for c in cortes:
        blocos.append((int(idx[ini]), int(idx[c])))
        ini = c + 1
    blocos.append((int(idx[ini]), int(idx[-1])))
    return blocos


# Salto máximo tolerado entre colunas vizinhas no caminho de RAMO ÚNICO, em
# múltiplos da espessura mediana do traço. Ver a `GUARDA DE CONTINUIDADE` em
# `mask_to_polyline` para a medição que fixa este valor.
SALTO_MAX_ESPESSURA: float = 8.0

# Tolerância da GUARDA DE EMENDA, em múltiplos da espessura mediana do traço:
# o quanto o segmento do outro lado de um vão pode estar LONGE de onde a
# continuação do segmento atual o colocaria. Ver `mask_to_polyline`.
EMENDA_MAX_ESPESSURA: float = 8.0

# Pontos usados para estimar a inclinação local à esquerda do vão. Curto de
# propósito: a inclinação que interessa é a de CHEGADA ao vão, não a média da
# curva. Oito pontos são ~8 colunas em traço sólido e ~14 em pontilhado (`:`
# deixa 43 % das colunas sem tinta, HANDOFF §4).
_EMENDA_JANELA: int = 8


def _emendas_suspeitas(x_arr: np.ndarray, y_arr: np.ndarray,
                       espessura: float) -> list[int]:
    """Índices `i` cuja ponte de `x_arr[i]` a `x_arr[i+1]` troca de objeto.

    O critério é CONTINUAÇÃO, não distância: extrapola a inclinação local da
    esquerda através do vão e mede o quanto o primeiro ponto da direita erra a
    previsão. Um vão dentro do mesmo traço (tracejado, pontilhado) acerta a
    previsão por construção, por mais íngreme que a curva esteja; um vão entre
    dois objetos diferentes erra pela distância que separa os objetos.
    """
    maus = []
    for i in range(x_arr.size - 1):
        vao = float(x_arr[i + 1] - x_arr[i])
        if vao <= 1.0:
            continue
        j = max(0, i - _EMENDA_JANELA)
        base = float(x_arr[i] - x_arr[j])
        incl = float(y_arr[i] - y_arr[j]) / base if base > 0.0 else 0.0
        predito = float(y_arr[i]) + incl * vao
        if abs(float(y_arr[i + 1]) - predito) > EMENDA_MAX_ESPESSURA * espessura:
            maus.append(i)
    return maus


def mask_to_polyline(mask: np.ndarray,
                     bbox: tuple[int, int, int, int] | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """UNIÃO das componentes conexas relevantes -> esqueleto -> mediana por
    coluna -> polilinha.

    NÃO usa só a MAIOR componente conexa: um traço tracejado/pontilhado
    (`line_style` em `-.`, `--`, `:`) é, por construção, uma sequência de
    componentes DESCONECTADAS — cada travessão/ponto é a sua própria
    componente. Manter só a maior descarta a curva quase inteira nesses
    estilos. Medido contra `mask.png` VERDADEIRA (sem nenhum ruído — o
    contrato da máscara garante isso): 40/300 amostras (todas com estilo
    tracejado/pontilhado) ficavam com menos de 10 pontos utilizáveis usando
    só a maior componente, e o RMSE mediano do estrato `traco=:` estourava o
    alvo (2,43 px contra 2 px). A união de TODAS as componentes acima de
    `MIN_COMPONENT_PX` resolve os dois: nada na máscara verdadeira além da
    curva, então a união é sempre segura ali; contra uma máscara PREDITA
    (Bloco 3/3b), o limiar ainda descarta ruído de 1 px isolado. Ver Ruling
    no HANDOFF_P2_4.md.
    """
    # Recorte à moldura (HANDOFF_P2_7 §34.2). Título, rótulo de eixo e legenda
    # externa vivem FORA do quadro e não são a curva; no caso real a polilinha ia
    # de y=21 a 551 com a moldura em 39..503. `bbox=None` preserva o
    # comportamento anterior byte a byte, porque `tests/part2` compara números
    # medidos sem moldura contra o histórico das rodadas 3 a 6.
    if bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        fora = np.ones(mask.shape, dtype=bool)
        fora[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = False
        mask = mask.copy()
        mask[fora] = 0

    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        return np.empty(0), np.empty(0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.empty(0), np.empty(0)
    uniao = np.zeros(binary.shape, dtype=bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX:
            uniao |= (lab == k)
    if not uniao.any():
        return np.empty(0), np.empty(0)
    skel = skeletonize(uniao)

    # Espessura mediana do traço (em px, medida na máscara ANTES do esqueleto):
    # separa um vão real entre dois objetos (curva x legenda, curva x reta de
    # referência) de uma quebra espúria dentro do MESMO objeto — anti-aliasing
    # ou o próprio padrão de um traço pontilhado (`:`), que fragmenta o
    # esqueleto em blocos verticais muito próximos dentro de uma coluna. Sem
    # este piso, a desambiguação por blocos piora o estrato `traco=:` e o
    # sintético (ver nota no Step 5 do brief da Task 3 / Ruling 46).
    espessuras_coluna = uniao.sum(axis=0)
    espessuras_coluna = espessuras_coluna[espessuras_coluna > 0]
    espessura_mediana = float(np.median(espessuras_coluna)) if espessuras_coluna.size else 1.0
    VAO_MIN_FRAC = 3.0

    xs, ys = [], []
    anterior = None
    ultimo_x = None
    for x in range(skel.shape[1]):
        coluna = skel[:, x]
        linhas = np.flatnonzero(coluna)
        if not linhas.size:
            continue
        if (anterior is not None and ultimo_x is not None
                and (x - ultimo_x) > VAO_MIN_FRAC * espessura_mediana):
            # Vão largo demais desde o último ponto: a referência está velha
            # demais para confiar — a curva pode ter se deslocado o bastante
            # no meio do vão para que "o bloco mais próximo do ponto
            # anterior" escolha o bloco ERRADO. Descarta a referência e cai
            # no fallback seguro (mediana de todas as linhas, o caminho de
            # ramo único), em vez de arriscar seguir o bloco errado com
            # confiança falsa.
            anterior = None
        blocos = _blocos(coluna)
        multi_ramo = False
        if len(blocos) > 1:
            bordas = sorted(blocos)
            vao_maximo = max(b[0] - a[1] for a, b in zip(bordas, bordas[1:]))
            multi_ramo = vao_maximo > VAO_MIN_FRAC * espessura_mediana
        if not multi_ramo or anterior is None:
            # Ramo único (ou blocos próximos demais para serem objetos
            # distintos): mediana de TODAS as linhas, idêntico ao
            # comportamento anterior. O Ruling 46 mediu que mexer aqui PIORA
            # o sintético.
            v = float(np.median(linhas))
            # GUARDA DE CONTINUIDADE. Aceitar o bloco único INCONDICIONALMENTE
            # era o defeito: numa curva TRACEJADA, no vão do traço a única
            # tinta da coluna é a LINHA DE ENTRADA desenhada no mesmo quadro.
            # O bloco único passa a ser o distrator, vira o novo `anterior`, e
            # da coluna seguinte em diante a desambiguação por ramos segue o
            # objeto errado com confiança. A polilinha fica AGARRADA na
            # entrada.
            #
            # Evidência do mecanismo (n=139 figuras de um degrau, geradores
            # reais): curva SÓLIDA — que não tem vãos — falha em 8/43
            # (18,6 %); tracejada ou pontilhada, em 46/96 (47,9 %). Fisher
            # OR=0,248, p=0,0012. Se a causa fosse "a máscara confunde os
            # objetos por proximidade", o estilo do traço não importaria.
            #
            # A guarda pula a coluna e PRESERVA `anterior`. O vão deixa de
            # corromper a referência, a interpolação final cobre o buraco, e
            # quando a tinta da curva volta a referência ainda está correta.
            # É conservador por construção: troca um ponto provavelmente
            # errado por um interpolado.
            #
            # Medido nas 297 figuras que calibram: extrações sujas
            # (NRMSE >= 0,05 contra a verdade analítica) caem de 115 para 48;
            # 113 figuras melhoram, 5 pioram, NENHUMA perde a polilinha — e as
            # 5 que pioram já estavam sujas, então a guarda não quebra
            # extração limpa nenhuma. Ponta a ponta: entrega física de 88,6 %
            # para 95,7 %, recusas de 34 para 13, truncagem espúria de 31 para
            # 17, e o núcleo de 1 degrau sem truncagem espúria vai de n=85
            # para n=116 com o MESMO |erro de K| mediano (0,0024).
            #
            # SALTO_MAX = 8 é o CENTRO de um platô, não a beira dele: entre 5 e
            # 12 o resultado é praticamente idêntico (5 vs 8 diferem em 9 de
            # 297 figuras; 8 vs 12, em 4), porque o salto para o distrator é
            # ordens de grandeza maior que o limiar. Mesma disciplina que
            # `_GANHO_MIN` documenta em `identify/classical.py`.
            #
            # ATENÇÃO: como toda constante a jusante do Estágio A, este 8
            # precisa ser REMEDIDO se a máscara mudar — as bordas do platô são
            # artefato do extrator atual, não do problema.
            #
            # REMEDIDO na promoção da época 17 do `render2` (14/09/2026), nas
            # mesmas 297 figuras. A guarda ficou INERTE: sem guarda, com 3, 5,
            # 8 e 12 dão o MESMO resultado — 15/297 extrações sujas
            # (NRMSE >= 0,05) e NRMSE p50 de 0,0033, idênticos até a quarta
            # casa. O defeito que ela existe para pegar quase desapareceu com
            # a máscara nova: eram 115/297 sujas SEM guarda nenhuma, hoje são
            # 15/297 sem guarda nenhuma.
            #
            # O 8 FICA, e a razão é a mesma que `_NRMSE_MAX` documenta sobre a
            # guarda de descontinuidade refutada: uma população que não contém
            # o modo de falha mede só o CUSTO da guarda, nunca o benefício.
            # Aqui o custo medido é ZERO e o mecanismo (polilinha agarrada na
            # linha de entrada no vão do tracejado) continua possível. Tirar a
            # guarda porque ela não dispara hoje seria confundir "não há
            # evidência de benefício nesta população" com "não há benefício".
            if (anterior is not None
                    and abs(v - anterior) > SALTO_MAX_ESPESSURA * espessura_mediana):
                continue
        else:
            # Ramo múltiplo de verdade (HANDOFF_P2_7 §34.2): a coluna tem mais
            # de um objeto — no caso real, a curva e a amostra de linha da
            # legenda. Segue o bloco mais próximo do ponto anterior e usa a
            # mediana DAQUELE bloco.
            a, b = min(blocos,
                       key=lambda t: 0.0 if t[0] <= anterior <= t[1]
                       else min(abs(t[0] - anterior), abs(t[1] - anterior)))
            dentro = linhas[(linhas >= a) & (linhas <= b)]
            v = float(np.median(dentro)) if dentro.size else float(np.median(linhas))
        xs.append(float(x))
        ys.append(v)
        anterior = v
        ultimo_x = x
    if len(xs) < 2:
        return np.empty(0), np.empty(0)

    x_arr, y_arr = np.asarray(xs), np.asarray(ys)

    # GUARDA DE EMENDA. `MAX_GAP_FRAC`, abaixo, só olha a DISTÂNCIA HORIZONTAL
    # do vão, e por isso não vê o defeito que esta guarda pega: dois OBJETOS
    # diferentes ligados por um vão curto.
    #
    # O caso que a isolou (`resposta_3.png`, matplotlib puro, um degrau, 2ª
    # ordem sobreamortecida, figura visualmente perfeita): o Estágio A perde a
    # cauda assentada a partir de x = 828 px — trecho perfeitamente reto, o
    # `Defeito B` que `tests/part2/test_caso_real_rg.py` já registra — e 45
    # colunas adiante acende a TRACEJADA DA ENTRADA `u(t) = 1`. A emenda cria
    # uma rampa de y = 0,80 para 1,00 que o gráfico não desenha. O vão tem 45 px
    # numa curva de 823 px: 5,5 %, bem abaixo dos 15 % de `MAX_GAP_FRAC`.
    #
    # O critério é CONTINUAÇÃO, não distância — ver `_emendas_suspeitas`. E o
    # segmento REJEITADO é descartado, não só a ponte: deixar a ponte em `nan` e
    # manter os dois lados foi medido na mesma figura e NÃO resolve (o platô
    # órfão da tracejada continua na série, `nrmse` do degrau único fica em
    # 0,0525 contra 0,0529 sem guarda nenhuma). Fica o segmento de maior
    # EXTENSÃO EM X, que é a curva sempre que o distrator é um pedaço de outro
    # objeto.
    #
    # MEDIDO nas 297 figuras que calibraram `SALTO_MAX_ESPESSURA`, contra a
    # verdade analítica, com a máscara promovida:
    #
    #   tolerância   NRMSE p50   extrações sujas (>= 0,05)   pareado vs. hoje
    #   ----------   ---------   -------------------------   -----------------
    #   sem guarda     0,0033            15/297              --
    #   4              0,0032             1/297              30 sobem / 1 cai
    #   8              0,0033             2/297              21 sobem / 1 cai
    #   12             0,0033             3/297              19 sobem / 0 caem
    #
    # NENHUMA figura perde a polilinha em nenhuma tolerância, e de 4 a 12 o
    # resultado é o mesmo platô. 8 é o CENTRO do platô, e é o mesmo valor e a
    # mesma unidade de `SALTO_MAX_ESPESSURA` — as duas guardas medem a mesma
    # coisa (distância até onde a curva deveria estar), uma entre colunas
    # vizinhas e outra através de um vão. Numa varredura anterior, SEM a
    # restrição de cauda logo abaixo, 2 já era ruído (43 sobem contra 42 caem);
    # a restrição não mexeu no platô (22/1 viraram 21/1 em 8).
    #
    # A ÚNICA que cai em 8 é `multi_20.png` (3 degraus), de 0,0532 para 0,0779 —
    # já suja antes da guarda. Como na guarda de continuidade, nenhuma extração
    # LIMPA é quebrada. Os maiores ganhos são polilinhas que estavam agarradas
    # num distrator inteiro: 0,768 -> 0,0079, 0,669 -> 0,0074, 0,340 -> 0,0105.
    #
    # PONTA A PONTA NOS QUATRO LOTES DE CONTROLE (700 figuras), veredito
    # conjuntivo de `acerto_conjuntivo.py`, sem guarda -> com guarda:
    #
    #   lote            n     ESTRITO      PRATICO      TOLERANTE
    #   -------------  ---   ----------   ----------   -----------
    #   lote_k_menor1  100    82 ->  84    89 ->  90    91 ->  91
    #   lote_k_maior1  100    94 ->  94    96 ->  96    96 ->  96
    #   lote_ruido     100    70 ->  69    85 ->  84    91 ->  90
    #   lote_misto2    400   304 -> 305   340 -> 341   365 -> 365
    #   TOTAL          700   550 -> 552   610 -> 611   643 -> 642
    #
    # A guarda muda o resultado em 17 das 700. No ESTRITO 4 sobem e 2 caem; no
    # PRATICO 3 sobem e 2 caem. É um EMPATE nesta população, e era de se
    # esperar: os lotes de controle já são quase todos de extração limpa, e o
    # ganho grande da guarda está em figura suja (no corpus de 297, extrações
    # sujas caem de 15 para 2). Onde ela age, age forte — NRMSE de curva
    # 0,0999 -> 0,0012, 0,0513 -> 0,0106, 0,0478 -> 0,0010, 0,0264 -> 0,0051.
    #
    # AS DUAS QUE CAEM, auditadas, e o modo de falha que elas expõem:
    #
    #   `ruido_Kmenor1_20dB_03.png` (0,0154 -> 0,0686). OCLUSÃO POR LEGENDA: a
    #   caixa tapa a curva de t = 100 s a 120 s e ela reaparece MAIS BAIXO. A
    #   guarda lê o reaparecimento como outro objeto e amputa 27 % da janela;
    #   sem o patamar assentado o erro de K vai de 0,8 % para 10 % e o de t_dom
    #   de 0,7 % para 35 %. É custo real e novo — ver OCLUSAO_LEGENDA.md.
    #
    #   `bal_K-_1deg_129.png` (0,0172 -> 0,1219). Curva pontilhada; a guarda
    #   corta só 6 % da cauda, e mesmo assim K vai de 2,5 % para 19 % e t_dom de
    #   7 % para 24 % — é a cauda que fixa K.
    #
    # LIMITAR O TAMANHO DA CAUDA DESCARTADA NÃO SEPARA os casos, e por isso não
    # está aqui: a `bal_K-_1deg_129` descarta 6 % e regride, enquanto a
    # `resposta_3.png`, que é o alvo, descarta 10 % e é o conserto.
    #
    # ATENÇÃO, como toda constante a jusante do Estágio A: as tabelas acima
    # valem para a máscara promovida. Mudou a máscara, remeça.
    maus = _emendas_suspeitas(x_arr, y_arr, espessura_mediana)
    if maus:
        bordas = [0] + [i + 1 for i in maus] + [x_arr.size]
        a, b = max(((bordas[k], bordas[k + 1]) for k in range(len(bordas) - 1)),
                   key=lambda seg: x_arr[seg[1] - 1] - x_arr[seg[0]])
        # SÓ CORTA A CAUDA. Quando o maior segmento não é o primeiro, a guarda
        # se declara incompetente e não mexe em nada.
        #
        # A assimetria NÃO é conveniência — as duas pontas carregam coisas
        # diferentes. A cauda carrega o patamar assentado, que o ajuste
        # extrapola: perdê-la custa PRECISÃO. A cabeça carrega o repouso e a
        # partida da resposta, de onde saem `theta`, o sinal do degrau e a
        # detecção de `resposta_inversa`: perdê-la custa SIGNIFICADO — a série
        # amputada vira uma curva bem-comportada diferente, e a pipeline passa
        # a ACEITAR em silêncio o que deveria recusar.
        #
        # Medido, e foi assim que a assimetria apareceu: com a regra "fique com
        # o maior segmento" sem restrição, em `data/val_nmp[:60]` (fase não
        # mínima, fora da família) a guarda cortava a CABEÇA em 21 das 60 e a
        # cauda em NENHUMA, e a recusa por `resposta_inversa` desabava de 90 %
        # para 58,3 % — o estrato existe exatamente para proteger essa guarda
        # (ver `tests/part2/test_estrato_fora_da_familia.py`). A máscara
        # fragmenta a curva fora da família (cobertura do platô de repouso 0,884
        # -> 0,286) e é a EMENDA por cima do mergulho que denuncia o defeito;
        # cortá-la apagava a evidência.
        if a == 0:
            x_arr, y_arr = x_arr[a:b], y_arr[a:b]
            if x_arr.size < 2:
                return np.empty(0), np.empty(0)

    x_full = np.arange(int(x_arr[0]), int(x_arr[-1]) + 1, dtype=float)
    y_full = np.interp(x_full, x_arr, y_arr)

    # Vão longo demais não é traço pontilhado: é ausência de dado. Descarta.
    largura = x_arr[-1] - x_arr[0]
    if largura > 0:
        vaos = np.diff(x_arr)
        for i in np.flatnonzero(vaos > MAX_GAP_FRAC * largura):
            corte = (x_full > x_arr[i]) & (x_full < x_arr[i + 1])
            y_full[corte] = np.nan
    ok = ~np.isnan(y_full)
    return x_full[ok], y_full[ok]


def polyline_to_series(x_px: np.ndarray, y_px: np.ndarray, cal) -> tuple[np.ndarray, np.ndarray]:
    """Pixels -> unidades físicas, com a afim estimada pelo Estágio B."""
    from identify.calibrate import px_to_data
    return px_to_data(cal, x_px, y_px)
