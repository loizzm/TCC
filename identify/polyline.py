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
