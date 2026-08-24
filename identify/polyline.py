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


def mask_to_polyline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    xs, ys = [], []
    for x in range(skel.shape[1]):
        linhas = np.flatnonzero(skel[:, x])
        if linhas.size:
            xs.append(float(x))
            ys.append(float(np.median(linhas)))
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
