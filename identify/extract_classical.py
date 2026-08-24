"""Estágio A sem rede — extrator clássico (PLANO_PARTE2.md §1.8, Bloco 3b).

Baseline e Plano B do risco de GPU: segmentação por cor + rejeição de
componentes retilíneas (grade, spines, distratoras), sem `torch` e sem treino.
Mesma assinatura de saída que `identify.extract.predict_mask` (Bloco 3):
`uint8` 0/255, mesma resolução da imagem de entrada — para ser intercambiável
no `identify/pipeline.py` (critério 2.10, Bloco 5).

A rejeição de retas de span completo (passo 4 do Bloco 3b) REUSA — copiada,
não importada, para não amarrar este módulo (deliberadamente leve, sem
`torch`, sem `pytest`/`sklearn`) ao pacote de testes — o algoritmo de
`tests/test_leakage.py::_spanning_rows`, calibrado em três iterações na
Parte 1 (ver a docstring de origem). As constantes abaixo são idênticas às de
lá.
"""
from __future__ import annotations

import cv2
import numpy as np

from identify.calibrate import detect_plot_bbox

# --- rejeição de retas de span completo (copiado de tests/test_leakage.py) --
INK_BG_TOL = 12
SPAN_FRAC = 0.98
MIN_INK_FRAC = 0.25
SPAN_BINS = 8
SPAN_MIN_BINS = 7
DASH_BRIDGE = 25         # px: fecha vãos de traço/reta pontilhada só para
                          # decidir se a linha/coluna "atravessa" (ver Ruling;
                          # varredura em {5,9,15,25}: mediana igual, p10 sobe
                          # de 0,068 para 0,384 — reduz o pior caso, não
                          # muda o alvo global)

# --- segmentação por cor ------------------------------------------------
QUANT = 32              # níveis de quantização por canal
MIN_MODE_FRAC = 0.001    # fração mínima de pixels para um modo ser candidato
BBOX_PAD = 4             # px descartados para dentro do bbox (evita as spines)
MIN_CURVE_PX = 15    # componente menor que isso é ruído, não curva
MIN_EXTENT_FRAC = 0.30   # candidato a curva precisa cobrir isso da largura útil


def _spanning_rows(ink: np.ndarray) -> np.ndarray:
    """Linhas (eixo 0) que atravessam a área de dados de ponta a ponta.

    Cópia exata de `tests/test_leakage.py::_spanning_rows` — não reescrever
    sem reler a docstring de lá; o denominador levou três iterações a acertar.
    """
    n = ink.shape[1]
    if n == 0 or ink.shape[0] == 0:
        return np.zeros(ink.shape[0], dtype=bool)
    cnt = ink.sum(1)
    first = np.argmax(ink, axis=1)
    last = n - 1 - np.argmax(ink[:, ::-1], axis=1)
    extent = np.where(cnt > 0, last - first, -1)
    edges = np.linspace(0, n, SPAN_BINS + 1).astype(int)
    filled = np.zeros(ink.shape[0], dtype=int)
    for a, bnd in zip(edges[:-1], edges[1:]):
        filled += ink[:, a:bnd].any(1)
    return ((extent >= SPAN_FRAC * (n - 1)) & (cnt >= MIN_INK_FRAC * n)
            & (filled >= SPAN_MIN_BINS))


def _bridge_gaps_1d(ink: np.ndarray, axis: int, k: int) -> np.ndarray:
    """Fecha vãos de até `k` px ao longo de `axis` (dilatação morfológica 1D).

    Só serve para DECIDIR se uma linha/coluna "atravessa" a área de dados
    (uma reta ou grade pontilhada/tracejada tem vãos regulares que reduzem a
    ocupação por linha abaixo do piso de `_spanning_rows` — medido: sem isso,
    ~14% das amostras escolhiam uma distratora pontilhada em vez da curva).
    O resultado NUNCA vai para a máscara de saída, só para a decisão.
    `cv2.dilate` com elemento estruturante 1D: O(1) por pixel (separável),
    contra o custo de um laço Python de `k` deslocamentos.
    """
    ksize = 2 * k + 1
    kernel = np.ones((ksize, 1), np.uint8) if axis == 0 else np.ones((1, ksize), np.uint8)
    return cv2.dilate(ink.astype(np.uint8), kernel) > 0


N_BUCKETS = 256 // QUANT   # baldes por canal (8, para QUANT=32)


def _bucket_key(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> um único inteiro 0..N_BUCKETS**3-1 (balde por canal).

    Evita `np.unique(..., axis=0)` sobre tuplas RGB, que é O(n log n) com
    comparação de tupla — lento demais para imagens de até 1600x1200 (~1,9 s
    medido, contra o alvo de 200 ms do critério G3b.4). `np.bincount` sobre um
    inteiro achatado é a mesma ideia em ordens de magnitude menos tempo.
    """
    b = (rgb.astype(np.int32) // QUANT)
    return (b[..., 0] * N_BUCKETS + b[..., 1]) * N_BUCKETS + b[..., 2]


def _modal_color_key(rgb: np.ndarray) -> int:
    key = _bucket_key(rgb).ravel()
    counts = np.bincount(key, minlength=N_BUCKETS ** 3)
    return int(np.argmax(counts))


def _color_modes(rgb: np.ndarray, exclude: np.ndarray) -> list[np.ndarray]:
    """Cores quantizadas mais frequentes fora do fundo, uma máscara por modo."""
    key = _bucket_key(rgb)
    flat = key.reshape(-1)
    valid = ~exclude.reshape(-1)
    if not valid.any():
        return []
    counts = np.bincount(flat[valid], minlength=N_BUCKETS ** 3)
    total = flat.size
    order = np.argsort(-counts)
    modes = []
    for k in order:
        if counts[k] / total < MIN_MODE_FRAC:
            break
        modes.append(key == k)
    return modes


def extract_mask_classical(image_rgb: np.ndarray,
                           bbox: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Segmentação da curva sem rede. Nunca levanta: falha devolve máscara vazia."""
    h, w = image_rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    try:
        if bbox is None:
            gray = (image_rgb.astype(np.float32)
                    @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
            bbox = detect_plot_bbox(gray.round().astype(np.uint8))
        if bbox is None:
            return out
        x0, y0, x1, y1 = bbox
        xa, xb = max(x0 + BBOX_PAD, 0), min(x1 - BBOX_PAD + 1, w)
        ya, yb = max(y0 + BBOX_PAD, 0), min(y1 - BBOX_PAD + 1, h)
        if xb - xa < 10 or yb - ya < 10:
            return out

        sub = image_rgb[ya:yb, xa:xb]
        # cor de fundo = moda de toda a imagem (mais robusta que só o recorte:
        # o recorte já é quase todo fundo, então a moda coincide, mas usar a
        # imagem inteira evita que uma curva muito espessa vire "fundo").
        bg_key = _modal_color_key(image_rgb)
        bg_mask_sub = _bucket_key(sub) == bg_key

        # Cada modo de cor pode ser o traço partido em muitos pedaços (estilo
        # tracejado/pontilhado): a rejeição de retas já opera em nível de
        # linha/coluna (não de componente), então depois dela o que sobra de
        # um modo é, por construção, o candidato a curva daquele modo — não
        # faz sentido escolher só o maior FRAGMENTO. Uma reta distratora muito
        # pontilhada (ciclo de trabalho baixo) pode escapar da rejeição de
        # span (que exige ocupação >= 25% por linha) e ainda assim atravessar
        # quase toda a largura — medido: em ~67% das amostras ela vencia a
        # curva de verdade só por ter extensão marginalmente maior. A curva
        # sempre tem MUITO mais tinta que um distrator pontilhado sobrevivente
        # (a curva ocupa cada coluna; o distrator só uma fração pequena) —
        # então o desempate entre candidatos que já cobrem uma largura
        # razoável (>= 30%) é por ÁREA (quantidade de tinta), não por
        # extensão pura.
        candidatos = []  # (area, extensao, mask_local)
        for mode_mask in _color_modes(sub, exclude=bg_mask_sub):
            cleaned = mode_mask.copy()
            # vãos horizontais (traço pontilhado/tracejado numa reta
            # HORIZONTAL) e vãos verticais (reta VERTICAL) precisam ser
            # fechados em eixos diferentes antes de cada checagem.
            bridged_h = _bridge_gaps_1d(cleaned, axis=1, k=DASH_BRIDGE)
            bridged_v = _bridge_gaps_1d(cleaned, axis=0, k=DASH_BRIDGE)
            rows = _spanning_rows(bridged_h)
            cols = _spanning_rows(bridged_v.T)
            cleaned[rows, :] = False
            cleaned[:, cols] = False
            area = int(cleaned.sum())
            if area < MIN_CURVE_PX:
                continue
            cols_ink = np.flatnonzero(cleaned.any(axis=0))
            extensao = int(cols_ink[-1] - cols_ink[0]) if cols_ink.size else 0
            if extensao < MIN_EXTENT_FRAC * cleaned.shape[1]:
                continue
            candidatos.append((area, extensao, cleaned))

        if candidatos:
            _, _, melhor = max(candidatos, key=lambda c: c[0])
            out[ya:yb, xa:xb][melhor] = 255
        return out
    except Exception:  # pragma: no cover - contrato: nunca levanta (contract.md §6)
        return np.zeros((h, w), dtype=np.uint8)
