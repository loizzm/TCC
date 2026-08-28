"""Estágio B — calibração dos eixos. Determinístico; o OCR entra no Bloco 2.

Convenções (ver PLANO_PARTE2.md, Restrições globais): origem de pixel no canto
superior esquerdo, centro do pixel i em i+0.5; bbox = [x0, y0, x1, y1] no centro
da spine; x_dados = sx*x_px + ox, y_dados = sy*y_px + oy com sy < 0.

Nenhuma função aqui levanta exceção: falha devolve None ou ok=False.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

# --- detecção da moldura (Bloco 1) ------------------------------------------
#
# `left` e `bottom` sempre existem (`dataset/randomize.py:289`); `right` e
# `top` aparecem com p = 0,45 cada, e o fundo dos eixos é sempre igual ao
# fundo da figura (`dataset/generator.py:200-202`) — ou seja, não há
# descontinuidade de cor a detectar na moldura da área de dados quando o
# spine correspondente está ausente. A única forma robusta de achar o lado
# ausente é observar que qualquer traço que cruze *toda* a área de dados
# (o spine garantido do lado oposto, uma linha de grade, uma reta
# distratora — todos sorteados via `axhline`/`axvline`, que o matplotlib
# recorta exatamente no retângulo dos eixos) revela os DOIS lados
# perpendiculares ao seu próprio traçado: a linha do spine inferior (sempre
# visível) atravessa exatamente de x0 a x1, e a do spine esquerdo (sempre
# visível), de y0 a y1. Não é preciso achar um traço no lado ausente; o
# traço do lado garantido já contém a informação do lado ausente na sua
# própria extensão.
INK_THR = 12.0          # desvio mínimo (em nível de cinza) para contar como "tinta"
SPINE_COVER = 0.55      # fração mínima da largura/altura que o traço deve cobrir
SPINE_MIN_FILL = 0.90   # fração mínima de pixels de tinta dentro do próprio intervalo
MIN_BBOX_PX = 8         # bbox menor que isso é considerado ruído, não moldura


def _ink_mask(gray: np.ndarray, bg: float) -> np.ndarray:
    return np.abs(gray.astype(np.float32) - bg) > INK_THR


def detect_plot_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Retângulo da área de dados. None quando nem left/bottom são achados.

    Varre de baixo para cima a primeira linha "cheia" (o spine inferior,
    sempre presente) — sua própria extensão horizontal já dá x0 e x1.
    Varre da esquerda para a direita a primeira coluna "cheia" (o spine
    esquerdo, sempre presente) — sua extensão vertical já dá y0 e y1.
    """
    h, w = gray.shape
    bg = float(np.median(gray))
    ink = _ink_mask(gray, bg)

    row_counts = ink.sum(axis=1)
    y1 = x0_row = x1_row = None
    for r in range(h - 1, -1, -1):
        if row_counts[r] == 0:
            continue
        cols = np.flatnonzero(ink[r])
        span = cols[-1] - cols[0]
        if span >= SPINE_COVER * w and cols.size / (span + 1) >= SPINE_MIN_FILL:
            y1, x0_row, x1_row = r, int(cols[0]), int(cols[-1])
            break

    col_counts = ink.sum(axis=0)
    x0 = y0_col = y1_col = None
    for c in range(w):
        if col_counts[c] == 0:
            continue
        rows = np.flatnonzero(ink[:, c])
        span = rows[-1] - rows[0]
        if span >= SPINE_COVER * h and rows.size / (span + 1) >= SPINE_MIN_FILL:
            x0, y0_col, y1_col = c, int(rows[0]), int(rows[-1])
            break

    if y1 is None or x0 is None:
        return None
    x1, y0 = x1_row, y0_col
    if x1 - x0 < MIN_BBOX_PX or y1 - y0 < MIN_BBOX_PX:
        return None
    return (x0, y0, x1, y1)


# --- detecção de ticks (Bloco 1) --------------------------------------------

TICK_BAND = 6          # px inspecionados fora da moldura, do lado dos rótulos
TICK_PROM = 0.25       # proeminência mínima do pico, relativa ao máximo da faixa


def _peaks(sig: np.ndarray, prom: float) -> list[float]:
    """Picos locais simples acima de `prom * max`, com refino por centroide."""
    if sig.size < 3:
        return []
    thr = prom * float(sig.max())
    out: list[float] = []
    i = 1
    while i < sig.size - 1:
        if sig[i] >= thr and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
            j = i
            while j + 1 < sig.size and sig[j + 1] == sig[i]:
                j += 1
            lo, hi = max(i - 1, 0), min(j + 2, sig.size)
            w = sig[lo:hi]
            idx = np.arange(lo, hi, dtype=float)
            out.append(float((w * idx).sum() / max(w.sum(), 1e-9)))
            i = j + 1
        else:
            i += 1
    return out


SPINE_PAD = 4    # px descartados nas pontas da faixa (evita achar o SPINE
                  # PERPENDICULAR como se fosse um tick — ver Ruling)


def _merge_close(vals: list[float], tol: float = 8.0) -> list[float]:
    """União de picos de duas faixas, descartando duplicatas a menos de `tol` px."""
    out: list[float] = []
    for v in sorted(vals):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _sem_paralela(faixa: np.ndarray, eixo_do_tick: int) -> np.ndarray:
    """Remove da faixa o que for PARALELO ao eixo, preservando o tick.

    Um tick e perpendicular ao eixo e LOCALIZADO na coordenada do eixo. Uma
    linha paralela — tipicamente a grade coincidindo com o proprio spine —
    contribui IGUALMENTE para todas as posicoes, entao e a mediana ao longo do
    eixo, e subtrai-la a anula sem tocar no tick.

    Por que isso existe: a faixa de DENTRO era a fonte de quase todas as
    deteccoes falsas. Medido em `data/test` (n=895, eixo x): 7 espurios
    medianos com as duas faixas, 1 usando so a de fora. E nas tres imagens
    reais do bloco a faixa de dentro devolvia 70, 69 e 68 deteccoes para ~6, 7
    e 9 ticks verdadeiros — uma a cada ~9 px ao longo de toda a largura, o
    padrao de uma grade PONTILHADA sobre a borda. O corpus nunca reproduz isso
    porque `y_margin_lo` empurra o `ylim` para baixo do minimo dos dados, e a
    grade de y=0 nunca cai sobre o spine; nas imagens reais o `ylim` comeca em
    0 e cai exatamente ali.

    Nao se resolve usando so a faixa de fora: isso perde os ticks de direcao
    "in" inteiramente (medido no Bloco 2, ~1/3 das amostras com recall ~0).

    Efeito medido (eixo x, n=895): espurios medianos 7 -> 3, recall mediano
    100 % nos dois, recall p10 40 % -> 30,5 %. Nas imagens reais: 70 -> 5,
    69 -> 6, 68 -> 8.

    NAO ESTA EM USO, E O MOTIVO E O RESULTADO. Aplicada em `detect_tick_pixels`,
    a correcao NAO MOVEU NADA a jusante: `ok` 716/900, `ok_x` 789, `ok_y` 796 e
    55 falsos positivos, digito a digito iguais aos de antes, e as tres imagens
    reais tambem inalteradas. Os ticks espurios ja eram inofensivos — os
    recortes que eles geram sao descartados pelo filtro `_NUM_RE` do OCR. Como
    ha custo medido (recall p10 40 % -> 30,5 %) e beneficio zero, a aplicacao
    foi revertida. Fica aqui documentada para que ninguem gaste a mesma
    investigacao: melhorar a PRECISAO da deteccao de ticks nao destrava a
    calibracao. O gargalo e o RECALL DO OCR.
    """
    if faixa.size == 0:
        return faixa
    return np.clip(faixa - np.median(faixa, axis=eixo_do_tick, keepdims=True),
                   0.0, None)


def detect_tick_pixels(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, list[float]]:
    """Ticks maiores por picos de tinta na faixa ao redor da moldura.

    `tick_direction` (dataset/randomize.py) é sorteado em {"in", "out",
    "inout"} e não está disponível aqui (só a imagem) — então a faixa é
    checada dos DOIS lados da moldura (dentro e fora), nunca só de um.
    Checar só o lado de fora (como a primeira versão fazia) perde os ticks
    "in" inteiramente: medido, ~1/3 das amostras tinham recall de tick
    próximo de 0 nesse caso, mascarado na mediana do portão G1.2 (Bloco 1)
    porque só ~1/3 das amostras sorteiam "in" — a mediana continuava 1,0
    com os 2/3 restantes intactos. Ver Ruling no HANDOFF_P2_2.md.
    """
    x0, y0, x1, y1 = bbox
    h, w = gray.shape
    g = gray.astype(np.float32)
    fundo = float(np.median(g))
    tinta = np.abs(g - fundo)

    # As pontas da faixa (perto de x0/x1 para os ticks em x, perto de y0/y1
    # para os ticks em y) cruzam o spine PERPENDICULAR (esquerdo/direito para
    # x, inferior/superior para y) — sem cortar essas pontas, o próprio spine
    # aparece como um pico de tick espúrio bem na borda. Medido em
    # `sample_00000`: pico falso em x≈x0 lido como "0.0" por acidente (o
    # recorte do OCR, largo, alcançava o rótulo do tick real vizinho).
    xa, xb = x0 + SPINE_PAD, x1 + 1 - SPINE_PAD
    ya, yb = y0 + SPINE_PAD, y1 + 1 - SPINE_PAD
    faixa_x_fora = tinta[min(y1 + 1, h - 1):min(y1 + 1 + TICK_BAND, h), xa:xb]
    faixa_x_dentro = tinta[max(y1 - TICK_BAND, y0):y1, xa:xb]
    faixa_y_fora = tinta[ya:yb, max(x0 - TICK_BAND, 0):max(x0, 1)]
    faixa_y_dentro = tinta[ya:yb, x0 + 1:min(x0 + 1 + TICK_BAND, x1 + 1)]

    # A faixa de DENTRO passa por `_sem_paralela`: e nela que a grade sobre o
    # spine aparece, e ela e paralela ao eixo. A de FORA nao precisa — ali nao
    # ha grade, e o que existe (rotulos) fica alem de TICK_BAND.
    # Para os ticks em x a faixa e (linhas, colunas) e a estrutura paralela e
    # constante ao longo das COLUNAS -> mediana no eixo 1. Para os ticks em y a
    # faixa e transposta em papel, e a mediana vai no eixo 0.
    px = []
    if faixa_x_fora.size:
        px += [xa + p for p in _peaks(faixa_x_fora.sum(axis=0), TICK_PROM)]
    if faixa_x_dentro.size:
        px += [xa + p for p in _peaks(faixa_x_dentro.sum(axis=0), TICK_PROM)]
    py = []
    if faixa_y_fora.size:
        py += [ya + p for p in _peaks(faixa_y_fora.sum(axis=1), TICK_PROM)]
    if faixa_y_dentro.size:
        py += [ya + p for p in _peaks(faixa_y_dentro.sum(axis=1), TICK_PROM)]
    return {"x": _merge_close(px), "y": _merge_close(py)}


# --- Bloco 2: OCR opcional, RANSAC, consistência ----------------------------

_NUM_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?$")
# SEM tessedit_char_whitelist: no engine LSTM padrão do Tesseract 4/5 (não o
# legado, que era o único que o respeitava direito), a whitelist quebra o
# reconhecimento em vez de restringi-lo — medido: '4' e '8' bem nítidos,
# recortados corretamente, voltavam vazios (`\x0c`, "nenhum texto") só por
# causa da whitelist; sem ela, o mesmo recorte lê certo. O filtro `_NUM_RE`
# abaixo já garante que só uma leitura puramente numérica é aceita, então a
# whitelist era redundante mesmo quando funcionava. Ver Ruling no
# HANDOFF_P2_2.md.
_OCR_CFG = "--psm 7"


def _ocr_number(crop: np.ndarray) -> float | None:
    """Lê um número de um recorte. Devolve None em qualquer ambiguidade."""
    if crop.size == 0:
        return None
    img = Image.fromarray(crop.astype(np.uint8), mode="L")
    img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
    txt = pytesseract.image_to_string(img, config=_OCR_CFG).strip()
    txt = txt.replace(" ", "")
    if not _NUM_RE.match(txt):
        return None
    try:
        return float(txt.replace(",", "."))
    except ValueError:
        return None


_OCR_UP = 3            # fator de ampliação antes do OCR (era inline em _ocr_number)
_LOTE_GAP = 400        # px de fundo entre recortes vizinhos no mosaico, JÁ ampliados.
                       # Varrido em 99 amostras contra a implementação anterior
                       # (HANDOFF_P2_7 Ruling 35): 60 -> 60,6 % de calibração ok,
                       # 200 -> 75,8 %, **400 -> 77,8 % (empata com a referência)**,
                       # 700 -> 74,7 %. Folga pequena funde dois rótulos numa
                       # palavra só e o `_NUM_RE` rejeita os dois; folga grande
                       # espalha demais e o `--psm 7` perde a linha. 400 é o ótimo
                       # medido, não um palpite.
_LOTE_PAD = 8          # px de fundo na borda do mosaico


def _texto_para_numero(txt: str) -> float | None:
    """`str` -> número, com o MESMO filtro de `_ocr_number` (contrato §1.7)."""
    txt = txt.strip().replace(" ", "")
    if not _NUM_RE.match(txt):
        return None
    try:
        return float(txt.replace(",", "."))
    except ValueError:
        return None


def _ocr_numeros_lote(crops: list[np.ndarray]) -> list[float | None]:
    """Lê N recortes numéricos em UMA invocação do tesseract.

    Motivo (HANDOFF_P2_7 Ruling 31): o custo do tesseract aqui é partida de
    processo, não reconhecimento. Medido com recortes sintéticos, 52,6 ms para
    um rótulo e 55,0 ms para dez — ~52 ms fixos por invocação e ~0,3 ms por
    rótulo. Com mediana de 16 chamadas por imagem (p95 57, máx 105), o estágio B
    gastava 790 dos seus 800 ms só abrindo processo.

    Os recortes são ampliados individualmente (igual ao `_ocr_number`), então
    alinhados numa ÚNICA LINHA horizontal separada por `_LOTE_GAP` de fundo.
    A linha única preserva o `--psm 7` do `_OCR_CFG`: o tesseract continua
    vendo "uma linha de texto", só que com vários números nela.

    O mapeamento de volta usa as caixas do `image_to_data`: cada palavra é
    atribuída ao recorte cuja faixa horizontal contém o centro dela, e as
    palavras do mesmo recorte são concatenadas em ordem de x. Isso reproduz o
    `.strip().replace(" ", "")` do `_ocr_number` para o caso em que o tesseract
    quebra "1.5" em vários pedaços.
    """
    n = len(crops)
    if n == 0:
        return []
    amp: list[np.ndarray | None] = []
    for c in crops:
        if c is None or c.size == 0 or c.shape[0] < 1 or c.shape[1] < 1:
            amp.append(None)
            continue
        img = Image.fromarray(c.astype(np.uint8), mode="L")
        img = img.resize((img.width * _OCR_UP, img.height * _OCR_UP), Image.LANCZOS)
        amp.append(np.asarray(img, dtype=np.uint8))
    validos = [i for i, a in enumerate(amp) if a is not None]
    if not validos:
        return [None] * n
    if len(validos) == 1:
        # nada a ganhar com mosaico, e evita qualquer divergência de leitura
        i = validos[0]
        out: list[float | None] = [None] * n
        out[i] = _ocr_number(crops[i])
        return out

    alt = max(amp[i].shape[0] for i in validos)
    fundo = int(np.median([int(np.median(amp[i])) for i in validos]))
    larguras = [amp[i].shape[1] for i in validos]
    larg_tot = _LOTE_PAD * 2 + sum(larguras) + _LOTE_GAP * (len(validos) - 1)
    mosaico = np.full((alt + _LOTE_PAD * 2, larg_tot), fundo, dtype=np.uint8)
    faixas: list[tuple[int, int, int]] = []      # (indice, x_ini, x_fim)
    cur = _LOTE_PAD
    for i in validos:
        a = amp[i]
        dy = _LOTE_PAD + (alt - a.shape[0]) // 2
        mosaico[dy:dy + a.shape[0], cur:cur + a.shape[1]] = a
        faixas.append((i, cur, cur + a.shape[1]))
        cur += a.shape[1] + _LOTE_GAP

    try:
        dados = pytesseract.image_to_data(
            Image.fromarray(mosaico, mode="L"), config=_OCR_CFG,
            output_type=pytesseract.Output.DICT)
    except Exception:
        return [_ocr_number(c) for c in crops]

    achados: dict[int, list[tuple[float, str]]] = {}
    for txt, left, width in zip(dados.get("text", []), dados.get("left", []),
                                dados.get("width", [])):
        if not txt or not txt.strip():
            continue
        centro = float(left) + float(width) / 2.0
        for i, xa, xb in faixas:
            if xa <= centro <= xb:
                achados.setdefault(i, []).append((float(left), txt))
                break

    out = [None] * n
    for i, itens in achados.items():
        itens.sort(key=lambda p: p[0])
        out[i] = _texto_para_numero("".join(t for _, t in itens))
    return out


MARGIN_X_H = 40    # altura da faixa de rótulos do eixo x, abaixo da moldura
MARGIN_Y_W = 90    # largura da faixa de rótulos do eixo y, à esquerda da moldura
TICK_GAP = 8       # px mais próximos da moldura, excluídos da busca de blob:
                    # é onde uma MARCA de tick (se existir) fica, e ela se
                    # funde com o dígito vizinho pela dilatação, confundindo
                    # o OCR (medido: "7.5" virava "75:" com a marca colada)
BLOB_DILATE_X = 8  # px: funde dígitos/sinal/ponto do MESMO número num só blob
BLOB_DILATE_Y = 2  # px: mantém rótulos de LINHAS diferentes (eixo y) separados
MIN_BLOB_AREA = 4  # blob menor que isso é ruído de antialiasing, não texto


def _text_blobs(strip: np.ndarray, fundo: float) -> list[tuple[int, int, int, int]]:
    """Componentes conexas prováveis de rótulo de texto num recorte.

    Dilata antes de rotular para fundir caracteres do MESMO número num só
    blob (`BLOB_DILATE_X`) sem fundir rótulos de ticks vizinhos entre si
    (`BLOB_DILATE_Y` pequeno mantém linhas do eixo y separadas). Devolve
    bboxes locais (x0, y0, x1, y1) dentro de `strip`, um por blob.
    """
    ink = (np.abs(strip.astype(np.float32) - fundo) > INK_THR).astype(np.uint8)
    if not ink.any():
        return []
    kernel = np.ones((2 * BLOB_DILATE_Y + 1, 2 * BLOB_DILATE_X + 1), np.uint8)
    dil = cv2.dilate(ink, kernel)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    boxes = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_BLOB_AREA:
            continue
        x, y = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
        bw, bh = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + bw, y + bh))
    return boxes


def read_tick_labels(gray: np.ndarray, bbox: tuple[int, int, int, int],
                     ticks: dict[str, list[float]]) -> dict[str, list[tuple[float, float]]]:
    """Lê os rótulos numéricos nas margens dos eixos. Pares (pixel, valor) lidos.

    NÃO usa `ticks` para posicionar o recorte do OCR (mantido no parâmetro só
    por compatibilidade de assinatura com o `PLANO_PARTE2.md`) — em vez
    disso, varre a margem inteira por BLOBS de texto e usa o próprio blob
    como janela de recorte e como posição do tick. Motivo, medido: um
    recorte de largura fixa centrado em cada `tick` detectado
    (`detect_tick_pixels`) lê o rótulo do tick VIZINHO sempre que há ticks
    menores sem rótulo entre os maiores (comuns quando `has_minor_ticks` ou
    quando `has_major_ticks=False` deixa só o texto do rótulo, sem marca
    nenhuma, no lugar onde a busca por "picos de tinta" esperava uma marca
    curta) — isso inflava os pares com duplicatas e valores relidos fora de
    posição, quebrando `_equiespacados` quase sempre (medido: só 2/30
    amostras chegavam a `ok=True` antes desta reescrita). Escanear a margem
    inteira por blobs de texto não depende de nenhuma marca de tick existir.
    Ver Ruling no HANDOFF_P2_2.md.
    """
    x0, y0, x1, y1 = bbox
    h, w = gray.shape
    fundo = float(np.median(gray))
    pares: dict[str, list[tuple[float, float]]] = {"x": [], "y": []}

    def _candidatos(strip, bx0, by0, bx1, by1, trim_top, trim_right):
        """Os MESMOS recortes de antes, em ordem de precedência.

        Antes cada um virava uma chamada separada ao tesseract e as duas
        tentativas extras só rodavam se a primeira falhasse. Agora todos vão
        no mesmo lote e a precedência é resolvida depois — o custo de um
        recorte a mais no mosaico é ~0,3 ms (Ruling 31), contra ~52 ms de uma
        invocação nova, então gerar os três sempre é mais barato que
        condicionar.
        """
        pad = 2
        cands = [strip[max(by0 - pad, 0):by1 + pad, max(bx0 - pad, 0):bx1 + pad]]
        if trim_top and by1 - by0 > TICK_GAP + 3:
            cands.append(strip[by0 + TICK_GAP:by1 + pad, max(bx0 - pad, 0):bx1 + pad])
        if trim_right and bx1 - bx0 > TICK_GAP + 3:
            cands.append(strip[max(by0 - pad, 0):by1 + pad, max(bx0 - pad, 0):bx1 - TICK_GAP])
        return cands

    faixa_x = gray[min(y1 + 1, h - 1):min(y1 + 1 + MARGIN_X_H, h), x0:x1 + 1]
    faixa_y = gray[y0:y1 + 1, max(x0 - MARGIN_Y_W, 0):x0]
    fy_w = faixa_y.shape[1]

    crops: list[np.ndarray] = []
    # (eixo, posicao_px, indices dos candidatos em ordem de precedencia)
    plano: list[tuple[str, float, list[int]]] = []
    for bx0, by0, bx1, by1 in _text_blobs(faixa_x, fundo):
        cs = _candidatos(faixa_x, bx0, by0, bx1, by1, by0 <= 1, False)
        plano.append(("x", x0 + (bx0 + bx1) / 2.0,
                      list(range(len(crops), len(crops) + len(cs)))))
        crops.extend(cs)
    for bx0, by0, bx1, by1 in _text_blobs(faixa_y, fundo):
        cs = _candidatos(faixa_y, bx0, by0, bx1, by1, False, bx1 >= fy_w - 1)
        plano.append(("y", y0 + (by0 + by1) / 2.0,
                      list(range(len(crops), len(crops) + len(cs)))))
        crops.extend(cs)

    lidos = _ocr_numeros_lote(crops)
    for eixo, pos, idxs in plano:
        for i in idxs:
            if lidos[i] is not None:
                pares[eixo].append((pos, lidos[i]))
                break
    return pares


RANSAC_TOL = 0.02      # tolerância relativa do inlier, em fração do span de valores
RANSAC_MIN = 2         # o PLANO: "bastam 2 ticks corretos por eixo"


def fit_axis_affine(pares: list[tuple[float, float]]) -> tuple[float, float, int] | None:
    """RANSAC exaustivo sobre pares (pixel, valor) -> (escala, offset, n_inliers).

    Exaustivo, não amostrado: são poucos ticks (tipicamente ≤ 12), então todos os
    pares cabem em O(n²) e o resultado fica determinístico — sem RNG, conforme as
    restrições globais.

    Desempate por RESÍDUO TOTAL (soma dos erros absolutos sobre TODOS os
    pares, não só os inliers) quando duas retas candidatas empatam em número
    de inliers — comum com poucos pontos (3 candidatos, 1 errado: quaisquer
    2 deles "empatam" em 2 inliers cada, porque 2 pontos sempre se ajustam
    um ao outro exatamente). Sem o desempate, a ordem de iteração decide
    arbitrariamente — medido: escolhia a reta formada por um ponto bom + um
    ponto com OCR errado, 20% de erro de escala, em vez da reta com os dois
    pontos corretos. O resíduo total sobre todos os pontos favorece a reta
    que mais se aproxima do CONJUNTO inteiro, não só do par que a define.
    Ver Ruling no HANDOFF_P2_2.md.
    """
    if len(pares) < RANSAC_MIN:
        return None
    vals = [v for _, v in pares]
    span = max(vals) - min(vals)
    tol = RANSAC_TOL * span if span > 0 else 1e-9
    melhor: tuple[float, float, int] | None = None
    melhor_residuo = float("inf")
    for i in range(len(pares)):
        for j in range(i + 1, len(pares)):
            (p1, v1), (p2, v2) = pares[i], pares[j]
            if abs(p2 - p1) < 1e-9:
                continue
            s = (v2 - v1) / (p2 - p1)
            o = v1 - s * p1
            resid_todos = [abs(s * p + o - v) for p, v in pares]
            inl = [(p, v) for (p, v), r in zip(pares, resid_todos) if r <= tol]
            if len(inl) < 2:
                continue
            residuo = float(sum(resid_todos))
            melhor_ate_agora = (
                melhor is None or len(inl) > melhor[2]
                or (len(inl) == melhor[2] and residuo < melhor_residuo)
            )
            if melhor_ate_agora:
                P = np.asarray([p for p, _ in inl], dtype=float)
                V = np.asarray([v for _, v in inl], dtype=float)
                A = np.vstack([P, np.ones_like(P)]).T
                sol, *_ = np.linalg.lstsq(A, V, rcond=None)
                melhor = (float(sol[0]), float(sol[1]), len(inl))
                melhor_residuo = residuo
    return melhor


@dataclass(frozen=True)
class Calibration:
    sx: float = float("nan")
    ox: float = float("nan")
    sy: float = float("nan")
    oy: float = float("nan")
    bbox_px: tuple[int, int, int, int] = (0, 0, 0, 0)
    n_pairs_x: int = 0
    n_pairs_y: int = 0
    ok: bool = False
    reason: str = ""
    # Estado POR EIXO. `ok` continua sendo `ok_x and ok_y`, entao todo
    # consumidor antigo se comporta igual. O ganho e para quem precisa de um
    # eixo so: o X sozinho da a JANELA em segundos, e com ela `wn` e `theta`
    # saem em unidade fisica sem o eixo Y. Medido em data/test (n=900):
    #   exigindo os dois (comportamento de hoje) .... 81,3%
    #   eixo X sozinho .............................. 89,3%
    #   eixo Y sozinho .............................. 89,1%
    # Nas tres imagens reais do bloco, duas tem X aprovado e Y reprovado — ou
    # seja, `wn` fisico estava disponivel e era descartado.
    ok_x: bool = False
    ok_y: bool = False


# Consistência interna (PLANO): ticks equiespaçados em valor E em pixel.
SPACING_TOL = 0.05     # desvio relativo máximo do espaçamento


def _equiespacados(pares: list[tuple[float, float]], tol: float) -> bool:
    """Ticks equiespaçados em valor E em pixel — TOLERANTE A LACUNAS.

    O OCR não lê 100% dos rótulos (medido: ler menos da metade é comum). Um
    tick perdido no MEIO da sequência não é inconsistência: é lacuna. A
    versão original comparava só diferenças CONSECUTIVAS contra a média, o
    que reprovava qualquer lacuna (uma diferença de "3 espaçamentos" não bate
    com a média de espaçamentos de "1") mesmo com todo par individualmente
    correto — medido: essa era a causa de quase toda reprovação por
    `calibration_failed` no Bloco 2 (não OCR errado, e sim OCR incompleto).
    Aqui, cada diferença consecutiva precisa ser próxima de um múltiplo
    INTEIRO do menor espaçamento observado — cobre tanto "sem lacuna" (razão
    1) quanto "faltou N ticks nesse trecho" (razão N+1), e ainda reprova um
    valor lido errado (razão longe de qualquer inteiro). Ver Ruling no
    HANDOFF_P2_2.md.
    """
    if len(pares) < 3:
        return True                       # 2 pontos não têm o que violar
    ps = np.asarray(sorted(p for p, _ in pares), dtype=float)
    vs = np.asarray(sorted(v for _, v in pares), dtype=float)
    for a in (ps, vs):
        d = np.diff(a)
        if d.size == 0 or float(np.min(d)) < 1e-9:
            return False
        unit = float(np.min(d))
        razao = d / unit
        n = np.round(razao)
        if np.any(n < 1):
            return False
        if float(np.max(np.abs(razao - n) / n)) > tol:
            return False
    return True


def _inliers(pares: list[tuple[float, float]], s: float, o: float) -> list[tuple[float, float]]:
    """Reconstrói o subconjunto inlier de um ajuste (mesma tolerância do RANSAC)."""
    vals = [v for _, v in pares]
    span = max(vals) - min(vals) if vals else 0.0
    tol = RANSAC_TOL * span if span > 0 else 1e-9
    return [(p, v) for p, v in pares if abs(s * p + o - v) <= tol]


def calibrate(image_rgb: np.ndarray) -> Calibration:
    """Estágio B completo. Nunca levanta: falha vira ok=False + reason.

    Ordem: RANSAC primeiro, consistência DEPOIS — sobre os INLIERS do
    RANSAC, não sobre os pares brutos do OCR. Checar consistência antes do
    RANSAC (a ordem do esqueleto original do `PLANO_PARTE2.md`) faz UM
    valor lido errado (comum: 1 em ~5 pares, medido) reprovar a amostra
    inteira, mesmo com todos os outros pares corretos — exatamente o tipo de
    outlier que o RANSAC existe para descartar. Consistência sobre o
    conjunto que o RANSAC já filtrou é o que sobrou depois de descartar leituras
    ruins; ela ainda pega o caso em que o próprio RANSAC converge para um
    subconjunto pequeno e espúrio. Ver Ruling no HANDOFF_P2_2.md.
    """
    g = (image_rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114],
                                                 dtype=np.float32))
    gray = g.round().astype(np.uint8)
    bbox = detect_plot_bbox(gray)
    if bbox is None:
        return Calibration(reason="bbox_not_found")
    ticks = detect_tick_pixels(gray, bbox)
    pares = read_tick_labels(gray, bbox, ticks)

    def _um_eixo(eixo: str) -> tuple[float, float, int, bool, str]:
        """Resolve UM eixo. Devolve (s, o, n_inliers, ok, motivo)."""
        p = pares[eixo]
        if len(p) < RANSAC_MIN:
            return float("nan"), float("nan"), len(p), False, "ocr_insuficiente"
        f = fit_axis_affine(p)
        if f is None:
            return float("nan"), float("nan"), len(p), False, "ransac_failed"
        s_, o_, n_ = f
        if not _equiespacados(_inliers(p, s_, o_), SPACING_TOL):
            return s_, o_, n_, False, "calibration_failed"
        # Sinal: x cresce para a direita; y da IMAGEM cresce para BAIXO, entao
        # a escala do eixo y tem de ser negativa. E estrutural, nao tolerancia.
        if not np.isfinite(s_) or (s_ <= 0 if eixo == "x" else s_ >= 0):
            return s_, o_, n_, False, "sinal_de_escala_invalido"
        return s_, o_, n_, True, ""

    sx, ox, nx, ok_x, motivo_x = _um_eixo("x")
    sy, oy, ny, ok_y, motivo_y = _um_eixo("y")
    if not (ok_x and ok_y):
        # `reason` mantem a semantica antiga: o motivo da falha, com o eixo X
        # tendo precedencia por ser o que destrava a janela temporal.
        return Calibration(sx=sx, ox=ox, sy=sy, oy=oy, bbox_px=bbox,
                           n_pairs_x=nx, n_pairs_y=ny, ok=False,
                           reason=motivo_x or motivo_y,
                           ok_x=ok_x, ok_y=ok_y)
    return Calibration(sx=sx, ox=ox, sy=sy, oy=oy, bbox_px=bbox,
                       n_pairs_x=nx, n_pairs_y=ny, ok=True, ok_x=True, ok_y=True)


def px_to_data(cal: Calibration, x_px: np.ndarray, y_px: np.ndarray):
    """Converte pixels em unidades físicas com a convenção da Parte 1."""
    return cal.sx * np.asarray(x_px, float) + cal.ox, \
           cal.sy * np.asarray(y_px, float) + cal.oy
