---
tags: [modulo, estagio-b]
aliases: [calibrate.py, identify/calibrate.py]
---

# `identify/calibrate.py`

> **746 linhas · Estágio B.** Descobre a transformação afim que leva coordenada de pixel a unidade física. Determinístico, sem RNG.

O estágio mais frágil do projeto e o que mais defeitos escondeu.

---

## O que faz

Responde uma pergunta só: **quanto vale um pixel**. São quatro números.

```
x_fisico = sx · x_px + ox
y_fisico = sy · y_px + oy        (com sy < 0: o pixel cresce para baixo)
```

Para obtê-los é preciso ler os rótulos numéricos dos eixos e descobrir a que posição cada um corresponde. A cadeia tem quatro camadas:

| # | Camada | Função pública |
|---|---|---|
| 1 | Moldura da área de dados | `detect_plot_bbox` |
| 2 | Blobs de texto nas margens | `read_tick_labels` |
| 3 | OCR de cada recorte | (interno, `_ocr_numeros_lote`) |
| 4 | Ajuste da afim | `fit_axis_affine` |

---

## Assinatura principal

```python
def calibrate(image_rgb: np.ndarray) -> Calibration:
    """Estágio B completo. Nunca levanta: falha vira ok=False + reason."""
```

### Recebe
`uint8[H,W,3]` — a imagem original, não a máscara. B é **independente de A**.

### Devolve

```python
@dataclass(frozen=True)
class Calibration:
    sx: float = float("nan");  ox: float = float("nan")
    sy: float = float("nan");  oy: float = float("nan")
    bbox_px: tuple[int, int, int, int] = (0, 0, 0, 0)
    n_pairs_x: int = 0;  n_pairs_y: int = 0
    ok: bool = False
    reason: str = ""
    ok_x: bool = False       # estado POR EIXO
    ok_y: bool = False
```

`ok` continua sendo `ok_x and ok_y`, então consumidores antigos se comportam igual. O ganho é para quem precisa de um eixo só.

> [!note] Aprovação por eixo
> **Só o eixo X já dá a janela em segundos**, e com ela `wn`, `tau` e `theta` saem em unidade física sem o eixo Y. Medido em `data/test` (n=900):
> - exigindo os dois: 81,3 %
> - eixo X sozinho: 89,3 %
> - eixo Y sozinho: 89,1 %
>
> Nas três imagens reais do bloco, duas tinham X aprovado e Y reprovado — ωn físico estava disponível e era descartado.

Repare que `bbox_px` **continua preenchido mesmo com `ok=False`**: a falha é no mapeamento de unidades, não na detecção do retângulo. [[identify.pipeline]] usa isso para recortar a polilinha.

---

## Como faz

### 1. O nível de fundo é a MODA DA BORDA

```python
def _fundo(gray: np.ndarray) -> float:
    """Nível de fundo da figura: a MODA DA BORDA da imagem, não a mediana."""
    b = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    valores, contagens = np.unique(b, return_counts=True)
    return float(valores[int(np.argmax(contagens))])
```

A mediana global assume **um fundo só**. Numa figura de tema escuro com `ax.set_facecolor()` diferente do fundo da figura, e sendo a área de dados a maior região, a mediana cai no fundo *dos eixos* — e a moldura inteira vira tinta. Medido: bbox `(0, 0, 799, 460)` em vez de `(100, 55, 720, 410)`, zero rótulos lidos, calibração perdida.

Verificado idêntico nas 900 amostras de `data/test`, com bbox igual em 900/900.

### 2. Blobs de texto, com a banda de tick cortada

```python
BLOB_DILATE_X = 3   # funde dígitos/sinal/ponto do MESMO número num só blob
BLOB_DILATE_Y = 2   # mantém rótulos de LINHAS diferentes separados
TICK_GAP = 8        # px mais próximos da moldura, excluídos da busca
```

```python
def _text_blobs(strip, fundo, corta_topo: int = 0, corta_direita: int = 0):
    ink = (np.abs(strip.astype(np.float32) - fundo) > INK_THR).astype(np.uint8)
    if corta_topo or corta_direita:
        ink = ink.copy()
        if corta_topo:    ink[:corta_topo] = 0
        if corta_direita and ink.shape[1] > corta_direita:
            ink[:, -corta_direita:] = 0
    ...
```

> [!bug] O defeito que isso conserta
> A faixa de rótulos começa colada na moldura, então as **marcas de tick** — traços de 1 px espaçados ~15,7 px — estão dentro dela. Com dilatação horizontal de 8 px (o valor antigo) elas se costuram entre si e grudam nos rótulos: o eixo inteiro vira *um blob só*, centrado no meio.
>
> Âncora do 1º tick a ≤3 px: **76,1 % → 99,9 %**.

### 3. OCR em lote, com teto

O Tesseract é chamado uma vez por mosaico de recortes, não uma vez por recorte — é ordens de magnitude mais rápido. Mas ele tem um modo de falha silencioso:

```python
_LOTE_MAX = 12

def _ocr_numeros_lote(crops: list[np.ndarray]) -> list[float | None]:
    out: list[float | None] = [None] * len(crops)
    for ini in range(0, len(crops), _LOTE_MAX):
        bloco = crops[ini:ini + _LOTE_MAX]
        r = _ocr_um_mosaico(bloco)
        uteis = sum(1 for c in bloco if c is not None and c.size and ...)
        if uteis >= 2 and all(v is None for v in r):
            # colapso do lote: recua para OCR individual
            r = [_ocr_number(c) if (c is not None and c.size) else None for c in bloco]
        out[ini:ini + len(r)] = r
    return out
```

> [!bug] Colapso do lote
> Com `--psm 7` o Tesseract devolve **zero palavras** quando a análise de layout falha — sem levantar exceção. O lote inteiro vira nulo. Não é previsível pelo tamanho: 14 recortes → 11 lidos, 16 → **0**, 20 → 17, 21 → **0**. Custava 16 amostras do corpus sem nenhum par.
>
> O conserto tem duas metades: um **teto** de 12 por mosaico, e um **recuo individual** quando o lote inteiro colapsa apesar de haver recortes úteis.

### 4. RANSAC exaustivo, depois consistência

```python
RANSAC_TOL = 0.02   # tolerância do inlier, em fração do span de valores
RANSAC_MIN = 2      # "bastam 2 ticks corretos por eixo"

def fit_axis_affine(pares) -> tuple[float, float, int] | None:
```

São tipicamente ≤ 12 rótulos por eixo, então **todos** os pares cabem em O(n²). A versão exaustiva é determinística — sem amostragem aleatória, sem semente. O desempate entre retas com igual número de inliers usa o resíduo total sobre *todos* os pontos; sem isso a ordem de iteração decidia, e escolhia a reta formada por um ponto bom mais um ponto com OCR errado.

A ordem das duas checagens é uma decisão medida:

```python
f = fit_axis_affine(p)                                    # RANSAC PRIMEIRO
if not _equiespacados(_inliers(p, s_, o_), SPACING_TOL):  # consistência DEPOIS
    return ..., "calibration_failed"
```

Checar consistência antes faz **um** valor lido errado (comum: 1 em ~5 pares) reprovar a amostra inteira — exatamente o outlier que o RANSAC existe para descartar.

### 5. Equiespaçamento tolerante a lacunas

```python
def _equiespacados(pares, tol) -> bool:
    """Ticks equiespaçados em valor E em pixel — TOLERANTE A LACUNAS."""
    if len(pares) < 3:
        return True
    for a in (ps, vs):
        d = np.diff(a)
        unit = float(np.min(d))
        razao = d / unit
        n = np.round(razao)
        if np.any(n < 1):                                     return False
        if float(np.max(np.abs(razao - n) / n)) > tol:        return False
    return True
```

O OCR não lê 100 % dos rótulos. Um tick perdido no **meio** da sequência não é inconsistência, é lacuna. A versão original comparava só diferenças consecutivas contra a média, o que reprovava qualquer lacuna — e essa era a causa de quase toda reprovação por `calibration_failed`. Aqui cada diferença precisa ser próxima de um **múltiplo inteiro** do menor espaçamento: cobre "sem lacuna" (razão 1) e "faltaram N ticks" (razão N+1), e ainda reprova um valor lido errado (razão longe de qualquer inteiro).

### 6. O sinal da escala é estrutural

```python
if not np.isfinite(s_) or (s_ <= 0 if eixo == "x" else s_ >= 0):
    return s_, o_, n_, False, "sinal_de_escala_invalido"
```

x cresce para a direita; y da **imagem** cresce para baixo, então `sy` tem de ser negativo. Isso não é tolerância, é impossibilidade física.

---

## Função auxiliar exportada

```python
def px_to_data(cal: Calibration, x_px, y_px):
    return cal.sx * np.asarray(x_px, float) + cal.ox, \
           cal.sy * np.asarray(y_px, float) + cal.oy
```

Usada por [[identify.polyline]]`.polyline_to_series`.

---

Ver também: [[Arquitetura do projeto#3. Estágio B — Calibração]] · [[Decisões de projeto#D4 · RANSAC antes da consistência]]
