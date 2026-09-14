---
tags: [modulo, estagio-a, baseline]
aliases: [extract_classical.py, extrator clássico]
---

# `identify/extract_classical.py`

> **184 linhas · Estágio A alternativo.** Segmentação da curva **sem rede neural**: cor modal + rejeição de retas de span completo.

Importa `numpy` e `cv2`. **Nunca `torch`** — e um teste assevera isso.

---

## Por que existe

Ele é **mais fraco** que a U-Net, e as duas razões de existir não são desempenho:

1. **Risco de projeto.** Se a GPU não estivesse disponível, o TCC precisaria de um Estágio A funcional.
2. **Epistemológica.** Ele é a *linha de base* contra a qual o ganho da rede é medido. Sem um extrator clássico honesto, "a U-Net funciona" seria afirmação sem contrafactual.

---

## Assinatura

```python
def extract_mask_classical(image_rgb: np.ndarray,
                           bbox: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Segmentação da curva sem rede. Nunca levanta: falha devolve máscara vazia."""
```

**Recebe** `uint8[H,W,3]` · **Devolve** `uint8[H,W]` 0/255, mesma resolução.

Contrato **idêntico** ao de `predict_mask` — é o que permite passá-lo como `extractor=` para [[identify.pipeline]] sem nenhum código condicional.

---

## Como faz

### 1. Recorta à moldura, com folga

```python
bbox = detect_plot_bbox(gray.round().astype(np.uint8))   # reusa o Estágio B
x0, y0, x1, y1 = bbox
xa, xb = max(x0 + BBOX_PAD, 0), min(x1 - BBOX_PAD + 1, w)
```

`BBOX_PAD = 4` px para dentro — evita que as próprias spines entrem como candidatas.

### 2. Quantiza a cor num inteiro só

```python
QUANT = 32
N_BUCKETS = 256 // QUANT      # 8 baldes por canal

def _bucket_key(rgb: np.ndarray) -> np.ndarray:
    b = (rgb.astype(np.int32) // QUANT)
    return (b[..., 0] * N_BUCKETS + b[..., 1]) * N_BUCKETS + b[..., 2]
```

> [!note] Por que não `np.unique(..., axis=0)`
> Sobre tuplas RGB é O(n log n) com comparação de tupla — **1,9 s medido** para imagens de até 1600×1200, contra o alvo de 200 ms do critério G3b.4. `np.bincount` sobre um inteiro achatado é a mesma ideia em ordens de magnitude menos tempo.

O fundo é a **moda de toda a imagem**, não do recorte: o recorte já é quase todo fundo, mas usar a imagem inteira evita que uma curva muito espessa vire "fundo".

### 3. Rejeita o que atravessa a área de dados

```python
SPAN_FRAC = 0.98        # extensão mínima, em fração da largura
MIN_INK_FRAC = 0.25     # ocupação mínima por linha
SPAN_BINS, SPAN_MIN_BINS = 8, 7

def _spanning_rows(ink: np.ndarray) -> np.ndarray:
    cnt = ink.sum(1)
    first = np.argmax(ink, axis=1)
    last = n - 1 - np.argmax(ink[:, ::-1], axis=1)
    extent = np.where(cnt > 0, last - first, -1)
    filled = ...            # em quantos dos 8 bins há tinta
    return ((extent >= SPAN_FRAC * (n - 1)) & (cnt >= MIN_INK_FRAC * n)
            & (filled >= SPAN_MIN_BINS))
```

Grade, spines e retas distratoras atravessam a área de ponta a ponta; a curva não. Os três critérios juntos — extensão, densidade e distribuição pelos bins — são o que separa uma reta de uma curva que por acaso cobre toda a largura.

> Cópia exata de `tests/test_leakage.py::_spanning_rows`, **copiada e não importada** para não amarrar este módulo (deliberadamente leve) ao pacote de testes. O denominador levou três iterações a acertar.

### 4. Fecha os vãos antes de decidir

```python
DASH_BRIDGE = 25

def _bridge_gaps_1d(ink, axis: int, k: int) -> np.ndarray:
    ksize = 2 * k + 1
    kernel = np.ones((ksize, 1), np.uint8) if axis == 0 else np.ones((1, ksize), np.uint8)
    return cv2.dilate(ink.astype(np.uint8), kernel) > 0
```

Uma reta distratora **pontilhada** tem vãos regulares que derrubam a ocupação por linha abaixo do piso de `_spanning_rows`. Sem esta ponte, ~14 % das amostras escolhiam uma distratora pontilhada em vez da curva.

O resultado **nunca vai para a máscara de saída** — só para a decisão. E os vãos horizontais (reta horizontal tracejada) e verticais (reta vertical) precisam ser fechados em eixos diferentes, daí as duas chamadas.

### 5. Desempate por ÁREA, não por extensão

```python
candidatos = []                                  # (area, extensao, mask_local)
for mode_mask in _color_modes(sub, exclude=bg_mask_sub):
    cleaned = mode_mask.copy()
    cleaned[_spanning_rows(bridged_h), :] = False
    cleaned[:, _spanning_rows(bridged_v.T)] = False
    if cleaned.sum() < MIN_CURVE_PX:                          continue
    if extensao < MIN_EXTENT_FRAC * cleaned.shape[1]:         continue
    candidatos.append((int(cleaned.sum()), extensao, cleaned))

_, _, melhor = max(candidatos, key=lambda c: c[0])            # por ÁREA
```

Uma reta distratora muito pontilhada (ciclo de trabalho baixo) pode escapar da rejeição de span e ainda atravessar quase toda a largura. Medido: em **~67 % das amostras** ela vencia a curva de verdade só por ter extensão marginalmente maior.

A curva sempre tem **muito mais tinta** que um distrator pontilhado sobrevivente — ela ocupa cada coluna, o distrator só uma fração. Então entre candidatos que já cobrem uma largura razoável (≥ 30 %), o desempate é por quantidade de tinta.

---

## Resultado medido

| Critério | Valor |
|---|---|
| G3b.1 · IoU mediana | 0,7153 |
| G3b.3 · importa sem torch | OK |
| G3b.4 · latência | mediana 12,2 ms, p95 29,0 ms |
| 2.10 · IoU vs. U-Net | clássico 0,7153 · U-Net 0,6482 |

> [!caution] O 2.10 compara pela métrica errada
> O clássico "ganha" no IoU porque produz máscaras **mais grossas**, e IoU numa curva fina mede espessura de traço. A comparação que decidiria a questão — erro perpendicular dos dois extratores — não é calculada em lugar nenhum. Ver [[Critérios de qualidade#O 2.10 mede pela métrica que o projeto demitiu]].

---

Ver também: [[identify.extract]] (a U-Net) · [[Critérios de qualidade]]
