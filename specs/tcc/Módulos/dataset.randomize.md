---
tags: [modulo, dataset, anti-vazamento]
aliases: [randomize.py, dataset/randomize.py, RenderStyle]
---

# `dataset/randomize.py`

> **396 linhas.** Sorteia **todo** o estilo visual de uma figura. Depende só de `numpy`.

O módulo mais curto do projeto com a restrição de projeto mais forte.

---

## A restrição que define o módulo

```python
def sample_style(rng: np.random.Generator) -> RenderStyle:
    """Sorteia o estilo visual completo. NAO recebe o sistema: sem vazamento."""
```

**A assinatura é `(rng)` e nada mais.** A função fisicamente não pode ver a `SystemSpec`, logo nenhum atributo visual pode se correlacionar com ordem, K, τ, θ, ωn ou ζ.

> [!important] Por que isso é estrutural e não convenção
> Sem essa separação, a rede poderia aprender atalhos — "figuras com grade tendem a ser de 1ª ordem" — e obter acurácia alta no corpus que **desaparece no mundo real**. A restrição não depende de disciplina de quem escreve o código: um teste assevera que a assinatura continua sendo exatamente `["rng"]`.
>
> Ver [[Decisões de projeto#D7 · O estilo não pode ver o sistema]].

---

## `RenderStyle`

Um dataclass com ~35 campos, agrupados por finalidade:

```python
@dataclass
class RenderStyle:
    # geometria
    width_px: int = 640;  height_px: int = 480;  dpi: int = 100
    axes_rect: tuple[float, float, float, float] = (0.15, 0.15, 0.80, 0.78)
    # cores
    line_color: str = "#1f77b4";  bg_color: str = "#ffffff";  axes_color: str = "#000000"
    # curva
    line_width: float = 1.5;  line_style: str = "-"
    marker: str | None = None;  markevery: int = 40;  marker_size: float = 4.0
    # eixos
    has_grid: bool = False;  grid_alpha: float = 0.3;  grid_style: str = ":"
    has_major_ticks: bool = True;  has_minor_ticks: bool = False
    tick_direction: str = "out";  tick_length: float = 3.0
    n_xbins: int = 5;  n_ybins: int = 5
    spines: tuple[bool, bool, bool, bool] = (True, True, False, False)
    # texto
    font_size: float = 9.0
    has_title / has_xlabel / has_ylabel / has_legend: bool = False
    annotations: list[tuple[str, float, float]]
    # distratores
    distractors: list[dict]
    # sinal
    snr_db: float = 40.0;  quantization_levels: int = 0
```

### Faixas sorteadas

| Atributo | Faixa |
|---|---|
| `width_px` × `height_px` | 240–1600 × 180–1200 |
| `dpi` | 60–200 |
| `line_width` | 0,8–3,0 pt, com piso em pixels |
| `line_style` | `-`, `--`, `-.`, `:` |
| `marker` | 30 % de chance de ter |
| `has_grid` | 50 % |
| `has_major_ticks` | 90 % |
| `has_minor_ticks` | 40 % |

### Os três campos que `sample_style` NUNCA toca

```python
has_reference_line: bool = False
has_annotation_arrow: bool = False
has_settling_band: bool = False
```

São **campos de render**, não sorteados. Quem os marca é `render_sample` via `dataclasses.replace`, quando o estrato correspondente é pedido. Manter isso fora do sorteio é o que preserva a assinatura `(rng)` — e portanto o teste anti-vazamento.

São **três campos e não um** porque são fenômenos separáveis: a caixa com seta desvia a máscara para uma estrutura *acima* da curva; a banda sombreada apaga o contraste *no patamar*. Separados dão ablação; juntos reproduzem o caso real.

---

## Contraste garantido por construção

```python
MIN_CONTRAST: float = 0.25        # curva x fundo (obrigatório no contrato)
MIN_AXES_CONTRAST: float = 0.20   # eixos/texto x fundo (legibilidade)
MIN_LINE_PX: float = 1.5          # largura mínima em pixels RENDERIZADOS
```

A luminância é a **relativa da WCAG**, não a média ingênua dos canais:

```python
def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def luminance(color) -> float:
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
```

A paleta é sorteada por **rejeição**, com fallback determinista:

```python
def _sample_palette(rng) -> tuple[str, str, str]:
    gray_mode = bool(rng.random() < 0.25)
    dark_bg   = bool(rng.random() < 0.35)      # 35% das figuras têm fundo escuro

    for _ in range(400):
        if dark_bg:
            bg = _sample_color(rng, cmode, 0.0, 0.30);  line = _sample_color(rng, cmode, 0.45, 1.0)
        else:
            bg = _sample_color(rng, cmode, 0.75, 1.0);  line = _sample_color(rng, cmode, 0.0, 0.60)
        if abs(luminance(line) - luminance(bg)) >= MIN_CONTRAST:
            break
    else:
        bg, line = ("#000000", "#ffffff") if dark_bg else ("#ffffff", "#000000")
```

O `else` do `for` nunca dispara na prática, mas garante que **o sorteio sempre termina** — um loop de rejeição sem saída seria não-determinismo disfarçado.

O piso de largura de linha existe porque abaixo dele o anti-aliasing pode **apagar a curva na máscara** (limiar > 127):

```python
line_width = float(min(3.0, max(line_width, MIN_LINE_PX * 72.0 / dpi)))
```

---

## Texto semanticamente vazio

```python
def _sample_text(rng) -> str:
    """Texto semanticamente vazio: da lista neutra ou string aleatoria curta."""
    if rng.random() < 0.75:
        return NEUTRAL_TEXTS[int(rng.integers(0, len(NEUTRAL_TEXTS)))]
    n = int(rng.integers(3, 9))
    return "".join(_ALPHABET[int(i)] for i in rng.integers(0, len(_ALPHABET), size=n))
```

Títulos e legendas não podem descrever o sistema — se dissessem "resposta de 1ª ordem", o rótulo estaria escrito na figura. Vazio semanticamente, presente visualmente: é exatamente o distrator que se quer.

---

## `to_meta()` — o conjunto FECHADO

```python
def to_meta(self) -> dict:
    """Bloco `render` do meta.json (nunca entra em modelo: so estratificacao)."""
    return {"dpi": ..., "size_px": ..., "has_grid": ..., "line_style": ...,
            "n_distractors": ..., "snr_db": ...,
            "has_reference_line": ..., "has_annotation_arrow": ..., "has_settling_band": ...}
```

O teste `tests/test_part1.py::_RENDER_KEYS` compara esse conjunto de chaves com `==`, não com `⊆`. Acrescentar um campo ao `RenderStyle` **obriga** a decidir conscientemente se ele entra ou não no `meta` — não há como esquecer.

O bloco `render` serve **só para estratificação** das métricas. Nunca entra em modelo nenhum: se entrasse, seria o vazamento que o módulo inteiro existe para impedir.

---

## Propriedades derivadas

```python
@property def has_marker(self)    -> bool:  return self.marker is not None
@property def n_annotations(self) -> int:   return len(self.annotations)
@property def n_distractors(self) -> int:   return len(self.distractors)
@property def n_spines(self)      -> int:   return int(sum(bool(s) for s in self.spines))
@property def figsize(self) -> tuple[float, float]:
    return (self.width_px / self.dpi, self.height_px / self.dpi)
```

---

Ver também: [[dataset.generator]] · [[Decisões de projeto#D7 · O estilo não pode ver o sistema]]
