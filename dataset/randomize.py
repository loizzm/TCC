"""Sorteio do estilo visual da figura (independente do sistema dinamico).

Regra estrutural anti-vazamento: `sample_style` NAO recebe o `SystemSpec`.
Ela fisicamente nao pode ver o rotulo, logo nenhum atributo visual pode ser
correlacionado a `order`, K, tau, theta, wn ou zeta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Vocabulario visual (nada aqui pode revelar ordem/amortecimento/parametros)
# --------------------------------------------------------------------------

NEUTRAL_TEXTS: tuple[str, ...] = (
    "Figura",
    "Serie A",
    "Serie B",
    "medicao",
    "canal 3",
    "amostra",
    "dados",
    "registro",
    "ensaio",
    "teste 2",
    "sinal",
    "aquisicao",
    "ponto",
    "item 4",
    "grupo B",
    "ref",
    "saida",
    "entrada",
    "valor",
    "coluna 1",
)

LINE_STYLES: tuple[str, ...] = ("-", "--", "-.", ":")
MARKERS: tuple[str, ...] = (".", "o", "s", "^", "x")
LEGEND_LOCS: tuple[str, ...] = (
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center right",
    "best",
)
TICK_DIRECTIONS: tuple[str, ...] = ("in", "out", "inout")
QUANT_LEVELS: tuple[int, ...] = (0, 64, 128, 256)

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# contraste minimo de luminancia entre curva e fundo (obrigatorio no contrato)
MIN_CONTRAST: float = 0.25
# contraste minimo entre eixos/texto e fundo (legibilidade)
MIN_AXES_CONTRAST: float = 0.20
# largura minima efetiva da linha, em PIXELS renderizados. Abaixo disso o
# anti-aliasing pode apagar a curva na mascara (limiar > 127).
MIN_LINE_PX: float = 1.5


# --------------------------------------------------------------------------
# Cores
# --------------------------------------------------------------------------


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(color: str | tuple[float, float, float]) -> float:
    """Luminancia relativa (WCAG) de uma cor hex '#rrggbb' ou tupla RGB 0-1."""
    if isinstance(color, str):
        s = color.lstrip("#")
        rgb = tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    else:
        rgb = tuple(float(c) for c in color[:3])
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(round(max(0.0, min(1.0, c)) * 255)):02x}" for c in rgb)


def _sample_color(rng: np.random.Generator, mode: str, lo: float, hi: float) -> str:
    """Sorteia uma cor. `mode` em {'gray','color'}; lo/hi limitam o brilho base."""
    v = float(rng.uniform(lo, hi))
    if mode == "gray":
        return _to_hex((v, v, v))
    base = rng.uniform(0.0, 1.0, size=3)
    span = float(base.max() - base.min())
    if span < 1e-9:
        base = np.array([v, v, v])
    else:
        base = (base - base.min()) / span
    # mistura entre a cor saturada e o brilho alvo
    mix = float(rng.uniform(0.35, 1.0))
    rgb = v * (mix * base + (1.0 - mix))
    rgb = np.clip(rgb + (v - float(np.mean(rgb))), 0.0, 1.0)
    return _to_hex(tuple(float(c) for c in rgb))


def _sample_palette(rng: np.random.Generator) -> tuple[str, str, str]:
    """(line_color, bg_color, axes_color) com contraste de luminancia garantido."""
    gray_mode = bool(rng.random() < 0.25)
    dark_bg = bool(rng.random() < 0.35)
    cmode = "gray" if gray_mode else "color"

    for _ in range(400):
        if dark_bg:
            bg = _sample_color(rng, cmode, 0.0, 0.30)
            line = _sample_color(rng, cmode, 0.45, 1.0)
        else:
            bg = _sample_color(rng, cmode, 0.75, 1.0)
            line = _sample_color(rng, cmode, 0.0, 0.60)
        if abs(luminance(line) - luminance(bg)) >= MIN_CONTRAST:
            break
    else:  # pragma: no cover - fallback determinista
        bg = "#000000" if dark_bg else "#ffffff"
        line = "#ffffff" if dark_bg else "#000000"

    lbg = luminance(bg)
    for _ in range(400):
        axes = _sample_color(rng, cmode, 0.0, 1.0)
        if abs(luminance(axes) - lbg) >= MIN_AXES_CONTRAST:
            break
    else:  # pragma: no cover
        axes = "#ffffff" if lbg < 0.5 else "#000000"
    return line, bg, axes


# --------------------------------------------------------------------------
# Textos
# --------------------------------------------------------------------------


def _sample_text(rng: np.random.Generator) -> str:
    """Texto semanticamente vazio: da lista neutra ou string aleatoria curta."""
    if rng.random() < 0.75:
        return NEUTRAL_TEXTS[int(rng.integers(0, len(NEUTRAL_TEXTS)))]
    n = int(rng.integers(3, 9))
    idx = rng.integers(0, len(_ALPHABET), size=n)
    return "".join(_ALPHABET[int(i)] for i in idx)


# --------------------------------------------------------------------------
# Estilo
# --------------------------------------------------------------------------


@dataclass
class RenderStyle:
    """Todos os atributos visuais sorteados de `rng_style`."""

    # geometria
    width_px: int = 640
    height_px: int = 480
    dpi: int = 100
    axes_rect: tuple[float, float, float, float] = (0.15, 0.15, 0.80, 0.78)
    # cores
    line_color: str = "#1f77b4"
    bg_color: str = "#ffffff"
    axes_color: str = "#000000"
    # curva
    line_width: float = 1.5
    line_style: str = "-"
    marker: str | None = None
    markevery: int = 40
    marker_size: float = 4.0
    # eixos
    has_grid: bool = False
    grid_alpha: float = 0.3
    grid_style: str = ":"
    has_major_ticks: bool = True
    has_minor_ticks: bool = False
    tick_direction: str = "out"
    tick_length: float = 3.0
    n_xbins: int = 5
    n_ybins: int = 5
    spines: tuple[bool, bool, bool, bool] = (True, True, False, False)
    x_margin_lo: float = 0.02
    x_margin_hi: float = 0.02
    y_margin_lo: float = 0.06
    y_margin_hi: float = 0.06
    # texto
    font_size: float = 9.0
    has_title: bool = False
    title_text: str = "Figura"
    has_xlabel: bool = False
    xlabel_text: str = "amostra"
    has_ylabel: bool = False
    ylabel_text: str = "valor"
    has_legend: bool = False
    legend_text: str = "Serie A"
    legend_loc: str = "best"
    legend_alpha: float = 0.8
    annotations: list[tuple[str, float, float]] = field(default_factory=list)
    # distratores (posicoes em FRACAO dos limites dos eixos)
    distractors: list[dict] = field(default_factory=list)
    # sinal
    snr_db: float = 40.0
    quantization_levels: int = 0

    # -- derivados --------------------------------------------------------
    @property
    def has_marker(self) -> bool:
        return self.marker is not None

    @property
    def n_annotations(self) -> int:
        return len(self.annotations)

    @property
    def n_distractors(self) -> int:
        return len(self.distractors)

    @property
    def n_spines(self) -> int:
        return int(sum(bool(s) for s in self.spines))

    @property
    def figsize(self) -> tuple[float, float]:
        return (self.width_px / self.dpi, self.height_px / self.dpi)

    def to_meta(self) -> dict:
        """Bloco `render` do meta.json (nunca entra em modelo: so estratificacao)."""
        return {
            "dpi": int(self.dpi),
            "size_px": [int(self.width_px), int(self.height_px)],
            "has_grid": bool(self.has_grid),
            "has_legend": bool(self.has_legend),
            "line_width": float(self.line_width),
            "line_style": str(self.line_style),
            "line_color": str(self.line_color),
            "bg_color": str(self.bg_color),
            "has_marker": bool(self.has_marker),
            "has_title": bool(self.has_title),
            "has_xlabel": bool(self.has_xlabel),
            "has_ylabel": bool(self.has_ylabel),
            "n_annotations": int(self.n_annotations),
            "n_distractors": int(self.n_distractors),
            "n_spines": int(self.n_spines),
            "snr_db": float(self.snr_db),
            "quantization_levels": int(self.quantization_levels),
        }


def sample_style(rng: np.random.Generator) -> RenderStyle:
    """Sorteia o estilo visual completo. NAO recebe o sistema: sem vazamento."""
    width_px = int(rng.integers(240, 1601))
    height_px = int(rng.integers(180, 1201))
    dpi = int(rng.integers(60, 201))

    line_color, bg_color, axes_color = _sample_palette(rng)

    # espessura em pontos; garante um minimo em pixels renderizados para que a
    # mascara (limiar > 127) nao fique com buracos de anti-aliasing.
    line_width = float(rng.uniform(0.8, 3.0))
    line_width = float(min(3.0, max(line_width, MIN_LINE_PX * 72.0 / dpi)))
    line_style = LINE_STYLES[int(rng.integers(0, len(LINE_STYLES)))]

    if rng.random() < 0.70:
        marker: str | None = None
    else:
        marker = MARKERS[int(rng.integers(0, len(MARKERS)))]
    markevery = int(rng.integers(20, 81))
    marker_size = float(rng.uniform(2.0, 7.0))

    left = float(rng.uniform(0.10, 0.22))
    bottom = float(rng.uniform(0.10, 0.22))
    right_pad = float(rng.uniform(0.03, 0.09))
    top_pad = float(rng.uniform(0.04, 0.12))
    axes_rect = (left, bottom, 1.0 - left - right_pad, 1.0 - bottom - top_pad)

    has_grid = bool(rng.random() < 0.5)
    grid_alpha = float(rng.uniform(0.15, 0.60))
    grid_style = LINE_STYLES[int(rng.integers(0, len(LINE_STYLES)))]
    has_major_ticks = bool(rng.random() < 0.90)
    has_minor_ticks = bool(rng.random() < 0.40)
    tick_direction = TICK_DIRECTIONS[int(rng.integers(0, len(TICK_DIRECTIONS)))]
    tick_length = float(rng.uniform(2.0, 5.0))
    n_xbins = int(rng.integers(3, 9))
    n_ybins = int(rng.integers(3, 9))
    # esquerda e inferior SEMPRE visiveis (moldura da area de dados, Estagio B)
    spines = (True, True, bool(rng.random() < 0.45), bool(rng.random() < 0.45))

    x_margin_lo = float(rng.uniform(0.01, 0.06))
    x_margin_hi = float(rng.uniform(0.01, 0.06))
    y_margin_lo = float(rng.uniform(0.03, 0.15))
    y_margin_hi = float(rng.uniform(0.03, 0.15))

    fig_min_in = min(width_px / dpi, height_px / dpi)
    font_size = (2.0 * fig_min_in + 3.0) * float(rng.uniform(0.85, 1.15))
    font_size = float(np.clip(font_size, 4.0, 15.0))

    has_title = bool(rng.random() < 0.5)
    title_text = _sample_text(rng)
    has_xlabel = bool(rng.random() < 0.5)
    xlabel_text = _sample_text(rng)
    has_ylabel = bool(rng.random() < 0.5)
    ylabel_text = _sample_text(rng)
    has_legend = bool(rng.random() < 0.5)
    legend_text = _sample_text(rng)
    legend_loc = LEGEND_LOCS[int(rng.integers(0, len(LEGEND_LOCS)))]
    legend_alpha = float(rng.uniform(0.4, 1.0))

    n_annotations = int(rng.integers(0, 3))
    annotations = [
        (_sample_text(rng), float(rng.uniform(0.05, 0.75)), float(rng.uniform(0.05, 0.92)))
        for _ in range(n_annotations)
    ]

    n_distractors = int(rng.integers(1, 4))
    distractors = []
    for _ in range(n_distractors):
        dcolor = _sample_color(rng, "gray" if rng.random() < 0.4 else "color", 0.0, 1.0)
        if abs(luminance(dcolor) - luminance(bg_color)) < 0.12:
            dcolor = axes_color
        distractors.append(
            {
                "orient": "h" if rng.random() < 0.5 else "v",
                "frac": float(rng.uniform(0.03, 0.97)),
                "color": dcolor,
                "line_style": LINE_STYLES[int(rng.integers(0, len(LINE_STYLES)))],
                "line_width": float(rng.uniform(0.5, 2.0)),
                "alpha": float(rng.uniform(0.30, 0.90)),
            }
        )

    snr_db = float(rng.uniform(20.0, 60.0))
    quantization_levels = int(QUANT_LEVELS[int(rng.integers(0, len(QUANT_LEVELS)))])

    return RenderStyle(
        width_px=width_px,
        height_px=height_px,
        dpi=dpi,
        axes_rect=axes_rect,
        line_color=line_color,
        bg_color=bg_color,
        axes_color=axes_color,
        line_width=line_width,
        line_style=line_style,
        marker=marker,
        markevery=markevery,
        marker_size=marker_size,
        has_grid=has_grid,
        grid_alpha=grid_alpha,
        grid_style=grid_style,
        has_major_ticks=has_major_ticks,
        has_minor_ticks=has_minor_ticks,
        tick_direction=tick_direction,
        tick_length=tick_length,
        n_xbins=n_xbins,
        n_ybins=n_ybins,
        spines=spines,
        x_margin_lo=x_margin_lo,
        x_margin_hi=x_margin_hi,
        y_margin_lo=y_margin_lo,
        y_margin_hi=y_margin_hi,
        font_size=font_size,
        has_title=has_title,
        title_text=title_text,
        has_xlabel=has_xlabel,
        xlabel_text=xlabel_text,
        has_ylabel=has_ylabel,
        ylabel_text=ylabel_text,
        has_legend=has_legend,
        legend_text=legend_text,
        legend_loc=legend_loc,
        legend_alpha=legend_alpha,
        annotations=annotations,
        distractors=distractors,
        snr_db=snr_db,
        quantization_levels=quantization_levels,
    )
