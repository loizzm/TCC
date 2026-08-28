"""Gerador saneado: sorteia sistema + estilo, renderiza imagem, mascara e meta.

Sem vazamento de rotulo: o estilo visual vem de um stream de RNG independente
do stream que sorteia o sistema dinamico (ver `randomize.sample_style`).
Determinismo bit-a-bit: mesma seed => mesmos bytes de image.png e mask.png.
A igualdade de BYTES so e garantida dentro de um ambiente pinado (ver
`requirements.txt`): versoes diferentes de matplotlib/pillow/libpng podem mudar
o rasterizador ou o encoder PNG. A igualdade dos VALORES (series, params, meta)
independe do ambiente.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, LinearLocator, MaxNLocator  # noqa: E402
from PIL import Image  # noqa: E402

from dataset.randomize import RenderStyle, sample_style  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

SCHEMA_VERSION: int = 1

ORDERS: tuple[str, str] = ("fopdt", "second")
N_SERIES: int = 512
STEP_AMPLITUDE: float = 1.0


# --------------------------------------------------------------------------
# Sistema dinamico
# --------------------------------------------------------------------------


def dominant_time_constant(
    order: str, tau: float | None, wn: float | None, zeta: float | None
) -> float:
    """Constante de tempo dominante (contrato §1, Ruling K). Fonte unica.

    fopdt              -> tau
    second, zeta <= 1  -> 1/(zeta*wn)
    second, zeta > 1   -> 1/(wn*(zeta - sqrt(zeta^2-1))), na forma racionalizada
                          (zeta + sqrt(zeta^2-1))/wn, que evita o cancelamento
                          catastrofico para zeta grande. Continua em zeta=1.
    """
    if order == "fopdt":
        return float(tau)
    zeta = float(zeta)
    wn = float(wn)
    if zeta <= 1.0:
        return float(1.0 / (zeta * wn))
    return float((zeta + np.sqrt(zeta * zeta - 1.0)) / wn)


@dataclass(frozen=True)
class SystemSpec:
    """Sistema sorteado e janela de observacao."""

    order: str
    K: float
    tau: float | None
    theta: float
    wn: float | None
    zeta: float | None
    t_start: float
    t_end: float
    step_amplitude: float

    @property
    def t_dom(self) -> float:
        """Constante de tempo dominante (delega para `dominant_time_constant`)."""
        return dominant_time_constant(self.order, self.tau, self.wn, self.zeta)


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


# Quantas constantes de tempo dominantes o estrato `reta_no_patamar` garante
# DEPOIS do tempo morto. `sample_system` sorteia a janela em
# `loguniform(0.5, 6.0) * t_dom`, e assentar a 1% leva ~4.6 t_dom — por isso o
# corpus praticamente nunca mostra o patamar: medido, a fracao final ja
# assentada tem MEDIANA de 0,98% da janela, e ZERO de 60 amostras chegam a 30%.
# As duas imagens reais do Ruling 55 usam 10 e ~21 t_dom, com ~50% da janela
# assentada. Sem patamar visivel a reta de referencia toca a curva so na ultima
# coluna, e nao ha trecho colinear para ocluir — foi o que a primeira tentativa
# deste estrato mediu (cobertura mediana 0,96, criterio pede < 0,75).
# Valor escolhido por VARREDURA medida (n=40 por ponto), tendo como alvo o
# maior vao de colunas sem tinta predita das duas imagens reais: 0,1669
# (Figure_1) e 0,3802 (resposta_degrau).
#   T_DOM   mediana     p90      max    >=0.15   >=0.20
#     9.2    0.0434   0.2710   0.4319    10/40     6/40
#    14.0    0.0479   0.4902   0.5744    11/40     9/40
#    20.0    0.0501   0.6007   0.7095    12/40     9/40   <- escolhido
#    30.0    0.0543   0.6724   0.8270    12/40    10/40
#    45.0    0.0516   0.6469   0.8907    14/40    12/40
# 20 casa com a janela das imagens reais (10 e ~21 t_dom). Acima disso o maximo
# vai a 0,83-0,89, FORA da faixa real, e a mediana nao sobe: os bloqueadores
# medidos sao ruido (Spearman SNR x vao +0,423, p=7e-4) e marcador (-0,346,
# p=7e-3), que fazem a curva escapar de tras da reta. Sao variacao legitima de
# estilo, entao o estrato exibe o fenomeno com forca em ~30% das amostras e
# isso e o correto — suprimi-los deixaria o estrato limpo demais para ser real.
_T_DOM_ESTRATO = 20.0


def sample_system(rng: np.random.Generator) -> SystemSpec:
    """Sorteia (order, K, tau|wn/zeta, theta) e a janela temporal."""
    order = ORDERS[int(rng.integers(0, 2))]
    K = _loguniform(rng, 0.2, 20.0)
    if order == "fopdt":
        tau: float | None = _loguniform(rng, 0.05, 50.0)
        wn: float | None = None
        zeta: float | None = None
    else:
        tau = None
        wn = _loguniform(rng, 0.02, 20.0)
        zeta = float(rng.uniform(0.10, 3.00))
    t_dom = dominant_time_constant(order, tau, wn, zeta)
    theta = _loguniform(rng, 0.05, 1.0) * t_dom
    t_end = theta + _loguniform(rng, 0.5, 6.0) * t_dom
    return SystemSpec(
        order=order,
        K=K,
        tau=tau,
        theta=float(theta),
        wn=wn,
        zeta=zeta,
        t_start=0.0,
        t_end=float(t_end),
        step_amplitude=STEP_AMPLITUDE,
    )


def step_response(spec: SystemSpec, t: np.ndarray) -> np.ndarray:
    """Resposta analitica limpa ao degrau (contrato §1). Vetorizada."""
    t = np.asarray(t, dtype=float)
    u = t - spec.theta
    active = u >= 0.0
    uu = np.where(active, u, 0.0)
    s = spec.step_amplitude * spec.K

    if spec.order == "fopdt":
        y = s * (1.0 - np.exp(-uu / float(spec.tau)))
    else:
        wn = float(spec.wn)
        zeta = float(spec.zeta)
        if abs(zeta - 1.0) <= 1e-6:
            y = s * (1.0 - np.exp(-wn * uu) * (1.0 + wn * uu))
        elif zeta < 1.0:
            wd = wn * np.sqrt(1.0 - zeta * zeta)
            y = s * (
                1.0
                - np.exp(-zeta * wn * uu)
                * (np.cos(wd * uu) + (zeta * wn / wd) * np.sin(wd * uu))
            )
        else:
            rad = np.sqrt(zeta * zeta - 1.0)
            r1 = wn * (-zeta + rad)
            r2 = wn * (-zeta - rad)
            y = s * (1.0 + (r2 * np.exp(r1 * uu) - r1 * np.exp(r2 * uu)) / (r1 - r2))

    return np.where(active, y, 0.0)


# --------------------------------------------------------------------------
# Ruido e quantizacao
# --------------------------------------------------------------------------


def _apply_noise(y: np.ndarray, style: RenderStyle, rng: np.random.Generator) -> np.ndarray:
    """Ruido gaussiano aditivo (SNR em dB sobre a variancia do sinal) + quantizacao."""
    y_out = np.asarray(y, dtype=float).copy()
    var = float(np.var(y_out))
    if var > 0.0 and np.isfinite(var):
        sigma = float(np.sqrt(var / (10.0 ** (style.snr_db / 10.0))))
        y_out = y_out + sigma * rng.standard_normal(y_out.size)

    levels = int(style.quantization_levels)
    if levels > 1:
        lo = float(np.min(y_out))
        hi = float(np.max(y_out))
        span = hi - lo
        if span > 1e-15:
            q = np.round((y_out - lo) / span * (levels - 1))
            y_out = lo + q * span / (levels - 1)
    return y_out


# --------------------------------------------------------------------------
# Renderizacao
# --------------------------------------------------------------------------


def _axis_limits(
    t: np.ndarray, y: np.ndarray, style: RenderStyle
) -> tuple[tuple[float, float], tuple[float, float]]:
    t0, t1 = float(np.min(t)), float(np.max(t))
    span_t = t1 - t0
    if span_t <= 0.0:
        span_t = max(abs(t1), 1.0)
    xlim = (t0 - style.x_margin_lo * span_t, t1 + style.x_margin_hi * span_t)

    y0, y1 = float(np.min(y)), float(np.max(y))
    span_y = y1 - y0
    if span_y <= 1e-12:
        span_y = max(abs(y1), 1.0) * 0.1
    ylim = (y0 - style.y_margin_lo * span_y, y1 + style.y_margin_hi * span_y)
    return xlim, ylim


def _new_figure(style: RenderStyle, facecolor: str) -> tuple[Figure, Axes]:
    fig = Figure(figsize=style.figsize, dpi=style.dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(facecolor)
    ax = fig.add_axes(list(style.axes_rect))
    ax.set_facecolor(facecolor)
    return fig, ax


# Luminancia ITU-R BT.601, a MESMA de `identify/extract.py::predict_mask`.
_PESO_CINZA = (0.299, 0.587, 0.114)


def _cinza(rgb) -> float:
    """Luminancia que a U-Net enxerga: ela recebe 1 canal, nao RGB."""
    return sum(p * c for p, c in zip(_PESO_CINZA, rgb))


def _cor_colidente(hex_curva: str) -> str:
    """Cor com a MESMA luminancia da curva e matiz oposta.

    O estagio A converte a imagem para cinza antes da U-Net
    (`identify/extract.py::predict_mask`), entao dois objetos de luminancia
    igual chegam a rede como o MESMO byte — medido nas duas imagens reais do
    Ruling 55: curva (44,160,44) e reta de referencia (230,61,61) viram ambas
    112. Separar um do outro deixa de ser dificil e passa a ser impossivel:
    nenhuma funcao de uma entrada distingue pontos onde a entrada e identica.

    O estrato precisa reproduzir ISSO, e nao apenas "existe uma reta". Uma cor
    sorteada ao acaso colide raramente (medido no gerador: 2,05% dos
    distratores ficam a menos de 2 bytes da curva) e, pior, colide sem se
    SOBREPOR — e a medicao mostrou que colisao sem sobreposicao e inofensiva
    (Spearman colisao x IoU = -0,006, p=0,86, n=900). Aqui a colisao e
    construida de proposito e combinada com sobreposicao no patamar.

    Depende so de `style.line_color`, que `sample_style` sorteia cego ao spec,
    entao nao abre caminho de vazamento (tests/test_part1.py:1115).
    """
    import colorsys
    rgb = tuple(int(hex_curva[i:i + 2], 16) for i in (1, 3, 5))
    alvo = _cinza(rgb)
    h, ll, s = colorsys.rgb_to_hls(*[c / 255.0 for c in rgb])
    girado = colorsys.hls_to_rgb((h + 0.5) % 1.0, ll, s)
    lg = _cinza([c * 255.0 for c in girado])
    if lg <= 1e-6:
        return hex_curva
    # reescala para casar a luminancia; se estourar, dessatura ate caber.
    for _ in range(24):
        k = alvo / lg
        cand = [c * 255.0 * k for c in girado]
        if max(cand) <= 255.0:
            return "#%02x%02x%02x" % tuple(int(round(min(255.0, max(0.0, c)))) for c in cand)
        s *= 0.85
        girado = colorsys.hls_to_rgb((h + 0.5) % 1.0, ll, s)
        lg = _cinza([c * 255.0 for c in girado])
        if lg <= 1e-6:
            break
    return hex_curva

def _plot_curve(ax, t: np.ndarray, y: np.ndarray, style: RenderStyle, color: str, label=None):
    kwargs = dict(
        color=color,
        linewidth=style.line_width,
        linestyle=style.line_style,
        solid_capstyle="round",
        antialiased=True,
    )
    if style.marker is not None:
        kwargs.update(
            marker=style.marker,
            markevery=style.markevery,
            markersize=style.marker_size,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=max(0.5, style.line_width * 0.6),
        )
    if label is not None:
        kwargs["label"] = label
    return ax.plot(t, y, **kwargs)


def _apply_locators(ax, style: RenderStyle, xlim, ylim) -> None:
    # limita o numero de ticks ao tamanho fisico do eixo (evita rotulos
    # sobrepostos em figuras pequenas). Depende so do estilo, nunca do rotulo.
    w_in = style.axes_rect[2] * style.figsize[0]
    h_in = style.axes_rect[3] * style.figsize[1]
    nx = int(np.clip(min(style.n_xbins, int(2.2 * w_in)), 3, 8))
    ny = int(np.clip(min(style.n_ybins, int(3.0 * h_in)), 3, 8))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nx, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ny, min_n_ticks=3))
    # hipotese §1.6.2: pelo menos 2 ticks maiores rotulados por eixo
    for axis, lim in ((ax.xaxis, xlim), (ax.yaxis, ylim)):
        vals = np.asarray(axis.get_major_locator()())
        inside = vals[(vals >= min(lim) - 1e-12) & (vals <= max(lim) + 1e-12)]
        if inside.size < 2:
            axis.set_major_locator(LinearLocator(3))


def _ticks_in_view(ax, axis_name: str, lim, height_px: int) -> list[list[float]]:
    """Ticks maiores visiveis dentro dos limites, em indices de pixel da imagem.

    Devolve [[coord_px, valor], ...]: coordenada horizontal para o eixo x e
    vertical (origem no topo) para o eixo y.
    """
    lo, hi = min(lim), max(lim)
    vals = ax.get_xticks() if axis_name == "x" else ax.get_yticks()
    out: list[list[float]] = []
    for v in np.asarray(vals, dtype=float):
        if not (lo - 1e-12 <= v <= hi + 1e-12):
            continue
        if axis_name == "x":
            coord = float(ax.transData.transform((v, lo))[0]) - 0.5
        else:
            py_mpl = float(ax.transData.transform((lo, v))[1])
            coord = float(height_px - py_mpl) - 0.5
        out.append([coord, float(v)])
    return out


def render_sample(
    spec: SystemSpec,
    style: RenderStyle,
    out_dir: str | Path,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
    *,
    seed: int | None = None,
    reta_no_patamar: bool = False,
) -> dict:
    """Escreve image.png, mask.png e meta.json em out_dir. Devolve o dict do meta."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rng is None:
        rng = np.random.default_rng(0)

    # Marca o estrato OOD numa COPIA do estilo (via `dataclasses.replace`),
    # nunca mutando o objeto `style` recebido do chamador: `render_sample` nao
    # e o dono desse objeto, e mutar um argumento e efeito colateral
    # observavel para quem chamou. `has_reference_line` e campo de RENDER
    # (dataset/randomize.py), nao sorteado por `sample_style`.
    style = replace(style, has_reference_line=bool(reta_no_patamar))

    t = np.linspace(spec.t_start, spec.t_end, N_SERIES)
    y_clean = step_response(spec, t)
    y_draw = _apply_noise(y_clean, style, rng) if add_noise else y_clean.copy()

    xlim, ylim = _axis_limits(t, y_draw, style)

    # ---------------- figura de verdade ----------------
    fig, ax = _new_figure(style, style.bg_color)
    _plot_curve(ax, t, y_draw, style, style.line_color,
                label=style.legend_text if style.has_legend else None)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _apply_locators(ax, style, xlim, ylim)

    # distratores: posicoes uniformes nos limites dos eixos, sem relacao com o rotulo
    distractors = list(style.distractors)
    if reta_no_patamar:
        # Estrato OOD, opt-in (HANDOFF_P2_7 §34.5). O laco acima sorteia
        # posicao uniforme; aqui se acrescenta uma reta horizontal FIXADA no
        # patamar da curva, que e o caso real do setpoint marcado e onde a
        # U-Net perde a curva. Entra no RENDER e nao em `sample_style`, porque
        # a posicao depende do sistema e `sample_style` tem de continuar cega
        # ao spec (tests/test_part1.py:1115). Sob `if`, para que o caminho
        # padrao nao mude um byte.
        # `line_style` tracejado ("--"), de proposito: e o traco real de um
        # setpoint marcado (a linha do caso real que motivou este estrato), e
        # e o caso DIFICIL: um traco tracejado fragmenta em varios blocos
        # separados por coluna, exatamente a condicao multi-bloco que o
        # extrator de polilinha precisa resolver. Solido seria mais facil de
        # extrair e menos fiel ao problema real — nao serviria de estrato de
        # teste para o defeito do §34.5.
        distractors = distractors + [
            {"orient": "h", "frac": None, "no_patamar": True,
             # Cor de LUMINANCIA IGUAL a da curva (ver `_cor_colidente`): em
             # cinza, que e o que a U-Net recebe, os dois objetos viram o mesmo
             # byte. Sem isso o estrato so testa "existe uma reta", e a medicao
             # mostrou que reta de cor qualquer nao degrada nada
             # (Spearman colisao x IoU = -0,006, p=0,86, n=900).
             "color": _cor_colidente(style.line_color), "line_style": "--",
             # ACIMA da curva: os distratores comuns ficam em `zorder=1`, sob
             # ela, e por isso nao ocluem. O fenomeno real do Ruling 55 e a
             # reta passar POR CIMA do trecho assentado — e a mistura
             # antialiasada dos dois que apaga o contraste (medido: cai a 32%
             # do original) e faz a mascara perder a curva.
             "zorder": 3,
             # Espessura acompanhando a da curva: uma reta mais fina deixaria
             # borda de curva visivel dos dois lados e nao ocluiria de fato.
             "line_width": max(2.0, style.line_width * 1.8), "alpha": 1.0}
        ]
    for d in distractors:
        if d["orient"] == "h":
            if d.get("no_patamar"):
                # SETPOINT comandado (`K * degrau`), nao `y_draw[-1]` nem
                # `y_clean[-1]`. Tres razoes, nesta ordem:
                #  - e o que um grafico real marca: nas duas imagens do
                #    Ruling 55 a reta esta exatamente em 1,0, o valor de
                #    referencia, nao "onde a curva calhou de terminar";
                #  - `y_draw[-1]` e uma amostra RUIDOSA, e a reta ancorada
                #    nela cobre a curva so em parte (medido: 12 px de desvio
                #    numa seed, e a oclusao vira parcial);
                #  - `y_clean[-1]` ainda oscila quando a janela nao basta para
                #    assentar (subamortecido), e ai a reta sai do patamar.
                # E derivavel do meta (`params.K` x `step_amplitude`), entao o
                # estrato fica localizavel sem chave nova no contrato.
                val = float(spec.K * spec.step_amplitude)
            else:
                val = ylim[0] + d["frac"] * (ylim[1] - ylim[0])
            ax.axhline(
                val, color=d["color"], linestyle=d["line_style"],
                linewidth=d["line_width"], alpha=d["alpha"],
                zorder=d.get("zorder", 1),
            )
        else:
            val = xlim[0] + d["frac"] * (xlim[1] - xlim[0])
            ax.axvline(
                val, color=d["color"], linestyle=d["line_style"],
                linewidth=d["line_width"], alpha=d["alpha"],
                zorder=d.get("zorder", 1),
            )

    if style.has_grid:
        ax.grid(True, color=style.axes_color, alpha=style.grid_alpha,
                linestyle=style.grid_style, linewidth=0.6, zorder=0)
    if style.has_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="minor", direction=style.tick_direction,
                       length=style.tick_length * 0.5, colors=style.axes_color)
    ax.tick_params(
        which="major",
        direction=style.tick_direction,
        length=style.tick_length if style.has_major_ticks else 0.0,
        colors=style.axes_color,
        labelsize=style.font_size * 0.85,
    )
    for name, visible in zip(("left", "bottom", "right", "top"), style.spines):
        ax.spines[name].set_visible(bool(visible))
        ax.spines[name].set_color(style.axes_color)

    if style.has_title:
        ax.set_title(style.title_text, color=style.axes_color, fontsize=style.font_size)
    if style.has_xlabel:
        ax.set_xlabel(style.xlabel_text, color=style.axes_color, fontsize=style.font_size)
    if style.has_ylabel:
        ax.set_ylabel(style.ylabel_text, color=style.axes_color, fontsize=style.font_size)
    if style.has_legend:
        leg = ax.legend(loc=style.legend_loc, fontsize=style.font_size * 0.8,
                        framealpha=style.legend_alpha)
        leg.get_frame().set_facecolor(style.bg_color)
        leg.get_frame().set_edgecolor(style.axes_color)
        for txt in leg.get_texts():
            txt.set_color(style.axes_color)
    for text, fx, fy in style.annotations:
        ax.text(fx, fy, text, transform=ax.transAxes, color=style.axes_color,
                fontsize=style.font_size * 0.8, zorder=5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    h_px, w_px = int(buf.shape[0]), int(buf.shape[1])

    # calibracao a partir da figura de verdade (nunca "chutada").
    # matplotlib usa display coords com origem no canto INFERIOR esquerdo; a
    # imagem usa indices de pixel com origem no canto SUPERIOR esquerdo e o
    # centro do pixel i na coordenada continua i+0.5 -> subtrai-se 0.5.
    p = ax.transData.transform(np.array([[xlim[0], ylim[0]], [xlim[1], ylim[1]]]))
    sx = float((xlim[1] - xlim[0]) / (p[1, 0] - p[0, 0]))
    ox = float(xlim[0] - sx * (p[0, 0] - 0.5))
    sy = float(-(ylim[1] - ylim[0]) / (p[1, 1] - p[0, 1]))
    oy = float(ylim[0] - sy * (h_px - p[0, 1] - 0.5))

    # `plot_bbox_px` delimita a AREA DE DADOS (o retangulo dos eixos), em
    # indices de pixel, [x0, y0, x1, y1] com origem no canto superior esquerdo.
    # Convencao: os limites sao o CENTRO da spine desenhada (mesma convencao de
    # indice usada pela afim), logo o retangulo fica ~0.5 px "para dentro" da
    # borda externa do traco da moldura. Nao alterar: a afim foi validada
    # geometricamente contra a mascara com essa convencao.
    bb = ax.get_window_extent()
    plot_bbox_px = [
        int(round(bb.x0 - 0.5)),
        int(round(h_px - bb.y1 - 0.5)),
        int(round(bb.x1 - 0.5)),
        int(round(h_px - bb.y0 - 0.5)),
    ]
    ticks = {
        "x": _ticks_in_view(ax, "x", xlim, h_px),
        "y": _ticks_in_view(ax, "y", ylim, h_px),
    }

    fig.savefig(out / "image.png", dpi=style.dpi, facecolor=style.bg_color,
                metadata={"Software": None})

    # ---------------- figura-mascara (mesma geometria) ----------------
    mfig, max_ = _new_figure(style, "#000000")
    _plot_curve(max_, t, y_draw, style, "#ffffff")
    max_.set_xlim(*xlim)
    max_.set_ylim(*ylim)
    max_.set_axis_off()
    mfig.canvas.draw()
    mbuf = np.asarray(mfig.canvas.buffer_rgba())
    if mbuf.shape[:2] != (h_px, w_px):  # pragma: no cover
        raise RuntimeError("geometria da mascara difere da imagem")
    mask = np.where(mbuf[:, :, 0] > 127, 255, 0).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(out / "mask.png", optimize=False)

    fig.clf()
    mfig.clf()

    with Image.open(out / "image.png") as im:
        img_size = im.size  # (W, H)
    if img_size != (w_px, h_px):  # pragma: no cover
        raise RuntimeError(f"image.png {img_size} != canvas {(w_px, h_px)}")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": out.name,
        "seed": seed,
        "order": spec.order,
        "params": {
            "K": float(spec.K),
            "tau": None if spec.tau is None else float(spec.tau),
            "theta": float(spec.theta),
            "wn": None if spec.wn is None else float(spec.wn),
            "zeta": None if spec.zeta is None else float(spec.zeta),
        },
        "step_amplitude": float(spec.step_amplitude),
        "t_window": [float(spec.t_start), float(spec.t_end)],
        "plot_bbox_px": plot_bbox_px,
        "axis_affine": {"sx": sx, "ox": ox, "sy": sy, "oy": oy},
        "ticks": ticks,
        "series": {"t": [float(v) for v in t], "y": [float(v) for v in y_draw]},
        "noise": {
            "enabled": bool(add_noise),
            "snr_db": float(style.snr_db),
            "quantization_levels": int(style.quantization_levels),
        },
        "render": style.to_meta(),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    return meta


# --------------------------------------------------------------------------
# API de alto nivel
# --------------------------------------------------------------------------


def generate_sample(out_dir: str | Path, seed: int, add_noise: bool = True,
                    reta_no_patamar: bool = False,
                    janela_assentada: bool = False) -> dict:
    """Sorteia sistema+estilo com streams independentes, renderiza e devolve o meta."""
    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(3)
    rng_sys = np.random.default_rng(children[0])
    rng_style = np.random.default_rng(children[1])
    rng_noise = np.random.default_rng(children[2])

    spec = sample_system(rng_sys)
    style = sample_style(rng_style)  # nao ve o spec: anti-vazamento estrutural
    if janela_assentada:
        # Janela longa o bastante para o PATAMAR ficar visivel. Eixo separado de
        # `reta_no_patamar` de proposito: sao dois fenomenos distintos e o
        # estrato OOD e a combinacao dos dois. Separados, dao ablacao (a reta
        # sozinha? a janela sozinha?) e permitem controle pareado no teste —
        # com os dois no mesmo parametro, a extensao muda a geometria da curva
        # e nenhum diferencial pixel a pixel e possivel.
        # Nao mexe no meta: a janela ja e observavel em `t_window`.
        t_dom = dominant_time_constant(spec.order, spec.tau, spec.wn, spec.zeta)
        spec = replace(spec, t_end=float(max(spec.t_end,
                                             spec.theta + _T_DOM_ESTRATO * t_dom)))
    return render_sample(spec, style, out_dir, add_noise=add_noise, rng=rng_noise,
                         seed=int(seed), reta_no_patamar=reta_no_patamar)


def _generate_one(args: tuple) -> str:
    out_dir, seed, add_noise, reta, janela = args
    generate_sample(out_dir, seed, add_noise=add_noise, reta_no_patamar=reta,
                    janela_assentada=janela)
    return str(out_dir)


def generate_dataset(
    out_dir: str | Path,
    n: int,
    seed: int = 0,
    workers: int | None = None,
    add_noise: bool = True,
    reta_no_patamar: bool = False,
    janela_assentada: bool = False,
) -> list[str]:
    """Gera n amostras em paralelo. Resultado independe do numero de workers."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(root / f"sample_{i:05d}"), int(seed) * 1_000_003 + i, bool(add_noise),
         bool(reta_no_patamar), bool(janela_assentada))
        for i in range(int(n))
    ]
    if workers is not None and workers <= 1:
        return [_generate_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_generate_one, jobs, chunksize=4))


def load_sample(sample_dir: str | Path) -> dict:
    """Le meta.json (series como np.ndarray) e anexa 'image' e 'mask' uint8."""
    d = Path(sample_dir)
    with open(d / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["series"] = {
        "t": np.asarray(meta["series"]["t"], dtype=float),
        "y": np.asarray(meta["series"]["y"], dtype=float),
    }
    with Image.open(d / "image.png") as im:
        meta["image"] = np.asarray(im.convert("RGB"), dtype=np.uint8)
    with Image.open(d / "mask.png") as im:
        meta["mask"] = np.asarray(im.convert("L"), dtype=np.uint8)
    return meta


if __name__ == "__main__":  # pragma: no cover
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    sd = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    dirs = generate_dataset(out, n, seed=sd, workers=os.cpu_count())
    print(f"{len(dirs)} amostras em {out}")
