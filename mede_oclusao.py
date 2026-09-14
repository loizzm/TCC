"""Mede EXATAMENTE quanto da curva a caixa da legenda tapa.

Por que um medidor novo. A primeira tentativa mediu "pixel da curva que nao
esta na cor da curva" e deu mediana 0,12 ate no corpus PADRAO — numero que nao
e' oclusao, e' anti-aliasing, quantizacao e sobreposicao com a grade. O segundo
mediu contra a cor de FUNDO e falhou pelo lado oposto: `framealpha` varia entre
0,4 e 1,0, entao a caixa translucida nao pinta o fundo puro.

O instrumento correto nao depende de cor nenhuma. `_new_figure` usa
`add_axes` com retangulo FIXO e `savefig` nao usa `bbox_inches`, logo a
legenda NAO move os eixos: renderizando a MESMA amostra com e sem legenda,
os pixels que diferem sao a pegada da caixa, exatamente. Cruzando essa pegada
com `mask.png` (que e' desenhada numa figura separada, sem legenda, e por isso
e' identica nas duas) sai a fracao dos pixels da curva que a legenda cobre.

    oclusao = |{img_com != img_sem} & {mask}| / |{mask}|

Uso:
    .venv/bin/python mede_oclusao.py [--n 120] [--seed 0]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from dataset.generator import (generate_sample, render_sample, sample_style,
                               sample_system)

RAIZ = Path(__file__).resolve().parent


def _rngs(seed: int):
    filhos = np.random.SeedSequence(int(seed)).spawn(3)
    return (np.random.default_rng(filhos[0]), np.random.default_rng(filhos[1]),
            np.random.default_rng(filhos[2]))


def _le(dst: Path):
    img = np.asarray(Image.open(dst / "image.png").convert("RGB"), dtype=np.int16)
    mask = np.asarray(Image.open(dst / "mask.png").convert("L")) > 127
    return img, mask


def _render(spec, style, dst: Path, seed: int, rng, **kw):
    render_sample(spec, style, dst, add_noise=True, rng=rng, seed=seed, **kw)
    return _le(dst)


def oclusao_de_um(seed: int, tmp: Path) -> tuple[float | None, float] | None:
    """(oclusao no padrao, oclusao com o estrato).

    O primeiro e' None quando `sample_style` nao sorteou legenda — o estrato
    FORCA a legenda, entao ele mede sempre, mas o padrao so tem numero nas
    amostras que de fato desenham uma.
    """
    rng_sys, rng_style, _ = _rngs(seed)
    spec = sample_system(rng_sys)
    style = sample_style(rng_style)

    # REFERENCIA sem legenda nenhuma. `rng_noise` e' recriado do seed a cada
    # render, e nenhum dos flags de legenda o consome antes da curva, entao as
    # tres figuras tem a MESMA curva bit a bit — a unica diferenca no PNG e' a
    # caixa. As duas variantes passam por `generate_sample` (e nao direto por
    # `render_sample`) porque o alargamento do rotulo vive la.
    sem, mask = _render(spec, replace(style, has_legend=False),
                        tmp / "sem", seed, _rngs(seed)[2])
    generate_sample(tmp / "ocl", seed=seed, legenda_oclusora=True)
    ocl, _ = _le(tmp / "ocl")

    n = int(mask.sum())
    if n == 0:
        return None
    f = lambda a: float(((np.abs(a - sem).sum(axis=2) > 0) & mask).sum()) / n
    o = f(ocl), *_vao(ocl, sem, mask)
    if not style.has_legend:
        return None, o
    generate_sample(tmp / "pad", seed=seed)
    pad, _ = _le(tmp / "pad")
    return f(pad), o


def _vao(img, sem, mask) -> tuple[float, float]:
    """(largura da caixa, colunas da curva no vao dela) — as duas em fracao da
    LARGURA DA CURVA. A largura e' a variavel causal do defeito (borda
    horizontal longa lida como patamar), entao ela e' medida, nao suposta."""
    fc = np.flatnonzero((np.abs(img - sem).sum(axis=2) > 0).any(axis=0))
    cc = np.flatnonzero(mask.any(axis=0))
    if not fc.size or cc.size < 2:
        return 0.0, 0.0
    larg = (fc[-1] - fc[0]) / (cc[-1] - cc[0])
    return float(larg), float(((cc >= fc[0]) & (cc <= fc[-1])).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="oclusao_"))
    pad, ocl, larg, colu = [], [], [], []
    try:
        for i in range(a.n):
            r = oclusao_de_um(int(a.seed) * 1_000_003 + i, tmp)
            if r is None:
                continue
            if r[0] is not None:
                pad.append(r[0])
            ocl.append(r[1][0])
            larg.append(r[1][1])
            colu.append(r[1][2])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    p, o = np.array(pad), np.array(ocl)
    print("OCLUSAO DA CURVA PELA CAIXA DA LEGENDA")
    print(f"  padrao: n={p.size} (so as que sortearam legenda) | "
          f"oclusora: n={o.size} (o estrato forca)\n")
    print(f"  {'variante':<14}{'p50':>9}{'p90':>9}{'max':>9}"
          f"{'> 1 %':>9}{'> 5 %':>9}{'> 10 %':>9}")
    for nome, v in (("padrao", p), ("oclusora", o)):
        if not v.size:
            continue
        print(f"  {nome:<14}{np.median(v):>9.4f}{np.percentile(v,90):>9.4f}"
              f"{v.max():>9.4f}{(v>0.01).mean():>9.1%}{(v>0.05).mean():>9.1%}"
              f"{(v>0.10).mean():>9.1%}")
    L, C = np.array(larg), np.array(colu)
    print(f"\n  GEOMETRIA DA CAIXA no estrato (fracao da largura da curva)")
    print(f"    largura da caixa      p50={np.median(L):.2f}  p90={np.percentile(L,90):.2f}")
    print(f"    colunas no vao dela   p50={np.median(C):.2f}  p90={np.percentile(C,90):.2f}")
    print(f"\n  ALVO REAL — `tests/fixtures/caso_real_neg_super.png`, a figura que")
    print(f"  o retreino existe para consertar: a caixa ocupa 0,43 da largura,")
    print(f"  tapa 0,48 das colunas e 0,1505 dos pixels da curva. O estrato tem")
    print(f"  de BRACAR esse ponto, nao empatar com ele.")


if __name__ == "__main__":
    main()
