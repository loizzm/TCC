"""Estrato opt-in LEGENDA OCLUSORA. Molde: `ruido_alto` (§69) — mexe no ESTILO.

POR QUE ELE EXISTE. O par controlado `caso_real_neg_super.png` x
`caso_real_neg_super_legenda_movida.png` isola uma variável e só uma: a posição
da caixa da legenda. Com ela em 'lower left', atravessando a faixa de
acomodação, `wn` e `zeta` saem com 26 % e 30 % de erro; com a MESMA imagem e a
legenda em 'upper right', 2,97 % e 2,23 %. O mecanismo é a rede seguir a BORDA
HORIZONTAL da caixa como se fosse patamar: isso antecipa a acomodação, e o
ajuste compensa com polo dominante mais lento e menos amortecimento.

POR QUE O CORPUS NÃO ENSINAVA ISSO. Medido em 120 amostras com o instrumento de
`mede_oclusao.py` (diferença de render, exata — `add_axes` é retângulo fixo e
`savefig` não usa `bbox_inches`, então a legenda não move os eixos): no corpus
base a caixa tapa MEDIANA 0,0000 dos pixels da curva, e passa de 1 % em só
14,5 % das amostras. `loc="best"` do matplotlib procura ativamente o espaço
LIVRE, e as outras cinco posições de `LEGEND_LOCS` são cantos. O corpus nunca
produziu o defeito, então a rede nunca teve como aprendê-lo.

DUAS VARIÁVEIS, NÃO UMA. Posição e LARGURA. A borda longa é o que se parece com
patamar, e o rótulo curto que `sample_style` sorteia dava caixa de 0,17 da
largura da curva contra 0,43 da figura real. Por isso o estrato também alarga o
rótulo (`_OCLUSAO_N_TEXTOS`), com texto do MESMO gerador neutro do corpus base —
o que muda é o comprimento, não a semântica.

ONDE O ESTRATO TEM DE CAIR. Bracejando a figura real, não empatando com ela:
medido, caixa 0,26/0,54 (p50/p90) contra 0,43 real; colunas tapadas 0,26/0,52
contra 0,48; pixels 0,078/0,186 contra 0,1505.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset.generator import (SCHEMA_VERSION, _OCLUSAO_CHEGADA, _OCLUSAO_FOLGA,
                               _OCLUSAO_JOELHO, generate_sample, load_sample)

SEEDS = (4242, 11, 202, 3003, 40004)


def _meta_sem_id(d: Path) -> dict:
    m = json.loads((d / "meta.json").read_text())
    m.pop("sample_id", None)
    return m


def test_o_padrao_nao_muda_um_byte():
    """Sem o flag, a amostra é idêntica à de antes desta mudança."""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            a, b = Path(tmp) / f"a{seed}", Path(tmp) / f"b{seed}"
            generate_sample(str(a), seed=seed)
            generate_sample(str(b), seed=seed, legenda_oclusora=False)
            assert (a / "image.png").read_bytes() == (b / "image.png").read_bytes()
            assert (a / "mask.png").read_bytes() == (b / "mask.png").read_bytes()
            assert _meta_sem_id(a) == _meta_sem_id(b)


def test_o_meta_do_base_nao_ganha_chave_nenhuma():
    """A chave nova é CONDICIONAL. `render` já carrega três booleanos de
    estrato que saem SEMPRE; um quarto mudaria os bytes de todo meta.json do
    corpus base só para dizer `false`."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "base"
        generate_sample(str(d), seed=4242)
        m = json.loads((d / "meta.json").read_text())
        assert "legenda_oclusora" not in m["render"]
        assert m["schema_version"] == SCHEMA_VERSION


def test_o_meta_do_estrato_se_declara():
    """Sem esta chave o estrato seria invisível: `render.has_legend` fica
    `true`, mas isso também vale em metade do corpus base."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ocl"
        generate_sample(str(d), seed=4242, legenda_oclusora=True)
        m = json.loads((d / "meta.json").read_text())
        assert m["render"]["legenda_oclusora"] is True
        assert m["render"]["has_legend"] is True


def test_a_legenda_e_forcada_em_toda_amostra():
    """`sample_style` só sorteia legenda em metade. Sem forçar, metade do
    corpus novo sairia byte a byte igual ao base — não seria material novo,
    seria o base com peso dobrado."""
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(24):
            d = Path(tmp) / f"s{i}"
            generate_sample(str(d), seed=931_002_793 + i, legenda_oclusora=True)
            assert load_sample(d)["render"]["has_legend"] is True


def test_a_curva_e_a_mascara_nao_mudam():
    """O estrato mexe SÓ na legenda. A física, o enquadramento e a máscara são
    os mesmos do seed sem o flag — é o que torna o estrato comparável amostra a
    amostra com o base, e o que garante que a verdade não se desloca."""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            a, b = Path(tmp) / f"a{seed}", Path(tmp) / f"b{seed}"
            base = generate_sample(str(a), seed=seed)
            ocl = generate_sample(str(b), seed=seed, legenda_oclusora=True)
            assert (a / "mask.png").read_bytes() == (b / "mask.png").read_bytes()
            assert base["series"]["y"] == ocl["series"]["y"]
            assert base["axis_affine"] == ocl["axis_affine"]
            assert base["plot_bbox_px"] == ocl["plot_bbox_px"]


def _oclusao(seed: int, tmp: Path) -> float:
    """Fração dos pixels da curva que a caixa cobre, pelo diferencial exato de
    `mede_oclusao.py`: a legenda não move os eixos, então tudo que difere entre
    o render com e sem ela É a pegada da caixa."""
    from dataclasses import replace

    from dataset.generator import render_sample, sample_style, sample_system

    f = np.random.SeedSequence(int(seed)).spawn(3)
    spec = sample_system(np.random.default_rng(f[0]))
    style = sample_style(np.random.default_rng(f[1]))
    render_sample(spec, replace(style, has_legend=False), tmp / f"sem{seed}",
                  add_noise=True, rng=np.random.default_rng(f[2]), seed=seed)
    generate_sample(tmp / f"ocl{seed}", seed=seed, legenda_oclusora=True)
    le = lambda d: np.asarray(Image.open(tmp / d / "image.png").convert("RGB"),
                              dtype=np.int16)
    sem, ocl = le(f"sem{seed}"), le(f"ocl{seed}")
    mask = np.asarray(Image.open(tmp / f"sem{seed}" / "mask.png").convert("L")) > 127
    n = int(mask.sum())
    assert n > 0
    return float(((np.abs(ocl - sem).sum(axis=2) > 0) & mask).sum()) / n


def test_a_caixa_realmente_tapa_a_curva():
    """O teste que a primeira implementação teria passado por engano. A métrica
    ingênua ('pixel da curva fora da cor da curva') dava 0,12 de mediana ATÉ no
    corpus padrão, porque conta anti-aliasing e grade. Esta mede a pegada
    exata da caixa."""
    with tempfile.TemporaryDirectory() as tmp:
        v = np.array([_oclusao(931_002_793 + i, Path(tmp)) for i in range(16)])
        # NÃO exige 100 %. A âncora RASPA a borda do patamar em vez de engolir a
        # curva — que é a geometria da figura real, medida: lá a curva cai
        # dentro da faixa vertical da caixa em só 28,7 % das colunas. Exigir que
        # toda amostra tape seria exigir a geometria ERRADA, a que o teste
        # pareado mostrou não reproduzir o defeito (79,1 % contra 78,3 %).
        # Medido em 120 amostras com a âncora atual: p50 0,0592, p90 0,1327,
        # máximo 0,2734, e 72,5 % acima de 1 %.
        assert (v > 0.01).mean() >= 0.60, (
            f"só {(v > 0.01).mean():.0%} das amostras tapam mais de 1 %")
        assert np.median(v) > 0.03, f"mediana {np.median(v):.4f} baixa demais"
        assert v.max() > 0.1505, (
            f"o estrato não alcança a figura real (0,1505): máximo {v.max():.4f}")


def _caixa_e_chegada(seed: int, tmp: Path) -> tuple[int, int, float]:
    """(x0, x1) da caixa e a coluna da CHEGADA ao patamar, na mesma figura.

    A pegada da caixa sai do diferencial contra o render SEM legenda — o mesmo
    de `_oclusao`, e sem ambiguidade: ali só existe uma caixa.
    """
    from dataclasses import replace

    from dataset.generator import render_sample, sample_style, sample_system

    f = np.random.SeedSequence(int(seed)).spawn(3)
    spec = sample_system(np.random.default_rng(f[0]))
    style = sample_style(np.random.default_rng(f[1]))
    render_sample(spec, replace(style, has_legend=False), tmp / f"s{seed}",
                  add_noise=True, rng=np.random.default_rng(f[2]), seed=seed)
    meta = generate_sample(tmp / f"o{seed}", seed=seed, legenda_oclusora=True)
    le = lambda d: np.asarray(Image.open(tmp / d / "image.png").convert("RGB"),
                              dtype=np.int16)
    dif = np.abs(le(f"o{seed}") - le(f"s{seed}")).sum(axis=2) > 0
    cols = np.nonzero(dif.any(axis=0))[0]
    assert cols.size, "o estrato não desenhou caixa nenhuma"

    af = meta["axis_affine"]
    ys = np.asarray(meta["series"]["y"], dtype=float)
    ts = np.asarray(meta["series"]["t"], dtype=float)
    rep = float(np.median(ys[: max(int(0.02 * ys.size), 3)]))
    fim = float(np.median(ys[max(int(0.8 * ys.size), 1) :]))
    s = float(np.sign(fim - rep)) or 1.0
    k = np.nonzero((ys - rep) * s >= _OCLUSAO_CHEGADA * abs(fim - rep))[0]
    t_arr = float(ts[k[0]]) if k.size else float(ts[-1])
    return int(cols.min()), int(cols.max()), (t_arr - af["ox"]) / af["sx"]


def test_a_caixa_cai_sobre_a_transicao_e_se_estende_para_tras():
    """O DESFECHO da âncora, não a intenção dela.

    A versão anterior deste teste conferia só os NÚMEROS da constante — que a
    âncora ficasse entre 0,5 e 4,0 `t_dom` depois de `theta` — e ficou verde
    enquanto a caixa caía, medida nos 200 pares de `val_parleg_*`, a 298 px
    (p50) do joelho, com ele DENTRO da caixa em 3,1 % das amostras. Asseverar a
    intenção não asseverou nada, e o estrato inteiro foi treinado em cima disso.

    A geometria que precisa valer, lida da figura real: a caixa cobre a CHEGADA
    ao patamar e se estende PARA TRÁS sobre o transitório e o tempo morto —
    lá, 254 px de largura para um transitório de 33 px, com a borda de cima
    correndo 14 px abaixo do patamar. É essa borda longa que a rede segue como
    se fosse patamar.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dentro = tras = 0
        for seed in SEEDS + (777, 8080, 91, 1234, 56789, 31337):
            x0, x1, xc = _caixa_e_chegada(seed, Path(tmp))
            dentro += int(x0 <= xc <= x1)
            tras += int(x0 < xc)
        n = len(SEEDS) + 6
        assert dentro >= n - 1, (
            f"a chegada caiu dentro da caixa em só {dentro}/{n} amostras")
        assert tras == n, (
            f"a caixa se estende para a FRENTE da chegada em {n - tras}/{n}: "
            "não cobre o transitório que vem antes")


def test_as_constantes_da_ancora_sao_coerentes():
    """`_OCLUSAO_JOELHO` é MEIA-LARGURA da caixa, não `t_dom`: a caixa é objeto
    de layout e `t_dom` é escala da dinâmica, e foi essa conversão inexistente
    que pôs a caixa no lugar errado. Com o teto abaixo de 1 a chegada fica
    dentro da caixa por construção."""
    lo, hi = _OCLUSAO_JOELHO
    assert 0.0 < lo < hi < 1.0, "acima de 1 meia-largura a chegada sai da caixa"
    assert 0.5 < _OCLUSAO_CHEGADA < 1.0, "a chegada é fração da excursão"
    a, b = _OCLUSAO_FOLGA
    assert 0.0 < a < b <= 0.10, "a folga é em fração do eixo, não em pixels"
