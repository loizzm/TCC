"""TERCEIRA FAMÍLIA DE RENDER (§73) — e o contrato de ISOLAMENTO dela.

POR QUE ELA EXISTE. A rede treina no render de `dataset/generator.py` e é
avaliada no de `rg_aleatorio.py`. A lacuna entre as duas famílias foi medida
com o mesmo instrumento dos dois lados — a fração das colunas em que a máscara
acende sobre a curva verdadeira: mediana **0,713** na avaliação contra **0,925**
no treino, Mann-Whitney p = 1,05e-12, com 79 % das figuras de avaliação abaixo
de 80 % de cobertura contra 29 % das de treino. Nenhum outro fator medido neste
estrato custa metade disso.

O CONTRATO QUE ESTE ARQUIVO GUARDA é o de poder VOLTAR ATRÁS. Um estrato que se
mistura com os anteriores não é reversível: para desfazê-lo seria preciso
reidentificar amostra por amostra qual veio de onde. Daí as quatro asserções:

  1. sem o flag, o corpus base não muda um byte;
  2. o meta do estrato DECLARA a família, e guarda o NOME DO TEMA — não um
     booleano —, então as amostras se separam por leitura de meta;
  3. a MÁSCARA e a FÍSICA são as mesmas do seed sem o flag, o que torna o
     estrato comparável amostra a amostra com o base;
  4. o corpus base não ganha chave nenhuma no meta.

A 3 é a mais delicada e por isso tem teste próprio. `matplotlib.rcParams` é
GLOBAL e por processo, e o `rg_aleatorio._estilo` documenta o estrago de
deixá-lo vazar: um estilo zerou `axes.linewidth`, o seguinte não restaurou, e
29 de 33 figuras saíram sem moldura — o que a calibração lê como
`bbox_not_found`. Aqui o tema é aplicado só à figura de verdade e restaurado
antes da figura-máscara.

O QUE ESTE ARQUIVO **NÃO** AFIRMA: que o estrato fecha a lacuna. Medido na
geração, `data/val_render2` tem cobertura mediana 0,903 contra 0,944 de
`data/val` (p = 0,23) — ou seja, ele é uma VARIAÇÃO da família 1, e continua a
19 pontos da família de avaliação. Ver o comentário de `_TEMAS_ALT`.
"""
import json
import tempfile
from pathlib import Path

from dataset.generator import (SCHEMA_VERSION, _TEMAS_ALT, generate_sample,
                               load_sample)

SEEDS = (4242, 11, 202, 3003, 40004)
SEED_BASE = 941_002_823          # primeira semente de `data/train_render2`


def _meta_sem_id(d: Path) -> dict:
    m = json.loads((d / "meta.json").read_text())
    m.pop("sample_id", None)
    return m


def test_o_padrao_nao_muda_um_byte():
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            a, b = Path(tmp) / f"a{seed}", Path(tmp) / f"b{seed}"
            generate_sample(str(a), seed=seed)
            generate_sample(str(b), seed=seed, familia_alt=False)
            assert (a / "image.png").read_bytes() == (b / "image.png").read_bytes()
            assert (a / "mask.png").read_bytes() == (b / "mask.png").read_bytes()
            assert _meta_sem_id(a) == _meta_sem_id(b)


def test_o_meta_do_base_nao_ganha_chave_nenhuma():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "base"
        generate_sample(str(d), seed=4242)
        m = json.loads((d / "meta.json").read_text())
        assert "familia_alt" not in m["render"]
        assert "companheiro" not in m["render"]
        assert m["schema_version"] == SCHEMA_VERSION


def test_o_meta_declara_a_familia_pelo_NOME():
    """Nome do tema, não booleano: é o que permite separar as amostras depois
    sem reidentificar uma a uma — o requisito de conseguir voltar atrás."""
    with tempfile.TemporaryDirectory() as tmp:
        vistos = set()
        for i in range(12):
            d = Path(tmp) / f"s{i}"
            m = generate_sample(str(d), seed=SEED_BASE + i, familia_alt=True)
            tema = m["render"]["familia_alt"]
            assert tema in _TEMAS_ALT
            assert isinstance(m["render"]["companheiro"], bool)
            vistos.add(tema)
        assert len(vistos) >= 3, f"sorteio de tema degenerado: {vistos}"


def test_nao_copia_a_familia_de_AVALIACAO():
    """Treinar no render do `rg_aleatorio` transformaria os lotes de controle em
    memorização de família e invalidaria todo número da Parte 2."""
    proibidos = {"seaborn-v0_8-darkgrid", "dark_background"}
    assert not (set(_TEMAS_ALT) & proibidos)


def test_a_mascara_e_a_fisica_nao_mudam():
    """O tema mexe SÓ na figura de verdade. `rcParams` é global e por processo —
    ver a docstring do módulo para o estrago que vazar já causou neste projeto."""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            a, b = Path(tmp) / f"a{seed}", Path(tmp) / f"b{seed}"
            base = generate_sample(str(a), seed=seed)
            alt = generate_sample(str(b), seed=seed, familia_alt=True)
            assert (a / "mask.png").read_bytes() == (b / "mask.png").read_bytes(), (
                f"a máscara mudou (seed {seed}): o tema vazou para a figura-máscara")
            assert base["series"]["y"] == alt["series"]["y"]
            assert base["axis_affine"] == alt["axis_affine"]
            assert base["plot_bbox_px"] == alt["plot_bbox_px"]
            assert base["params"] == alt["params"]


def test_a_imagem_muda_de_verdade():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        generate_sample(str(a), seed=4242)
        generate_sample(str(b), seed=4242, familia_alt=True)
        assert (a / "image.png").read_bytes() != (b / "image.png").read_bytes()


def test_o_tema_nao_vaza_entre_amostras_seguidas():
    """Duas amostras do estrato seguidas de uma do BASE: a do base tem de sair
    idêntica à que sairia isolada. É o modo de falha exato que o
    `rg_aleatorio._estilo` registra (29 de 33 figuras sem moldura)."""
    with tempfile.TemporaryDirectory() as tmp:
        sozinha = Path(tmp) / "sozinha"
        generate_sample(str(sozinha), seed=4242)
        generate_sample(str(Path(tmp) / "x0"), seed=SEED_BASE, familia_alt=True)
        generate_sample(str(Path(tmp) / "x1"), seed=SEED_BASE + 1, familia_alt=True)
        depois = Path(tmp) / "depois"
        generate_sample(str(depois), seed=4242)
        assert (sozinha / "image.png").read_bytes() == (depois / "image.png").read_bytes()
        assert (sozinha / "mask.png").read_bytes() == (depois / "mask.png").read_bytes()


def test_o_corpus_gerado_se_declara_inteiro():
    """Toda amostra de `data/train_render2` carrega a chave. Sem isso, separar
    as famílias depois exigiria reidentificação amostra a amostra."""
    import pytest
    d = Path(__file__).resolve().parents[2] / "data" / "val_render2"
    if not d.exists():
        pytest.skip("corpus ausente (gere com generate_dataset(..., familia_alt=True))")
    amostras = sorted(d.glob("sample_*"))[:60]
    assert amostras
    for s in amostras:
        m = json.loads((s / "meta.json").read_text())
        assert m["render"].get("familia_alt") in _TEMAS_ALT
