"""Estrato opt-in FORA DA FAMILIA: fase não-mínima (§65). Molde: `ganho_negativo`.

É o primeiro estrato que sai da família de modelos do Estágio D de propósito.
Todos os outros — `reta_no_patamar`, `ganho_negativo`, `multi_degrau`,
`plato_no_meio` — variam render, sinal, entrada ou enquadramento de curvas que
CONTINUAM sendo FOPDT ou 2ª ordem. Este muda a planta: multiplica-a por
`(1 - a·s)`, um zero no semiplano direito.

POR QUE ELE EXISTE. O Estágio A aprendeu, por acidente do corpus, que curva de
dados é "aproximação monótona a um patamar" — nenhuma outra forma existia para
ele ver. Medido em `data/val_nmp` contra `data/val` com o modelo promovido, a
cobertura do platô de repouso cai de 0,884 para 0,286 (60,7 % das amostras
abaixo de metade, contra 8,6 %), enquanto o recall global da máscara mal se
mexe (0,784 -> 0,729): a rede segmenta o trecho que se parece com o que ela
conhece e larga o resto. Nenhum estrato DENTRO da família produz essa
geometria — em FOPDT e 2ª ordem a resposta é contínua em `t = θ` e vale zero
ali, igual ao repouso, então não há descontinuidade para ensinar.

POR QUE `order` NÃO GANHA UM TERCEIRO VALOR. O zero não mexe nos polos:
`order`, `tau`/`wn`/`zeta` e portanto `t_dom` continuam descrevendo a mesma
dinâmica de polos e continuam corretos no meta. O que deixa de valer é tratar
`params` como um modelo da CURVA INTEIRA, e é isso que a chave
`fora_da_familia` diz. Um terceiro valor em `order` faria três estragos: todo
`if order == "fopdt": ... else: <supõe second>` passaria a rotear errado em
silêncio; a matriz de confusão dos relatórios ganharia uma linha que é 100 %
erro por construção, porque o ajustador não pode predizer um rótulo que não
está no espaço dele; e a estrutura de polos, que é real e é útil, seria
jogada fora.

O QUE OS ORÁCULOS FAZEM COM ELE. Não pulam — INVERTEM. Para uma amostra fora
da família o comportamento certo da pipeline não é ajustar bem, é RECUSAR com
`resposta_inversa`. Pular deixaria um buraco silencioso justamente na guarda
mais frágil do sistema; inverter transforma o estrato em suíte de regressão
dela. Ver `test_a_pipeline_recusa_com_o_motivo_certo`, e em especial a
exigência de que a MÁSCARA TENHA ACHADO A CURVA antes: sem isso o teste
passaria hoje pelo motivo errado (máscara vazia também produz recusa) e
continuaria verde mesmo se o retreino não resolvesse nada.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dataset.generator import (SCHEMA_VERSION, SCHEMA_VERSION_FORA_DA_FAMILIA,
                               _NMP_MERGULHO, generate_sample, load_sample)

SEEDS = (4242, 11, 202, 3003, 40004)


def _meta_sem_id(d: Path) -> dict:
    m = json.loads((d / "meta.json").read_text())
    m.pop("sample_id", None)
    return m


def _gera(tmp, seed, **kw):
    d = Path(tmp) / f"s{seed}_{int(kw.get('fase_nao_minima', False))}"
    generate_sample(str(d), seed=seed, **kw)
    return load_sample(d)


def test_o_padrao_nao_muda_um_byte():
    """Sem o flag, a amostra é idêntica à de antes desta mudança."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"; b = Path(tmp) / "b"
        generate_sample(str(a), seed=4242)
        generate_sample(str(b), seed=4242, fase_nao_minima=False)
        assert (a / "image.png").read_bytes() == (b / "image.png").read_bytes()
        assert (a / "mask.png").read_bytes() == (b / "mask.png").read_bytes()
        assert _meta_sem_id(a) == _meta_sem_id(b)


def test_o_meta_do_base_nao_ganha_chave_nenhuma():
    """As chaves novas são CONDICIONAIS. Se saíssem sempre, todo meta.json do
    corpus base mudaria de bytes para dizer `false`/`null`."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _gera(tmp, 4242)
        assert "fora_da_familia" not in m
        assert "a" not in m["params"]
        assert m["schema_version"] == SCHEMA_VERSION


def test_o_meta_do_estrato_se_declara():
    with tempfile.TemporaryDirectory() as tmp:
        m = _gera(tmp, 4242, fase_nao_minima=True)
        assert m["fora_da_familia"] is True
        assert m["schema_version"] == SCHEMA_VERSION_FORA_DA_FAMILIA
        assert m["params"]["a"] > 0.0


def test_os_polos_continuam_descritos_e_intactos():
    """`order` e os parâmetros de polo são os MESMOS do seed sem o flag — é o
    que torna o estrato comparável amostra a amostra com o base, e o que
    justifica não inventar um terceiro valor para `order`."""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            base = _gera(tmp, seed)
            nmp = _gera(tmp, seed, fase_nao_minima=True)
            assert nmp["order"] == base["order"] in ("fopdt", "second")
            for k in ("K", "tau", "wn", "zeta", "theta"):
                assert nmp["params"][k] == base["params"][k], f"{k} mudou (seed {seed})"


def test_a_curva_mergulha_antes_de_subir():
    """A assinatura de fase não-mínima, na série VERDADEIRA, com a mesma
    aritmética que o gerador usa para calibrar `a`."""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            m = _gera(tmp, seed, fase_nao_minima=True)
            y = np.asarray(m["series"]["y"], dtype=float)
            d = 1.0 if m["params"]["K"] >= 0 else -1.0
            merg = -float(np.min(y * d)) / float(np.ptp(y))
            assert _NMP_MERGULHO[0] - 0.02 <= merg <= _NMP_MERGULHO[1] + 0.02, (
                f"mergulho {merg:.3f} fora da faixa sorteada (seed {seed})")


def test_a_curva_volta_e_assenta_acima_do_repouso():
    """Sem isto o estrato produz figuras que são só um decaimento monótono:
    a curva mergulha, cruza o zero e o quadro acaba antes de ela voltar. Nessas
    `_undershoot` lê 0,000 e ACERTA — recusar seria acertar pelo motivo errado.
    Foi medido em 2 de 24 antes de `_NMP_T_DOM_MIN` existir."""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            m = _gera(tmp, seed, fase_nao_minima=True)
            y = np.asarray(m["series"]["y"], dtype=float)
            d = 1.0 if m["params"]["K"] >= 0 else -1.0
            repouso = float(np.median(y[:5])) * d
            final = float(np.median(y[-max(1, y.size // 10):])) * d
            assert final > repouso, f"a curva não volta acima do repouso (seed {seed})"


def test_a_soma_com_ganho_negativo_inverte_o_mergulho():
    """Os dois flags compõem: com K < 0 a curva sobe antes de descer. Sem isto
    o estrato ensinaria a rede uma assinatura de mão única."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _gera(tmp, 4242, fase_nao_minima=True, ganho_negativo=True)
        y = np.asarray(m["series"]["y"], dtype=float)
        assert m["params"]["K"] < 0.0
        assert float(np.max(y)) > 0.0, "com K < 0 o mergulho tem de ser para CIMA"
        assert float(np.median(y[-max(1, y.size // 10):])) < 0.0


# --------------------------------------------------------------------------
# Oráculo INVERTIDO: aqui a pipeline tem de RECUSAR, e pelo motivo certo
# --------------------------------------------------------------------------
_CORPUS = Path(__file__).resolve().parents[2] / "data" / "val_nmp"
_MODELO = Path(__file__).resolve().parents[2] / "models" / "unet_stageA.pt"
_N_AMOSTRAS = 60

# Mínimo de pontos para dizer que a MÁSCARA ACHOU A CURVA. Sem este piso o
# teste passaria pelo motivo errado: máscara vazia também produz recusa, e o
# teste ficaria verde hoje e continuaria verde se o retreino não resolvesse
# nada. `data/val` tem mediana de 734 pontos; 200 é folgado o bastante para
# não brigar com variação de largura de figura e apertado o bastante para
# reprovar uma máscara que só pegou um pedaço.
_PONTOS_MIN = 200

# Fração de `data/val_nmp` que a pipeline recusa por `resposta_inversa`.
# MEDIDA, não desejada.
#
# HISTÓRICO, porque ele é o argumento do estrato:
#   base 32 antes do retreino ......  66/120 = 55,0 %   (piso era 0,50)
#   época 17, promovida ............  58/60  = 96,7 %   (piso agora 0,90)
# (a época 16 dava 98,3 % aqui e foi PRETERIDA: ela pagava o ganho com a cauda
#  assentada — cobertura p10 do último quinto caía de 0,593 para 0,124 em
#  `data/train_janela`, e fazia a figura de controle de um degrau truncar. A 17
#  entrega o mesmo ganho em figura real com a cauda intacta. Ver §66.)
# O retreino com o estrato `fase_nao_minima` entregou o que se pediu dele: o
# teto era a MÁSCARA, não o limiar — antes, 34 das 49 que escapavam tinham
# undershoot extraído abaixo de 0,02, ou seja, o mergulho nem chegava na
# guarda. Agora chega.
#
# ATENÇÃO ao que este número NÃO diz. Ele é medido no render de TREINO. Nas
# 40 figuras de fase não-mínima com render do `rg_aleatorio` a mesma rede
# recusa 50 % e ainda devolve 40 % de resposta confiante e errada. A lacuna
# render–física continua aberta e é o próximo trabalho; este piso protege o
# ganho, não o declara completo.
_RECUSA_MIN = 0.90


@pytest.fixture(scope="module")
def _pipeline():
    torch = pytest.importorskip("torch")
    if not _MODELO.exists():
        pytest.skip(f"checkpoint ausente: {_MODELO}")
    if not _CORPUS.exists():
        pytest.skip(f"corpus ausente: {_CORPUS} "
                    "(gere com generate_dataset(..., fase_nao_minima=True))")
    from PIL import Image

    from identify.extract import load_model
    from identify.pipeline import identify_from_image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(_MODELO), dev)
    modelo.eval()
    out = []
    for s in sorted(_CORPUS.glob("sample_*"))[:_N_AMOSTRAS]:
        img = np.asarray(Image.open(s / "image.png").convert("RGB"))
        out.append((s.name, identify_from_image(img, modelo, dev)))
    return out


# Motivos que vêm do ESTÁGIO B (calibração de eixos) e não do ajuste. Eles têm
# taxa de fundo própria e nada a ver com a família de modelos: medida, ela é de
# 4/57 (7,0 %) em `data/val_nmp` e 8/120 (6,7 %) em `data/val` — a mesma. Uma
# figura cujo OCR falhou nunca chega na guarda, então incluí-la aqui mediria o
# OCR fingindo medir a guarda.
_MOTIVOS_DE_CALIBRACAO = frozenset({
    "bbox_not_found", "ocr_insuficiente", "ransac_failed",
    "calibration_failed", "sinal_de_escala_invalido", "polilinha_curta",
})


def test_a_pipeline_recusa_com_o_motivo_certo(_pipeline):
    """Quem chega no ajuste ou sai `ok` ou sai `resposta_inversa`. Nada mais.

    Dois filtros, e os dois são o teste. `_PONTOS_MIN` separa "acertou" de
    "acertou pelo motivo errado": máscara vazia também produz recusa, e sem o
    piso este teste ficaria verde hoje e continuaria verde se o retreino não
    resolvesse nada. `_MOTIVOS_DE_CALIBRACAO` tira o que falhou antes do
    ajuste, que é ruído de outra etapa.

    O que sobra e REPROVA é a recusa com diagnóstico errado — `ajuste_
    inconsistente` numa figura de fase não-mínima, por exemplo: a guarda
    específica deixou passar, o resíduo alto pegou, e o usuário recebe a causa
    errada. Sair `ok` NÃO reprova aqui; isso é o teto da máscara, e quem mede
    é `test_a_taxa_de_recusa_nao_regride`.
    """
    achou = [(n, r) for n, r in _pipeline
             if r["n_points"] >= _PONTOS_MIN
             and (r["reason"] or "") not in _MOTIVOS_DE_CALIBRACAO]
    assert achou, "a máscara não achou a curva em nenhuma amostra do estrato"
    erradas = [(n, r["reason"]) for n, r in achou
               if not r["ok"] and r["reason"] != "resposta_inversa"]
    assert not erradas, (
        f"recusa pelo motivo errado em {len(erradas)}/{len(achou)}: {erradas[:5]}")


def test_a_taxa_de_recusa_nao_regride(_pipeline):
    """Piso na fração recusada por `resposta_inversa`. Ver `_RECUSA_MIN`."""
    n = len(_pipeline)
    recusou = sum(1 for _, r in _pipeline if r["reason"] == "resposta_inversa")
    assert recusou / n >= _RECUSA_MIN, (
        f"só {recusou}/{n} ({recusou/n:.1%}) recusadas por resposta_inversa, "
        f"piso {_RECUSA_MIN:.0%} — o Estágio A regrediu em curva fora da família")
