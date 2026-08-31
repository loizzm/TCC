"""Regressão do CASO REAL (HANDOFF_P2_7 §34, Ruling 55).

Imagem produzida FORA do gerador do projeto, com a verdade declarada pelo autor
do sistema: num = [wn**2], den = [1, 2*zeta*wn, wn**2], zeta = 0.5, wn = 2.0,
degrau unitário, janela de 10 s, theta = 0.

Este teste existe porque UMA imagem real encontrou dois defeitos que 300 amostras
sintéticas não encontraram. Ele não substitui a suíte sintética: guarda o eixo que
ela não cobre.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "caso_real_2ordem.png"

CASO_REAL = {
    "path": FIXTURE,
    "order": "second",
    "zeta": 0.5,
    "wn": 2.0,
    "K": 1.0,
    "theta": 0.0,
    "T": 10.0,          # janela em segundos, lida do eixo x da figura
}

# Tolerâncias. O caso real é mais difícil que o sintético (Ruling 55): a máscara
# enfrenta reta de referência coincidente com o patamar, legenda dentro da moldura
# e texto fora dela. 5 % é folgado sobre os 0,4 % que a cadeia consertada entrega
# e apertado o bastante para reprovar os 12,6 % do estimador defeituoso.
TOL_ZETA = 0.05
TOL_WN = 0.05


@pytest.fixture(scope="module")
def caso_real() -> np.ndarray:
    assert FIXTURE.exists(), f"fixture ausente: {FIXTURE}"
    return np.asarray(Image.open(FIXTURE).convert("RGB"))


def test_caso_real_acerta_a_ordem(caso_real):
    """A ordem é o portão: com ordem errada, zeta e wn_T saem nulos."""
    import torch
    from identify.extract import load_model
    from identify.pipeline import identify_from_image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    r = identify_from_image(caso_real, model, dev)
    assert r["order"] == CASO_REAL["order"], (
        f"ordem {r['order']!r}, esperada {CASO_REAL['order']!r}; "
        f"calibração: {r['calibration']}"
    )


def test_caso_real_recupera_zeta_e_wn(caso_real):
    """zeta e wn*T pelo nível ADIMENSIONAL — a calibração falha nesta imagem
    (reta de referência + legenda), e é exatamente o cenário da Decisão E."""
    import torch
    from identify.extract import load_model
    from identify.pipeline import identify_from_image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    r = identify_from_image(caso_real, model, dev)
    d = r["dimensionless"]

    assert d["zeta"] is not None, f"zeta ausente; order={r['order']!r}"
    e_zeta = abs(d["zeta"] - CASO_REAL["zeta"]) / CASO_REAL["zeta"]
    assert e_zeta <= TOL_ZETA, (
        f"zeta = {d['zeta']:.4f}, esperado {CASO_REAL['zeta']:.4f} "
        f"(erro {e_zeta:.1%}, tolerância {TOL_ZETA:.0%})"
    )

    assert d["wn_T"] is not None, "wn_T ausente"
    wn = d["wn_T"] / CASO_REAL["T"]
    e_wn = abs(wn - CASO_REAL["wn"]) / CASO_REAL["wn"]
    assert e_wn <= TOL_WN, (
        f"wn = {wn:.4f}, esperado {CASO_REAL['wn']:.4f} "
        f"(erro {e_wn:.1%}, tolerância {TOL_WN:.0%})"
    )


def test_polilinha_segue_o_bloco_mais_proximo_do_ponto_anterior():
    """Coluna com dois blocos de tinta bem separados: a polilinha segue o
    bloco mais próximo do ponto anterior, não a mediana de todas as linhas.

    Nota (fix round 1): a versão anterior deste teste dizia respeito à
    amostra de linha da legenda no caso real e passava mesmo sem a
    desambiguação — não discriminava nada. A medição correta (feita sobre a
    máscara predita pelo U-Net, não sobre uma máscara de limiar de cor)
    mostra que os 44 blocos duplos do caso real ficam todos em x=[81,389],
    na parte ascendente da curva, e a amostra da legenda (25 px de tinta em
    y~453) não contamina nenhuma coluna ali — o mecanismo é real, mas o
    exemplo da legenda estava errado no documento de diagnóstico. Este teste
    troca o caso real por uma máscara sintética determinística: uma curva
    suave atravessando toda a largura, mais um segundo bloco de tinta bem
    separado (imitando outro objeto — reta de referência, marcador, etc.)
    só num trecho de colunas. Sem a desambiguação por blocos, a mediana da
    coluna cai para o meio dos dois blocos, longe da curva.
    """
    from identify.polyline import mask_to_polyline

    H, W = 60, 40
    mask = np.zeros((H, W), dtype=np.uint8)
    curva_y = {}
    for x in range(W):
        y = int(round(10 + 0.3 * x))
        curva_y[x] = y
        mask[max(y - 1, 0):y + 2, x] = 255
    # segundo bloco, bem separado da curva (gap >> 3 * espessura), só num
    # trecho de colunas — imita um segundo objeto dentro da moldura.
    bloco_extra_y = 45
    for x in range(15, 26):
        mask[bloco_extra_y - 1:bloco_extra_y + 2, x] = 255

    xp, yp = mask_to_polyline(mask)
    trecho = (xp >= 16) & (xp <= 24)
    assert trecho.sum() >= 5, "trecho multi-bloco vazio na polilinha"
    for x, y in zip(xp[trecho], yp[trecho]):
        esperado = curva_y[int(x)]
        assert abs(y - esperado) <= 2.0, (
            f"x={x:.0f}: y={y:.1f} longe da curva (esperado ~{esperado}); "
            "a polilinha provavelmente caiu no meio dos dois blocos"
        )


def test_polilinha_reseta_a_referencia_apos_vao_largo():
    """Depois de um vão real (sem tinta) largo o bastante para disparar o
    reset (identify/polyline.py, linhas ~97-107), a referência `anterior`
    tem que ser descartada — seguir "o bloco mais próximo do ponto
    anterior" usaria, com confiança falsa, uma referência velha demais: a
    curva pode ter se deslocado bastante durante o vão.

    Desenho: uma curva plana em y=20 nas colunas 0-49; um vão real de 12
    colunas (50-61) sem tinta nenhuma; a curva reaparece em y=100 (deslocada
    80 px) a partir da coluna 62; e nas colunas 62-63 um segundo traço fino
    (1 px) exatamente na posição ANTIGA (y=20) cria, ali, uma coluna com
    dois blocos — a ambiguidade multi-bloco que o reset precisa resolver
    bem na hora em que é mais perigoso (logo após o vão).

    Números medidos (não presumidos): espessura_mediana = 3 (a maioria das
    colunas tem 3 px de tinta), então o gatilho do reset é
    3 * 3 = 9 px. O vão real mede 14 colunas do último ponto de esqueleto
    antes do vão (x=48) até o primeiro depois (x=62) — 14 > 9, dispara o
    reset. Ao mesmo tempo esse vão não é descartado como ausência de dado:
    MAX_GAP_FRAC (0,15) vezes a largura da polilinha (138 px) dá 20,7 px
    de folga, e 14 <= 20,7.

    Sem o reset: `anterior` ainda vale ~19 (valor antes do vão) e a coluna
    x=62 (dois blocos: y=20 e y=100) segue o bloco mais próximo de 19 —
    o bloco ERRADO em y=20, a 80 px do valor real (y=100).

    Com o reset: `anterior` é descartado antes de decidir, e a coluna cai
    no fallback (mediana de TODAS as linhas: mediana({20, 100}) = 60) — a
    60 px do valor real, bem mais perto que os 80 px do bloco errado.
    """
    from identify.polyline import mask_to_polyline

    H, W = 140, 140
    mask = np.zeros((H, W), dtype=np.uint8)

    y_velho = 20
    for x in range(0, 50):
        mask[y_velho - 1:y_velho + 2, x] = 255

    # vão real: colunas 50-61 (12 colunas) sem tinta nenhuma.

    y_real = 100
    for x in range(62, W):
        mask[y_real - 1:y_real + 2, x] = 255

    # segundo traço fino (1 px), na posição ANTIGA, só nas 2 colunas logo
    # depois do vão -- cria a ambiguidade multi-bloco exatamente onde o
    # reset precisa agir.
    mask[y_velho, 62:64] = 255

    xp, yp = mask_to_polyline(mask)
    idx = np.flatnonzero(xp == 62.0)
    assert idx.size == 1, "coluna x=62 (logo após o vão) ausente da polilinha"
    y62 = float(yp[idx[0]])

    esperado_com_reset = 60.0  # mediana({y_velho=20, y_real=100}), o fallback
    erro = abs(y62 - esperado_com_reset)
    assert erro <= 2.0, (
        f"x=62: y={y62:.1f}, esperado ~{esperado_com_reset:.0f} px (fallback: "
        f"mediana de todas as linhas da coluna, erro medido {erro:.1f} px); "
        "a polilinha provavelmente seguiu o bloco antigo em y=20 (a 80 px "
        "do valor real) por não ter descartado a referência velha depois "
        "do vão"
    )


def test_polilinha_nao_sai_da_moldura(caso_real):
    """Título e rótulo de eixo ficam FORA da moldura e não são a curva."""
    import torch
    from identify.calibrate import detect_plot_bbox
    from identify.extract import load_model, predict_mask
    from identify.polyline import mask_to_polyline
    from tests.part2.conftest import to_gray

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    bbox = detect_plot_bbox(to_gray(caso_real))
    assert bbox is not None
    x0, y0, x1, y1 = bbox

    mask = predict_mask(model, caso_real, dev)
    xp, yp = mask_to_polyline(mask, bbox=bbox)
    assert xp.size >= 10, "polilinha curta"
    assert yp.min() >= y0, f"y mínimo {yp.min():.0f} acima da moldura (y0={y0})"
    assert yp.max() <= y1, f"y máximo {yp.max():.0f} abaixo da moldura (y1={y1})"
    assert xp.min() >= x0 and xp.max() <= x1, "x fora da moldura"
