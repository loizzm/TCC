"""Regressão das três imagens de GANHO NEGATIVO (Bloco 9, caminho C).

As três vêm de `rg_negativo.py`, versionado na raiz, com a verdade declarada na
própria função de transferência — nada aqui foi inferido de gráfico. Foi assim
que se descobriu que o título da Figura 2 dizia zeta=0,3 enquanto o sistema tem
zeta=0,2, e que o ajuste (0,201) estava certo e a leitura visual errada.

    `caso_real_neg_fopdt.png`   TransferFunction([2],[1,2]), theta=1, degrau
                                de amplitude -2 em t=2 s
                                -> FOPDT, K=-2, tau=0,5, theta da janela = 3 s
    `caso_real_neg_sub.png`     TransferFunction([25],[1,2,25]), theta=2,
                                degrau de amplitude -1 em t=0
                                -> 2a ordem, K=-1, wn=5, zeta=0,2, theta=2 s
    `caso_real_neg_super.png`   TransferFunction([16],[1,10,16]), theta=0,5,
                                degrau de amplitude -3 em t=3 s
                                -> 2a ordem, K=-3, wn=4, zeta=1,25,
                                   theta da janela = 3,5 s

Antes do caminho C as três eram recusadas, e pelos motivos ERRADOS: `K_BOUNDS`
é positivo por construção, o ajuste travava K no piso 0,001 e saía com NRMSE
0,90-0,96. `resposta_inversa` na primeira era diagnóstico FALSO — não é fase
não-mínima, é degrau negativo.

O que cada uma protege, e o que cada uma ainda não entrega:

  - `neg_sub` FECHA fim a fim, e é o único teste que assevera isso sobre uma
    imagem real. Os testes sintéticos cobrem `_sinal_do_degrau` e `identify`
    isoladamente; nenhum deles pega uma regressão no Estágio A. Esta imagem
    pega, porque a série extraída dela perde o platô inicial (o `axvspan` cinza
    cai exatamente sobre ele) e mesmo assim a direção tem de sair certa.
  - `neg_fopdt` segue RECUSADA pelo defeito 4: o degrau de ENTRADA é plotado
    como tracejada branca no mesmo quadro, e a polilinha começa SOBRE ELA
    (y[0] = -1,995, o patamar da entrada) em vez de sobre a resposta, cujo
    repouso só aparece 38 amostras depois. O dano é anterior ao Estágio D e
    contamina tudo o que vem depois: o mínimo precede o máximo, a direção é
    lida como subida, o espelho não dispara e K trava no piso (0,001, NRMSE
    0,90). Os testes abaixo MEDEM o defeito e põem a recuperação em `xfail`
    estrito, para o dia em que alguém separar os dois objetos.
  - `neg_super` responde com K certo e wn/zeta/theta errados, por causa do
    §39.3 (o Estágio A come o platô e a cauda, cobertura de 62,5 %). Os três
    parâmetros errados entram com `xfail` estrito: quando o retreino fechar
    aquele defeito, viram XPASS, a suíte reprova e obriga a conversão em portão.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIX = Path(__file__).resolve().parents[1] / "fixtures"

NEG_SUB = {"path": FIX / "caso_real_neg_sub.png",
           "order": "second", "K": -1.0, "wn": 5.0, "zeta": 0.2, "theta": 2.0}
NEG_FOPDT = {"path": FIX / "caso_real_neg_fopdt.png",
             "order": "fopdt", "K": -2.0, "tau": 0.5, "theta": 3.0}
NEG_SUPER = {"path": FIX / "caso_real_neg_super.png",
             "order": "second", "K": -3.0, "wn": 4.0, "zeta": 1.25, "theta": 3.5}

# Mesma tolerância de `test_caso_real_rg.py`: folgada sobre o que a cadeia
# entrega e apertada o bastante para reprovar o defeito.
TOL = 0.06


@pytest.fixture(scope="module")
def modelo():
    import torch
    from identify.extract import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model("models/unet_stageA.pt", dev), dev


def _roda(caso, modelo):
    from identify.pipeline import identify_from_image
    m, dev = modelo
    img = np.asarray(Image.open(caso["path"]).convert("RGB"))
    return identify_from_image(img, m, dev)


def _serie(caso, modelo):
    """Série extraída, ANTES das guardas de plausibilidade da pipeline."""
    from identify.calibrate import calibrate
    from identify.extract import predict_mask
    from identify.polyline import mask_to_polyline
    from identify.pipeline import polyline_to_series
    m, dev = modelo
    img = np.asarray(Image.open(caso["path"]).convert("RGB"))
    cal = calibrate(img)
    mask = predict_mask(m, img, dev)
    xp, yp = mask_to_polyline(mask, bbox=cal.bbox_px if any(cal.bbox_px) else None)
    t, y = polyline_to_series(xp, yp, cal)
    o = np.argsort(t)
    return t[o], y[o]


def _erro(medido, esperado):
    return abs(medido - esperado) / abs(esperado)


# --------------------------------------------------------------------------- #
# A direção, lida das três séries reais
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("caso", [NEG_SUB, NEG_SUPER], ids=["sub", "super"])
def test_a_direcao_sai_negativa(caso, modelo):
    """As duas DESCEM. `_sinal_do_degrau` é medido aqui contra série real, não
    sintética: é sobre `sub` que a formulação por decil falhava, porque o
    `axvspan` cinza cai sobre o platô inicial e o Estágio A o come."""
    from identify.classical import _sinal_do_degrau
    _, y = _serie(caso, modelo)
    assert _sinal_do_degrau(y) == -1.0


# --------------------------------------------------------------------------- #
# neg_sub — fecha fim a fim
# --------------------------------------------------------------------------- #

def test_neg_sub_fecha_fim_a_fim(modelo):
    """A imagem inteira, do PNG aos parâmetros físicos, com K < 0."""
    r = _roda(NEG_SUB, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r} cal={r['calibration']}"
    assert r["order"] == "second", f"ordem {r['order']!r}, esperada 'second'"
    p = r["params"]
    assert p["K"] < 0.0, f"K = {p['K']:.4f}, tinha de ser negativo"
    for nome in ("K", "wn", "zeta", "theta"):
        e = _erro(p[nome], NEG_SUB[nome])
        assert e <= TOL, (f"{nome} = {p[nome]:.4f}, esperado {NEG_SUB[nome]} "
                          f"(erro {e:.1%}, tolerância {TOL:.0%})")


# --------------------------------------------------------------------------- #
# neg_fopdt — recusada pelo defeito 4, com a física certa por baixo
# --------------------------------------------------------------------------- #

def test_neg_fopdt_e_recusada_e_nomeia_a_causa(modelo):
    """Comportamento ATUAL, documentado: a pipeline recusa em vez de devolver
    número errado em silêncio. A causa é o degrau de entrada plotado no mesmo
    quadro (defeito 4), que faz a polilinha pular entre dois objetos."""
    r = _roda(NEG_FOPDT, modelo)
    assert not r["ok"]
    assert r["reason"] in ("resposta_inversa", "ajuste_inconsistente"), r["reason"]


def test_neg_fopdt_a_polilinha_comeca_no_degrau_de_entrada(modelo):
    """MEDE o defeito 4, para que ele tenha um número e não só um nome.

    A série extraída começa em -1,995 — o patamar da TRACEJADA DE ENTRADA, que
    `rg_negativo.py` desenha em y=-2 a partir de t=2 s. O repouso da RESPOSTA
    (y=0) só aparece 38 amostras depois. A polilinha, portanto, não está
    seguindo um objeto só.

    A consequência é dupla e está asseverada abaixo: a direção é lida ao
    contrário (o mínimo vem antes do máximo, então a série "sobe") e o ajuste
    trava K no piso. Quem consertar o defeito 4 tem de fazer este teste falhar.
    """
    _, y = _serie(NEG_FOPDT, modelo)
    assert float(y[0]) < -1.5, (
        f"y[0] = {float(y[0]):.4f}; se a polilinha passou a seguir a RESPOSTA, "
        "ela começa perto de 0 e o defeito 4 foi consertado")
    assert int(np.argmin(y)) < int(np.argmax(y))


@pytest.mark.xfail(strict=True, reason=
                   "defeito 4: o degrau de ENTRADA é plotado como tracejada "
                   "branca no mesmo quadro, a polilinha pula entre os dois "
                   "objetos e começa sobre a entrada (y[0]=-1,995, o patamar "
                   "dela) em vez de sobre a resposta. Com isso o mínimo precede "
                   "o máximo, `_sinal_do_degrau` lê subida, o espelho não "
                   "dispara e K trava no piso de K_BOUNDS (0,001, NRMSE 0,90). "
                   "Fechar exige distinguir dois objetos de curva no mesmo "
                   "quadro — envelope novo, spec própria. Quando fechar, isto "
                   "vira XPASS e reprova a suíte.")
@pytest.mark.parametrize("nome", ["K", "tau", "theta"])
def test_neg_fopdt_recupera_os_parametros(nome, modelo):
    from identify.classical import identify
    t, y = _serie(NEG_FOPDT, modelo)
    f = identify(t, y)
    e = _erro(f.params[nome], NEG_FOPDT[nome])
    assert e <= TOL, (f"{nome} = {f.params[nome]:.4f}, "
                      f"esperado {NEG_FOPDT[nome]} (erro {e:.1%})")


# --------------------------------------------------------------------------- #
# neg_super — K certo, o resto preso no §39.3
# --------------------------------------------------------------------------- #

def test_neg_super_recupera_o_ganho(modelo):
    """K sai certo mesmo com o Estágio A comendo platô e cauda: o ganho se lê da
    excursão, que sobrevive à truncagem."""
    r = _roda(NEG_SUPER, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r}"
    assert r["order"] == "second"
    e = _erro(r["params"]["K"], NEG_SUPER["K"])
    assert e <= TOL, f"K = {r['params']['K']:.4f}, esperado {NEG_SUPER['K']} (erro {e:.1%})"


@pytest.mark.xfail(strict=True, reason=
                   "§39.3: o Estágio A perde o platô e a cauda desta imagem "
                   "(cobertura 62,5 %), e sem eles a DINÂMICA não é "
                   "identificável — só o ganho. Exige RETREINO, não conserta no "
                   "Estágio D. Quando o retreino fechar, isto vira XPASS e "
                   "reprova a suíte, obrigando a virar portão de regressão.")
@pytest.mark.parametrize("nome", ["wn", "zeta", "theta"])
def test_neg_super_recupera_a_dinamica(nome, modelo):
    r = _roda(NEG_SUPER, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r}"
    e = _erro(r["params"][nome], NEG_SUPER[nome])
    assert e <= TOL, (f"{nome} = {r['params'][nome]:.4f}, "
                      f"esperado {NEG_SUPER[nome]} (erro {e:.1%})")
