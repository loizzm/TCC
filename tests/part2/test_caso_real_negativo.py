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
  - `neg_fopdt` FECHA, desde o retreino do §40.7. Ela era o caso do "defeito
    4": o degrau de ENTRADA é plotado como tracejada branca no mesmo quadro, e
    a polilinha começava SOBRE ELA (y[0] = -1,995) em vez de sobre a resposta,
    o que fazia o mínimo preceder o máximo, a direção sair como subida, o
    espelho não disparar e K travar no piso (0,001, NRMSE 0,90).

    A ATRIBUIÇÃO DA CAUSA ESTAVA ERRADA, e vale registrar porque custou uma
    conclusão: aquilo foi lido como "dois objetos de curva no mesmo quadro,
    envelope novo, spec própria". Não era. Era o MESMO prior de posição do
    §40.7 — a rede não via o platô de repouso da resposta, que fica no topo, e
    a única linha visível naquela altura era a da entrada. Um retreino fechou
    os dois defeitos porque eram um só. Hoje: K 0,15 %, tau 0,01 %, theta
    0,02 %.
  - `neg_super` recupera K (0,42 %) e theta (0,94 %) — o theta entrou com o
    retreino. wn e zeta seguem errados (26 % e 30 %) e ficam em `xfail`
    estrito: eles se leem da FORMA do transitório, que depende da cauda
    assentada, e essa a rede ainda perde.
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

@pytest.mark.parametrize("caso", [NEG_SUB, NEG_SUPER, NEG_FOPDT],
                         ids=["sub", "super", "fopdt"])
def test_a_direcao_sai_negativa(caso, modelo):
    """As três DESCEM. `_sinal_do_degrau` é medido aqui contra série real, não
    sintética: é sobre `sub` que a formulação por decil falhava, porque o
    `axvspan` cinza cai sobre o platô inicial e o Estágio A o come.

    A `fopdt` entrou nesta lista DEPOIS do retreino do §40.7. Antes dele a
    polilinha começava sobre a tracejada de ENTRADA (y[0] = -1,995) e a direção
    saía +1; hoje começa sobre a resposta (y[0] = +0,005) e sai -1."""
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

def test_neg_fopdt_fecha_fim_a_fim(modelo):
    """PORTÃO NOVO (§40.11). Esta imagem era RECUSADA até o retreino do §40.7.

    Ela era o caso do "defeito 4": o degrau de ENTRADA é plotado como tracejada
    branca no mesmo quadro, e a polilinha começava sobre ELA em vez de sobre a
    resposta. A atribuição da causa estava errada — ver §40.11. Não era "dois
    objetos confundem a polilinha": era o MESMO prior de posição do §40.7. A
    rede não via o platô de repouso da resposta, que fica no topo, e a única
    linha visível naquela altura era a da entrada."""
    r = _roda(NEG_FOPDT, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r} cal={r['calibration']}"
    assert r["order"] == "fopdt", f"ordem {r['order']!r}"
    assert r["params"]["K"] < 0.0, f"K = {r['params']['K']:.4f}"


def test_neg_fopdt_a_polilinha_segue_a_RESPOSTA_e_nao_a_entrada(modelo):
    """PORTÃO NOVO (§40.11). Mede o defeito 4 pelo lado consertado.

    A versão anterior deste teste asseverava o DEFEITO: `y[0] < -1,5`, o
    patamar da tracejada de entrada, com o mínimo precedendo o máximo. Ele
    existia para dar um número ao defeito 4 e para falhar no dia em que alguém
    o consertasse. Falhou — o retreino do §40.7 consertou.

    Agora assevera o contrário, e é o portão que impede a volta: a série tem de
    começar no repouso da RESPOSTA (perto de 0) e não no patamar da entrada
    (-2), e o máximo tem de vir antes do mínimo, que é o que faz
    `_sinal_do_degrau` ler descida.
    """
    _, y = _serie(NEG_FOPDT, modelo)
    assert abs(float(y[0])) < 0.5, (
        f"y[0] = {float(y[0]):.4f}; perto de -2 significa que a polilinha "
        "voltou a começar sobre a tracejada de ENTRADA (defeito 4 de volta)")
    assert int(np.argmax(y)) < int(np.argmin(y)), (
        "o máximo tem de vir antes do mínimo numa resposta que desce")


@pytest.mark.parametrize("nome", ["K", "tau", "theta"])
def test_neg_fopdt_recupera_os_parametros(nome, modelo):
    """PORTÃO NOVO (§40.11): era `xfail` estrito pelo defeito 4, virou portão.
    Medido depois do retreino: K 0,15 %, tau 0,01 %, theta 0,02 %."""
    r = _roda(NEG_FOPDT, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r}"
    e = _erro(r["params"][nome], NEG_FOPDT[nome])
    assert e <= TOL, (f"{nome} = {r['params'][nome]:.4f}, "
                      f"esperado {NEG_FOPDT[nome]} (erro {e:.1%})")


def test_neg_super_recupera_o_ganho(modelo):
    """K sai certo mesmo com o Estágio A comendo platô e cauda: o ganho se lê da
    excursão, que sobrevive à truncagem."""
    r = _roda(NEG_SUPER, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r}"
    assert r["order"] == "second"
    e = _erro(r["params"]["K"], NEG_SUPER["K"])
    assert e <= TOL, f"K = {r['params']['K']:.4f}, esperado {NEG_SUPER['K']} (erro {e:.1%})"


def test_neg_super_recupera_o_theta(modelo):
    """PORTÃO NOVO (§40.11): era `xfail` pelo §39.3, virou portão.

    O retreino recuperou o ATRASO desta imagem (0,94 % de erro) mas não a
    dinâmica — ver o `xfail` abaixo. Faz sentido: theta se lê de ONDE a resposta
    arranca, e o arranque voltou a ser visível quando o platô de repouso no topo
    passou a ser segmentado. wn e zeta se leem da FORMA do transitório, que
    depende da cauda, ainda perdida."""
    r = _roda(NEG_SUPER, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r}"
    e = _erro(r["params"]["theta"], NEG_SUPER["theta"])
    assert e <= TOL, (f"theta = {r['params']['theta']:.4f}, "
                      f"esperado {NEG_SUPER['theta']} (erro {e:.1%})")


@pytest.mark.xfail(strict=True, reason=
                   "§39.3, o que SOBROU dele depois do retreino do §40.7. O "
                   "platô de repouso desta imagem foi recuperado (theta sai a "
                   "0,94 %, K a 0,42 %), mas a CAUDA assentada não — e wn e zeta "
                   "se leem da forma do transitório, que precisa dela. Medido: "
                   "wn 26 % de erro, zeta 30 %. Exige mais retreino ou um "
                   "estrato de cauda; não conserta no Estágio D. Quando fechar, "
                   "vira XPASS e reprova a suíte.")
@pytest.mark.parametrize("nome", ["wn", "zeta"])
def test_neg_super_recupera_a_dinamica(nome, modelo):
    r = _roda(NEG_SUPER, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r}"
    e = _erro(r["params"][nome], NEG_SUPER[nome])
    assert e <= TOL, (f"{nome} = {r['params'][nome]:.4f}, "
                      f"esperado {NEG_SUPER[nome]} (erro {e:.1%})")
