"""Direção do degrau, lida da série. Base do caminho C (§40.2)."""
import numpy as np
import pytest

from identify.classical import _sinal_do_degrau, model_response


def _serie(K, order="fopdt", n=400):
    t = np.linspace(0.0, 12.0, n)
    p = ({"K": K, "tau": 1.0, "theta": 2.0, "wn": None, "zeta": None}
         if order == "fopdt" else
         {"K": K, "tau": None, "theta": 1.0, "wn": 3.0, "zeta": 0.3})
    return t, model_response(order, p, t)


@pytest.mark.parametrize("order", ["fopdt", "second"])
def test_degrau_positivo_da_mais_um(order):
    _, y = _serie(2.0, order)
    assert _sinal_do_degrau(y) == 1.0


@pytest.mark.parametrize("order", ["fopdt", "second"])
def test_degrau_negativo_da_menos_um(order):
    _, y = _serie(-2.0, order)
    assert _sinal_do_degrau(y) == -1.0


def test_subamortecido_negativo_nao_se_confunde_com_o_sobressinal():
    """zeta baixo faz a curva ULTRAPASSAR o patamar. A direção é do patamar,
    não do pico — por isso o detector usa decis, não o extremo."""
    _, y = _serie(-1.0, "second")
    assert float(np.min(y)) < -1.0        # confirma que há sobressinal
    assert _sinal_do_degrau(y) == -1.0


def test_serie_plana_devolve_mais_um():
    assert _sinal_do_degrau(np.zeros(200)) == 1.0


def test_serie_curta_nao_levanta():
    assert _sinal_do_degrau(np.array([1.0, 2.0])) in (-1.0, 1.0)
    assert _sinal_do_degrau(np.array([])) == 1.0


def test_ruido_nao_inverte_o_sinal():
    rng = np.random.default_rng(7)
    _, y = _serie(-2.0)
    y = y + rng.normal(0.0, 0.05, size=y.size)
    assert _sinal_do_degrau(y) == -1.0
