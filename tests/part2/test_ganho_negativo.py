"""Caminho C: espelhar a serie descendente (§40.3).

O ganho negativo era estruturalmente inexprimivel — `K_BOUNDS = (1e-3, 1e4)`
trava K positivo e o ajuste saia com NRMSE 0,90-0,96. Espelhar e negar K
depois e exatamente equivalente a parametrizar `K = s*|K|`, porque o modelo e
linear em K e a base e livre de sinal.
"""
import numpy as np
import pytest

from identify.classical import identify, identify_both, model_response

VERDADES = [
    ("fopdt",  {"K": -2.0, "tau": 1.4, "theta": 2.0, "wn": None, "zeta": None}),
    ("fopdt",  {"K": -0.4, "tau": 0.5, "theta": 0.0, "wn": None, "zeta": None}),
    ("second", {"K": -3.0, "tau": None, "theta": 1.0, "wn": 2.0, "zeta": 0.30}),
    ("second", {"K": -1.0, "tau": None, "theta": 0.5, "wn": 4.0, "zeta": 1.50}),
]


def _serie(order, p, n=512, t_fim=14.0):
    t = np.linspace(0.0, t_fim, n)
    return t, model_response(order, p, t)


@pytest.mark.parametrize("order, p", VERDADES)
def test_recupera_ganho_negativo(order, p):
    t, y = _serie(order, p)
    r = identify(t, y)
    assert r.order == order, f"ordem {r.order!r}, esperada {order!r}"
    for nome in ("K", "tau", "theta", "wn", "zeta"):
        alvo = p[nome]
        if alvo is None or abs(alvo) < 1e-9:
            continue
        obtido = r.params[nome]
        assert obtido is not None, f"{nome} ausente"
        erro = abs(obtido - alvo) / abs(alvo)
        assert erro <= 0.01, f"{nome} = {obtido:.5f}, esperado {alvo} (erro {erro:.2%})"


@pytest.mark.parametrize("order, p", VERDADES)
def test_o_K_devolvido_e_negativo(order, p):
    t, y = _serie(order, p)
    assert identify(t, y).params["K"] < 0.0


def test_theta_zero_sai_proximo_de_zero():
    """theta=0 e excluido do teste relativo acima (denominador nulo); aqui vai
    o absoluto, normalizado pela janela."""
    p = {"K": -0.4, "tau": 0.5, "theta": 0.0, "wn": None, "zeta": None}
    t, y = _serie("fopdt", p)
    r = identify(t, y)
    assert abs(r.params["theta"]) <= 0.01 * float(t[-1] - t[0])


@pytest.mark.parametrize("order, p", [
    ("fopdt",  {"K": 2.0, "tau": 1.4, "theta": 2.0, "wn": None, "zeta": None}),
    ("second", {"K": 3.0, "tau": None, "theta": 1.0, "wn": 2.0, "zeta": 0.30}),
])
def test_caminho_positivo_intocado(order, p):
    """Com K > 0 o espelho nao dispara e o resultado tem de ser IDENTICO ao que
    `identify_both` + a regra de escolha produzem — nao 'parecido'."""
    t, y = _serie(order, p)
    r = identify(t, y)
    r1, r2 = identify_both(t, y)
    referencia = r1 if r.order == "fopdt" else r2
    assert r.params == referencia.params
    assert r.sse == referencia.sse
    assert r.nrmse == referencia.nrmse


def test_metricas_sao_invariantes_ao_espelho():
    """sse/nrmse dependem so de residuos ao quadrado: a versao negativa de uma
    serie tem de dar exatamente as mesmas metricas que a positiva."""
    p_pos = {"K": 2.0, "tau": 1.4, "theta": 2.0, "wn": None, "zeta": None}
    p_neg = {**p_pos, "K": -2.0}
    t, y_pos = _serie("fopdt", p_pos)
    _, y_neg = _serie("fopdt", p_neg)
    a, b = identify(t, y_pos), identify(t, y_neg)
    assert a.nrmse == pytest.approx(b.nrmse, rel=1e-9)
    assert a.params["tau"] == pytest.approx(b.params["tau"], rel=1e-9)
    assert a.params["K"] == pytest.approx(-b.params["K"], rel=1e-9)
