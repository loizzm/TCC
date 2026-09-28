"""Nivel ESTENDIDO do acerto conjuntivo: NRMSE da curva num horizonte que passa
da janela desenhada.

O NRMSE na janela e cego a erro de EXTRAPOLACAO: com a janela curta, varias
combinacoes (K, zeta, theta) desenham a mesma subida e assentam em patamares
diferentes. O caso de regressao abaixo e o `ruido_Kmaior1_15dB_17` do
`lote_misto2`: K errado em 52 % com NRMSE de janela de 0,6 %.

Nao roda pipeline nem modelo: so as duas funcoes de medida.
"""
from __future__ import annotations

import numpy as np
import pytest

from acerto_conjuntivo import PRATICO, avalia_estendido
from analisa_aleatorias import nrmse_horizonte


def _fopdt(K=1.0, tau=1.0, theta=0.0, t_fim=1.5):
    return {"order": "fopdt", "K": K, "tau": tau, "wn": None, "zeta": None,
            "theta": theta, "t_fim": t_fim, "t_dom": tau}


def _p(v, **mud):
    p = {k: v[k] for k in ("K", "tau", "wn", "zeta", "theta")}
    p.update(mud)
    return p


def test_parametros_identicos_dao_nrmse_zero():
    v = _fopdt()
    assert nrmse_horizonte(v, "fopdt", _p(v), 6.0) == pytest.approx(0.0, abs=1e-12)


def test_erro_puro_de_K_converge_para_o_erro_de_K_no_horizonte_longo():
    # K +5 %, theta = 0, horizonte 6 tau: a curva erro e 0,05*(1-e^-t),
    # cujo RMS sobre [0, 6] e 0,0433 da faixa. Valor analitico, nao medido.
    v = _fopdt()
    t = np.linspace(0.0, 6.0, 200_000)
    esperado = np.sqrt(np.mean((0.05 * (1 - np.exp(-t))) ** 2)) / (1 - np.exp(-6.0))
    got = nrmse_horizonte(v, "fopdt", _p(v, K=1.05), 6.0)
    assert got == pytest.approx(esperado, rel=2e-3)


def test_horizonte_nunca_e_mais_curto_que_a_janela_desenhada():
    # janela de 8 tau, horizonte pedido de 0,01 tau: mede a janela inteira
    v = _fopdt(t_fim=8.0)
    p = _p(v, K=1.1, tau=1.3)
    curto = nrmse_horizonte(v, "fopdt", p, 0.01)
    na_janela = nrmse_horizonte(v, "fopdt", p, (v["t_fim"] - v["theta"]) / v["t_dom"])
    assert curto == pytest.approx(na_janela, rel=1e-9)


def test_caso_15dB_17_passa_na_janela_e_reprova_no_estendido():
    # ruido_Kmaior1_15dB_17.png: verdade e ajuste reais do lote_misto2
    v = {"order": "second", "K": -1.479605504769435, "tau": None,
         "wn": 0.07618897328332647, "zeta": 0.7327927028385658,
         "theta": 4.240187726278811, "t_fim": 31.838606360121407,
         "t_dom": 17.911284942527622}
    p = {"K": -2.2514837101751177, "tau": None, "theta": 4.932338589407499,
         "wn": 0.07045429920562699, "zeta": 1.2668878737779068}
    janela = (v["t_fim"] - v["theta"]) / v["t_dom"]
    assert nrmse_horizonte(v, "second", p, janela) < 0.01
    assert nrmse_horizonte(v, "second", p, 6.0) > 0.20


def _fig(**mud):
    x = {"ok": True, "reason": "", "order_true": "second", "order_hat": "second",
         "err_K": 0.0, "err_theta_T": 0.0, "sinal_K_ok": True,
         "nrmse_curva": 0.001, "nrmse_estendido": 0.001,
         "nrmse_horizontes": {"6.0": 0.001, "12.0": 0.001}}
    x.update(mud)
    return x


def test_aprova_com_estrutura_certa_e_nrmse_estendido_abaixo_de_2pct():
    assert avalia_estendido(_fig(nrmse_estendido=0.0199)) == []


def test_limiar_e_estrito_2pct_reprova():
    assert avalia_estendido(_fig(nrmse_estendido=0.02)) == ["curva estendida"]


def test_nao_cobra_K_nem_theta_diretamente():
    # K e theta ficam cobertos pelo NRMSE estendido, nao por limiar proprio
    assert avalia_estendido(_fig(err_K=0.5, err_theta_T=0.1,
                                 nrmse_estendido=0.01)) == []


def test_troca_de_estrutura_reprova():
    assert avalia_estendido(_fig(order_hat="fopdt")) == ["estrutura"]


def test_nao_entregou_reprova_sozinho():
    f = avalia_estendido(_fig(ok=False, reason="polilinha_curta"))
    assert f == ["nao entregou (polilinha_curta)"]


def test_nrmse_ausente_reprova():
    assert avalia_estendido(_fig(nrmse_estendido=None)) == ["curva estendida"]


def test_horizonte_explicito_le_nrmse_horizontes():
    x = _fig(nrmse_estendido=0.001, nrmse_horizontes={"6.0": 0.001, "12.0": 0.05})
    assert avalia_estendido(x, horizonte=12.0) == ["curva estendida"]
    assert avalia_estendido(x, horizonte=6.0) == []


# --- nivel PRATICO: o ESTENDIDO sem exigir o rotulo de estrutura -----------

@pytest.mark.parametrize("verdade, predito", [("second", "fopdt"), ("fopdt", "second")])
def test_pratico_aceita_troca_de_estrutura_nos_dois_sentidos(verdade, predito):
    x = _fig(order_true=verdade, order_hat=predito, nrmse_estendido=0.01)
    assert avalia_estendido(x, PRATICO) == []


def test_pratico_ainda_reprova_troca_que_estraga_a_curva():
    x = _fig(order_true="second", order_hat="fopdt", nrmse_estendido=0.03)
    assert avalia_estendido(x, PRATICO) == ["curva estendida"]


def test_pratico_nao_entregou_reprova():
    x = _fig(ok=False, reason="polilinha_curta")
    assert avalia_estendido(x, PRATICO) == ["nao entregou (polilinha_curta)"]
