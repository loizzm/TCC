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


# --------------------------------------------------------------------------- #
# O primeiro decil NÃO é o nível de repouso quando o Estágio A come a cabeça.
#
# `_sinal_do_degrau` comparava mediana do primeiro decil com a do último,
# tratando o primeiro decil como proxy do repouso. Quando o §39.3 corta o
# platô inicial, o primeiro decil cai DENTRO do transitório — e numa
# subamortecida o transitório passa ALÉM do valor final, pelo lado oposto.
# A comparação lê ao contrário.
#
# Medido em `Figure_dn2.png` (2ª ordem, zeta=0,2, theta=2 s): a série extraída
# começa em t=2,09 s, logo depois do platô, e o detector devolvia +1 numa
# resposta que DESCE. O corte não precisa passar de theta para quebrar: onde
# a fronteira cai depende da amostragem, porque o decil pousa onde a
# oscilação estiver — e é justamente isso que torna a fração do decil
# insintonizável.
#
# O defeito é SIMÉTRICO: com degrau positivo e a mesma cabeça cortada o
# detector devolvia -1, espelhando uma série ascendente. Não é um problema de
# ganho negativo — é um problema do estimador.
# --------------------------------------------------------------------------- #

def _sem_cabeca(K, zeta=0.2, theta=2.0, wn=5.0, t0=2.09, n=480):
    """Série de 2ª ordem com o platô inicial CORTADO, como o §39.3 faz."""
    t = np.linspace(t0, 12.0, n)
    p = {"K": K, "tau": None, "theta": theta, "wn": wn, "zeta": zeta}
    return model_response("second", p, t)


@pytest.mark.parametrize("zeta", [0.05, 0.1, 0.2, 0.4])
def test_subamortecida_sem_cabeca_ainda_le_a_descida(zeta):
    assert _sinal_do_degrau(_sem_cabeca(-2.0, zeta=zeta)) == -1.0


@pytest.mark.parametrize("zeta", [0.05, 0.1, 0.2, 0.4])
def test_subamortecida_sem_cabeca_nao_espelha_uma_subida(zeta):
    """O controle que prova que o defeito não era do ganho negativo."""
    assert _sinal_do_degrau(_sem_cabeca(+2.0, zeta=zeta)) == 1.0


@pytest.mark.parametrize("t0", [0.0, 1.0, 1.9, 2.0, 2.09, 2.2])
def test_a_direcao_nao_depende_de_onde_a_cabeca_foi_cortada(t0):
    """Varre o corte da cabeça DENTRO do contrato: enquanto o nível de repouso
    sobreviver na série, nenhum t0 pode inverter a leitura. Em theta=2 s o
    repouso ainda está no dado até t0 = 2,2 s.

    `Figure_dn2.png` cai aqui dentro: a série extraída começa em t=2,09 s com
    y=-0,156, contra o assentado -0,998 — o repouso sobreviveu."""
    assert _sinal_do_degrau(_sem_cabeca(-2.0, t0=t0)) == -1.0


@pytest.mark.xfail(strict=True, reason=
                   "limite do contrato: cortada a cabeça ALÉM do repouso, o "
                   "extremo mais distante do assentado passa a ser o primeiro "
                   "sobressinal, que fica do lado DO degrau e inverte a regra. "
                   "Recuperar exige ajustar o envelope decadente. Se algum dia "
                   "isto passar, o limite mudou e a suíte tem de reprovar para "
                   "obrigar a revisão do contrato.")
@pytest.mark.parametrize("t0", [2.3, 2.4, 2.5, 2.7])
def test_o_limite_do_contrato_quando_o_repouso_sai_da_serie(t0):
    assert _sinal_do_degrau(_sem_cabeca(-2.0, t0=t0)) == -1.0


# --------------------------------------------------------------------------- #
# ... e o valor ASSENTADO também não é confiável quando a janela acaba antes de
# a resposta assentar.
#
# A segunda tentativa estimava o repouso como o extremo MAIS DISTANTE da
# mediana do último decil. Isso pressupõe que o último decil É o valor final —
# e numa 2ª ordem muito subamortecida com janela curta, o último decil pousa no
# ringing. Aí o PRIMEIRO PICO fica mais longe dele que o repouso, o pico é
# eleito repouso, e a leitura inverte.
#
# Medido no oráculo do corpus (900 séries verdadeiras, TODAS com K > 0, logo
# nenhuma pode ser espelhada): duas amostras viravam — `sample_00307`
# (zeta=0,104, wn=1,046, theta=3,69) e `sample_00889` (zeta=0,121, wn=0,355,
# theta=2,13). O efeito na Parte 1 era visível: MAPE(K) do estrato limpo w<3
# saía de 0,000 % para 0,239 %.
#
# Os parâmetros abaixo são os dessas duas amostras, reproduzidos aqui para o
# teste não depender de `data/test`.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("p, t_fim", [
    ({"K": 0.7454976203500854, "tau": None, "theta": 3.6934207649311657,
      "wn": 1.0461979340328496, "zeta": 0.1035450977456918}, 16.40366623495255),
    ({"K": 4.318184176887993, "tau": None, "theta": 2.1310134462874215,
      "wn": 0.35494500547912816, "zeta": 0.12147567544869976}, 21.616401406038005),
])
def test_janela_curta_e_subamortecida_nao_espelha_uma_subida(p, t_fim):
    """Corpus tem K > 0 por construção (`generator.py`): espelhar aqui é erro."""
    t = np.linspace(0.0, t_fim, 512)
    assert _sinal_do_degrau(model_response("second", p, t)) == 1.0
