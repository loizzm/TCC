"""Unidade da truncagem no 1º degrau (spec de 04/09/2026).

Séries SINTÉTICAS de propósito: esta tarefa é sobre a lógica de decisão, e
misturar erro do Estágio A aqui esconderia qual das duas coisas quebrou.
"""
import numpy as np
import pytest

from identify.classical import (identify, identify_com_truncagem,
                                model_response, _PISO_SUSPEITA, _GANHO_MIN)

P1 = {"K": 2.0, "tau": 0.5, "theta": 1.5}
P2 = {"K": 1.5, "tau": 0.5, "theta": 4.5}


def _t():
    return np.linspace(0, 12, 600)


def _um_degrau():
    t = _t()
    return t, model_response("fopdt", P1, t)


def _dois_degraus():
    t = _t()
    return t, model_response("fopdt", P1, t) + model_response("fopdt", P2, t)


def test_um_degrau_nao_trunca():
    """Série de degrau único fica abaixo do piso e não paga varredura nenhuma."""
    t, y = _um_degrau()
    r = identify_com_truncagem(t, y)
    assert r.truncado_em is None, f"truncou em {r.truncado_em}"
    assert r.ganho is None
    assert r.nrmse_full <= _PISO_SUSPEITA


def test_caminho_comum_identico_ao_de_hoje():
    """Abaixo do piso, o resultado é o MESMO objeto de parâmetros que `identify`
    devolve. É o portão 3 da spec: prova que 98 % das imagens não mudaram."""
    t, y = _um_degrau()
    esperado = identify(t, y)
    r = identify_com_truncagem(t, y)
    assert r.fit.order == esperado.order
    assert r.fit.params == esperado.params
    assert r.fit.nrmse == esperado.nrmse
    assert r.t.size == t.size and r.y.size == y.size


def test_dois_degraus_trunca_e_recupera_o_primeiro():
    """Recupera K, tau e theta do 1º degrau a partir da série de dois."""
    t, y = _dois_degraus()
    r = identify_com_truncagem(t, y)
    assert r.truncado_em is not None, "não truncou uma série de dois degraus"
    assert r.ganho >= _GANHO_MIN
    assert r.fit.order == "fopdt"
    for nome, esperado in (("K", 2.0), ("tau", 0.5), ("theta", 1.5)):
        obtido = r.fit.params[nome]
        assert abs(obtido - esperado) / esperado <= 0.06, (
            f"{nome} = {obtido:.4f}, esperado {esperado}")


def test_serie_truncada_acompanha_o_ajuste():
    """`t`/`y` devolvidos são o PREFIXO, não a série inteira — a pipeline usa
    esses dois para a guarda e para o bloco adimensional, e usar a série inteira
    ali julgaria uma série que o ajuste não descreve."""
    t, y = _dois_degraus()
    r = identify_com_truncagem(t, y)
    assert r.t.size < t.size
    assert r.t[-1] == pytest.approx(r.truncado_em)
    assert r.y.size == r.t.size


def test_ajuste_completo_que_falha_nao_dispara_varredura():
    """Sem denominador não há ganho definível: o comportamento é o de hoje.

    NÃO usar série constante aqui: foi medido que `identify` NÃO falha nela —
    devolve `success=True` com `nrmse = 1,2e11`, finito e acima do piso, o que
    dispararia a varredura. `nan` é o que de fato produz `success=False`.
    """
    t = _t()
    y = np.full_like(t, np.nan)
    r = identify_com_truncagem(t, y)
    assert r.truncado_em is None
    assert r.ganho is None


def test_serie_curta_demais_nao_trunca():
    """Nenhum corte candidato sobrevive ao piso de 30 pontos.

    30 pontos exatos: o maior corte é `int(0,95 x 30) = 28`, abaixo do piso, então
    a lista de candidatos sai VAZIA. E `nrmse = 0,05999` está acima do piso de
    suspeita, o que garante que o código CHEGA ao laço de varredura — sem isso o
    teste passaria pelo caminho errado, provando nada.
    """
    t = np.linspace(0, 12, 30)
    y = model_response("fopdt", P1, t) + model_response("fopdt", P2, t)
    r = identify_com_truncagem(t, y)
    assert r.nrmse_full > _PISO_SUSPEITA, "o teste precisa CHEGAR ao laço"
    assert r.truncado_em is None
