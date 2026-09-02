"""O sinal negativo que o tesseract devolve não é o hífen ASCII.

O matplotlib desenha U+2212 (MINUS SIGN); o tesseract lê o dígito certo e
devolve o sinal como em-dash. Medido em `Figure_dn2.png`, texto cru: '—0.2',
'—0.4', '—0.6', '—-0.8', '—1.0', '=1.2', '—1.4', '—16'. Ver §40.1.
"""
import pytest

from identify.calibrate import _texto_para_numero


@pytest.mark.parametrize("txt, esperado", [
    ("—0.2", -0.2),    # em-dash, o que o tesseract mais devolve
    ("−0.2", -0.2),    # minus sign, o que o matplotlib desenha
    ("–0.2", -0.2),    # en-dash
    ("—-0.8", -0.8),   # em-dash + hífen: sinal duplicado, um só vale
    ("-0.2", -0.2),         # hífen ASCII, que já funcionava
    ("—1.0", -1.0),
])
def test_le_sinal_negativo(txt, esperado):
    assert _texto_para_numero(txt) == pytest.approx(esperado)


@pytest.mark.parametrize("txt", [
    "=1.2",     # '=' NÃO é sinal: mapear seria inventar leitura
    "abc",
    "",
    "--",
    "1.2.3",
    "—",   # sinal sozinho, sem dígito
])
def test_rejeita_o_que_nao_e_numero(txt):
    assert _texto_para_numero(txt) is None


@pytest.mark.parametrize("txt, esperado", [
    ("0.0", 0.0), ("12", 12.0), ("1,5", 1.5), ("+3", 3.0), ("2e3", 2000.0),
])
def test_nao_regride_o_caminho_positivo(txt, esperado):
    assert _texto_para_numero(txt) == pytest.approx(esperado)
