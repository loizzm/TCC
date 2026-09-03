"""Estrato opt-in de ganho negativo (§40.5). Molde: `reta_no_patamar` (§34.5).

O corpus tinha ZERO amostras de ganho negativo: `sample_system` sorteia
`K = _loguniform(rng, 0.2, 20.0)`. Sem treino, sem teste, sem critério — tudo
o que o Ruling 63 afirma sobre ganho negativo se apoiava em três imagens reais
e em séries sintéticas de teste, nunca no corpus.

Opt-in, e NÃO um sorteio dentro de `sample_system`, porque mexer no sorteio
moveria toda amostra do corpus base e com ela todo número histórico da Parte 1
e da Parte 2. O corpus base fica byte a byte idêntico e o estrato entra ao lado.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dataset.generator import generate_sample, load_sample


def _meta_sem_id(d: Path) -> dict:
    """meta.json sem `sample_id`, que vem do NOME do diretório de propósito e
    portanto difere entre duas gerações em pastas distintas."""
    m = json.loads((d / "meta.json").read_text())
    m.pop("sample_id", None)
    return m


def _gera(tmp, seed, **kw):
    d = Path(tmp) / f"s{seed}_{int(kw.get('ganho_negativo', False))}"
    generate_sample(str(d), seed=seed, **kw)
    return load_sample(d)


def test_o_padrao_nao_muda_um_byte():
    """Sem o flag, a amostra tem de ser idêntica à de antes desta mudança."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"; b = Path(tmp) / "b"
        generate_sample(str(a), seed=4242)
        generate_sample(str(b), seed=4242, ganho_negativo=False)
        assert (a / "image.png").read_bytes() == (b / "image.png").read_bytes()
        assert (a / "mask.png").read_bytes() == (b / "mask.png").read_bytes()
        assert _meta_sem_id(a) == _meta_sem_id(b)


def test_com_o_flag_o_ganho_e_negativo():
    with tempfile.TemporaryDirectory() as tmp:
        m = _gera(tmp, 4242, ganho_negativo=True)
        assert m["params"]["K"] < 0.0


def test_a_serie_desce():
    with tempfile.TemporaryDirectory() as tmp:
        m = _gera(tmp, 4242, ganho_negativo=True)
        y = np.asarray(m["series"]["y"], dtype=float)
        assert float(np.median(y[-max(1, y.size // 10):])) < float(y[0])


def test_o_detector_de_direcao_le_o_estrato_inteiro():
    """A prova que liga o estrato ao Ruling 63: `_sinal_do_degrau` tem de dar
    -1 em TODA amostra do estrato e +1 em toda amostra do base, sobre a série
    VERDADEIRA. É o portão que faltava — antes disto, nenhuma amostra de ganho
    negativo passava perto do detector."""
    from identify.classical import _sinal_do_degrau
    with tempfile.TemporaryDirectory() as tmp:
        for seed in (11, 202, 3003, 40004, 500005):
            neg = _gera(tmp, seed, ganho_negativo=True)
            pos = _gera(tmp, seed)
            yn = np.asarray(neg["series"]["y"], dtype=float)
            yp = np.asarray(pos["series"]["y"], dtype=float)
            assert _sinal_do_degrau(yn) == -1.0, f"seed {seed}: estrato negativo"
            assert _sinal_do_degrau(yp) == 1.0, f"seed {seed}: base positivo"


def test_a_magnitude_e_a_mesma_do_positivo():
    """Só o SINAL muda: o mesmo seed tem de dar |K| idêntico, senão o estrato
    não é comparável amostra a amostra com o base."""
    with tempfile.TemporaryDirectory() as tmp:
        pos = _gera(tmp, 4242)
        neg = _gera(tmp, 4242, ganho_negativo=True)
        assert abs(neg["params"]["K"]) == abs(pos["params"]["K"])
        assert neg["params"]["theta"] == pos["params"]["theta"]
        assert neg["order"] == pos["order"]
        assert neg["t_window"] == pos["t_window"]


def test_determinismo():
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"; b = Path(tmp) / "b"
        generate_sample(str(a), seed=99, ganho_negativo=True)
        generate_sample(str(b), seed=99, ganho_negativo=True)
        assert (a / "image.png").read_bytes() == (b / "image.png").read_bytes()
        assert _meta_sem_id(a) == _meta_sem_id(b)


def test_a_mascara_continua_valida():
    with tempfile.TemporaryDirectory() as tmp:
        m = _gera(tmp, 4242, ganho_negativo=True)
        mk = np.asarray(m["mask"])
        assert set(np.unique(mk)) <= {0, 255}
        acesos = int((mk > 127).sum())
        assert acesos >= 40, f"mascara degenerada: {acesos} px acesos"


def test_generate_dataset_propaga_o_flag():
    """Task 5 precisa gerar um estrato inteiro; sem isto, o flag morre em
    `generate_sample` e o corpus novo não existe."""
    from dataset.generator import generate_dataset
    with tempfile.TemporaryDirectory() as tmp:
        dirs = generate_dataset(tmp, n=3, seed=7, workers=1,
                                add_noise=False, ganho_negativo=True)
        assert len(dirs) == 3
        for d in dirs:
            assert load_sample(d)["params"]["K"] < 0.0


def _taxa_de_recuperacao(neg: bool, ruido: bool, seeds=range(9000, 9060)):
    """Fração das amostras em que K sai com o SINAL certo e a <= 5 % da verdade,
    pelo caminho ORÁCULO (série do meta, sem render).

    Oráculo e não imagem: estes portões medem o Estágio D, que é o que o caminho
    C mudou. O Estágio A no ganho negativo é assunto do §39.3 e está coberto por
    `test_caso_real_negativo.py`.
    """
    import identify.classical as CL
    ac = n = 0
    falhas = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            d = Path(tmp) / f"s{seed}"
            generate_sample(str(d), seed=seed, ganho_negativo=neg, add_noise=ruido)
            m = load_sample(d)
            tc, yc = CL._clean(np.asarray(m["series"]["t"], dtype=float),
                               np.asarray(m["series"]["y"], dtype=float))
            if tc.size < 3:
                continue
            n += 1
            k = CL.identify(tc, yc).params.get("K")
            v = m["params"]["K"]
            if k is not None and (k < 0) == (v < 0) and abs(k - v) / abs(v) <= 0.05:
                ac += 1
            else:
                falhas.append(seed)
    return ac / n, n, falhas


def test_o_estrato_negativo_e_o_positivo_ESPELHADO():
    """A prova de equivalência exata do caminho C, no estrato do gerador.

    Sem ruído, a série de ganho negativo é exatamente `-1` vezes a positiva do
    mesmo seed (o modelo é linear em K). Espelhar antes do ajuste e negar o K
    depois tem, portanto, de devolver os mesmos parâmetros. Este é o portão mais
    forte do estrato: qualquer assimetria de sinal que entre em `classical.py`
    no futuro cai aqui.

    A comparação é por tolerância relativa e não bit a bit, de propósito. A
    equivalência é MATEMÁTICA, não numérica: ajustar `-y` e ajustar `y` percorre
    o mesmo `least_squares` com entradas de sinal trocado, e ponto flutuante não
    é simétrico ao sinal — somas acumulam em ordem diferente. Medido, o desvio
    fica em ~1,5e-14 relativo. `1e-9` deixa cinco ordens de margem e ainda pega
    qualquer assimetria REAL, que apareceria em 1e-2 ou pior.
    """
    import identify.classical as CL
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(9000, 9020):
            saidas = {}
            for neg in (False, True):
                d = Path(tmp) / f"s{seed}_{int(neg)}"
                generate_sample(str(d), seed=seed, ganho_negativo=neg, add_noise=False)
                m = load_sample(d)
                tc, yc = CL._clean(np.asarray(m["series"]["t"], dtype=float),
                                   np.asarray(m["series"]["y"], dtype=float))
                saidas[neg] = CL.identify(tc, yc)
            pos, neg = saidas[False], saidas[True]
            assert neg.order == pos.order, f"seed {seed}: ordem divergiu"
            assert neg.params["K"] == pytest.approx(-pos.params["K"], rel=1e-9), (
                f"seed {seed}: K = {neg.params['K']!r} contra "
                f"-({pos.params['K']!r})")
            for nome in ("tau", "theta", "wn", "zeta"):
                if pos.params[nome] is None:
                    assert neg.params[nome] is None, f"seed {seed}: {nome}"
                    continue
                assert neg.params[nome] == pytest.approx(pos.params[nome], rel=1e-9), (
                    f"seed {seed}: {nome} divergiu "
                    f"({neg.params[nome]!r} contra {pos.params[nome]!r})")


def test_pipeline_fecha_no_estrato_negativo():
    """Portão do estrato, série LIMPA: recuperação total.

    Sem ruído de medição o caminho oráculo não tem desculpa — e de fato fecha
    60/60. Um alvo abaixo de 100 % aqui esconderia regressão.
    """
    taxa, n, falhas = _taxa_de_recuperacao(neg=True, ruido=False)
    assert n >= 55, f"amostras úteis de menos: {n}"
    assert taxa == 1.0, f"K com sinal em {taxa:.1%} de {n}; falharam: {falhas}"


def test_com_ruido_o_estrato_negativo_acompanha_o_positivo():
    """Portão de PARIDADE, e não de valor absoluto.

    Com ruído, parte das amostras deixa de fechar a 5 % — mas isso NÃO é do
    ganho negativo: o caminho positivo, histórico e sem espelho nenhum, falha
    praticamente nas mesmas. O que este portão exige é que o negativo acompanhe
    o positivo; um alvo absoluto aqui mediria a dificuldade do estrato de janela
    truncada (RULING C: MAPE(K) de 127 % a 20 dB), não o caminho C.

    A pequena diferença que sobra é ESPERADA e não é defeito: o ruído é somado
    DEPOIS da inversão do sinal, então `-y_negativo = y_limpo - ruído` enquanto
    `y_positivo = y_limpo + ruído`. São realizações diferentes, e nada obriga as
    duas a falharem exatamente nas mesmas amostras. Medido: 88,3 % contra
    90,0 %, divergindo em uma única amostra de 60.
    """
    t_neg, n_neg, f_neg = _taxa_de_recuperacao(neg=True, ruido=True)
    t_pos, n_pos, f_pos = _taxa_de_recuperacao(neg=False, ruido=True)
    assert n_neg == n_pos
    assert t_pos - t_neg <= 0.05, (
        f"negativo {t_neg:.1%} contra positivo {t_pos:.1%} em n={n_neg}; "
        f"só no negativo: {sorted(set(f_neg) - set(f_pos))}")
