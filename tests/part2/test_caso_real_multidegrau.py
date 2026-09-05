"""Regressão do par controlado de multi-degrau (spec de 04/09/2026).

As três imagens vêm de `rg_multidegrau.py`, com a verdade declarada na própria
função de transferência — não há estimativa envolvida. Ver o docstring daquele
arquivo para os sistemas.

O PAR CONTROLADO é `caso_real_multi_fopdt.png` (dois degraus) contra
`caso_real_multi_fopdt_1degrau.png` (um degrau): mesma planta, mesmo render,
mesmos limites de eixo. Uma variável muda, e o resultado muda com ela. Enquanto
os dois coexistirem, nenhuma explicação alternativa sobrevive (Ruling 66).
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIX = Path(__file__).resolve().parents[1] / "fixtures"

SUB = {"path": FIX / "caso_real_multi_sub.png", "order": "second",
       "K": -1.0, "wn": 3.16227766, "zeta": 0.31622777, "theta": 1.0}
FOPDT_2 = {"path": FIX / "caso_real_multi_fopdt.png", "order": "fopdt",
           "K": 2.0, "tau": 0.5, "theta": 1.5}
FOPDT_1 = {"path": FIX / "caso_real_multi_fopdt_1degrau.png", "order": "fopdt",
           "K": 2.0, "tau": 0.5, "theta": 1.5}

# Mesma tolerância de `test_caso_real_rg.py`.
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


def _erro(medido, esperado):
    return abs(medido - esperado) / abs(esperado)


def _confere(r, caso):
    assert r["ok"], f"sem físico: reason={r['reason']!r} cal={r['calibration']}"
    assert r["order"] == caso["order"], (
        f"ordem {r['order']!r}, esperada {caso['order']!r}; params={r['params']}")
    for nome in [k for k in ("K", "tau", "wn", "zeta", "theta") if k in caso]:
        e = _erro(r["params"][nome], caso[nome])
        assert e <= TOL, (f"{nome} = {r['params'][nome]:.4f}, esperado "
                          f"{caso[nome]} (erro {e:.1%}, tolerância {TOL:.0%})")


# --------------------------------------------------------------------------- #
# Portão 1 — recuperação do 1º degrau
# --------------------------------------------------------------------------- #

def test_sub_recupera_o_primeiro_degrau(modelo):
    """Sem a truncagem esta imagem é RECUSADA (`ajuste_inconsistente`, nrmse
    0,143): a série inteira não sustenta modelo nenhum."""
    r = _roda(SUB, modelo)
    assert r["truncado_em"] is not None, "não truncou — a imagem tem dois degraus"
    _confere(r, SUB)


def test_fopdt_recupera_o_primeiro_degrau(modelo):
    """Sem a truncagem esta imagem devolve número CONFIANTE e errado: tau sai
    3,777 s contra 0,5 s verdadeiro, 7,6x, com nrmse 0,069 — sob o limiar de
    recusa. É o erro silencioso que motivou a spec."""
    r = _roda(FOPDT_2, modelo)
    assert r["truncado_em"] is not None, "não truncou — a imagem tem dois degraus"
    _confere(r, FOPDT_2)


# --------------------------------------------------------------------------- #
# Portão 2 — CONTROLE NEGATIVO do par
# --------------------------------------------------------------------------- #

def test_um_degrau_nao_dispara_truncagem(modelo):
    """A metade de controle do par. É este teste que impede alguém de apertar o
    `_GANHO_MIN` até truncar tudo: baixá-lo o bastante quebra AQUI antes de
    quebrar em qualquer outro lugar."""
    r = _roda(FOPDT_1, modelo)
    assert r["truncado_em"] is None, (
        f"truncou em t={r['truncado_em']} uma figura de UM degrau "
        f"(ganho {r['ganho_truncagem']}) — falso positivo no controle negativo")
    _confere(r, FOPDT_1)


def test_o_par_difere_em_uma_variavel_so():
    """Documenta em CÓDIGO que o par é controlado: mesma planta, mesmo render,
    mesmos eixos. Se alguém regerar as fixtures mudando o render de uma e não da
    outra, o par deixa de isolar a variável e este teste avisa."""
    from identify.calibrate import calibrate

    img2 = np.asarray(Image.open(FOPDT_2["path"]).convert("RGB"))
    img1 = np.asarray(Image.open(FOPDT_1["path"]).convert("RGB"))
    assert img2.shape == img1.shape, (
        "as duas metades do par têm tamanhos de imagem diferentes")

    cal2 = calibrate(img2)
    cal1 = calibrate(img1)
    assert cal2.bbox_px == cal1.bbox_px, (
        f"molduras diferentes: {cal2.bbox_px} contra {cal1.bbox_px} — o par "
        f"deixou de diferir em uma variável só")
