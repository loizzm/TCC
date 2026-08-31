"""Estrato OOD: reta de referência COINCIDENTE com o patamar (Ruling 55 §34.5).

O gerador sorteia distratores em posição uniforme; uma reta no patamar — o caso
real do setpoint marcado — quase nunca sai por acaso, e é exatamente onde a U-Net
perde a curva. Este teste fixa o contrato do estrato novo E a invariante
anti-vazamento que ele não pode violar.
"""
import inspect
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import generator, randomize


def test_sample_style_continua_cega_ao_sistema():
    """Regressão da guarda de tests/test_part1.py:1115. O estrato entra no
    RENDER, não no sorteio de estilo."""
    nomes = list(inspect.signature(randomize.sample_style).parameters)
    assert nomes == ["rng"], f"assinatura de sample_style mudou: {nomes}"


def test_padrao_nao_gera_reta_de_referencia():
    """Sem opt-in, o meta declara ausência da reta."""
    tmp = Path(tempfile.mkdtemp())
    try:
        meta = generator.generate_sample(tmp / "s", seed=7)
        assert meta["render"]["has_reference_line"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_opt_in_gera_reta_e_declara_no_meta():
    tmp = Path(tempfile.mkdtemp())
    try:
        meta = generator.generate_sample(tmp / "s", seed=7, reta_no_patamar=True)
        assert meta["render"]["has_reference_line"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_padrao_reproduz_bit_a_bit():
    """O opt-in não pode alterar UM BYTE do caminho padrão: os splits
    data/train|val|test e todos os números medidos dependem disso."""
    tmp = Path(tempfile.mkdtemp())
    try:
        generator.generate_sample(tmp / "a", seed=99)
        generator.generate_sample(tmp / "b", seed=99, reta_no_patamar=False)
        for nome in ("image.png", "mask.png"):
            assert (tmp / "a" / nome).read_bytes() == (tmp / "b" / nome).read_bytes(), \
                f"{nome} mudou com reta_no_patamar=False"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reta_fica_no_setpoint_e_so_ali():
    """A reta aparece NO setpoint comandado (`K x degrau`), e so ali.

    Desenho, e por que cada peca esta aqui:

    - **Diferencial pareado**, nao teste absoluto. Uma versao anterior exigia
      que a linha do setpoint fosse a mais cheia da imagem (`argmax`), e
      qualquer distrator horizontal de span completo ganhava por acaso. Outra
      exigia so "ha muita tinta nessa linha", e a moldura tambem tem.

    - **Controle com a MESMA janela** (`janela_assentada=True` nos dois lados).
      O estrato estende a janela temporal para o patamar ficar visivel; se o
      controle nao a estendesse, as duas imagens teriam curvas de geometria
      diferente e nenhum diferencial pixel a pixel seria possivel. E por isso
      que `janela_assentada` e um parametro SEPARADO de `reta_no_patamar`.

    - **Contagem de pixels que MUDARAM**, nao de pixels escuros. O gerador
      sorteia o fundo, e um limiar fixo tipo `img < 200` conta a imagem
      inteira quando o fundo e escuro (medido: seed 7 tem fundo #070707) e
      nao conta a curva quando ela e clara (seed 42, curva com cinza 200,9).
      O diferencial pareado dispensa limiar.

    - **Ancoragem no SETPOINT**, derivado do meta. `series.y` e a serie
      RUIDOSA: ancorar nela erra ate 12 px (medido, seed 123). E o ultimo
      ponto limpo ainda oscila quando a janela nao basta para assentar
      (medido, seed 22).

    Limiares de medicao real (n=40 seeds): a fracao da largura que muda dentro
    de +-12 px do setpoint tem MINIMO 0,411; fora dela o MAXIMO e 0,085 — e
    esse maximo nao e a reta, e a legenda de `loc='best'` mudando de lugar
    quando a reta ocupa o topo. 0,25 separa os dois com folga dos dois lados.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        for seed in (11, 22, 123):
            off = generator.generate_sample(tmp / f"off{seed}", seed=seed,
                                            reta_no_patamar=False,
                                            janela_assentada=True)
            on = generator.generate_sample(tmp / f"on{seed}", seed=seed,
                                           reta_no_patamar=True,
                                           janela_assentada=True)
            with Image.open(tmp / f"off{seed}" / "image.png") as im:
                A = np.asarray(im.convert("RGB"), dtype=np.int16)
            with Image.open(tmp / f"on{seed}" / "image.png") as im:
                B = np.asarray(im.convert("RGB"), dtype=np.int16)
            assert A.shape == B.shape, "a reta nao pode mudar a geometria da figura"

            mudou = (np.abs(A - B).sum(axis=2) > 0)
            altura, largura = mudou.shape
            frac = mudou.sum(axis=1) / largura

            a = on["axis_affine"]
            px = (on["params"]["K"] * on["step_amplitude"] - a["oy"]) / a["sy"]
            perto = [r for r in range(altura) if abs(r - px) <= 12]
            longe = [r for r in range(altura) if abs(r - px) > 12]
            assert perto, f"seed={seed}: setpoint em y={px:.1f} fora da figura"

            f_perto = max(frac[r] for r in perto)
            assert f_perto >= 0.25, (
                f"seed={seed}: a reta nao apareceu no setpoint (y={px:.1f}); "
                f"so {f_perto:.1%} da largura mudou ali "
                f"(esperado >= 25%; minimo medido em 40 seeds = 41,1%)"
            )

            f_longe = max(frac[r] for r in longe) if longe else 0.0
            assert f_longe < 0.25, (
                f"seed={seed}: uma linha longe do setpoint mudou {f_longe:.1%} "
                f"da largura (esperado < 25%; maximo medido em 40 seeds = 8,5%) "
                "— a reta pode nao estar ancorada no setpoint"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
