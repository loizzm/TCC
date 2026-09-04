"""Checkpoint por época, para que a seleção não fique refém do `IoU_val`.

Por que existe (§40.9). `train_unet.py` guarda só o melhor checkpoint segundo
o `IoU_val`, e essa métrica é quase CEGA ao defeito do platô: o estrato que a
rede não sabe segmentar custa 5 pontos de IoU (0,7814 -> 0,7304) enquanto a
cobertura do platô nele desaba 42 (0,944 -> 0,527). O platô é uma linha fina,
poucos pixels contra o corpo da curva, e o IoU é dominado pelo corpo.

A consequência é que a época que aprender o platô pode não ser a selecionada.
Guardar todas e escolher depois, pela métrica de platô, resolve isso sem tocar
em métrica nenhuma — o `IoU_val` continua sendo reportado igual, para os
números seguirem comparáveis com as rodadas históricas.

Opt-in: sem o flag, o comportamento é o de antes.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from dataset.generator import generate_dataset

RAIZ = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mini_corpus():
    """Duas pastas minúsculas, o bastante para uma época rodar."""
    with tempfile.TemporaryDirectory() as tmp:
        tr = str(Path(tmp) / "tr")
        va = str(Path(tmp) / "va")
        generate_dataset(tr, n=2, seed=515151, workers=1, add_noise=False)
        generate_dataset(va, n=2, seed=525252, workers=1, add_noise=False)
        yield tr, va, tmp


def _treina(tr, va, out, extra=()):
    cmd = [sys.executable, "train_unet.py",
           "--epochs", "2", "--batch", "1", "--size", "64", "--base", "4",
           "--in-ch", "3", "--device", "cpu", "--batches-per-epoch", "1",
           "--train-dir", tr, "--val-dir", va, "--out", out, *extra]
    p = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, f"treino falhou:\n{p.stdout}\n{p.stderr}"
    return p.stdout


def test_sem_o_flag_nada_alem_do_melhor_e_escrito(mini_corpus):
    tr, va, tmp = mini_corpus
    d = Path(tmp) / "sem_flag"
    d.mkdir()
    _treina(tr, va, str(d / "m.pt"))
    assert sorted(p.name for p in d.iterdir()) == ["m.pt"]


def test_com_o_flag_sai_um_checkpoint_por_epoca(mini_corpus):
    tr, va, tmp = mini_corpus
    d = Path(tmp) / "com_flag"
    ep = Path(tmp) / "epocas"
    d.mkdir()
    _treina(tr, va, str(d / "m.pt"), ("--save-epoch-dir", str(ep)))
    nomes = sorted(p.name for p in ep.iterdir())
    assert nomes == ["epoca_00.pt", "epoca_01.pt"], nomes
    assert (d / "m.pt").exists(), "o melhor por IoU_val continua sendo escrito"


def test_os_checkpoints_por_epoca_carregam(mini_corpus):
    """Não basta existir: têm de ser state_dicts utilizáveis."""
    import torch
    from identify.extract import UNet
    tr, va, tmp = mini_corpus
    d = Path(tmp) / "carrega"
    ep = Path(tmp) / "epocas_carrega"
    d.mkdir()
    _treina(tr, va, str(d / "m.pt"), ("--save-epoch-dir", str(ep)))
    for arq in sorted(ep.iterdir()):
        sd = torch.load(arq, map_location="cpu")
        UNet(base=4, in_ch=3).load_state_dict(sd)


def test_o_relatorio_de_IoU_val_nao_muda_de_formato(mini_corpus):
    """As rodadas históricas são comparadas por este texto; ele não pode mudar."""
    tr, va, tmp = mini_corpus
    d = Path(tmp) / "formato"
    ep = Path(tmp) / "epocas_formato"
    d.mkdir()
    saida = _treina(tr, va, str(d / "m.pt"), ("--save-epoch-dir", str(ep)))
    assert "epoca 00  IoU_val=" in saida
    assert "melhor IoU_val=" in saida
