"""Portão G0.1/G0.2/G0.3/G0.4: o ambiente da Parte 2 existe e o dispositivo é conhecido."""
import shutil

import pytest


def test_g0_1_dependencias_importam():
    import cv2            # noqa: F401
    import pytesseract    # noqa: F401
    import skimage        # noqa: F401
    import torch          # noqa: F401


def test_g0_1_binario_tesseract_no_path():
    assert shutil.which("tesseract") is not None, (
        "instale o pacote de sistema: sudo apt-get install tesseract-ocr"
    )


def test_g0_2_dispositivo_declarado():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev in ("cuda", "cpu")
    print(f"TCC_DEVICE={dev}")


def test_g0_4_rodar_parte2_nao_apaga_relatorio_da_parte1(tmp_path):
    """Portão G0.4: sessão sem critérios da Parte 1 não reescreve o relatório."""
    import tests.conftest as c1

    saved_criteria = dict(c1.RESULTS["criteria"])
    c1.RESULTS["criteria"] = {}
    try:
        antes = c1.REPORT_PATH.read_bytes() if c1.REPORT_PATH.exists() else None
        c1.pytest_sessionfinish(_FakeSession(), 0)
        depois = c1.REPORT_PATH.read_bytes() if c1.REPORT_PATH.exists() else None
        assert antes == depois, "o relatório da Parte 1 foi alterado"
    finally:
        c1.RESULTS["criteria"] = saved_criteria


class _FakeSession:
    class config:
        class option:
            markexpr = ""
            keyword = ""
        args = ["tests/part2"]


def test_g0_3_splits_disjuntos_e_completos():
    """Portão G0.3: 4200/900/900 e nenhuma seed compartilhada entre splits."""
    from pathlib import Path
    import json

    esperado = {"train": 4200, "val": 900, "test": 900}
    seeds: dict[str, set[int]] = {}
    for split, n in esperado.items():
        root = Path("data") / split
        dirs = sorted(root.glob("sample_*"))
        assert len(dirs) == n, f"{split}: {len(dirs)} != {n}"
        seeds[split] = {
            json.loads((d / "meta.json").read_text(encoding="utf-8"))["seed"]
            for d in dirs
        }
    assert not seeds["train"] & seeds["val"]
    assert not seeds["train"] & seeds["test"]
    assert not seeds["val"] & seeds["test"]
