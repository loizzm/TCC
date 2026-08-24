"""Fixtures da Parte 2. NÃO reescreve o relatório da Parte 1 (ver G0.4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset.generator import load_sample

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "part2_strata.md"
N_EVAL = 300           # amostras de `data/test` usadas nos portões geométricos
RESULTS_P2: dict = {"criteria": {}, "blocks": {}}


def record_p2(cid: str, name: str, target: str, measured: str, ok: bool | None) -> None:
    RESULTS_P2["criteria"][cid] = {
        "name": name, "target": target, "measured": measured, "ok": ok,
    }


def to_gray(image: np.ndarray) -> np.ndarray:
    """RGB uint8 -> cinza uint8. Luma ITU-R BT.601, sem depender do OpenCV."""
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (image.astype(np.float32) @ w).round().astype(np.uint8)


@pytest.fixture(scope="session")
def test_samples() -> list[dict]:
    root = ROOT / "data" / "test"
    dirs = sorted(root.glob("sample_*"))[:N_EVAL]
    assert dirs, "rode o Passo 12 do Bloco 0: data/test está vazio"
    return [load_sample(d) for d in dirs]


def _write_report_p2() -> None:
    """Gera reports/part2_strata.md com os critérios acumulados em RESULTS_P2."""
    L: list[str] = []
    A = L.append
    A("# Parte 2 — Estratificação e critérios")
    A("")
    A("Relatório gerado automaticamente por `tests/part2/` "
      "(`pytest_sessionfinish` em `tests/part2/conftest.py`). Não editar à mão.")
    A("")
    A("| Critério | Nome | Alvo | Medido | Veredito |")
    A("|---|---|---|---|---|")
    for cid, c in RESULTS_P2["criteria"].items():
        v = "✅" if c["ok"] else ("❓" if c["ok"] is None else "❌")
        A(f"| {cid} | {c['name']} | {c['target']} | {c['measured']} | {v} |")
    A("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    if not RESULTS_P2["criteria"]:
        return
    try:
        _write_report_p2()
    except Exception as exc:  # pragma: no cover - o relatorio nunca derruba a sessao
        print(f"\n[tests/part2/conftest] falha ao escrever {REPORT_PATH}: {exc!r}")
