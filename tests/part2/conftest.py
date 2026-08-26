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


# Limiar da métrica geométrica de percepção, em px (PLANO §2.1/§2.2 revisados,
# HANDOFF_P2_7 Ruling 50). Derivado do ORÇAMENTO do critério 2.6 (3 p.p.), não do
# resultado medido: com 0,800 px de erro perpendicular a contribuição da rede à
# degradação de ζ é +0,127 p.p. (~4 % do orçamento, nem significativa). 1,0 px a
# mantém em ~5 %, com 25 % de folga sobre o medido — apertado o bastante para
# derrubar uma regressão real.
PERP_MED_MAX = 1.0
PERP_P95_MAX = 2.0

# Limiar de erro de escala, COMUM aos critérios 2.3 e 2.5 (PLANO §2.3/§2.5
# revisados, Ruling 52). Antes o 2.3 usava 1 % e o 2.5 usava 5 %, e as amostras na
# faixa intermediária eram simultaneamente "não deviam ter sido rejeitadas" (2.5) e
# "ruins o bastante para estragar o 2.3" — nenhum subconjunto satisfazia os dois.
# Escolhido 1 %, o valor fisicamente motivado: o erro de escala propaga direto para
# K e τ, e 5 % consumiria sozinho quase o dobro do orçamento do 2.6.
ESCALA_TOL = 0.01


def erro_perpendicular(x_px, y_px, meta) -> dict | None:
    """Distância PERPENDICULAR da polilinha à curva verdadeira, em px.

    Substitui a diferença VERTICAL usada até o Bloco 7. Motivo medido
    (Ruling 50): num trecho de inclinação `m`, um erro geométrico de meio pixel
    aparece como `m/2` px de erro vertical, então a métrica vertical responde à
    declividade do render e não à geometria. Correlação com a inclinação:
    vertical +0,869, perpendicular +0,326.

    PONTO CEGO, e a razão de o 2.6[theta] ser obrigatório ao lado desta métrica:
    a distância perpendicular NÃO penaliza erro AO LONGO da curva — uma polilinha
    deslocada no tempo, mas sobre a curva, pontua zero. Num degrau isso é
    exatamente o θ, que o 2.6[theta] mede.
    """
    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)
    if x_px.size < 10:
        return None
    a = meta["axis_affine"]
    t = np.asarray(meta["series"]["t"], dtype=float)
    y = np.asarray(meta["series"]["y"], dtype=float)
    dentro = (a["sx"] * x_px + a["ox"] >= t[0]) & (a["sx"] * x_px + a["ox"] <= t[-1])
    if dentro.sum() < 10:
        return None
    px, py = x_px[dentro], y_px[dentro]
    # curva verdadeira densificada em coordenadas de PIXEL
    cx = (t - a["ox"]) / a["sx"]
    cy = (y - a["oy"]) / a["sy"]
    s = np.linspace(float(cx.min()), float(cx.max()), max(4000, cx.size * 8))
    cyi = np.interp(s, cx, cy)
    # menor distância euclidiana de cada ponto à poligonal, em blocos para não
    # materializar uma matriz de 630 x 4000 de uma vez em amostras grandes
    e = np.empty(px.size, dtype=float)
    passo = 128
    for i in range(0, px.size, passo):
        dx = s[None, :] - px[i:i + passo, None]
        dy = cyi[None, :] - py[i:i + passo, None]
        e[i:i + passo] = np.sqrt((dx * dx + dy * dy).min(axis=1))
    return {"med": float(np.median(e)), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "p95": float(np.percentile(e, 95)), "n": int(e.size)}


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
