"""Criterios de aceitacao 1.1, 1.2, 1.5, 1.6, 1.7 + testes de contrato.

A suite MEDE a realidade. Quando um criterio nao e' assertavel por decisao ja
registrada (RULING C para janela truncada, RULING N para omega_n/zeta em
zeta >= 1,6), o numero e' medido e vai para o relatorio SEM assertiva -- nunca
com o limiar afrouxado.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from dataset import randomize
from dataset.generator import (
    N_SERIES,
    SCHEMA_VERSION,
    SystemSpec,
    _apply_noise,
    generate_dataset,
    generate_sample,
    load_sample,
    sample_system,
    step_response,
)
from dataset.randomize import sample_style
from identify.classical import _estimate_gain, fit_second, model_response

from tests.conftest import (
    FIXED_SNR_DB,
    MP_CTX,
    W_TRUNC,
    WORKERS,
    ZETA_SPLIT,
    meta_w,
    read_meta,
    record_block,
    record_criterion,
    record_gate_n,
    t_fast,
    t_slow,
)

# ==========================================================================
# Utilitarios de erro
# ==========================================================================
NEG_SHIFT_PX = 3.0        # deslocamento do controle negativo do criterio 1.5
# Orcamento de elementos por bloco nas distancias ponto-polilinha. Mantido baixo
# de proposito: sao 16 workers em paralelo, e um orcamento grande faz o pico de
# memoria da suite estourar a RAM da maquina (o custo em tempo e' desprezivel,
# cada bloco continua vetorizado).
_DIST_BLOCK = 500_000
MASK_RMSE_PX = 1.5        # RULING H
MASK_VBIAS_PX = 0.3       # RULING H
# RULING S: cobertura = extensao horizontal acesa / projecao de `t_window` pela
# `axis_affine` (a largura em pixels da JANELA DE DADOS), nao a largura do
# `plot_bbox_px`. Ver a docstring de `test_mask_is_clean_and_binary`.
MASK_MIN_COVERAGE = 0.93
MASK_MIN_LIT_ABS = 40     # RULING R: guarda de degeneracao (mascara vazia)
MASK_MAX_FRAC = 0.10      # teto, inalterado desde o RULING I

# Vao do padrao de tracejado do matplotlib, em multiplos da espessura da linha.
# '--' = (3.7, 1.6) ; '-.' = (6.4, 1.6, 1, 1.6) ; ':' = (1, 1.65).
DASH_GAP_K: dict[str, float] = {"-": 0.0, "--": 1.6, "-.": 1.6, ":": 1.65}


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def _mape(vals: list[float]) -> float:
    return float(100.0 * np.mean(vals)) if vals else float("nan")


def _errors(rows: list[dict]) -> dict[str, list[float]]:
    """Erros relativos por parametro.

    `K` e `theta` saem de `identify()` -- o pipeline real, sem descartar amostra
    alguma. `tau`, `wn` e `zeta` sao especificos da estrutura e sairiam `None`
    quando o AIC escolhe a outra estrutura; para nao descartar amostras (o que
    inflaria o numero) eles saem de `identify_both()` com a ordem VERDADEIRA
    imposta. A acuracia de selecao de estrutura e' reportada separadamente.
    """
    out: dict[str, list[float]] = {k: [] for k in
                                   ("K", "tau", "theta", "theta_mape", "wn", "zeta")}
    for r in rows:
        true, sel, imp = r["params"], r["sel_params"], r["imp_params"]
        out["K"].append(_rel(sel["K"], true["K"]))
        out["theta"].append(abs(sel["theta"] - true["theta"]) / r["t_dom"])
        out["theta_mape"].append(_rel(sel["theta"], true["theta"]))
        if r["order"] == "fopdt":
            out["tau"].append(_rel(imp["tau"], true["tau"]))
        else:
            out["wn"].append(_rel(imp["wn"], true["wn"]))
            out["zeta"].append(_rel(imp["zeta"], true["zeta"]))
    return out


def _summarise(rows: list[dict]) -> dict:
    """MAPE (media) e mediana do erro relativo -- a mediana mostra quanto do
    MAPE vem de poucas amostras patologicas, o que importa no estrato truncado."""
    e = _errors(rows)
    d = {k: _mape(v) for k, v in e.items()}
    d.update({f"{k}_med": (float(100.0 * np.median(v)) if v else float("nan"))
              for k, v in e.items()})
    d["n"] = len(rows)
    d["n_fopdt"] = sum(1 for r in rows if r["order"] == "fopdt")
    d["n_second"] = sum(1 for r in rows if r["order"] == "second")
    return d


def _strata(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "w>=3": [r for r in rows if r["w"] >= W_TRUNC],
        "w<3": [r for r in rows if r["w"] < W_TRUNC],
        "todos": list(rows),
    }


def _record_mape(tag: str, rows: list[dict]) -> dict:
    blk = {k: _summarise(v) for k, v in _strata(rows).items()}
    hits = sum(1 for r in rows if r["sel_order"] == r["order"])
    blk["order_acc"] = hits / max(len(rows), 1)
    blk["order_hits"] = hits
    blk["order_n"] = len(rows)
    record_block(f"mape_{tag}", blk)
    return blk


# ==========================================================================
# 1.1 — pipeline-oraculo sobre serie limpa
# ==========================================================================
def test_1_1_oracle_pipeline_clean(fits_clean):
    """MAPE < 1% em todos os parametros, no estrato nao truncado (RULING C)."""
    blk = _record_mape("clean", fits_clean)
    s = blk["w>=3"]
    targets = {"K": 1.0, "tau": 1.0, "theta": 1.0, "wn": 1.0, "zeta": 1.0}

    worst_name, worst = max(
        ((k, s[k]) for k in targets if np.isfinite(s[k])), key=lambda p: p[1]
    )
    shown = ", ".join(f"{lbl} {s[k]:.4f}%" for k, lbl in
                      (("K", "K"), ("tau", "τ"), ("theta", "θ"),
                       ("wn", "ωn"), ("zeta", "ζ")) if np.isfinite(s[k]))
    record_criterion(
        "1.1", "Pipeline-oráculo, série limpa (estrato `w ≥ 3`)",
        "MAPE < 1% em K, τ, θ, ωn, ζ",
        f"{shown} — pior = {worst_name} ({worst:.4f}%), n = {s['n']}",
        worst < 1.0,
    )
    record_gate_n("1.1", "`w ≥ 3` — K, θ", s["n"])
    record_gate_n("1.1", "`w ≥ 3` ∩ fopdt — τ", s["n_fopdt"])
    record_gate_n("1.1", "`w ≥ 3` ∩ second — ωn, ζ", s["n_second"])
    failures = {k: s[k] for k, lim in targets.items()
                if np.isfinite(s[k]) and s[k] >= lim}
    assert s["n"] >= 40, f"estrato w>=3 pequeno demais: n={s['n']}"
    assert not failures, f"MAPE >= 1% no estrato w>=3: {failures}"


# ==========================================================================
# 1.2 — pipeline com ruido a 20 dB
# ==========================================================================
def test_1_2_oracle_pipeline_noisy(fits_noisy):
    """MAPE < 5% no estrato `w >= 3`, com a excecao do RULING N para ωn/ζ."""
    blk = _record_mape("noisy", fits_noisy)
    rows = [r for r in fits_noisy if r["w"] >= W_TRUNC]
    s = blk["w>=3"]

    # ------ parametros sem excecao: K, tau, theta ------
    base = {k: s[k] for k in ("K", "tau", "theta") if np.isfinite(s[k])}

    # ------ RULING N: omega_n e zeta so sao assertados em zeta < 1.6 ------
    lo = [r for r in rows if r["order"] == "second" and r["params"]["zeta"] < ZETA_SPLIT]
    hi = [r for r in rows if r["order"] == "second" and r["params"]["zeta"] >= ZETA_SPLIT]
    lo_err = _errors(lo)
    wn_lo = _mape(lo_err["wn"])
    zeta_lo = _mape(lo_err["zeta"])

    # ------ RULING N em zeta >= 1.6: K, T_lento e NRMSE de reconstrucao ------
    hi_K = _mape([_rel(r["sel_params"]["K"], r["params"]["K"]) for r in hi])
    hi_ts = _mape([
        _rel(t_slow(r["imp_params"]["wn"], r["imp_params"]["zeta"]),
             t_slow(r["params"]["wn"], r["params"]["zeta"])) for r in hi
    ])
    hi_nrmse = float(np.mean([r["nrmse_rec"] for r in hi])) if hi else float("nan")

    measured = {"K": base.get("K"), "τ": base.get("tau"), "θ": base.get("theta"),
                f"ωn (ζ<{ZETA_SPLIT})": wn_lo, f"ζ (ζ<{ZETA_SPLIT})": zeta_lo}
    measured = {k: v for k, v in measured.items() if v is not None}
    worst_name, worst = max(((k, v) for k, v in measured.items() if np.isfinite(v)),
                            key=lambda p: p[1])
    shown = ", ".join(f"{k} {v:.3f}%" for k, v in measured.items() if np.isfinite(v))
    record_criterion(
        "1.2", f"Pipeline com ruído SNR = {FIXED_SNR_DB:.0f} dB (estrato `w ≥ 3`)",
        "MAPE < 5% (ωn/ζ só em ζ < 1,6 — RULING N)",
        f"{shown} — pior = {worst_name} ({worst:.3f}%); "
        f"n = {s['n']} (ζ<{ZETA_SPLIT}: n = {len(lo)})",
        worst < 5.0,
    )
    record_criterion(
        "1.2b", "RULING N — 2ª ordem com ζ ≥ 1,6 (ωn/ζ não identificáveis)",
        "MAPE(K) < 5%, MAPE(T_lento) < 5%, NRMSE recon. < 0,05",
        f"K = {hi_K:.3f}%, T_lento = {hi_ts:.3f}%, NRMSE = {hi_nrmse:.3e} (n = {len(hi)})",
        bool(hi_K < 5.0 and hi_ts < 5.0 and hi_nrmse < 0.05),
    )

    record_gate_n("1.2", "`w ≥ 3` — K, θ", s["n"])
    record_gate_n("1.2", "`w ≥ 3` ∩ fopdt — τ", s["n_fopdt"])
    record_gate_n("1.2", f"`w ≥ 3` ∩ second ∩ ζ < {ZETA_SPLIT} — ωn, ζ", len(lo))
    record_gate_n("1.2b", f"`w ≥ 3` ∩ second ∩ ζ ≥ {ZETA_SPLIT} — K, T_lento, NRMSE",
                  len(hi))

    nz = float(np.mean([r["nrmse_noise"] for r in fits_noisy]))
    nr = float(np.mean([r["nrmse_rec"] for r in fits_noisy]))
    record_block("nrmse_noise_mean", nz)
    record_block("nrmse_rec_over_noise", nz / max(nr, 1e-12))

    failures = {k: v for k, v in measured.items() if np.isfinite(v) and v >= 5.0}
    assert not failures, f"MAPE >= 5% no estrato w>=3 (SNR {FIXED_SNR_DB:.0f} dB): {failures}"
    assert hi_K < 5.0, f"RULING N: MAPE(K) em ζ>=1.6 = {hi_K:.3f}%"
    assert hi_ts < 5.0, f"RULING N: MAPE(T_lento) em ζ>=1.6 = {hi_ts:.3f}%"
    assert hi_nrmse < 0.05, f"RULING N: NRMSE de reconstrução em ζ>=1.6 = {hi_nrmse:.3e}"


def test_1_1_1_2_truncated_stratum_is_measured(fits_clean, fits_noisy):
    """RULING C: o estrato `w < 3` e' MEDIDO e reportado, sem assertiva.

    Nao ha assertiva de exatidao aqui de proposito: uma curva cortada antes do
    regime permanente nao determina K a 1% nem com informacao perfeita. O unico
    invariante checado e' que o estrato existe e foi medido.
    """
    for tag, rows in (("clean", fits_clean), ("noisy", fits_noisy)):
        trunc = [r for r in rows if r["w"] < W_TRUNC]
        assert trunc, f"conjunto {tag} sem estrato truncado — sorteio da janela quebrado?"
    c = _summarise([r for r in fits_clean if r["w"] < W_TRUNC])
    n = _summarise([r for r in fits_noisy if r["w"] < W_TRUNC])
    record_criterion(
        "C", "RULING C — estrato truncado `w < 3` (resultado, não critério)",
        "sem alvo: medido e reportado",
        f"limpo: MAPE(K) = {c['K']:.3f}% (n = {c['n']}); "
        f"{FIXED_SNR_DB:.0f} dB: MAPE(K) = {n['K']:.3f}% (n = {n['n']})",
        None,
    )


# ==========================================================================
# RULING N — populacao dedicada de 2a ordem (tabela de evidencia da monografia)
# ==========================================================================
N_RULING_N = 1800         # seeds sorteadas; ~metade cai em `second`
ZETA_BANDS = [(0.10, 1.00, "[0,10 ; 1,00)"), (1.00, 1.60, "[1,00 ; 1,60)"),
              (1.60, 2.20, "[1,60 ; 2,20)"), (2.20, 3.01, "[2,20 ; 3,00]")]


def _ruling_n_worker(seed: int) -> dict | None:
    """Uma série de 2ª ordem a 20 dB, ajustada com a `order` imposta.

    Não renderiza imagem: a tabela do RULING N é sobre a SÉRIE, e a figura não
    entra na conta. O ruído vem de `generator._apply_noise` — a mesma função do
    pipeline, com o mesmo estilo sorteado e `snr_db` forçado em 20 dB — para que
    a convenção do Ruling L (potência = variância) e a quantização sejam
    idênticas às da geração real.
    """
    children = np.random.SeedSequence(int(seed)).spawn(3)
    spec = sample_system(np.random.default_rng(children[0]))
    if spec.order != "second":
        return None
    style = sample_style(np.random.default_rng(children[1]))
    style.snr_db = FIXED_SNR_DB
    t = np.linspace(spec.t_start, spec.t_end, N_SERIES)
    y_clean = step_response(spec, t)
    y = _apply_noise(y_clean, style, np.random.default_rng(children[2]))

    res = fit_second(t, y)
    y_hat = model_response("second", res.params, t)
    rng_y = float(np.max(y_clean) - np.min(y_clean))
    return {
        "order": "second",
        "params": {"K": spec.K, "tau": None, "theta": spec.theta,
                   "wn": spec.wn, "zeta": spec.zeta},
        "imp_params": res.params,
        "w": (spec.t_end - spec.theta) / spec.t_dom,
        "nrmse_rec": float(np.sqrt(np.mean((y_hat - y_clean) ** 2)) / max(rng_y, 1e-12)),
        "nrmse_noise": float(np.sqrt(np.mean((y - y_clean) ** 2)) / max(rng_y, 1e-12)),
        "success": bool(res.success),
    }


@pytest.fixture(scope="session")
def ruling_n_population() -> list[dict]:
    seeds = [7_000_003 * 31 + i for i in range(N_RULING_N)]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=MP_CTX) as ex:
        rows = list(ex.map(_ruling_n_worker, seeds, chunksize=32))
    return [r for r in rows if r is not None]


def _zeta_band_table(pool: list[dict], stratum: str) -> list[dict]:
    table = []
    for a, b, label in ZETA_BANDS:
        sub = [r for r in pool if a <= r["params"]["zeta"] < b]
        if not sub:
            continue
        e_wn = [(r["imp_params"]["wn"] - r["params"]["wn"]) / r["params"]["wn"]
                for r in sub]
        e_z = [(r["imp_params"]["zeta"] - r["params"]["zeta"]) / r["params"]["zeta"]
               for r in sub]
        over = [r for r in sub if r["params"]["zeta"] > 1.0]
        table.append({
            "stratum": stratum, "band": label, "n": len(sub),
            "K": _mape([_rel(r["imp_params"]["K"], r["params"]["K"]) for r in sub]),
            "wn": _mape([abs(v) for v in e_wn]),
            "zeta": _mape([abs(v) for v in e_z]),
            "t_slow": _mape([_rel(t_slow(r["imp_params"]["wn"], r["imp_params"]["zeta"]),
                                  t_slow(r["params"]["wn"], r["params"]["zeta"]))
                             for r in over]),
            "t_fast": _mape([_rel(t_fast(r["imp_params"]["wn"], r["imp_params"]["zeta"]),
                                  t_fast(r["params"]["wn"], r["params"]["zeta"]))
                             for r in over]),
            "nrmse_rec": float(np.mean([r["nrmse_rec"] for r in sub])),
            "corr": float(np.corrcoef(e_wn, e_z)[0, 1]) if len(sub) > 2 else float("nan"),
        })
    return table


def test_ruling_n_evidence_table(ruling_n_population):
    """Tabela estratificada por ζ do RULING N, com n suficiente por faixa.

    O conjunto de aceitação de 300 amostras deixa ~10 séries por faixa de ζ —
    pouco para uma tabela de monografia. Esta população dedicada mede o mesmo
    fenômeno com ordem de grandeza a mais de amostras. A ASSERTIVA do gate
    continua sendo a 1.2b, sobre o conjunto de aceitação.
    """
    pool = ruling_n_population
    assert len(pool) > 500, f"população de 2ª ordem pequena demais: {len(pool)}"
    ok = [r for r in pool if r["w"] >= W_TRUNC]
    table = _zeta_band_table(ok, "w>=3") + _zeta_band_table(pool, "todos")
    record_block("ruling_n", table)
    record_block("ruling_n_pop", {
        "n": len(pool), "n_ok": len(ok),
        "nrmse_noise": float(np.mean([r["nrmse_noise"] for r in pool])),
        "nrmse_rec": float(np.mean([r["nrmse_rec"] for r in pool])),
    })

    hi = [r for r in ok if r["params"]["zeta"] >= ZETA_SPLIT]
    lo = [r for r in ok if r["params"]["zeta"] < ZETA_SPLIT]
    hi_K = _mape([_rel(r["imp_params"]["K"], r["params"]["K"]) for r in hi])
    hi_ts = _mape([_rel(t_slow(r["imp_params"]["wn"], r["imp_params"]["zeta"]),
                        t_slow(r["params"]["wn"], r["params"]["zeta"])) for r in hi])
    hi_nrmse = float(np.mean([r["nrmse_rec"] for r in hi]))
    lo_wn = _mape([_rel(r["imp_params"]["wn"], r["params"]["wn"]) for r in lo])
    lo_z = _mape([_rel(r["imp_params"]["zeta"], r["params"]["zeta"]) for r in lo])
    record_criterion(
        "1.2c", f"RULING N na população dedicada (n = {len(ok)} em `w ≥ 3`)",
        "ζ < 1,6: MAPE(ωn), MAPE(ζ) < 5%; ζ ≥ 1,6: MAPE(K), MAPE(T_lento) < 5% "
        "e NRMSE recon. < 0,05",
        f"ζ<1,6 (n = {len(lo)}): ωn = {lo_wn:.3f}%, ζ = {lo_z:.3f}% | "
        f"ζ≥1,6 (n = {len(hi)}): K = {hi_K:.3f}%, T_lento = {hi_ts:.3f}%, "
        f"NRMSE = {hi_nrmse:.3e}",
        bool(lo_wn < 5.0 and lo_z < 5.0 and hi_K < 5.0 and hi_ts < 5.0
             and hi_nrmse < 0.05),
    )
    record_gate_n("1.2c", f"pop. dedicada, `w ≥ 3` ∩ ζ < {ZETA_SPLIT}", len(lo))
    record_gate_n("1.2c", f"pop. dedicada, `w ≥ 3` ∩ ζ ≥ {ZETA_SPLIT}", len(hi))
    assert lo_wn < 5.0 and lo_z < 5.0, (
        f"ζ<1.6: MAPE(ωn) = {lo_wn:.3f}%, MAPE(ζ) = {lo_z:.3f}%")
    assert hi_K < 5.0, f"ζ>=1.6: MAPE(K) = {hi_K:.3f}%"
    assert hi_ts < 5.0, f"ζ>=1.6: MAPE(T_lento) = {hi_ts:.3f}%"
    assert hi_nrmse < 0.05, f"ζ>=1.6: NRMSE de reconstrução = {hi_nrmse:.3e}"


# ==========================================================================
# 1.5 — mascara x calibracao
# ==========================================================================
def _signed_normal_offsets(P: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Offset normal ASSINADO de cada pixel ate a polilinha, em pixels.

    O sinal vem do produto vetorial com a tangente local, entao o traco (que e'
    simetrico em torno do eixo da curva) contribui com media ~0 qualquer que
    seja a sua espessura. E' isso que isola o erro de CALIBRACAO da espessura
    desenhada.
    """
    A, B = poly[:-1], poly[1:]
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    L2 = np.where(L2 > 0, L2, 1.0)
    out = np.empty(P.shape[0])
    step = max(1, _DIST_BLOCK // max(A.shape[0], 1))
    for i in range(0, P.shape[0], step):
        Q = P[i:i + step]
        AP = Q[:, None, :] - A[None, :, :]
        tt = np.clip(np.einsum("mij,ij->mi", AP, AB) / L2, 0.0, 1.0)
        D = Q[:, None, :] - (A[None, :, :] + tt[:, :, None] * AB[None, :, :])
        j = np.argmin(np.einsum("mij,mij->mi", D, D), axis=1)
        m = np.arange(Q.shape[0])
        cross = D[m, j, 0] * AB[j, 1] - D[m, j, 1] * AB[j, 0]
        out[i:i + step] = cross / np.sqrt(L2[j])
    return out


def _unsigned_distances(P: np.ndarray, poly: np.ndarray) -> np.ndarray:
    A, B = poly[:-1], poly[1:]
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    L2 = np.where(L2 > 0, L2, 1.0)
    out = np.empty(P.shape[0])
    step = max(1, _DIST_BLOCK // max(A.shape[0], 1))
    for i in range(0, P.shape[0], step):
        Q = P[i:i + step]
        AP = Q[:, None, :] - A[None, :, :]
        tt = np.clip(np.einsum("mij,ij->mi", AP, AB) / L2, 0.0, 1.0)
        D = Q[:, None, :] - (A[None, :, :] + tt[:, :, None] * AB[None, :, :])
        out[i:i + step] = np.sqrt(np.einsum("mij,mij->mi", D, D)).min(axis=1)
    return out


def _mask_worker(sample_dir: str) -> dict:
    m = load_sample(sample_dir)
    mask, img = m["mask"], m["image"]
    a = m["axis_affine"]
    r = m["render"]

    lit = mask > 127
    res = {
        "same_shape": tuple(mask.shape) == tuple(img.shape[:2]),
        "binary": bool(np.isin(np.unique(mask), (0, 255)).all()),
        "n_lit": int(lit.sum()),
        "frac_lit": float(lit.mean()),
        "ls": r["line_style"],
        "marker": bool(r["has_marker"]),
        "half_width": 0.5 * r["line_width"] * r["dpi"] / 72.0,
        "size_px": list(r["size_px"]),
        "dpi": int(r["dpi"]),
        "sample_dir": str(sample_dir),
    }

    px = (m["series"]["t"] - a["ox"]) / a["sx"]
    py = (m["series"]["y"] - a["oy"]) / a["sy"]
    poly = np.column_stack([px, py])

    ys, xs = np.nonzero(lit)
    if ys.size == 0:
        return res
    P_all = np.column_stack([xs.astype(float), ys.astype(float)])
    P = P_all
    if P.shape[0] > 6000:
        P = P[np.linspace(0, P.shape[0] - 1, 6000).astype(int)]

    res["bias"] = float(np.mean(_signed_normal_offsets(P, poly)))
    shifted = np.column_stack([px, py + NEG_SHIFT_PX])
    res["bias_shift"] = float(np.mean(_signed_normal_offsets(P, shifted)))
    d = _unsigned_distances(P, poly)
    res["raw_rmse"] = float(np.sqrt(np.mean(d ** 2)))

    # mediana por coluna: erro do extrator ingenuo (RULING H, so reportado)
    cols = np.unique(xs)
    devs, n_gap = [], 0
    lo_c, hi_c = int(np.floor(px.min())), int(np.ceil(px.max()))
    col_set = set(int(c) for c in cols)
    for c in range(max(lo_c, 0), hi_c + 1):
        if c not in col_set:
            n_gap += 1
            continue
        rows_c = ys[xs == c]
        pyc = np.interp(float(c), px, py, left=np.nan, right=np.nan)
        if np.isfinite(pyc):
            devs.append(float(np.median(rows_c) - pyc))
    span_cols = max(hi_c - max(lo_c, 0) + 1, 1)
    res["medcol_err"] = float(np.mean(np.abs(devs))) if devs else float("nan")
    res["medcol_bias"] = float(np.mean(devs)) if devs else float("nan")
    res["gap_pct"] = 100.0 * n_gap / span_cols

    if r["line_style"] == "-" and not r["has_marker"]:
        # cobertura curva -> tinta: so faz sentido em traco continuo (RULING H).
        # Calculada em blocos para nao estourar a memoria com 16 workers.
        best = np.full(poly.shape[0], np.inf)
        step = max(1, _DIST_BLOCK // max(poly.shape[0], 1))
        for i in range(0, P_all.shape[0], step):
            Q = P_all[i:i + step]
            d2 = ((poly[:, None, :] - Q[None, :, :]) ** 2).sum(-1).min(axis=1)
            np.minimum(best, d2, out=best)
        res["cov_rmse"] = float(np.sqrt(np.mean(best)))
    return res


@pytest.fixture(scope="session")
def mask_stats(clean_dataset) -> list[dict]:
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=MP_CTX) as ex:
        return list(ex.map(_mask_worker, clean_dataset, chunksize=2))


def test_1_5_mask_matches_affine(mask_stats):
    """RULING H: consistencia mutua entre `mask.png` e `axis_affine`."""
    bias = np.array([s["bias"] for s in mask_stats])
    bias_shift = np.array([s["bias_shift"] for s in mask_stats])
    raw = np.array([s["raw_rmse"] for s in mask_stats])
    hw = np.array([s["half_width"] for s in mask_stats])

    rmse = float(np.sqrt(np.mean(bias ** 2)))
    rmse_shift = float(np.sqrt(np.mean(bias_shift ** 2)))
    ratio = rmse_shift / max(rmse, 1e-12)

    solid = [s for s in mask_stats if s["ls"] == "-" and not s["marker"]]
    cov = np.array([s["cov_rmse"] for s in solid])
    vbias = np.array([s["medcol_bias"] for s in solid])
    vbias = vbias[np.isfinite(vbias)]

    medcol = []
    for ls in ("-", "--", "-.", ":"):
        for mk in (False, True):
            sub = [s for s in mask_stats if s["ls"] == ls and s["marker"] == mk]
            if not sub:
                continue
            medcol.append({
                "ls": ls, "marker": mk, "n": len(sub),
                "err": float(np.nanmean([s["medcol_err"] for s in sub])),
                "gap_pct": float(np.mean([s["gap_pct"] for s in sub])),
            })

    b_no = np.array([s["bias"] for s in mask_stats if not s["marker"]])
    b_mk = np.array([s["bias"] for s in mask_stats if s["marker"]])

    n_lit = np.array([s["n_lit"] for s in mask_stats])
    frac = np.array([s["frac_lit"] for s in mask_stats])
    record_block("mask", {
        "n": len(mask_stats),
        "n_marker": int(b_mk.size), "n_nomarker": int(b_no.size),
        "bias_rmse_nomarker": float(np.sqrt(np.mean(b_no ** 2))),
        "bias_max_nomarker": float(np.max(np.abs(b_no))),
        "bias_rmse_marker": float(np.sqrt(np.mean(b_mk ** 2))),
        "bias_max_marker": float(np.max(np.abs(b_mk))),
        "bias_rmse": rmse,
        "bias_mean": float(np.mean(bias)),
        "bias_p95": float(np.percentile(np.abs(bias), 95)),
        "bias_max_abs": float(np.max(np.abs(bias))),
        "n_solid": len(solid),
        "cov_mean": float(np.mean(cov)), "cov_p95": float(np.percentile(cov, 95)),
        "cov_max": float(np.max(cov)),
        "vbias_mean": float(np.mean(vbias)),
        "vbias_max_abs": float(np.max(np.abs(vbias))),
        "raw_mean": float(np.mean(raw)), "raw_p95": float(np.percentile(raw, 95)),
        "raw_max": float(np.max(raw)),
        "half_width_max": float(np.max(hw)),
        "shift_px": NEG_SHIFT_PX, "shift_rmse": rmse_shift, "shift_ratio": ratio,
        "medcol": medcol,
        "lit": {"min": int(n_lit.min()), "median": int(np.median(n_lit)),
                "max": int(n_lit.max()), "frac_min": float(frac.min()),
                "frac_median": float(np.median(frac)), "frac_max": float(frac.max())},
    })
    record_criterion(
        "1.5", "Máscara reprojetada pela `axis_affine` × `series`",
        f"RMSE do viés normal < {MASK_RMSE_PX} px; "
        f"|viés vertical| < {MASK_VBIAS_PX} px (sólida s/ marcador)",
        f"RMSE = {rmse:.4f} px ({float(np.sqrt(np.mean(b_no ** 2))):.4f} px sem "
        f"marcador); viés vertical = {float(np.mean(vbias)):+.4f} px; "
        f"cobertura = {float(np.max(cov)):.3f} px (máx)",
        bool(rmse < MASK_RMSE_PX and abs(float(np.mean(vbias))) < MASK_VBIAS_PX
             and float(np.max(cov)) < MASK_RMSE_PX),
    )
    record_criterion(
        "1.5c", "Controle negativo do critério 1.5",
        f"deslocar a afim em {NEG_SHIFT_PX:.0f} px deve piorar o RMSE ≥ 10×",
        f"{rmse:.4f} px → {rmse_shift:.4f} px ({ratio:.1f}×)",
        ratio >= 10.0,
    )

    record_gate_n("1.5", "todas as amostras — viés normal", len(mask_stats))
    record_gate_n("1.5", "linha sólida s/ marcador — viés vertical e cobertura",
                  len(solid))
    assert rmse < MASK_RMSE_PX, f"RMSE do viés normal = {rmse:.4f} px"
    assert abs(float(np.mean(vbias))) < MASK_VBIAS_PX, \
        f"viés vertical no estrato sólido = {float(np.mean(vbias)):+.4f} px"
    assert float(np.max(cov)) < MASK_RMSE_PX, \
        f"cobertura curva→tinta (sólida s/ marcador) = {float(np.max(cov)):.3f} px"
    # a metrica tem que responder a erro de calibracao, senao nao mede nada
    assert ratio >= 10.0, f"controle negativo fraco: {ratio:.1f}×"


def _occupancy_worker(sample_dir: str) -> dict:
    """Ocupacao da mascara: barato o bastante para rodar nas 1200 amostras.

    Nao faz geometria de distancia (isso e' `_mask_worker`, so no conjunto
    limpo): so conta tinta e mede a extensao horizontal acesa.
    """
    m = load_sample(sample_dir)
    mask, img = m["mask"], m["image"]
    lit = mask > 127
    bb = m["plot_bbox_px"]
    bbox_w = float(bb[2] - bb[0])
    a = m["axis_affine"]
    t0, t1 = m["t_window"]
    # Teto atingivel: a curva ocupa apenas a janela de dados, e os eixos tem
    # margens laterais sorteadas -- entao a extensao maxima possivel NAO e' a
    # largura do bbox, e sim a projecao da janela temporal pela afim.
    span_px = abs((t1 - a["ox"]) / a["sx"] - (t0 - a["ox"]) / a["sx"])
    ys, xs = np.nonzero(lit)
    extent = float(xs.max() - xs.min()) if xs.size else 0.0
    ls = m["render"]["line_style"]
    lw_px = m["render"]["line_width"] * m["render"]["dpi"] / 72.0
    return {
        "sample_dir": str(sample_dir),
        "sample": Path(sample_dir).name,
        "same_shape": tuple(mask.shape) == tuple(img.shape[:2]),
        "binary": bool(np.isin(np.unique(mask), (0, 255)).all()),
        "n_lit": int(lit.sum()),
        "frac_lit": float(lit.mean()),
        "extent_px": extent,
        "bbox_w": bbox_w,
        "span_px": span_px,
        # RULING S: denominador = projecao da janela de dados
        "coverage": extent / span_px if span_px > 0 else 0.0,
        # denominador antigo (RULING R), mantido so para a comparacao auditavel
        "coverage_bbox": extent / bbox_w if bbox_w > 0 else 0.0,
        "ceiling": span_px / bbox_w if bbox_w > 0 else 0.0,
        # deficit maximo que o vao do tracejado pode causar numa ponta
        "gap_frac": (DASH_GAP_K[ls] * lw_px / span_px) if span_px > 0 else 0.0,
        "ls": ls,
        "size_px": list(m["render"]["size_px"]),
        "dpi": int(m["render"]["dpi"]),
        "lw_px": lw_px,
    }


@pytest.fixture(scope="session")
def mask_occupancy(clean_dataset, noisy_dataset) -> list[dict]:
    dirs = list(clean_dataset) + list(noisy_dataset)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=MP_CTX) as ex:
        return list(ex.map(_occupancy_worker, dirs, chunksize=8))


def test_mask_is_clean_and_binary(mask_occupancy):
    """Mascara binaria, mesma resolucao da imagem e nao degenerada (RULING S).

    A pergunta que este teste faz e': *a mascara contem uma curva que atravessa
    o grafico inteiro, em vez de uma mascara degenerada?* Chegar a uma grandeza
    que responda isso exigiu TRES denominadores. O registro dos dois primeiros
    erros importa mais que o resultado final -- e' a mesma classe de defeito se
    repetindo: **um limiar fixo comparado contra uma grandeza cujo teto fisico
    varia de amostra para amostra**.

    1. **Piso ABSOLUTO de pixels acesos (>= 200 px, RULING I). ERRADO.**
       Contagem absoluta nao e' invariante de escala: a largura da figura varia
       por fator ~6,7 (240..1600 px) e o traco pode ter ciclo de trabalho de 38%
       (`line_style` ':'). A tinta disponivel tem teto proporcional ao
       comprimento da curva em pixels, entao piso e teto se cruzam no canto de
       figura pequena. Medido: falhou em 1/600 (`clean/sample_00345`, 171 px,
       374x210 a 70 dpi, pontilhado) -- e nenhuma mascara correta daquele estilo
       poderia passar.

    2. **Extensao horizontal / largura do `plot_bbox_px` (>= 0.90, RULING R).
       ERRADO.** `plot_bbox_px` e' o retangulo dos EIXOS, que inclui as margens
       laterais sorteadas; a curva so ocupa a janela de dados. O teto por
       amostra e' `1/(1 + x_margin_lo + x_margin_hi)`, com as duas margens
       uniformes em [0.01, 0.06], logo o teto vive em [0.893, 0.980] -- e o
       limiar de 0.90 caiu DENTRO dessa faixa. Medido: falhou em 15/1200, das
       quais 10 tinham teto abaixo de 0.90 (impossivel de passar). Uma das que
       falharam era 1598x765 com 3685 px acesos, o que encerra qualquer duvida
       de que nao era mascara degenerada.

    3. **Extensao horizontal / projecao de `t_window` pela `axis_affine`
       (>= 0.93, RULING S). CORRETO.** O denominador passa a ser a largura em
       pixels da JANELA DE DADOS de fato, e nao a do retangulo dos eixos. Essa
       grandeza e' invariante a resolucao, a espessura do traco, ao tracejado
       E as margens, e o seu teto e' 1.0 para toda amostra -- nenhum limiar
       fixo abaixo de 1.0 pode colidir com ele.

    A troca tambem FORTALECE a assertiva em relacao ao piso original: uma
    mascara com 5000 px concentrados em 20% da largura passava no piso de 200 px
    e falha aqui.

    Assertivas finais:
    1. extensao horizontal >= 0.93 * projecao de `t_window` (RULING S);
    2. guarda de degeneracao: >= 40 px acesos (bem abaixo de qualquer caso real);
    3. teto: <= 10% dos pixels da imagem.
    """
    bad_shape = [s["sample"] for s in mask_occupancy if not s["same_shape"]]
    bad_bin = [s["sample"] for s in mask_occupancy if not s["binary"]]
    degenerate = [(s["sample"], s["n_lit"]) for s in mask_occupancy
                  if s["n_lit"] < MASK_MIN_LIT_ABS]
    too_many = [(s["sample"], s["frac_lit"]) for s in mask_occupancy
                if s["frac_lit"] > MASK_MAX_FRAC]
    narrow = [s for s in mask_occupancy if s["coverage"] < MASK_MIN_COVERAGE]

    cov = np.array([s["coverage"] for s in mask_occupancy])
    cov_bbox = np.array([s["coverage_bbox"] for s in mask_occupancy])
    ceil_ = np.array([s["ceiling"] for s in mask_occupancy])
    n_lit = np.array([s["n_lit"] for s in mask_occupancy])
    frac = np.array([s["frac_lit"] for s in mask_occupancy])
    gap = np.array([s["gap_frac"] for s in mask_occupancy])

    def q(a):
        return {"min": float(a.min()), "p1": float(np.percentile(a, 1)),
                "median": float(np.median(a)), "max": float(a.max())}

    by_style = []
    for ls in ("-", "--", "-.", ":"):
        sub = np.array([s["coverage"] for s in mask_occupancy if s["ls"] == ls])
        g = np.array([s["gap_frac"] for s in mask_occupancy if s["ls"] == ls])
        if sub.size:
            by_style.append({"ls": ls, "n": int(sub.size), "min": float(sub.min()),
                             "median": float(np.median(sub)),
                             "gap_max": float(g.max()) if g.size else 0.0})

    # Veredito vai para o relatorio ANTES das assertivas: se algo falhar, o
    # numero real tem de aparecer na tabela mestre, nao so no traceback.
    record_block("ruling_r", {
        "n": len(mask_occupancy),
        "cov": q(cov), "cov_bbox": q(cov_bbox),
        "n_below": int(len(narrow)), "threshold": MASK_MIN_COVERAGE,
        "n_below_bbox": int((cov_bbox < 0.90).sum()),
        "n_lit_min": int(n_lit.min()), "n_lit_median": int(np.median(n_lit)),
        "n_lit_max": int(n_lit.max()),
        "frac_max": float(frac.max()),
        "ceiling_min": float(ceil_.min()), "ceiling_median": float(np.median(ceil_)),
        "ceiling_max": float(ceil_.max()),
        "gap_max": float(gap.max()), "gap_p99": float(np.percentile(gap, 99)),
        "by_style": by_style,
        "offenders": [
            {"sample": s["sample"], "coverage": s["coverage"],
             "coverage_bbox": s["coverage_bbox"], "n_lit": s["n_lit"],
             "size_px": s["size_px"], "dpi": s["dpi"], "ls": s["ls"],
             "lw_px": s["lw_px"], "gap_frac": s["gap_frac"]}
            for s in sorted(narrow, key=lambda s: s["coverage"])[:6]
        ],
        "worst": [
            {"sample": s["sample"], "coverage": s["coverage"],
             "n_lit": s["n_lit"], "size_px": s["size_px"], "dpi": s["dpi"],
             "ls": s["ls"], "lw_px": s["lw_px"], "gap_frac": s["gap_frac"]}
            for s in sorted(mask_occupancy, key=lambda s: s["coverage"])[:5]
        ],
    })
    record_criterion(
        "R", "RULING S — máscara não degenerada (curva atravessa a janela)",
        f"extensão horizontal ≥ {MASK_MIN_COVERAGE:.2f}·projeção de `t_window`; "
        f"≥ {MASK_MIN_LIT_ABS} px acesos; ≤ {MASK_MAX_FRAC:.0%} da imagem",
        f"cobertura: mín {cov.min():.4f}, p1 {np.percentile(cov, 1):.4f}, "
        f"mediana {np.median(cov):.4f}, máx {cov.max():.4f} — "
        f"{len(narrow)}/{len(mask_occupancy)} abaixo de {MASK_MIN_COVERAGE:.2f}; "
        f"mín de px acesos = {int(n_lit.min())}; fração máxima = {frac.max():.4f}",
        not (narrow or degenerate or too_many or bad_shape or bad_bin),
    )

    assert not bad_shape, f"máscara com resolução != imagem: {bad_shape[:3]}"
    assert not bad_bin, f"máscara não binária: {bad_bin[:3]}"
    assert not degenerate, (
        f"máscara degenerada, < {MASK_MIN_LIT_ABS} px acesos: {degenerate[:3]}")
    assert not too_many, f"máscara com > {MASK_MAX_FRAC:.0%} da imagem: {too_many[:3]}"
    assert not narrow, (
        f"cobertura horizontal < {MASK_MIN_COVERAGE:.2f} em "
        f"{len(narrow)}/{len(mask_occupancy)} amostras; pior = "
        + ", ".join(f"{s['sample']} {s['coverage']:.4f}"
                    for s in sorted(narrow, key=lambda s: s["coverage"])[:3]))


# ==========================================================================
# 1.6 — determinismo bit-a-bit
# ==========================================================================
RECORDED_DET: dict = {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta_without_id(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    m.pop("sample_id", None)
    return m


@pytest.mark.parametrize("seed", [0, 1, 7, 12345, 987654321])
@pytest.mark.parametrize("add_noise", [True, False])
def test_1_6_determinism_same_seed(tmp_path, seed, add_noise):
    """Mesma seed duas vezes => bytes identicos de image.png e mask.png."""
    a = tmp_path / "a" / "sample_00000"
    b = tmp_path / "b" / "sample_00000"
    generate_sample(a, seed, add_noise=add_noise)
    generate_sample(b, seed, add_noise=add_noise)

    ha_i, hb_i = _sha256(a / "image.png"), _sha256(b / "image.png")
    ha_m, hb_m = _sha256(a / "mask.png"), _sha256(b / "mask.png")
    meta_eq = _meta_without_id(a / "meta.json") == _meta_without_id(b / "meta.json")

    if add_noise:
        det = RECORDED_DET.setdefault("rows", [])
        det.append({"seed": seed, "image": ha_i, "mask": ha_m, "meta_equal": meta_eq})
        record_block("determinism", det)
        ok = all(d["meta_equal"] for d in det)
        record_criterion(
            "1.6", "Determinismo bit-a-bit (mesma seed ⇒ mesmos bytes)",
            "sha256 idêntico de image.png e mask.png + meta.json idêntico",
            f"{len(det)} seeds × 2 gerações: todos idênticos" if ok else "divergência",
            ok and ha_i == hb_i and ha_m == hb_m,
        )

    assert ha_i == hb_i, f"image.png difere para seed={seed}, add_noise={add_noise}"
    assert ha_m == hb_m, f"mask.png difere para seed={seed}, add_noise={add_noise}"
    assert meta_eq, f"meta.json difere para seed={seed}, add_noise={add_noise}"


def test_1_6_dataset_independent_of_workers(tmp_path):
    """`generate_dataset` produz os mesmos bytes com 1 ou 8 workers (contrato §5)."""
    d1 = generate_dataset(tmp_path / "w1", 8, seed=31, workers=1, add_noise=True)
    d8 = generate_dataset(tmp_path / "w8", 8, seed=31, workers=8, add_noise=True)
    assert len(d1) == len(d8) == 8
    for a, b in zip(sorted(d1), sorted(d8)):
        for name in ("image.png", "mask.png"):
            assert _sha256(Path(a) / name) == _sha256(Path(b) / name), \
                f"{name} depende do número de workers em {a}"
        assert _meta_without_id(Path(a) / "meta.json") == \
            _meta_without_id(Path(b) / "meta.json")


# ==========================================================================
# 1.7 — desempenho
# ==========================================================================
@pytest.mark.slow
def test_1_7_generation_throughput(tmp_path):
    """200 amostras com workers=16, extrapolado linearmente para 6000."""
    import time

    n = 200
    t0 = time.perf_counter()
    dirs = generate_dataset(tmp_path / "perf", n, seed=777, workers=WORKERS,
                            add_noise=True)
    dt = time.perf_counter() - t0
    assert len(dirs) == n
    minutes = dt / n * 6000 / 60.0
    record_block("perf", {"n": n, "workers": WORKERS, "seconds": dt,
                          "per_sample": dt / n, "minutes_6000": minutes})
    record_criterion(
        "1.7", "Tempo de geração extrapolado para 6000 amostras",
        "< 15 min (folga 2× sobre os 30 min do PLANO)",
        f"{dt:.2f} s para {n} amostras ⇒ {minutes:.2f} min",
        minutes < 15.0,
    )
    assert minutes < 15.0, f"extrapolação = {minutes:.2f} min para 6000 amostras"


# ==========================================================================
# Verificacao cruzada dos dois modelos independentes
# ==========================================================================
def test_model_cross_check():
    """`step_response` (dataset) × `model_response` (identify) coincidem.

    A 2ª ordem de `identify` e' uma expressao analitica unica (Ruling M), sem os
    tres ramos do contrato §1; comparam-se VALORES, nunca estrutura. Perto de
    zeta = 1 a diferenca e' proporcional a K, entao a tolerancia e' relativa a K.
    """
    rng = np.random.default_rng(2024)
    max_abs = 0.0
    max_rel = 0.0
    n = 200
    for i in range(n):
        order = "fopdt" if i % 2 == 0 else "second"
        K = float(np.exp(rng.uniform(np.log(0.2), np.log(20.0))))
        if order == "fopdt":
            tau, wn, zeta = float(np.exp(rng.uniform(np.log(0.05), np.log(50.0)))), None, None
            t_dom = tau
        else:
            tau = None
            wn = float(np.exp(rng.uniform(np.log(0.02), np.log(20.0))))
            # zeta atravessando 1, incluindo a vizinhanca exata do ramo critico
            pick = i % 6
            if pick == 1:
                zeta = 1.0
            elif pick == 3:
                zeta = 1.0 + float(rng.uniform(-1e-7, 1e-7))
            else:
                zeta = float(rng.uniform(0.05, 4.0))
            t_dom = (1.0 / (zeta * wn) if zeta <= 1.0
                     else (zeta + np.sqrt(zeta * zeta - 1.0)) / wn)
        theta = float(rng.uniform(0.05, 1.0)) * t_dom
        t_end = theta + float(rng.uniform(0.5, 6.0)) * t_dom
        spec = SystemSpec(order, K, tau, theta, wn, zeta, 0.0, t_end, 1.0)
        t = np.linspace(0.0, t_end, 257)
        a = step_response(spec, t)
        b = model_response(order, {"K": K, "tau": tau, "theta": theta,
                                   "wn": wn, "zeta": zeta}, t)
        diff = float(np.max(np.abs(a - b)))
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, diff / K)
        assert diff <= max(1e-9, 1e-6 * K), (
            f"divergência {diff:.3e} em order={order} K={K:.4g} zeta={zeta}")
    record_block("cross_check", {"n": n, "max_abs": max_abs, "max_rel": max_rel})


# ==========================================================================
# Contrato do meta.json e da API
# ==========================================================================
_META_KEYS = {"schema_version", "sample_id", "seed", "order", "params",
              "step_amplitude", "t_window", "plot_bbox_px", "axis_affine",
              "ticks", "series", "noise", "render"}
_RENDER_KEYS = {"dpi", "size_px", "has_grid", "has_legend", "line_width",
                "line_style", "line_color", "bg_color", "has_marker",
                "has_title", "has_xlabel", "has_ylabel", "n_annotations",
                "n_distractors", "n_spines", "snr_db", "quantization_levels",
                "has_reference_line"}


def test_meta_contract(clean_dataset, noisy_dataset):
    """Todas as chaves do contrato §4, com os tipos e invariantes certos."""
    for sample_dir in list(clean_dataset) + list(noisy_dataset):
        m = read_meta(sample_dir)
        assert set(m) == _META_KEYS, f"chaves erradas em {sample_dir}: {set(m) ^ _META_KEYS}"
        assert m["schema_version"] == SCHEMA_VERSION
        assert m["sample_id"] == Path(sample_dir).name
        assert isinstance(m["seed"], int)
        assert m["order"] in ("fopdt", "second")
        assert m["step_amplitude"] == 1.0

        p = m["params"]
        assert set(p) == {"K", "tau", "theta", "wn", "zeta"}
        assert p["K"] > 0 and p["theta"] > 0
        if m["order"] == "fopdt":
            assert p["tau"] is not None and p["wn"] is None and p["zeta"] is None
        else:
            assert p["tau"] is None and p["wn"] is not None and p["zeta"] is not None

        t0, t1 = m["t_window"]
        assert t0 == 0.0 and t1 > t0
        bbox = m["plot_bbox_px"]
        assert len(bbox) == 4 and all(isinstance(v, int) for v in bbox)
        assert bbox[0] < bbox[2] and bbox[1] < bbox[3]

        a = m["axis_affine"]
        assert set(a) == {"sx", "ox", "sy", "oy"}
        assert a["sx"] > 0, "sx deve ser positivo (px cresce com t)"
        assert a["sy"] < 0, "sy deve ser negativo (py cresce para baixo)"

        assert set(m["ticks"]) == {"x", "y"}
        for axis in ("x", "y"):
            assert len(m["ticks"][axis]) >= 2, f"menos de 2 ticks em {axis}"
            assert all(len(e) == 2 for e in m["ticks"][axis])

        s = m["series"]
        assert len(s["t"]) == N_SERIES == 512 and len(s["y"]) == N_SERIES
        assert np.allclose(np.diff(s["t"]), (t1 - t0) / (N_SERIES - 1)), \
            "series.t não é uniforme"
        assert np.isclose(s["t"][0], t0) and np.isclose(s["t"][-1], t1)
        assert all(np.isfinite(s["y"]))

        nz = m["noise"]
        assert set(nz) == {"enabled", "snr_db", "quantization_levels"}
        assert isinstance(nz["enabled"], bool)
        assert set(m["render"]) == _RENDER_KEYS
        assert m["render"]["size_px"][0] >= 240 and m["render"]["size_px"][1] >= 180


def test_series_is_what_was_drawn(clean_dataset, noisy_dataset):
    """`series.y` limpa == resposta analítica; com ruído, difere (contrato §4)."""
    for sample_dir in clean_dataset[:40]:
        m = read_meta(sample_dir)
        t = np.asarray(m["series"]["t"])
        y = np.asarray(m["series"]["y"])
        y_true = model_response(m["order"], m["params"], t)
        assert np.max(np.abs(y - y_true)) <= 1e-6 * m["params"]["K"], \
            f"series.y limpa não bate com o modelo em {sample_dir}"
    diffs = []
    for sample_dir in noisy_dataset[:40]:
        m = read_meta(sample_dir)
        t = np.asarray(m["series"]["t"])
        y = np.asarray(m["series"]["y"])
        y_true = model_response(m["order"], m["params"], t)
        rngy = float(np.max(y_true) - np.min(y_true))
        diffs.append(float(np.sqrt(np.mean((y - y_true) ** 2)) / max(rngy, 1e-12)))
    # SNR de 20 dB sobre a VARIANCIA (Ruling L) => ruido claramente presente
    assert float(np.median(diffs)) > 1e-3, \
        "series.y do conjunto ruidoso está limpa demais — ruído não foi aplicado?"


def test_load_sample_shapes(clean_dataset):
    """`load_sample` devolve series como ndarray e os dois PNGs coerentes."""
    m = load_sample(clean_dataset[0])
    assert isinstance(m["series"]["t"], np.ndarray)
    assert isinstance(m["series"]["y"], np.ndarray)
    assert m["image"].dtype == np.uint8 and m["mask"].dtype == np.uint8
    assert m["image"].shape[:2] == m["mask"].shape
    w, h = m["render"]["size_px"]
    assert m["image"].shape[:2] == (h, w)


def test_baselines_vs_identify(fits_clean):
    """Compara os 3 baselines clássicos com `identify` nas mesmas séries limpas.

    Só faz sentido nas amostras FOPDT: os baselines são estimadores FOPDT de
    dois/três pontos. Não há assertiva de limiar — é a tabela comparativa da
    monografia. O único invariante checado é que `identify` não é pior que o
    melhor baseline no estrato não truncado, que é o que justifica o Estágio D.
    """
    fopdt = [r for r in fits_clean if r["order"] == "fopdt"]
    assert fopdt, "conjunto limpo sem amostras FOPDT"

    methods = [("identify", None), ("tangente", "tangent"),
               ("Smith", "smith"), ("Sundaresan–Krishnaswamy", "sk")]
    table = []
    summary: dict[tuple[str, str], float] = {}
    for label, key in methods:
        for stratum, sub in (("w>=3", [r for r in fopdt if r["w"] >= W_TRUNC]),
                             ("w<3", [r for r in fopdt if r["w"] < W_TRUNC])):
            eK, etau, eth, n_ok = [], [], [], 0
            for r in sub:
                true = r["params"]
                est = r["imp_params"] if key is None else r["baselines"][key]
                K, tau, th = est.get("K"), est.get("tau"), est.get("theta")
                if not (K is not None and tau is not None and th is not None
                        and np.isfinite(K) and np.isfinite(tau) and np.isfinite(th)):
                    continue
                n_ok += 1
                eK.append(_rel(K, true["K"]))
                etau.append(_rel(tau, true["tau"]))
                eth.append(abs(th - true["theta"]) / r["t_dom"])
            row = {"method": label, "stratum": stratum, "n": len(sub),
                   "coverage": n_ok / max(len(sub), 1),
                   "K": _mape(eK), "tau": _mape(etau), "theta": _mape(eth)}
            table.append(row)
            summary[(label, stratum)] = row["tau"]
    record_block("baselines", table)

    best_baseline = min(
        v for (label, st), v in summary.items()
        if st == "w>=3" and label != "identify" and np.isfinite(v)
    )
    ours = summary[("identify", "w>=3")]
    record_criterion(
        "B", "Baselines clássicos × `identify` (FOPDT limpo, `w ≥ 3`)",
        "sem alvo: comparação da monografia",
        f"MAPE(τ): identify = {ours:.4f}% vs melhor baseline = {best_baseline:.4f}%",
        None,
    )
    assert ours <= best_baseline, (
        f"identify (MAPE τ = {ours:.4f}%) pior que o melhor baseline "
        f"({best_baseline:.4f}%) — o Estágio D não se justifica")


GAIN_MAPE_MAX = 1.0       # criterio G: MAPE de `_estimate_gain` no estrato w<3
GAIN_SHORTCUT_MIN = 10.0  # controle positivo: quanto max(y) erra no mesmo estrato


def _gain_worker(sample_dir: str) -> dict | None:
    """`_estimate_gain` x o atalho proibido max(y), numa serie limpa FOPDT."""
    m = read_meta(sample_dir)
    if m["order"] != "fopdt":
        return None
    t = np.asarray(m["series"]["t"], dtype=float)
    y = np.asarray(m["series"]["y"], dtype=float)
    K = float(m["params"]["K"])
    k_est = _estimate_gain(t, y)
    return {"w": meta_w(m), "ok": bool(np.isfinite(k_est)),
            "e_est": abs(k_est - K) / K if np.isfinite(k_est) else float("nan"),
            "e_max": abs(float(np.max(y)) - K) / K}


def test_gain_estimator_does_not_fall_back_to_max_y(clean_dataset):
    """`_estimate_gain` recupera K em janela truncada — e max(y) não.

    Existe por TESTE DE MUTAÇÃO (HANDOFF §3.5). O mutante que devolve `max(y)`
    em `_estimate_gain` — que foi um bug real, corrigido na Tarefa 2 —
    atravessava a suíte inteira com 30 passed. Motivo: `_estimate_gain` só
    alimenta os três baselines clássicos, e a única assertiva sobre eles
    (`test_baselines_vs_identify`) compara `identify` com o MELHOR baseline;
    piorar todos os baselines de uma vez não quebra essa comparação. Ou seja, a
    tabela de baselines da monografia estava sem guarda.

    O estrato é o truncado (`w < 3`), onde os dois se separam: sem regime
    permanente na janela, `max(y)` subestima K por construção. Fora dele a
    curva já assentou e `max(y)` acerta — por isso a assertiva **precisa** viver
    aqui. O alvo é acompanhado de um **controle positivo** (o atalho tem de
    errar ≥ 10%), que impede o teste de virar vácuo caso a distribuição de
    janelas mude e o estrato deixe de ser difícil.

    Medido em 2894 séries FOPDT limpas com `w < 3`: `_estimate_gain` erra
    **0,000000%** (máximo, não média) com cobertura 100%, contra **30,3%** de
    `max(y)` (mínimo por amostra: 4,98%).
    """
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=MP_CTX) as ex:
        rows = [r for r in ex.map(_gain_worker, clean_dataset, chunksize=8)
                if r is not None]
    trunc = [r for r in rows if r["w"] < W_TRUNC]
    assert trunc, "conjunto limpo sem FOPDT truncado"

    ok = [r for r in trunc if r["ok"]]
    coverage = len(ok) / len(trunc)
    mape_est = _mape([r["e_est"] for r in ok])
    mape_max = _mape([r["e_max"] for r in trunc])

    record_block("gain", {
        "n_fopdt": len(rows), "n_trunc": len(trunc), "coverage": coverage,
        "mape_est": mape_est, "mape_shortcut": mape_max,
        "ratio": mape_max / max(mape_est, 1e-12),
        "worst_est": 100.0 * max((r["e_est"] for r in ok), default=float("nan")),
    })
    record_criterion(
        "G", "`_estimate_gain` em janela truncada (`w < 3`, FOPDT limpo)",
        f"MAPE < {GAIN_MAPE_MAX}% e cobertura = 100%; controle positivo: o "
        f"atalho max(y) erra ≥ {GAIN_SHORTCUT_MIN}% no mesmo estrato",
        f"MAPE = {mape_est:.4f}% (n = {len(trunc)}, cobertura "
        f"{coverage:.3f}) vs max(y) = {mape_max:.2f}%",
        bool(mape_est < GAIN_MAPE_MAX and coverage == 1.0
             and mape_max >= GAIN_SHORTCUT_MIN),
    )
    record_gate_n("G", "`w < 3` ∩ fopdt ∩ limpo — ganho estático", len(trunc))
    assert coverage == 1.0, f"`_estimate_gain` devolveu nan em {1 - coverage:.1%}"
    assert mape_est < GAIN_MAPE_MAX, (
        f"MAPE de `_estimate_gain` = {mape_est:.4f}% no estrato truncado — "
        "o estimador voltou a um atalho?")
    assert mape_max >= GAIN_SHORTCUT_MIN, (
        f"controle positivo fraco: max(y) erra só {mape_max:.2f}% neste "
        "estrato, então o teste não separa mais o atalho do estimador")


def test_sample_style_signature_cannot_see_the_label():
    """Anti-vazamento ESTRUTURAL: `sample_style` só recebe o RNG de estilo."""
    sig = inspect.signature(randomize.sample_style)
    names = list(sig.parameters)
    assert names == ["rng"], f"assinatura de sample_style mudou: {names}"
    forbidden = {"spec", "system", "order", "params", "K", "tau", "theta", "wn", "zeta"}
    assert not (set(names) & forbidden)
