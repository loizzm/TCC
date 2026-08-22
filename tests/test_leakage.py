"""Criterios 1.3 e 1.4 — ausencia de vazamento de rotulo no estilo visual.

Este e' o teste que existe por causa do gerador legado `img.py`, que
desenhava `axhline(K)`, `axvline(theta)` e coloria por tipo de amortecimento —
e por isso "acertava" 93% com um classificador que so olhava a figura.

RULING G: 1.3 e 1.4a rodam no NIVEL DE SORTEIO, com n = 20000 e sem renderizar
imagem alguma. E' onde a independencia e' gerada e onde ela pode ser medida com
potencia estatistica suficiente.

1.4b repete a medida sobre as 300 amostras de fato renderizadas. O valor literal
do RULING G (|rho| < 0.20) e' MEDIDO E REPORTADO, inclusive quando excedido; a
ASSERTIVA e' o teste de permutacao equivalente, porque o limiar literal aplicado
ao MAXIMO sobre ~154 pares e' excedido em ~60% das vezes sob independencia
perfeita (medido). Ver a docstring de `test_1_4b_spearman_on_rendered_dataset` e
o §4.4 do relatorio. A verificacao com poder real sobre o caminho de
renderizacao e' `test_1_4b_render_block_is_exactly_the_sampled_style`, que exige
igualdade EXATA entre o bloco `render` do meta e o estilo sorteado.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from dataset.generator import load_sample, sample_system
from dataset.randomize import sample_style
from tests.conftest import (
    FEATURE_NAMES,
    FIXED_SNR_DB,
    MP_CTX,
    N_SAMPLING,
    WORKERS,
    read_meta,
    record_block,
    record_criterion,
    render_features,
)

GBM_ACC_MAX = 0.55        # criterio 1.3
RHO_MAX_SAMPLING = 0.05   # criterio 1.4a (RULING G)
RHO_MAX_DATASET = 0.20    # criterio 1.4b (RULING G)
BONFERRONI_ALPHA = 0.05

# Parametros contra os quais cada atributo visual e' correlacionado.
PARAM_NAMES = ("is_second", "K", "tau", "theta", "wn", "zeta", "w_window")


def _spearman_grid(X: np.ndarray, params: dict, names: tuple[str, ...]) -> dict:
    """|rho| de Spearman de cada (atributo de render, parametro) aplicavel."""
    pairs = []
    for pname in PARAM_NAMES:
        pv = params.get(pname)
        if pv is None:
            continue
        ok = np.isfinite(pv)
        if int(ok.sum()) < 20:
            continue
        for j, fname in enumerate(names):
            x = X[ok, j]
            if np.all(x == x[0]):
                continue
            rho, p = spearmanr(x, pv[ok])
            if not np.isfinite(rho):
                continue
            pairs.append((fname, pname, float(rho), float(p)))
    absr = np.array([abs(r) for _, _, r, _ in pairs])
    ps = np.array([p for *_, p in pairs])
    k = np.argmax(absr)
    alpha = BONFERRONI_ALPHA / max(len(pairs), 1)
    top = sorted(pairs, key=lambda t: -abs(t[2]))[:10]
    return {
        "n_pairs": len(pairs),
        "max_abs": float(absr.max()),
        "mean_abs": float(absr.mean()),
        "argmax": f"{pairs[k][0]} × {pairs[k][1]}",
        "min_p": float(ps.min()),
        "bonferroni_alpha": alpha,
        "n_significant": int((ps < alpha).sum()),
        "significant": [(a, b, r, p) for a, b, r, p in pairs if p < alpha],
        "top": top,
    }


# ==========================================================================
# 1.3 — classificador que so ve o estilo
# ==========================================================================
def test_1_3_render_attributes_do_not_predict_order(sampling_population):
    """GBM treinado SO com atributos de `render` não deve prever `order`."""
    X = sampling_population["X"]
    y = (sampling_population["order"] == "second").astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y)
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    majority = float(max(np.mean(y_te), 1.0 - np.mean(y_te)))

    imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                 key=lambda p: -p[1])[:8]
    record_block("gbm", {
        "dataset": f"sorteio puro (n = {N_SAMPLING})",
        "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
        "acc": acc, "majority": majority,
        "top": [(k, float(v)) for k, v in imp],
    })
    record_criterion(
        "1.3", "Vazamento: GBM só com atributos de `render` prevendo `order`",
        f"acurácia de teste ≤ {GBM_ACC_MAX}",
        f"{acc:.4f} (classe majoritária = {majority:.4f}, n_teste = {X_te.shape[0]})",
        acc <= GBM_ACC_MAX,
    )
    assert acc <= GBM_ACC_MAX, (
        f"acurácia {acc:.4f} > {GBM_ACC_MAX}: o estilo visual carrega a ordem — "
        "vazamento de rótulo")


# ==========================================================================
# 1.4a — Spearman no nivel de sorteio (assertiva forte)
# ==========================================================================
def test_1_4a_spearman_at_sampling_level(sampling_population):
    """|rho| < 0.05 em todos os pares, com n = 20000 (RULING G)."""
    sp = _spearman_grid(sampling_population["X"], sampling_population["params"],
                        sampling_population["names"])
    sp["n"] = int(sampling_population["X"].shape[0])
    record_block("spearman_sampling", sp)
    record_criterion(
        "1.4a", f"Spearman render × parâmetro no sorteio (n = {sp['n']})",
        f"|ρ| < {RHO_MAX_SAMPLING} em todos os {sp['n_pairs']} pares; "
        "nenhum significativo após Bonferroni",
        f"max |ρ| = {sp['max_abs']:.4f} ({sp['argmax']}); "
        f"pares significativos = {sp['n_significant']}",
        bool(sp["max_abs"] < RHO_MAX_SAMPLING and sp["n_significant"] == 0),
    )
    assert sp["max_abs"] < RHO_MAX_SAMPLING, (
        f"max |ρ| = {sp['max_abs']:.4f} em {sp['argmax']}")
    assert sp["n_significant"] == 0, (
        f"pares significativos após Bonferroni (α = {sp['bonferroni_alpha']:.2e}): "
        f"{sp['significant'][:5]}")


# ==========================================================================
# 1.4b — sanidade sobre o dataset realmente renderizado
# ==========================================================================
N_PERM = 4000             # replicas do nulo de permutacao de 1.4b
N_PERM_SUB = 2000         # replicas do nulo truncado em n = 300 (RULING O)
PERM_ALPHA = 1e-3         # orcamento de falha espuria de um portao deterministico


def _dataset_matrices(metas: list[dict]) -> tuple[np.ndarray, dict]:
    X = np.asarray([[render_features(m["render"])[k] for k in FEATURE_NAMES]
                    for m in metas], dtype=float)

    def col(key):
        return np.asarray([np.nan if m["params"][key] is None else m["params"][key]
                           for m in metas], dtype=float)

    params = {
        "is_second": np.asarray([1.0 if m["order"] == "second" else 0.0 for m in metas]),
        "K": col("K"), "tau": col("tau"), "theta": col("theta"),
        "wn": col("wn"), "zeta": col("zeta"),
        "w_window": np.asarray([m["t_window"][1] for m in metas], dtype=float),
    }
    return X, params


# Dados do nulo de permutacao. Ficam num global de modulo e sao herdados pelos
# workers via fork, em vez de serem serializados em cada uma das milhares de
# tarefas -- serializar X e params por tarefa colocaria centenas de MB na fila
# do pool.
_PERM_DATA: dict = {}


def _perm_max_abs(args: tuple) -> float:
    """max |rho| depois de embaralhar o pareamento estilo <-> sistema.

    Permutar o pareamento destroi qualquer associacao e preserva exatamente as
    marginais e a estrutura de dados faltantes (quais linhas tem tau, wn, zeta).
    E' o nulo exato do proprio dataset. `n_sub`, quando dado, trunca a replica
    para medir a calibracao do limiar em outro tamanho de amostra.
    """
    seed, n_sub = args
    X = _PERM_DATA["X"]
    params = _PERM_DATA["params"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    shuffled = {k: v[idx] for k, v in params.items()}
    if n_sub is not None and n_sub < X.shape[0]:
        X = X[:n_sub]
        shuffled = {k: v[:n_sub] for k, v in shuffled.items()}
    return _spearman_grid(X, shuffled, FEATURE_NAMES)["max_abs"]


def _perm_null(X, params, n_replicas: int, seed0: int, n_sub=None) -> np.ndarray:
    _PERM_DATA["X"] = X
    _PERM_DATA["params"] = params
    jobs = [(seed0 + i, n_sub) for i in range(n_replicas)]
    # o pool e' criado DEPOIS de popular `_PERM_DATA`: com fork, os filhos herdam
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=MP_CTX) as ex:
        return np.fromiter(ex.map(_perm_max_abs, jobs, chunksize=64),
                           dtype=float, count=n_replicas)


@pytest.mark.parametrize("which", ["clean", "noisy"])
def test_1_4b_spearman_on_rendered_dataset(which, clean_dataset, noisy_dataset):
    """Spearman render × parâmetro sobre as 300 amostras de fato renderizadas.

    RULING G manda assertar |rho| < 0.20 aqui. MEDIDO: esse limiar, aplicado ao
    MAXIMO sobre os ~154 pares, e' excedido em ~60% das vezes sob independencia
    perfeita (nulo de permutacao, 4000 replicas). Ou seja, ele sofre do mesmo
    defeito que o RULING G identificou no alvo original de 0.05 do PLANO, um
    nivel abaixo: a aritmetica "0.20 = 3.4 erros padrao" vale para UM par com
    n = 300, mas (i) tau/wn/zeta so existem em ~metade das amostras, logo o erro
    padrao real deles e' ~0.082 e 0.20 vale so 2.4 erros padrao, e (ii) o
    estatistico assertado e' o maximo sobre ~154 pares, nao um par.

    Portanto o numero literal do RULING G e' MEDIDO E REPORTADO (inclusive
    quando excede 0.20), e a ASSERTIVA e' o teste de permutacao correspondente,
    com orcamento de falha espuria de 1/1000 -- o que um portao deterministico
    exige. A assertiva forte de independencia continua sendo a 1.4a, com
    n = 20000, onde um |rho| verdadeiro de 0.05 ja seria detectado.
    """
    dirs = clean_dataset if which == "clean" else noisy_dataset
    metas = [read_meta(d) for d in dirs]
    X, params = _dataset_matrices(metas)
    sp = _spearman_grid(X, params, FEATURE_NAMES)
    sp["n"] = len(metas)

    null = _perm_null(X, params, N_PERM, 900_000)
    p_perm = float((null >= sp["max_abs"]).sum() + 1) / (N_PERM + 1)
    sp["perm"] = {
        "n_replicas": N_PERM,
        "p_value": p_perm,
        "alpha": PERM_ALPHA,
        "null_median": float(np.median(null)),
        "null_p99": float(np.percentile(null, 99)),
        "null_p999": float(np.percentile(null, 99.9)),
        "null_frac_over_020": float((null >= RHO_MAX_DATASET).mean()),
    }
    if which == "clean":
        # RULING O: calibracao do limiar literal no tamanho de amostra que o
        # RULING G tinha em mente (n = 300). Medido no mesmo nulo de permutacao,
        # so truncando as replicas -- e' a evidencia viva de que 0.20 aplicado ao
        # MAXIMO sobre ~154 pares nao separa vazamento de acaso nesse n.
        null300 = _perm_null(X, params, N_PERM_SUB, 500_000, n_sub=300)
        sp["perm"]["n300"] = {
            "n_replicas": N_PERM_SUB,
            "median": float(np.median(null300)),
            "p95": float(np.percentile(null300, 95)),
            "frac_over_020": float((null300 >= RHO_MAX_DATASET).mean()),
        }
    record_block(f"spearman_dataset_{which}", sp)
    if which == "clean":
        record_block("spearman_dataset", sp)
    record_criterion(
        f"1.4b-{which}",
        f"Spearman render × parâmetro no dataset renderizado `{which}` (n = {sp['n']})",
        f"p de permutação > {PERM_ALPHA:g} "
        f"(literal do RULING G: |ρ| < {RHO_MAX_DATASET})",
        f"max |ρ| = {sp['max_abs']:.4f} ({sp['argmax']}), p = {p_perm:.4f}"
        + ("" if sp["max_abs"] < RHO_MAX_DATASET
           else f" — EXCEDE o literal {RHO_MAX_DATASET}, ver §4.4"),
        p_perm > PERM_ALPHA,
    )
    assert p_perm > PERM_ALPHA, (
        f"[{which}] max |ρ| = {sp['max_abs']:.4f} em {sp['argmax']} é "
        f"significativo no nulo de permutação (p = {p_perm:.5f})")


def test_1_4b_render_block_is_exactly_the_sampled_style(clean_dataset, noisy_dataset):
    """O bloco `render` do meta é EXATAMENTE o estilo sorteado — round-trip.

    Esta é a assertiva de vazamento com poder real sobre o caminho de
    renderização: se qualquer etapa entre o sorteio e o meta.json alterasse um
    atributo visual em função do sistema, a igualdade exata quebraria. Ao
    contrário de uma correlação com n = 300, aqui a taxa de falso positivo é
    zero e a de falso negativo também.
    """
    n_checked = 0
    for dirs, snr in ((clean_dataset, None), (noisy_dataset, FIXED_SNR_DB)):
        for sample_dir in dirs:
            n_checked += 1
            m = read_meta(sample_dir)
            children = np.random.SeedSequence(int(m["seed"])).spawn(3)
            spec = sample_system(np.random.default_rng(children[0]))
            style = sample_style(np.random.default_rng(children[1]))
            if snr is not None:
                style.snr_db = float(snr)
            assert style.to_meta() == m["render"], (
                f"bloco render != estilo sorteado em {sample_dir}")
            assert spec.order == m["order"]
            assert {"K": spec.K, "tau": spec.tau, "theta": spec.theta,
                    "wn": spec.wn, "zeta": spec.zeta} == m["params"], (
                f"params != sistema sorteado em {sample_dir}")
    record_block("render_roundtrip", {"n": n_checked})
    record_criterion(
        "1.4c", "Round-trip exato: bloco `render` do meta == estilo sorteado",
        "igualdade exata em todas as amostras (falso positivo zero)",
        f"{n_checked} amostras verificadas, todas idênticas",
        True,
    )


# ==========================================================================
# 1.3 (replica) — GBM sobre o dataset renderizado, só reportado
# ==========================================================================
def test_1_3b_gbm_on_rendered_dataset(clean_dataset):
    """Réplica de 1.3 nas 300 amostras renderizadas (potência baixa, reportado)."""
    metas = [read_meta(d) for d in clean_dataset]
    X = np.asarray([[render_features(m["render"])[k] for k in FEATURE_NAMES]
                    for m in metas], dtype=float)
    y = np.asarray([1 if m["order"] == "second" else 0 for m in metas])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y)
    clf = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    record_criterion(
        "1.3b", f"Réplica de 1.3 no dataset renderizado (n = {len(metas)})",
        "sem alvo: n pequeno demais para separar 0,55 do acaso",
        f"acurácia de teste = {acc:.4f} (n_teste = {X_te.shape[0]})",
        None,
    )
    # Sem assertiva de limiar: com n_teste = 90 o erro padrão da acurácia é
    # ~0.053, então 0.55 está a 1 erro padrão do acaso. A assertiva vive em 1.3.
    assert 0.0 <= acc <= 1.0


# ==========================================================================
# 1.4d / 1.4e — vazamento no NIVEL DE PIXEL
# ==========================================================================
# Motivo de existirem: TESTE DE MUTACAO (HANDOFF §3.5). Os criterios 1.3 e
# 1.4a-c leem o bloco `render` do meta, nunca a `image.png`. Isso deixava passar
# exatamente a forma do defeito do `img.py` -- um vazamento que vive SO nos
# pixels. Dois mutantes atravessaram a suite inteira com 30 passed:
#
#   (i)  gerador que escolhe a cor da curva em funcao de `order` sem tocar no
#        meta (o `render` continua batendo com o estilo sorteado, entao 1.4c
#        passa);
#   (ii) gerador que redesenha `axhline(K)` e `axvline(theta)` na figura.
#
# Os dois testes abaixo fecham esses dois buracos lendo os pixels. Ambos os
# limiares foram fixados DEPOIS de medir o teto atingivel em 600 amostras
# (metodologia dos RULINGS O/R/S), e a separacao foi verificada contra os
# proprios mutantes -- os numeros estao nas docstrings.
INK_MODE_MIN_FRAC = 0.50   # so se afirma sobre amostras com cor modal dominante
INK_BG_TOL = 12            # distancia de canal para o pixel ser "nao-fundo"
SPAN_FRAC = 0.98           # extensao de ponta a ponta da area de dados
MIN_INK_FRAC = 0.25        # piso de ocupacao (pontilhado tem ciclo de ~0.38)
SPAN_BINS = 8              # a linha e' dividida em 8 faixas iguais...
SPAN_MIN_BINS = 7          # ...e a tinta tem que aparecer em pelo menos 7
CURVE_DILATION = 3         # px de folga em torno da mascara (franja do traco)


def _rgb(s: str) -> np.ndarray:
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], dtype=int)


def _n_groups(flags: np.ndarray) -> int:
    """Numero de blocos de indices consecutivos marcados (linha grossa = 1)."""
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(idx) > 1))


def _spanning_rows(ink: np.ndarray) -> np.ndarray:
    """Linhas (eixo 0) que atravessam a area de dados de ponta a ponta.

    Tres condicoes, e cada uma existe por um falso positivo MEDIDO:

    1. **Extensao** entre o primeiro e o ultimo pixel com tinta >= 98% da
       largura util.
    2. **Piso de ocupacao** de 25%, e nao continuidade: um distrator pode ser
       tracejado ou pontilhado (ciclo de trabalho de ate 0.38), e exigir
       continuidade descartaria justamente as linhas tracejadas do `img.py`.
    3. **Tinta em >= 7 das 8 faixas** da linha. So (1) e (2) marcavam uma linha
       que passa pela LEGENDA (canto direito, ~4 faixas) e por um TICK PARA
       DENTRO (borda esquerda, 1 faixa): duas coisas sem relacao nas duas
       pontas, somando 26% de ocupacao e extensao total. Uma reta de verdade
       aparece em todas as faixas; a folga de uma faixa cobre a fase do
       tracejado (com 8/8 exigidas, 2 retas reais em 585 escapavam).
    """
    n = ink.shape[1]
    cnt = ink.sum(1)
    first = np.argmax(ink, axis=1)
    last = n - 1 - np.argmax(ink[:, ::-1], axis=1)
    extent = np.where(cnt > 0, last - first, -1)
    edges = np.linspace(0, n, SPAN_BINS + 1).astype(int)
    filled = np.zeros(ink.shape[0], dtype=int)
    for a, bnd in zip(edges[:-1], edges[1:]):
        filled += ink[:, a:bnd].any(1)
    return ((extent >= SPAN_FRAC * (n - 1)) & (cnt >= MIN_INK_FRAC * n)
            & (filled >= SPAN_MIN_BINS))


def _pixel_worker(sample_dir: str) -> dict:
    m = load_sample(sample_dir)
    img = m["image"].astype(int)
    lit = m["mask"] > 127
    r = m["render"]
    out = {"sample_dir": str(sample_dir), "has_grid": bool(r["has_grid"]),
           "n_distractors": int(r["n_distractors"]), "line_style": r["line_style"],
           "lw_px": float(r["line_width"]) * int(r["dpi"]) / 72.0}

    # ---- (1.4d) cor da tinta desenhada x `render.line_color` ----
    if lit.any():
        px = img[lit]
        uniq, cnt = np.unique(px, axis=0, return_counts=True)
        mode = uniq[int(np.argmax(cnt))]
        out["mode_frac"] = float(cnt.max() / cnt.sum())
        out["ink_err"] = float(np.max(np.abs(mode - _rgb(r["line_color"]))))
    else:  # pragma: no cover - mascara vazia e' barrada pelo RULING S
        out["mode_frac"] = 0.0
        out["ink_err"] = float("nan")

    # ---- (1.4e) linhas retas de span completo x orcamento de distratores ----
    h, w = lit.shape
    x0, y0, x1, y1 = m["plot_bbox_px"]
    pad = 4 + int(np.ceil(int(r["dpi"]) / 72.0))   # descarta as spines
    xa, xb = max(x0 + pad, 0), min(x1 - pad, w)
    ya, yb = max(y0 + pad, 0), min(y1 - pad, h)
    if xb - xa < 20 or yb - ya < 20:  # pragma: no cover - figura degenerada
        out["span_skipped"] = True
        return out
    out["span_skipped"] = False

    # A curva nao conta, e a mascara (limiar > 127) e' mais estreita que o traco
    # desenhado: sobra a franja de anti-aliasing. Com 1 px de folga, o trecho
    # ja assentado de uma curva pontilhada virava "reta de span completo" (3
    # falsos positivos em 1200 medidos). 3 px cobrem a franja de qualquer
    # espessura sorteada.
    cur = lit[ya:yb, xa:xb]
    halo = cur
    for _ in range(CURVE_DILATION):
        nxt = halo.copy()
        nxt[1:, :] |= halo[:-1, :]
        nxt[:-1, :] |= halo[1:, :]
        nxt[:, 1:] |= halo[:, :-1]
        nxt[:, :-1] |= halo[:, 1:]
        halo = nxt
    sub = img[ya:yb, xa:xb]
    ink = (np.abs(sub - _rgb(r["bg_color"])).max(axis=2) > INK_BG_TOL) & ~halo

    out["n_h"] = _n_groups(_spanning_rows(ink))
    out["n_v"] = _n_groups(_spanning_rows(ink.T))
    out["excess"] = out["n_h"] + out["n_v"] - out["n_distractors"]
    return out


@pytest.fixture(scope="session")
def pixel_stats(clean_dataset, noisy_dataset) -> list[dict]:
    dirs = list(clean_dataset) + list(noisy_dataset)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=MP_CTX) as ex:
        return list(ex.map(_pixel_worker, dirs, chunksize=8))


def test_1_4d_ink_color_in_the_image_is_the_sampled_line_color(pixel_stats):
    """A cor de fato desenhada na `image.png` é exatamente `render.line_color`.

    Fecha o buraco do mutante (i): cor da curva escolhida em função de `order`
    sem tocar no meta. Como a curva é o artista de maior `zorder` sobre a
    máscara, a cor MODAL dos pixels acesos é a cor pura do traço.

    **Estrato assertado:** amostras cuja cor modal é dominante
    (`mode_frac ≥ 0.50`). Fora dele não existe pixel de interior puro — traço no
    piso de 1,5 px, ou curva recortada demais pelo ruído — e a moda é uma
    mistura do anti-aliasing. Medido num lote limpo de 600: as 3 únicas
    amostras com `ink_err > 0` têm `mode_frac` entre 0,13 e 0,26, todas com
    `line_width` no piso de 1,5 px; no estrato assertado, 0 violações em 561.

    Contra o gerador mutante: **279/279 violações**, com erro mediano de 144
    níveis de canal. Separação total, falso positivo zero.
    """
    dom = [s for s in pixel_stats if s["mode_frac"] >= INK_MODE_MIN_FRAC]
    assert dom, "nenhuma amostra com cor modal dominante"
    err = np.array([s["ink_err"] for s in dom])
    bad = [s for s in dom if s["ink_err"] > 0]
    excluded = len(pixel_stats) - len(dom)

    record_block("ink_color", {
        "n_total": len(pixel_stats), "n_asserted": len(dom),
        "excluded_frac": excluded / len(pixel_stats),
        "err_max": float(err.max()), "n_violations": len(bad),
        "worst": [(s["sample_dir"], s["ink_err"], s["mode_frac"], s["lw_px"])
                  for s in sorted(dom, key=lambda s: -s["ink_err"])[:3]],
    })
    record_criterion(
        "1.4d", "Cor da tinta na `image.png` == `render.line_color` (nível de pixel)",
        f"erro de canal = 0 em todas as amostras com cor modal dominante "
        f"(`mode_frac ≥ {INK_MODE_MIN_FRAC:.2f}`)",
        f"{len(bad)} violações em {len(dom)} amostras "
        f"({excluded} excluídas por traço fino); erro máximo = {err.max():.0f}",
        len(bad) == 0,
    )
    assert not bad, (
        f"{len(bad)} amostras com cor desenhada != `render.line_color` "
        f"(pior: {sorted(bad, key=lambda s: -s['ink_err'])[0]['sample_dir']}) — "
        "vazamento visível só nos pixels")


def test_1_4e_no_straight_line_unexplained_by_the_style(pixel_stats):
    """Nenhuma reta de span completo além das que o estilo declara.

    Fecha o buraco do mutante (ii): `axhline(K)` / `axvline(θ)` de volta na
    figura — o defeito literal do `img.py`, em que a posição da reta É o
    parâmetro. O invariante é o do contrato §2: todo elemento visual tem de
    estar declarado no estilo, que por construção não vê o rótulo.

    **Estrato assertado:** amostras sem grade. Com grade, as linhas da própria
    grade são retas de span completo e não há orçamento declarado para elas
    (`render` só registra `has_grid`, não quantas linhas) — medido: excesso de
    até 15 nas amostras com grade, contra 0 sem grade.

    Medido: **0 violações em 916 amostras sãs sem grade** (585 dos dois
    conjuntos de aceitação + 331 de um lote independente); contra **143/157
    (91,1%) no gerador mutante**. A assertiva é sobre o conjunto, então a
    detecção é certa mesmo com a potência por amostra em 91%.
    """
    pool = [s for s in pixel_stats if not s["span_skipped"] and not s["has_grid"]]
    assert pool, "nenhuma amostra sem grade"
    bad = [s for s in pool if s["excess"] > 0]
    ex = np.array([s["excess"] for s in pool])

    record_block("span_lines", {
        "n_total": len(pixel_stats), "n_asserted": len(pool),
        "n_grid": sum(1 for s in pixel_stats if s["has_grid"]),
        "excess_max": int(ex.max()), "excess_median": float(np.median(ex)),
        "n_violations": len(bad),
        "worst": [(s["sample_dir"], s["n_h"], s["n_v"], s["n_distractors"])
                  for s in sorted(pool, key=lambda s: -s["excess"])[:3]],
    })
    record_criterion(
        "1.4e", "Retas de span completo na área de dados × distratores declarados",
        "nenhuma amostra sem grade com mais retas que `render.n_distractors`",
        f"{len(bad)} violações em {len(pool)} amostras sem grade; "
        f"excesso máximo = {int(ex.max())}",
        len(bad) == 0,
    )
    assert not bad, (
        f"{len(bad)} amostras com reta não declarada pelo estilo "
        f"(pior: {sorted(bad, key=lambda s: -s['excess'])[0]}) — "
        "elemento visual fora do estilo sorteado")
