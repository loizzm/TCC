"""Identificação clássica por mínimos quadrados (Estágio D do PLANO).

Recebe uma série de resposta ao degrau `(t, y)` e devolve os parâmetros da
planta, escolhendo entre FOPDT e 2ª ordem canônica pelo menor AIC. Também
implementa os baselines clássicos (tangente, Smith, Sundaresan–Krishnaswamy).

Este módulo NÃO importa nada de `dataset/` — `model_response` é uma
reimplementação independente dos modelos de `contract.md` §1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

__all__ = [
    "FitResult",
    "PARAM_KEYS",
    "STEP_AMPLITUDE",
    "model_response",
    "initial_guess_fopdt",
    "initial_guess_second",
    "fit_fopdt",
    "fit_second",
    "identify",
    "identify_both",
    "baseline_tangent",
    "baseline_smith",
    "baseline_sundaresan_krishnaswamy",
]

# Amplitude do degrau: sempre 1.0 neste trabalho (contract.md §1).
#
# CONSEQUENCIA QUE JA CAUSOU LEITURA ERRADA, e que nao e limitacao de
# implementacao: o `K` que este modulo devolve e a EXCURSAO DA SAIDA POR UNIDADE
# DE ENTRADA, ou seja o PRODUTO `K_planta x U`, onde U e a amplitude do degrau
# realmente aplicado. Nao e o ganho DC da planta, a menos que U = 1.
#
# Da curva de SAIDA sozinha, `K_planta` e `U` nao sao separaveis — so o produto
# e observavel. Estas tres situacoes geram a MESMA curva, ponto a ponto:
#
#     K_planta = 1,997   com   U = -1
#     K_planta = 0,999   com   U = -2
#     K_planta = 0,499   com   U = -4
#
# Nenhum algoritmo distingue entre elas sem conhecer U. Medido em
# `Figure_dn.png` (`rg_negativo.py`, sistema 1): a planta e `2/(s+2)`, cujo
# ganho DC e 1, e o degrau aplicado tem amplitude -2. A pipeline devolve
# K = -1,997, que e 1 x (-2) e esta CORRETO sob esta convencao — a curva
# reconstruida a partir dela bate com a verdade analitica com erro maximo de
# 0,0039 e RMSE 0,0026. Quem comparar esse -1,997 com o "K = 1" escrito na
# funcao de transferencia vai concluir, erradamente, que houve falha de ajuste.
#
# Recuperar `K_planta` exige LER a amplitude do degrau, o que so e possivel
# quando a entrada esta plotada no mesmo quadro — envelope proprio, ainda nao
# implementado. Ver ARQUITETURA.md §4 e HANDOFF_P2_7 §42.
STEP_AMPLITUDE: float = 1.0

PARAM_KEYS: tuple[str, ...] = ("K", "tau", "theta", "wn", "zeta")

# Bounds obrigatórios (task-2-brief). `theta` tem limite superior t[-1].
K_BOUNDS = (1e-3, 1e4)
TAU_BOUNDS = (1e-4, 1e4)
WN_BOUNDS = (1e-4, 1e3)
ZETA_BOUNDS = (1e-3, 10.0)

_TINY = 1e-300

# Tamanho do trecho contíguo da guarda `_polo_rapido_e_artefato`, em fração da
# série. É o ÚNICO número dela escolhido por medição — o limiar (1,0) é
# estrutural. Varrido nos dois caminhos que usam `identify` (§38):
#
#   frac    imagem (n=837)          oráculo 20 dB (n=300)
#   -----   ---------------------   ---------------------
#   0,005   90,08 %  (custo  -1)    --
#   0,01    91,64 %  (custo  -3)    88,0 %
#   0,02    92,71 %  (custo  -6)    87,3 %
#   0,03    92,95 %  (custo  -7)    87,7 %   <- escolhido
#   0,05    92,95 %  (custo  -9)    87,7 %
#   0,08    91,64 %  (custo -23)    86,7 %
#
# 0,03 e 0,05 empatam no platô do caminho da imagem e no oráculo; 0,03 é o
# escolhido por ter o MENOR custo dentro do platô (7 contra 9 amostras de 2ª
# ordem rebaixadas) e por ser a janela mais curta — trecho menor é afirmação
# mais forte de localidade, que é o que a guarda quer dizer.
TRECHO_FRAC = 0.03


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    """Resultado de um ajuste (uma estrutura de modelo)."""

    order: str
    params: dict = field(default_factory=dict)
    aic: float = float("inf")
    nrmse: float = float("nan")
    sse: float = float("nan")
    success: bool = False
    n_params: int = 0


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def _num(value, default: float) -> float:
    """Converte para float tratando None/nan como `default`."""
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _fopdt_basis(t, tau, theta):
    """Resposta FOPDT com ganho unitário. `tau`/`theta` podem ser arrays."""
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-30)
    u = np.asarray(t, dtype=float) - np.asarray(theta, dtype=float)
    uu = np.maximum(u, 0.0)
    return np.where(u >= 0.0, 1.0 - np.exp(-uu / tau), 0.0)


def _second_terms(zeta: float, v, need_d: bool = True):
    """Termos base da resposta de 2ª ordem, em função de `v = wn*(t-theta) >= 0`.

    Com `w = (1 - zeta^2) * v^2`, `C = cos(sqrt(w))` e `S = sin(sqrt(w))/sqrt(w)`
    são **funções inteiras de w** (para w < 0 viram cosh e sinh(.)/.), logo a
    resposta é uma única expressão analítica, contínua e sem NaN atravessando
    zeta = 1 — em w = 0 vale C = S = 1 e a fórmula degenera exatamente no ramo
    criticamente amortecido `1 - exp(-wn*u)*(1 + wn*u)` do contract.md §1.

    Devolve `(EC, ES, ED)` = `E*C`, `E*S`, `E*dS/dw`, com `E = exp(-zeta*v)`,
    já na forma de produto para evitar overflow de cosh quando zeta > 1.
    """
    zeta = float(zeta)
    v = np.asarray(v, dtype=float)
    w = (1.0 - zeta * zeta) * v * v
    E = np.exp(-zeta * v)
    if zeta <= 1.0:
        x = np.sqrt(np.maximum(w, 0.0))
        EC = E * np.cos(x)
        ES = E * np.sinc(x / np.pi)          # sin(x)/x, estável em x -> 0
    else:
        beta = np.sqrt(zeta * zeta - 1.0)
        x = beta * v
        a = np.exp(-(zeta - beta) * v)       # ambos os expoentes <= 0
        b = np.exp(-(zeta + beta) * v)
        EC = 0.5 * (a + b)
        small = x <= 1e-6
        ES = np.where(small,
                      E * (1.0 + x * x / 6.0),
                      (a - b) / (2.0 * np.where(small, 1.0, x)))
    if not need_d:
        return EC, ES, None
    tiny_w = np.abs(w) <= 1e-4
    ED = np.where(tiny_w,
                  E * (-1.0 / 6.0 + w / 60.0),
                  (EC - ES) / (2.0 * np.where(tiny_w, 1.0, w)))
    return EC, ES, ED


def _second_basis(t, wn, zeta, theta):
    """Resposta de 2ª ordem canônica com ganho unitário.

    `zeta` é escalar; `wn` e `theta` podem ser arrays (broadcasting).
    """
    wn = np.maximum(np.asarray(wn, dtype=float), 1e-30)
    u = np.asarray(t, dtype=float) - np.asarray(theta, dtype=float)
    uu = np.maximum(u, 0.0)
    v = wn * uu
    EC, ES, _ = _second_terms(zeta, v, need_d=False)
    return np.where(u >= 0.0, 1.0 - (EC + zeta * v * ES), 0.0)


def model_response(order: str, params: dict, t: np.ndarray) -> np.ndarray:
    """Resposta ao degrau analítica a partir de `(order, params)` (contract §1).

    Levanta `ValueError` se `order` não for "fopdt" nem "second". É a única
    função pública do módulo que levanta, e é intencional: `order` vem de código
    (não de dado medido), então um valor desconhecido é erro de programação, não
    uma série difícil. A regra "nunca lance exceção" do contrato §6 vale para
    `identify`/`fit_*`, que rodam em lote sobre dados — esses nunca levantam.
    """
    t = np.asarray(t, dtype=float)
    K = _num(params.get("K"), 0.0)
    theta = _num(params.get("theta"), 0.0)
    if order == "fopdt":
        base = _fopdt_basis(t, _num(params.get("tau"), 1.0), theta)
    elif order == "second":
        base = _second_basis(t,
                             _num(params.get("wn"), 1.0),
                             _num(params.get("zeta"), 1.0),
                             theta)
    else:
        raise ValueError(f"order desconhecida: {order!r}")
    return (K * STEP_AMPLITUDE) * base


def _params_dict(order: str, K, tau, theta, wn, zeta) -> dict:
    """Dict com as 5 chaves; None onde o parâmetro não se aplica."""
    if order == "fopdt":
        return {"K": float(K), "tau": float(tau), "theta": float(theta),
                "wn": None, "zeta": None}
    return {"K": float(K), "tau": None, "theta": float(theta),
            "wn": float(wn), "zeta": float(zeta)}


# --------------------------------------------------------------------------- #
# Utilitários de série
# --------------------------------------------------------------------------- #
def _clean(t, y) -> tuple[np.ndarray, np.ndarray]:
    """Alinha, ordena e remove pontos não finitos (robustez a NaN)."""
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(t.size, y.size)
    t, y = t[:n], y[:n]
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size > 1 and np.any(np.diff(t) < 0):
        idx = np.argsort(t, kind="stable")
        t, y = t[idx], y[idx]
    return t, y


def _span(t: np.ndarray) -> float:
    if t.size < 2:
        return 1.0
    s = float(t[-1] - t[0])
    return s if s > 0 else 1.0


def _sinal_do_degrau(y: np.ndarray) -> float:
    """Direção do degrau: `-1.0` se a resposta DESCE, `+1.0` caso contrário.

    O repouso é o extremo que aparece PRIMEIRO. Se o máximo vem antes do
    mínimo, a série desceu. É tudo.

    Por que o TEMPO e não o VALOR. Uma resposta ao degrau nunca cruza de volta
    o nível de onde partiu — o sobressinal de 1ª/2ª ordem nunca ultrapassa
    100 % do salto, com igualdade só no limite ζ→0. Então o repouso É um dos
    dois extremos da série, e o que o distingue do sobressinal não é a
    magnitude, é a ORDEM: o repouso vem antes. Ler a ordem dispensa saber onde
    a resposta assenta, que é justamente o que uma janela curta esconde.

    Duas formulações por VALOR foram implementadas, medidas e DESCARTADAS —
    cada uma quebra numa ponta diferente da série, e as duas quebras são reais:

    1. **Mediana do primeiro decil contra a do último.** Usa o primeiro decil
       como proxy do repouso, e ele deixa de ser o repouso quando o Estágio A
       come o platô inicial (§39.3): cai dentro do transitório, que numa
       subamortecida passa ALÉM do valor final, pelo lado oposto. Errava 8 de
       44 casos sintéticos (todas as subamortecidas ζ ≤ 0,4 sem cabeça) e era a
       causa de `Figure_dn2.png` não fechar. Simétrico: espelhava subida
       também. Ajustar a FRAÇÃO não conserta — a leitura oscila com ela
       (0,05 → -1; 0,10 → +1; 0,20 → -1; 0,30 → +1 na MESMA imagem).
    2. **Extremo mais distante da mediana do último decil.** Pressupõe que o
       último decil é o valor FINAL, e não é quando a janela acaba antes de
       assentar: numa 2ª ordem muito subamortecida o último decil pousa no
       ringing, o primeiro pico fica mais longe dele que o repouso, e o pico é
       eleito repouso. Media 0 erros no sintético mas espelhava 2 das 900
       séries do oráculo do corpus — todas com K > 0, logo 2 amostras boas
       destruídas. Aparecia na Parte 1 como MAPE(K) do estrato limpo w<3
       saindo de 0,000 % para 0,239 %.

    A leitura por ordem temporal mede 0 erros nas duas provas: 44/44 no
    sintético (ζ e τ varridos, dois sinais, com e sem cabeça) e 0 espelhos
    indevidos nas 900 do oráculo. É imune a ruído nesta aplicação (0 erros em
    1600 séries com σ até 0,30 sobre um salto de 2).

    CONTRATO, e o limite medido. Vale enquanto o repouso ainda ESTIVER na
    série. Cortada a cabeça além dele, o primeiro extremo que sobra é um
    sobressinal, e a regra inverte — não há estatística barata que recupere
    isso, porque a direção passa a viver só no envelope decadente, que exige
    ajuste. Asseverado com xfail estrito em
    `test_o_limite_do_contrato_quando_o_repouso_sai_da_serie`.

    NÃO reusa `pipeline._nivel_de_repouso`: ele lê as 5 PRIMEIRAS colunas e
    supõe a curva parada ali — a mesma suposição que o §39.3 quebra, e que o
    próprio `pipeline.py:96-104` documenta como frágil. Além disso
    `classical.py` é a camada de baixo e não pode importar `pipeline`.

    Empate (série plana, ou extremos no mesmo índice) devolve `+1.0`: sem
    excursão não há sinal a recuperar, e `+1.0` mantém o caminho anterior byte
    a byte.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size < 4:
        return 1.0
    return -1.0 if int(np.argmax(y)) < int(np.argmin(y)) else 1.0


def _profiled_sse(basis: np.ndarray, y: np.ndarray, yy: float):
    """Perfila K analiticamente (o modelo é linear em K) e devolve (K, SSE)."""
    bb = np.einsum("...i,...i->...", basis, basis)
    by = basis @ y
    K = np.where(bb > _TINY, by / np.maximum(bb, _TINY), K_BOUNDS[0])
    K = np.clip(K, K_BOUNDS[0], K_BOUNDS[1])
    sse = yy - 2.0 * K * by + K * K * bb
    return K, sse


# --------------------------------------------------------------------------- #
# Chutes iniciais
# --------------------------------------------------------------------------- #
def _integral_guess_fopdt(t: np.ndarray, y: np.ndarray) -> dict | None:
    """Regressão linear de y contra sua própria integral.

    Para t >= theta vale  I(t) = K*t - K*theta - tau*y(t), com
    I(t) = integral de y. Logo [t, 1, -y] @ [K, -K*theta, tau] = I(t) é linear
    nos três parâmetros e não depende de y[-1] nem do regime permanente.

    LIMITAÇÃO CONHECIDA (medida na revisão): com janela abaixo de ~1.8*tau a
    identidade acima é violada por uma fração grande da amostra (os pontos com
    t < theta, onde ela não vale) e a regressão converge para K e tau
    **negativos**. Por isso a função **rejeita** qualquer solução com K <= 0 ou
    tau <= 0 e devolve `None` (falha explícita) em vez de um valor sem sentido:
    quem chama tem de ter um caminho alternativo. Nesse regime quem sustenta o
    ajuste é a busca em grade com K perfilado (`_grid_guess_fopdt`).
    """
    if t.size < 6:
        return None
    integral = cumulative_trapezoid(y, t, initial=0.0)
    mask = np.ones(t.size, dtype=bool)
    out = None
    for _ in range(3):
        if int(mask.sum()) < 4:
            break
        A = np.column_stack([t[mask], np.ones(int(mask.sum())), -y[mask]])
        try:
            c, *_ = np.linalg.lstsq(A, integral[mask], rcond=None)
        except np.linalg.LinAlgError:
            break
        K, c1, tau = float(c[0]), float(c[1]), float(c[2])
        if not (np.isfinite(K) and np.isfinite(tau)):
            break
        # Rejeição de solução sem sentido físico (K e tau são positivos por
        # construção no domínio deste trabalho). Não sobrescreve um iterado
        # válido anterior: melhor devolver o bom antigo do que o ruim novo.
        if K <= 0.0 or tau <= 0.0:
            break
        theta = -c1 / K
        if not np.isfinite(theta):
            break
        out = {"K": K, "tau": tau, "theta": theta}
        mask = t >= max(theta, t[0])
    return out


def _integral_guess_second(t: np.ndarray, y: np.ndarray) -> dict | None:
    """Regressão por dupla integração (equivalente a um ajuste de momentos).

    Para t >= theta:  a*y + b*I1 + I2 = K*(t-theta)^2/2, com a = 1/wn^2 e
    b = 2*zeta/wn. Isso é linear em [a, b, K/2, -K*theta, K*theta^2/2] usando
    os regressores [-y, -I1, t^2, t, 1] contra I2.
    """
    if t.size < 8:
        return None
    i1 = cumulative_trapezoid(y, t, initial=0.0)
    i2 = cumulative_trapezoid(i1, t, initial=0.0)
    mask = np.ones(t.size, dtype=bool)
    out = None
    for _ in range(3):
        n = int(mask.sum())
        if n < 6:
            break
        tm = t[mask]
        A = np.column_stack([-y[mask], -i1[mask], tm * tm, tm, np.ones(n)])
        try:
            c, *_ = np.linalg.lstsq(A, i2[mask], rcond=None)
        except np.linalg.LinAlgError:
            break
        a, b, c2, c3 = float(c[0]), float(c[1]), float(c[2]), float(c[3])
        K = 2.0 * c2
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(K)):
            break
        if a <= 0.0 or abs(K) < 1e-12:
            break
        wn = 1.0 / np.sqrt(a)
        zeta = 0.5 * b * wn
        theta = -c3 / K
        if not (np.isfinite(wn) and np.isfinite(zeta) and np.isfinite(theta)):
            break
        out = {"K": K, "wn": wn, "zeta": zeta, "theta": theta}
        mask = t >= max(theta, t[0])
    return out


def _subsample(t, y, n_max=160):
    """Decima a série para a busca em grade (o chute grosseiro não precisa de 512)."""
    if t.size <= n_max:
        return t, y
    idx = np.unique(np.linspace(0, t.size - 1, n_max).astype(int))
    return t[idx], y[idx]


def _grid_guess_fopdt(t, y, n_keep=3) -> list[dict]:
    """Busca em grade 2-D (theta, tau) com K perfilado analiticamente."""
    span = _span(t)
    t0 = float(t[0])
    t, y = _subsample(t, y)
    yy = float(y @ y)
    thetas = np.linspace(0.0, 0.85 * span, 18) + t0
    taus = np.geomspace(span / 400.0, 30.0 * span, 26)
    basis = _fopdt_basis(t[None, None, :], taus[None, :, None],
                         thetas[:, None, None])
    K, sse = _profiled_sse(basis, y, yy)
    flat = sse.ravel()
    order_idx = np.argsort(flat)[:max(1, n_keep)]
    out = []
    for idx in order_idx:
        i, j = np.unravel_index(idx, sse.shape)
        out.append({"K": float(K[i, j]), "tau": float(taus[j]),
                    "theta": float(thetas[i])})
    return out


def _grid_guess_second(t, y, n_keep=3) -> list[dict]:
    """Busca em grade 3-D (theta, wn, zeta) com K perfilado analiticamente."""
    span = _span(t)
    t0 = float(t[0])
    t, y = _subsample(t, y)
    yy = float(y @ y)
    thetas = np.linspace(0.0, 0.8 * span, 10) + t0
    wns = np.geomspace(0.05 / span, 120.0 / span, 15)
    zetas = np.geomspace(0.05, 8.0, 10)
    best = []
    for zeta in zetas:
        basis = _second_basis(t[None, None, :], wns[None, :, None], zeta,
                              thetas[:, None, None])
        K, sse = _profiled_sse(basis, y, yy)
        idx = int(np.argmin(sse))
        i, j = np.unravel_index(idx, sse.shape)
        best.append((float(sse[i, j]),
                     {"K": float(K[i, j]), "wn": float(wns[j]),
                      "zeta": float(zeta), "theta": float(thetas[i])}))
    best.sort(key=lambda p: p[0])
    return [p[1] for p in best[:max(1, n_keep)]]


def initial_guess_fopdt(t, y) -> dict:
    """Chute inicial FOPDT robusto a janela truncada."""
    t, y = _clean(t, y)
    if t.size < 3:
        return {"K": 1.0, "tau": 1.0, "theta": 0.0, "wn": None, "zeta": None}
    span = _span(t)
    cands: list[dict] = []
    g = _integral_guess_fopdt(t, y)
    if g is not None:
        cands.append(g)
    cands.append(baseline_sundaresan_krishnaswamy(t, y))
    cands.extend(_grid_guess_fopdt(t, y, n_keep=2))
    fallback = {"K": max(float(np.max(np.abs(y))), K_BOUNDS[0]),
                "tau": 0.25 * span, "theta": 0.0}
    best, best_sse = fallback, np.inf
    for c in cands:
        p = _sanitize_fopdt(c, t, fallback)
        sse = float(np.sum((model_response("fopdt", p, t) - y) ** 2))
        if np.isfinite(sse) and sse < best_sse:
            best, best_sse = p, sse
    return {"K": best["K"], "tau": best["tau"], "theta": best["theta"],
            "wn": None, "zeta": None}


def initial_guess_second(t, y) -> dict:
    """Chute inicial de 2ª ordem (usa sobressinal quando existe)."""
    t, y = _clean(t, y)
    if t.size < 3:
        return {"K": 1.0, "tau": None, "theta": 0.0, "wn": 1.0, "zeta": 1.0}
    cands: list[dict] = []
    g = _integral_guess_second(t, y)
    if g is not None:
        cands.append(g)
    o = _overshoot_guess(t, y)
    if o is not None:
        cands.append(o)
    cands.append(_overdamped_guess(t, y))
    cands.extend(_grid_guess_second(t, y, n_keep=2))
    fallback = {"K": max(float(np.max(np.abs(y))), K_BOUNDS[0]),
                "wn": 4.0 / _span(t), "zeta": 1.0, "theta": 0.0}
    best, best_sse = fallback, np.inf
    for c in cands:
        p = _sanitize_second(c, t, fallback)
        sse = float(np.sum((model_response("second", p, t) - y) ** 2))
        if np.isfinite(sse) and sse < best_sse:
            best, best_sse = p, sse
    return {"K": best["K"], "tau": None, "theta": best["theta"],
            "wn": best["wn"], "zeta": best["zeta"]}


def _overshoot_guess(t, y) -> dict | None:
    """Chute por sobressinal + período de oscilação (caso subamortecido)."""
    if t.size < 8:
        return None
    K = _estimate_gain(t, y)
    if not np.isfinite(K) or K <= 0:
        return None
    ymax = float(np.max(y))
    if ymax <= K * 1.005:
        return None
    mp = min(max((ymax - K) / K, 1e-6), 0.99)
    lg = np.log(mp)
    zeta = -lg / np.sqrt(np.pi ** 2 + lg ** 2)
    tp = float(t[int(np.argmax(y))])
    theta = _crossing_time(t, y, 0.05 * K)
    if not np.isfinite(theta):
        theta = float(t[0])
    dt_peak = tp - theta
    if dt_peak <= 0:
        return None
    wd = np.pi / dt_peak
    wn = wd / max(np.sqrt(max(1.0 - zeta * zeta, 1e-9)), 1e-9)
    return {"K": K, "wn": wn, "zeta": zeta, "theta": theta}


def _overdamped_guess(t, y) -> dict:
    """Chute superamortecido: converte o ajuste FOPDT em (wn, zeta)."""
    g = _integral_guess_fopdt(t, y)
    span = _span(t)
    if g is None or not np.isfinite(g.get("tau", np.nan)) or g["tau"] <= 0:
        return {"K": max(float(np.max(np.abs(y))), K_BOUNDS[0]),
                "wn": 4.0 / span, "zeta": 1.5, "theta": 0.0}
    tau = max(g["tau"], 1e-6)
    # Polo lento em -1/tau, polo rápido 5x mais rápido.
    p1, p2 = 1.0 / tau, 5.0 / tau
    wn = np.sqrt(p1 * p2)
    zeta = 0.5 * (p1 + p2) / wn
    return {"K": g["K"], "wn": wn, "zeta": zeta, "theta": max(g["theta"], 0.0)}


# --------------------------------------------------------------------------- #
# Saneamento de pontos de partida
# --------------------------------------------------------------------------- #
def _theta_bounds(t: np.ndarray) -> tuple[float, float]:
    lo = 0.0
    hi = float(t[-1]) if t.size else 1.0
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _sanitize_fopdt(p: dict, t: np.ndarray, fallback: dict) -> dict:
    tlo, thi = _theta_bounds(t)
    K = _num(p.get("K"), fallback["K"])
    tau = _num(p.get("tau"), fallback["tau"])
    theta = _num(p.get("theta"), fallback["theta"])
    if K <= 0:
        K = fallback["K"]
    if tau <= 0:
        tau = fallback["tau"]
    return {"K": float(np.clip(K, *K_BOUNDS)),
            "tau": float(np.clip(tau, *TAU_BOUNDS)),
            "theta": float(np.clip(theta, tlo, thi * 0.98)),
            "wn": None, "zeta": None}


def _sanitize_second(p: dict, t: np.ndarray, fallback: dict) -> dict:
    tlo, thi = _theta_bounds(t)
    K = _num(p.get("K"), fallback["K"])
    wn = _num(p.get("wn"), fallback["wn"])
    zeta = _num(p.get("zeta"), fallback["zeta"])
    theta = _num(p.get("theta"), fallback["theta"])
    if K <= 0:
        K = fallback["K"]
    if wn <= 0:
        wn = fallback["wn"]
    if zeta <= 0:
        zeta = fallback["zeta"]
    return {"K": float(np.clip(K, *K_BOUNDS)),
            "tau": None,
            "theta": float(np.clip(theta, tlo, thi * 0.98)),
            "wn": float(np.clip(wn, *WN_BOUNDS)),
            "zeta": float(np.clip(zeta, *ZETA_BOUNDS))}


def _dedupe(starts: list[np.ndarray], rtol: float = 1e-3) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for s in starts:
        if not np.all(np.isfinite(s)):
            continue
        dup = False
        for k in kept:
            denom = np.maximum(np.abs(k), 1e-9)
            if np.all(np.abs(s - k) / denom < rtol):
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def _metrics(y: np.ndarray, resid: np.ndarray, k: int) -> tuple[float, float, float]:
    n = int(resid.size)
    sse = float(resid @ resid)
    if not np.isfinite(sse):
        return float("inf"), float("nan"), float("inf")
    rng = float(np.max(y) - np.min(y)) if n else 0.0
    denom = rng if rng > 1e-12 else 1e-12
    nrmse = float(np.sqrt(sse / max(n, 1)) / denom)
    sse_safe = max(sse / max(n, 1), 1e-300)
    aic = float(n * np.log(sse_safe) + 2 * k)
    return sse, nrmse, aic


# --------------------------------------------------------------------------- #
# Ajustes
# --------------------------------------------------------------------------- #
_LSQ_COARSE = dict(method="trf", x_scale="jac", ftol=1e-9, xtol=1e-9,
                   gtol=1e-9, max_nfev=120)
_LSQ_FINE = dict(method="trf", x_scale="jac", ftol=1e-15, xtol=1e-15,
                 gtol=1e-15, max_nfev=400)


def _run_lsq(fun, jac, x0, lo, hi, kw):
    x0 = np.clip(np.asarray(x0, dtype=float),
                 np.asarray(lo) * (1.0 + 1e-12) + 1e-15,
                 np.asarray(hi) * (1.0 - 1e-12))
    try:
        res = least_squares(fun, x0, jac=jac, bounds=(lo, hi), **kw)
    except Exception:
        return None
    if not np.all(np.isfinite(res.x)):
        return None
    return res


def _multistart(fun, jac, starts, lo, hi, n_coarse=5, n_fine=2):
    """Multi-start em dois passes: triagem barata + polimento fino.

    Passe 0: ordena os pontos de partida pelo SSE bruto.
    Passe 1: `least_squares` com tolerância folgada nos `n_coarse` melhores.
    Passe 2: `least_squares` com tolerância apertada nos `n_fine` melhores
    resultados do passe 1. Fica com o menor SSE global.
    """
    scored = []
    for x in starts:
        try:
            r = fun(x)
            s = float(r @ r)
        except Exception:
            continue
        scored.append((s if np.isfinite(s) else np.inf, x))
    if not scored:
        return None, np.inf, False
    scored.sort(key=lambda p: p[0])

    coarse = []
    for _, x0 in scored[:max(1, n_coarse)]:
        res = _run_lsq(fun, jac, x0, lo, hi, _LSQ_COARSE)
        if res is None:
            continue
        s = float(res.fun @ res.fun)
        if np.isfinite(s):
            coarse.append((s, res.x))
    if not coarse:
        return scored[0][1], scored[0][0], False
    coarse.sort(key=lambda p: p[0])

    best_sse, best_x = coarse[0]
    for _, x0 in coarse[:max(1, n_fine)]:
        x, prev = x0, np.inf
        # Re-partidas do ponto convergido: em casos muito mal condicionados
        # (2ª ordem superamortecida truncada) o trf esgota `max_nfev` antes do
        # ótimo; reiniciar a região de confiança termina o serviço. Sai já na
        # 2ª volta quando não há mais ganho, então custa quase nada.
        for _ in range(4):
            res = _run_lsq(fun, jac, x, lo, hi, _LSQ_FINE)
            if res is None:
                break
            s = float(res.fun @ res.fun)
            if not np.isfinite(s):
                break
            x = res.x
            if s < best_sse:
                best_sse, best_x = s, x
            if s > prev * (1.0 - 1e-6):
                break
            prev = s
    return best_x, best_sse, True


def _fopdt_lsq(t, y, cands, n_coarse=5, n_fine=2):
    """Núcleo do ajuste FOPDT: bounds, resíduo, jacobiano analítico e multi-start.

    Compartilhado por `fit_fopdt` e por `_estimate_gain`. `_estimate_gain` passa
    uma lista de candidatos que NÃO inclui os baselines — é o que quebra a
    recursão (os baselines chamam `_estimate_gain`).
    """
    span = _span(t)
    tlo, thi = _theta_bounds(t)
    lo = np.array([K_BOUNDS[0], TAU_BOUNDS[0], tlo])
    hi = np.array([K_BOUNDS[1], TAU_BOUNDS[1], thi])
    fallback = {"K": max(float(np.max(np.abs(y))), K_BOUNDS[0]),
                "tau": 0.25 * span, "theta": 0.0}

    starts = _dedupe([np.array([p["K"], p["tau"], p["theta"]])
                      for p in (_sanitize_fopdt(c, t, fallback) for c in cands)])
    if not starts:
        starts = [np.array([fallback["K"], fallback["tau"], fallback["theta"]])]

    def fun(x):
        return _fopdt_basis(t, x[1], x[2]) * x[0] - y

    def jac(x):
        K, tau, theta = x[0], max(x[1], 1e-30), x[2]
        u = t - theta
        e = np.where(u >= 0.0, np.exp(-np.maximum(u, 0.0) / tau), 0.0)
        uu = np.maximum(u, 0.0)
        col_K = np.where(u >= 0.0, 1.0 - e, 0.0)
        return np.column_stack([col_K,
                                -K * e * uu / (tau * tau),
                                -K * e / tau])

    best_x, _, ok = _multistart(fun, jac, starts, lo, hi,
                                n_coarse=n_coarse, n_fine=n_fine)
    if best_x is None:
        best_x, ok = starts[0], False
    return best_x, fun, bool(ok)


def fit_fopdt(t, y, p0: dict | None = None) -> FitResult:
    """Ajusta FOPDT (K, tau, theta) por mínimos quadrados com multi-start."""
    t, y = _clean(t, y)
    n = t.size
    if n < 4:
        return FitResult("fopdt",
                         {"K": float("nan"), "tau": float("nan"),
                          "theta": float("nan"), "wn": None, "zeta": None},
                         float("inf"), float("nan"), float("nan"), False, 3)

    cands: list[dict] = []
    if p0:
        cands.append(p0)
    g = _integral_guess_fopdt(t, y)
    if g is not None:
        cands.append(g)
    for d in (baseline_sundaresan_krishnaswamy(t, y), baseline_smith(t, y),
              baseline_tangent(t, y)):
        # Baseline que devolveu `nan` (percentil não atingido) não é ponto de
        # partida: entraria como o fallback e só geraria duplicata.
        if np.isfinite(d.get("tau", np.nan)) and np.isfinite(d.get("K", np.nan)):
            cands.append(d)
    cands.extend(_grid_guess_fopdt(t, y, n_keep=3))

    best_x, fun, ok = _fopdt_lsq(t, y, cands)
    resid = fun(best_x)
    sse, nrmse, aic = _metrics(y, resid, 3)
    params = _params_dict("fopdt", best_x[0], best_x[1], best_x[2], None, None)
    return FitResult("fopdt", params, aic, nrmse, sse, bool(ok), 3)


def fit_second(t, y, p0: dict | None = None) -> FitResult:
    """Ajusta 2ª ordem canônica (K, wn, zeta, theta) com multi-start."""
    t, y = _clean(t, y)
    n = t.size
    if n < 5:
        return FitResult("second",
                         {"K": float("nan"), "tau": None,
                          "theta": float("nan"), "wn": float("nan"),
                          "zeta": float("nan")},
                         float("inf"), float("nan"), float("nan"), False, 4)

    span = _span(t)
    tlo, thi = _theta_bounds(t)
    lo = np.array([K_BOUNDS[0], WN_BOUNDS[0], ZETA_BOUNDS[0], tlo])
    hi = np.array([K_BOUNDS[1], WN_BOUNDS[1], ZETA_BOUNDS[1], thi])
    fallback = {"K": max(float(np.max(np.abs(y))), K_BOUNDS[0]),
                "wn": 4.0 / span, "zeta": 1.0, "theta": 0.0}

    cands: list[dict] = []
    if p0:
        cands.append(p0)
    g = _integral_guess_second(t, y)
    if g is not None:
        cands.append(g)
    o = _overshoot_guess(t, y)
    if o is not None:
        cands.append(o)
    cands.append(_overdamped_guess(t, y))
    cands.extend(_grid_guess_second(t, y, n_keep=3))
    # FOPDT convertido em 2ª ordem bem superamortecida (polo rápido distante).
    gf = _integral_guess_fopdt(t, y)
    if gf is not None and np.isfinite(gf.get("tau", np.nan)) and gf["tau"] > 0:
        p1, p2 = 1.0 / gf["tau"], 40.0 / gf["tau"]
        cands.append({"K": gf["K"], "wn": float(np.sqrt(p1 * p2)),
                      "zeta": float(0.5 * (p1 + p2) / np.sqrt(p1 * p2)),
                      "theta": gf["theta"]})

    starts = _dedupe([np.array([p["K"], p["wn"], p["zeta"], p["theta"]])
                      for p in (_sanitize_second(c, t, fallback) for c in cands)])
    if not starts:
        starts = [np.array([fallback["K"], fallback["wn"], fallback["zeta"],
                            fallback["theta"]])]

    def fun(x):
        return _second_basis(t, x[1], x[2], x[3]) * x[0] - y

    def jac(x):
        # d(f)/dv = E*v*S  =>  d/dwn = v^2*ES/wn ; d/dtheta = -wn*v*ES ;
        # d/dzeta = 2*v^3*ED  (ver _second_terms).
        K, wn, zeta, theta = x[0], max(x[1], 1e-30), x[2], x[3]
        u = t - theta
        m = u >= 0.0
        v = wn * np.maximum(u, 0.0)
        EC, ES, ED = _second_terms(zeta, v)
        base = np.where(m, 1.0 - (EC + zeta * v * ES), 0.0)
        vES = np.where(m, v * ES, 0.0)
        return np.column_stack([base,
                                K * v * vES / wn,
                                K * 2.0 * v * v * np.where(m, v * ED, 0.0),
                                -K * wn * vES])

    best_x, _, ok = _multistart(fun, jac, starts, lo, hi)
    if best_x is None:
        best_x, ok = starts[0], False

    resid = fun(best_x)
    sse, nrmse, aic = _metrics(y, resid, 4)
    params = _params_dict("second", best_x[0], None, best_x[3], best_x[1],
                          best_x[2])
    return FitResult("second", params, aic, nrmse, sse, bool(ok), 4)


def identify_both(t, y) -> tuple[FitResult, FitResult]:
    """Ajusta as duas estruturas e devolve (fopdt, second) sem escolher."""
    return fit_fopdt(t, y), fit_second(t, y)


def _rho1(resid: np.ndarray) -> float:
    """Autocorrelação de defasagem 1 do resíduo, saturada em [0, 0.99]."""
    if resid.size < 3:
        return 0.0
    r = resid - resid.mean()
    d = float(r @ r)
    if not np.isfinite(d) or d <= 0.0:
        return 0.0
    return float(np.clip(float(r[:-1] @ r[1:]) / d, 0.0, 0.99))


def _n_efetivo(t: np.ndarray, y: np.ndarray, fit: FitResult) -> float:
    """Nº de pontos EFETIVAMENTE independentes, `n*(1-rho)/(1+rho)`.

    O AIC do `_metrics` usa `n` cru. Isso é correto para observações
    independentes e errado aqui: a série vem de uma polilinha extraída de
    imagem, onde pixels vizinhos carregam erro de extração correlacionado.
    Medido em `data/test` (HANDOFF_P2_7 §9.1), o resíduo tem autocorrelação de
    defasagem 1 de ~0,71 e `n` mediano 738 contra `n_eff` mediano 112 — o AIC
    tratava 738 pontos correlacionados como 738 evidências independentes.

    A consequência era estrutural, não de calibração: a 2ª ordem vence quando
    `SSE1/SSE2 > exp(2/n)`, e com n=806 bastava 0,234 % de ganho de SSE. Ela
    consegue isso ajustando o próprio artefato de extração com o polo extra —
    o NRMSE das duas estruturas empatava (0,00353 x 0,00351). Resultado: 32 %
    das plantas de 1ª ordem eram classificadas como 2ª ordem, contra 6 % quando
    o mesmo estágio D recebe a série verdadeira.

    Trocar AIC por BIC NÃO resolve (corrige só 40 % dos casos): o problema não
    é a constante da penalidade, é o `n` inflado.

    O `rho` sai do resíduo da estrutura MAIS FLEXÍVEL (2ª ordem) de propósito:
    num modelo subespecificado a correlação do resíduo mistura ruído de
    extração com erro de estrutura, e superestimaria a correção.
    """
    resid = y - model_response(fit.order, fit.params, t)
    rho = _rho1(resid)
    n_eff = float(t.size) * (1.0 - rho) / (1.0 + rho)
    return max(n_eff, float(fit.n_params) + 2.0)


def _ganho_num_trecho_so(t: np.ndarray, y: np.ndarray,
                         r1: FitResult, r2: FitResult, frac: float = TRECHO_FRAC) -> float:
    """Quanto da vantagem de SSE do 2ª ordem sobre o FOPDT cabe no MELHOR
    trecho CONTÍGUO de `frac` da série. Acima de 1, fora desse trecho o
    2ª ordem é PIOR que o FOPDT.

    Contíguo, e não "os pontos de maior ganho onde quer que estejam": a
    diferença é o que separa artefato de RENDER de ruído de medição. Um canto
    rasterizado é um acidente LOCAL — ocupa pixels vizinhos; o ruído de
    aquisição é disperso por construção, e a versão dispersa da estatística
    confunde os dois. Medido no estrato do critério 1.2 (série ORÁCULO, sem
    render nenhum, SNR fixo de 20 dB, n=300), acurácia de seleção de estrutura:

        sem guarda ................... 88,3 %
        DISPERSO, 1 % dos pontos ..... 85,7 %   (-2,6 p.p.)
        trecho CONTÍGUO de 3 % ....... 87,7 %   (-0,6 p.p.)

    A versão dispersa custava 2,6 p.p. num caminho que nem tem render — era a
    guarda pegando picos de ruído. A contígua fica dentro do ruído amostral de
    n=300 (1 sigma = 1,9 p.p.).

    `t` precisa estar ordenado — `_clean` garante isso e é por onde `identify`
    passa.
    """
    g = ((y - model_response("fopdt", r1.params, t)) ** 2
         - (y - model_response("second", r2.params, t)) ** 2)
    total = float(g.sum())
    if not np.isfinite(total) or total <= 0.0 or g.size == 0:
        return 0.0
    k = max(1, int(round(frac * g.size)))
    if k >= g.size:
        return 1.0
    c = np.concatenate(([0.0], np.cumsum(g)))
    return float(np.max(c[k:] - c[:-k])) / total


def _polo_rapido_e_artefato(t: np.ndarray, y: np.ndarray,
                            r1: FitResult, r2: FitResult) -> bool:
    """O polo extra do 2ª ordem descreve DINÂMICA ou o traço do próprio render?

    O critério de `identify` compara SSE agregado, e SSE agregado não sabe
    ONDE o ganho aconteceu. Uma 2ª ordem superamortecida pode vencer um FOPDT
    colocando o segundo polo tão rápido que ele só arredonda o CANTO do
    degrau — e o canto de uma curva rasterizada já vem arredondado pela
    espessura da linha e pelo antialiasing, então o "ganho" é o modelo
    ajustando o desenho, não a planta.

    Medido na imagem real do `rg.py` (FOPDT verdadeiro: K=5, tau=1, theta=4):
    o ajuste FOPDT acerta os três parâmetros (K=5,02, tau=0,997, theta=3,99,
    NRMSE=0,0020) e mesmo assim perdia para uma 2ª ordem cujo polo extra fica
    em tau = 45 ms = 2,35 amostras = 3,6 px. DOIS pontos, de 406, respondiam
    por 107,5 % de toda a vantagem de SSE; no resto da série o 2ª ordem era,
    no líquido, pior.

    Daí o teste: se UM trecho contíguo de `TRECHO_FRAC` da série responde por
    100 % ou mais do ganho, então FORA dele o 2ª ordem não vence — a estrutura
    inteira está apoiada num punhado de pixels vizinhos. O limiar 1,0 não é
    calibrado: é a fronteira estrutural do enunciado. Quem foi escolhido por
    medição é só o TAMANHO do trecho (ver `TRECHO_FRAC`).

    Medido no corpus (`data/test`, n=837 com calibração e ajuste), ordem
    contra o `order` do meta:

        ordem correta em plantas de 1ª ordem .... 82,6 % -> 92,1 %  (+41)
        ordem correta em plantas de 2ª ordem .... 95,6 % -> 93,9 %  ( -7)
        ordem correta global ................... 88,89 % -> 92,95 %

    As amostras perdidas são plantas de 2ª ordem genuinamente
    superamortecidas (zeta de 2 a 3,5) cujo polo rápido é quase invisível na
    janela; rebaixá-las a FOPDT preserva o polo DOMINANTE — medido numa
    amostragem delas, o tau reportado ficou a 0,1 % a 1,5 % do tau dominante
    verdadeiro e K a menos de 1 %. Perde-se o rótulo, não a física.

    SÓ VALE PARA `zeta > 1`. Com polos complexos não existe "polo rápido
    separado" a descartar: a oscilação é a assinatura inteira e nenhum FOPDT a
    representa, então rebaixar destruiria justamente o zeta — a grandeza que o
    nível adimensional da Decisão E existe para entregar. Sem a restrição o
    corpus rebaixaria 1 amostra a mais (e ela é de 1ª ordem, ou seja, ganho):
    a restrição custa quase nada e fecha um modo de falha inteiro.
    """
    z = r2.params.get("zeta")
    if z is None or not np.isfinite(z) or z <= 1.0:
        return False
    return _ganho_num_trecho_so(t, y, r1, r2) > 1.0


def identify(t, y) -> FitResult:
    """Estágio D: ajusta FOPDT e 2ª ordem e escolhe pela verossimilhança
    penalizada com nº de pontos EFETIVO (ver `_n_efetivo`), com uma guarda
    contra polo extra que só ajusta o traço do render
    (`_polo_rapido_e_artefato`).

    Equivale ao AIC quando o resíduo é branco; difere dele exatamente na medida
    em que a polilinha extraída é autocorrelacionada. Os campos `.aic` dos dois
    `FitResult` continuam sendo o AIC clássico e NÃO mudaram — `tests/conftest`
    os reporta na Parte 1.

    **Ganho negativo (§40.1).** `K_BOUNDS` é positivo por construção, e alargá-lo
    poria K=0 dentro da caixa — o modelo degenerado, um mínimo local trivial que
    hoje não existe. Em vez disso, uma resposta que DESCE é espelhada antes do
    ajuste e o `K` devolvido troca de sinal. É exatamente equivalente a
    parametrizar `K = s·|K|`, porque `model_response` é linear em K e a base é
    livre de sinal — e não toca uma linha da matemática do módulo. `sse`,
    `nrmse` e `aic` são invariantes ao espelho (dependem de resíduos ao
    quadrado), então nenhuma métrica precisa de correção.

    Com `s = +1` o caminho é byte a byte o de antes desta mudança, e o teste
    `test_caminho_positivo_intocado` assevera isso contra `identify_both`.
    """
    s = _sinal_do_degrau(y)
    if s < 0.0:
        r = _identify_ascendente(t, -np.asarray(y, dtype=float))
        r.params = {k: (-v if (k == "K" and v is not None) else v)
                    for k, v in r.params.items()}
        return r
    return _identify_ascendente(t, y)


def _identify_ascendente(t, y) -> FitResult:
    """O `identify` de sempre, supondo resposta que SOBE. Ver `identify`."""
    r1, r2 = identify_both(t, y)
    if not np.isfinite(r1.aic) and not np.isfinite(r2.aic):
        return r1
    if not np.isfinite(r2.aic):
        return r1
    if not np.isfinite(r1.aic):
        return r2
    tc, yc = _clean(t, y)
    if tc.size < 3:
        return r1 if r1.aic <= r2.aic else r2
    n_eff = _n_efetivo(tc, yc, r2)
    ganho = n_eff * np.log(max(r1.sse, 1e-300) / max(r2.sse, 1e-300))
    if ganho <= 2.0 * (r2.n_params - r1.n_params):
        return r1
    return r1 if _polo_rapido_e_artefato(tc, yc, r1, r2) else r2


# --------------------------------------------------------------------------- #
# Baselines clássicos (só FOPDT)
# --------------------------------------------------------------------------- #
_GAIN_CACHE: dict[tuple[bytes, bytes], float] = {}
_GAIN_CACHE_MAX = 8


def _estimate_gain(t: np.ndarray, y: np.ndarray) -> float:
    """Ganho estático estimado, ou `nan` quando não é identificável na janela.

    Três caminhos, nesta ordem:

    1. **Cauda assentada**: média da cauda. O limiar é apertado (variação da
       cauda < 0.02 % da faixa) de propósito — com 0.5 % o atalho disparava já
       em janela de 5*tau, onde ainda falta e^-5 = 0.67 % para o regime
       permanente, e o ganho saía 0.76 % baixo. A 0.02 % o viés residual fica
       abaixo de 0.05 % e o caso duvidoso cai no caminho 2, que é exato.
    2. **Extrapolação por ajuste FOPDT completo** (grade com K perfilado +
       `least_squares`), que é o mesmo mecanismo que sustenta `fit_fopdt` em
       janela truncada. NÃO usa a regressão integral crua, que colapsa para
       K < 0 abaixo de ~1.8*tau (ver `_integral_guess_fopdt`), e NÃO usa
       `max(y)`/`y[-1]`, proibidos pelo brief.
    3. **`nan`** se o ajuste falhar ou devolver ganho sem sentido. Devolver
       `nan` é o comportamento correto: os baselines então devolvem `nan` em vez
       de inventar, que é exatamente o que o brief pede.

    O caminho 2 é um multi-start de `least_squares`; como os três baselines o
    pedem para a mesma série, o resultado é memoizado pelos bytes de `(t, y)`.
    """
    if t.size < 4:
        return float("nan")
    ntail = max(2, t.size // 20)
    tail = y[-ntail:]
    rng = float(np.max(y) - np.min(y))
    if rng > 0 and float(np.max(tail) - np.min(tail)) < 2e-4 * rng:
        return float(np.mean(tail))

    key = (t.tobytes(), y.tobytes())
    hit = _GAIN_CACHE.get(key)
    if hit is not None:
        return hit

    K = float("nan")
    cands: list[dict] = []
    g = _integral_guess_fopdt(t, y)
    if g is not None:
        cands.append(g)
    cands.extend(_grid_guess_fopdt(t, y, n_keep=3))
    try:
        best_x, _, ok = _fopdt_lsq(t, y, cands, n_coarse=3, n_fine=1)
    except Exception:
        ok, best_x = False, None
    if ok and best_x is not None:
        k = float(best_x[0])
        if np.isfinite(k) and K_BOUNDS[0] < k < K_BOUNDS[1]:
            K = k

    if len(_GAIN_CACHE) >= _GAIN_CACHE_MAX:
        _GAIN_CACHE.clear()
    _GAIN_CACHE[key] = K
    return K


def _crossing_time(t: np.ndarray, y: np.ndarray, level: float) -> float:
    """Primeiro instante em que y cruza `level` (interpolação linear)."""
    if not np.isfinite(level) or t.size < 2:
        return float("nan")
    idx = np.flatnonzero(y >= level)
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return float(t[i])
    frac = (level - y0) / (y1 - y0)
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def _nan_fopdt(K: float) -> dict:
    return {"K": float(K), "tau": float("nan"), "theta": float("nan")}


def baseline_tangent(t, y) -> dict:
    """Método da tangente: reta no ponto de máxima inclinação."""
    t, y = _clean(t, y)
    if t.size < 4:
        return _nan_fopdt(float("nan"))
    K = _estimate_gain(t, y)
    if not np.isfinite(K) or K <= 0:
        return _nan_fopdt(K)
    try:
        # `t` com valores repetidos faz np.gradient dividir por zero; o
        # resultado (inf/nan) já é tratado abaixo, mas o aviso poluiria a saída
        # de um lote de milhares de séries.
        with np.errstate(divide="ignore", invalid="ignore"):
            dydt = np.gradient(y, t)
    except Exception:
        return _nan_fopdt(K)
    dydt = np.where(np.isfinite(dydt), dydt, -np.inf)
    i = int(np.argmax(dydt))
    m = float(dydt[i])
    if not np.isfinite(m) or m <= 0:
        return _nan_fopdt(K)
    t0, y0 = float(t[i]), float(y[i])
    theta = t0 - y0 / m          # interseção da tangente com y = 0
    tau = K / m                  # (interseção com y = K) - theta
    if not np.isfinite(theta) or not np.isfinite(tau) or tau <= 0:
        return _nan_fopdt(K)
    return {"K": float(K), "tau": float(tau), "theta": float(theta)}


def baseline_smith(t, y) -> dict:
    """Smith: theta = 1.5*t28 - 0.5*t63 ; tau = 1.5*(t63 - t28)."""
    t, y = _clean(t, y)
    if t.size < 4:
        return _nan_fopdt(float("nan"))
    K = _estimate_gain(t, y)
    if not np.isfinite(K) or K <= 0:
        return _nan_fopdt(K)
    t28 = _crossing_time(t, y, 0.283 * K)
    t63 = _crossing_time(t, y, 0.632 * K)
    if not (np.isfinite(t28) and np.isfinite(t63)):
        return _nan_fopdt(K)
    theta = 1.5 * t28 - 0.5 * t63
    tau = 1.5 * (t63 - t28)
    if tau <= 0:
        return _nan_fopdt(K)
    return {"K": float(K), "tau": float(tau), "theta": float(theta)}


def baseline_sundaresan_krishnaswamy(t, y) -> dict:
    """S–K: theta = 1.3*t35 - 0.29*t85 ; tau = 0.67*(t85 - t35)."""
    t, y = _clean(t, y)
    if t.size < 4:
        return _nan_fopdt(float("nan"))
    K = _estimate_gain(t, y)
    if not np.isfinite(K) or K <= 0:
        return _nan_fopdt(K)
    t35 = _crossing_time(t, y, 0.353 * K)
    t85 = _crossing_time(t, y, 0.853 * K)
    if not (np.isfinite(t35) and np.isfinite(t85)):
        return _nan_fopdt(K)
    theta = 1.3 * t35 - 0.29 * t85
    tau = 0.67 * (t85 - t35)
    if tau <= 0:
        return _nan_fopdt(K)
    return {"K": float(K), "tau": float(tau), "theta": float(theta)}
