"""Cola dos estágios A, B e D. Única porta de entrada para a Parte 3."""
from __future__ import annotations

import time

import numpy as np

from identify.calibrate import calibrate
from identify.classical import identify
from identify.extract import predict_mask
from identify.polyline import mask_to_polyline, polyline_to_series

# Fração inicial de colunas usada para estimar o nível de repouso no quadro
# normalizado. O degrau parte de zero e o tempo morto mantém a curva parada,
# então as primeiras colunas são o zero. 8 % foi o valor medido no
# HANDOFF_P2_7 §21 (ζ recuperado a 2,93 % de MAPE).
_FRAC_REPOUSO = 0.08


def _serie_normalizada(x_px: np.ndarray, y_px: np.ndarray):
    """Polilinha em pixels -> série adimensional, sem depender de calibração.

    `t` em [0, 1] (a janela observada vira a unidade de tempo) e `y` com o zero
    no nível de repouso e o sinal invertido, porque o pixel cresce para baixo.
    A escala de `y` é arbitrária: ζ é invariante a ela, e `K` sai já dividido
    pela faixa (é o `K_yrange` do PLANO §1.7).

    Devolve `(None, None)` quando não há o que normalizar.
    """
    if x_px.size < 10:
        return None, None
    k = np.argsort(x_px)
    x = np.asarray(x_px, dtype=float)[k]
    y = np.asarray(y_px, dtype=float)[k]
    span = float(x[-1] - x[0])
    if not np.isfinite(span) or span <= 0:
        return None, None
    n0 = max(3, int(_FRAC_REPOUSO * x.size))
    repouso = float(np.median(y[:n0]))
    desvio = repouso - y                       # inverte: pixel cresce p/ baixo
    escala = float(np.max(np.abs(desvio)))
    if not np.isfinite(escala) or escala <= 0:
        return None, None
    return (x - x[0]) / span, desvio / escala


def _vazio_adimensional() -> dict:
    """Bloco `dimensionless` com as chaves presentes e valores nulos.

    O PLANO §1.7 exige o bloco SEMPRE preenchido (critério 2.11). Quando não há
    polilinha suficiente para ajustar nada, o contrato é honrado pela estrutura:
    as chaves existem, os valores são `None`. Nunca ausente, nunca exceção.
    """
    return {"zeta": None, "wn_T": None, "tau_T": None,
            "theta_T": None, "theta_tau": None, "K_yrange": None}


def _adimensional(params: dict, T: float, y_faixa: float) -> dict:
    """Grandezas adimensionais a partir de um ajuste e das escalas observadas.

    `T` é a duração da janela e `y_faixa` a amplitude do sinal, ambas nas MESMAS
    unidades do ajuste. No quadro normalizado as duas valem 1 por construção, o
    que faz esta função servir aos dois caminhos sem ramificar.
    """
    d = _vazio_adimensional()
    T = float(T) if np.isfinite(T) and T > 0 else float("nan")
    yf = float(y_faixa) if np.isfinite(y_faixa) and y_faixa > 0 else float("nan")
    z, wn, tau = params.get("zeta"), params.get("wn"), params.get("tau")
    th, K = params.get("theta"), params.get("K")
    if z is not None:
        d["zeta"] = float(z)
    if wn is not None and np.isfinite(T):
        d["wn_T"] = float(wn) * T
    if tau is not None and np.isfinite(T) and T > 0:
        d["tau_T"] = float(tau) / T
    if th is not None and np.isfinite(T) and T > 0:
        d["theta_T"] = float(th) / T
    if th is not None and tau not in (None, 0) and tau is not None:
        try:
            d["theta_tau"] = float(th) / float(tau)
        except ZeroDivisionError:
            pass
    if K is not None and np.isfinite(yf) and yf > 0:
        d["K_yrange"] = float(K) / yf
    return d


def identify_from_image(image_rgb: np.ndarray, model, device: str = "cpu",
                        extractor=None) -> dict:
    """Imagem -> parâmetros. Nunca levanta: falha vira ok=False.

    `extractor`: opcional, `callable(image_rgb) -> mask uint8 0/255` — troca
    `predict_mask` (U-Net, Bloco 3) por outro extrator com a mesma
    assinatura de saída, ex. `identify.extract_classical.extract_mask_classical`
    (Bloco 3b). Quando `None` (padrão, assinatura idêntica à do
    `PLANO_PARTE2.md`), usa a U-Net via `model`/`device` como sempre.

    **Decisão E do PLANO §1.7 (HANDOFF_P2_7 Rulings 34 e 44).** A saída tem dois
    níveis. O `dimensionless` sai SEMPRE (critério 2.11); o `physical` só quando
    a calibração fecha, e é `None` caso contrário. Antes desta mudança
    `cal.ok == False` abortava a amostra inteira — contra o plano, e custando 56
    das ~68 amostras perdidas do pipeline (Ruling 42), justamente onde ζ é
    recuperável sem calibração nenhuma (medido: 53 recuperadas, ζ a 2,93 % de
    MAPE, contra 2,40 % do caminho físico).

    Duas decisões de compatibilidade, deliberadas:

    1. **`params` e `order` no topo continuam existindo** e continuam sendo os do
       nível FÍSICO, porque `tests/part2` e a Parte 3 os consomem. Os campos
       `physical`/`dimensionless`/`calibration` do PLANO §1.7 entram ao lado.
    2. **`ok` continua significando "há saída física"**, não "há resposta". É o
       que o próprio §1.7 pede em "Consequência nos critérios": 2.3, 2.4 e 2.5
       passam a ser medidos sobre o subconjunto em que a calibração declarou
       sucesso. Quem quiser o nível adimensional lê `dimensionless`, que nunca
       é nulo.

    Quando a calibração fecha, o bloco adimensional é DERIVADO do ajuste físico
    em vez de sair de um segundo ajuste. Isso evita um ajuste a mais por imagem e
    faz os dois níveis concordarem por construção — o §21.3 mediu 50,5 % de
    divergência no p95 entre ajustes independentes, e essa divergência deixa de
    existir aqui.
    """
    t0 = time.perf_counter()

    def _saida(order, params, ok, reason, dim, cal, n_pts):
        fis = dict(params) if (ok and params) else None
        return {
            "order": order, "params": params, "ok": bool(ok), "reason": reason,
            "dimensionless": dim, "physical": fis,
            "calibration": {"ok": bool(cal.ok), "reason": cal.reason,
                            "n_pairs_x": int(cal.n_pairs_x),
                            "n_pairs_y": int(cal.n_pairs_y)},
            "latency_ms": (time.perf_counter() - t0) * 1e3,
            "n_points": int(n_pts),
        }

    cal = calibrate(image_rgb)
    mask = extractor(image_rgb) if extractor is not None else predict_mask(model, image_rgb, device)
    x_px, y_px = mask_to_polyline(mask)

    if x_px.size < 10:
        # Sem polilinha não há nível nenhum: o bloco adimensional existe, vazio.
        return _saida("", {}, False, "polilinha_curta",
                      _vazio_adimensional(), cal, x_px.size)

    if cal.ok:
        t, y = polyline_to_series(x_px, y_px, cal)
        ordem = np.argsort(t)
        t, y = t[ordem], y[ordem]
        fit = identify(t, y)
        dim = (_adimensional(fit.params, float(t[-1] - t[0]), float(np.ptp(y)))
               if fit.success else _vazio_adimensional())
        return _saida(fit.order, fit.params, bool(fit.success),
                      "" if fit.success else "ajuste_falhou", dim, cal, x_px.size)

    # Calibração falhou: só o nível adimensional é possível. No quadro
    # normalizado T = 1 e a faixa de y = 1, então `_adimensional` recebe as duas
    # como 1 e devolve as grandezas já normalizadas.
    tn, yn = _serie_normalizada(x_px, y_px)
    if tn is None:
        return _saida("", {}, False, cal.reason,
                      _vazio_adimensional(), cal, x_px.size)
    g = identify(tn, yn)
    dim = _adimensional(g.params, 1.0, 1.0) if g.success else _vazio_adimensional()
    # `order` e `params` do topo são do nível FÍSICO, que aqui não existe.
    return _saida("", {}, False, cal.reason, dim, cal, x_px.size)
