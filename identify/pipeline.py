"""Cola dos estágios A, B e D. Única porta de entrada para a Parte 3."""
from __future__ import annotations

import time

import numpy as np

from identify.calibrate import calibrate
from identify.classical import identify
from identify.extract import predict_mask
from identify.polyline import mask_to_polyline, polyline_to_series


def identify_from_image(image_rgb: np.ndarray, model, device: str = "cpu",
                        extractor=None) -> dict:
    """Imagem -> parâmetros físicos. Nunca levanta: falha vira ok=False.

    `extractor`: opcional, `callable(image_rgb) -> mask uint8 0/255` — troca
    `predict_mask` (U-Net, Bloco 3) por outro extrator com a mesma
    assinatura de saída, ex. `identify.extract_classical.extract_mask_classical`
    (Bloco 3b). Quando `None` (padrão, assinatura idêntica à do
    `PLANO_PARTE2.md`), usa a U-Net via `model`/`device` como sempre.
    """
    t0 = time.perf_counter()
    vazio = {"order": "", "params": {}, "ok": False, "reason": "",
             "latency_ms": 0.0, "n_points": 0}

    cal = calibrate(image_rgb)
    if not cal.ok:
        return {**vazio, "reason": cal.reason,
                "latency_ms": (time.perf_counter() - t0) * 1e3}

    mask = extractor(image_rgb) if extractor is not None else predict_mask(model, image_rgb, device)
    x_px, y_px = mask_to_polyline(mask)
    if x_px.size < 10:
        return {**vazio, "reason": "polilinha_curta",
                "latency_ms": (time.perf_counter() - t0) * 1e3}

    t, y = polyline_to_series(x_px, y_px, cal)
    ordem = np.argsort(t)
    fit = identify(t[ordem], y[ordem])
    return {"order": fit.order, "params": fit.params, "ok": bool(fit.success),
            "reason": "" if fit.success else "ajuste_falhou",
            "latency_ms": (time.perf_counter() - t0) * 1e3,
            "n_points": int(x_px.size)}
