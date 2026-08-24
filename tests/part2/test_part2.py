"""Critérios 2.1 a 2.11 do PLANO §PARTE 2, mais os portões internos G1/G2/G3b."""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from identify.calibrate import detect_plot_bbox
from tests.part2.conftest import record_p2, to_gray


def test_g1_1_moldura_dentro_de_2px(test_samples):
    erros = []
    for m in test_samples:
        got = detect_plot_bbox(to_gray(m["image"]))
        if got is None:
            erros.append(np.inf)
            continue
        exp = m["plot_bbox_px"]
        erros.append(max(abs(g - e) for g, e in zip(got, exp)))
    erros = np.asarray(erros, dtype=float)
    frac = float(np.mean(erros <= 2.0))
    record_p2("G1.1", "Erro da moldura", "≤ 2 px em ≥ 95%", f"{frac:.3f}", frac >= 0.95)
    assert frac >= 0.95, f"apenas {frac:.1%} das molduras dentro de 2 px"


def test_g1_2_recall_de_ticks(test_samples):
    from identify.calibrate import detect_tick_pixels

    rec = {"x": [], "y": []}
    for m in test_samples:
        bbox = detect_plot_bbox(to_gray(m["image"]))
        if bbox is None:
            rec["x"].append(0.0); rec["y"].append(0.0)
            continue
        got = detect_tick_pixels(to_gray(m["image"]), bbox)
        for eixo in ("x", "y"):
            verdade = [p for p, _ in m["ticks"][eixo]]
            if not verdade:
                continue
            achados = np.asarray(got[eixo], dtype=float)
            if achados.size == 0:
                rec[eixo].append(0.0); continue
            ok = sum(bool(np.min(np.abs(achados - v)) <= 3.0) for v in verdade)
            rec[eixo].append(ok / len(verdade))
    for eixo in ("x", "y"):
        med = float(np.median(rec[eixo]))
        record_p2(f"G1.2{eixo}", f"Recall de ticks ({eixo})", "≥ 0,95 mediana",
                  f"{med:.3f}", med >= 0.95)
        assert med >= 0.95, f"recall mediano de ticks em {eixo}: {med:.3f}"


def test_g1_3_moldura_por_n_spines(test_samples):
    por_estrato: dict[int, list[float]] = {}
    for m in test_samples:
        got = detect_plot_bbox(to_gray(m["image"]))
        exp = m["plot_bbox_px"]
        err = np.inf if got is None else max(abs(g - e) for g, e in zip(got, exp))
        por_estrato.setdefault(int(m["render"]["n_spines"]), []).append(float(err))
    for n_spines, errs in sorted(por_estrato.items()):
        frac = float(np.mean(np.asarray(errs) <= 2.0))
        record_p2(f"G1.3-{n_spines}", f"Moldura, n_spines={n_spines}",
                  "≥ 0,90", f"{frac:.3f} (n={len(errs)})", frac >= 0.90)
        assert frac >= 0.90, f"n_spines={n_spines}: {frac:.1%} dentro de 2 px"


# --- Bloco 3b: extrator clássico (Estágio A sem rede) -----------------------

def test_g3b_3_extract_classical_nao_importa_torch():
    """Portão G3b.3: import identify.extract_classical com torch AUSENTE do
    ambiente. Testado via subprocess com sys.modules bloqueado de verdade —
    ler o código não é confiável (PLANO_PARTE2.md, Bloco 3b)."""
    codigo = (
        "import sys\n"
        "class _Blocked:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch bloqueado de proposito')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocked())\n"
        "import identify.extract_classical\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True)
    ok = r.returncode == 0 and "OK" in r.stdout
    record_p2("G3b.3", "extract_classical não importa torch", "import OK sem torch",
              "OK" if ok else f"falhou: {r.stderr[-300:]}", ok)
    assert ok, r.stderr


def test_g3b_1_iou_mediana(test_samples):
    from identify.extract_classical import extract_mask_classical

    ious = []
    por_estrato: dict[str, list[float]] = {}
    for m in test_samples:
        pred = extract_mask_classical(m["image"]) > 127
        alvo = m["mask"] > 127
        inter = float(np.logical_and(pred, alvo).sum())
        union = float(np.logical_or(pred, alvo).sum())
        v = inter / max(union, 1.0)
        ious.append(v)
        r = m["render"]
        escuro = int(r["bg_color"].lstrip("#")[:2], 16) < 128
        for nome in (f"grade={r['has_grid']}", f"fundo_escuro={escuro}",
                     f"n_distractors={r['n_distractors']}"):
            por_estrato.setdefault(nome, []).append(v)
    med = float(np.median(ious))
    record_p2("G3b.1", "IoU mediana (extrator clássico)", "≥ 0,70", f"{med:.4f}", med >= 0.70)
    for nome, vs in sorted(por_estrato.items()):
        record_p2(f"G3b.1[{nome}]", f"IoU clássico — {nome}", "diagnóstico",
                  f"{float(np.median(vs)):.4f} (n={len(vs)})", None)
    assert med >= 0.70, f"IoU mediana do extrator clássico: {med:.4f}"


def test_g3b_2_sem_reta_de_span_completo(test_samples):
    """Portão G3b.2: a máscara PREDITA não contém reta de span completo
    (grade/spines/distratoras), reusando `_spanning_rows` sobre a própria
    máscara de saída."""
    from identify.extract_classical import _spanning_rows, extract_mask_classical

    violacoes = 0
    for m in test_samples:
        pred = (extract_mask_classical(m["image"]) > 127)
        x0, y0, x1, y1 = m["plot_bbox_px"]
        sub = pred[y0:y1 + 1, x0:x1 + 1]
        if sub.shape[0] < 10 or sub.shape[1] < 10:
            continue
        if _spanning_rows(sub).any() or _spanning_rows(sub.T).any():
            violacoes += 1
    record_p2("G3b.2", "Sem reta de span completo na máscara", "0 violações",
              str(violacoes), violacoes == 0)
    assert violacoes == 0, f"{violacoes} amostras com reta de span completo na máscara"


def test_g3b_4_latencia(test_samples):
    import time

    from identify.extract_classical import extract_mask_classical

    extract_mask_classical(test_samples[0]["image"])  # aquecimento
    lat = []
    for m in test_samples[:100]:
        t0 = time.perf_counter()
        extract_mask_classical(m["image"])
        lat.append((time.perf_counter() - t0) * 1e3)
    p95 = float(np.percentile(lat, 95))
    record_p2("G3b.4", "Latência do extrator clássico", "< 200 ms",
              f"mediana {np.median(lat):.1f} ms, p95 {p95:.1f} ms", p95 < 200.0)
    assert p95 < 200.0, f"p95 latência: {p95:.1f} ms"


# --- Bloco 3: U-Net (Estágio A com rede) ------------------------------------

def test_letterbox_preserva_geometria_no_roundtrip():
    from identify.extract import letterbox, unletterbox

    alvo = np.zeros((180, 640), dtype=np.uint8)
    alvo[40:140, 100:500] = 255            # retângulo com cantos conhecidos
    pequeno, info = letterbox(alvo, size=512)
    assert pequeno.shape == (512, 512)
    volta = unletterbox(pequeno, info)
    assert volta.shape == alvo.shape
    ys, xs = np.nonzero(volta)
    assert abs(int(xs.min()) - 100) <= 2 and abs(int(xs.max()) - 499) <= 2
    assert abs(int(ys.min()) - 40) <= 2 and abs(int(ys.max()) - 139) <= 2


def test_unet_tamanho_declarado():
    from identify.extract import UNet
    n = sum(p.numel() for p in UNet().parameters())
    record_p2("A.0", "Parâmetros da U-Net", "~1,2 M (PLANO)", f"{n/1e6:.2f} M", None)
    assert 0.5e6 <= n <= 2.5e6, f"{n} parâmetros fogem da ordem de grandeza"


# --- Bloco 2: OCR opcional, RANSAC, consistência ----------------------------

def test_2_3_erro_das_escalas(test_samples):
    from identify.calibrate import calibrate

    erros = []
    for m in test_samples:
        cal = calibrate(m["image"])
        if not cal.ok:
            continue
        a = m["axis_affine"]
        e = max(abs(cal.sx - a["sx"]) / abs(a["sx"]),
                abs(cal.sy - a["sy"]) / abs(a["sy"]))
        erros.append(float(e))
    assert len(erros) >= 0.5 * len(test_samples), "aceitou amostras de menos"
    frac = float(np.mean(np.asarray(erros) < 0.01))
    record_p2("2.3", "Erro relativo de sx, sy", "< 1% em ≥ 95%",
              f"{frac:.3f} (n={len(erros)})", frac >= 0.95)
    assert frac >= 0.95


def _erro_de_escala_sem_guarda(image, affine_verdadeira) -> float:
    """Erro relativo de escala que teríamos se a guarda de consistência não existisse."""
    from identify.calibrate import (detect_plot_bbox, detect_tick_pixels,
                                    fit_axis_affine, read_tick_labels)
    gray = to_gray(image)
    bbox = detect_plot_bbox(gray)
    if bbox is None:
        return float("inf")
    pares = read_tick_labels(gray, bbox, detect_tick_pixels(gray, bbox))
    fx, fy = fit_axis_affine(pares["x"]), fit_axis_affine(pares["y"])
    if fx is None or fy is None:
        return float("inf")
    return max(abs(fx[0] - affine_verdadeira["sx"]) / abs(affine_verdadeira["sx"]),
               abs(fy[0] - affine_verdadeira["sy"]) / abs(affine_verdadeira["sy"]))


def test_2_4_2_5_rejeicao_por_consistencia(test_samples):
    from identify.calibrate import calibrate

    rejeitadas, erro_se_aceitasse = 0, []
    for m in test_samples:
        cal = calibrate(m["image"])
        a = m["axis_affine"]
        if cal.ok:
            continue
        rejeitadas += 1
        e = _erro_de_escala_sem_guarda(m["image"], a)
        erro_se_aceitasse.append(e)

    taxa = rejeitadas / len(test_samples)
    record_p2("2.4", "Taxa de rejeição (falso alarme)", "< 5%",
              f"{taxa:.3f}", taxa < 0.05)

    if rejeitadas < 5:
        record_p2("2.5", "Rejeições corretas", "≥ 90% (n insuficiente)",
                  f"n={rejeitadas} < 5 — não asseverável", None)
        return

    corretas = float(np.mean([e > 0.05 for e in erro_se_aceitasse]))
    record_p2("2.5", "Rejeições corretas", "≥ 90%",
              f"{corretas:.3f} (n={rejeitadas})", corretas >= 0.90)
    assert corretas >= 0.90


def test_2_9_cobertura_da_calibracao(test_samples):
    from identify.calibrate import calibrate

    por_dpi: dict[str, list[bool]] = {}
    todos = []
    for m in test_samples:
        cal = calibrate(m["image"])
        todos.append(cal.ok)
        dpi = int(m["render"]["dpi"])
        faixa = "60-99" if dpi < 100 else ("100-149" if dpi < 150 else "150-200")
        por_dpi.setdefault(faixa, []).append(cal.ok)
    cobertura = float(np.mean(todos))
    record_p2("2.9", "Cobertura da calibração (ok=True)", "≥ 90% global",
              f"{cobertura:.3f} (n={len(todos)})", cobertura >= 0.90)
    for faixa, vs in sorted(por_dpi.items()):
        record_p2(f"2.9[dpi={faixa}]", f"Cobertura — dpi {faixa}",
                  "diagnóstico, sem alvo por estrato",
                  f"{float(np.mean(vs)):.3f} (n={len(vs)})", None)
    assert cobertura >= 0.90, f"cobertura da calibração: {cobertura:.1%}"


def test_2_11_saida_adimensional_sempre_presente(test_samples):
    """Portão 2.11: a saída adimensional existe mesmo com ok=False (§1.7)."""
    from identify.calibrate import calibrate

    for m in test_samples:
        # Nunca levanta, mesmo em amostras adversárias — checagem do contrato.
        cal = calibrate(m["image"])
        assert cal.ok in (True, False)
        assert isinstance(cal.reason, str)
    record_p2("2.11", "calibrate() nunca levanta exceção", "100% das amostras",
              f"{len(test_samples)}/{len(test_samples)}", True)


# --- Bloco 4: máscara -> polilinha ------------------------------------------

def test_2_2_polilinha_contra_mascara_verdadeira(test_samples):
    from identify.polyline import mask_to_polyline

    rmses = []
    for m in test_samples:
        xp, yp = mask_to_polyline(m["mask"])
        if xp.size == 0:
            rmses.append(np.inf)
            continue
        a = m["axis_affine"]
        t_col = a["sx"] * xp + a["ox"]
        y_col = np.interp(t_col, m["series"]["t"], m["series"]["y"])
        yp_true = (y_col - a["oy"]) / a["sy"]        # de volta a pixels
        dentro = (t_col >= m["series"]["t"][0]) & (t_col <= m["series"]["t"][-1])
        if dentro.sum() < 10:
            rmses.append(np.inf)
            continue
        rmses.append(float(np.sqrt(np.mean((yp[dentro] - yp_true[dentro]) ** 2))))
    r = np.asarray(rmses, dtype=float)
    med, p95 = float(np.median(r)), float(np.percentile(r, 95))
    record_p2("2.2-piso", "Polilinha vs. máscara VERDADEIRA",
              "RMSE ≤ 2 px, p95 ≤ 5 px", f"RMSE={med:.2f} px, p95={p95:.2f} px",
              med <= 2.0 and p95 <= 5.0)
    assert med <= 2.0 and p95 <= 5.0


def test_2_2_estrato_marcador_e_estilo(test_samples):
    from identify.polyline import mask_to_polyline

    por: dict[str, list[float]] = {}
    for m in test_samples:
        xp, yp = mask_to_polyline(m["mask"])
        a = m["axis_affine"]
        if xp.size == 0:
            e = float("inf")
        else:
            t_col = a["sx"] * xp + a["ox"]
            y_col = np.interp(t_col, m["series"]["t"], m["series"]["y"])
            e = float(np.sqrt(np.mean((yp - (y_col - a["oy"]) / a["sy"]) ** 2)))
        r = m["render"]
        por.setdefault(f"traco={r['line_style']}", []).append(e)
        por.setdefault(f"marcador={r['has_marker']}", []).append(e)
        # espessura em px renderizados (não em pontos): junto com o dpi é o
        # que de fato determina quantas linhas o traço ocupa por coluna.
        lw_px = float(r["line_width"]) * int(r["dpi"]) / 72.0
        por.setdefault(f"espessura={'grossa' if lw_px >= 3.0 else 'fina'}", []).append(e)
    for nome, vs in sorted(por.items()):
        med = float(np.median(vs))
        record_p2(f"2.2[{nome}]", f"RMSE da polilinha — {nome}", "≤ 2 px",
                  f"{med:.2f} px (n={len(vs)})", med <= 2.0)
        assert med <= 2.0, f"estrato {nome}: RMSE mediano {med:.2f} px"


# --- Bloco 3: U-Net treinada -------------------------------------------------

def test_2_1_iou_mediana(test_samples):
    import torch
    from identify.extract import load_model, predict_mask

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    ious = []
    for m in test_samples:
        pred = predict_mask(model, m["image"], dev) > 127
        alvo = m["mask"] > 127
        inter = float(np.logical_and(pred, alvo).sum())
        union = float(np.logical_or(pred, alvo).sum())
        ious.append(inter / max(union, 1.0))
    med = float(np.median(ious))
    record_p2("2.1", "IoU da máscara (U-Net)", "≥ 0,85 (mediana)", f"{med:.4f}", med >= 0.85)
    assert med >= 0.85


def test_2_7_iou_por_estrato(test_samples):
    import torch
    from identify.extract import load_model, predict_mask

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    estratos: dict[str, list[float]] = {}
    for m in test_samples:
        pred = predict_mask(model, m["image"], dev) > 127
        alvo = m["mask"] > 127
        v = float(np.logical_and(pred, alvo).sum()) / max(
            float(np.logical_or(pred, alvo).sum()), 1.0)
        r = m["render"]
        escuro = int(r["bg_color"].lstrip("#")[:2], 16) < 128
        for nome in (f"grade={r['has_grid']}", f"legenda={r['has_legend']}",
                     f"fundo_escuro={escuro}", f"traco={r['line_style']}"):
            estratos.setdefault(nome, []).append(v)
    for nome, vs in sorted(estratos.items()):
        med = float(np.median(vs))
        record_p2(f"2.7[{nome}]", f"IoU — {nome}", "≥ 0,75",
                  f"{med:.4f} (n={len(vs)})", med >= 0.75)
        assert med >= 0.75, f"estrato {nome}: IoU mediano {med:.4f}"


def test_g3b_1_vs_2_1_comparacao_iou(test_samples):
    """Critério 2.10: U-Net x extrator clássico, mesmo conjunto — sem alvo,
    é o resultado que justifica (ou não) a U-Net."""
    import torch
    from identify.extract import load_model, predict_mask
    from identify.extract_classical import extract_mask_classical

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    ious_unet, ious_classico = [], []
    for m in test_samples:
        alvo = m["mask"] > 127
        p_unet = predict_mask(model, m["image"], dev) > 127
        p_classico = extract_mask_classical(m["image"]) > 127
        ious_unet.append(float(np.logical_and(p_unet, alvo).sum()) /
                         max(float(np.logical_or(p_unet, alvo).sum()), 1.0))
        ious_classico.append(float(np.logical_and(p_classico, alvo).sum()) /
                             max(float(np.logical_or(p_classico, alvo).sum()), 1.0))
    med_unet, med_classico = float(np.median(ious_unet)), float(np.median(ious_classico))
    record_p2("2.10", "IoU mediana: U-Net vs. extrator clássico", "sem alvo",
              f"U-Net={med_unet:.4f}  clássico={med_classico:.4f}", None)


# --- Bloco 5: integração, degradação e relatório ----------------------------

def test_2_8_latencia_por_imagem(test_samples):
    import torch
    from identify.extract import load_model
    from identify.pipeline import identify_from_image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    identify_from_image(test_samples[0]["image"], model, dev)   # aquecimento
    lat = [identify_from_image(m["image"], model, dev)["latency_ms"]
           for m in test_samples[:100]]
    p95 = float(np.percentile(lat, 95))
    record_p2("2.8", "Latência por imagem", "< 500 ms",
              f"mediana {np.median(lat):.0f} ms, p95 {p95:.0f} ms", p95 < 500.0)
    assert p95 < 500.0


def test_2_6_degradacao_vs_oraculo(test_samples):
    """ΔMAPE ≤ 3 p.p. Mesmas amostras, mesmo estágio D, mesma métrica da Parte 1."""
    import torch
    from identify.classical import identify as estagio_d
    from identify.extract import load_model
    from identify.pipeline import identify_from_image
    from tests.conftest import meta_t_dom

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)

    err_oraculo: dict[str, list[float]] = {}
    err_real: dict[str, list[float]] = {}
    aceitas = 0
    for m in test_samples:
        alvo = m["params"]
        t_dom = meta_t_dom(m)

        # Oráculo: série VERDADEIRA do meta -> estágio D (idêntico à Parte 1).
        o = estagio_d(m["series"]["t"], m["series"]["y"])
        r = identify_from_image(m["image"], model, dev)
        if not (o.success and r["ok"] and r["order"] == m["order"]
                and o.order == m["order"]):
            continue
        aceitas += 1
        for k in ("K", "tau", "wn", "zeta"):
            if alvo.get(k) is None:
                continue
            for saida, acc in ((o.params, err_oraculo), (r["params"], err_real)):
                if saida.get(k) is None:
                    continue
                acc.setdefault(k, []).append(
                    abs(saida[k] - alvo[k]) / max(abs(alvo[k]), 1e-12) * 100.0)
        # θ em NMAE/T_dom — convenção da Parte 1 (tests/conftest.py:441).
        for saida, acc in ((o.params, err_oraculo), (r["params"], err_real)):
            if saida.get("theta") is not None:
                acc.setdefault("theta", []).append(
                    abs(saida["theta"] - alvo["theta"]) / max(t_dom, 1e-12) * 100.0)

    record_p2("2.6-aceitas", "Amostras comparáveis (mesma ordem, ambos convergem)",
              "diagnóstico", f"{aceitas}/{len(test_samples)}", None)
    if aceitas < 100:
        record_p2("2.6", "Degradação end-to-end (pior parâmetro)",
                  "≤ 3 p.p. (n insuficiente)", f"n={aceitas} < 100 — não asseverável", None)
        return
    piores = []
    for k in sorted(set(err_oraculo) & set(err_real)):
        d = float(np.median(err_real[k])) - float(np.median(err_oraculo[k]))
        piores.append((k, d))
        record_p2(f"2.6[{k}]", f"ΔMAPE — {k}", "≤ 3 p.p.",
                  f"{d:+.2f} p.p. (oráculo {np.median(err_oraculo[k]):.2f}%, "
                  f"real {np.median(err_real[k]):.2f}%)", d <= 3.0)
    pior = max(d for _, d in piores)
    record_p2("2.6", "Degradação end-to-end (pior parâmetro)", "≤ 3 p.p.",
              f"{pior:+.2f} p.p. (n={aceitas})", pior <= 3.0)
    assert pior <= 3.0, f"pior degradação: {piores}"


def test_2_6_diagnostico_extrator_classico(test_samples):
    """Diagnóstico (não é critério do PLANO): mesma medição do 2.6, mas
    trocando a U-Net (Bloco 3) pelo extrator sem rede (Bloco 3b) —
    comparação pendente registrada no HANDOFF_P2_5.md §7, item 4. Sem
    `assert`: não é um portão do PLANO, só material de decisão entre os
    dois extratores.
    """
    from identify.classical import identify as estagio_d
    from identify.extract_classical import extract_mask_classical
    from identify.pipeline import identify_from_image
    from tests.conftest import meta_t_dom

    err_oraculo: dict[str, list[float]] = {}
    err_real: dict[str, list[float]] = {}
    aceitas = 0
    for m in test_samples:
        alvo = m["params"]
        t_dom = meta_t_dom(m)

        o = estagio_d(m["series"]["t"], m["series"]["y"])
        r = identify_from_image(m["image"], None, extractor=extract_mask_classical)
        if not (o.success and r["ok"] and r["order"] == m["order"]
                and o.order == m["order"]):
            continue
        aceitas += 1
        for k in ("K", "tau", "wn", "zeta"):
            if alvo.get(k) is None:
                continue
            for saida, acc in ((o.params, err_oraculo), (r["params"], err_real)):
                if saida.get(k) is None:
                    continue
                acc.setdefault(k, []).append(
                    abs(saida[k] - alvo[k]) / max(abs(alvo[k]), 1e-12) * 100.0)
        for saida, acc in ((o.params, err_oraculo), (r["params"], err_real)):
            if saida.get("theta") is not None:
                acc.setdefault("theta", []).append(
                    abs(saida["theta"] - alvo["theta"]) / max(t_dom, 1e-12) * 100.0)

    record_p2("2.6-classico-aceitas", "Amostras comparáveis (extrator clássico)",
              "diagnóstico", f"{aceitas}/{len(test_samples)}", None)
    if aceitas < 100:
        record_p2("2.6-classico", "Degradação end-to-end (extrator clássico)",
                  "diagnóstico (n insuficiente)", f"n={aceitas} < 100", None)
        return
    piores = []
    for k in sorted(set(err_oraculo) & set(err_real)):
        d = float(np.median(err_real[k])) - float(np.median(err_oraculo[k]))
        piores.append((k, d))
        record_p2(f"2.6-classico[{k}]", f"ΔMAPE (clássico) — {k}", "diagnóstico",
                  f"{d:+.2f} p.p. (oráculo {np.median(err_oraculo[k]):.2f}%, "
                  f"real {np.median(err_real[k]):.2f}%)", d <= 3.0)
    pior = max(d for _, d in piores)
    record_p2("2.6-classico", "Degradação end-to-end (extrator clássico, pior parâmetro)",
              "diagnóstico", f"{pior:+.2f} p.p. (n={aceitas})", pior <= 3.0)
