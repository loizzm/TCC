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
    from tests.part2.conftest import ESCALA_TOL

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
    # ESCALA_TOL, não 0.01 literal: o alinhamento com o 2.5 (Ruling 52) precisa ser
    # garantido por construção, não por coincidência de valor.
    frac = float(np.mean(np.asarray(erros) < ESCALA_TOL))
    record_p2("2.3", "Erro relativo de sx, sy", f"< {ESCALA_TOL:.0%} em ≥ 95%",
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
    from tests.part2.conftest import ESCALA_TOL

    rejeitadas, erro_se_aceitasse = 0, []
    for m in test_samples:
        cal = calibrate(m["image"])
        a = m["axis_affine"]
        if cal.ok:
            continue
        rejeitadas += 1
        e = _erro_de_escala_sem_guarda(m["image"], a)
        erro_se_aceitasse.append(e)

    # 2.4 APOSENTADO como critério (PLANO §2.4, HANDOFF_P2_7 Ruling 52):
    # `rejeitadas/total` é exatamente `1 - cobertura` do 2.9 — a MESMA grandeza,
    # com limiar diferente (< 5% de rejeição x >= 90% de cobertura). O 2.4
    # subsumia o 2.9 e dava peso duplo a uma medição. Fica como diagnóstico; o
    # veredito de cobertura é do 2.9.
    taxa = rejeitadas / len(test_samples)
    record_p2("2.4", "Taxa de rejeição (diagnóstico — unificado no 2.9)",
              "sem alvo próprio: é 1 − cobertura do 2.9 (Ruling 52)",
              f"{taxa:.3f}", None)

    if rejeitadas < 5:
        record_p2("2.5", "Rejeições corretas", "≥ 90% (n insuficiente)",
                  f"n={rejeitadas} < 5 — não asseverável", None)
        return

    # 2.5 com o limiar ALINHADO ao do 2.3 (Ruling 52). Antes usava 5% enquanto o
    # 2.3 usava 1%, e as amostras na faixa intermediária eram simultaneamente
    # "não deviam ter sido rejeitadas" (2.5) e "ruins o bastante para estragar o
    # 2.3" — nenhum subconjunto satisfazia os dois.
    corretas = float(np.mean([e > ESCALA_TOL for e in erro_se_aceitasse]))
    record_p2("2.5", "Rejeições corretas",
              f"≥ 90% (erro > {ESCALA_TOL:.0%}, alinhado ao 2.3)",
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


CHAVES_ADIM = {"zeta", "wn_T", "tau_T", "theta_T", "theta_tau", "K_yrange"}


def test_2_11_saida_adimensional_sempre_presente(test_samples):
    """Portão 2.11: a saída adimensional existe mesmo com ok=False (§1.7).

    Este teste MEDIA COISA OUTRA até o Bloco 7 (HANDOFF_P2_7 Ruling 34): ele só
    verificava que `calibrate()` não levanta exceção, e passava com a Decisão E
    inteira não implementada. O `PLANO.md:299` pede "100% das amostras com
    `dimensionless` preenchido e nenhuma exceção levantada", o que é sobre a
    saída de `identify_from_image`, não sobre o calibrador.
    """
    import torch
    from identify.calibrate import calibrate
    from identify.extract import load_model
    from identify.pipeline import identify_from_image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)

    com_bloco = com_valor = 0
    sem_calibracao = adim_sem_calibracao = 0
    for m in test_samples:
        # Nenhuma das duas portas levanta, nem em amostras adversárias.
        cal = calibrate(m["image"])
        assert cal.ok in (True, False)
        assert isinstance(cal.reason, str)

        r = identify_from_image(m["image"], model, dev)
        # Estrutura: o bloco existe sempre, com todas as chaves.
        assert isinstance(r.get("dimensionless"), dict), r.get("reason")
        assert CHAVES_ADIM <= set(r["dimensionless"]), r["dimensionless"].keys()
        com_bloco += 1
        # `physical` é null exatamente quando não há saída física.
        assert (r["physical"] is None) == (not r["ok"])
        assert r["calibration"]["ok"] == cal.ok
        if any(v is not None for v in r["dimensionless"].values()):
            com_valor += 1
        if not cal.ok:
            sem_calibracao += 1
            if r["dimensionless"].get("zeta") is not None or \
               r["dimensionless"].get("tau_T") is not None:
                adim_sem_calibracao += 1

    n = len(test_samples)
    record_p2("2.11", "Bloco `dimensionless` presente em toda amostra",
              "100% das amostras, sem exceção",
              f"{com_bloco}/{n} com bloco; {com_valor}/{n} com valor; "
              f"{adim_sem_calibracao}/{sem_calibracao} das sem calibração",
              com_bloco == n)
    assert com_bloco == n


# --- Bloco 4: máscara -> polilinha ------------------------------------------

def test_2_2_polilinha_contra_mascara_verdadeira(test_samples):
    """Portão 2.2 REVISADO (PLANO §2.2, HANDOFF_P2_7 Ruling 50).

    Media diferença VERTICAL até o Bloco 7. Num trecho de inclinação `m`, um erro
    geométrico de meio pixel aparece como `m/2` px de erro vertical — a métrica
    respondia à declividade do render (Spearman +0,869 com a inclinação) e não à
    geometria (perpendicular: +0,326). Dezenove tentativas de melhorar a redução
    coluna->ponto ficaram todas PIORES que a atual, e um extrator oráculo que
    passa a métrica antiga não recupera acurácia significativa (Ruling 49): o erro
    que o critério antigo penalizava não existia.

    O número vertical continua REPORTADO como diagnóstico, sem alvo.
    """
    from identify.polyline import mask_to_polyline
    from tests.part2.conftest import (PERP_MED_MAX, PERP_P95_MAX,
                                      erro_perpendicular)

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
    record_p2("2.2-piso-vertical",
              "Polilinha vs. máscara VERDADEIRA, diferença VERTICAL (diagnóstico)",
              "sem alvo: responde à declividade do render (Ruling 50)",
              f"RMSE={med:.2f} px, p95={p95:.2f} px", None)

    # --- métrica REVISADA: distância perpendicular (PLANO §2.2, Ruling 50) ----
    pm = [erro_perpendicular(*mask_to_polyline(m["mask"])[:2], m) for m in test_samples]
    pm = [e for e in pm if e is not None]
    pmed = float(np.median([e["rmse"] for e in pm]))
    pp95 = float(np.percentile([e["rmse"] for e in pm], 95))
    record_p2("2.2-piso", "Polilinha vs. máscara VERDADEIRA (perpendicular)",
              f"RMSE ≤ {PERP_MED_MAX:.1f} px, p95 ≤ {PERP_P95_MAX:.1f} px",
              f"RMSE={pmed:.3f} px, p95={pp95:.3f} px (n={len(pm)})",
              pmed <= PERP_MED_MAX and pp95 <= PERP_P95_MAX)
    assert pmed <= PERP_MED_MAX and pp95 <= PERP_P95_MAX


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

def test_2_1_erro_geometrico_da_mascara(test_samples):
    """Portão 2.1 REVISADO (PLANO §2.1, HANDOFF_P2_7 Ruling 50).

    Media IoU de máscara até o Bloco 7. Numa curva fina a área é dominada pela
    espessura do traço: IoU correlaciona +0,860 com a tinta por coluna e -0,879
    com a razão de espessura, contra só +0,284 com o deslocamento real. Em todas
    as faixas de espessura o erro de centerline é 1,00 px CONSTANTE enquanto o
    IoU vai de 0,468 a 0,782 — a métrica lia a largura de linha que o gerador
    sorteia. Dice não ajudaria: é `2*IoU/(1+IoU)` por identidade exata.

    O IoU continua REPORTADO como diagnóstico, sem alvo, para preservar a
    comparabilidade com as rodadas 3 a 6.
    """
    import torch
    from identify.extract import load_model, predict_mask
    from identify.polyline import mask_to_polyline
    from tests.part2.conftest import (PERP_MED_MAX, PERP_P95_MAX,
                                      erro_perpendicular)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    ious, meds, p95s = [], [], []
    for m in test_samples:
        pred = predict_mask(model, m["image"], dev) > 127
        alvo = m["mask"] > 127
        inter = float(np.logical_and(pred, alvo).sum())
        union = float(np.logical_or(pred, alvo).sum())
        ious.append(inter / max(union, 1.0))
        xq, yq = mask_to_polyline((pred * 255).astype(np.uint8))
        e = erro_perpendicular(xq, yq, m)
        if e is not None:
            # RMSE por amostra, a MESMA estatística de que o limiar foi derivado
            # (Ruling 50: 0,800 px medido -> +0,127 p.p. em ζ). Usar a mediana por
            # amostra aqui deixaria o critério mais frouxo que a justificativa.
            meds.append(e["rmse"]); p95s.append(e["p95"])

    med = float(np.median(meds)); p95 = float(np.percentile(meds, 95))
    record_p2("2.1", "Erro perpendicular da máscara (U-Net)",
              f"≤ {PERP_MED_MAX:.1f} px mediana, ≤ {PERP_P95_MAX:.1f} px p95",
              f"{med:.3f} px / {p95:.3f} px (n={len(meds)})",
              med <= PERP_MED_MAX and p95 <= PERP_P95_MAX)
    record_p2("2.1-iou", "IoU da máscara (diagnóstico — ver Ruling 50)",
              "sem alvo: mede espessura de traço, não acurácia",
              f"{float(np.median(ious)):.4f}", None)
    assert med <= PERP_MED_MAX and p95 <= PERP_P95_MAX


def test_2_7_iou_por_estrato(test_samples):
    import torch
    from identify.extract import load_model, predict_mask

    from identify.polyline import mask_to_polyline
    from tests.part2.conftest import PERP_MED_MAX, erro_perpendicular

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    estratos: dict[str, list[tuple]] = {}
    for m in test_samples:
        pred = predict_mask(model, m["image"], dev) > 127
        alvo = m["mask"] > 127
        v = float(np.logical_and(pred, alvo).sum()) / max(
            float(np.logical_or(pred, alvo).sum()), 1.0)
        xq, yq = mask_to_polyline((pred * 255).astype(np.uint8))
        e = erro_perpendicular(xq, yq, m)
        v = (v, None if e is None else e["rmse"])
        r = m["render"]
        escuro = int(r["bg_color"].lstrip("#")[:2], 16) < 128
        for nome in (f"grade={r['has_grid']}", f"legenda={r['has_legend']}",
                     f"fundo_escuro={escuro}", f"traco={r['line_style']}"):
            estratos.setdefault(nome, []).append(v)
    for nome, vs in sorted(estratos.items()):
        med = float(np.median([v for v, _ in vs]))
        perp = [p for _, p in vs if p is not None]
        pmed = float(np.median(perp)) if perp else float("inf")
        record_p2(f"2.7[{nome}]", f"Erro perpendicular — {nome}",
                  f"≤ {PERP_MED_MAX:.1f} px mediana",
                  f"{pmed:.3f} px (n={len(perp)})", pmed <= PERP_MED_MAX)
        record_p2(f"2.7-iou[{nome}]", f"IoU — {nome} (diagnóstico)",
                  "sem alvo (Ruling 50)", f"{med:.4f} (n={len(vs)})", None)
        assert pmed <= PERP_MED_MAX, f"estrato {nome}: perpendicular {pmed:.3f} px"


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
    # Nível ADIMENSIONAL (Decisão E, PLANO §1.7). ζ é adimensional e não depende
    # de calibração, então este acumulador NÃO exige `r["ok"]` — só que a
    # estrutura tenha sido acertada. É o que faz as amostras sem calibração
    # entrarem no 2.6, que antes as descartava inteiras (HANDOFF_P2_7 Ruling 42:
    # a calibração derrubava 56 das ~68 amostras perdidas).
    err_orac_adim: list[float] = []
    err_real_adim: list[float] = []
    aceitas_adim = 0
    sem_calib_adim = 0
    # ωₙ adimensional (ω_n·T), diagnóstico (fix round 1, B3): não existia
    # nenhum critério `2.6-adim[wn]` — só `2.6-adim[zeta]`, e ζ é invariante à
    # escala de `t`. Isso deixou uma regressão de escala de `t` passar verde
    # pela suíte inteira. `T_true` vem de `t_window` (janela REAL da série, sem
    # a margem do matplotlib que `plot_bbox_px`/a moldura incluem) — é o mesmo
    # `T` que `_adimensional` usaria se a calibração fechasse. Sem meta de
    # aprovação de propósito: o pedido é só tornar o número visível.
    #
    # Os acumuladores são DOIS (fix round 2, C4). O caminho que este
    # diagnóstico vigia — `_serie_normalizada` — só roda quando a calibração
    # FALHA, e essas são minoria aqui (medido: 33 das 143 aceitas). Medir só o
    # corpus dilui o sinal ~7×: reintroduzindo a moldura incondicional, a linha
    # do corpus se move +0,78 p.p. (+1,04 -> +1,82) e a das sem calibração
    # +5,80 p.p. (+0,63 -> +6,43). A linha do corpus fica porque é a que se
    # compara com `2.6-adim[zeta]`; a das sem calibração é a que denuncia.
    err_orac_wnT_adim: list[float] = []
    err_real_wnT_adim: list[float] = []
    aceitas_wnT_adim = 0
    err_orac_wnT_sc: list[float] = []
    err_real_wnT_sc: list[float] = []
    # θ adimensional (θ/T), diagnóstico (revisão final, §6). MESMO padrão e
    # MESMO motivo do bloco de ωₙ acima, um eixo ao lado: `_serie_normalizada`
    # produz DOIS parâmetros de referência de tempo — escala E origem — e até
    # aqui a suíte vigiava só a escala. ζ é invariante às duas, ωₙ·T só pega a
    # escala; quem pega a ORIGEM é θ. Medido pela revisão: reverter
    # `origem = bbox_px[0]` para `x[0]` em `identify/pipeline.py` deixa a suíte
    # inteira verde — o mesmo ponto cego do Ruling 67, no mesmo arquivo. E não é
    # hipotético: essa origem já valeu ~25 % de erro em θ (fix round 1, B1),
    # achada por revisor humano e não por teste.
    #
    # A métrica é a da Parte 1 para θ (NMAE sobre a janela, tests/conftest.py),
    # não MAPE: θ pode ser 0 e a razão relativa explode. Aqui a janela é T, e
    # `theta_T` JÁ é θ/T, então a diferença absoluta ×100 é o erro em % de T.
    # Dois acumuladores pelo mesmo motivo do ωₙ (fix round 2, C4): o caminho
    # vigiado só roda sem calibração, e medir só o corpus dilui o sinal ~7×.
    # Sem meta de aprovação: o pedido é tornar o número visível.
    err_orac_thT_adim: list[float] = []
    err_real_thT_adim: list[float] = []
    aceitas_thT_adim = 0
    err_orac_thT_sc: list[float] = []
    err_real_thT_sc: list[float] = []
    # Acerto de ORDEM (revisão final, §6). `order` é saída ADIMENSIONAL por
    # contrato (PLANO §1.7) e o PRIMEIRO dos dois defeitos que a imagem real
    # expôs era ordem errada — mas nenhum `record_p2` da suíte media ordem: ela
    # entrava só como FILTRO de aceitação dos blocos acima. Essa é a forma
    # clássica do ponto cego: uma regressão de ordem não piora mediana nenhuma,
    # ela ENCOLHE a amostra, e as que somem são as difíceis, então as medianas
    # que restam até melhoram. Já em movimento sem ninguém ver:
    # `2.6-classico-aceitas` caiu 183 -> 181 neste bloco, e o estrato da reta de
    # referência vira a ordem de 1 em 30 amostras (HANDOFF_P2_7 §35.6).
    ordem_ok = 0
    ordem_ok_sc = 0
    ordem_n_sc = 0
    # K adimensional (K/faixa de y), diagnóstico (re-review, R-5). TERCEIRO
    # eixo do mesmo padrão, depois da escala e da origem de `t` — e os dois
    # primeiros já morderam. `K_yrange` é a única das seis grandezas do bloco
    # `dimensionless` sensível à escala de **y**, e nada a vigiava: `2.6[K]` é
    # do caminho FÍSICO, que só existe quando a calibração fecha. A verdade é
    # `K_alvo / ptp(série verdadeira)`, as duas do meta, na mesma unidade em
    # que o oráculo é avaliado. Sem meta de aprovação, como os irmãos.
    err_orac_Kyr_adim: list[float] = []
    err_real_Kyr_adim: list[float] = []
    aceitas_Kyr_adim = 0
    err_orac_Kyr_sc: list[float] = []
    err_real_Kyr_sc: list[float] = []
    for m in test_samples:
        alvo = m["params"]
        t_dom = meta_t_dom(m)

        # Oráculo: série VERDADEIRA do meta -> estágio D (idêntico à Parte 1).
        o = estagio_d(m["series"]["t"], m["series"]["y"])
        r = identify_from_image(m["image"], model, dev)

        z_adim = (r.get("dimensionless") or {}).get("zeta")
        if (o.success and o.order == m["order"] and r["order"] == m["order"]
                and z_adim is not None and alvo.get("zeta") is not None
                and o.params.get("zeta") is not None):
            aceitas_adim += 1
            if not r["calibration"]["ok"]:
                sem_calib_adim += 1
            esc = max(abs(alvo["zeta"]), 1e-12)
            err_real_adim.append(abs(z_adim - alvo["zeta"]) / esc * 100.0)
            err_orac_adim.append(abs(o.params["zeta"] - alvo["zeta"]) / esc * 100.0)

        wnT_adim = (r.get("dimensionless") or {}).get("wn_T")
        twin = m.get("t_window")
        if (o.success and o.order == m["order"] and r["order"] == m["order"]
                and wnT_adim is not None and alvo.get("wn") is not None
                and o.params.get("wn") is not None and twin):
            t_true = float(twin[1] - twin[0])
            if t_true > 0:
                aceitas_wnT_adim += 1
                wnT_alvo = alvo["wn"] * t_true
                esc_wn = max(abs(wnT_alvo), 1e-12)
                err_real_wnT_adim.append(abs(wnT_adim - wnT_alvo) / esc_wn * 100.0)
                err_orac_wnT_adim.append(
                    abs(o.params["wn"] * t_true - wnT_alvo) / esc_wn * 100.0)
                if not r["calibration"]["ok"]:
                    err_real_wnT_sc.append(err_real_wnT_adim[-1])
                    err_orac_wnT_sc.append(err_orac_wnT_adim[-1])

        thT_adim = (r.get("dimensionless") or {}).get("theta_T")
        if (o.success and o.order == m["order"] and r["order"] == m["order"]
                and thT_adim is not None and alvo.get("theta") is not None
                and o.params.get("theta") is not None and twin):
            t_true = float(twin[1] - twin[0])
            if t_true > 0:
                aceitas_thT_adim += 1
                thT_alvo = float(alvo["theta"]) / t_true
                err_real_thT_adim.append(abs(thT_adim - thT_alvo) * 100.0)
                err_orac_thT_adim.append(
                    abs(o.params["theta"] / t_true - thT_alvo) * 100.0)
                if not r["calibration"]["ok"]:
                    err_real_thT_sc.append(err_real_thT_adim[-1])
                    err_orac_thT_sc.append(err_orac_thT_adim[-1])

        Kyr_adim = (r.get("dimensionless") or {}).get("K_yrange")
        faixa_true = float(np.ptp(np.asarray(m["series"]["y"], dtype=float)))
        if (o.success and o.order == m["order"] and r["order"] == m["order"]
                and Kyr_adim is not None and alvo.get("K") is not None
                and o.params.get("K") is not None and faixa_true > 0):
            aceitas_Kyr_adim += 1
            Kyr_alvo = float(alvo["K"]) / faixa_true
            esc_K = max(abs(Kyr_alvo), 1e-12)
            err_real_Kyr_adim.append(abs(Kyr_adim - Kyr_alvo) / esc_K * 100.0)
            err_orac_Kyr_adim.append(
                abs(o.params["K"] / faixa_true - Kyr_alvo) / esc_K * 100.0)
            if not r["calibration"]["ok"]:
                err_real_Kyr_sc.append(err_real_Kyr_adim[-1])
                err_orac_Kyr_sc.append(err_orac_Kyr_adim[-1])

        # Ordem: sobre TODAS as amostras, sem filtro nenhum — é justamente o
        # filtro dos blocos acima que se quer medir aqui.
        if r["order"] == m["order"]:
            ordem_ok += 1
        if not r["calibration"]["ok"]:
            ordem_n_sc += 1
            if r["order"] == m["order"]:
                ordem_ok_sc += 1

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
    # O `n` insuficiente do caminho FÍSICO suspende só o veredito do 2.6 — NÃO
    # pode encerrar a função. Até a Task 6 do bloco do caso real havia aqui um
    # `return`, e o acoplamento era invertido: os blocos adimensionais abaixo
    # existem justamente para medir o que NÃO depende da calibração, e sumiam
    # do relatório exatamente quando a calibração piorava — sem sequer registrar
    # "n insuficiente". Era o ponto cego do Ruling 67 reabrindo em silêncio.
    piores: list[tuple[str, float]] = []
    pior: float | None = None
    if aceitas < 100:
        record_p2("2.6", "Degradação end-to-end (pior parâmetro)",
                  "≤ 3 p.p. (n insuficiente)", f"n={aceitas} < 100 — não asseverável", None)
    else:
        for k in sorted(set(err_oraculo) & set(err_real)):
            d = float(np.median(err_real[k])) - float(np.median(err_oraculo[k]))
            piores.append((k, d))
            record_p2(f"2.6[{k}]", f"ΔMAPE — {k}", "≤ 3 p.p.",
                      f"{d:+.2f} p.p. (oráculo {np.median(err_oraculo[k]):.2f}%, "
                      f"real {np.median(err_real[k]):.2f}%)", d <= 3.0)
        pior = max(d for _, d in piores)
        record_p2("2.6", "Degradação end-to-end (pior parâmetro)", "≤ 3 p.p.",
                  f"{pior:+.2f} p.p. (n={aceitas})", pior <= 3.0)

    # --- nível adimensional: ζ sem depender de calibração (Decisão E) ---------
    d_adim = None
    # A CONTAGEM vai SEMPRE, fora do portão de `n`: esta é a única linha do bloco
    # adimensional que carrega o `n` e o tamanho do subconjunto sem calibração, e
    # dentro do `if` ela sumiria em silêncio com `n` baixo — o mesmo modo de falha
    # que o `return` do 2.6 causava um nível acima (§35.7-2 do HANDOFF_P2_7).
    # Um diagnóstico que desaparece quando a população encolhe é pior que
    # nenhum: quem compara duas rodadas não vê que a base mudou.
    record_p2("2.6-adim-aceitas",
              "Amostras comparáveis no nível adimensional (dispensa calibração)",
              "diagnóstico",
              f"{aceitas_adim}/{len(test_samples)} "
              f"({sem_calib_adim} sem calibração)", None)
    if aceitas_adim >= 100:
        d_adim = float(np.median(err_real_adim)) - float(np.median(err_orac_adim))
        record_p2("2.6-adim[zeta]", "ΔMAPE adimensional — zeta", "≤ 3 p.p.",
                  f"{d_adim:+.2f} p.p. (oráculo {np.median(err_orac_adim):.2f}%, "
                  f"real {np.median(err_real_adim):.2f}%)", d_adim <= 3.0)
    else:
        record_p2("2.6-adim[zeta]", "ΔMAPE adimensional — zeta",
                  "≤ 3 p.p. (n insuficiente)",
                  f"n={aceitas_adim} < 100 — não asseverável", None)

    # ωₙ adimensional (ω_n·T) — diagnóstico, SEM meta de aprovação (fix round
    # 1, B3): o ponto cego que deixou a regressão de escala de `t` do B2
    # passar era não medir ωₙ aqui, só ζ (invariante à escala). Ainda não há
    # limiar decidido para este número — só torná-lo visível.
    if aceitas_wnT_adim >= 100:
        d_wnT_adim = float(np.median(err_real_wnT_adim)) - float(np.median(err_orac_wnT_adim))
        record_p2("2.6-adim[wn_T]", "ΔMAPE adimensional — ωₙ·T (diagnóstico)",
                  "diagnóstico",
                  f"{d_wnT_adim:+.2f} p.p. (oráculo {np.median(err_orac_wnT_adim):.2f}%, "
                  f"real {np.median(err_real_wnT_adim):.2f}%, n={aceitas_wnT_adim})", None)
    else:
        record_p2("2.6-adim[wn_T]", "ΔMAPE adimensional — ωₙ·T (diagnóstico)",
                  "diagnóstico (n insuficiente)",
                  f"n={aceitas_wnT_adim} < 100 — não asseverável", None)

    # A MESMA grandeza restrita às amostras SEM CALIBRAÇÃO (fix round 2, C4).
    # Esta é a linha sensível: é o único subconjunto em que `_serie_normalizada`
    # roda, e por isso a única em que uma regressão de escala de `t` aparece com
    # tamanho suficiente para alguém notar numa tabela de ~80 linhas. Sem
    # exigência de n ≥ 100 (o subconjunto é pequeno por construção — a
    # calibração falha em ~20 % das amostras): o `n` vai escrito na medição, e
    # quem for comparar duas rodadas compara também o `n`.
    if err_real_wnT_sc:
        d_wnT_sc = float(np.median(err_real_wnT_sc)) - float(np.median(err_orac_wnT_sc))
        record_p2("2.6-adim[wn_T/sem-calib]",
                  "ΔMAPE adimensional — ωₙ·T, só sem calibração (diagnóstico)",
                  "diagnóstico",
                  f"{d_wnT_sc:+.2f} p.p. (oráculo {np.median(err_orac_wnT_sc):.2f}%, "
                  f"real {np.median(err_real_wnT_sc):.2f}%, n={len(err_real_wnT_sc)})", None)
    else:
        record_p2("2.6-adim[wn_T/sem-calib]",
                  "ΔMAPE adimensional — ωₙ·T, só sem calibração (diagnóstico)",
                  "diagnóstico (sem amostras)", "n=0", None)

    # θ adimensional (θ/T) — diagnóstico, SEM meta de aprovação (revisão final,
    # §6). Fecha a ORIGEM de `t`, o eixo irmão da escala que o par `wn_T` acima
    # fechou. Erro em pontos percentuais DA JANELA (NMAE/T), convenção da Parte
    # 1 para θ; o "Δ" é a diferença de medianas real − oráculo, como nos demais.
    if aceitas_thT_adim >= 100:
        d_thT_adim = (float(np.median(err_real_thT_adim))
                      - float(np.median(err_orac_thT_adim)))
        record_p2("2.6-adim[theta_T]", "Δ(NMAE/T) adimensional — θ/T (diagnóstico)",
                  "diagnóstico",
                  f"{d_thT_adim:+.2f} p.p. (oráculo {np.median(err_orac_thT_adim):.2f}%, "
                  f"real {np.median(err_real_thT_adim):.2f}%, n={aceitas_thT_adim})", None)
    else:
        record_p2("2.6-adim[theta_T]", "Δ(NMAE/T) adimensional — θ/T (diagnóstico)",
                  "diagnóstico (n insuficiente)",
                  f"n={aceitas_thT_adim} < 100 — não asseverável", None)

    # A MESMA grandeza restrita às amostras SEM CALIBRAÇÃO — a linha sensível,
    # pelo mesmo motivo do `wn_T/sem-calib`: é o único subconjunto em que
    # `_serie_normalizada` roda, e por isso o único em que uma regressão de
    # ORIGEM de `t` aparece sem diluição. Sem exigência de n ≥ 100 (o
    # subconjunto é pequeno por construção): o `n` vai escrito na medição.
    if err_real_thT_sc:
        d_thT_sc = (float(np.median(err_real_thT_sc))
                    - float(np.median(err_orac_thT_sc)))
        record_p2("2.6-adim[theta_T/sem-calib]",
                  "Δ(NMAE/T) adimensional — θ/T, só sem calibração (diagnóstico)",
                  "diagnóstico",
                  f"{d_thT_sc:+.2f} p.p. (oráculo {np.median(err_orac_thT_sc):.2f}%, "
                  f"real {np.median(err_real_thT_sc):.2f}%, n={len(err_real_thT_sc)})", None)
    else:
        record_p2("2.6-adim[theta_T/sem-calib]",
                  "Δ(NMAE/T) adimensional — θ/T, só sem calibração (diagnóstico)",
                  "diagnóstico (sem amostras)", "n=0", None)

    # K adimensional (K/faixa de y) — diagnóstico, SEM meta (re-review, R-5).
    # Fecha a escala de **y**, o terceiro eixo depois da escala e da origem de
    # `t`. MAPE relativo: `K_yrange` de um degrau vale ~1/(1+overshoot) e não
    # passa por zero, então a razão relativa é bem definida aqui (ao contrário
    # de θ, que pode ser 0 e por isso usa NMAE).
    if aceitas_Kyr_adim >= 100:
        d_Kyr_adim = (float(np.median(err_real_Kyr_adim))
                      - float(np.median(err_orac_Kyr_adim)))
        record_p2("2.6-adim[K_yrange]", "ΔMAPE adimensional — K/faixa de y (diagnóstico)",
                  "diagnóstico",
                  f"{d_Kyr_adim:+.2f} p.p. (oráculo {np.median(err_orac_Kyr_adim):.2f}%, "
                  f"real {np.median(err_real_Kyr_adim):.2f}%, n={aceitas_Kyr_adim})", None)
    else:
        record_p2("2.6-adim[K_yrange]", "ΔMAPE adimensional — K/faixa de y (diagnóstico)",
                  "diagnóstico (n insuficiente)",
                  f"n={aceitas_Kyr_adim} < 100 — não asseverável", None)

    # A MESMA grandeza restrita às amostras SEM CALIBRAÇÃO, pelo mesmo motivo
    # dos irmãos: é o único subconjunto em que `_serie_normalizada` roda, e por
    # isso o único em que uma regressão de escala de `y` aparece sem diluição.
    if err_real_Kyr_sc:
        d_Kyr_sc = (float(np.median(err_real_Kyr_sc))
                    - float(np.median(err_orac_Kyr_sc)))
        record_p2("2.6-adim[K_yrange/sem-calib]",
                  "ΔMAPE adimensional — K/faixa de y, só sem calibração (diagnóstico)",
                  "diagnóstico",
                  f"{d_Kyr_sc:+.2f} p.p. (oráculo {np.median(err_orac_Kyr_sc):.2f}%, "
                  f"real {np.median(err_real_Kyr_sc):.2f}%, n={len(err_real_Kyr_sc)})", None)
    else:
        record_p2("2.6-adim[K_yrange/sem-calib]",
                  "ΔMAPE adimensional — K/faixa de y, só sem calibração (diagnóstico)",
                  "diagnóstico (sem amostras)", "n=0", None)

    # --- acerto de ORDEM (revisão final, §6) ----------------------------------
    # Diagnóstico, sem meta: não há limiar decidido, e o pedido é o mesmo dos
    # blocos acima — tornar visível um número que hoje só existe como filtro.
    # Sem portão de `n`: a população é o corpus inteiro e vai escrita.
    n_total = len(test_samples)
    if n_total:
        record_p2("2.12-ordem", "Acerto de ordem (diagnóstico)", "diagnóstico",
                  f"{100.0 * ordem_ok / n_total:.1f}% "
                  f"({ordem_ok}/{n_total}, n={n_total})", None)
    else:
        record_p2("2.12-ordem", "Acerto de ordem (diagnóstico)",
                  "diagnóstico (sem amostras)", "n=0", None)
    if ordem_n_sc:
        record_p2("2.12-ordem[sem-calib]",
                  "Acerto de ordem, só sem calibração (diagnóstico)", "diagnóstico",
                  f"{100.0 * ordem_ok_sc / ordem_n_sc:.1f}% "
                  f"({ordem_ok_sc}/{ordem_n_sc}, n={ordem_n_sc})", None)
    else:
        record_p2("2.12-ordem[sem-calib]",
                  "Acerto de ordem, só sem calibração (diagnóstico)",
                  "diagnóstico (sem amostras)", "n=0", None)

    if pior is not None:
        assert pior <= 3.0, f"pior degradação: {piores}"
    if d_adim is not None:
        assert d_adim <= 3.0, f"degradação adimensional de zeta: {d_adim:+.2f} p.p."


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
