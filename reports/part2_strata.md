# Parte 2 — Estratificação e critérios

Relatório gerado automaticamente por `tests/part2/` (`pytest_sessionfinish` em `tests/part2/conftest.py`). Não editar à mão.

| Critério | Nome | Alvo | Medido | Veredito |
|---|---|---|---|---|
| G1.1 | Erro da moldura | ≤ 2 px em ≥ 95% | 0.997 | ✅ |
| G1.2x | Recall de ticks (x) | ≥ 0,95 mediana | 1.000 | ✅ |
| G1.2y | Recall de ticks (y) | ≥ 0,95 mediana | 1.000 | ✅ |
| G1.3-2 | Moldura, n_spines=2 | ≥ 0,90 | 1.000 (n=85) | ✅ |
| G1.3-3 | Moldura, n_spines=3 | ≥ 0,90 | 1.000 (n=156) | ✅ |
| G1.3-4 | Moldura, n_spines=4 | ≥ 0,90 | 0.983 (n=59) | ✅ |
| G3b.3 | extract_classical não importa torch | import OK sem torch | OK | ✅ |
| G3b.1 | IoU mediana (extrator clássico) | ≥ 0,70 | 0.7153 | ✅ |
| G3b.1[fundo_escuro=False] | IoU clássico — fundo_escuro=False | diagnóstico | 0.6799 (n=173) | ❓ |
| G3b.1[fundo_escuro=True] | IoU clássico — fundo_escuro=True | diagnóstico | 0.7382 (n=127) | ❓ |
| G3b.1[grade=False] | IoU clássico — grade=False | diagnóstico | 0.6928 (n=150) | ❓ |
| G3b.1[grade=True] | IoU clássico — grade=True | diagnóstico | 0.7301 (n=150) | ❓ |
| G3b.1[n_distractors=1] | IoU clássico — n_distractors=1 | diagnóstico | 0.7313 (n=116) | ❓ |
| G3b.1[n_distractors=2] | IoU clássico — n_distractors=2 | diagnóstico | 0.7153 (n=92) | ❓ |
| G3b.1[n_distractors=3] | IoU clássico — n_distractors=3 | diagnóstico | 0.6727 (n=92) | ❓ |
| G3b.2 | Sem reta de span completo na máscara | 0 violações | 0 | ✅ |
| G3b.4 | Latência do extrator clássico | < 200 ms | mediana 14.2 ms, p95 34.8 ms | ✅ |
| A.0 | Parâmetros da U-Net | ~1,2 M (PLANO) | 1.94 M | ❓ |
| 2.3 | Erro relativo de sx, sy | < 1% em ≥ 95% | 0.986 (n=280) | ✅ |
| 2.4 | Taxa de rejeição (diagnóstico — unificado no 2.9) | sem alvo próprio: é 1 − cobertura do 2.9 (Ruling 52) | 0.067 | ❓ |
| 2.5 | Rejeições corretas | ≥ 90% (erro > 1%, alinhado ao 2.3) | 0.900 (n=20) | ✅ |
| 2.9 | Cobertura da calibração (ok=True) | ≥ 90% global | 0.933 (n=300) | ✅ |
| 2.9[dpi=100-149] | Cobertura — dpi 100-149 | diagnóstico, sem alvo por estrato | 0.983 (n=115) | ❓ |
| 2.9[dpi=150-200] | Cobertura — dpi 150-200 | diagnóstico, sem alvo por estrato | 0.970 (n=100) | ❓ |
| 2.9[dpi=60-99] | Cobertura — dpi 60-99 | diagnóstico, sem alvo por estrato | 0.824 (n=85) | ❓ |
| 2.11 | Bloco `dimensionless` presente em toda amostra | 100% das amostras, sem exceção | 300/300 com bloco; 292/300 com valor; 20/20 das sem calibração | ✅ |
| 2.2-piso-vertical | Polilinha vs. máscara VERDADEIRA, diferença VERTICAL (diagnóstico) | sem alvo: responde à declividade do render (Ruling 50) | RMSE=1.49 px, p95=6.70 px | ❓ |
| 2.2-piso | Polilinha vs. máscara VERDADEIRA (perpendicular) | RMSE ≤ 1.0 px, p95 ≤ 2.0 px | RMSE=0.615 px, p95=1.137 px (n=300) | ✅ |
| 2.2[espessura=fina] | RMSE da polilinha — espessura=fina | ≤ 2 px | 1.15 px (n=142) | ✅ |
| 2.2[espessura=grossa] | RMSE da polilinha — espessura=grossa | ≤ 2 px | 1.76 px (n=158) | ✅ |
| 2.2[marcador=False] | RMSE da polilinha — marcador=False | ≤ 2 px | 1.33 px (n=214) | ✅ |
| 2.2[marcador=True] | RMSE da polilinha — marcador=True | ≤ 2 px | 1.91 px (n=86) | ✅ |
| 2.2[traco=-] | RMSE da polilinha — traco=- | ≤ 2 px | 1.08 px (n=94) | ✅ |
| 2.2[traco=--] | RMSE da polilinha — traco=-- | ≤ 2 px | 1.52 px (n=72) | ✅ |
| 2.2[traco=-.] | RMSE da polilinha — traco=-. | ≤ 2 px | 1.61 px (n=67) | ✅ |
| 2.2[traco=:] | RMSE da polilinha — traco=: | ≤ 2 px | 1.81 px (n=67) | ✅ |
| 2.1 | Erro perpendicular da máscara (U-Net) | ≤ 1.0 px mediana, ≤ 2.0 px p95 | 0.799 px / 1.703 px (n=300) | ✅ |
| 2.1-iou | IoU da máscara (diagnóstico — ver Ruling 50) | sem alvo: mede espessura de traço, não acurácia | 0.6482 | ❓ |
| 2.7[fundo_escuro=False] | Erro perpendicular — fundo_escuro=False | ≤ 1.0 px mediana | 0.786 px (n=173) | ✅ |
| 2.7-iou[fundo_escuro=False] | IoU — fundo_escuro=False (diagnóstico) | sem alvo (Ruling 50) | 0.6244 (n=173) | ❓ |
| 2.7[fundo_escuro=True] | Erro perpendicular — fundo_escuro=True | ≤ 1.0 px mediana | 0.810 px (n=127) | ✅ |
| 2.7-iou[fundo_escuro=True] | IoU — fundo_escuro=True (diagnóstico) | sem alvo (Ruling 50) | 0.6738 (n=127) | ❓ |
| 2.7[grade=False] | Erro perpendicular — grade=False | ≤ 1.0 px mediana | 0.792 px (n=150) | ✅ |
| 2.7-iou[grade=False] | IoU — grade=False (diagnóstico) | sem alvo (Ruling 50) | 0.6423 (n=150) | ❓ |
| 2.7[grade=True] | Erro perpendicular — grade=True | ≤ 1.0 px mediana | 0.801 px (n=150) | ✅ |
| 2.7-iou[grade=True] | IoU — grade=True (diagnóstico) | sem alvo (Ruling 50) | 0.6579 (n=150) | ❓ |
| 2.7[legenda=False] | Erro perpendicular — legenda=False | ≤ 1.0 px mediana | 0.781 px (n=155) | ✅ |
| 2.7-iou[legenda=False] | IoU — legenda=False (diagnóstico) | sem alvo (Ruling 50) | 0.6774 (n=155) | ❓ |
| 2.7[legenda=True] | Erro perpendicular — legenda=True | ≤ 1.0 px mediana | 0.810 px (n=145) | ✅ |
| 2.7-iou[legenda=True] | IoU — legenda=True (diagnóstico) | sem alvo (Ruling 50) | 0.6147 (n=145) | ❓ |
| 2.7[traco=-] | Erro perpendicular — traco=- | ≤ 1.0 px mediana | 0.684 px (n=94) | ✅ |
| 2.7-iou[traco=-] | IoU — traco=- (diagnóstico) | sem alvo (Ruling 50) | 0.7064 (n=94) | ❓ |
| 2.7[traco=--] | Erro perpendicular — traco=-- | ≤ 1.0 px mediana | 0.756 px (n=72) | ✅ |
| 2.7-iou[traco=--] | IoU — traco=-- (diagnóstico) | sem alvo (Ruling 50) | 0.6727 (n=72) | ❓ |
| 2.7[traco=-.] | Erro perpendicular — traco=-. | ≤ 1.0 px mediana | 0.827 px (n=67) | ✅ |
| 2.7-iou[traco=-.] | IoU — traco=-. (diagnóstico) | sem alvo (Ruling 50) | 0.6158 (n=67) | ❓ |
| 2.7[traco=:] | Erro perpendicular — traco=: | ≤ 1.0 px mediana | 0.981 px (n=67) | ✅ |
| 2.7-iou[traco=:] | IoU — traco=: (diagnóstico) | sem alvo (Ruling 50) | 0.5264 (n=67) | ❓ |
| 2.10 | IoU mediana: U-Net vs. extrator clássico | sem alvo | U-Net=0.6482  clássico=0.7153 | ❓ |
| 2.8 | Latência por imagem | < 500 ms | mediana 210 ms, p95 349 ms | ✅ |
| 2.6-aceitas | Amostras comparáveis (mesma ordem, ambos convergem) | diagnóstico | 254/300 | ❓ |
| 2.6[K] | ΔMAPE — K | ≤ 3 p.p. | +0.16 p.p. (oráculo 0.12%, real 0.28%) | ✅ |
| 2.6[tau] | ΔMAPE — tau | ≤ 3 p.p. | +0.21 p.p. (oráculo 0.23%, real 0.44%) | ✅ |
| 2.6[theta] | ΔMAPE — theta | ≤ 3 p.p. | +0.21 p.p. (oráculo 0.06%, real 0.27%) | ✅ |
| 2.6[wn] | ΔMAPE — wn | ≤ 3 p.p. | +1.00 p.p. (oráculo 0.65%, real 1.66%) | ✅ |
| 2.6[zeta] | ΔMAPE — zeta | ≤ 3 p.p. | +1.63 p.p. (oráculo 1.09%, real 2.72%) | ✅ |
| 2.6 | Degradação end-to-end (pior parâmetro) | ≤ 3 p.p. | +1.63 p.p. (n=254) | ✅ |
| 2.6-adim-aceitas | Amostras comparáveis no nível adimensional (dispensa calibração) | diagnóstico | 133/300 (11 sem calibração) | ❓ |
| 2.6-adim[zeta] | ΔMAPE adimensional — zeta | ≤ 3 p.p. | +1.59 p.p. (oráculo 1.09%, real 2.68%) | ✅ |
| 2.6-adim[wn_T] | ΔMAPE adimensional — ωₙ·T (diagnóstico) | diagnóstico | +1.41 p.p. (oráculo 0.64%, real 2.05%, n=133) | ❓ |
| 2.6-adim[wn_T/sem-calib] | ΔMAPE adimensional — ωₙ·T, só sem calibração (diagnóstico) | diagnóstico | +6.30 p.p. (oráculo 0.30%, real 6.60%, n=11) | ❓ |
| 2.6-adim[theta_T] | Δ(NMAE/T) adimensional — θ/T (diagnóstico) | diagnóstico | +0.11 p.p. (oráculo 0.03%, real 0.14%, n=273) | ❓ |
| 2.6-adim[theta_T/sem-calib] | Δ(NMAE/T) adimensional — θ/T, só sem calibração (diagnóstico) | diagnóstico | +2.18 p.p. (oráculo 0.02%, real 2.20%, n=19) | ❓ |
| 2.6-adim[K_yrange] | ΔMAPE adimensional — K/faixa de y (diagnóstico) | diagnóstico | +0.64 p.p. (oráculo 0.11%, real 0.76%, n=273) | ❓ |
| 2.6-adim[K_yrange/sem-calib] | ΔMAPE adimensional — K/faixa de y, só sem calibração (diagnóstico) | diagnóstico | +0.87 p.p. (oráculo 0.08%, real 0.95%, n=19) | ❓ |
| 2.12-ordem | Acerto de ordem (diagnóstico) | diagnóstico | 92.3% (277/300, n=300) | ❓ |
| 2.12-ordem[sem-calib] | Acerto de ordem, só sem calibração (diagnóstico) | diagnóstico | 95.0% (19/20, n=20) | ❓ |
| 2.6-classico-aceitas | Amostras comparáveis (extrator clássico) | diagnóstico | 195/300 | ❓ |
| 2.6-classico[K] | ΔMAPE (clássico) — K | diagnóstico | +0.30 p.p. (oráculo 0.16%, real 0.46%) | ✅ |
| 2.6-classico[tau] | ΔMAPE (clássico) — tau | diagnóstico | +0.56 p.p. (oráculo 0.26%, real 0.82%) | ✅ |
| 2.6-classico[theta] | ΔMAPE (clássico) — theta | diagnóstico | +0.46 p.p. (oráculo 0.06%, real 0.52%) | ✅ |
| 2.6-classico[wn] | ΔMAPE (clássico) — wn | diagnóstico | +0.98 p.p. (oráculo 0.65%, real 1.63%) | ✅ |
| 2.6-classico[zeta] | ΔMAPE (clássico) — zeta | diagnóstico | +1.05 p.p. (oráculo 1.47%, real 2.52%) | ✅ |
| 2.6-classico | Degradação end-to-end (extrator clássico, pior parâmetro) | diagnóstico | +1.05 p.p. (n=195) | ✅ |

