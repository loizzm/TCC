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
| G3b.4 | Latência do extrator clássico | < 200 ms | mediana 13.3 ms, p95 33.9 ms | ✅ |
| A.0 | Parâmetros da U-Net | ~1,2 M (PLANO) | 1.94 M | ❓ |
| 2.3 | Erro relativo de sx, sy | < 1% em ≥ 95% | 0.958 (n=239) | ✅ |
| 2.4 | Taxa de rejeição (diagnóstico — unificado no 2.9) | sem alvo próprio: é 1 − cobertura do 2.9 (Ruling 52) | 0.203 | ❓ |
| 2.5 | Rejeições corretas | ≥ 90% (erro > 1%, alinhado ao 2.3) | 0.885 (n=61) | ❌ |
| 2.9 | Cobertura da calibração (ok=True) | ≥ 90% global | 0.797 (n=300) | ❌ |
| 2.9[dpi=100-149] | Cobertura — dpi 100-149 | diagnóstico, sem alvo por estrato | 0.826 (n=115) | ❓ |
| 2.9[dpi=150-200] | Cobertura — dpi 150-200 | diagnóstico, sem alvo por estrato | 0.770 (n=100) | ❓ |
| 2.9[dpi=60-99] | Cobertura — dpi 60-99 | diagnóstico, sem alvo por estrato | 0.788 (n=85) | ❓ |
| 2.11 | Bloco `dimensionless` presente em toda amostra | 100% das amostras, sem exceção | 300/300 com bloco; 300/300 com valor; 61/61 das sem calibração | ✅ |
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
| 2.1 | Erro perpendicular da máscara (U-Net) | ≤ 1.0 px mediana, ≤ 2.0 px p95 | 0.802 px / 1.529 px (n=300) | ✅ |
| 2.1-iou | IoU da máscara (diagnóstico — ver Ruling 50) | sem alvo: mede espessura de traço, não acurácia | 0.6478 | ❓ |
| 2.7[fundo_escuro=False] | Erro perpendicular — fundo_escuro=False | ≤ 1.0 px mediana | 0.792 px (n=173) | ✅ |
| 2.7-iou[fundo_escuro=False] | IoU — fundo_escuro=False (diagnóstico) | sem alvo (Ruling 50) | 0.6330 (n=173) | ❓ |
| 2.7[fundo_escuro=True] | Erro perpendicular — fundo_escuro=True | ≤ 1.0 px mediana | 0.812 px (n=127) | ✅ |
| 2.7-iou[fundo_escuro=True] | IoU — fundo_escuro=True (diagnóstico) | sem alvo (Ruling 50) | 0.6741 (n=127) | ❓ |
| 2.7[grade=False] | Erro perpendicular — grade=False | ≤ 1.0 px mediana | 0.792 px (n=150) | ✅ |
| 2.7-iou[grade=False] | IoU — grade=False (diagnóstico) | sem alvo (Ruling 50) | 0.6449 (n=150) | ❓ |
| 2.7[grade=True] | Erro perpendicular — grade=True | ≤ 1.0 px mediana | 0.806 px (n=150) | ✅ |
| 2.7-iou[grade=True] | IoU — grade=True (diagnóstico) | sem alvo (Ruling 50) | 0.6583 (n=150) | ❓ |
| 2.7[legenda=False] | Erro perpendicular — legenda=False | ≤ 1.0 px mediana | 0.794 px (n=155) | ✅ |
| 2.7-iou[legenda=False] | IoU — legenda=False (diagnóstico) | sem alvo (Ruling 50) | 0.6748 (n=155) | ❓ |
| 2.7[legenda=True] | Erro perpendicular — legenda=True | ≤ 1.0 px mediana | 0.807 px (n=145) | ✅ |
| 2.7-iou[legenda=True] | IoU — legenda=True (diagnóstico) | sem alvo (Ruling 50) | 0.6164 (n=145) | ❓ |
| 2.7[traco=-] | Erro perpendicular — traco=- | ≤ 1.0 px mediana | 0.663 px (n=94) | ✅ |
| 2.7-iou[traco=-] | IoU — traco=- (diagnóstico) | sem alvo (Ruling 50) | 0.7099 (n=94) | ❓ |
| 2.7[traco=--] | Erro perpendicular — traco=-- | ≤ 1.0 px mediana | 0.751 px (n=72) | ✅ |
| 2.7-iou[traco=--] | IoU — traco=-- (diagnóstico) | sem alvo (Ruling 50) | 0.6753 (n=72) | ❓ |
| 2.7[traco=-.] | Erro perpendicular — traco=-. | ≤ 1.0 px mediana | 0.808 px (n=67) | ✅ |
| 2.7-iou[traco=-.] | IoU — traco=-. (diagnóstico) | sem alvo (Ruling 50) | 0.6435 (n=67) | ❓ |
| 2.7[traco=:] | Erro perpendicular — traco=: | ≤ 1.0 px mediana | 0.956 px (n=67) | ✅ |
| 2.7-iou[traco=:] | IoU — traco=: (diagnóstico) | sem alvo (Ruling 50) | 0.5317 (n=67) | ❓ |
| 2.10 | IoU mediana: U-Net vs. extrator clássico | sem alvo | U-Net=0.6478  clássico=0.7153 | ❓ |
| 2.8 | Latência por imagem | < 500 ms | mediana 184 ms, p95 270 ms | ✅ |
| 2.6-aceitas | Amostras comparáveis (mesma ordem, ambos convergem) | diagnóstico | 214/300 | ❓ |
| 2.6[K] | ΔMAPE — K | ≤ 3 p.p. | +0.19 p.p. (oráculo 0.10%, real 0.29%) | ✅ |
| 2.6[tau] | ΔMAPE — tau | ≤ 3 p.p. | +0.31 p.p. (oráculo 0.21%, real 0.51%) | ✅ |
| 2.6[theta] | ΔMAPE — theta | ≤ 3 p.p. | +0.27 p.p. (oráculo 0.06%, real 0.33%) | ✅ |
| 2.6[wn] | ΔMAPE — wn | ≤ 3 p.p. | +0.82 p.p. (oráculo 0.71%, real 1.54%) | ✅ |
| 2.6[zeta] | ΔMAPE — zeta | ≤ 3 p.p. | +0.99 p.p. (oráculo 1.40%, real 2.40%) | ✅ |
| 2.6 | Degradação end-to-end (pior parâmetro) | ≤ 3 p.p. | +0.99 p.p. (n=214) | ✅ |
| 2.6-adim-aceitas | Amostras comparáveis no nível adimensional (dispensa calibração) | diagnóstico | 143/300 (33 sem calibração) | ❓ |
| 2.6-adim[zeta] | ΔMAPE adimensional — zeta | ≤ 3 p.p. | +0.89 p.p. (oráculo 1.19%, real 2.08%) | ✅ |
| 2.6-adim[wn_T] | ΔMAPE adimensional — ωₙ·T (diagnóstico) | diagnóstico | +1.04 p.p. (oráculo 0.72%, real 1.76%, n=143) | ❓ |
| 2.6-adim[wn_T/sem-calib] | ΔMAPE adimensional — ωₙ·T, só sem calibração (diagnóstico) | diagnóstico | +0.63 p.p. (oráculo 0.72%, real 1.35%, n=33) | ❓ |
| 2.6-classico-aceitas | Amostras comparáveis (extrator clássico) | diagnóstico | 181/300 | ❓ |
| 2.6-classico[K] | ΔMAPE (clássico) — K | diagnóstico | +0.43 p.p. (oráculo 0.11%, real 0.54%) | ✅ |
| 2.6-classico[tau] | ΔMAPE (clássico) — tau | diagnóstico | +0.58 p.p. (oráculo 0.23%, real 0.82%) | ✅ |
| 2.6-classico[theta] | ΔMAPE (clássico) — theta | diagnóstico | +0.56 p.p. (oráculo 0.06%, real 0.61%) | ✅ |
| 2.6-classico[wn] | ΔMAPE (clássico) — wn | diagnóstico | +1.05 p.p. (oráculo 0.55%, real 1.61%) | ✅ |
| 2.6-classico[zeta] | ΔMAPE (clássico) — zeta | diagnóstico | +0.93 p.p. (oráculo 1.47%, real 2.40%) | ✅ |
| 2.6-classico | Degradação end-to-end (extrator clássico, pior parâmetro) | diagnóstico | +1.05 p.p. (n=181) | ✅ |

