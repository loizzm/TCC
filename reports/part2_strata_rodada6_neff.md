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
| G3b.4 | Latência do extrator clássico | < 200 ms | mediana 12.9 ms, p95 30.2 ms | ✅ |
| A.0 | Parâmetros da U-Net | ~1,2 M (PLANO) | 1.94 M | ❓ |
| 2.3 | Erro relativo de sx, sy | < 1% em ≥ 95% | 0.950 (n=241) | ✅ |
| 2.4 | Taxa de rejeição (falso alarme) | < 5% | 0.197 | ❌ |
| 2.5 | Rejeições corretas | ≥ 90% | 0.678 (n=59) | ❌ |
| 2.9 | Cobertura da calibração (ok=True) | ≥ 90% global | 0.803 (n=300) | ❌ |
| 2.9[dpi=100-149] | Cobertura — dpi 100-149 | diagnóstico, sem alvo por estrato | 0.878 (n=115) | ❓ |
| 2.9[dpi=150-200] | Cobertura — dpi 150-200 | diagnóstico, sem alvo por estrato | 0.770 (n=100) | ❓ |
| 2.9[dpi=60-99] | Cobertura — dpi 60-99 | diagnóstico, sem alvo por estrato | 0.741 (n=85) | ❓ |
| 2.11 | calibrate() nunca levanta exceção | 100% das amostras | 300/300 | ✅ |
| 2.2-piso | Polilinha vs. máscara VERDADEIRA | RMSE ≤ 2 px, p95 ≤ 5 px | RMSE=1.49 px, p95=6.70 px | ❌ |
| 2.2[espessura=fina] | RMSE da polilinha — espessura=fina | ≤ 2 px | 1.15 px (n=142) | ✅ |
| 2.2[espessura=grossa] | RMSE da polilinha — espessura=grossa | ≤ 2 px | 1.76 px (n=158) | ✅ |
| 2.2[marcador=False] | RMSE da polilinha — marcador=False | ≤ 2 px | 1.33 px (n=214) | ✅ |
| 2.2[marcador=True] | RMSE da polilinha — marcador=True | ≤ 2 px | 1.91 px (n=86) | ✅ |
| 2.2[traco=-] | RMSE da polilinha — traco=- | ≤ 2 px | 1.08 px (n=94) | ✅ |
| 2.2[traco=--] | RMSE da polilinha — traco=-- | ≤ 2 px | 1.52 px (n=72) | ✅ |
| 2.2[traco=-.] | RMSE da polilinha — traco=-. | ≤ 2 px | 1.61 px (n=67) | ✅ |
| 2.2[traco=:] | RMSE da polilinha — traco=: | ≤ 2 px | 1.80 px (n=67) | ✅ |
| 2.1 | IoU da máscara (U-Net) | ≥ 0,85 (mediana) | 0.6478 | ❌ |
| 2.7[fundo_escuro=False] | IoU — fundo_escuro=False | ≥ 0,75 | 0.6330 (n=173) | ❌ |
| 2.10 | IoU mediana: U-Net vs. extrator clássico | sem alvo | U-Net=0.6478  clássico=0.7153 | ❓ |
| 2.8 | Latência por imagem | < 500 ms | mediana 891 ms, p95 2325 ms | ❌ |
| 2.6-aceitas | Amostras comparáveis (mesma ordem, ambos convergem) | diagnóstico | 215/300 | ❓ |
| 2.6[K] | ΔMAPE — K | ≤ 3 p.p. | +0.19 p.p. (oráculo 0.09%, real 0.28%) | ✅ |
| 2.6[tau] | ΔMAPE — tau | ≤ 3 p.p. | +0.39 p.p. (oráculo 0.19%, real 0.58%) | ✅ |
| 2.6[theta] | ΔMAPE — theta | ≤ 3 p.p. | +0.27 p.p. (oráculo 0.06%, real 0.33%) | ✅ |
| 2.6[wn] | ΔMAPE — wn | ≤ 3 p.p. | +0.73 p.p. (oráculo 0.77%, real 1.50%) | ✅ |
| 2.6[zeta] | ΔMAPE — zeta | ≤ 3 p.p. | +0.93 p.p. (oráculo 1.17%, real 2.10%) | ✅ |
| 2.6 | Degradação end-to-end (pior parâmetro) | ≤ 3 p.p. | +0.93 p.p. (n=215) | ✅ |
| 2.6-classico-aceitas | Amostras comparáveis (extrator clássico) | diagnóstico | 179/300 | ❓ |
| 2.6-classico[K] | ΔMAPE (clássico) — K | diagnóstico | +0.65 p.p. (oráculo 0.10%, real 0.75%) | ✅ |
| 2.6-classico[tau] | ΔMAPE (clássico) — tau | diagnóstico | +1.14 p.p. (oráculo 0.22%, real 1.36%) | ✅ |
| 2.6-classico[theta] | ΔMAPE (clássico) — theta | diagnóstico | +0.56 p.p. (oráculo 0.06%, real 0.62%) | ✅ |
| 2.6-classico[wn] | ΔMAPE (clássico) — wn | diagnóstico | +1.33 p.p. (oráculo 0.73%, real 2.06%) | ✅ |
| 2.6-classico[zeta] | ΔMAPE (clássico) — zeta | diagnóstico | +2.42 p.p. (oráculo 1.31%, real 3.73%) | ✅ |
| 2.6-classico | Degradação end-to-end (extrator clássico, pior parâmetro) | diagnóstico | +2.42 p.p. (n=179) | ✅ |

