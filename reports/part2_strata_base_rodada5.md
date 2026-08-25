# Parte 2 — Estratificação e critérios

Relatório gerado automaticamente por `tests/part2/` (`pytest_sessionfinish` em `tests/part2/conftest.py`). Não editar à mão.

| Critério | Nome | Alvo | Medido | Veredito |
|---|---|---|---|---|
| 2.1 | IoU da máscara (U-Net) | ≥ 0,85 (mediana) | 0.6205 | ❌ |
| 2.7[fundo_escuro=False] | IoU — fundo_escuro=False | ≥ 0,75 | 0.5856 (n=173) | ❌ |
| 2.10 | IoU mediana: U-Net vs. extrator clássico | sem alvo | U-Net=0.6205  clássico=0.7153 | ❓ |
| 2.6-aceitas | Amostras comparáveis (mesma ordem, ambos convergem) | diagnóstico | 173/300 | ❓ |
| 2.6[K] | ΔMAPE — K | ≤ 3 p.p. | +1.01 p.p. (oráculo 0.11%, real 1.12%) | ✅ |
| 2.6[tau] | ΔMAPE — tau | ≤ 3 p.p. | +2.44 p.p. (oráculo 0.24%, real 2.68%) | ✅ |
| 2.6[theta] | ΔMAPE — theta | ≤ 3 p.p. | +0.50 p.p. (oráculo 0.07%, real 0.58%) | ✅ |
| 2.6[wn] | ΔMAPE — wn | ≤ 3 p.p. | +2.03 p.p. (oráculo 0.93%, real 2.96%) | ✅ |
| 2.6[zeta] | ΔMAPE — zeta | ≤ 3 p.p. | +3.65 p.p. (oráculo 1.31%, real 4.97%) | ❌ |
| 2.6 | Degradação end-to-end (pior parâmetro) | ≤ 3 p.p. | +3.65 p.p. (n=173) | ❌ |
| 2.6-classico-aceitas | Amostras comparáveis (extrator clássico) | diagnóstico | 176/300 | ❓ |
| 2.6-classico[K] | ΔMAPE (clássico) — K | diagnóstico | +0.94 p.p. (oráculo 0.11%, real 1.05%) | ✅ |
| 2.6-classico[tau] | ΔMAPE (clássico) — tau | diagnóstico | +1.84 p.p. (oráculo 0.23%, real 2.07%) | ✅ |
| 2.6-classico[theta] | ΔMAPE (clássico) — theta | diagnóstico | +0.72 p.p. (oráculo 0.07%, real 0.79%) | ✅ |
| 2.6-classico[wn] | ΔMAPE (clássico) — wn | diagnóstico | +2.29 p.p. (oráculo 0.80%, real 3.09%) | ✅ |
| 2.6-classico[zeta] | ΔMAPE (clássico) — zeta | diagnóstico | +4.36 p.p. (oráculo 1.24%, real 5.60%) | ❌ |
| 2.6-classico | Degradação end-to-end (extrator clássico, pior parâmetro) | diagnóstico | +4.36 p.p. (n=176) | ❌ |

