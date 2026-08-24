# Parte 2 — Estratificação e critérios

Relatório gerado automaticamente por `tests/part2/` (`pytest_sessionfinish` em `tests/part2/conftest.py`). Não editar à mão.

| Critério | Nome | Alvo | Medido | Veredito |
|---|---|---|---|---|
| 2.1 | IoU da máscara (U-Net) | ≥ 0,85 (mediana) | 0.6205 | ❌ |
| 2.7[fundo_escuro=False] | IoU — fundo_escuro=False | ≥ 0,75 | 0.5856 (n=173) | ❌ |
| 2.10 | IoU mediana: U-Net vs. extrator clássico | sem alvo | U-Net=0.6205  clássico=0.7153 | ❓ |
| 2.8 | Latência por imagem | < 500 ms | mediana 3353 ms, p95 9981 ms | ❌ |
| 2.6-aceitas | Amostras comparáveis (mesma ordem, ambos convergem) | diagnóstico | 168/300 | ❓ |
| 2.6[K] | ΔMAPE — K | ≤ 3 p.p. | +1.08 p.p. (oráculo 0.11%, real 1.20%) | ✅ |
| 2.6[tau] | ΔMAPE — tau | ≤ 3 p.p. | +2.57 p.p. (oráculo 0.25%, real 2.82%) | ✅ |
| 2.6[theta] | ΔMAPE — theta | ≤ 3 p.p. | +0.54 p.p. (oráculo 0.08%, real 0.62%) | ✅ |
| 2.6[wn] | ΔMAPE — wn | ≤ 3 p.p. | +2.11 p.p. (oráculo 0.90%, real 3.01%) | ✅ |
| 2.6[zeta] | ΔMAPE — zeta | ≤ 3 p.p. | +3.73 p.p. (oráculo 1.24%, real 4.97%) | ❌ |
| 2.6 | Degradação end-to-end (pior parâmetro) | ≤ 3 p.p. | +3.73 p.p. (n=168) | ❌ |

