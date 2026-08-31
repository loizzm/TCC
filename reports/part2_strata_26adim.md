# Parte 2 — Estratificação e critérios

> ## ⚠ INSTANTÂNEO HISTÓRICO — ESTADO ANTIGO DOS CRITÉRIOS. NÃO É REFERÊNCIA CORRENTE.
>
> **De quando é:** commit `7964230` (2026-08-26), a rodada em que o ζ adimensional
> entrou no critério 2.6 (`HANDOFF_P2_7.md` §25, Ruling 47). Guardado como registro
> daquele momento.
>
> **Estado dos critérios AQUI dentro:** é **anterior à revisão de critérios do
> Ruling 53** (`HANDOFF_P2_7.md` §32, `PLANO.md` §2.12). Nesta tabela o `2.1` ainda
> é **IoU com alvo ≥ 0,85** e reprova; o `2.2-piso` ainda é **diferença VERTICAL**
> com alvo ≤ 2/5 px e reprova; o `2.4` ainda tem alvo próprio; o `2.5` está em
> **0,721** contra um limiar de rejeição de 5 %, não de 1 %.
>
> **O que mudou depois:** o Ruling 53 trocou IoU e diferença vertical por **erro
> perpendicular** (classe A do §2.12 — as métricas antigas mediam espessura de traço
> e declividade do render, não acurácia), unificou o `2.4` no `2.9` (classe B) e
> alinhou `2.3`/`2.5` no mesmo τ = 1 % (classe C). IoU e vertical continuam na tabela
> corrente, mas como **diagnóstico sem alvo** (`2.1-iou`, `2.2-piso-vertical`).
>
> **Referência corrente:** `reports/part2_strata.md` (regenerado a cada rodada da
> suíte) e, para o antes/depois do bloco do caso real,
> `reports/part2_strata_pos_caso_real.md`. Para isolar o efeito de UM bloco, a
> comparação certa é contra `git show HEAD:reports/part2_strata.md`.
>
> **Por que este aviso existe:** o plano do bloco do caso real mandava diffar a
> rodada nova contra ESTE arquivo, e a expectativa que ele declarava ("mudanças
> apenas em `2.6-adim[zeta]`") era inalcançável — não por regressão, mas porque a
> referência é de duas revisões de critério atrás e o diff devolve ~60 linhas da
> revisão, não do bloco. Registrado em `HANDOFF_P2_7.md` §35.5.

Relatório gerado automaticamente por `tests/part2/` (`pytest_sessionfinish` em `tests/part2/conftest.py`). Não editar à mão.

*(O aviso acima foi acrescentado à mão, deliberadamente: este arquivo saiu do
gerador e depois foi congelado como registro histórico — não é regenerado.)*

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
| G3b.4 | Latência do extrator clássico | < 200 ms | mediana 12.4 ms, p95 31.7 ms | ✅ |
| A.0 | Parâmetros da U-Net | ~1,2 M (PLANO) | 1.94 M | ❓ |
| 2.3 | Erro relativo de sx, sy | < 1% em ≥ 95% | 0.958 (n=239) | ✅ |
| 2.4 | Taxa de rejeição (falso alarme) | < 5% | 0.203 | ❌ |
| 2.5 | Rejeições corretas | ≥ 90% | 0.721 (n=61) | ❌ |
| 2.9 | Cobertura da calibração (ok=True) | ≥ 90% global | 0.797 (n=300) | ❌ |
| 2.9[dpi=100-149] | Cobertura — dpi 100-149 | diagnóstico, sem alvo por estrato | 0.826 (n=115) | ❓ |
| 2.9[dpi=150-200] | Cobertura — dpi 150-200 | diagnóstico, sem alvo por estrato | 0.770 (n=100) | ❓ |
| 2.9[dpi=60-99] | Cobertura — dpi 60-99 | diagnóstico, sem alvo por estrato | 0.788 (n=85) | ❓ |
| 2.11 | Bloco `dimensionless` presente em toda amostra | 100% das amostras, sem exceção | 300/300 com bloco; 300/300 com valor; 61/61 das sem calibração | ✅ |
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
| 2.8 | Latência por imagem | < 500 ms | mediana 183 ms, p95 269 ms | ✅ |
| 2.6-aceitas | Amostras comparáveis (mesma ordem, ambos convergem) | diagnóstico | 214/300 | ❓ |
| 2.6[K] | ΔMAPE — K | ≤ 3 p.p. | +0.20 p.p. (oráculo 0.10%, real 0.30%) | ✅ |
| 2.6[tau] | ΔMAPE — tau | ≤ 3 p.p. | +0.33 p.p. (oráculo 0.21%, real 0.54%) | ✅ |
| 2.6[theta] | ΔMAPE — theta | ≤ 3 p.p. | +0.28 p.p. (oráculo 0.06%, real 0.34%) | ✅ |
| 2.6[wn] | ΔMAPE — wn | ≤ 3 p.p. | +0.82 p.p. (oráculo 0.71%, real 1.54%) | ✅ |
| 2.6[zeta] | ΔMAPE — zeta | ≤ 3 p.p. | +0.99 p.p. (oráculo 1.40%, real 2.40%) | ✅ |
| 2.6 | Degradação end-to-end (pior parâmetro) | ≤ 3 p.p. | +0.99 p.p. (n=214) | ✅ |
| 2.6-adim-aceitas | Amostras comparáveis no nível adimensional (dispensa calibração) | diagnóstico | 141/300 (31 sem calibração) | ❓ |
| 2.6-adim[zeta] | ΔMAPE adimensional — zeta | ≤ 3 p.p. | +1.53 p.p. (oráculo 1.24%, real 2.78%) | ✅ |
| 2.6-classico-aceitas | Amostras comparáveis (extrator clássico) | diagnóstico | 183/300 | ❓ |
| 2.6-classico[K] | ΔMAPE (clássico) — K | diagnóstico | +0.59 p.p. (oráculo 0.10%, real 0.69%) | ✅ |
| 2.6-classico[tau] | ΔMAPE (clássico) — tau | diagnóstico | +0.77 p.p. (oráculo 0.22%, real 0.99%) | ✅ |
| 2.6-classico[theta] | ΔMAPE (clássico) — theta | diagnóstico | +0.66 p.p. (oráculo 0.06%, real 0.72%) | ✅ |
| 2.6-classico[wn] | ΔMAPE (clássico) — wn | diagnóstico | +1.19 p.p. (oráculo 0.64%, real 1.83%) | ✅ |
| 2.6-classico[zeta] | ΔMAPE (clássico) — zeta | diagnóstico | +1.62 p.p. (oráculo 1.71%, real 3.33%) | ✅ |
| 2.6-classico | Degradação end-to-end (extrator clássico, pior parâmetro) | diagnóstico | +1.62 p.p. (n=183) | ✅ |

