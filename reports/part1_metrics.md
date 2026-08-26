# Parte 1 — Métricas de aceitação

Relatório gerado automaticamente por `tests/` (`pytest_sessionfinish` em `tests/conftest.py`). Não editar à mão.

- Amostras renderizadas por conjunto: **600** (`clean`, `add_noise=False`; `noisy`, SNR fixo = 20 dB) — RULING Q, elevado de 300 para 600 por poder estatístico do portão; nenhum limiar foi alterado
- Sorteios sem renderização (critérios 1.3 e 1.4a): **20000**
- Estrato assertado nos critérios 1.1/1.2 (RULING C): `w = (t_end - θ)/T_dom ≥ 3`
- Workers: 16
- Tempo total da suíte: **576.5 s**

## 1. Critérios de aceitação

| # | critério | alvo | medido | veredito |
|---|---|---|---|---|
| **1.1** | Pipeline-oráculo, série limpa (estrato `w ≥ 3`) | MAPE < 1% em K, τ, θ, ωn, ζ | K 0.0000%, τ 0.0000%, θ 0.0000%, ωn 0.0000%, ζ 0.0000% — pior = zeta (0.0000%), n = 181 | **PASSA** |
| **1.2** | Pipeline com ruído SNR = 20 dB (estrato `w ≥ 3`) | MAPE < 5% (ωn/ζ só em ζ < 1,6 — RULING N) | K 0.385%, τ 0.979%, θ 1.568%, ωn (ζ<1.6) 3.111%, ζ (ζ<1.6) 3.557% — pior = ζ (ζ<1.6) (3.557%); n = 158 (ζ<1.6: n = 44) | **PASSA** |
| **1.2b** | RULING N — 2ª ordem com ζ ≥ 1,6 (ωn/ζ não identificáveis) | MAPE(K) < 5%, MAPE(T_lento) < 5%, NRMSE recon. < 0,05 | K = 0.351%, T_lento = 1.798%, NRMSE = 2.731e-03 (n = 32) | **PASSA** |
| **1.2c** | RULING N na população dedicada (n = 256 em `w ≥ 3`) | ζ < 1,6: MAPE(ωn), MAPE(ζ) < 5%; ζ ≥ 1,6: MAPE(K), MAPE(T_lento) < 5% e NRMSE recon. < 0,05 | ζ<1,6 (n = 122): ωn = 3.259%, ζ = 3.789% \| ζ≥1,6 (n = 134): K = 0.307%, T_lento = 1.479%, NRMSE = 2.513e-03 | **PASSA** |
| **1.3** | Vazamento: GBM só com atributos de `render` prevendo `order` | acurácia de teste ≤ 0.55 | 0.4985 (classe majoritária = 0.5045, n_teste = 6000) | **PASSA** |
| **1.3b** | Réplica de 1.3 no dataset renderizado (n = 600) | sem alvo: n pequeno demais para separar 0,55 do acaso | acurácia de teste = 0.4944 (n_teste = 180) | **medido** |
| **1.4a** | Spearman render × parâmetro no sorteio (n = 20000) | \|ρ\| < 0.05 em todos os 161 pares; nenhum significativo após Bonferroni | max \|ρ\| = 0.0194 (has_legend × wn); pares significativos = 0 | **PASSA** |
| **1.4b-clean** | Spearman render × parâmetro no dataset renderizado `clean` (n = 600) | p de permutação > 0.001 (literal do RULING G: \|ρ\| < 0.2) | max \|ρ\| = 0.1133 (aspect × tau), p = 0.9633 | **PASSA** |
| **1.4b-noisy** | Spearman render × parâmetro no dataset renderizado `noisy` (n = 600) | p de permutação > 0.001 (literal do RULING G: \|ρ\| < 0.2) | max \|ρ\| = 0.1762 (dpi × tau), p = 0.1292 | **PASSA** |
| **1.4c** | Round-trip exato: bloco `render` do meta == estilo sorteado | igualdade exata em todas as amostras (falso positivo zero) | 1200 amostras verificadas, todas idênticas | **PASSA** |
| **1.4d** | Cor da tinta na `image.png` == `render.line_color` (nível de pixel) | erro de canal = 0 em todas as amostras com cor modal dominante (`mode_frac ≥ 0.50`) | 0 violações em 900 amostras (300 excluídas por traço fino); erro máximo = 0 | **PASSA** |
| **1.4e** | Retas de span completo na área de dados × distratores declarados | nenhuma amostra sem grade com mais retas que `render.n_distractors` | 0 violações em 585 amostras sem grade; excesso máximo = 0 | **PASSA** |
| **1.5** | Máscara reprojetada pela `axis_affine` × `series` | RMSE do viés normal < 1.5 px; \|viés vertical\| < 0.3 px (sólida s/ marcador) | RMSE = 0.1649 px (0.0298 px sem marcador); viés vertical = +0.0027 px; cobertura = 0.492 px (máx) | **PASSA** |
| **1.5c** | Controle negativo do critério 1.5 | deslocar a afim em 3 px deve piorar o RMSE ≥ 10× | 0.1649 px → 2.2293 px (13.5×) | **PASSA** |
| **1.6** | Determinismo bit-a-bit (mesma seed ⇒ mesmos bytes) | sha256 idêntico de image.png e mask.png + meta.json idêntico | 5 seeds × 2 gerações: todos idênticos | **PASSA** |
| **1.7** | Tempo de geração extrapolado para 6000 amostras | < 15 min (folga 2× sobre os 30 min do PLANO) | 2.44 s para 200 amostras ⇒ 1.22 min | **PASSA** |
| **B** | Baselines clássicos × `identify` (FOPDT limpo, `w ≥ 3`) | sem alvo: comparação da monografia | MAPE(τ): identify = 0.0000% vs melhor baseline = 0.0491% | **medido** |
| **C** | RULING C — estrato truncado `w < 3` (resultado, não critério) | sem alvo: medido e reportado | limpo: MAPE(K) = 0.000% (n = 419); 20 dB: MAPE(K) = 127.627% (n = 442) | **medido** |
| **G** | `_estimate_gain` em janela truncada (`w < 3`, FOPDT limpo) | MAPE < 1.0% e cobertura = 100%; controle positivo: o atalho max(y) erra ≥ 10.0% no mesmo estrato | MAPE = 0.0000% (n = 191, cobertura 1.000) vs max(y) = 30.48% | **PASSA** |
| **R** | RULING S — máscara não degenerada (curva atravessa a janela) | extensão horizontal ≥ 0.93·projeção de `t_window`; ≥ 40 px acesos; ≤ 10% da imagem | cobertura: mín 0.9594, p1 0.9894, mediana 1.0021, máx 1.0486 — 0/1200 abaixo de 0.93; mín de px acesos = 171; fração máxima = 0.0731 | **PASSA** |

> Critérios marcados como `medido` são reportados sem assertiva por decisão registrada (RULING C para o estrato truncado, RULING N para ωn/ζ em ζ ≥ 1,6).

### 1.1 Tamanho de amostra de cada subgrupo efetivamente assertado

Onde o portão é fino. `w ≥ 3` retém ≈ ln 2 / ln 12 = 28% das amostras (consequência do sorteio log-uniforme da janela em [0,5 ; 6,0]·T_dom), e os parâmetros específicos de estrutura vivem em ≈ metade delas.

| critério | subgrupo | n |
|---|---|---|
| `1.1` | `w ≥ 3` — K, θ | 181 |
| `1.1` | `w ≥ 3` ∩ fopdt — τ | 102 |
| `1.1` | `w ≥ 3` ∩ second — ωn, ζ | 79 |
| `1.2` | `w ≥ 3` — K, θ | 158 |
| `1.2` | `w ≥ 3` ∩ fopdt — τ | 82 |
| `1.2` | `w ≥ 3` ∩ second ∩ ζ < 1.6 — ωn, ζ | 44 |
| `1.2b` | `w ≥ 3` ∩ second ∩ ζ ≥ 1.6 — K, T_lento, NRMSE | 32 |
| `1.2c` | pop. dedicada, `w ≥ 3` ∩ ζ < 1.6 | 122 |
| `1.2c` | pop. dedicada, `w ≥ 3` ∩ ζ ≥ 1.6 | 134 |
| `1.5` | todas as amostras — viés normal | 600 |
| `1.5` | linha sólida s/ marcador — viés vertical e cobertura | 109 |
| `G` | `w < 3` ∩ fopdt ∩ limpo — ganho estático | 191 |

**Os dois subgrupos mais finos do portão são `1.2b` (2ª ordem, `w ≥ 3`, ζ ≥ 1,6) e `1.2` (idem com ζ < 1,6). Ambos são corroborados pela população dedicada do critério `1.2c`**, que mede exatamente os mesmos dois subgrupos com uma ordem de grandeza a mais de séries (§3) e chega aos mesmos valores. O portão fino não deixa o resultado sem evidência: a evidência com poder de verdade está em 1.2c, e 1.2/1.2b são o gate sobre o conjunto de aceitação.

## 2.1 1.1 — pipeline-oráculo, série limpa

Erro por parâmetro, estratificado pela largura da janela `w`. `K`, `τ`, `ωn`, `ζ` em MAPE; **`θ` em NMAE normalizado por `T_dom`** (RULING J) — o MAPE de θ aparece ao lado apenas como número secundário.

| estrato | n | n fopdt / second | K (MAPE) | τ (MAPE) | θ (NMAE/T_dom) | θ (MAPE, secund.) | ωn (MAPE) | ζ (MAPE) |
|---|---|---|---|---|---|---|---|---|
| `w>=3` | 181 | 102 / 79 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `w<3` | 419 | 191 / 228 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `todos` | 600 | 293 / 307 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |

Mediana do mesmo erro (mostra quanto do MAPE vem de poucas amostras patológicas — decisivo no estrato truncado):

| estrato | n | K | τ | θ (/T_dom) | ωn | ζ |
|---|---|---|---|---|---|---|
| `w>=3` | 181 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `w<3` | 419 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| `todos` | 600 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |

- Acurácia de seleção de estrutura por AIC (`identify`): **1.000** (600/600)
- `K` e `θ` vêm de `identify()` (o pipeline real, todas as amostras). `τ`, `ωn` e `ζ` são específicos da estrutura e por isso vêm de `identify_both()` com a ordem verdadeira imposta — assim nenhuma amostra é descartada e o número não pode ser inflado por seleção de estrutura.

## 2.2 1.2 — série com ruído (SNR = 20 dB)

Erro por parâmetro, estratificado pela largura da janela `w`. `K`, `τ`, `ωn`, `ζ` em MAPE; **`θ` em NMAE normalizado por `T_dom`** (RULING J) — o MAPE de θ aparece ao lado apenas como número secundário.

| estrato | n | n fopdt / second | K (MAPE) | τ (MAPE) | θ (NMAE/T_dom) | θ (MAPE, secund.) | ωn (MAPE) | ζ (MAPE) |
|---|---|---|---|---|---|---|---|---|
| `w>=3` | 158 | 82 / 76 | 0.385% | 0.979% | 1.568% | 9.577% | 26.121% | 25.430% |
| `w<3` | 442 | 224 / 218 | 127.627% | 3.461% | 0.907% | 5.848% | 23.734% | 62.556% |
| `todos` | 600 | 306 / 294 | 94.120% | 2.796% | 1.081% | 6.830% | 24.351% | 52.958% |

Mediana do mesmo erro (mostra quanto do MAPE vem de poucas amostras patológicas — decisivo no estrato truncado):

| estrato | n | K | τ | θ (/T_dom) | ωn | ζ |
|---|---|---|---|---|---|---|
| `w>=3` | 158 | 0.213% | 0.816% | 0.733% | 4.996% | 5.208% |
| `w<3` | 442 | 1.809% | 2.417% | 0.462% | 7.707% | 14.939% |
| `todos` | 600 | 1.036% | 1.819% | 0.520% | 7.056% | 11.621% |

- Acurácia de seleção de estrutura por AIC (`identify`): **0.888** (533/600)
- `K` e `θ` vêm de `identify()` (o pipeline real, todas as amostras). `τ`, `ωn` e `ζ` são específicos da estrutura e por isso vêm de `identify_both()` com a ordem verdadeira imposta — assim nenhuma amostra é descartada e o número não pode ser inflado por seleção de estrutura.

**Estrato truncado (RULING C).** Com `w < 3` a curva é cortada antes do regime permanente e o ganho deixa de ser identificável: MAPE(K) = 127.6% contra mediana de 1.809%. A distância entre média e mediana diz que o erro vem de poucas amostras em que a extrapolação do patamar diverge, não de uma degradação uniforme. Isso é **limite de informação da janela**, não do método, e por isso é reportado sem assertiva — é resultado da monografia.

## 3. Não-identificabilidade prática em 2ª ordem (RULING N)

População dedicada de **872** séries de 2ª ordem a 20 dB (das quais **256** com `w ≥ 3`), ajustadas com a `order` imposta (`fit_second`). O ruído vem da mesma função do pipeline (`_apply_noise`), com o mesmo estilo sorteado e `snr_db` forçado — convenção do Ruling L e quantização idênticas às da geração real. Erros relativos médios (MAPE); `corr(erro_ωn, erro_ζ)` sobre os erros **relativos assinados**.

| estrato | faixa ζ | n | K | ωn | ζ | T_lento | T_rápido | NRMSE recon. | corr(e_ωn, e_ζ) |
|---|---|---|---|---|---|---|---|---|---|
| `w>=3` | [0,10 ; 1,00) | 83 | 0.583% | 1.45% | 2.42% | n/d | n/d | 2.669e-03 | 0.9191 |
| `w>=3` | [1,00 ; 1,60) | 39 | 0.547% | 7.10% | 6.70% | 4.859% | 17.53% | 2.825e-03 | 0.9711 |
| `w>=3` | [1,60 ; 2,20) | 59 | 0.342% | 22.61% | 20.80% | 1.733% | 31.54% | 2.590e-03 | 0.9996 |
| `w>=3` | [2,20 ; 3,00] | 75 | 0.280% | 87.66% | 83.56% | 1.279% | 63.35% | 2.453e-03 | 0.9997 |
| `todos` | [0,10 ; 1,00) | 269 | 588.223% | 12.89% | 95.83% | n/d | n/d | 2.718e-03 | -0.8024 |
| `todos` | [1,00 ; 1,60) | 170 | 167.389% | 13.95% | 47.80% | 328.152% | 26.59% | 2.727e-03 | -0.7457 |
| `todos` | [1,60 ; 2,20) | 204 | 6.488% | 18.54% | 22.87% | 10.707% | 27.65% | 2.855e-03 | 0.9433 |
| `todos` | [2,20 ; 3,00] | 229 | 2.651% | 69.01% | 69.17% | 4.542% | 53.22% | 2.757e-03 | 0.9953 |

- NRMSE do ruído injetado nesta população: **3.328e-02**; NRMSE de reconstrução médio: 2.762e-03 → a reconstrução é **12.1×** melhor que o ruído.
- No conjunto de aceitação `noisy` (300 amostras, as duas ordens): NRMSE do ruído = 3.331e-02, reconstrução 13.2× melhor.

Leitura: no estrato `w ≥ 3`, onde a janela contém informação suficiente, ωn e ζ erram dezenas de por cento na faixa superamortecida enquanto `K` e a constante do polo lento continuam corretos, e o erro de `T_lento` **melhora** com ζ — direção oposta à de ωn/ζ. A correlação → +1 entre os dois erros é a assinatura de deslizamento ao longo de uma curva de nível onde a dinâmica observável é a mesma. É limite de informação, não deficiência do estimador (a auditoria com partida no oráculo já descartou essa hipótese).

O bloco `todos` inclui as janelas truncadas (`w < 3`), onde nem `K` é identificável; ele está aqui para contraste, e não é a evidência do RULING N — a leitura acima vale para o bloco `w ≥ 3`.

## 4. Vazamento de rótulo

### 4.1 (1.3) Classificador treinado só com atributos de `render`

| conjunto | n treino | n teste | acurácia teste | baseline (classe majoritária) | alvo |
|---|---|---|---|---|---|
| sorteio puro (n = 20000) | 14000 | 6000 | **0.4985** | 0.5045 | ≤ 0.55 |

Um `GradientBoostingClassifier` que só vê atributos visuais não consegue prever a ordem da planta: a acurácia fica no acaso. É a contraprova direta do defeito do gerador legado `img.py`.

Importâncias mais altas (todas irrelevantes na prática):

| atributo | importância |
|---|---|
| line_width | 0.1524 |
| snr_db | 0.1167 |
| aspect | 0.1053 |
| height_px | 0.1000 |
| width_px | 0.0942 |
| dpi | 0.0728 |
| line_b | 0.0582 |
| line_g | 0.0570 |

### 4.2 (1.4a) Spearman no nível de sorteio — assertiva forte

n = 20000, 161 pares (atributo de `render` × parâmetro). Limiar |ρ| < 0.05, mais teste de significância com correção de Bonferroni sobre todos os pares (RULING G).

- **max |ρ| = 0.0194** no par `has_legend × wn`
- média de |ρ| = 0.0060
- menor p-valor = 3.100e-02; limiar de Bonferroni = 3.106e-04; pares significativos: **0**

Dez maiores |ρ|:

| atributo de render | parâmetro | ρ | p |
|---|---|---|---|
| has_legend | wn | 0.0194 | 0.051 |
| n_annotations | wn | 0.0191 | 0.055 |
| quantization_levels | wn | -0.0177 | 0.076 |
| has_xlabel | zeta | 0.0170 | 0.088 |
| bg_g | zeta | 0.0157 | 0.114 |
| dpi | is_second | -0.0153 | 0.031 |
| bg_r | is_second | -0.0150 | 0.034 |
| aspect | tau | -0.0144 | 0.151 |
| n_annotations | theta | -0.0143 | 0.043 |
| bg_b | tau | -0.0140 | 0.163 |

Com n = 20000 o erro padrão de ρ é ≈ 0,007: um ρ verdadeiro de 0,05 seria detectado com folga. Esta é a medida que de fato decide se o estilo visual carrega o rótulo.

### 4.3 (1.4b) Spearman no dataset renderizado — `clean`

n = 600, 161 pares. O bloco `render` do meta é byte-a-byte o estilo sorteado (verificado por round-trip exato), então esta medida é a **mesma variável aleatória** da 4.2, só que com n = 600 em vez de 20000.

- **max |ρ| = 0.1133** no par `aspect × tau`  (literal do RULING G: |ρ| < 0.20 → dentro)
- média de |ρ| = 0.0353
- **p de permutação = 0.9633** (4000 réplicas; limiar do portão: p > 0.001)
- nulo de permutação de `max |ρ|`: mediana 0.1472, p99 0.2185, p99,9 0.2508
- **sob independência perfeita, o limiar literal de 0,20 é excedido em 3% das réplicas** — ele não separa vazamento de acaso neste tamanho de amostra.

Dez maiores |ρ|:

| atributo de render | parâmetro | ρ | p |
|---|---|---|---|
| aspect | tau | 0.1133 | 0.0527 |
| width_px | tau | 0.1133 | 0.0527 |
| bg_g | wn | -0.1057 | 0.0645 |
| snr_db | wn | 0.1011 | 0.0769 |
| quantization_levels | w_window | -0.0933 | 0.0222 |
| width_px | K | -0.0910 | 0.0257 |
| quantization_levels | wn | 0.0862 | 0.1318 |
| aspect | w_window | 0.0856 | 0.0360 |
| bg_r | wn | -0.0824 | 0.1496 |
| aspect | theta | 0.0822 | 0.0442 |

### 4.4 (1.4b) Spearman no dataset renderizado — `noisy`

n = 600, 154 pares. O bloco `render` do meta é byte-a-byte o estilo sorteado (verificado por round-trip exato), então esta medida é a **mesma variável aleatória** da 4.2, só que com n = 600 em vez de 20000.

- **max |ρ| = 0.1762** no par `dpi × tau`  (literal do RULING G: |ρ| < 0.20 → dentro)
- média de |ρ| = 0.0370
- **p de permutação = 0.1292** (4000 réplicas; limiar do portão: p > 0.001)
- nulo de permutação de `max |ρ|`: mediana 0.1463, p99 0.2190, p99,9 0.2524
- **sob independência perfeita, o limiar literal de 0,20 é excedido em 3% das réplicas** — ele não separa vazamento de acaso neste tamanho de amostra.

Dez maiores |ρ|:

| atributo de render | parâmetro | ρ | p |
|---|---|---|---|
| dpi | tau | 0.1762 | 0.0020 |
| line_style | wn | 0.1207 | 0.0387 |
| line_style | zeta | -0.1136 | 0.0518 |
| has_marker | zeta | 0.1031 | 0.0774 |
| line_style | tau | 0.1027 | 0.0728 |
| n_annotations | tau | -0.0985 | 0.0853 |
| dpi | w_window | 0.0915 | 0.0249 |
| quantization_levels | wn | 0.0864 | 0.1396 |
| has_title | zeta | 0.0857 | 0.1427 |
| aspect | is_second | 0.0829 | 0.0424 |

### 4.5 Por que a assertiva de 1.4b não é o limiar literal (RULING O)

O RULING G fixou |ρ| < 0,20 raciocinando sobre o erro padrão de **um par** com n = 300. O estatístico que 1.4b de fato assere é o **máximo sobre ~161 pares**, e `τ`, `ωn` e `ζ` só existem em cerca de metade das amostras — o que derruba o n efetivo desses pares pela metade. Medindo o nulo por permutação do próprio conjunto:

| tamanho de amostra | mediana de max \|ρ\| sob H₀ | p95 / p99 | P(max \|ρ\| ≥ 0,20) sob H₀ |
|---|---|---|---|
| n = 300 (2000 réplicas) | 0.2076 | 0.2713 | **60%** |
| n = 600 (4000 réplicas) | 0.1472 | 0.2185 | **3%** |

Com n = 300, o limiar literal é excedido em **60% das réplicas sob independência perfeita** — ele reprova o gerador correto na maioria das vezes. É a mesma patologia que o RULING G identificou no alvo original de 0,05 do PLANO, um nível abaixo. Por isso a assertiva do portão é o teste de permutação (`p > 1e-3`), que é o nulo correto para uma estatística de máximo, somado ao round-trip exato do bloco `render` (§4.6), que tem poder real e falso positivo zero. O valor literal continua **medido e reportado** acima.

### 4.6 (1.4c) Round-trip exato do bloco `render`

Para cada uma das **1200** amostras renderizadas, o estilo é re-derivado a partir de `meta["seed"]` (mesmo `SeedSequence.spawn(3)`) e exige-se `style.to_meta() == meta["render"]` e `spec == meta["params"]`, com igualdade exata. Todas passaram.

Esta é a verificação anti-vazamento com **poder real sobre o caminho de renderização**: se qualquer etapa entre o sorteio e o meta.json alterasse um atributo visual em função do sistema, a igualdade quebraria. Ao contrário de uma correlação com n finito, aqui a taxa de falso positivo é zero e a de falso negativo também. Somada à §4.2 (n = 20000, que limita qualquer ρ verdadeiro a < 0,05), ela é o que de fato sustenta a alegação de ausência de vazamento.

### 4.7 (1.4d, 1.4e) Vazamento no nível de pixel — o que o meta não vê

Os critérios 1.3 e 1.4a–c leem o bloco `render` do meta; nenhum deles abre a `image.png`. O **teste de mutação** (HANDOFF §3.5) mostrou que isso deixava passar exatamente a forma do defeito do `img.py`: um vazamento que vive só nos pixels. Dois mutantes atravessaram a suíte com 30 passed — cor da curva escolhida por `order` sem tocar no meta, e `axhline(K)`/`axvline(θ)` redesenhados na figura. Os dois critérios abaixo fecham esses buracos.

**1.4d — cor da tinta.** A cor modal dos pixels acesos da máscara, lida na `image.png`, tem de ser exatamente `render.line_color`. Assertado nas **900** de 1200 amostras com cor modal dominante (`mode_frac ≥ 0,50`); nas 25.0% restantes não há pixel de interior puro — traço no piso de 1,5 px, ou curva muito recortada pelo ruído — e a moda é mistura de anti-aliasing. Medido: **0 violações**, erro máximo de canal = 0. No gerador mutante que escolhe a cor por `order`, 279/279 violações, com erro mediano de 144 níveis: separação total.

**1.4e — retas não declaradas.** Toda reta que atravessa a área de dados de ponta a ponta tem de estar no orçamento de `render.n_distractors`. Assertado nas **585** amostras sem grade (615 com grade ficam fora: as linhas da grade também são retas de span completo e o `render` não declara quantas). Medido: **0 violações**, excesso máximo = 0. No gerador mutante que redesenha `axhline(K)`/`axvline(θ)`, 143/157 (91,1%) — como a assertiva é sobre o conjunto, a detecção é certa.

O critério é o do contrato §2: todo elemento visual tem de estar declarado no estilo, e o estilo por construção não vê o rótulo. Os dois limiares foram fixados **depois** de medir o teto atingível (916 amostras sem grade para o 1.4e, entre os conjuntos de aceitação e um lote independente), como manda o padrão dos RULINGS O/R/S — e cada condição do 1.4e existe por um falso positivo medido, documentado na docstring de `_spanning_rows`.

## 5. (1.5) Consistência entre `mask.png` e `axis_affine`

A `series` é reprojetada para pixels pela afim inversa (`px = (t − ox)/sx`, `py = (y − oy)/sy`) e comparada com os pixels acesos da máscara.

| métrica | n | média | p95 | máx | assertiva |
|---|---|---|---|---|---|
| viés normal assinado por amostra (px) | 600 | -0.0627 | 0.3521 | 1.4942 | RMSE < 1,5 px |
| **RMSE do viés normal (px)** | 600 | **0.1649** | -- | -- | < 1,5 px |
| cobertura curva→tinta, sólida s/ marcador (px) | 109 | 0.406 | 0.444 | 0.492 | < 1,5 px |
| viés vertical (mediana por coluna), sólida s/ marcador (px) | 109 | 0.0027 | -- | 0.2096 | \|média\| < 0,3 px |
| distância bruta pixel→polilinha, sem correção (px) | 600 | 1.232 | 2.434 | 5.266 | só reportada |

Viés normal separado por presença de marcador:

| estrato | n | RMSE do viés (px) | máx \|viés\| (px) |
|---|---|---|---|
| sem marcador | 428 | 0.0298 | 0.1230 |
| com marcador | 172 | 0.3043 | 1.4942 |

Praticamente todo o viés residual vem das amostras **com marcador**. O glifo é centrado no ponto de dado, mas a sua massa de pixels não é simétrica em relação à **tangente local** da curva — um `^` tem centroide acima do centro, e qualquer glifo concentra área num punhado de pontos esparsos onde a inclinação da curva é uma só. Isso desloca a média do offset normal sem que a afim tenha erro nenhum: no estrato sem marcador o RMSE cai para **0.0298 px** e o pior caso para 0.1230 px, ou seja, a calibração está correta em bem menos de um décimo de pixel.

**Controle negativo:** injetando um deslocamento de 3 px na afim, o RMSE do viés normal salta de 0.1649 px para 2.2293 px (**13.5×**). A métrica tem sensibilidade real a erro de calibração.

A distância **bruta** pixel→polilinha não é assertada porque é dominada pela **espessura do traço desenhado** (meia-largura de até 4.0 px no conjunto), que é geometria pretendida e não erro de calibração: um traço de largura `L` produz sozinho um RMSE de `L/(2√3)` px mesmo com afim perfeita. O que mede calibração é o **viés assinado**, que é insensível à espessura (o traço é simétrico em torno do eixo da curva) e é a métrica assertada acima.

### 5.1 Erro do extrator ingênuo "mediana por coluna" (medido, sem assertiva — RULING H)

Quanto o extrator do Estágio A (Parte 2) terá de interpolar em traço descontínuo. Erro absoluto médio, em pixels.

| line_style | marcador | n | erro médio (px) | colunas sem tinta (%) |
|---|---|---|---|---|
| `-` | não | 109 | 0.184 | 0.0 |
| `-` | sim | 43 | 0.373 | 0.0 |
| `--` | não | 101 | 0.599 | 20.3 |
| `--` | sim | 38 | 0.786 | 16.8 |
| `-.` | não | 118 | 0.509 | 20.9 |
| `-.` | sim | 50 | 0.875 | 15.6 |
| `:` | não | 100 | 0.896 | 43.8 |
| `:` | sim | 41 | 1.096 | 36.8 |

## 5.2 Máscara não degenerada (RULING S)

Medido sobre as **1200** amostras renderizadas (`clean` + `noisy`). A pergunta é: *a máscara contém uma curva que atravessa o gráfico inteiro, em vez de uma máscara degenerada?*

### 5.2.1 Os três denominadores (registro de metodologia)

Chegar a uma grandeza que responda essa pergunta exigiu três tentativas. O registro dos dois erros vale mais que o resultado final, porque é a **mesma classe de defeito se repetindo**: um limiar fixo comparado contra uma grandeza cujo **teto físico varia de amostra para amostra**.

| # | grandeza assertada | teto físico | veredito | o que falhou |
|---|---|---|---|---|
| 1 | contagem absoluta de px acesos ≥ 200 (RULING I) | ∝ comprimento da curva × espessura × ciclo do tracejado | **errado** | 1/600 (`sample_00345`, 171 px, 374×210 a 70 dpi, `:`) — figura pequena; nenhuma máscara correta daquele estilo passaria |
| 2 | extensão / largura do `plot_bbox_px` ≥ 0,90 (RULING R) | `1/(1+m_lo+m_hi)` ∈ [0,893 ; 0,980] | **errado** | 15/1200, das quais 10 com teto abaixo de 0,90; uma delas 1598×765 com 3685 px acesos — não era degeneração |
| 3 | extensão / projeção de `t_window` ≥ 0.93 (RULING S) | **1,0 para toda amostra** | **correto** | nenhum limiar fixo abaixo de 1,0 pode colidir com esse teto |

O denominador do RULING S é a largura em pixels da **janela de dados** de fato — `|px(t_end) − px(t_start)|` pela `axis_affine` — e não a do retângulo dos eixos. É invariante a resolução, espessura do traço, tracejado **e margens**. A troca também **fortalece** a assertiva em relação ao piso original: uma máscara com 5000 px concentrados em 20% da largura passava nos 200 px e falha aqui.

### 5.2.2 Distribuição medida

| grandeza | mín | p1 | mediana | máx | limite | abaixo do limite |
|---|---|---|---|---|---|---|
| **cobertura (RULING S) = extensão / projeção de `t_window`** | **0.9594** | 0.9894 | 1.0021 | 1.0486 | ≥ 0.93 | **0/1200** |
| cobertura antiga (RULING R) = extensão / largura do `plot_bbox_px` | 0.8895 | 0.8986 | 0.9384 | 0.9923 | ≥ 0,90 (revogado) | 15/1200 |
| pixels acesos | 171 | -- | 4519 | 42557 | ≥ 40 | 0 |
| fração da imagem | -- | -- | -- | 0.07311 | ≤ 0,10 | 0 |

As duas primeiras linhas são a mesma extensão acesa medida contra denominadores diferentes: a comparação lado a lado deixa auditável que a mudança de veredito (15 → 0 reprovações) vem do **denominador**, não de qualquer mudança no gerador ou na máscara. O teto de margens que condenava o denominador antigo foi medido em [0.8958 ; 0.9775], mediana 0.9353.

Mediana da cobertura acima de 1,0 é esperada: `solid_capstyle="round"` estende o traço meia-largura além dos extremos da curva.

Por estilo de traço, com o **déficit máximo que o vão do tracejado pode causar** naquele estrato (`k·line_width/projeção`, com `k` = 1,6 para `--` e `-.`, 1,65 para `:`) — é o mecanismo que produz a cauda inferior:

| line_style | n | cobertura mín | cobertura mediana | déficit máx. possível por vão |
|---|---|---|---|---|
| `-` | 308 | 0.9995 | 1.0042 | 0.0000 |
| `--` | 289 | 0.9594 | 1.0014 | 0.0519 |
| `-.` | 308 | 0.9850 | 1.0010 | 0.0477 |
| `:` | 295 | 0.9798 | 1.0012 | 0.0487 |

As cinco menores coberturas do conjunto:

| amostra | cobertura | px acesos | figura (px) | dpi | line_style | traço (px) | déficit por vão |
|---|---|---|---|---|---|---|---|
| `sample_00371` | 0.9594 | 1133 | 267×382 | 150 | `--` | 5.57 | 0.0506 |
| `sample_00280` | 0.9798 | 2236 | 667×810 | 181 | `:` | 7.08 | 0.0231 |
| `sample_00215` | 0.9804 | 3376 | 324×1095 | 87 | `--` | 2.86 | 0.0189 |
| `sample_00310` | 0.9837 | 860 | 599×413 | 176 | `:` | 4.32 | 0.0172 |
| `sample_00397` | 0.9844 | 1514 | 763×307 | 170 | `:` | 6.70 | 0.0201 |

**Margem do limiar.** O pior caso medido é 0.9594, contra o limiar de 0.93: folga de 0.0294. O mecanismo da cauda é estocástico (traço tracejado que termina em vão), e o déficit máximo por vão medido neste conjunto é 0.0519 (p99 = 0.0406). O limiar é **empírico, não um limite derivado**: o canto analítico do espaço de estilos (figura estreita × dpi alto × traço grosso × pontilhado terminando em vão) admite déficits maiores do que os observados. Isso está registrado para que uma reprovação futura seja lida como cauda de estilo, e não como defeito do gerador, antes de qualquer conclusão.

## 6. (1.6) Determinismo bit-a-bit

| seed | sha256 image.png | sha256 mask.png | meta idêntico |
|---|---|---|---|
| 0 | `67a31c765a577073…` | `833f7ec93ec12837…` | sim |
| 1 | `b4661c5a318219d9…` | `0524022846c94495…` | sim |
| 7 | `11f85baf11fb209f…` | `3d7eae205a3df921…` | sim |
| 12345 | `9508b5b8148353ad…` | `39413e9fcc502536…` | sim |
| 987654321 | `84777e639aa96b83…` | `54a6d964065df33f…` | sim |

Cada seed foi gerada duas vezes, em diretórios distintos; os hashes são dos dois arquivos produzidos e coincidem. O `meta.json` é comparado ignorando `sample_id`, que por contrato é o basename do diretório.

## 7. (1.7) Desempenho de geração

| n medido | workers | tempo (s) | s/amostra | extrapolado p/ 6000 (min) | alvo |
|---|---|---|---|---|---|
| 200 | 16 | 2.44 | 0.0122 | **1.22** | < 15 min (folga 2× sobre os 30 min do PLANO) |

## 8. Baselines clássicos × `identify` (mesmas séries limpas, FOPDT)

Erro sobre as amostras FOPDT do conjunto limpo. `θ` em NMAE/`T_dom`. `cobertura` é a fração de séries em que o método devolveu um resultado finito (os baselines devolvem `nan` quando o percentil exigido não é atingido dentro da janela).

| método | estrato | n | cobertura | K (MAPE) | τ (MAPE) | θ (NMAE/T_dom) |
|---|---|---|---|---|---|---|
| identify | `w>=3` | 102 | 1.000 | 0.000% | 0.000% | 0.000% |
| identify | `w<3` | 191 | 1.000 | 0.000% | 0.000% | 0.000% |
| tangente | `w>=3` | 102 | 1.000 | 0.000% | 1.346% | 0.009% |
| tangente | `w<3` | 191 | 1.000 | 0.000% | 0.504% | 0.002% |
| Smith | `w>=3` | 102 | 1.000 | 0.000% | 0.049% | 0.081% |
| Smith | `w<3` | 191 | 0.602 | 0.000% | 0.049% | 0.082% |
| Sundaresan–Krishnaswamy | `w>=3` | 102 | 1.000 | 0.000% | 0.712% | 1.296% |
| Sundaresan–Krishnaswamy | `w<3` | 191 | 0.309 | 0.000% | 0.712% | 1.313% |

Como ler a tabela, sem superestimar os baselines:

1. Os três baselines obtêm `K` de `identify.classical._estimate_gain`, que por sua vez roda um ajuste FOPDT completo quando a cauda não está assentada. Por isso o `K` deles é exato aqui: o que está sendo comparado são apenas as fórmulas de `τ` e `θ`, não a estimação do ganho.
2. O erro de `τ` de Smith (≈ 0,05%) e de S–K (≈ 0,7%) é o **viés intrínseco das constantes** de cada fórmula sobre uma FOPDT exata (1,5·ln(0,717/0,368) = 1,0008 e 0,67·ln(0,647/0,147) = 0,9931), e por isso não muda entre estratos.
3. Onde os baselines de fato quebram é na **cobertura**: no estrato truncado, Smith responde em pouco mais da metade das séries e S–K em cerca de um quarto — os percentis de 63,2% e 85,3% simplesmente não são atingidos dentro da janela. `identify` responde em 100% delas porque ajusta o modelo inteiro por mínimos quadrados com multi-start, sem depender de cruzamentos.

### 8.1 (G) Guarda do ganho estático — por que o item 1 acima é verdade

A leitura da tabela depende de uma afirmação: o `K` dos baselines é exato porque vem de `_estimate_gain`, que ajusta o modelo em vez de ler `max(y)`. Essa afirmação passou a ser **assertada**, e não apenas escrita, depois que o teste de mutação (HANDOFF §3.5) mostrou que um `_estimate_gain` devolvendo `max(y)` — um bug real, corrigido na Tarefa 2 — atravessava a suíte inteira sem quebrar nada.

No estrato truncado (`w < 3`, FOPDT limpo, n = 191), onde a curva não assenta e o atalho necessariamente erra: `_estimate_gain` fica em **MAPE = 0.0000%** com cobertura 1.000, contra **30.48%** de `max(y)` no mesmo estrato. O segundo número é um **controle positivo**: se um dia ele cair abaixo de 10%, o estrato deixou de separar o atalho do estimador e o teste avisa em vez de virar vácuo.

## 9. Verificação cruzada dos modelos

`dataset.generator.step_response` × `identify.classical.model_response`, 200 conjuntos de parâmetros aleatórios (as duas ordens, ζ atravessando 1).

| grandeza | valor |
|---|---|
| máx \|Δ\| absoluto | 3.687e-07 |
| máx \|Δ\| / K | 4.439e-08 |
| tolerância | 1e-9 (geral); 1e-6·\|K\| na vizinhança \|ζ−1\| ≤ 1e-6 |

