# Plano de Execução — Sistema de Identificação Automática de Plantas a partir de Imagens de Resposta ao Degrau

**Aluno:** Luiz Miguel De Jesus Santiago · **Orientador:** Prof. Dr. Emerson Alves da Silva
**Horizonte:** 3 semanas · **Hardware:** notebook local (RTX 4050 6 GB, 16 threads, 15 GB RAM)

---

## 0. Diagnóstico do estado atual (base honesta para o Ciclo 4)

Antes de planejar, é preciso registrar o que de fato existe no repositório, porque três afirmações dos relatórios anteriores não são sustentadas pelo código:

| Afirmado nos relatórios | Estado real em `img.py` |
|---|---|
| Script "completamente reparametrizado": resolução, cor, estilo e ruído aleatórios (Ciclo 3, §2.1) | Resolução fixa (`figsize=(6,4)`, `dpi=150`), cor fixa (`#378ADD`), sem ruído, sem randomização de estilo |
| Dataset de 1.200 → 50.000 imagens | 300 imagens (150 de 1ª ordem + 150 de 2ª ordem) |
| Dataset com "baixa variabilidade intraclasse" como limitação | O problema é mais grave que baixa variabilidade: **há vazamento de rótulo** |

### 0.1 O vazamento de rótulo (defeito crítico)

O gerador atual desenha a resposta do gabarito dentro da imagem:

- `save_step_responses` traça `axhline(K)` — o valor de regime permanente, que é exatamente o parâmetro K, aparece como uma reta tracejada;
- traça `axvline(theta + tau)` — a soma dos dois outros parâmetros a estimar;
- traça `axvline(theta)` **somente quando existe atraso**, ou seja, a presença da linha pontilhada é o rótulo `has_delay`;
- `save_second_order_step_responses` escolhe a **cor da curva em função de `damping_type`** (`#378ADD`/`#EF9F27`/`#D85A30`), isto é, a cor codifica o regime de amortecimento.

Uma CNN treinada nesse conjunto aprende a localizar linhas tracejadas e a ler o canal de cor — não a dinâmica. Isso explica de forma plausível a acurácia de 93 % do Ciclo 1 e o platô do Ciclo 2: o modelo já estava saturando um atalho, e nem mais dados nem mais camadas removem um atalho.

**Consequência para o plano:** todo resultado anterior (93 %, <95 %) deve ser tratado como não-válido e re-medido. Isso não é um retrocesso a esconder — é um achado metodológico legítimo e, redigido corretamente, vale uma seção da monografia sobre *shortcut learning* / vazamento em datasets sintéticos.

---

## 1. Decisões de arquitetura e suas justificativas

> A literatura que sustenta cada decisão desta seção está mapeada em
> **`REFERENCIAS.md`**, organizada por decisão — §1.1, §1.2, §1.3, §1.4, §1.5,
> §1.7 e §1.8 têm entrada própria lá, com as lacunas conhecidas declaradas ao fim.

### 1.1 Decisão A — Saída em unidades físicas, com calibração de eixos explícita

**Escolha:** o sistema devolve K, τ, θ, ωₙ, ζ em unidades físicas, lendo a escala dos eixos da própria imagem.

**Justificativa.** A partir dos pixels da curva isoladamente, os parâmetros dimensionais **não são identificáveis**: a mesma geometria de traçado corresponde a (K = 2, τ = 1 s) e a (K = 200, τ = 1 min). Apenas grandezas adimensionais — a ordem do sistema, ζ, e as razões τ/T_janela, θ/T_janela, K/y_máx — são invariantes de escala. Portanto a leitura dos eixos não é um acessório do sistema: é a condição necessária para que a saída tenha significado físico. Imagens sem eixos numerados estão, por construção, fora do domínio de aplicabilidade — e isso deve ser declarado como hipótese do trabalho, não como limitação envergonhada.

**Consequência arquitetural:** o requisito "funcionar com imagens sem texto" é reinterpretado como **"invariante a texto irrelevante"** (títulos, legendas, anotações, marca d'água), mantendo a dependência do **texto relevante** (rótulos numéricos dos ticks).

### 1.2 Decisão B — Pipeline em estágios, não CNN fim-a-fim

**Escolha:** `imagem → [A] extração da curva → [B] calibração dos eixos → série y(t) → [D] identificação por mínimos quadrados`.

> **Revisão de 22/08/2026.** O plano original tinha **quatro** estágios, com um `[C] estimador neural 1D` entre B e D. O estágio C foi medido e removido — ver §1.3. Os quatro argumentos abaixo continuam válidos: eles justificam o pipeline **em estágios** contra a CNN fim-a-fim, e não dependem de C existir.

**Justificativa em quatro pontos:**

1. **Invariância por construção, não por esperança.** Numa CNN 2D fim-a-fim, a invariância a resolução, cor, grade e legenda só pode ser induzida por *data augmentation* — e permanece uma propriedade empírica, não garantida, que falha silenciosamente fora da distribuição. No pipeline em dois estágios, a série y(t) normalizada é a **mesma representação** quer a figura tenha 300×200 px em preto e branco, quer tenha 1920×1080 px colorida com legenda. A invariância vira uma propriedade estrutural, demonstrável.

2. **Vazamento de rótulo detectável, não invisível.** Este é o argumento mais forte *neste* projeto, pela sua própria história: o `img.py` legado vazava o rótulo dentro da imagem, e um GBM olhando apenas atributos visuais acertava a ordem do sistema em 93 %. Uma CNN fim-a-fim é exatamente a arquitetura que explora esse tipo de atalho **sem deixar rastro** — não há representação intermediária para inspecionar, e a métrica de teste sobe, não desce. No pipeline em estágios a máscara e a série são artefatos inspecionáveis, e foi assim que os critérios 1.4d/1.4e fecharam o elo pixel→meta→estilo e provaram a ausência de vazamento. Uma arquitetura em que o vazamento é *indetectável* não é aceitável num trabalho cuja contribuição metodológica é justamente ter encontrado um vazamento.

3. **Atribuição de erro.** Com fim-a-fim, um MAPE de 15 % em τ é um número sem diagnóstico. Aqui, cada estágio tem métrica própria e é possível afirmar, por exemplo: "a segmentação contribui com 2 % do erro, a calibração com 1 %, o estimador com 12 %" — que é o tipo de análise que sustenta uma defesa.

4. **Viabilidade no hardware disponível.** Com a remoção do estágio C (§1.3) resta **um único modelo treinado** — a U-Net do estágio A — e a §1.8 põe um extrator clássico como alternativa que dispensa `torch` por completo. Contra isso, a alternativa fim-a-fim exige treinar uma ResNet-18 sobre ~50 k imagens numa GPU de 6 GB cujo driver ainda nem está instalado, e não tem caminho de contingência: se a GPU não sobe, não há trabalho a entregar.

**Contrapartida assumida:** são três estágios a integrar em vez de um passo único, e o gerador precisa emitir a máscara da curva como rótulo adicional (custo marginal ~zero, já que quem desenha a curva sabe onde ela está). A alternativa fim-a-fim **não é descartada como experimento**: ela está planejada em `PLANO_CNN_FIM_A_FIM.md` para rodar como *baseline* no fim do trabalho, quando o dataset já existir, convertendo esta decisão de argumento em medição (critério 3.9).

### 1.3 Decisão C — Estágio D como estimador único, sem estimador neural intermediário

**Escolha:** a identificação paramétrica é feita **inteiramente** por `scipy.optimize.least_squares` com multi-start clássico e arbitragem de estrutura por AIC (`identify/classical.py`, já implementado e medido na Parte 1). **Não há rede neural estimadora de parâmetros.**

**Histórico da decisão.** O plano original previa um estágio C — CNN 1D dilatada com três cabeças, emitindo a ordem do sistema e o chute inicial — e o estágio D como refinamento sobre essa predição. O argumento era o clássico: otimização não-linear atinge erro sub-1 % (Ljung, *System Identification*) mas falha de dois modos, escolha de estrutura e mínimo local por inicialização ruim; uma rede é boa nesses dois e ruim em precisão fina, o otimizador é o oposto.

O argumento é correto **em geral**. Ele foi medido **neste problema** em 22/08/2026, e não se sustenta: as duas tarefas que o estágio C existia para resolver já estão resolvidas.

**Medição 1 — seleção de estrutura por AIC, n = 900, SNR = 20 dB.** Acurácia estratificada por ζ:

| estrato | acerto | NRMSE de reconstrução quando erra |
|---|---|---|
| 2ª ordem, ζ ∈ [0,10; 0,50) | **61/61 — 100 %** | — |
| 2ª ordem, ζ ∈ [0,50; 0,80) | **42/42 — 100 %** | — |
| 2ª ordem, ζ ∈ [0,80; 1,30) | **70/70 — 100 %** | — |
| 2ª ordem, ζ ∈ [1,30; 1,60) | **44/44 — 100 %** | — |
| 2ª ordem, ζ ∈ [1,60; 2,20) | 91/107 — 85,0 % | mediana 4,1e-03 |
| 2ª ordem, ζ ≥ 2,20 | 50/118 — 42,4 % | mediana 3,0e-03 |
| FOPDT | 412/458 — 90,0 % | mediana 3,5e-03 |
| **global** | 770/900 — 85,6 % | **máximo 5,8e-03** |

Leitura: **todo** erro de estrutura está na região onde as duas famílias são observacionalmente equivalentes (ζ alto, onde a 2ª ordem superamortecida é indistinguível de um FOPDT — §1.5). Onde a distinção é observável, ζ < 1,6, o AIC está em **100 %**. E quando ele erra, o custo é 9e-04 de NRMSE — **50× abaixo** do limiar de 0,05, com 0/900 amostras acima do limiar. Não é erro de estimador: é ausência de informação no dado. Nenhuma rede pode superar isso, porque o que falta não está na entrada.

**Medição 2 — inicialização por multi-start clássico, n = 400, SNR = 20 dB.** `identify` convergiu em **400/400** amostras; 0/400 acima do limiar de NRMSE. O ganho do refinamento sobre o chute inicial puro — que é exatamente o que o critério 3.7 mediria contra a rede — foi:

| | chute inicial | após o estágio D | ganho |
|---|---|---|---|
| NRMSE mediana | 8,674e-03 | 2,460e-03 | 3,5× |
| NRMSE p95 | 2,119e-02 | 4,401e-03 | 4,8× |

Redução de **72 %**, contra o alvo de 40 % do critério 3.7 original. O estágio D se justifica com folga; o que não se justifica é *quem* fornece o chute, porque as quatro heurísticas de `_multistart` (regressão integral, grade com K perfilado, heurística de sobressinal, heurística superamortecida) já não falham.

Os dois experimentos usaram ruído a 20 dB, correspondente a σ ≈ 2,4 % da faixa do sinal — cerca de **4× maior** que o erro de percepção de 0,649 % medido nos estágios A+B. O teste é mais duro que a degradação que a Parte 2 introduz, não mais fácil.

**Consequências.**

1. Sai um modelo treinado inteiro, as 200.000 séries de treino, a calibração de λ, a parametrização log/logit, o adaptador de reamostragem entre B e C, e `dataset/series.py`.
2. Sai o critério 3.1 (acurácia de ordem ≥ 95 %) — que a Medição 1 mostra ser **inalcançável em princípio** no estrato ζ ≥ 1,6, por qualquer método. Ele é substituído pelo critério 3.1 novo, que mede a coisa certa: a estratificação, mais o **custo limitado** do erro de estrutura.
3. Saem os critérios 3.2 e 3.7 na forma original.
4. O trabalho **não perde a componente de aprendizado profundo**: o estágio A é uma U-Net, e segmentar uma curva de um gráfico com resolução, paleta, grade, legenda e distratores arbitrários é o problema de aprendizado legítimo deste trabalho. Passa a haver uma rede em vez de duas.

**O que se ganha na monografia.** A pergunta de banca "por que não usar só mínimos quadrados?" ganha uma resposta *melhor* que a original: não é "porque a rede escolhe a estrutura", é **"medimos, e o AIC atinge 100 % onde a distinção é observável; onde ele erra, as duas estruturas são indistinguíveis e o erro custa 9e-04 de NRMSE"**. Isso é um resultado com estratificação e limite de informação, não uma escolha de projeto a defender.

**Gatilho para revisão (condição de volta atrás).** Reintroduzir o estágio C fica como trabalho futuro com condição objetiva: **se** o critério 2.6 mostrar que a degradação de percepção derruba a taxa de convergência de `identify` abaixo de 99 % ou empurra o NRMSE p95 acima de 0,02, **então** o chute inicial voltou a ser um problema e a especificação original do estágio C — preservada no histórico do git e no `HANDOFF.md §5` — volta à mesa. Sem esse gatilho disparado, uma segunda rede é peso morto que a banca vai, com razão, perguntar por que está ali.

### 1.4 Decisão D — Cobertura: FOPDT + segunda ordem canônica

- 1ª ordem com tempo morto: `G(s) = K·e^{-θs}/(τs + 1)` — 3 parâmetros. É o modelo de referência da sintonia PID industrial (Ziegler–Nichols, Cohen–Coon, IMC), o que ancora a relevância prática do trabalho.
- 2ª ordem canônica: `G(s) = Kωₙ²/(s² + 2ζωₙs + ωₙ²)` — 3 parâmetros, cobrindo ζ subamortecido, crítico e superamortecido.

Fora de escopo (declarado): zeros, fase não-mínima, integradores, ordens superiores.

### 1.5 A ambiguidade estrutural que precisa ser tratada de frente

Um sistema de 2ª ordem **superamortecido** (ζ > 1) produz uma resposta ao degrau visualmente quase indistinguível de um FOPDT. Isso não é deficiência do modelo — é **não-identificabilidade prática** do problema: as duas famílias geram curvas cuja diferença é menor que o ruído de extração.

Tratamento adotado:
- não forçar decisão binária dura: o estágio D ajusta **sempre as duas** estruturas e escolhe pelo critério de resíduo penalizado (AIC). Como não há mais um classificador a montante (§1.3), não há faixa de probabilidade a arbitrar — a arbitragem é sempre a posteriori, sobre dois ajustes convergidos. Medido: acurácia de 100 % em ζ < 1,6 e custo máximo de 5,8e-03 de NRMSE quando erra;
- a **métrica primária do trabalho passa a ser o erro de reconstrução da curva** (NRMSE entre a resposta do modelo identificado e a série extraída), não a acurácia de classificação. Justificativa: para uso em controle, o que importa é que o modelo reproduza a dinâmica, não que ele carregue o rótulo "certo" numa região onde o rótulo é ambíguo;
- a acurácia de classificação continua reportada, mas **estratificada por ζ**, evidenciando que o erro se concentra em ζ ≈ 1 — o que é um resultado, não uma falha.

### 1.6 Hipóteses explícitas (a declarar na monografia)

1. O gráfico é uma resposta a um degrau de **amplitude conhecida** (unitária, salvo anotação). Sem isso, apenas o produto K·A é identificável.
2. Os eixos são lineares (não logarítmicos). Ao menos 2 ticks rotulados por eixo são necessários **para a saída em unidades físicas**; sem eles o sistema ainda entrega a resposta adimensional, com degradação graciosa (§1.7).
3. A curva é única na figura (sem múltiplas séries sobrepostas) — múltiplas curvas ficam como teste de estresse, não como requisito.
4. O instante do degrau pode ser ≠ 0 e é absorvido em θ.

### 1.7 Decisão E — OCR opcional, não estrutural: saída em dois níveis

**Escolha:** o sistema sempre entrega o **nível adimensional** da resposta; o **nível físico** é acrescentado quando — e só quando — a calibração dos eixos tiver sucesso. `Calibration.ok = False` degrada a saída, não a invalida.

**Justificativa.** O §5 classifica "OCR falha em fontes pequenas / DPI baixo" com probabilidade **Alta**, e os critérios 2.3, 2.4 e 2.5 existem inteiramente por causa dessa fragilidade. Só que boa parte da resposta não depende de OCR nenhum:

| grandeza | precisa de calibração? |
|---|---|
| `order` (estrutura) | **não** |
| **ζ** | **não** — adimensional, vem da forma da curva |
| θ/τ, ωₙ·T, θ/T | **não** — todas adimensionais |
| τ, θ, ωₙ absolutos | só da escala do eixo **x** |
| K | só da escala do eixo **y** |

Fazer o OCR carregar a resposta inteira transforma o elo mais frágil da cadeia em ponto único de falha para grandezas que **não precisam dele**. Pior: no conjunto OOD — figuras de Ogata e Nise digitalizadas, capturas de planilha com tipografia que o Tesseract nunca viu — é onde o OCR tem mais chance de falhar e onde a resposta adimensional é provavelmente a única robusta.

**Interface.** `identify_from_image` passa a devolver sempre:

```jsonc
{
  "order": "second",
  "dimensionless": {"zeta": 0.3238, "wn_T": 15.27, "theta_T": 0.0544, "K_yrange": 0.7437},
  "physical":      {"K": 0.6073, "wn": 0.3361, "zeta": 0.3238, "theta": 2.4676},  // ou null
  "calibration":   {"ok": true, "reason": "", "n_pairs_x": 5, "n_pairs_y": 3},
  "ok": true, "latency_ms": 61.4, "n_points": 312
}
```

O bloco `dimensionless` é sempre preenchido; `physical` é `null` quando `calibration.ok` é falso. Nenhuma exceção é levantada — a regra do contrato §6 vale aqui também.

**Relação com a Decisão A.** Não a revoga: a saída em unidades físicas continua sendo o objetivo, e continua sendo a que o critério 3.9 (PID/IMC) consome. O que muda é que ela deixa de ser condição de existência da resposta. Imagens sem eixos numerados saem do "fora do domínio de aplicabilidade" e entram no "domínio de aplicabilidade parcial, com saída adimensional" — o que é uma hipótese mais honesta e um sistema mais útil.

**Consequência nos critérios.** 2.3, 2.4 e 2.5 continuam válidos, mas passam a ser medidos **sobre o subconjunto em que a calibração declarou sucesso**, e ganham um companheiro: a **cobertura** da calibração (fração de amostras com `ok = True`), reportada por estrato de DPI. Um calibrador que rejeita tudo passaria vacuamente em 2.3 e 2.4 — a cobertura é o que impede isso, e é a mesma lógica de par precisão/cobertura que já rege 2.4 e 2.5 entre si.

### 1.8 Decisão F — extrator clássico como Plano B do risco de GPU e como baseline do estágio A

**Escolha:** implementar um extrator de curva **sem rede neural** — segmentação por cor + rejeição de componentes retilíneas — em `identify/extract_classical.py`, com a mesma assinatura de `predict_mask`. Ele cumpre dois papéis, e nenhum deles é substituir a U-Net.

**Papel 1 — Plano B do maior risco de cronograma.** O §5 lista "Driver NVIDIA não instala" com mitigação "tudo em CPU". Medido: isso custa **20–35 h de parede por rodada de treino**, o que na prática significa uma ou duas tentativas no orçamento inteiro da Parte 2. Um extrator clássico custa **zero** de treino e remove `torch` do caminho crítico. A contingência atual é "treinar devagar"; a nova é "não treinar".

**Papel 2 — baseline que justifica a U-Net.** Sem ele, "usamos uma U-Net" é uma escolha a defender. Com ele, é uma conclusão medida: a U-Net entra no trabalho porque superou o extrator clássico no critério 2.1 (IoU) e, principalmente, no conjunto OOD, onde artefato de JPEG, gradiente de iluminação em figura fotografada e anti-aliasing agressivo quebram a segmentação por cor.

**Como funciona.** A estrutura de um gráfico é rígida, e é isso que dá tração ao método clássico:

1. cor modal da imagem = fundo; agrupamento das cores restantes em modos;
2. componentes conexas de cada modo;
3. **rejeição de retas**: grade, *spines* e distratores são segmentos retos de span completo — a mesma detecção que o critério 1.4e já implementa na suíte de testes (`_spanning_rows`, com a verificação de cobertura de bins);
4. entre os componentes sobreviventes, o que tem maior extensão horizontal dentro da `plot_bbox_px` é a curva;
5. saída `uint8` 0/255, idêntica em formato à de `predict_mask`, alimentando o mesmo `mask_to_polyline`.

O passo 3 é a razão de o método ser viável: o que confunde um limiar de cor ingênuo — grade e linhas de referência — tem assinatura geométrica exata e já foi implementado e validado na Parte 1.

**Contrapartida assumida.** Fragilidade fora da distribuição, e é exatamente por isso que ele é o baseline e não a solução: se ele empatasse com a U-Net no OOD, a U-Net não teria razão de existir no trabalho.

---

## 2. Contrato de dados (define as fronteiras entre os entregáveis)

Cada amostra gerada produz:

```
sample_XXXXX/
  image.png            # resolução, DPI, cores, estilo, textos — todos aleatórios
  mask.png             # uint8, mesma resolução: 255 onde há curva, 0 caso contrário
  meta.json
```

```jsonc
{
  "order": "fopdt" | "second",
  "params": { "K":2.13, "tau":0.87, "theta":0.31, "wn":null, "zeta":null },
  "step_amplitude": 1.0,
  "plot_bbox_px": [x0, y0, x1, y1],          // moldura da área de dados
  "axis_affine":  { "sx":0.0123, "ox":-1.5,  // t = sx*px + ox
                    "sy":-0.0087, "oy":4.2 },// y = sy*py + oy
  "ticks": { "x": [[px,val],...], "y": [[px,val],...] },
  "series": { "t": [...], "y": [...] },      // 512 amostras, verdade de terra
  "render": { "dpi":..., "size_px":[W,H], "has_grid":true, "has_legend":false, ... }
}
```

O bloco `render` **nunca** entra em nenhum modelo — existe só para estratificar as métricas (ex.: "IoU cai 4 pontos quando há grade").

**Regra anti-vazamento (invariante testável):** nenhum elemento visual pode ser estatisticamente correlacionado ao rótulo. Cor, espessura, estilo de linha, presença de grade, de legenda, de título e de anotações são sorteados **independentemente** de (`order`, K, τ, θ, ωₙ, ζ). As linhas auxiliares do `img.py` atual (`axhline(K)`, `axvline(θ+τ)`) são removidas; podem retornar como **distratores** — linhas de referência em posições aleatórias descorrelacionadas dos parâmetros — o que torna o modelo robusto a figuras reais que possuem marcações.

---

## 3. Os três entregáveis

Cada parte é auto-contida, tem critério de aceitação numérico e um teste automatizado (`pytest`) que decide aprovação sem julgamento subjetivo.

---

### PARTE 1 — Gerador saneado + prova de solubilidade com oráculo (Semana 1)

**Ideia central:** antes de treinar qualquer rede, provar que o problema é solúvel e que os rótulos estão corretos, substituindo os estágios A e B por seus valores de verdade de terra ("oráculo"). Se o pipeline não fecha com informação perfeita, nenhuma rede o fará.

**Escopo:**
1. Reescrever `img.py` → `dataset/generator.py` com randomização completa e sem vazamento:
   - resolução 240×180 a 1600×1200 px, DPI 60–200;
   - paleta da curva e do fundo aleatórias (inclui escala de cinza e fundo escuro);
   - espessura 0,8–3,0 px, estilos `-`, `--`, `-.`, marcadores esparsos;
   - grade on/off, ticks maiores/menores, molduras (`spines`) variáveis;
   - título, rótulos de eixo, legenda e anotações **com texto aleatório e semanticamente vazio**, presentes em ~50 % das amostras;
   - ruído aditivo gaussiano e de quantização sobre y(t), SNR 20–60 dB;
   - janela temporal variável (0,5× a 6× a constante de tempo dominante) — inclui curvas **truncadas antes do regime permanente**, que é o caso difícil e realista;
   - degrau iniciando em t₀ ≠ 0;
   - 1 a 3 linhas de referência distratoras em posições descorrelacionadas.
2. Emitir máscara e `meta.json` conforme §2.
3. Implementar `identify/classical.py`: ajuste por `least_squares` das duas estruturas + critério AIC, e os baselines clássicos (método da tangente, Smith, Sundaresan–Krishnaswamy) para comparação na monografia.
4. Implementar o teste de vazamento (§2).

**Volumes:** 6.000 imagens para os estágios A/B (renderizadas em disco, ~2 GB), divididas em 4.200 treino / 900 validação / 900 teste com seeds disjuntos. A geração de séries em memória sem limite deixou de ser necessária com a remoção do estágio C (§1.3).

**Critérios de aceitação (mensuráveis):**

| # | Critério | Alvo |
|---|---|---|
| 1.1 | Pipeline-oráculo (máscara e calibração verdadeiras → estágio D) recupera os parâmetros | MAPE < 1 % em K, τ, θ, ωₙ, ζ, sem ruído |
| 1.2 | Idem, com ruído de 20 dB | MAPE < 5 % |
| 1.3 | Teste de vazamento: um GBM treinado **apenas** com os atributos de `render` (cor, dpi, grade, espessura…) prediz `order` | acurácia ≤ 55 % (≈ acaso) |
| 1.4 | Teste de vazamento paramétrico: correlação de Spearman entre cada atributo de `render` e cada parâmetro | \|ρ\| < 0,05 |
| 1.5 | Consistência da máscara: reprojetar `mask.png` pela `axis_affine` reproduz `series` | RMSE < 1,5 px |
| 1.6 | Reprodutibilidade: mesma seed → hash de bytes idêntico | igualdade exata |
| 1.7 | Custo de geração | 6.000 imagens em < 30 min nos 16 threads |

**Teste automatizado:** `pytest tests/test_part1.py` — cobre 1.1 a 1.7 sobre uma amostra de 300 exemplos.

**Por que este entregável primeiro:** ele é o único que, se falhar, invalida todo o resto — e não depende de GPU, de driver, nem de PyTorch. Mesmo no pior cenário de infraestrutura, a Semana 1 entrega resultado defendível (gerador + comparação de métodos clássicos de identificação).

---

### PARTE 2 — Estágio A (extração da curva) + Estágio B (calibração dos eixos) (Semana 2)

**Ideia central:** substituir o oráculo por percepção real e medir **quanto** o pipeline degrada. A métrica de sucesso é relativa ao oráculo da Parte 1, não absoluta.

**Estágio A — segmentação da curva.**
- Arquitetura: U-Net compacta (~1,2 M parâmetros, 4 níveis, base 16 canais). Justificativa: segmentação binária de estrutura fina e alongada com pouquíssima semântica; as *skip connections* preservam a localização em nível de pixel, que é exatamente o que determina a precisão de y(t). Uma ResNet-18 classificadora seria a ferramenta errada — ela descarta resolução espacial por *pooling* agressivo, que é justamente a informação que queremos.
- Entrada: imagem redimensionada para 512×512 com preenchimento que **preserva a razão de aspecto** (a distorção anisotrópica alteraria a geometria da curva); saída reprojetada para a resolução original.
- Perda: Dice + BCE (a classe positiva ocupa < 2 % dos pixels; BCE puro colapsa para "tudo fundo").
- Pós-processamento: maior componente conexa → esqueletonização → para cada coluna x, mediana das linhas y ativas → polilinha. Vive em `identify/polyline.py`, separado da rede: é determinístico, não importa `torch` e é testável contra a máscara verdadeira sem modelo nenhum.
- **Baseline obrigatório (§1.8):** `identify/extract_classical.py` — segmentação por cor + rejeição de componentes retilíneas, mesma assinatura, zero treino. É o Plano B do risco de GPU e o baseline que justifica a U-Net (critério 2.10).

**Estágio B — calibração dos eixos (determinístico + OCR opcional, §1.7).**
- Detecção da moldura da área de dados por projeção de gradientes (linhas retas longas) e dos ticks por picos na projeção sobre a moldura;
- recorte dos rótulos numéricos vizinhos a cada tick → OCR (Tesseract, com EasyOCR como alternativa);
- ajuste **robusto** da transformação afim px→dados por RANSAC sobre os pares (pixel do tick, valor lido), o que descarta automaticamente leituras erradas do OCR;
- **teste de consistência interna**: os ticks devem ser equiespaçados em valor e em pixel; violação → `Calibration.ok = False` com `reason` preenchido, em vez de produzir um número silenciosamente errado;
- **degradação graciosa (§1.7):** `ok = False` **não** aborta a amostra. A saída adimensional (`order`, ζ, ωₙ·T, θ/T, K/y_faixa) é produzida de qualquer forma, porque não depende da calibração; só o bloco `physical` fica `null`.

Justificativa da escolha determinística: a estrutura de um gráfico cartesiano é rígida e conhecida; treinar uma rede para redescobri-la gastaria semana de orçamento sem ganho. O componente aprendido fica onde há ambiguidade real (a curva e os parâmetros). Além disso, o RANSAC transforma o OCR — o elo mais frágil — em um componente tolerante a falhas: bastam 2 ticks corretos por eixo.

**Critérios de aceitação:**

| # | Critério | Alvo |
|---|---|---|
| 2.1 | IoU da máscara no conjunto de teste | ≥ 0,85 (mediana) |
| 2.2 | Erro da polilinha extraída vs. verdade | RMSE ≤ 2 px (p95 ≤ 5 px) |
| 2.3 | Erro relativo das escalas `sx`, `sy`, **no subconjunto `ok = True`** | < 1 % em ≥ 95 % do subconjunto |
| 2.4 | Taxa de rejeição por consistência (falso alarme) | < 5 % |
| 2.5 | Rejeições corretas: quando rejeita, o erro de escala seria de fato > 5 % | ≥ 90 % das rejeições, com n ≥ 5 |
| 2.6 | **Degradação end-to-end** vs. oráculo da Parte 1 (mesmas amostras, estágio D idêntico) | ΔMAPE ≤ 3 pontos percentuais |
| 2.7 | Estratificação: IoU por presença de grade / legenda / fundo escuro | nenhum estrato < 0,75 |
| 2.8 | Tempo de inferência por imagem | < 500 ms |
| **2.9** | **Cobertura da calibração** (§1.7): fração de amostras com `ok = True`, estratificada por DPI | ≥ 90 % global; reportada por estrato, sem alvo por estrato |
| **2.10** | **U-Net × extrator clássico** (§1.8): IoU mediana das duas no mesmo conjunto de teste | sem alvo — é o resultado que justifica (ou não) a U-Net |
| **2.11** | Saída adimensional produzida mesmo com `ok = False` | 100 % das amostras com `dimensionless` preenchido e nenhuma exceção levantada |

**Por que 2.9 existe.** Sem ele, 2.3 e 2.4 são vacuamente satisfeitos por um calibrador que rejeita quase tudo: zero falsos alarmes e erro de escala baixíssimo no punhado que sobra. 2.9 é o lado "cobertura" do par precisão/cobertura, e usa a mesma lógica que já liga 2.4 a 2.5.

**Por que 2.11 existe.** A Decisão E só vale se a degradação graciosa for testada. Um `ok = False` que na prática produz `None` em tudo é a falha silenciosa que a §1.7 existe para evitar.

**Teste automatizado:** `pytest tests/test_part2.py`, mais um relatório estratificado `reports/part2_strata.md` gerado automaticamente.

---

### PARTE 3 — Validação fora da distribuição, utilidade para controle e baseline fim-a-fim (Semana 3)

> **Reescrita em 22/08/2026.** A versão original desta parte era "Estágio C (estimador neural) + D (refinamento) + validação OOD". O estágio C foi medido e removido (§1.3); o estágio D já está implementado e validado desde a Parte 1. O que resta na Semana 3 é o que sempre foi o argumento central do trabalho — **a validação fora da distribuição** — mais dois experimentos que convertem decisões de projeto em medições.

**Ideia central:** o pipeline completo já existe ao entrar nesta parte (A e B da Parte 2, D da Parte 1). A Semana 3 não constrói componente novo: ela **mede** o sistema onde a medição é difícil e onde as afirmações da monografia são de fato testadas.

**Validação fora da distribuição (OOD) — o teste que sustenta a afirmação central do trabalho.**
Conjunto separado, **nunca visto**, com ~60 imagens de origens distintas do gerador:
- capturas de tela de respostas ao degrau do MATLAB/Simulink e do Python Control (renderizador diferente, tipografia diferente);
- figuras de livros-texto (Ogata, Nise) fotografadas ou digitalizadas — com ruído de JPEG, leve perspectiva e distorção;
- gráficos gerados em planilha (Excel/LibreOffice), cujo estilo visual não aparece no treino;
- ~10 curvas de **plantas reais** medidas (se houver acesso a bancada no LSI/DELT; caso contrário, simulação de ordem superior — ex.: 4ª ordem — para medir a degradação graciosa quando a hipótese estrutural é falsa).

Justificativa: o requisito "independente da resolução e dos elementos da imagem" só é comprovável em dados que não vieram do próprio gerador. Sem esse conjunto, o trabalho demonstra apenas que a rede aprendeu a inverter o gerador — que é uma afirmação muito mais fraca.

O conjunto OOD é também onde a Decisão E (§1.7) e a Decisão F (§1.8) são de fato testadas: é ali que o OCR tem mais chance de falhar (e a saída adimensional de provar sua utilidade), e ali que o extrator clássico deve quebrar (e a U-Net de provar a sua).

**Experimento de utilidade para controle.** Sintonizar um PID por IMC sobre o modelo identificado e sobre o modelo verdadeiro, comparar as duas malhas fechadas em simulação. É o que fecha o círculo com o título do curso: mostra que o erro paramétrico residual — inclusive nos casos de ζ alto em que §1.3 mostrou que a estrutura escolhida pode estar "errada" — é irrelevante para a finalidade de controle.

**Baseline fim-a-fim.** Treinar a CNN 2D que a Decisão B (§1.2) rejeitou, sobre o mesmo dataset, e comparar. Plano de execução completo, com prós, contras e passos, em **`PLANO_CNN_FIM_A_FIM.md`**. O objetivo não é adotá-la: é converter a §1.2 de argumento em medição, e — mais importante — **testar empiricamente a afirmação de vazamento**, medindo se a CNN fim-a-fim generaliza pior no OOD do que o pipeline em estágios, que é a assinatura observável de um atalho aprendido.

**Critérios de aceitação:**

| # | Critério | Alvo |
|---|---|---|
| 3.1 | Acurácia de estrutura por AIC, estrato **ζ < 1,6** (onde a distinção é observável) | ≥ 95 % — medido em 100 % (217/217) na Parte 1 |
| 3.2 | Acurácia de estrutura, estratificada por ζ, **incluindo** ζ ≥ 1,6 e FOPDT | sem alvo — é resultado, com o limite de informação declarado (§1.3) |
| 3.3 | Custo do erro de estrutura: NRMSE de reconstrução nas amostras em que o AIC errou | ≤ 0,02 (medido: máx 5,8e-03) |
| 3.4 | MAPE de K, τ, θ (FOPDT) após o estágio D, teste sintético | ≤ 5 % |
| 3.5 | MAPE de K, ωₙ, ζ (2ª ordem, ζ < 1,6) após o estágio D | ≤ 8 % (ζ), ≤ 5 % (K, ωₙ) |
| 3.6 | **NRMSE de reconstrução da curva** (métrica primária), teste sintético | ≤ 3 % |
| 3.7 | NRMSE de reconstrução, conjunto **OOD** | ≤ 8 % |
| 3.8 | Comparação com baselines clássicos (tangente, Smith, S–K) sobre a mesma série extraída | superar em MAPE em ≥ 2 dos 3 parâmetros de FOPDT |
| 3.9 | **CNN fim-a-fim × pipeline em estágios** (`PLANO_CNN_FIM_A_FIM.md`) | sem alvo no sintético; **no OOD, o pipeline deve degradar menos** — é o teste da §1.2 |
| 3.10 | Utilidade para controle: PID por IMC sobre o modelo identificado vs. sobre o verdadeiro | sobressinal < 10 % e tempo de acomodação < 15 % de diferença em malha fechada simulada |
| 3.11 | Latência total imagem → parâmetros | < 2 s em CPU |
| 3.12 | Taxa de convergência de `identify` sobre séries **extraídas** (não do oráculo) | ≥ 99 % — é o **gatilho da §1.3**: abaixo disso, o estágio C volta à mesa |

**Teste automatizado:** `pytest tests/test_part3.py` + `reports/final_report.md` com todas as tabelas prontas para a monografia.

Dois critérios carregam o argumento da monografia. O **3.10** fecha o círculo com o curso: o erro residual não importa para controlar. O **3.9** fecha o círculo com a metodologia: a arquitetura escolhida não foi preferência, foi medição. E o **3.12** é a condição de honestidade da Decisão C — o único número que, se falhar, obriga a reabrir a decisão de remover o estágio C.

---

## 4. Cronograma (3 semanas)

| Dia | Atividade | Marco |
|---|---|---|
| **1** | Ambiente: venv com **Python 3.11**, PyTorch, OpenCV, Tesseract. Driver NVIDIA (RPM Fusion `akmod-nvidia`) — timebox de 3 h | `torch.cuda.is_available()` ou decisão formal de ir por CPU |
| **2–3** | `dataset/generator.py` com randomização e máscara | Parte 1, itens 1–2 |
| **4** | `identify/classical.py` (least_squares + AIC + baselines) | Parte 1, item 3 |
| **5** | Testes de vazamento e de oráculo | ✅ **Entregável 1** (critérios 1.1–1.7) |
| **6–7** | Geração do conjunto de 6.000 imagens; folga | dataset em disco |
| **8** | `extract_classical.py` (§1.8) — extrator sem rede, e baseline do estágio A | máscara 0/255 com a mesma assinatura de `predict_mask` |
| **9** | U-Net + treino do estágio A | IoU ≥ 0,85 |
| **10–11** | Calibração de eixos: moldura, ticks, OCR opcional (§1.7), RANSAC, consistência | escalas < 1 % de erro e cobertura ≥ 90 % |
| **12** | Integração A+B com o estágio D; medição da degradação | ✅ **Entregável 2** (critérios 2.1–2.11) |
| **13–14** | Folga / correção do estrato mais fraco | — |
| **15–16** | Baseline CNN fim-a-fim (`PLANO_CNN_FIM_A_FIM.md`) | critério 3.9 medido |
| **17** | Anotação do conjunto OOD (coleta vem correndo desde o Dia 1) | 60 imagens externas |
| **18** | Avaliação OOD completa e estratificações | 3.7, 3.9 no OOD |
| **19** | Experimento de PID/IMC e fechamento do relatório | ✅ **Entregável 3** (critérios 3.1–3.12) |
| **20–21** | Redação: metodologia, resultados, discussão da ambiguidade estrutural | capítulos rascunhados |

Os dias 6–7 e 13–14 são folga deliberada. Um cronograma de 3 semanas sem folga é um cronograma de 3 semanas que não é cumprido.

**Efeito da revisão de 22/08/2026 no cronograma.** A remoção do estágio C (§1.3) libera os dias 15–16, antes ocupados pelo gerador de séries e pelo treino do estimador 1D. Eles foram realocados para o baseline fim-a-fim (critério 3.9) — que é medição, não construção de componente — e a anotação do OOD subiu do Dia 18 para o 17, dando um dia inteiro a mais para a avaliação que sustenta o argumento central. O Dia 8 ganhou o extrator clássico (§1.8), que é também a contingência do risco de GPU.

---

## 5. Riscos e mitigações

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| **Driver NVIDIA não instala** (Fedora 43 + Secure Boot exige assinatura de módulo MOK) | Média | **Baixo** (era Médio) | Timebox de 3 h no Dia 1. Plano B **medido**: treinar a U-Net em CPU custa 20–35 h de parede por rodada, o que na prática dá uma ou duas tentativas no orçamento — insuficiente. Plano B real (§1.8): `extract_classical.py`, **zero treino**, remove `torch` do caminho crítico. A U-Net passa a ser melhoria mensurável (critério 2.10), não pré-requisito. |
| **PyTorch sem wheel para Python 3.14** | Alta | Baixo | venv com `python3.11`, já presente no sistema |
| **RAM insuficiente** (só ~6 GB livres) | Média | Médio | Carregamento por lote via `DataLoader` com memória mapeada; nunca carregar o dataset inteiro; lote 8–16 |
| **OCR falha em fontes pequenas / DPI baixo** | Alta | **Baixo** (era Médio) | RANSAC tolera erros individuais (bastam 2 ticks corretos); consistência rejeita em vez de errar em silêncio. **E, pela Decisão E (§1.7), a falha deixou de ser fatal:** a saída adimensional (`order`, ζ, ωₙ·T, θ/T) não depende de OCR e é sempre produzida. Cobertura medida pelo critério 2.9 |
| **Ambiguidade FOPDT × 2ª ordem superamortecida derruba a acurácia** | **Confirmada, medida** | Baixo | Tratada por desenho (§1.5) e **quantificada** (§1.3): 100 % de acerto em ζ < 1,6; 42,4 % em ζ ≥ 2,2, com custo máximo de 5,8e-03 de NRMSE. Virou resultado com limite de informação declarado, e é a base para remover o estágio C |
| **Conjunto OOD não fica pronto a tempo** | Média | Alto (é o argumento central) | Montá-lo no Dia 18 é tarde: coletar capturas de tela **em paralelo, a partir do Dia 1**, em pastas soltas; anotação só no Dia 18 |
| **Escopo cresce** (zeros, ordens superiores, múltiplas curvas) | Média | Alto | Congelado em §1.4; qualquer extensão vai para "trabalhos futuros" |
| **Cronograma estoura** (a Parte 1 sozinha consumiu o orçamento das 3 semanas) | **Alta, já materializado** | Alto | Revisão de 22/08/2026: um modelo treinado em vez de dois (§1.3), GPU fora do caminho crítico (§1.8), OCR fora do caminho crítico (§1.7). O caminho crítico da Parte 2 passou a não ter nenhuma dependência de infraestrutura não instalada |

---

## 6. Estrutura de diretórios proposta

```
TCC-2/
├── dataset/
│   ├── generator.py            # renderização + máscara + meta.json (Parte 1)
│   └── randomize.py            # sorteio de estilo visual, isolado e auditável
├── identify/
│   ├── classical.py            # least_squares, AIC, baselines (Parte 1) — o estágio D
│   ├── extract.py              # U-Net (Parte 2)
│   ├── extract_classical.py    # extrator sem rede: Plano B e baseline (§1.8)
│   ├── polyline.py             # máscara → polilinha → série (determinístico)
│   ├── calibrate.py            # moldura, ticks, OCR opcional, RANSAC (§1.7)
│   └── pipeline.py             # A→B→D, com modo oráculo comutável
├── tests/
│   ├── test_part1.py  test_leakage.py
│   ├── part2/                  # critérios 2.1–2.11
│   └── test_part3.py           # critérios 3.1–3.12
├── e2e/                        # baseline fim-a-fim (PLANO_CNN_FIM_A_FIM.md)
├── reports/                # tabelas e figuras geradas, prontas para a monografia
├── data/                   # não versionado
└── ood/                    # conjunto fora da distribuição, coletado à mão
```

O modo oráculo comutável em `pipeline.py` é um detalhe importante: é ele que permite, a qualquer momento, medir a degradação de cada estágio isoladamente sem reescrever código de avaliação.

Duas ausências são deliberadas. Não há `estimator.py` nem `dataset/series.py`: os dois existiam para o estágio C, removido pela §1.3. E `e2e/` fica **fora** do pacote `identify/` de propósito — é experimento comparativo, não componente do sistema, e nada em `identify/` deve poder importá-lo.

---

## 7. O que muda nos relatórios

Para o Ciclo 4, sugere-se registrar com transparência:

1. que a auditoria do gerador revelou vazamento de rótulo, invalidando as métricas dos Ciclos 1 e 2 — apresentado como achado metodológico;
2. que a parametrização anunciada no Ciclo 3 foi de fato implementada neste ciclo, junto com a máscara de segmentação;
3. que o dataset de 50.000 imagens foi **substituído** por 6.000 imagens + geração ilimitada de séries, com a justificativa de eficiência da §1.2 — uma decisão de projeto melhor fundamentada que o número original;
4. que o escopo passou a incluir a calibração de eixos, sem a qual a saída não teria unidade física (§1.1);
5. **que o estimador neural intermediário (estágio C) foi projetado, medido e removido** — com a estratificação por ζ que mostra o AIC em 100 % onde a distinção é observável e o custo limitado do erro onde não é (§1.3). Isto é o achado metodológico do Ciclo 5, e é da mesma natureza do vazamento de rótulo do Ciclo 4: uma decisão de arquitetura derrubada por medição própria. Um TCC que remove um componente porque mediu que ele não fazia nada é mais forte que um que o mantém por simetria de arquitetura;
6. que o OCR deixou de ser estrutural (§1.7) e que a GPU saiu do caminho crítico (§1.8) — duas reduções de risco que tornam o cronograma restante executável no hardware que de fato existe.

Nenhum desses pontos enfraquece o trabalho. Um TCC que documenta por que uma abordagem inicial falhou e como isso mudou o projeto é, em banca, mais forte que um que só relata sucessos.
```
