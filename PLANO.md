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

### 1.1 Decisão A — Saída em unidades físicas, com calibração de eixos explícita

**Escolha:** o sistema devolve K, τ, θ, ωₙ, ζ em unidades físicas, lendo a escala dos eixos da própria imagem.

**Justificativa.** A partir dos pixels da curva isoladamente, os parâmetros dimensionais **não são identificáveis**: a mesma geometria de traçado corresponde a (K = 2, τ = 1 s) e a (K = 200, τ = 1 min). Apenas grandezas adimensionais — a ordem do sistema, ζ, e as razões τ/T_janela, θ/T_janela, K/y_máx — são invariantes de escala. Portanto a leitura dos eixos não é um acessório do sistema: é a condição necessária para que a saída tenha significado físico. Imagens sem eixos numerados estão, por construção, fora do domínio de aplicabilidade — e isso deve ser declarado como hipótese do trabalho, não como limitação envergonhada.

**Consequência arquitetural:** o requisito "funcionar com imagens sem texto" é reinterpretado como **"invariante a texto irrelevante"** (títulos, legendas, anotações, marca d'água), mantendo a dependência do **texto relevante** (rótulos numéricos dos ticks).

### 1.2 Decisão B — Pipeline em dois estágios, não CNN fim-a-fim

**Escolha:** `imagem → [A] extração da curva → [B] calibração dos eixos → série y(t) em unidades físicas → [C] estimador neural 1D → [D] refinamento por mínimos quadrados`.

**Justificativa em quatro pontos:**

1. **Invariância por construção, não por esperança.** Numa CNN 2D fim-a-fim, a invariância a resolução, cor, grade e legenda só pode ser induzida por *data augmentation* — e permanece uma propriedade empírica, não garantida, que falha silenciosamente fora da distribuição. No pipeline em dois estágios, a série y(t) normalizada é a **mesma representação** quer a figura tenha 300×200 px em preto e branco, quer tenha 1920×1080 px colorida com legenda. A invariância vira uma propriedade estrutural, demonstrável.

2. **Descolamento entre custo de renderização e volume de treino.** O estimador do estágio C consome *séries temporais*, não imagens. Séries podem ser geradas analiticamente aos milhões, em memória, a custo desprezível — enquanto renderizar 50.000 PNGs com matplotlib leva horas e ocupa disco. O desenho em dois estágios permite treinar o componente mais crítico com ordens de magnitude mais dados que a abordagem fim-a-fim, no mesmo orçamento de tempo.

3. **Atribuição de erro.** Com fim-a-fim, um MAPE de 15 % em τ é um número sem diagnóstico. Aqui, cada estágio tem métrica própria e é possível afirmar, por exemplo: "a segmentação contribui com 2 % do erro, a calibração com 1 %, o estimador com 12 %" — que é o tipo de análise que sustenta uma defesa.

4. **Viabilidade no hardware disponível.** Um estimador 1D (CNN dilatada ou GRU sobre 512 amostras) treina em minutos, inclusive em CPU. Isso permite dezenas de experimentos por dia dentro das 3 semanas, contra poucos ciclos de uma ResNet-18 sobre 50k imagens numa GPU de 6 GB cujo driver ainda nem está instalado.

**Contrapartida assumida:** são dois modelos a manter em vez de um, e o gerador precisa emitir a máscara da curva como rótulo adicional (custo marginal ~zero, já que quem desenha a curva sabe onde ela está).

### 1.3 Decisão C — Estágio D: refinamento por mínimos quadrados sobre a predição da rede

**Escolha:** a saída da rede não é a resposta final; ela é o **chute inicial e a seleção de estrutura** para um ajuste `scipy.optimize.least_squares` do modelo paramétrico à série extraída.

**Justificativa.** Identificação de sistemas por otimização não-linear é uma técnica consolidada (Ljung, *System Identification*) e atinge erro paramétrico sub-1 % — quando parte de uma inicialização boa e da estrutura correta. Seus dois modos de falha clássicos são exatamente (i) escolha de estrutura e (ii) mínimos locais em inicialização ruim. Uma rede neural é excelente em ambos e medíocre em precisão numérica fina; o otimizador é o oposto. Compor os dois entrega o melhor dos dois regimes e — mais importante para o TCC — **define com precisão qual é a contribuição da rede neural**: ela resolve o problema de estrutura e inicialização, que é onde os métodos clássicos falham. Sem esse enquadramento, a pergunta de banca "por que não usar só mínimos quadrados?" não tem resposta boa.

### 1.4 Decisão D — Cobertura: FOPDT + segunda ordem canônica

- 1ª ordem com tempo morto: `G(s) = K·e^{-θs}/(τs + 1)` — 3 parâmetros. É o modelo de referência da sintonia PID industrial (Ziegler–Nichols, Cohen–Coon, IMC), o que ancora a relevância prática do trabalho.
- 2ª ordem canônica: `G(s) = Kωₙ²/(s² + 2ζωₙs + ωₙ²)` — 3 parâmetros, cobrindo ζ subamortecido, crítico e superamortecido.

Fora de escopo (declarado): zeros, fase não-mínima, integradores, ordens superiores.

### 1.5 A ambiguidade estrutural que precisa ser tratada de frente

Um sistema de 2ª ordem **superamortecido** (ζ > 1) produz uma resposta ao degrau visualmente quase indistinguível de um FOPDT. Isso não é deficiência do modelo — é **não-identificabilidade prática** do problema: as duas famílias geram curvas cuja diferença é menor que o ruído de extração.

Tratamento adotado:
- não forçar decisão binária dura: o estágio C emite probabilidade de classe, e o estágio D ajusta **as duas** estruturas quando a probabilidade fica na faixa incerta (0,3–0,7), escolhendo pelo critério de resíduo penalizado (AIC);
- a **métrica primária do trabalho passa a ser o erro de reconstrução da curva** (NRMSE entre a resposta do modelo identificado e a série extraída), não a acurácia de classificação. Justificativa: para uso em controle, o que importa é que o modelo reproduza a dinâmica, não que ele carregue o rótulo "certo" numa região onde o rótulo é ambíguo;
- a acurácia de classificação continua reportada, mas **estratificada por ζ**, evidenciando que o erro se concentra em ζ ≈ 1 — o que é um resultado, não uma falha.

### 1.6 Hipóteses explícitas (a declarar na monografia)

1. O gráfico é uma resposta a um degrau de **amplitude conhecida** (unitária, salvo anotação). Sem isso, apenas o produto K·A é identificável.
2. Os eixos são lineares (não logarítmicos) e possuem ao menos 2 ticks rotulados por eixo.
3. A curva é única na figura (sem múltiplas séries sobrepostas) — múltiplas curvas ficam como teste de estresse, não como requisito.
4. O instante do degrau pode ser ≠ 0 e é absorvido em θ.

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

**Volumes:** 6.000 imagens para os estágios A/B (renderizadas em disco, ~2 GB) e gerador de séries em memória, sem limite, para o estágio C.

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
- Pós-processamento: maior componente conexa → esqueletonização → para cada coluna x, mediana das linhas y ativas → polilinha.

**Estágio B — calibração dos eixos (determinístico + OCR).**
- Detecção da moldura da área de dados por projeção de gradientes (linhas retas longas) e dos ticks por picos na projeção sobre a moldura;
- recorte dos rótulos numéricos vizinhos a cada tick → OCR (Tesseract, com EasyOCR como alternativa);
- ajuste **robusto** da transformação afim px→dados por RANSAC sobre os pares (pixel do tick, valor lido), o que descarta automaticamente leituras erradas do OCR;
- **teste de consistência interna**: os ticks devem ser equiespaçados em valor e em pixel; violação → amostra rejeitada com bandeira `calibration_failed`, em vez de produzir um número silenciosamente errado.

Justificativa da escolha determinística: a estrutura de um gráfico cartesiano é rígida e conhecida; treinar uma rede para redescobri-la gastaria semana de orçamento sem ganho. O componente aprendido fica onde há ambiguidade real (a curva e os parâmetros). Além disso, o RANSAC transforma o OCR — o elo mais frágil — em um componente tolerante a falhas: bastam 2 ticks corretos por eixo.

**Critérios de aceitação:**

| # | Critério | Alvo |
|---|---|---|
| 2.1 | IoU da máscara no conjunto de teste | ≥ 0,85 (mediana) |
| 2.2 | Erro da polilinha extraída vs. verdade | RMSE ≤ 2 px (p95 ≤ 5 px) |
| 2.3 | Erro relativo das escalas `sx`, `sy` | < 1 % em ≥ 95 % das amostras |
| 2.4 | Taxa de rejeição por consistência (falso alarme) | < 5 % |
| 2.5 | Rejeições corretas: quando rejeita, o erro de escala seria de fato > 5 % | ≥ 90 % das rejeições |
| 2.6 | **Degradação end-to-end** vs. oráculo da Parte 1 (mesmas amostras, estágio D idêntico) | ΔMAPE ≤ 3 pontos percentuais |
| 2.7 | Estratificação: IoU por presença de grade / legenda / fundo escuro | nenhum estrato < 0,75 |
| 2.8 | Tempo de inferência por imagem | < 500 ms |

**Teste automatizado:** `pytest tests/test_part2.py`, mais um relatório estratificado `reports/part2_strata.md` gerado automaticamente.

---

### PARTE 3 — Estágio C (estimador neural) + D (refinamento) + validação OOD (Semana 3)

**Estágio C — estimador de estrutura e parâmetros.**
- Entrada canônica: série reamostrada em **512 pontos uniformes** sobre a janela do gráfico, normalizada por `t ← (t−t_ini)/T_janela` e `y ← y/y_escala`. Assim a rede opera num espaço **exatamente adimensional** — a invariância de escala é estrutural, não aprendida.
- Arquitetura: CNN 1D dilatada (dilatações 1,2,4,8,16,32 — campo receptivo cobrindo as 512 amostras) + *pooling* estatístico + três cabeças:
  - classificação de ordem (2 logits);
  - regressão FOPDT: (log K, log τ/T, θ/T);
  - regressão 2ª ordem: (log K, log ωₙT, logit ζ/3).
  Justificativa da parametrização logarítmica: os parâmetros variam por ordens de grandeza e a métrica de interesse é o **erro relativo**; regredir em log torna o MSE no espaço de treino equivalente ao erro relativo no espaço físico, sem que amostras de ganho alto dominem o gradiente.
- Perda: `CE(ordem) + λ·MSE(cabeça da classe verdadeira)`, com a cabeça da classe errada mascarada (não faz sentido penalizar ωₙ de um sistema de 1ª ordem). λ calibrado para igualar as magnitudes dos gradientes.
- **Treino com degradação simulada:** as séries de treino recebem exatamente as perturbações que os estágios A e B introduzem — jitter de extração (~2 px convertidos em unidades), erro de escala de ±1 %, truncamento, falhas de coluna. Isso alinha a distribuição de treino à de operação sem precisar renderizar imagens, e é o mecanismo que substitui as 50.000 imagens do plano original.
- Volume: 200.000 séries geradas em memória (~10 min de CPU), sem custo de disco.

**Estágio D — refinamento.** `least_squares` (Levenberg–Marquardt) partindo da predição da rede; quando `p(ordem) ∈ [0,3; 0,7]`, ajusta as duas estruturas e escolhe por AIC.

**Validação fora da distribuição (OOD) — o teste que sustenta a afirmação central do trabalho.**
Conjunto separado, **nunca visto**, com ~60 imagens de origens distintas do gerador:
- capturas de tela de respostas ao degrau do MATLAB/Simulink e do Python Control (renderizador diferente, tipografia diferente);
- figuras de livros-texto (Ogata, Nise) fotografadas ou digitalizadas — com ruído de JPEG, leve perspectiva e distorção;
- gráficos gerados em planilha (Excel/LibreOffice), cujo estilo visual não aparece no treino;
- ~10 curvas de **plantas reais** medidas (se houver acesso a bancada no LSI/DELT; caso contrário, simulação de ordem superior — ex.: 4ª ordem — para medir a degradação graciosa quando a hipótese estrutural é falsa).

Justificativa: o requisito "independente da resolução e dos elementos da imagem" só é comprovável em dados que não vieram do próprio gerador. Sem esse conjunto, o trabalho demonstra apenas que a rede aprendeu a inverter o gerador — que é uma afirmação muito mais fraca.

**Critérios de aceitação:**

| # | Critério | Alvo |
|---|---|---|
| 3.1 | Acurácia de ordem, excluindo a faixa ambígua ζ ∈ [0,8; 1,3] | ≥ 95 % |
| 3.2 | Acurácia de ordem, global (com a faixa ambígua) | reportada e estratificada por ζ (sem alvo — é resultado) |
| 3.3 | MAPE de K, τ, θ (FOPDT) após estágio D, teste sintético | ≤ 5 % |
| 3.4 | MAPE de K, ωₙ, ζ (2ª ordem) após estágio D | ≤ 8 % (ζ), ≤ 5 % (K, ωₙ) |
| 3.5 | **NRMSE de reconstrução da curva** (métrica primária), teste sintético | ≤ 3 % |
| 3.6 | NRMSE de reconstrução, conjunto **OOD** | ≤ 8 % |
| 3.7 | Ganho do estágio D sobre a rede pura | redução ≥ 40 % no MAPE |
| 3.8 | Comparação com baselines clássicos (tangente, Smith, S–K) sobre a mesma série extraída | superar em MAPE em ≥ 2 dos 3 parâmetros de FOPDT |
| 3.9 | Utilidade para controle: PID sintonizado por IMC sobre o modelo identificado vs. sobre o modelo verdadeiro | diferença de sobressinal < 10 % e de tempo de acomodação < 15 % em malha fechada simulada |
| 3.10 | Latência total imagem → parâmetros | < 2 s em CPU |

**Teste automatizado:** `pytest tests/test_part3.py` + `reports/final_report.md` com todas as tabelas prontas para a monografia.

O critério **3.9** é o que fecha o círculo com o título do curso: mostra que o erro paramétrico residual é irrelevante para a finalidade de controle — o argumento mais forte que este trabalho pode apresentar em banca.

---

## 4. Cronograma (3 semanas)

| Dia | Atividade | Marco |
|---|---|---|
| **1** | Ambiente: venv com **Python 3.11**, PyTorch, OpenCV, Tesseract. Driver NVIDIA (RPM Fusion `akmod-nvidia`) — timebox de 3 h | `torch.cuda.is_available()` ou decisão formal de ir por CPU |
| **2–3** | `dataset/generator.py` com randomização e máscara | Parte 1, itens 1–2 |
| **4** | `identify/classical.py` (least_squares + AIC + baselines) | Parte 1, item 3 |
| **5** | Testes de vazamento e de oráculo | ✅ **Entregável 1** (critérios 1.1–1.7) |
| **6–7** | Geração do conjunto de 6.000 imagens; folga | dataset em disco |
| **8–9** | U-Net + treino do estágio A | IoU ≥ 0,85 |
| **10–11** | Calibração de eixos: moldura, ticks, OCR, RANSAC, consistência | escalas < 1 % de erro |
| **12** | Integração A+B com o estágio D; medição da degradação | ✅ **Entregável 2** (critérios 2.1–2.8) |
| **13–14** | Folga / correção do estrato mais fraco | — |
| **15–16** | Gerador de séries + estimador 1D; treino | MAPE da rede pura |
| **17** | Estágio D + política AIC na faixa ambígua | ganho ≥ 40 % |
| **18** | Montagem e anotação do conjunto OOD | 60 imagens externas |
| **19** | Avaliação completa, estratificações, experimento de PID/IMC | ✅ **Entregável 3** (critérios 3.1–3.10) |
| **20–21** | Redação: metodologia, resultados, discussão da ambiguidade estrutural | capítulos rascunhados |

Os dias 6–7 e 13–14 são folga deliberada. Um cronograma de 3 semanas sem folga é um cronograma de 3 semanas que não é cumprido.

---

## 5. Riscos e mitigações

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| **Driver NVIDIA não instala** (Fedora 43 + Secure Boot exige assinatura de módulo MOK) | Média | Médio | Timebox de 3 h no Dia 1. Plano B: **tudo em CPU** — o estimador 1D treina em minutos e a U-Net compacta a 512² treina em ~2 h nos 16 threads. O plano foi dimensionado para ser executável sem GPU; a GPU só encurta a Semana 2. |
| **PyTorch sem wheel para Python 3.14** | Alta | Baixo | venv com `python3.11`, já presente no sistema |
| **RAM insuficiente** (só ~6 GB livres) | Média | Médio | Carregamento por lote via `DataLoader` com memória mapeada; nunca carregar o dataset inteiro; lote 8–16 |
| **OCR falha em fontes pequenas / DPI baixo** | Alta | Médio | RANSAC tolera erros individuais (bastam 2 ticks corretos); teste de consistência rejeita em vez de errar em silêncio; limite inferior de DPI declarado como hipótese |
| **Ambiguidade FOPDT × 2ª ordem superamortecida derruba a acurácia** | Alta | Baixo | Já é tratada por desenho (§1.5) — vira resultado analisado, não falha |
| **Conjunto OOD não fica pronto a tempo** | Média | Alto (é o argumento central) | Montá-lo no Dia 18 é tarde: coletar capturas de tela **em paralelo, a partir do Dia 1**, em pastas soltas; anotação só no Dia 18 |
| **Escopo cresce** (zeros, ordens superiores, múltiplas curvas) | Média | Alto | Congelado em §1.4; qualquer extensão vai para "trabalhos futuros" |

---

## 6. Estrutura de diretórios proposta

```
TCC-2/
├── dataset/
│   ├── generator.py        # renderização + máscara + meta.json (Parte 1)
│   ├── series.py           # geração analítica de séries + degradação (Parte 3)
│   └── randomize.py        # sorteio de estilo visual, isolado e auditável
├── identify/
│   ├── classical.py        # least_squares, AIC, baselines (Parte 1)
│   ├── extract.py          # U-Net + polilinha (Parte 2)
│   ├── calibrate.py        # moldura, ticks, OCR, RANSAC (Parte 2)
│   ├── estimator.py        # CNN 1D multi-cabeça (Parte 3)
│   └── pipeline.py         # A→B→C→D, com modo oráculo comutável
├── tests/
│   ├── test_part1.py  test_part2.py  test_part3.py
│   └── test_leakage.py
├── reports/                # tabelas e figuras geradas, prontas para a monografia
├── data/                   # não versionado
└── ood/                    # conjunto fora da distribuição, coletado à mão
```

O modo oráculo comutável em `pipeline.py` é um detalhe importante: é ele que permite, a qualquer momento, medir a degradação de cada estágio isoladamente sem reescrever código de avaliação.

---

## 7. O que muda nos relatórios

Para o Ciclo 4, sugere-se registrar com transparência:

1. que a auditoria do gerador revelou vazamento de rótulo, invalidando as métricas dos Ciclos 1 e 2 — apresentado como achado metodológico;
2. que a parametrização anunciada no Ciclo 3 foi de fato implementada neste ciclo, junto com a máscara de segmentação;
3. que o dataset de 50.000 imagens foi **substituído** por 6.000 imagens + geração ilimitada de séries, com a justificativa de eficiência da §1.2 — uma decisão de projeto melhor fundamentada que o número original;
4. que o escopo passou a incluir a calibração de eixos, sem a qual a saída não teria unidade física (§1.1).

Nenhum desses pontos enfraquece o trabalho. Um TCC que documenta por que uma abordagem inicial falhou e como isso mudou o projeto é, em banca, mais forte que um que só relata sucessos.
```
