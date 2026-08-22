# REFERÊNCIAS — obras que sustentam a arquitetura e as decisões

Este arquivo não é a bibliografia final da monografia. É o **mapa entre cada decisão
de projeto e a literatura que a sustenta** — organizado por decisão, não por autor,
para que ao redigir cada seção você saiba exatamente o que citar e por quê.

---

## ⚠ Antes de usar: verifique cada entrada

As entradas abaixo foram compiladas de memória técnica, **sem acesso às fontes no
momento da redação**. Autores, títulos e veículos estão corretos com alta confiança;
**ano, volume, páginas e DOI precisam ser conferidos** contra a fonte antes de
entrarem na monografia.

Cada entrada leva um marcador de confiança:

- **[✓]** — obra canônica, amplamente citada; a existência e a atribuição são seguras.
- **[?]** — a obra existe e o conteúdo é o descrito, mas **confirme os dados
  bibliográficos** (edição, ano, páginas, veículo exato) antes de citar.

Citação com dado bibliográfico errado é problema de banca. Trate este arquivo como
lista de leitura verificável, não como bibliografia pronta.

---

# Parte I — Por decisão de arquitetura

## §1.1 Decisão A — saída em unidades físicas, calibração explícita

O argumento é que os parâmetros dimensionais **não são identificáveis** a partir dos
pixels da curva isoladamente; só grandezas adimensionais o são.

- **[✓]** BELLMAN, R.; ÅSTRÖM, K. J. On structural identifiability. *Mathematical
  Biosciences*, v. 7, n. 3-4, p. 329–339, 1970.
  → A distinção formal entre identificabilidade estrutural e prática. É a base
  conceitual para afirmar que a leitura dos eixos não é acessório, e sim condição
  necessária.
- **[✓]** LJUNG, L. *System Identification: Theory for the User*. 2. ed. Upper Saddle
  River: Prentice Hall, 1999.
  → Capítulos sobre identificabilidade e informatividade do experimento. Referência
  obrigatória do trabalho inteiro.
- **[✓]** KAY, S. M. *Fundamentals of Statistical Signal Processing: Estimation
  Theory*. Englewood Cliffs: Prentice Hall, 1993.
  → Limite de Cramér–Rao. Sustenta a auditoria registrada em `HANDOFF.md §3.6`, que
  mostra por que o critério 1.2 é limite de informação e não falha de estimador.

## §1.2 Decisão B — pipeline em estágios, não CNN fim-a-fim

Quatro argumentos: invariância por construção, vazamento detectável, atribuição de
erro, viabilidade no hardware. **O segundo é o mais forte, e é o mais bem apoiado na
literatura.**

### Aprendizado por atalho — o argumento central

- **[✓]** GEIRHOS, R. *et al.* Shortcut learning in deep neural networks. *Nature
  Machine Intelligence*, v. 2, p. 665–673, 2020. arXiv:2004.07780.
  → **A referência principal desta decisão e do achado do Ciclo 4.** Define atalho,
  mostra por que ele é invisível em teste dentro da distribuição, e por que só aparece
  fora dela. É o enquadramento teórico exato do vazamento encontrado no `img.py`.
- **[✓]** LAPUSCHKIN, S. *et al.* Unmasking Clever Hans predictors and assessing what
  machines really learn. *Nature Communications*, v. 10, art. 1096, 2019.
  → O caso do classificador que reconhecia cavalos pela marca d'água do fotógrafo.
  Análogo direto do `img.py`, que codificava `damping_type` na cor da curva.
- **[✓]** RIBEIRO, M. T.; SINGH, S.; GUESTRIN, C. "Why should I trust you?":
  Explaining the predictions of any classifier. In: *KDD*, 2016.
  → O exemplo husky/lobo-na-neve. Útil como ilustração didática do mecanismo.
- **[✓]** TORRALBA, A.; EFROS, A. A. Unbiased look at dataset bias. In: *CVPR*, 2011.
  → Viés de dataset como propriedade mensurável. Sustenta a existência dos critérios
  1.3 e 1.4 como testes de dataset, não de modelo.
- **[✓]** ZHANG, C. *et al.* Understanding deep learning requires rethinking
  generalization. In: *ICLR*, 2017. arXiv:1611.03530.
  → Redes memorizam rótulos aleatórios. Explica por que "mais dados e mais camadas"
  não removeram o platô dos Ciclos 1–2: o atalho não é problema de capacidade.
- **[✓]** D'AMOUR, A. *et al.* Underspecification presents challenges for credibility
  in modern machine learning. *Journal of Machine Learning Research*, v. 23, 2022.
  arXiv:2011.03395.
  → Modelos com métrica de teste idêntica podem diferir radicalmente fora da
  distribuição. É a justificativa formal do critério 3.9 (razão de degradação
  OOD/sintético como o número que importa).

### Generalização fora da distribuição — sustenta o conjunto OOD

- **[✓]** RECHT, B. *et al.* Do ImageNet classifiers generalize to ImageNet? In:
  *ICML*, 2019. arXiv:1902.10811.
  → Queda sistemática em conjunto de teste novo colhido pelo mesmo protocolo. É o
  argumento de que teste dentro da distribuição superestima desempenho — exatamente o
  que o conjunto OOD do §PARTE 3 existe para evitar.
- **[✓]** TOBIN, J. *et al.* Domain randomization for transferring deep neural
  networks from simulation to the real world. In: *IROS*, 2017. arXiv:1703.06907.
  → **Sustenta a randomização de estilo do gerador.** Randomizar agressivamente
  atributos irrelevantes (cor, textura, iluminação) força o modelo a depender só do
  sinal invariante. É precisamente o papel do `randomize.py`.

### Atribuição de erro e engenharia de pipelines de ML

- **[✓]** SCULLEY, D. *et al.* Hidden technical debt in machine learning systems. In:
  *NeurIPS*, 2015.
  → Discute *pipeline jungles* e o custo de fronteiras mal definidas. Use com cuidado:
  ele argumenta **contra** pipelines longos, então é a melhor referência para a
  **contrapartida** assumida no §1.2, não para o argumento a favor. Citar o
  contra-argumento fortalece a defesa.

## §1.3 Decisão C — estágio D como estimador único, sem rede intermediária

Duas frentes: (a) por que mínimos quadrados com multi-start basta; (b) por que a
ambiguidade estrutural é limite de informação e não deficiência do estimador.

### Otimização não-linear e o núcleo numérico

- **[✓]** LEVENBERG, K. A method for the solution of certain non-linear problems in
  least squares. *Quarterly of Applied Mathematics*, v. 2, p. 164–168, 1944.
- **[✓]** MARQUARDT, D. W. An algorithm for least-squares estimation of nonlinear
  parameters. *SIAM Journal on Applied Mathematics*, v. 11, n. 2, p. 431–441, 1963.
  → Os dois acima são o método clássico. **Atenção:** a implementação usa TRF, não LM,
  porque o `method='lm'` do SciPy não aceita limites de caixa. Cite LM como contexto
  histórico e TRF como o que foi de fato usado — ver `PLANO.md §1.3`.
- **[✓]** BRANCH, M. A.; COLEMAN, T. F.; LI, Y. A subspace, interior, and conjugate
  gradient method for large-scale bound-constrained minimization problems. *SIAM
  Journal on Scientific Computing*, v. 21, n. 1, p. 1–23, 1999.
  → **É o algoritmo por trás de `method="trf"` do `scipy.optimize.least_squares`.**
  Esta é a citação correta para o que o estágio D realmente executa.
- **[?]** COLEMAN, T. F.; LI, Y. An interior trust region approach for nonlinear
  minimization subject to bounds. *SIAM Journal on Optimization*, v. 6, n. 2,
  p. 418–445, 1996.
  → Base da reflexão nos limites de caixa.
- **[✓]** NOCEDAL, J.; WRIGHT, S. J. *Numerical Optimization*. 2. ed. New York:
  Springer, 2006.
  → Região de confiança, Gauss–Newton, escalamento de variáveis. Sustenta a explicação
  do `x_scale="jac"` (K ~ 20 contra τ ~ 0,05).
- **[✓]** GOLUB, G. H.; PEREYRA, V. The differentiation of pseudo-inverses and
  nonlinear least squares problems whose variables separate. *SIAM Journal on
  Numerical Analysis*, v. 10, n. 2, p. 413–432, 1973.
  → **Projeção variável (*variable projection*): é exatamente a técnica do "K
  perfilado"** em `_grid_guess_*` e `_profiled_sse`. Como K entra linearmente, o K
  ótimo sai em forma fechada para cada ponto da grade, reduzindo a dimensão da busca.
  Citação obrigatória para essa parte do código.
- **[✓]** BARD, Y. *Nonlinear Parameter Estimation*. New York: Academic Press, 1974.
- **[✓]** SEBER, G. A. F.; WILD, C. J. *Nonlinear Regression*. New York: Wiley, 1989.
  → Os dois acima: condicionamento, mal-condicionamento e o efeito da parametrização
  sobre a geometria da superfície de resíduos. Base para a discussão de `(ωₙ, ζ)`
  contra coeficientes do denominador.
- **[?]** RINNOOY KAN, A. H. G.; TIMMER, G. T. Stochastic global optimization methods.
  *Mathematical Programming*, v. 39, 1987.
  → Fundamento de multi-start. Sustenta a estratégia de `_multistart` como método, não
  como truque de implementação.

### Seleção de modelo

- **[✓]** AKAIKE, H. A new look at the statistical model identification. *IEEE
  Transactions on Automatic Control*, v. 19, n. 6, p. 716–723, 1974.
  → O AIC. É a citação da arbitragem de estrutura do `identify()`.
- **[✓]** BURNHAM, K. P.; ANDERSON, D. R. *Model Selection and Multimodel Inference:
  A Practical Information-Theoretic Approach*. 2. ed. New York: Springer, 2002.
  → **Muito útil aqui:** discute ΔAIC e o que significa uma diferença pequena entre
  modelos. É o apoio para a leitura de que ΔAIC alto (2015, no exemplo documentado) é
  decisão inequívoca, enquanto ΔAIC baixo em ζ alto indica modelos equivalentes — e
  não um classificador ruim. Também traz o AICc, para amostra pequena.

### Identificabilidade prática — a base do resultado central

- **[✓]** BECK, J. V.; ARNOLD, K. J. *Parameter Estimation in Engineering and
  Science*. New York: Wiley, 1977.
  → Sensibilidade e correlação entre parâmetros; o caso clássico de parâmetros que só
  aparecem em combinação. Explica `corr(erro_ωₙ, erro_ζ) = +0,9997`.
- **[✓]** RAUE, A. *et al.* Structural and practical identifiability analysis of
  partially observed dynamical models by exploiting the profile likelihood.
  *Bioinformatics*, v. 25, n. 15, p. 1923–1929, 2009.
  → **A melhor referência para o achado do RULING N.** Formaliza exatamente o fenômeno
  medido: a verossimilhança é plana ao longo de uma curva de nível, os parâmetros
  individuais deslizam, e a grandeza combinada permanece bem determinada.
- **[?]** BRUN, R.; REICHERT, P.; KÜNSCH, H. R. Practical identifiability analysis of
  large environmental simulation models. *Water Resources Research*, v. 37, n. 4,
  p. 1015–1030, 2001.
  → Índice de colinearidade a partir da matriz de sensibilidade. É a métrica que
  quantifica o que o experimento `exp_cond.py` mediu (correlação das colunas do
  jacobiano, 66× melhor em coeficientes do denominador).
- **[✓]** GOLUB, G. H.; VAN LOAN, C. F. *Matrix Computations*. 4. ed. Baltimore: Johns
  Hopkins University Press, 2013.
  → Número de condição e SVD. Base formal da comparação de condicionamento entre
  parametrizações.

## §1.4 Decisão D — cobertura FOPDT + 2ª ordem canônica

- **[✓]** ÅSTRÖM, K. J.; HÄGGLUND, T. *Advanced PID Control*. Research Triangle Park:
  ISA, 2006.
  → Por que FOPDT é o modelo de referência da indústria. Ancora a relevância prática.
- **[✓]** SEBORG, D. E.; EDGAR, T. F.; MELLICHAMP, D. A.; DOYLE, F. J. *Process
  Dynamics and Control*. Hoboken: Wiley. (confira a edição)
  → Modelos de baixa ordem em controle de processos; resposta inversa e zeros — útil
  para justificar o **escopo declarado como fora** (zeros, fase não-mínima).
- **[✓]** OGATA, K. *Modern Control Engineering*. 5. ed. Upper Saddle River: Prentice
  Hall, 2010.
- **[✓]** NISE, N. S. *Control Systems Engineering*. 7. ed. Hoboken: Wiley, 2015.
  → Os dois acima: forma canônica de 2ª ordem, relações sobressinal↔ζ (usada em
  `_overshoot_guess`), tempo de acomodação. **E são as fontes das figuras do conjunto
  OOD** — cite-os nessa função também.

### Baselines clássicos implementados

- **[✓]** ZIEGLER, J. G.; NICHOLS, N. B. Optimum settings for automatic controllers.
  *Transactions of the ASME*, v. 64, p. 759–768, 1942.
  → Método da tangente (`baseline_tangent`).
- **[?]** SMITH, C. L. *Digital Computer Process Control*. Scranton: Intext
  Educational Publishers, 1972.
  → Método dos dois pontos (`baseline_smith`). **Confirme o veículo e o ano** — este
  método é frequentemente citado de segunda mão.
- **[✓]** SUNDARESAN, K. R.; KRISHNASWAMY, P. R. Estimation of time delay time
  constant parameters in time, frequency, and Laplace domains. *The Canadian Journal
  of Chemical Engineering*, v. 56, n. 2, p. 257–262, 1978.
  → `baseline_sundaresan_krishnaswamy`.
- **[?]** COHEN, G. H.; COON, G. A. Theoretical consideration of retarded control.
  *Transactions of the ASME*, v. 75, p. 827–834, 1953.
  → Citado no §1.4 como âncora de relevância; não implementado.
- **[✓]** RIVERA, D. E.; MORARI, M.; SKOGESTAD, S. Internal model control: PID
  controller design. *Industrial & Engineering Chemistry Process Design and
  Development*, v. 25, n. 1, p. 252–265, 1986.
  → **IMC-PID: é o método do critério 3.10**, o experimento que fecha o círculo com o
  título do curso.

## §1.5 A ambiguidade estrutural

As referências de identificabilidade prática do §1.3 (RAUE, BECK & ARNOLD, BRUN)
são as principais. Acrescente:

- **[✓]** LJUNG, L. *op. cit.*, capítulo sobre seleção de estrutura de modelo.
  → Sustenta a decisão de reportar erro de reconstrução em vez de acurácia de
  classificação quando as estruturas são equivalentes na saída.

## §1.7 Decisão E — OCR opcional, saída em dois níveis

- **[✓]** SMITH, R. An overview of the Tesseract OCR engine. In: *ICDAR*, 2007.
  → A ferramenta usada, e suas condições de operação (DPI mínimo, tipografia).
- **[✓]** SAVVA, M. *et al.* ReVision: Automated classification, analysis and redesign
  of chart images. In: *UIST*, 2011.
  → **Documenta a fragilidade do OCR em rótulos de eixo de gráficos** — poucos pixels,
  fontes variadas, orientação. É a evidência externa de que tratar o OCR como elo
  frágil não é pessimismo local, e sim característica conhecida do problema.
- **[✓]** POCO, J.; HEER, J. Reverse-engineering visualizations: Recovering visual
  encodings from chart images. *Computer Graphics Forum* (EuroVis), 2017.
  → Pipeline de recuperação de codificação visual, incluindo texto de eixo. Boa fonte
  para taxas de acerto de OCR em gráficos reais.

## §1.8 Decisão F — extrator clássico como Plano B e baseline

- **[✓]** DUDA, R. O.; HART, P. E. Use of the Hough transformation to detect lines and
  curves in pictures. *Communications of the ACM*, v. 15, n. 1, p. 11–15, 1972.
  → Detecção de retas. É a alternativa canônica ao passo de "rejeição de componentes
  retilíneas" (grade, *spines*, distratores).
- **[✓]** OTSU, N. A threshold selection method from gray-level histograms. *IEEE
  Transactions on Systems, Man, and Cybernetics*, v. 9, n. 1, p. 62–66, 1979.
- **[✓]** CANNY, J. A computational approach to edge detection. *IEEE Transactions on
  Pattern Analysis and Machine Intelligence*, v. 8, n. 6, p. 679–698, 1986.
  → Limiarização e bordas: as ferramentas do extrator clássico e da detecção de moldura
  por projeção de gradientes.
- **[✓]** CLICHE, M. *et al.* Scatteract: Automated extraction of data from scatter
  plots. In: *ECML PKDD*, 2017. arXiv:1704.06687.
  → **Compara abordagem clássica e aprendida no mesmo problema.** É o precedente
  metodológico exato do critério 2.10.

---

# Parte II — Por componente

## Estágio A — segmentação da curva (U-Net)

- **[✓]** RONNEBERGER, O.; FISCHER, P.; BROX, T. U-Net: Convolutional networks for
  biomedical image segmentation. In: *MICCAI*, 2015, p. 234–241. arXiv:1505.04597.
  → A arquitetura. O argumento de que as *skip connections* preservam localização em
  nível de pixel — que é o que determina a precisão de y(t) — está no artigo original.
- **[✓]** LONG, J.; SHELHAMER, E.; DARRELL, T. Fully convolutional networks for
  semantic segmentation. In: *CVPR*, 2015.
  → Antecedente. Útil para justificar por que uma rede classificadora (ResNet-18) é a
  ferramenta errada: *pooling* agressivo descarta a resolução espacial desejada.
- **[✓]** MILLETARI, F.; NAVAB, N.; AHMADI, S.-A. V-Net: Fully convolutional neural
  networks for volumetric medical image segmentation. In: *3DV*, 2016.
  arXiv:1606.04797.
  → **Perda Dice.** A citação para `dice_bce_loss`.
- **[✓]** SUDRE, C. H. *et al.* Generalised Dice overlap as a deep learning loss
  function for highly unbalanced segmentations. In: *MICCAI DLMIA*, 2017.
  arXiv:1707.03237.
  → **Sustenta quantitativamente a decisão de não usar BCE pura**: com classe positiva
  em torno de 0,6% dos pixels (medido: 3.400 de 577.980), BCE colapsa para "tudo
  fundo".
- **[✓]** HE, K. *et al.* Deep residual learning for image recognition. In: *CVPR*,
  2016. arXiv:1512.03385.
  → ResNet. Usada como tronco do baseline fim-a-fim (`PLANO_CNN_FIM_A_FIM.md §5`).

## Estágio B — calibração dos eixos

- **[✓]** FISCHLER, M. A.; BOLLES, R. C. Random sample consensus: A paradigm for model
  fitting with applications to image analysis and automated cartography.
  *Communications of the ACM*, v. 24, n. 6, p. 381–395, 1981.
  → RANSAC. A citação do ajuste robusto da afim sobre os pares (pixel, valor).
- **[✓]** HARTLEY, R.; ZISSERMAN, A. *Multiple View Geometry in Computer Vision*.
  2. ed. Cambridge: Cambridge University Press, 2004.
  → Capítulo de estimação robusta: RANSAC, limiar de consenso, número mínimo de
  amostras. Sustenta a afirmação de que "bastam 2 ticks corretos por eixo".

## Estágio C — removido, mas a especificação foi reaproveitada

As referências abaixo continuam relevantes porque a **cabeça multi-tarefa** do
Estágio C é a cabeça do baseline fim-a-fim (`PLANO_CNN_FIM_A_FIM.md §5`), e porque a
decisão de remover o estágio precisa ser argumentada contra a literatura que o
justificaria.

- **[✓]** YU, F.; KOLTUN, V. Multi-scale context aggregation by dilated convolutions.
  In: *ICLR*, 2016. arXiv:1511.07122.
  → Convoluções dilatadas e o crescimento exponencial do campo receptivo. **É também a
  fonte para verificar a aritmética do campo receptivo** — o `PLANO.md` original
  afirmava cobertura de 512 amostras com dilatações 1–32, o que dá 127
  (`1 + 2·(1+2+…+32)`).
- **[✓]** VAN DEN OORD, A. *et al.* WaveNet: A generative model for raw audio.
  arXiv:1609.03499, 2016.
  → Convoluções dilatadas empilhadas em sinal 1D. O precedente da arquitetura.
- **[✓]** BAI, S.; KOLTER, J. Z.; KOLTUN, V. An empirical evaluation of generic
  convolutional and recurrent networks for sequence modeling. arXiv:1803.01271, 2018.
  → **Sustenta a escolha de CNN dilatada em vez de GRU/LSTM**: convolução temporal
  iguala ou supera recorrente em modelagem de sequência, e paraleliza no tempo.
- **[?]** SNYDER, D. *et al.* X-vectors: Robust DNN embeddings for speaker recognition.
  In: *ICASSP*, 2018.
  → *Pooling* estatístico (média + desvio-padrão ao longo do tempo) como agregador de
  tamanho fixo. É a origem da técnica prevista para a cabeça.

## Estágio D — já implementado

Ver §1.3 acima. As citações centrais são BRANCH/COLEMAN/LI (o algoritmo real),
GOLUB & PEREYRA (o K perfilado), AKAIKE e BURNHAM & ANDERSON (a arbitragem).

---

# Parte III — Trabalhos relacionados: extração de dados de gráficos

Este é o campo em que o trabalho se insere, e a seção de trabalhos relacionados da
monografia sai daqui. **Nenhum destes resolve o problema deste TCC** — todos extraem
*a série de dados* do gráfico; nenhum vai da série aos **parâmetros de uma planta
dinâmica**. Essa é a lacuna a declarar.

- **[✓]** SAVVA, M. *et al.* ReVision: Automated classification, analysis and redesign
  of chart images. In: *UIST*, 2011.
  → Pioneiro. Classifica o tipo de gráfico e extrai os dados. Abordagem clássica.
- **[✓]** CLICHE, M. *et al.* Scatteract: Automated extraction of data from scatter
  plots. In: *ECML PKDD*, 2017. arXiv:1704.06687.
  → Detecção de objetos + OCR para eixos. **Muito próximo em espírito do estágio B**,
  inclusive no uso de dados sintéticos para treino.
- **[✓]** POCO, J.; HEER, J. Reverse-engineering visualizations: Recovering visual
  encodings from chart images. *Computer Graphics Forum* (EuroVis), 2017.
- **[✓]** LUO, J. *et al.* ChartOCR: Data extraction from charts images via a deep
  hybrid framework. In: *WACV*, 2021.
  → Híbrido de rede e regra — a mesma filosofia deste trabalho, e boa referência para
  defender a arquitetura em estágios.
- **[✓]** METHANI, N. *et al.* PlotQA: Reasoning over scientific plots. In: *WACV*,
  2020. arXiv:1909.00997.
- **[✓]** MASRY, A. *et al.* ChartQA: A benchmark for question answering about charts
  with visual and logical reasoning. In: *ACL Findings*, 2022. arXiv:2203.10244.
- **[✓]** LIU, F. *et al.* DePlot: One-shot visual language reasoning by plot-to-table
  translation. arXiv:2212.10505, 2023.
- **[✓]** LIU, F. *et al.* MatCha: Enhancing visual language pretraining with math
  reasoning and chart derendering. arXiv:2212.09662, 2023.
  → Os quatro acima: estado da arte recente, baseado em modelos multimodais grandes.
  Úteis para situar o trabalho e para justificar por que **não** se seguiu esse
  caminho (custo, ausência de garantia de invariância, e o fato de que a saída
  desejada aqui é paramétrica e não tabular).
- **[?]** LineEX: Data extraction from scientific line charts. In: *WACV*, 2023.
  → Especificamente sobre gráficos de linha, que é o caso deste trabalho.
  **Confirme autores e veículo** antes de citar.

---

# Parte IV — Metodologia de verificação

O rigor de verificação da Parte 1 (33 testes, campanha de mutação, controles
negativos) é um diferencial do trabalho e merece fundamentação própria.

- **[✓]** DEMILLO, R. A.; LIPTON, R. J.; SAYWARD, F. G. Hints on test data selection:
  Help for the practicing programmer. *Computer*, v. 11, n. 4, p. 34–41, 1978.
  → Origem do teste de mutação.
- **[✓]** JIA, Y.; HARMAN, M. An analysis and survey of the development of mutation
  testing. *IEEE Transactions on Software Engineering*, v. 37, n. 5, p. 649–678, 2011.
- **[?]** PAPADAKIS, M. *et al.* Mutation testing advances: An analysis and survey.
  *Advances in Computers*, v. 112, p. 275–378, 2019.
  → Os dois acima sustentam a campanha de 17 mutantes do `HANDOFF.md §3.5.1` como
  método reconhecido, e o princípio "critério não testado contra defeito injetado é
  decoração".
- **[✓]** CHEN, T. Y. *et al.* Metamorphic testing: A review of challenges and
  opportunities. *ACM Computing Surveys*, v. 51, n. 1, 2018.
  → **Enquadra os controles negativos do trabalho.** O critério 1.5c (deslocar a afim
  em 3 px deve piorar o RMSE ≥ 10×) é exatamente uma relação metamórfica. Dá nome
  formal a uma prática que hoje está no código sem rótulo.
- **[✓]** DUNN, O. J. Multiple comparisons among means. *Journal of the American
  Statistical Association*, v. 56, n. 293, p. 52–64, 1961.
  → Correção de Bonferroni, usada no critério 1.4a (161 pares testados).
- **[✓]** BENJAMINI, Y.; HOCHBERG, Y. Controlling the false discovery rate: A
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society, Series B*, v. 57, n. 1, p. 289–300, 1995.
  → Alternativa menos conservadora ao Bonferroni. Vale citar ao justificar a escolha.
- **[✓]** GOOD, P. *Permutation, Parametric, and Bootstrap Tests of Hypotheses*.
  3. ed. New York: Springer, 2005.
  → Testes de permutação. Sustenta o nulo empírico do critério 1.4b.
- **[✓]** FRIEDMAN, J. H. Greedy function approximation: A gradient boosting machine.
  *Annals of Statistics*, v. 29, n. 5, p. 1189–1232, 2001.
  → O GBM usado como sonda de vazamento no critério 1.3.
- **[?]** PINEAU, J. *et al.* Improving reproducibility in machine learning research.
  *Journal of Machine Learning Research*, v. 22, 2021.
  → Sustenta o critério 1.6 (determinismo bit-a-bit, hash de PNG) e o
  `requirements.txt` pinado como prática, não como excesso de zelo.

---

# Parte V — Ferramentas e software

Cite o software que produziu os resultados — é exigência crescente de reprodutibilidade.

- **[✓]** VIRTANEN, P. *et al.* SciPy 1.0: Fundamental algorithms for scientific
  computing in Python. *Nature Methods*, v. 17, p. 261–272, 2020.
  → `scipy.optimize.least_squares` — o estágio D.
- **[✓]** HARRIS, C. R. *et al.* Array programming with NumPy. *Nature*, v. 585,
  p. 357–362, 2020.
  → Inclui a documentação de `SeedSequence.spawn`, base do determinismo do gerador.
- **[✓]** HUNTER, J. D. Matplotlib: A 2D graphics environment. *Computing in Science &
  Engineering*, v. 9, n. 3, p. 90–95, 2007.
  → O renderizador. **Relevante além do crédito:** a calibração é lida de
  `ax.transData.transform()`, ou seja, a corretude do `axis_affine` depende da
  semântica documentada dessa transformação.
- **[✓]** PEDREGOSA, F. *et al.* Scikit-learn: Machine learning in Python. *Journal of
  Machine Learning Research*, v. 12, p. 2825–2830, 2011.
  → O GBM do critério 1.3.
- **[✓]** PASZKE, A. *et al.* PyTorch: An imperative style, high-performance deep
  learning library. In: *NeurIPS*, 2019.
- **[✓]** VAN DER WALT, S. *et al.* scikit-image: Image processing in Python. *PeerJ*,
  v. 2, e453, 2014.
- **[?]** BRADSKI, G. The OpenCV Library. *Dr. Dobb's Journal of Software Tools*, 2000.

## Repositórios e ferramentas de referência

| recurso | por que interessa |
|---|---|
| `github.com/scipy/scipy` | implementação de `least_squares`; a docstring do `trf` aponta a citação correta |
| `github.com/matplotlib/matplotlib` | semântica de `transData`, backend `Agg`, determinismo de renderização |
| `github.com/python-control/python-control` | **fonte de imagens do conjunto OOD** com renderizador diferente; e referência de implementação de resposta ao degrau |
| `github.com/ankitrohatgi/WebPlotDigitizer` | a ferramenta manual de referência do domínio. Útil como comparação de linha de base humana e para discutir o que a automação acrescenta |
| `github.com/tesseract-ocr/tesseract` | OCR do estágio B; a documentação de `--psm` e de DPI mínimo |
| `github.com/milesial/Pytorch-UNet` | implementação de referência de U-Net em PyTorch, para conferir a arquitetura |
| `github.com/qubvel/segmentation_models.pytorch` | implementações de perdas de segmentação (Dice, Focal, Tversky), úteis para comparar com `dice_bce_loss` |

---

# Lacunas conhecidas

Registro honesto do que **não** tem referência boa neste arquivo, para não dar a
impressão de cobertura completa:

1. **Parametrização de 2ª ordem por coeficientes do denominador** contra a forma
   canônica `(ωₙ, ζ)`, do ponto de vista de condicionamento. O resultado medido
   (`exp_cond.py`: colinearidade de −0,99992 contra +0,002; condicionamento 66× melhor)
   está apoiado nos textos gerais de estimação não-linear (BARD, SEBER & WILD,
   BECK & ARNOLD), **mas não encontrei um artigo específico sobre esta troca de
   parametrização neste modelo**. Se for para a monografia, apresente como resultado
   próprio com a fundamentação geral, não como reprodução de resultado conhecido.
2. **Identificação de plantas a partir de imagens de gráfico** — a lacuna que o
   trabalho ocupa. A Parte III mostra trabalhos que extraem *séries*; nenhum vai à
   parametrização dinâmica. **Se você encontrar um que vá, ele é o trabalho
   relacionado mais importante da monografia e precisa entrar aqui.** Vale uma busca
   dirigida por: "system identification from step response image", "transfer function
   estimation from plot image", "chart to model parameters".
3. **Extração de curva de gráfico com traço tracejado**, onde a máscara sai
   descontínua (medido: 46 de 358 colunas vazias em linha `-.`). O pós-processamento
   interpola os vãos, mas não localizei tratamento específico na literatura de
   extração de gráficos.
