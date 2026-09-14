---
tags: [tcc, referencias, proveniencia, originalidade]
aliases: [Origem da arquitetura, O que é nosso]
---

# Proveniência

> De onde veio cada peça da arquitetura, o que já estava na literatura, e o que é de fato deste trabalho.

Fonte primária no repositório: `REFERENCIAS.md` (27 KB, organizado por decisão de arquitetura). Esta nota é o resumo executivo dele, com as lacunas destacadas.

Ver também: [[Arquitetura do projeto]] · [[Decisões de projeto]]

---

## Resumo em uma frase

**Nenhum componente é invenção do projeto**, e o *formato* do pipeline também já existe na literatura de extração de dados de gráficos. O que é deste trabalho é o **acoplamento** — levar a extração de gráfico até a identificação paramétrica de uma planta dinâmica — e um conjunto de mecanismos que só existem porque as duas metades se tocam.

---

## Camada 1 · As peças, todas de prateleira

| Peça | Citação canônica |
|---|---|
| U-Net, conexões de salto | **Ronneberger, Fischer & Brox**, MICCAI 2015 · arXiv:1505.04597 |
| Por que não um classificador com pooling | **Long, Shelhamer & Darrell**, CVPR 2015 (FCN) |
| Perda Dice | **Milletari, Navab & Ahmadi**, 3DV 2016 (V-Net) · arXiv:1606.04797 |
| Dice+BCE para classe rara | **Sudre et al.**, MICCAI DLMIA 2017 · arXiv:1707.03237 |
| RANSAC | **Fischler & Bolles**, *CACM* 24(6), 1981 |
| "bastam 2 pontos", limiar de consenso | **Hartley & Zisserman**, *Multiple View Geometry*, 2. ed., cap. de estimação robusta |
| Tesseract | **Smith**, ICDAR 2007 |
| Otimizador do Estágio D (`method="trf"`) | **Branch, Coleman & Li**, *SIAM J. Sci. Comput.* 21(1), 1999 |
| K perfilado (*variable projection*) | **Golub & Pereyra**, *SIAM J. Numer. Anal.* 10(2), 1973 |
| Multistart | **Rinnooy Kan & Timmer**, *Math. Prog.* 39, 1987 |
| Escalamento `x_scale="jac"` | **Nocedal & Wright**, *Numerical Optimization*, 2. ed., 2006 |
| AIC | **Akaike**, *IEEE TAC* 19(6), 1974 |
| ΔAIC pequeno = modelos equivalentes | **Burnham & Anderson**, 2002 |
| Randomização de estilo do gerador | **Tobin et al.**, IROS 2017 — *domain randomization* |
| Aprendizado por atalho (anti-vazamento) | **Geirhos et al.**, *Nature Mach. Intell.* 2, 2020 |
| Identificabilidade estrutural × prática | **Bellman & Åström**, 1970 · **Raue et al.**, *Bioinformatics* 25(15), 2009 |
| Baselines FOPDT | Ziegler–Nichols 1942 · Smith 1972 · Sundaresan–Krishnaswamy 1978 |

> [!note] Sobre o RANSAC exaustivo
> **Não é variante nova.** É o que se faz quando *n* é pequeno — Hartley & Zisserman discutem a contagem de amostras necessárias, e com ≤ 12 rótulos por eixo o cálculo dá "teste todos os pares". A escolha do projeto foi **elevar o determinismo a restrição global**, não inventar um algoritmo.

---

## Camada 2 · O formato do pipeline também já existe

Segmentar a curva + ler os eixos por OCR + calibrar por ajuste robusto é a **receita estabelecida** da literatura de *chart data extraction*.

| Trabalho | Relação |
|---|---|
| **Scatteract** — Cliche et al., ECML PKDD 2017 | Detecção + OCR de eixos, treinado com **dados sintéticos**. "Muito próximo em espírito do estágio B"; é o **precedente metodológico exato do critério 2.10** (comparar clássico e aprendido no mesmo problema) |
| **ChartOCR** — Luo et al., WACV 2021 | Híbrido de rede e regra. "A mesma filosofia deste trabalho" |
| **ReVision** — Savva et al., UIST 2011 | O pioneiro. Fonte externa que documenta a **fragilidade do OCR em rótulos de eixo** |
| **Poco & Heer**, EuroVis 2017 | Recuperação de codificação visual, incluindo texto de eixo |
| **LineEX**, WACV 2023 | Especificamente gráficos de linha — o caso deste trabalho |
| PlotQA, ChartQA, DePlot, MatCha | Estado da arte multimodal. Úteis para justificar por que **não** se seguiu esse caminho |

> [!important] Consequência para a monografia
> **Estágios A + B + C não são arquitetura nova.** São a aplicação competente de uma receita conhecida, com a U-Net no lugar do detector do Scatteract.
>
> Isso é defensável — e é *mais* defensável dizendo assim do que fingindo originalidade onde não há. A contribuição está em outro lugar.

---

## Camada 3 · O que é deste projeto

`REFERENCIAS.md` declara a lacuna, e ela é a afirmação central da monografia:

> **Nenhum destes resolve o problema deste TCC** — todos extraem *a série de dados* do gráfico; nenhum vai da série aos **parâmetros de uma planta dinâmica**. Essa é a lacuna a declarar.

**O acoplamento é a contribuição.** A literatura de gráficos **para** em `(t, y)`. A literatura de identificação de sistemas **começa** em `(t, y)`. Ninguém atravessou a fronteira.

O Estágio D não é novo isoladamente — é LSQ com atraso, que Ljung cobre. Mas ele **nunca recebeu uma série extraída de imagem** na literatura consultada. E é daí que saem os achados que só existem porque as duas metades se tocam:

### A correção `n_eff` do AIC

O resíduo é autocorrelacionado (ρ ≈ 0,71) **porque a série veio de uma polilinha de imagem** — pixels vizinhos carregam erro de extração correlacionado. O AIC clássico tratava 738 pontos como 738 evidências independentes, e **32 % das plantas de 1ª ordem** viravam 2ª ordem.

Um pesquisador de identificação de sistemas não veria isso: a série dele não vem de pixel. Um de extração de gráficos também não: ele para antes do AIC.

Detalhe em [[AIC e seleção de ordem]] · Implementa: [[Módulos/identify.classical|identify.classical]]

### A saída em dois níveis

Fundamentada em Bellman & Åström — ζ é adimensional, logo identificável sem calibração; ωn não é. Mas **estruturar a API nisso**, com `dimensionless` sempre presente e `physical` nulo exatamente quando a calibração falha, é decisão de projeto.

Ver [[Decisões de projeto#D1 · Saída em dois níveis]]

### E mais

| Mecanismo | Nota |
|---|---|
| Aprovação por eixo (`ok_x` / `ok_y`) | [[Decisões de projeto#D9 · Aprovação por eixo, não conjunta]] |
| Moda-da-borda como nível de fundo | [[Decisões de projeto#D10 · O nível de fundo é a moda da borda]] |
| Blob de texto como posição do tick | [[Decisões de projeto#D3 · O blob de texto é a posição do tick]] |
| Teto do lote de OCR + recuo individual | [[Decisões de projeto#D11 · O lote de OCR tem teto e recuo]] |
| Guardas de plausibilidade | [[Decisões de projeto#D12 · Guardas de plausibilidade]] |
| Metodologia de estratos opt-in | [[Arquitetura do projeto#Estratos: o corpus é onde os defeitos aparecem ou se escondem]] |

---

## Três lacunas declaradas

`REFERENCIAS.md` tem uma seção "Lacunas conhecidas" que existe para **não dar impressão de cobertura completa**. Elas são dívida antes da monografia.

### 1. A correção `n_eff` está sem citação

Verificado: o termo não aparece em `REFERENCIAS.md`. É a peça algorítmica mais citável do projeto e a única **sem fundamentação bibliográfica registrada**.

A ideia de *tamanho amostral efetivo* para dados autocorrelacionados é clássica — Bartlett (1935), Bayley & Hammersley (1946) — e há literatura sobre critérios de informação com resíduo correlacionado. **Vale uma busca dirigida.**

Se não achar nada específico, o enquadramento honesto é: **aplicação de um resultado estatístico conhecido a um contexto novo**, apresentada com a fundamentação geral. Nem resultado próprio, nem reprodução.

### 2. A lacuna que o trabalho ocupa não foi buscada exaustivamente

`REFERENCIAS.md` pede busca dirigida por:

- `system identification from step response image`
- `transfer function estimation from plot image`
- `chart to model parameters`

E avisa: **se existir um trabalho assim, ele é o trabalho relacionado mais importante da monografia.**

> Enquanto essa busca não for feita, a afirmação de originalidade está sustentada por *ausência de evidência*, não por *evidência de ausência*.

### 3. Parametrização `(ωₙ, ζ)` contra coeficientes do denominador

O resultado medido em `exp_cond.py` — colinearidade de −0,99992 contra +0,002, condicionamento **66× melhor** — está apoiado em textos gerais (Bard, Seber & Wild, Beck & Arnold), mas **não há artigo específico sobre esta troca de parametrização neste modelo**.

Recomendação registrada: apresentar como **resultado próprio com fundamentação geral**, não como reprodução de resultado conhecido.

---

## Como enquadrar na monografia

1. **Trabalhos relacionados** — a Parte III de `REFERENCIAS.md` inteira, terminando na frase da lacuna.
2. **Arquitetura** — creditar Scatteract e ChartOCR como o precedente do formato em estágios. Não reivindicar novidade aqui.
3. **Contribuição** — o acoplamento, e os mecanismos da Camada 3.
4. **Ameaças à validade** — as três lacunas acima, mais a ressalva da validação externa ([[Arquitetura do projeto#9. Envelope de validade e limites conhecidos]]).

Há ainda uma referência que **fortalece a defesa por ser contrária**: Sculley et al., NeurIPS 2015, argumenta *contra* pipelines longos (*pipeline jungles*). `REFERENCIAS.md` recomenda citá-la como a contrapartida assumida no §1.2 — citar o contra-argumento é mais forte que omiti-lo.
