---
tags: [tcc, arquitetura, visao-macro]
aliases: [Do Pixel ao Parâmetro, Visão macro]
---

# Arquitetura — Do Pixel ao Parâmetro

> Um gráfico de resposta ao degrau entra como imagem. Sai `K`, `τ`, `θ`, `ωn`, `ζ` e a estrutura do modelo. Entre as duas pontas há quatro estágios, cada um trocando a representação do dado — e é essa troca que organiza o projeto inteiro.

| | |
|---|---|
| Estágios | 4, cada um com contrato próprio |
| Código de produção | 4 032 linhas |
| Suíte | 69 testes, todos passando |
| Cobertura da calibração física | 93,0 % |
| Erro geométrico da máscara (mediana) | 0,799 px |

Documentos irmãos: [[Decisões de projeto]] · [[Módulos]] · [[Critérios de qualidade]]

---

## 1. A arquitetura é uma cadeia de representações

O problema tem uma estrutura natural que a arquitetura copia. Uma imagem de gráfico carrega **duas informações independentes**: *onde a curva está* (geometria em pixels) e *o que os pixels significam* (a escala dos eixos). São problemas diferentes, resolvidos por técnicas diferentes, e que falham de formas diferentes.

Separá-los em estágios distintos não é organização cosmética — é o que permite **entregar resposta parcial quando só um dos dois funciona**.

```
                 ┌──────────────────────────────┐
   imagem RGB ──►│ A · Segmentação  (U-Net)     │──► máscara 0/255
        │        └──────────────────────────────┘        │
        │                                                ▼
        │                                     ┌────────────────────┐
        │                                     │ C · Polilinha      │──► x_px, y_px
        │                                     └────────────────────┘        │
        │        ┌──────────────────────────────┐                           │
        └───────►│ B · Calibração  (OCR+RANSAC) │──► sx, ox, sy, oy         │
                 └──────────────────────────────┘        │                  │
                                                         ▼                  ▼
                                            ┌─────────────────────────────────┐
                                            │ D · Ajuste  (LSQ + AIC n_eff)   │
                                            └─────────────────────────────────┘
                                                         │
                                              dimensionless (sempre)
                                              physical      (só se B fecha)
```

**A e B são independentes** e leem a mesma imagem. A produz geometria, B produz unidades. Só o estágio D precisa dos dois — e é por isso que a saída tem dois níveis: quando B falha, a geometria ainda é suficiente para responder a estrutura e o amortecimento.

### O contrato em cada fronteira

| Fronteira | Tipo | Invariante |
|---|---|---|
| imagem → A | `uint8[H,W,3]` | RGB puro, sem canal alfa |
| A → C | `uint8[H,W]` 0/255 | Mesma resolução da entrada. Contrato deliberadamente estreito: qualquer extrator que o cumpra é intercambiável |
| B → D | `Calibration` | `x = sx·x_px + ox`, `y = sy·y_px + oy`, com `sy < 0` (o pixel cresce para baixo) |
| C → D | dois `float64[N]` | Ordenados em `t`, uma amostra por coluna de pixel |
| D → Parte 3 | `dict` | `dimensionless` nunca é nulo; `physical` é nulo exatamente quando a calibração falha |

> [!important] Decisão · Saída em dois níveis
> Antes desta decisão, a calibração falhar abortava a amostra inteira. Isso custava **56 das ~68 amostras perdidas** — justamente aquelas em que ζ é recuperável sem calibração nenhuma, porque *amortecimento é adimensional*: ele se lê da forma da curva, não da escala dos eixos. O que a calibração dá é a *unidade*; ωn em rad/s precisa dela, ζ não.
>
> Separar os dois níveis transformou uma perda total numa perda parcial, e o teste que assevera isso mede 300/300 amostras com bloco adimensional preenchido. Ver [[Decisões de projeto#D1 · Saída em dois níveis]].

---

## 2. Estágio A — Segmentação

`[[identify.extract]]` (U-Net) · `[[identify.extract_classical]]` (alternativa sem rede)

A entrada é uma figura completa: moldura, grade, legenda, título, retas de referência, anotações. A saída deve conter *apenas* os pixels da curva. É segmentação semântica pixel a pixel, e a dificuldade não é a curva — é a quantidade de coisas que se parecem com ela.

### Por que U-Net

> Diagrama completo, com canais e parâmetros por bloco: [[Anatomia da U-Net]]

A U-Net é uma rede encoder–decoder com **conexões de salto** entre níveis simétricos. O encoder reduz a resolução para capturar contexto — "isto é uma legenda, não uma curva" exige olhar longe — e o decoder devolve à resolução original. As conexões de salto reinjetam a informação de alta frequência que o encoder descartou.

Essa combinação é exatamente o que o problema pede. Decidir *o que* é curva exige contexto amplo; marcar *onde* ela passa exige precisão de um pixel. Uma rede só convolucional sem downsampling não teria contexto; um classificador com downsampling e sem skips devolveria máscara borrada, e o erro de posição viraria erro de parâmetro no estágio D.

| Escolha | Valor | Razão |
|---|---|---|
| Profundidade | 4 níveis | Campo receptivo de 173 px (medido) — 34 % do lado do quadrado do letterbox |
| Canais base | 32 | Elevado de 24 depois que a saturação foi *medida*, não suposta |
| Parâmetros | 7 763 041 | Cabe em 6 GB de VRAM com lote 6 a 512×512 |
| Entrada | 3 canais (RGB) | A mudança mais consequente do projeto — ver abaixo |
| Perda | Dice + BCE | A curva ocupa ~1 % dos pixels; BCE puro converge para "tudo fundo" |
| Alvo | contínuo, não binário | Preserva o gradiente do anti-aliasing |

### Letterbox, não redimensionamento

A rede opera a 512×512, mas as figuras têm proporções variadas. O pré-processamento usa **letterbox** — escala isotrópica mais preenchimento — em vez de esticar a imagem. Distorção anisotrópica mudaria a inclinação local da curva, e a inclinação é o que o estágio D lê como constante de tempo. Esticar a imagem envenenaria o parâmetro.

> [!important] Decisão · Três canais em vez de um
> A versão original projetava RGB em luminância antes da rede, pela fórmula ITU-R BT.601 `0,299R + 0,587G + 0,114B`. Parecia inofensivo.
>
> É uma projeção de ℝ³ em ℝ¹, e portanto **destrutiva**. Numa imagem real, a curva verde `(44,160,44)` e a reta de referência vermelha `(230,61,61)` viram *o mesmo byte 112*. Onde se cruzavam, o contraste caía para 32 % do original e a máscara perdia **244 colunas consecutivas** — 39 % da largura. Separá-las não era difícil: era *impossível*, porque a informação já tinha sido descartada antes da rede ver a imagem.
>
> Ver [[Decisões de projeto#D2 · RGB em vez de luminância]].

### O extrator clássico, e por que ele existe

`extract_classical.py` resolve o mesmo problema sem rede: segmentação por cor modal seguida de rejeição de componentes retilíneas de span completo. Ele é **mais fraco** que a U-Net e existe por duas razões que não são desempenho.

A primeira é risco de projeto: se a GPU não estivesse disponível, o TCC precisaria de um Estágio A funcional. A segunda é epistemológica — ele é a *linha de base* contra a qual o ganho da rede é medido. Sem um extrator clássico honesto, "a U-Net funciona" seria afirmação sem contrafactual. Ele importa `numpy` e `cv2`, **nunca `torch`**, e um teste assevera isso.

---

## 3. Estágio B — Calibração

`[[identify.calibrate]]` — determinístico, sem RNG

Este estágio responde uma pergunta só: qual a transformação afim que leva coordenada de pixel a unidade física. São quatro números — `sx, ox, sy, oy` — e obtê-los exige ler os rótulos numéricos dos eixos e descobrir a que posição cada um corresponde. É **o estágio mais frágil do projeto** e o que mais defeitos escondeu.

| Camada | O que faz | Como |
|---|---|---|
| 1. Moldura | Acha o retângulo da área de dados | Varre de baixo para cima a primeira linha "cheia" de tinta — o spine inferior, sempre presente |
| 2. Blobs | Localiza cada rótulo na margem | Componentes conexas após dilatação anisotrópica, com a banda de marcas de tick removida |
| 3. OCR | Lê o número de cada blob | Tesseract em *lote*: recortes num mosaico só, com teto de 12 e recuo individual |
| 4. Ajuste | Converte pares (pixel, valor) na afim | RANSAC exaustivo, depois porta de consistência sobre os *inliers* |

> [!important] Decisão · O blob é a posição, não a marca de tick
> A abordagem intuitiva é detectar as marcas de tick e recortar o texto em volta de cada uma. Ela falha sempre que há ticks menores sem rótulo entre os maiores: o recorte lê o rótulo do tick *vizinho*, e os pares saem com valor certo em posição errada. Medido, essa versão levava só **2 de 30** amostras a uma calibração aceita.

> [!important] Decisão · RANSAC primeiro, consistência depois
> O esboço original checava a consistência dos pares *antes* de ajustar. Com um valor lido errado em cerca de um a cada cinco pares, isso reprovava a amostra inteira por causa de um outlier — exatamente o que o RANSAC existe para descartar.

**Por que RANSAC exaustivo.** São tipicamente ≤ 12 rótulos por eixo, então todos os pares cabem em O(n²). A versão exaustiva é **determinística** — sem amostragem aleatória, sem semente — o que é restrição global do projeto. O desempate entre retas com igual número de *inliers* usa o resíduo total sobre *todos* os pontos; sem isso a ordem de iteração decidia.

> [!important] Decisão · Aprovação por eixo, não conjunta
> A calibração declara `ok_x` e `ok_y` separados. **Só o eixo x já dá a janela em segundos**, e com ela ωn, τ e θ saem em unidade física sem o eixo y. Medido em 900 amostras — exigir os dois: 81,3 %; só o x: 89,3 %; só o y: 89,1 %.

### Três defeitos que só imagens externas revelaram

| Defeito | Mecanismo | Efeito |
|---|---|---|
| Blobs fundidos com as marcas de tick | A faixa começa colada na moldura, então as marcas (1 px, espaçadas ~15,7 px) estão dentro dela. Com dilatação horizontal de 8 px elas se costuram e grudam nos rótulos: o eixo vira *um blob só* | Âncora do 1º tick a ≤3 px: 76,1 % → **99,9 %** |
| Colapso do lote de OCR | Com `--psm 7` o Tesseract devolve zero palavras quando a análise de layout falha — sem exceção. O lote inteiro vira nulo. Não é previsível pelo tamanho: 14 recortes → 11 lidos, 16 → **0**, 20 → 17, 21 → **0** | 16 amostras do corpus sem nenhum par |
| Dois fundos na figura | O fundo era a mediana do quadro, o que assume um fundo só. Com `ax.set_facecolor()` diferente do fundo da figura, a mediana cai no fundo *dos eixos* | bbox devolvia a imagem inteira |

Os dois primeiros afetavam o corpus sintético o tempo todo, diluídos entre outras falhas, sem nunca apontar causa. O terceiro é uma premissa do gerador que o matplotlib real viola. Consertados os três, a cobertura foi de **79,6 % para 93,0 %** e o falso positivo caiu de **55 para 20** — cobertura e precisão subindo juntas, o que nenhum ajuste de limiar tinha conseguido.

---

## 4. Estágio C — Polilinha

`[[identify.polyline]]` — determinístico, sem torch

A máscara é uma região de largura variável; o estágio D precisa de uma função *y(t)* com um valor por instante. O caminho é: **componentes conexas → esqueletização → mediana por coluna → interpolação de vãos**.

> [!important] Decisão · União das componentes, não a maior
> A escolha padrão seria manter a maior componente conexa. Ela é *catastrófica* aqui, porque uma curva tracejada ou pontilhada é, por construção, uma sequência de componentes **desconectadas** — cada travessão é a sua própria componente.
>
> Medido contra a máscara verdadeira: **40 de 300** amostras ficavam com menos de 10 pontos utilizáveis, e o estrato de estilo `:` estourava o alvo de erro.

**Interpolação de vãos não é enfeite.** O estilo pontilhado deixa **43 % das colunas sem tinta nenhuma**. Sem interpolar, quase metade do domínio temporal some. Mas interpolar cegamente é igualmente perigoso: um vão pode ser *traço* (legítimo) ou *falha de extração* (fabricar dado). A discriminação usa a espessura mediana do traço medida antes da esqueletização — um vão maior que a escala do próprio traço é falha, não estilo.

---

## 5. Estágio D — Ajuste

`[[identify.classical]]` — mínimos quadrados não-lineares

Duas estruturas de modelo são ajustadas à série e comparadas:

```
FOPDT      G(s) = K·e^(−θs) / (τs + 1)
2ª ordem   G(s) = K·ωn²·e^(−θs) / (s² + 2ζωn·s + ωn²)
```

A família é deliberadamente pequena. Ela cobre a esmagadora maioria das plantas industriais que um curso de controle trata, e — mais importante — é *identificável* a partir de uma única resposta ao degrau. Ampliá-la teria custo imediato em ambiguidade.

**Como o ajuste é feito**

1. **Chute inicial por integrais, não por tentativa.** Momentos da curva (áreas acumuladas) dão estimativas fechadas de K, τ e θ sem otimização.
2. **SSE perfilado.** K entra *linearmente* no modelo, então é resolvido em forma fechada para cada combinação dos demais. Reduz a dimensão do problema não-linear e elimina uma direção inteira de mínimos locais.
3. **Multistart com refino em dois níveis.** A superfície de erro com atraso é multimodal — θ e τ trocam de papel com facilidade.
4. **Baselines clássicos, mantidos.** Tangente, Smith e Sundaresan–Krishnaswamy existem como referência de comparação no texto do TCC.

> [!important] Decisão · O AIC precisa do número EFETIVO de pontos
> O AIC clássico usa *n* cru, o que pressupõe observações independentes — e aqui não são. A série vem de uma polilinha extraída de imagem, onde pixels vizinhos carregam erro de extração **correlacionado**.
>
> Medido: autocorrelação de defasagem 1 do resíduo ~0,71, e *n* mediano de 738 corresponde a *n* efetivo de **112**. Bastava 0,234 % de ganho de SSE para a 2ª ordem vencer, e ela consegue isso ajustando o *próprio artefato de extração* com o polo extra. Resultado: **32 % das plantas de 1ª ordem** eram classificadas como 2ª ordem, contra 6 % quando o mesmo estágio recebe a série verdadeira.
>
> A correção substitui *n* por `n·(1−ρ)/(1+ρ)`. Trocar AIC por BIC *não* resolveria — corrige só 40 % dos casos, porque o problema não é a constante da penalidade, é o *n* inflado.

**Onde a seleção de ordem ainda falha.** Uma 2ª ordem **criticamente amortecida** (ζ = 1) com atraso é quase indistinguível de uma FOPDT com atraso maior. Num caso real medido, a 2ª ordem tinha SSE apenas **1,5 % melhor** — o ganho no teste ficou em 0,308 contra limiar 2,0, e a estrutura mais simples venceu. Isso não é defeito de implementação: é *identificabilidade*.

---

## 6. Guardas: recusar é melhor que errar com confiança

`[[identify.pipeline]]`

Uma pipeline que sempre responde é pior que uma que às vezes recusa, porque o consumidor não tem como distinguir uma resposta boa de uma ruim.

| Guarda | Limiar | Detecta | Precisão / custo |
|---|---|---|---|
| `ajuste_inconsistente` | nrmse > 0,13 | A série não sustenta modelo nenhum | 88,2 % de precisão, custo 0,24 % |
| `resposta_inversa` | undershoot > 0,08 | Fase não-mínima (zero no semiplano direito) | custo 0,44 %, benefício com n=1 |

> [!failure] Refutada · Descontinuidade da máscara
> A guarda mais intuitiva era medir o maior buraco entre colunas com tinta. Numa imagem externa que errava, o buraco era **19,3 %** contra 5,5 % da segunda pior — separação convincente em n=8.
>
> Contra o corpus de 895 amostras ela **não sobrevive**: Spearman com o erro real de **+0,020** (p = 0,57), e o maior buraco do próprio corpus (20,9 %) é *maior* que o da imagem que errava.
>
> O motivo é estrutural e vale mais que o resultado: **o corpus não contém o modo de falha que essa guarda existia para pegar**, então ele media só o custo dela, nunca o benefício.

> [!warning] Um limiar calibrado com um exemplo é frágil a mudanças a montante
> O limiar de *undershoot* era 0,10, medido com o modelo anterior. Ao promover uma máscara melhor, a métrica mudou de escala — ela é normalizada pela faixa de *y* capturada, que é o **denominador** — e a mesma imagem caiu de 0,143 para 0,0916, furando o limiar por baixo e virando resposta confiante e errada. Recalibrado para 0,08. Qualquer mudança no Estágio A exige remedir este número.

---

## 7. O gerador é parte da arquitetura, não uma ferramenta auxiliar

`[[dataset.generator]]` · `[[dataset.randomize]]`

Treinar a U-Net exige pares imagem/máscara com verdade exata. Nenhum acervo real oferece isso, então o corpus é sintetizado: sorteia-se um sistema, renderiza-se a figura com matplotlib, e a máscara sai de um **segundo render contendo só a curva**. A verdade é exata por construção, não anotada.

> [!important] Decisão · O estilo visual não pode ver o sistema
> A função que sorteia o estilo visual **não recebe** a especificação do sistema. Ela fisicamente não pode ver o rótulo, logo nenhum atributo visual pode se correlacionar com ordem, K, τ, θ, ωn ou ζ.
>
> Sem essa separação, a rede poderia aprender atalhos — "figuras com grade tendem a ser de 1ª ordem" — e obter acurácia alta no corpus que desaparece no mundo real. Um teste assevera que a assinatura da função continua sendo `(rng)` e nada mais.

**Determinismo bit a bit.** A mesma semente produz os mesmos bytes de `image.png` e `mask.png`. Sem ele, nenhuma comparação entre rodadas de treino é interpretável, porque a diferença medida poderia vir do dado e não do modelo.

### Estratos: o corpus é onde os defeitos aparecem ou se escondem

Cada defeito que uma imagem real expôs virou um **estrato opt-in** no gerador, com o caminho padrão preservado byte a byte.

| Estrato | Fenômeno | Degradação medida |
|---|---|---|
| `reta_no_patamar` | Reta de referência coincidente com o patamar, em cor de luminância colidente | motivou a mudança para RGB |
| `janela_assentada` | Janela longa o bastante para o patamar ser visível | eixo separado, permite ablação |
| `banda_de_acomodacao` | Faixa sombreada de ±5 % em torno do setpoint | IoU −0,063 (p = 2,6e−10) |
| `anotacao_com_seta` | Caixa de texto ligada por seta ao pico | IoU −0,113 (p = 1,6e−11) |
| combinados | Os dois juntos | IoU −0,190 · **31 % da mediana** |

O critério para acrescentar um estrato é que ele *degrade o modelo atual de forma mensurável*. Um estrato que não move a métrica não reproduz o fenômeno.

**Como a saturação do modelo foi testada, e não suposta.** Três modelos de 4,37 M de parâmetros treinados com combinações diferentes de estratos **trocavam** um caso real pelo outro. A hipótese — capacidade esgotada — foi testada elevando os canais base de 24 para 32 (7,76 M), sem mudar mais nada. O modelo maior é o **primeiro que acerta as duas ao mesmo tempo**, e o ganho em IoU de validação foi de apenas +0,005.

---

## 8. Mapa de módulos e dependências

| Módulo | Linhas | Papel | Depende de |
|---|---:|---|---|
| [[identify.pipeline]] | 572 | Cola dos estágios. **Única porta de entrada** | todos os abaixo |
| [[identify.classical]] | 950 | Estágio D — ajuste e seleção de estrutura | numpy, scipy |
| [[identify.calibrate]] | 746 | Estágio B — moldura, blobs, OCR, afim | cv2, pytesseract |
| [[identify.extract]] | 139 | Estágio A — U-Net e inferência | torch |
| [[identify.extract_classical]] | 184 | Estágio A alternativo, sem rede | cv2 · *nunca torch* |
| [[identify.polyline]] | 155 | Estágio C — máscara → série | cv2, skimage |
| [[dataset.generator]] | 709 | Síntese do corpus e da verdade | matplotlib |
| [[dataset.randomize]] | 396 | Sorteio do estilo visual, cego ao sistema | numpy |
| [[train_unet]] | 172 | Treino, com estratos acumuláveis por CLI | torch |
| [[identificar]] | 204 | Casca de linha de comando para uso avulso | — |

A direção das dependências é intencional. `pipeline.py` conhece todos; **nenhum estágio conhece outro**. `polyline.py` não importa `torch`, o que permite testar o estágio C contra máscaras verdadeiras sem GPU — e essa separação foi o que permitiu descobrir que a métrica de erro do estágio C estava medindo declividade e não geometria.

A troca de extrator é um parâmetro: `identify_from_image(img, model, device, extractor=...)` aceita qualquer função que devolva máscara `uint8` 0/255.

---

## 9. Envelope de validade e limites conhecidos

| Parâmetro | Faixa treinada |
|---|---|
| K | 0,2 a 20 (log-uniforme) |
| τ (1ª ordem) | 0,05 a 50 s |
| ωn (2ª ordem) | 0,02 a 20 rad/s |
| ζ | 0,10 a 3,00 |
| θ | 0,05 a 1,0 × constante de tempo dominante |
| janela | θ + 0,5 a 6 × constante dominante |

### Fora do envelope

- **Fase não-mínima** — *recusa correta.* Não pertence à família. A guarda de resposta inversa detecta e recusa.
- **ζ = 0 exato** — *viés conhecido.* O piso de `ZETA_BOUNDS` é 1e−3, então ζ = 0 verdadeiro sai como 0,006. Não é medição, é o otimizador encostando na borda da caixa.
- **3ª ordem ou superior** — *sem guarda.* Será forçada na estrutura de 2ª ordem.
- **Zeros no semiplano esquerdo, ganho negativo** — *sem guarda.* Passariam em silêncio.
- **Ordem ambígua** — *sem sinalização.* Hoje a pipeline escolhe a mais simples sem avisar; deveria declarar a ordem como incerta.
- **Parâmetro na borda da caixa** — *sem guarda.* Uma planta instável devolveu `K` no teto exato e `ζ` no piso exato com `nrmse` de 0,032 e `ok=true`.

> [!caution] Uma ressalva metodológica sobre a validação externa
> Treze imagens produzidas fora do gerador foram testadas, e dez saem corretas ou são corretamente recusadas. Mas **cinco delas foram usadas para diagnosticar** os defeitos que depois foram consertados. Elas serviram de diagnóstico e agora servem de validação, o que infla o resultado. Uma validação limpa exige imagens novas, geradas depois de todos os consertos e nunca vistas durante o desenvolvimento. Registrar isso é parte do resultado, não uma nota de rodapé.

---

*Documento gerado a partir do código em `identify/` e `dataset/`. Os números vêm de medições sobre o corpus de teste (n = 900) e da suíte automatizada, rastreados nos handoffs do projeto.*
