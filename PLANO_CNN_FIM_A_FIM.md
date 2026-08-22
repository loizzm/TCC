# Plano alternativo — CNN 2D fim-a-fim

> **Status:** planejado, não iniciado. **Papel:** *baseline* comparativo do
> `PLANO.md §PARTE 3`, critério **3.9**. Não é a arquitetura do sistema.
>
> **Spec:** `PLANO.md §1.2` (Decisão B, que este documento existe para testar),
> `§1.3` (Decisão C, de onde a cabeça multi-tarefa é reaproveitada), `§1.7`
> (Decisão E, que define o nível de comparação), `§PARTE 3` critério 3.9.
> Referências que sustentam os argumentos: `REFERENCIAS.md`.

---

## 1. Por que este documento existe

O `PLANO.md §1.2` rejeitou a CNN fim-a-fim com quatro argumentos escritos. Argumento
escrito é hipótese, não resultado. Ao fim da Parte 2 o dataset de 6.000 imagens já
existe em disco, com máscara e `meta.json`, e o conjunto OOD já foi coletado — ou
seja, **o custo marginal de testar a hipótese é baixo e o valor de defesa é alto**.

Dois objetivos, nesta ordem de importância:

1. **Converter a Decisão B de argumento em medição.** Em banca, "medimos as duas e a
   fim-a-fim degradou 3× mais fora da distribuição" é resposta; "escolhemos o pipeline
   em estágios porque a invariância é estrutural" é opinião fundamentada.
2. **Testar a hipótese de vazamento pelo seu sintoma observável.** A afirmação central
   do §1.2 é que uma CNN fim-a-fim explora atalhos visuais de forma indetectável. Isso
   não é diretamente observável — mas tem uma assinatura que é: **desempenho bom no
   sintético e ruim fora da distribuição**. Se a fim-a-fim empatar no sintético e cair
   no OOD, o atalho está medido, mesmo sem ser visto.

E um objetivo de fundo: o Estágio C removido pelo `§1.3` deixou uma especificação boa
de cabeça multi-tarefa (três cabeças, parametrização log/logit, perda mascarada). Ela
não é desperdiçada — é exatamente a cabeça que este modelo usa. O que muda é o tronco:
uma CNN 2D sobre a imagem em lugar de uma CNN 1D sobre a série.

---

## 2. A limitação estrutural que define o nível da comparação

**Uma CNN fim-a-fim não pode entregar unidades físicas.** Isso não é escolha de
projeto, é consequência da arquitetura:

- os parâmetros dimensionais só são recuperáveis lendo os rótulos numéricos dos ticks
  (`PLANO.md §1.1`);
- os rótulos são texto de poucos pixels de altura;
- o tronco convolucional opera sobre a imagem reduzida (224² ou 256²), onde o texto do
  tick já não existe — foi destruído pela reamostragem.

Alimentar o modelo em resolução nativa para preservar o texto elevaria a memória a um
patamar impraticável na GPU de 6 GB, e ainda exigiria que a rede aprendesse OCR
implicitamente a partir de 6.000 exemplos — tarefa para a qual esse volume é
irrisório.

**Consequência para o experimento:** a comparação acontece no **nível adimensional** da
Decisão E (`PLANO.md §1.7`) — `order`, ζ, ωₙ·T, θ/T, K/y_faixa. É a saída que o
pipeline em estágios também produz sempre, então a comparação é justa e bem definida.

Isso é, por si só, um resultado a registrar na monografia: a arquitetura fim-a-fim é
**estruturalmente incapaz** de cumprir o requisito de saída em unidades físicas, que é
a Decisão A do trabalho. O baseline não é apenas pior — ele resolve um problema menor.

---

## 3. Prós

| # | Prós | Peso |
|---|---|---|
| P1 | **Simplicidade estrutural máxima.** Um modelo, uma perda, um treino, uma inferência. Sem máscara, sem polilinha, sem OCR, sem RANSAC, sem otimizador, sem adaptador entre estágios. Menos código, menos interfaces, menos lugares para errar | alto |
| P2 | **Nenhuma dependência de infraestrutura clássica.** Dispensa Tesseract, `cv2` e `scikit-image`. A pilha inteira é `torch` + `numpy` | médio |
| P3 | **Não herda erro de estágio anterior.** No pipeline, um erro de segmentação contamina a calibração e a identificação em cascata. Aqui não há cascata | médio |
| P4 | **Latência previsível e baixa.** Uma passada direta, sem otimização iterativa. O `least_squares` do estágio D tem cauda longa (máx 267 ms medidos, contra mediana de 55 ms); uma CNN tem tempo praticamente constante | médio |
| P5 | **Aprende invariâncias que ninguém especificou.** O pipeline é robusto ao que foi projetado para ser robusto. Uma rede pode absorver variações que não estão no plano — desde que estejam no treino | baixo |
| P6 | **É a arquitetura que a literatura de extração de dados de gráficos mais usa.** Comparar contra ela é comparar contra a prática corrente, não contra um espantalho | médio |

**Sobre o P5, com honestidade:** é o argumento mais sedutor e o menos confiável. "Pode
absorver variações que estão no treino" é o mesmo mecanismo que absorve atalhos que
estão no treino. É a mesma moeda: capacidade de aprender correlações não especificadas.

---

## 4. Contras

| # | Contras | Peso |
|---|---|---|
| C1 | **Não entrega unidades físicas** (§2 acima). Viola a Decisão A por construção | **eliminatório** |
| C2 | **Vazamento de rótulo indetectável.** Sem representação intermediária inspecionável, um atalho aprendido se manifesta como métrica boa, não como erro. Neste projeto isso não é hipotético: o `img.py` legado vazava, e um GBM só com atributos visuais chegava a 93% de acurácia. Os critérios 1.4d/1.4e fecharam o elo pixel→meta→estilo justamente porque havia artefato inspecionável | **eliminatório** |
| C3 | **Sem atribuição de erro.** Um MAPE de 15% em τ é um número sem diagnóstico: não se sabe se a curva foi mal percebida, se a escala foi mal lida ou se o ajuste falhou. O pipeline decompõe; este não | alto |
| C4 | **Invariância por esperança, não por construção.** A invariância a resolução, paleta, grade e legenda só existe na medida em que o treino a cobriu. É propriedade empírica, e falha silenciosamente fora da distribuição | alto |
| C5 | **Volume de dados desfavorável.** 6.000 imagens é pouco para regredir 4 parâmetros contínuos a partir de pixels. O pipeline treina a U-Net com o mesmo volume, mas para um problema muito mais fácil (segmentação binária com verdade de terra perfeita e ~3.400 pixels positivos por amostra como sinal denso) | alto |
| C6 | **Precisão numérica ruim onde ela importa.** Redes são medíocres em precisão fina. O estágio D atinge MAPE 0,0000% na série limpa; nenhuma regressão direta chega perto disso | alto |
| C7 | **Depende de GPU.** Contraria a §1.8, que tirou a GPU do caminho crítico. Como baseline isso é aceitável — se não treinar, o critério 3.9 fica "não medido" e a §1.2 volta a ser argumento | médio |
| C8 | **Não trata a ambiguidade estrutural com honestidade.** O §1.3 mostrou que em ζ ≥ 2,2 as duas estruturas são indistinguíveis, e a resposta certa é reportar o limite de informação. Uma cabeça de classificação treinada com entropia cruzada vai aprender a chutar a classe majoritária do estrato e reportar confiança alta — o oposto de declarar ignorância | médio |

---

## 5. Arquitetura proposta

Deliberadamente **a mais favorável possível** ao baseline. Um baseline enfraquecido de
propósito não prova nada — se o pipeline vai ganhar, tem de ganhar da melhor versão
razoável do concorrente.

```
image_rgb  uint8[H,W,3]
   │
   ├─ letterbox para 256×256, preservando razão de aspecto     (reusa identify/extract.py)
   │
   ▼  float32[B,3,256,256]
ResNet-18, pesos do ImageNet, primeira camada mantida em 3 canais
   │
   ▼  float32[B,512]   (após global average pooling)
   ├──► cabeça 1: classificação de ordem            2 logits
   ├──► cabeça 2: regressão FOPDT      (log K/y_faixa, log τ/T, θ/T)
   └──► cabeça 3: regressão 2ª ordem   (log K/y_faixa, log ωₙ·T, logit ζ/3)
```

**Três escolhas e as razões.**

*ResNet-18 com pesos do ImageNet, não uma CNN do zero.* Transferência de
características de baixo nível — bordas, cantos, texturas — é gratuita e ajuda com
6.000 amostras. Treinar do zero seria enfraquecer o baseline artificialmente (C5).

*Letterbox, não redimensionamento anisotrópico.* Distorcer a razão de aspecto altera a
geometria da curva, que é precisamente o sinal. Mesmo argumento e mesma implementação
do estágio A.

*As três cabeças e a parametrização log/logit vêm do Estágio C removido.* Regredir em
log torna o MSE equivalente ao erro relativo, impedindo que amostras de ganho alto
dominem o gradiente; `logit(ζ/3)` torna a rede estruturalmente incapaz de emitir ζ
fora de (0, 3). A perda é a mesma: `CE(ordem) + λ·MSE(cabeça da classe verdadeira)`,
com a cabeça da classe errada mascarada.

**Os alvos são adimensionais** (`K/y_faixa`, `τ/T`, `ωₙ·T`), por causa do §2. `T` é a
extensão da janela temporal e `y_faixa` a amplitude vertical do desenho — ambos lidos
do `meta.json` no treino. Na inferência não são conhecidos, e é exatamente por isso
que a saída física é inacessível.

---

## 6. Passos de implementação

Tudo em `e2e/`, **fora** do pacote `identify/`. É experimento comparativo, não
componente do sistema, e nada em `identify/` deve poder importá-lo.

### Bloco E0 — Preparação e portões de honestidade

- [ ] **Passo 1: criar `e2e/dataset.py`** — `Dataset` do PyTorch sobre `data/train`,
      `data/val`, `data/test`, os **mesmos splits** e as mesmas seeds da Parte 2.
      Usar splits diferentes invalidaria a comparação.

- [ ] **Passo 2: escrever o teste que garante que os alvos são adimensionais.**
      Este é o portão de honestidade do experimento: se um alvo dimensional escapar,
      a rede aparentemente "resolve" o problema físico e a comparação fica sem
      sentido. Asserte que todo alvo é invariante a uma mudança de escala de eixo.

- [ ] **Passo 3: escrever o teste que garante que o bloco `render` não entra.**
      A mesma regra anti-vazamento da Parte 1 (`PLANO.md §2`). O `meta.json` traz
      `render`, e o `Dataset` **não pode** tocá-lo.

- [ ] **Passo 4: registrar a contagem de parâmetros e a memória de pico**, para a
      tabela de custo comparativo da monografia.

### Bloco E1 — Treino

- [ ] **Passo 5: `e2e/model.py`** — a arquitetura da §5.
- [ ] **Passo 6: `e2e/train.py`** — Adam, `lr` 3e-4 com *cosine decay*, lote 16 (limite
      de 6 GB), *early stopping* na perda de validação, semente fixa registrada.
      Aumento de dados **nenhum**: o gerador já randomiza estilo, e acrescentar
      aumento aqui mudaria a distribuição efetiva de treino em relação à da U-Net,
      quebrando a comparação.
- [ ] **Passo 7: calibrar λ** medindo as magnitudes dos gradientes das duas parcelas
      na primeira época. Registrar o valor e o método.
- [ ] **Passo 8: treinar até convergir e salvar a curva de aprendizado.**

### Bloco E2 — Avaliação e comparação

- [ ] **Passo 9: avaliar no teste sintético**, nas mesmas métricas adimensionais do
      pipeline. Estratificar por ζ, exatamente como o critério 3.2.
- [ ] **Passo 10: avaliar no conjunto OOD.** É o passo que decide o experimento.
- [ ] **Passo 11: calcular a razão de degradação** `métrica_OOD / métrica_sintético`
      para as duas arquiteturas. **Esta razão é o resultado**, não os valores
      absolutos: ela mede quanto cada arquitetura depende de a distribuição de teste
      parecer com a de treino, que é a assinatura observável do atalho aprendido.
- [ ] **Passo 12: teste de atalho por oclusão.** Zerar a região da legenda, depois a
      da grade, depois as linhas distratoras, e medir a queda de desempenho de cada
      arquitetura. O pipeline deve ser praticamente insensível (nada disso entra na
      máscara); se a fim-a-fim degradar, o atalho ficou localizado — o que é evidência
      direta, não circunstancial.
- [ ] **Passo 13: escrever `HANDOFF_E2E.md`** com o veredito e as tabelas.

---

## 7. Critérios

| # | Critério | Alvo |
|---|---|---|
| E.1 | Alvos comprovadamente adimensionais | teste do Passo 2 passa |
| E.2 | Bloco `render` não acessado pelo `Dataset` | teste do Passo 3 passa |
| E.3 | Splits idênticos aos da Parte 2 | igualdade exata dos `sample_id` |
| E.4 | Desempenho no teste **sintético**, nível adimensional | sem alvo — é resultado |
| E.5 | Desempenho no **OOD** | sem alvo — é resultado |
| E.6 | **Razão de degradação OOD/sintético**, pipeline × fim-a-fim | **a favor do pipeline** — é o teste da §1.2 (critério 3.9) |
| E.7 | Sensibilidade à oclusão de legenda / grade / distratores | pipeline praticamente insensível; a fim-a-fim é o que se mede |
| E.8 | Custo: parâmetros, memória de pico, tempo de treino, latência | reportado nas duas arquiteturas |

Os critérios E.1 a E.3 são portões de validade: **se qualquer um falhar, o experimento
não tem valor e nada do que ele produzir pode ir para a monografia.** Faça-os primeiro.

---

## 8. O que fazer com cada resultado possível

Um experimento cujo resultado não muda decisão nenhuma não vale o tempo. Os quatro
cenários, decididos de antemão:

| resultado | interpretação | ação |
|---|---|---|
| pipeline ganha no sintético **e** no OOD | a §1.2 estava certa nos dois planos | reportar; a Decisão B passa a ser medida |
| empate no sintético, **pipeline ganha no OOD** | **o cenário previsto pela §1.2.** A fim-a-fim aprendeu atalhos que não sobrevivem à mudança de distribuição | reportar como o resultado central do critério 3.9, com o teste de oclusão como evidência de mecanismo |
| fim-a-fim ganha no sintético, **perde no OOD** | idem, com o atalho ainda mais evidente | idem, e o contraste fica mais forte |
| fim-a-fim ganha nos **dois** | a §1.2 estava errada para este problema | **reportar honestamente.** Não muda a arquitetura entregue — o C1 (não entrega unidades físicas) continua eliminatório para o objetivo do trabalho — mas entra na discussão e em "trabalhos futuros" como caminho promissor caso o requisito de unidade física seja relaxado |

O quarto cenário é o que exige compromisso escrito antes de medir. Ele é improvável
pelos argumentos da §4, mas se acontecer, o registro honesto vale mais que a defesa
da decisão anterior. É a mesma regra que já governa a suíte da Parte 1: **a medição
mede a realidade; ela não existe para confirmar o plano.**

---

## 9. Custo e condições de desistência

| item | estimativa |
|---|---|
| implementação (E0 + E1) | 0,5–1 dia |
| treino, com GPU | 0,5–2 h por rodada |
| treino, só CPU | 8–15 h por rodada — **inviável para iterar** |
| avaliação e tabelas (E2) | 0,5 dia |

**Condições de desistência, decididas de antemão:**

1. **Sem GPU, não faça.** O baseline em CPU permitiria uma única rodada, e uma rodada
   sem ajuste de hiperparâmetro produz um baseline artificialmente fraco — o que
   inverte o sinal do experimento e é pior que não medir. Nesse caso, registre o
   critério 3.9 como **"não medido, por falta de GPU"** e mantenha a §1.2 declarada
   como argumento, não como resultado. Isso é honesto; um baseline enfraquecido não é.
2. **Se a Parte 2 atrasar, corte isto primeiro.** É o item de menor risco de corte no
   trabalho inteiro: a §1.2 permanece defensável como argumento fundamentado, e a
   ausência do baseline é uma limitação a declarar, não um furo.
3. **Não corte os portões E.1–E.3 para ganhar tempo.** Eles são baratos e são o que
   separa um experimento de uma anedota.
