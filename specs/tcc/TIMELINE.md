# TIMELINE — decisões e problemas do TCC-2

Registro cronológico único do projeto: **o que quebrou, o que foi medido, o que
foi decidido e o que a decisão custou**. Substitui os dez arquivos `HANDOFF*.md`,
condensados aqui em 04/09/2026.

Este arquivo é **histórico e append-only**: cada bloco novo entra no fim, e o que
já está escrito só muda para registrar uma **retratação** (com o texto antigo
marcado, nunca apagado — foi assim que os handoffs funcionavam e é o que dá valor
ao registro). O que está **aberto** vive em [`nextSteps.md`](nextSteps.md), que é
reescrito o tempo todo.

Contexto de projeto (o que o sistema é, como rodar, arquitetura, bibliografia):
`README.md`, `ARQUITETURA.md`, `PLANO.md`, `PLANO_PARTE2.md`, `REFERENCIAS.md`.

---

## 0. Como ler este arquivo

### 0.1 Os `Ruling`s, e a armadilha da numeração

Um **Ruling** é uma divergência entre o plano e a realidade, resolvida **com
medição** e registrada com o número que a resolveu. É a unidade de decisão do
projeto e o material direto da seção de metodologia da monografia.

**A numeração é irregular, e isso morde quem lê referências cruzadas:**

| Faixa | Escopo | Onde nasceu |
|---|---|---|
| `Ruling C, J, K, L, N, O, H→P, I→R→S, Q` | Parte 1, letras | tabela §6 do handoff da Parte 1 |
| Rulings 1–4, 1–2, 1–7, 1–4, 1–3, 1–4 | **locais ao bloco** | Blocos 0, 1, 2, 3b, 4, 5 — cada um recomeça em 1 |
| **Rulings 1–10** | **globais** | Bloco 3 (a U-Net). É esta sequência que continua adiante |
| Rulings 11–17 | globais | Bloco 6 |
| Rulings 18–53 | globais | Bloco 7 |
| Rulings 54–62 | globais | Bloco 8 |
| Rulings 63–66 | globais | Bloco 9 |

Ou seja: **"Ruling 4" sozinho é ambíguo** (existe no Bloco 0, no 2, no 3, no 3b e
no 5), enquanto "Ruling 30" é único. Neste arquivo os locais aparecem sempre
qualificados (`B2/Ruling 4`); os globais, pelo número puro.

### 0.2 Os handoffs aposentados

Os dez arquivos foram removidos da árvore, **não do histórico**. Cada um continua
recuperável por inteiro:

```bash
git show bc41fea:HANDOFF_P2_7.md          # o maior, 4.416 linhas
git show bc41fea:HANDOFF.md               # Parte 1
```

| Arquivo | Bloco | Rulings | Data |
|---|---|---|---|
| `HANDOFF.md` | Parte 1 | letras C…S | 22/08/2026 |
| `HANDOFF_P2_0.md` | 0 — infra | B0/1–4 | 22/08 |
| `HANDOFF_P2_1.md` | 1 — geometria dos eixos | B1/1–2 | 22/08 |
| `HANDOFF_P2_2.md` | 2 — OCR, RANSAC, consistência | B2/1–7 | 22/08 |
| `HANDOFF_P2_3.md` | 3 — U-Net e treino | **1–10 (globais)** | 22–24/08 |
| `HANDOFF_P2_3b.md` | 3b — extrator clássico | B3b/1–4 | 22/08 |
| `HANDOFF_P2_4.md` | 4 — polilinha | B4/1–3 | 22/08 |
| `HANDOFF_P2_5.md` | 5 — integração e relatório | B5/1–4 | 22–24/08 |
| `HANDOFF_P2_6.md` | 6 — triagem capacidade × dados | 11–17 | 25/08 |
| `HANDOFF_P2_7.md` | 7, 8 e 9 | 18–66 | 26/08–04/09 |

**Comentários de código ainda citam `HANDOFF_P2_7 §40.9` e afins.** As seções
`§NN` referenciadas por eles não existem mais na árvore; use a tabela de Rulings
acima para achar o assunto aqui, ou o `git show` para o texto literal.

---

# PARTE 1 — gerador saneado e identificação clássica

**Concluída em 22/08/2026, 33/33 testes verdes.** O portão dos sete critérios
1.1–1.7 está fechado, com os números em `reports/part1_metrics.md` (arquivo
**gerado** pela suíte — não editar à mão).

## P1.1 O problema de origem: o `img.py` vazava o rótulo

O gerador legado (`img.py`, preservado na raiz como **evidência, não como
código**) desenhava `axhline(K)`, `axvline(θ)` e coloria a curva por tipo de
amortecimento. Por isso "acertava" 93 % — estava lendo a resposta escrita na
própria imagem.

**Decisão estrutural, não uma promessa:** `sample_style(rng)` **não recebe** o
`SystemSpec`. Ela fisicamente não pode ver o rótulo, e sistema e estilo vêm de
streams de RNG independentes (`SeedSequence.spawn(3)`).

**Contraprova medida:** um GBM que só vê atributos visuais fica em **0,4985** de
acurácia (n=20.000) — o acaso.

## P1.2 As decisões que mudaram a FORMA de um critério

Vários critérios do `PLANO` eram estatisticamente ou fisicamente impossíveis de
verificar como escritos. **Nenhum alvo foi afrouxado sem que a grandeza medida
fosse trocada por uma bem posta.**

| # | O que o PLANO pedia | Problema medido | O que passou a valer |
|---|---|---|---|
| **C** | MAPE < 1 % / < 5 % sobre toda a distribuição | a janela é sorteada entre 0,5× e 6× a constante dominante; a 0,5× a curva não determina K a 1 % nem com informação perfeita | assertiva no estrato `w ≥ 3`; truncado medido e reportado |
| **J** | MAPE de θ | θ é sorteado até 0,05·T_dom, então o MAPE mede o piso de sorteio | erro normalizado \|θ̂−θ\|/T_dom, com o MAPE ao lado |
| **K** | `T_dom = 1/(ζωn)` | **errado para ζ>1**: para ζ=3, ωn=1 dava 0,333 s contra 5,83 s reais | `T_dom = (ζ+√(ζ²−1))/ωn` para ζ>1 |
| **L** | (convenção de SNR indefinida) | dois agentes escolheriam convenções diferentes | potência do sinal = **variância** |
| **N** | MAPE < 5 % em ωn e ζ | não-identificabilidade prática comprovada em ζ ≥ 1,6 | ωn/ζ assertados só em ζ<1,6; acima, K, polo lento e NRMSE |
| **O** | \|ρ\| < 0,05 (n=300) | o erro padrão de ρ é 0,058 e a estatística é o **máximo sobre ~154 pares** — sob independência perfeita até \|ρ\|<0,20 é excedido em 60 % das réplicas | teste de permutação (p > 1e-3) + round-trip exato |
| **H→P** | distância ponto→polilinha | dominada pela meia-largura do traço (até 4 px), não pela calibração | offset normal assinado + controle negativo (3 px ⇒ 13,5× de piora) |
| **I→R→S** | fração mínima de pixels na máscara | dois denominadores colidiram com tetos físicos | cobertura horizontal ≥ 0,93 da projeção de `t_window` (teto 1,0 por construção) |
| **Q** | `N_DATASET` = 300 | subgrupo ζ<1,6 ficava com n=23 | 600 |

> **O padrão dos erros é sempre o mesmo: limiar fixado sem calcular antes o
> MÁXIMO ATINGÍVEL da grandeza medida.** Aconteceu em 1.4 (ruído amostral do
> máximo), em 1.5 (espessura do traço) e **duas vezes** na sanidade da máscara.
> Três desses erros eram do plano de execução, não do `PLANO.md`, e foram achados
> pelos próprios implementadores **medindo** — inclusive o `T_dom`, que corrompia
> sistematicamente toda a população superamortecida sem que nenhum teste
> acusasse.

## P1.3 A não-identificabilidade prática (resultado, não falha)

Em ζ ∈ [2,2; 3,0], ωn e ζ erram **~85 %** enquanto K erra 0,28 % e a constante do
polo lento erra 1,28 %, com `corr(erro_ωn, erro_ζ) = +0,9997`. É deslizamento ao
longo de uma curva de nível onde a dinâmica observável é a mesma.

**Auditada por ceticismo, não aceita por alegação.** Um teste *oracle-start*
(partir dos parâmetros VERDADEIROS, tolerância 1e-15, 12 restarts) achou SSE
menor em **4/200** séries FOPDT e **1/200** de 2ª ordem; no pior estrato (ζ≥2,2,
n=80, 40 restarts): **0/80**, MAPE idêntico dígito a dígito. Mediana de
`|erro|/desvio_CRLB` entre 0,59 e 0,74, contra 0,674 esperado de um estimador
eficiente. **O estimador está no piso de informação.**

Isto é o que justifica o NRMSE de reconstrução como métrica primária do trabalho.

## P1.4 O teste de mutação — e por que ele autoriza citar todos os outros números

17 mutantes + 1 controle, cada um uma cópia do repositório com uma substituição
exata e a suíte **inteira** rodando. **Encontrou três buracos, todos da suíte,
nenhum do gerador.**

Os dois primeiros têm a mesma causa raiz, e ela é o achado metodológico:

> **Os critérios 1.3 e 1.4a–c nunca abriam a `image.png`.** Mediam correlação
> entre o bloco `render` do meta e o rótulo. Um vazamento que existisse **só nos
> pixels** — que é *literalmente* o defeito do `img.py` — era invisível para a
> suíte inteira.

O terceiro (`_estimate_gain` voltando a `max(y)`) passava porque a única
assertiva sobre os baselines comparava `identify` com o **melhor** deles — piorar
todos de uma vez não quebra a comparação.

Três testes novos fecharam os três, cada um com **poder medido** contra o defeito
injetado (1.4d: 279/279; 1.4e: 143/157; critério G: `max(y)` erra 30,5 % no mesmo
estrato). O 1.4e custou três iterações de limiar, pela mesma razão de sempre
(limiar fixo contra teto variável).

**Corolário que vale para o projeto inteiro: um critério que nunca falha em
nenhum mutante não é um critério, é decoração.**

## P1.5 Regras que sobrevivem a tudo

1. **`img.py` é evidência, não código.** Não editar, não apagar, não importar.
2. **`reports/*.md` são gerados.** Regenerar com `pytest`, nunca editar à mão.
3. **A suíte mede a realidade; ela não existe para passar.** Limiar mal calibrado
   se corrige **com medição**, nunca com afrouxamento.
4. **Nunca `np.random` global**, nunca `time`/`uuid`/hash dependente de
   `PYTHONHASHSEED` — quebram o determinismo bit-a-bit.
5. **Paralelize com processos, nunca threads** — `_estimate_gain` é memoizado em
   estado global de módulo.

## P1.6 Revisão de arquitetura (22/08/2026)

O **Estágio C (estimador neural 1D) foi medido e removido** do plano. O pipeline
passou de quatro para três estágios: **A → B → D**. Duas decisões de robustez
acompanham: OCR opcional (Decisão E) e extrator clássico como contingência de
GPU. O critério **3.12** existe como gatilho de ressurreição do Estágio C —
convergência de `identify` ≥ 99 % e NRMSE p95 ≤ 0,02 sobre as séries
**extraídas**. Nenhum dos dois falhou até hoje.

---

# PARTE 2 — o pipeline de imagem (Blocos 0 a 9)

## Bloco 0 — infraestrutura e a guarda do relatório (22/08/2026)

**Problema que abriu o bloco:** o `PLANO_PARTE2.md` supunha a máquina do
`PLANO.md` (RTX 4050, Fedora). O ambiente real era outro.

- **B0/Ruling 1 — não há timebox de driver NVIDIA porque não há hardware NVIDIA.**
  `lspci` não mostrava controlador NVIDIA. `TCC_DEVICE = "cpu"` por **ausência de
  hardware**, não por falha de driver. *(Esta afirmação foi refutada no Bloco 6 —
  ver Ruling 17. O erro de método está registrado lá, e é dos mais úteis do
  projeto.)*
- **B0/Ruling 2** — ambiente Debian/Ubuntu, `apt` e não `dnf`.
- **B0/Ruling 3 — a guarda do relatório da Parte 1.** O passo literal do plano
  recalculava a string de seleção sem gravá-la de volta; sem uma segunda chamada
  a `record_block("selection", …)` o banner de "relatório parcial" nunca
  dispararia para seleção por caminho.
- **B0/Ruling 4** — `add_noise` ficou no default (`True`), coerente com o
  conjunto "noisy" da Parte 1.

**A armadilha que mordeu de verdade:** rodar o teste sintético da guarda **antes**
de aplicar a guarda **sobrescreveu** `reports/part1_metrics.md`, reduzindo-o de
~500 para 16 linhas. Recuperado com 13 min de `pytest` completo.

**Números:** splits 4.200/900/900, seeds disjuntas, 422 MB, 6m40s de geração. A
máquina media ~6× mais lenta que a da Parte 1 em wall-clock (787 s contra 128 s
na suíte completa) — o primeiro sinal de que o Bloco 3 custaria caro.

## Bloco 1 — moldura e ticks, sem OCR (22/08/2026)

**Problema:** o esqueleto literal do plano (`_edges` + `_long_lines` por
gradiente) dava **13,3 %** de acerto em G1.1, estável entre todas as combinações
de parâmetros testadas — nenhuma passava de 20 %.

- **B1/Ruling 1 — a causa raiz, medida nos pixels.** Quando os spines `right`/`top`
  estão ausentes, o retângulo `plot_bbox_px` é um limite puramente geométrico com
  **nenhum traço desenhado ali**. A única informação recuperável é indireta: o
  matplotlib **recorta o spine inferior exatamente no retângulo dos eixos**, então
  a extensão dele já é `[x0, x1]`. Reescrito para varrer a primeira linha "cheia"
  e usar a extensão *dela mesma*. **Medido: 13,3 % → 99,7 %** (299/300), 98,3 % no
  pior estrato.
- **B1/Ruling 2** — trabalhar em cinza BT.601 (a assinatura publicada) custa 1/300
  amostras: o gerador garante contraste **WCAG sobre sRGB linearizado**, e as
  duas métricas divergem para matizes diferentes com luma parecida. Registrado
  como limite conhecido, não corrigido.

> **A armadilha do bloco, que não é óbvia até medir:** um cruzamento de curva
> imita uma "linha longa" se você só olhar o **alcance** da tinta. Uma resposta
> subamortecida cruza o mesmo nível de `y` em `x` bem afastados, produzindo
> alcance de ~1000 px com **~1 % de preenchimento**. É `SPINE_MIN_FILL = 0,90`
> que separa spine de curva. Qualquer detecção geométrica de linha neste dataset
> **tem que** checar preenchimento, não só alcance.

## Bloco 2 — OCR, RANSAC e consistência (22/08/2026)

**Problema:** com a implementação literal do plano, **2 de 30** amostras
calibravam. Seis correções, cada uma com efeito medido, levaram a 77 %.

| # | Correção | Por que a versão original quebrava | Efeito medido |
|---|---|---|---|
| B2/1 | **RANSAC → consistência**, não o inverso | checar equiespaçamento sobre os pares BRUTOS faz um único valor errado reprovar a amostra inteira — o outlier que o RANSAC existe para descartar | 18/30 → 24/30 |
| B2/2 | **sem whitelist no Tesseract** | `tessedit_char_whitelist` **quebra** o engine LSTM: um "4" nítido voltava vazio com ela e lia certo sem ela | — |
| B2/3 | **ticks bidirecionais** (dentro E fora) | `tick_direction` é sorteado em `{in, out, inout}` e a implementação só olhava fora — perdia ~1/3. **O portão G1.2 não pegou porque mede recall por MEDIANA:** com 2/3 intactas a mediana continuava 1,0 | — |
| B2/4 | **rótulo por blob de texto**, não por marca de tick | recorte de largura fixa centrado no tick lê o rótulo do VIZINHO sempre que há tick sem marca ou sem rótulo | 2/30 → 24/30 |
| B2/5 | **`_equiespacados` tolera LACUNAS** | comparar diferenças consecutivas contra a média reprova quando um tick do MEIO não é lido | eliminou a maioria dos `calibration_failed` |
| B2/6 | **desempate por RESÍDUO TOTAL** | com 3 pares e 1 errado, quaisquer 2 "empatam" em 2 inliers — a **ordem de iteração** decidia. Caso concreto: 20 % de erro de escala numa amostra reportada `ok=True` | 2.3: 76,5 % → 95,65 % em lote de 30 |
| B2/7 | 2.11 cobre só o **contrato** de `calibrate()` | a afirmação completa do PLANO depende de `identify_from_image`, que não existia | marcado parcial |

**Resultado do bloco:** 232/300 (77,3 %) calibram. **Os quatro critérios que
falham (2.3, 2.4, 2.5, 2.9) têm a MESMA causa raiz** — cobertura de OCR — não
quatro problemas independentes.

**Custo de iteração:** o ciclo completo de 300 amostras levava **~2 horas**,
dominado por spawn de subprocesso `tesseract`. Toda a depuração foi feita em
lotes de 20–60. (Este custo vira Ruling 31 e depois Ruling 35, no Bloco 7.)

## Bloco 3b — o extrator clássico (22/08/2026)

Existe para **tirar a GPU do caminho crítico** e para dar à U-Net um número que
justifique sua existência. Se um dia empatar com ela em 2.10, o achado correto é
"a U-Net não se justifica" — não um problema a esconder.

- **B3b/Ruling 1 — escolher o candidato por ÁREA, não por extensão.** Uma reta
  distratora muito pontilhada tinha extensão marginalmente maior que a curva
  (`sample_00001`: modo de 5.097 px e extensão 1.031 perdia para modo de 989 px e
  extensão 1.075 — a distratora ganhava por 44 px de extensão com **5× menos
  tinta**). **Mediana de IoU 0,0 → 0,708.**
- **B3b/Ruling 2 — fechar vãos antes de decidir se é reta de span completo.** Uma
  pontilhada com ciclo de trabalho < 25 % nunca era marcada como reta e sobrevivia
  inteira. `_bridge_gaps_1d` (dilatação 1D, só para a DECISÃO, nunca aplicada à
  saída) fecha vãos de até 25 px. Mediana idêntica, mas **p10 sobe de 0,068 para
  0,384**.
- **B3b/Ruling 3 — `np.unique(axis=0)` sobre tuplas RGB custava 1,9 s/amostra**,
  10× o orçamento. Reescrito com `_bucket_key` + `np.bincount`: **1,93 s →
  0,07 s**; p95 de 203,8 ms (falhando por 3,8) para **128,5 ms**.
- **B3b/Ruling 4** — `_spanning_rows` foi **copiado** e não importado, porque
  `tests/test_leakage.py` arrasta `pytest`, `sklearn` e `tests.conftest` inteiro.

**Números:** IoU mediana **0,7153** (≥ 0,70 ✅), 0 violações de span, latência p95
128,5 ms, e não importa `torch` (verificado em subprocess com `sys.meta_path`
bloqueando o import — leitura de código não bastaria).

## Bloco 4 — máscara → polilinha (22/08/2026)

- **B4/Ruling 1 — não pode ficar só com a MAIOR componente conexa.** O plano usava
  `argmax(area)`. Isso quebra estruturalmente para traço tracejado: o matplotlib
  desenha cada travessão desconectado, então a máscara — mesmo a **verdadeira** —
  tem o traço partido em dezenas de componentes. **Medido antes:** 44/300 amostras
  (14,7 %) com menos de 10 pontos utilizáveis (mediana de 2 a 9 pixels!).
  **Depois** (união de tudo com área ≥ 2 px): **0 amostras**, mediana 2,01 → 1,49 px.
- **B4/Ruling 2 — o p95 de 6,70 px é limitação estrutural.** Os piores casos são
  imagens de proporção extrema (260×1158, 284×1023) onde a curva tem trechos
  quase verticais e "uma mediana por coluna" perde informação. **Alvo não
  ajustado.** *(Isto vira Rulings 43, 46, 48, 49 e finalmente **50** no Bloco 7,
  que mostra que o critério é que estava errado.)*
- **B4/Ruling 3** — a suíte precisou de um estrato novo (`espessura`) para ter
  poder contra o mutante que pula `skeletonize`.

## Bloco 3 — a U-Net e as cinco rodadas de treino (22–24/08/2026)

**Este é o bloco de onde vem a numeração global dos Rulings.** Cinco rodadas,
**cada uma motivada por uma causa raiz medida**, não por tentativa cega.

| Rodada | O que mudou | IoU teste | 2.6 pior parâmetro |
|---|---|---|---|
| 1 | LR fixo | 0,544 | +8,00 p.p. |
| 2 | + `ReduceLROnPlateau` | 0,572 | +8,10 p.p. |
| 3 | alvo com limiar 0 | 0,495 (**piorou**) | +3,92 p.p. |
| 4 | alvo com limiar 32 | 0,560 | **+3,64 p.p.** ← melhor 2.6 |
| 5 | alvo **contínuo** | **0,6205** ← melhor IoU | +3,73 p.p. |

- **Ruling 1** — contagem real: **1.942.289** parâmetros, não os "~1,2 M" do
  PLANO. Não é desvio de implementação: é a contagem da arquitetura exatamente
  como descrita.
- **Ruling 2** — **512² abandonado sem completar 1 época** (parado aos 95,5 min),
  disparando a regra do plano. 256² mede 29,3 min/época.
- **Ruling 3/5 — o platô é real, e a decisão de interromper foi tomada com
  números.** Sete épocas seguidas (6 a 13) oscilando entre 0,6522 e 0,6734, sem
  tendência. Os parâmetros do scheduler **não foram palpite**: foram medidos
  rodando a lógica de decisão, fora do laço de treino, contra a sequência real de
  14 IoUs, comparando `patience ∈ {1,2}` × `threshold ∈ {1e-4 rel, 0,005 abs,
  0,01 abs}`.
- **Ruling 6 — o scheduler ajudou (+2 pontos) e não resolveu.** Nas últimas épocas
  o LR estava em 5,9e-7 — seis ordens abaixo do inicial — e o IoU não passou de
  0,685. **Isso descarta taxa de aprendizado como único fator.**
- **Ruling 7 — a causa do platô era o ALVO de treino, medido sem treinar nada:**

  | Preparo do alvo | Cobertura de colunas (mediana) | Pior caso (p10) |
  |---|---|---|
  | máscara original | 96,6 % | 73,1 % |
  | `INTER_AREA` + limiar 127 (o que estava rodando) | 68,8 % | **0,4 %** |
  | `INTER_AREA` + limiar 1 | 100 % | 95,0 %+ |

  Em imagens grandes (**68 % do conjunto**) o `letterbox` reduz até ~6,25×; uma
  linha de 1–2 px vira média bem abaixo de 127 e **desaparece do alvo**. A rede
  treinava, numa fração real do dataset, contra rótulos que já tinham perdido a
  informação.
- **Ruling 8 — o limiar 0 resolveu o sumiço e superajustou para o lado oposto.**
  IoU_val disparou para 0,9055 e o IoU de **teste caiu** de 0,572 para 0,495: com
  `INTER_AREA`, qualquer bloco que toque a curva produz valor > 0, e aceitar
  qualquer valor > 0 captura o halo inteiro. Área do alvo **1,47× a 3,03×** maior
  que a esperada. **Não era overfitting — a régua de treino e a régua de avaliação
  é que eram diferentes.** Varredura escolheu o limiar 32 (cobertura 85,1 %,
  inflação 2,14×).
- **Ruling 9 — em vez de um quarto limiar, o alvo virou CONTÍNUO.** Depois de três
  limiares com a mesma troca (cobertura × inflação), a pergunta certa deixou de
  ser "qual o próximo limiar". **Resultado: melhor IoU das cinco rodadas e ζ NÃO
  fechou** — piorou 0,09 p.p., na direção oposta da esperada. **A hipótese
  específica não se confirmou.**
- **Ruling 10** — as duas hipóteses que sobram: **(a) capacidade** (`base=24`,
  4,37 M; `base=32`, 7,76 M) e **(b) tamanho do dataset**. Com a alínea **10c**:
  se ζ não fechar de novo, olhar amostra a amostra *quais* casos erram.

**A armadilha silenciosa deste bloco:** `test_2_7_iou_por_estrato` usa `assert`
DENTRO do laço, então para no primeiro estrato que reprova e **nunca chega aos
demais**. "Nenhum estrato < 0,75" nunca foi de fato verificado em todos.

## Bloco 5 — integração e o consolidado (22–24/08/2026)

- **B5/Ruling 3 — a lacuna mais importante da Parte 2: a Decisão E não está
  implementada.** `identify_from_image` devolve só o nível **físico**; o bloco
  `dimensionless`/`physical` do `PLANO §1.7` não existe. **O critério 2.11, na
  leitura estrita, NÃO está fechado** — o que está fechado é "nunca levanta
  exceção". *(Vira Ruling 34, depois 44 e finalmente **45** no Bloco 7.)*
- **B5/Ruling 4** — a guarda do relatório da Parte 1 segurou por toda a Parte 2
  (hash idêntico em dezenas de verificações).

**A medição que justifica a U-Net estar no trabalho.** Em IoU de máscara puro o
extrator clássico **vence** em todas as cinco rodadas (0,7153 × 0,6205). Medindo
2.6 — degradação end-to-end dos parâmetros físicos — a **U-Net vence**:

| Parâmetro | U-Net (rodada 4) | clássico | vencedor |
|---|---|---|---|
| K | +0,92 | +1,03 | U-Net (margem pequena) |
| τ | +2,00 | +1,85 | clássico (margem pequena) |
| θ | +0,55 | +0,78 | **U-Net** |
| ωₙ | +1,78 | **+3,10 (reprova)** | **U-Net** |
| ζ | +3,64 | +4,38 | **U-Net** |

**IoU de máscara e utilidade final para estimar parâmetros não são a mesma
coisa** — e este é o número que prova isso, não a intuição. *(O Ruling 30
depois mostra que parte desta vantagem era ruído de seleção de ordem, e o Ruling
25 que o IoU não prediz o end-to-end de jeito nenhum.)*

**Estado ao fim do Bloco 5: nenhum dos onze critérios numéricos fecha
integralmente.** 2.6 a 0,64 p.p. do alvo. As duas armadilhas de honestidade
registradas: `reports/part2_strata.md` era regenerado do zero a cada processo
`pytest` (refletia só a última seleção), e a suíte completa **nunca foi executada
numa única invocação** — cada bloco validado com `-k` isolado.

## Bloco 6 — a triagem capacidade × dados (25/08/2026)

Ambiente reconstruído em máquina nova. **A linha de base reproduziu os números
documentados** (2.1, 2.7, 2.10 idênticos), o que valida a portabilidade.

- **Ruling 11 — o dataset é reprodutível entre máquinas; os sha256 NÃO são.** RNG
  e geometria idênticos, IoU batendo em quatro casas: **os pixels são os mesmos e
  a diferença é o encoder PNG** (zlib/libpng). O teste de determinismo por hash é
  válido **dentro** de uma máquina, não **entre** máquinas.
- **Ruling 12 — os critérios de OCR são sensíveis à versão do tesseract.** A única
  diferença real na linha de base (173 × 168 amostras comparáveis) vem do
  `tesseract 5.5.3`. Quem comparar OCR entre máquinas precisa registrar a versão.
- **Ruling 13 — a rede SUB-AJUSTA, e o diagnóstico previu a triagem.**

  | Checkpoint | IoU_train | IoU_val | gap |
  |---|---|---|---|
  | rodada 4 | 0,5551 | 0,5511 | +0,0040 |
  | rodada 5 | 0,7502 | 0,7439 | +0,0062 |

  Meio ponto percentual de gap significa que o modelo **não sobre-ajusta — ele nem
  consegue ajustar os dados que já tem**. É viés, não variância. **A previsão foi
  feita ANTES de qualquer treino:** a hipótese (b) não vai ajudar, porque não há
  sobre-ajuste para mais dados combaterem. **Uma medição de 20 minutos previu
  corretamente o resultado de 12 épocas de treino.** As cinco rodadas anteriores
  nunca a fizeram.
- **A triagem fatorial 2×2** (4 pilotos, 3 épocas, 525 passos fixos para isolar
  diversidade de passos de gradiente):

  | | 4.200 | 8.400 | efeito dos dados |
  |---|---|---|---|
  | `base=16` | 0,6794 | 0,6405 | **−0,039** |
  | `base=24` | **0,7445** | 0,7203 | **−0,024** |
  | **efeito da capacidade** | **+0,065** | **+0,080** | |

  **Cada efeito aparece nas duas linhas, com sinal igual e magnitude parecida** —
  não são pontos soltos. Capacidade **APROVADA**, dados **REPROVADOS**. A célula
  de interação existe justamente para que "dados não ajudam" não se confunda com
  "a rede é pequena demais para aproveitá-los".
- **Ruling 14** — a trajetória desta máquina é sistematicamente melhor com o mesmo
  código e os mesmos dados (piloto A1 passou em **3** épocas o que a rodada 5 levou
  25 para atingir). Causa provável: versão do torch. **Não afirmado sem uma rodada
  completa.**
- **Ruling 15** — a triagem mede IoU, não ζ. **Ela autoriza gastar as 21 h; não
  promete o resultado.**
- **Ruling 16 — `base=24` custa 1,71× o `base=16`, não 2,25×.** O Ruling 10a
  estimava pela contagem de parâmetros, mas convolução não escala assim. **O
  experimento sempre foi ~35 % mais barato do que o handoff supunha.**
- **Ruling 17 — esta máquina TEM uma RTX 4050, e nenhum treino do projeto a
  usou.** Dois bloqueios: driver `nouveau` (sem CUDA) e build `torch 2.13.0+cpu`.

  > **O erro de método, registrado para não se repetir:** o registro de ambiente
  > dizia "GPU: nenhuma", inferido de `nvidia-smi` ausente do PATH e de
  > `torch.cuda.is_available() == False`. **Nenhum dos dois mede presença de
  > hardware** — o primeiro mede o driver proprietário, e o segundo era
  > **circular**, porque uma build `+cpu` retorna `False` por construção. A
  > verificação correta é `lspci | grep -i vga`, que lê o barramento. **Nunca
  > concluir ausência de hardware a partir de uma biblioteca compilada sem suporte
  > a ele.**

**Armadilha operacional:** pressão de memória degrada o treino de forma não
óbvia — o piloto C levou 4h37 de parede para 2h39 somadas nas épocas, com o swap
saturado. ~2 h caíram **fora** dos cronômetros por época.

## Bloco 7 — GPU, o fechamento de 2.6, e a revisão dos critérios (26/08/2026)

O bloco mais denso do projeto: Rulings 18 a 53.

### 7.1 A rodada 6 fecha o 2.6

**11,7× mais rápido em GPU** (258 s/época contra 3.030 estimados em CPU; 1,79 h
contra ~21 h). Batch 8 e 512² **preservados** em 4,74 de 5,64 GB, então a
comparabilidade com os pilotos está intacta.

| Parâmetro | Base (rodada 5) | **Rodada 6 (`base=24`)** | Alvo |
|---|---|---|---|
| K | +1,01 | **+0,25** | ≤ 3,00 ✅ |
| τ | +2,44 | **+0,54** | ✅ |
| θ | +0,50 | **+0,27** | ✅ |
| ωₙ | +2,03 | **+0,82** | ✅ |
| **ζ** | **+3,65 ❌** | **+1,05 ✅** | |

- **Ruling 18 — trocar CPU por GPU não move nenhum critério de acurácia.** A linha
  de base foi remedida inteira em GPU e reproduziu a de CPU em **todas as casas
  decimais**. O temor do Bloco 6 era legítimo em princípio e **falso na prática**.
  O único critério que muda é o 2.8 (latência), e esse muda por construção.
- **O controle interno que valida o resultado:** o diff dos relatórios mostra que
  **só os critérios que dependem da U-Net mudaram**. O mais forte é o
  **2.6-clássico: +4,36 p.p., inalterado até a segunda casa** — o extrator
  clássico não usa a rede, logo tinha que ficar parado, e ficou. **A única
  variável que se moveu foi o checkpoint.**
- **Ruling 21 — IoU de máscara é proxy ruim, e o critério 2.10 inverte de sinal:**
  o clássico ganha em IoU (0,7153 × 0,6478) e perde em ζ por fator 4. **Não é
  diferença sutil: é inversão completa de ranking.**
- **Ruling 22 — as 25 épocas foram desnecessárias.** O melhor saiu na época 15, e
  a **época 05 já entregava 99,2 % do resultado final**. Nove cortes de LR não
  destravaram patamar novo.
- **Ruling 23 — as falhas restantes se dividem em DUAS famílias, e o diff prova a
  separação.** Trocar o checkpoint mexeu em 2.1/2.7/2.8 e deixou 2.2-piso, 2.4,
  2.5 e 2.9 **byte a byte idênticos**. **Qualquer plano que trate "os critérios
  que faltam" como um bloco único está errado.**
- **Ruling 24 — a rede AINDA sub-ajusta em `base=24`** (gap +0,0105). Mais dados,
  regularização e *augmentation* seguem **previsivelmente inúteis**.

### 7.2 O Ruling 10c executado: o IoU não prediz nada

- **Ruling 25 — o IoU não prediz o erro end-to-end, e a estimativa pontual é
  ANTICORRELACIONADA.** Erro de ζ por quartil de IoU (n=119): 1,60 % no pior
  quartil de máscara contra **4,22 % no melhor**, monótono nos quatro. Spearman
  +0,192 (p=0,036).

  **Ressalva que impede leitura causal:** o erro do *oráculo* também sobe no Q4, e
  o oráculo não vê a imagem — o Q4 concentra problemas intrinsecamente mais
  difíceis. A correlação da **degradação** não é significativa (p=0,255).
  **Afirmação defensável:** não há evidência de que subir o IoU reduza o erro
  end-to-end. **NÃO afirmar** que IoU alto causa erro alto.

  Evidência independente: as **112 descartadas têm IoU mediano MAIOR (0,6583) que
  as 188 aceitas (0,6456)**.
- **Ruling 26 — a hipótese original do 10c está errada.** Os **mais
  subamortecidos estão entre os melhores** (ζ<0,3: +0,49 p.p.), o oposto do que o
  Bloco 3 supôs. Com n de 9 a 11 por faixa isto **serve para descartar a hipótese,
  não para eleger uma culpada**.
- **Ruling 27 — as 112 descartadas: 103 vêm de duas causas que nenhum treino de
  segmentação alcança** — 51 de ordem errada e 55 de OCR/calibração. Assimetria
  forte: FOPDT aceita 46,3 % e diverge de ordem em 32,2 %; 2ª ordem aceita 78,8 %
  e diverge em 2,0 %. **Não é falta de dado** — as que divergiram têm *mais*
  pontos extraídos. Falha de OCR por dpi **é um U, não uma rampa** (25,9 % /
  12,2 % / 23,0 %), o que **desmonta "upscalar imagens de baixo dpi" como conserto
  suficiente**.

  *Armadilha 7:* `render.bg_color` é hex arbitrário (200+ valores distintos em
  300), não um rótulo `"white"`/`"black"` — derivar "fundo escuro" por comparação
  de string dá 299/300 e está errado.

### 7.3 A confusão de ordem, e o `n` efetivo

- **Ruling 28 — o AIC decide pelo `n` INFLADO da polilinha, não pelo ajuste.**
  Mesmo estágio D, mesmo AIC: o oráculo acerta a ordem em 94,1 % com n mediano 512;
  o pipeline acerta 59,3 % com n mediano **806**. Com n=806 o limiar `exp(2/n)` é
  1,00234 — basta **0,234 %** de ganho de SSE, e o observado é 1,010 %. **Não é
  ganho real de modelo:** NRMSE 0,00353 (fopdt) × 0,00351 (segunda), empate
  técnico. A polilinha tem ~800 pontos que **não são independentes**, o polo extra
  absorve a correlação, e o AIC trata isso como 800 evidências. **Trocar AIC por
  BIC NÃO resolve** — 29 das 48 ainda escolheriam 2ª ordem.
- **Ruling 29 — 2ª ordem forçada recupera K e τ e só custa θ.** ζ atribuído foi
  ≥ 1 em **100 %** dos casos: a 2ª ordem **nunca** inventou um sistema
  subamortecido a partir de um de 1ª, como a teoria prevê.
- **Ruling 30 — o conserto: AIC com `n` efetivo.** `n_eff = n·(1−ρ)/(1+ρ)`, com ρ
  medido no resíduo da estrutura **mais flexível** (num modelo subespecificado a
  correlação mistura ruído com erro de estrutura e superestimaria a correção).

  | Critério | fopdt | second | total |
  |---|---|---|---|
  | AIC com `n` cru | 59,3 % | 97,6 % | 78,8 % |
  | BIC com `n` cru | 75,4 % | 94,3 % | 85,1 % |
  | **`n_eff` (adotado)** | **89,8 %** | 92,7 % | **91,3 %** |

  Troca 6 acertos por 36. `n` mediano 738 → `n_eff` **112,5**. A varredura mostra
  que `n_eff × 0,25` chega a 92,9 %, só 1,6 p.p. acima — **não vale introduzir um
  fator ajustado à mão**, o `n_eff` puro não tem parâmetro livre.

  **Todos os cinco parâmetros melhoraram COM 27 amostras a mais na conta** — o
  oposto do risco esperado. E **o 2.6-clássico virou de ❌ para ✅** (+4,36 →
  +2,42): a confusão de ordem estava no estágio D, **compartilhado** pelos dois
  extratores. **Parte do que o Ruling 25 creditou à U-Net era ruído de seleção de
  ordem.** A correção é **no-op em série verdadeira** (resíduo branco → ρ≈0), e a
  Parte 1 passou 33/33.

### 7.4 A latência: 98,7 % era spawn de processo

- **Ruling 31 — não é OCR, é partida de processo.** `read_tick_labels` = **98,9 %**
  do estágio B; o tesseract dentro dele, 98,7 %. Todo o resto soma **4,5 ms**.
  Medido com recortes sintéticos: **~52 ms fixos por invocação, ~0,3 ms por rótulo
  adicional** (1 rótulo 52,6 ms; 100 rótulos 82,1 ms). Mediana de **16** chamadas
  por imagem, máx 105.
- **Ruling 35 — OCR em lote fecha o 2.8.** Todos os candidatos num mosaico
  horizontal de uma linha só (preserva o `--psm 7`), mapeados de volta pelas
  caixas do `image_to_data`. **A folga do mosaico é o parâmetro crítico e foi
  varrida:** `gap=60` degradou muito (funde dois rótulos numa palavra e o `_NUM_RE`
  rejeita ambos); `gap=200` funde, `gap=700` espalha e o `psm 7` perde a linha;
  **`gap=400` empata com a referência lendo mais pares**. A variante em cascata
  **perdeu** — registrado porque era a hipótese mais "óbvia".

  **2.8: 891 ms → 172 ms (p95 2.325 → 254).** `read_tick_labels` 797,8 → 98,3 ms.
  **Ressalva honesta: NÃO é superconjunto** — 24 amostras viraram falha e 22
  viraram sucesso, saldo −2.

### 7.5 A Decisão E, finalmente implementada

- **Ruling 34 — a Decisão E não está implementada, e o teste que leva o nome do
  critério não testa o critério.** `grep -rn "dimensionless" identify/*.py` → zero
  ocorrências. E `test_2_11_saida_adimensional_sempre_presente`, cuja docstring
  diz *"a saída adimensional existe mesmo com ok=False"*, tinha por corpo
  `assert cal.ok in (True, False)`. **Reportava ✅ 300/300.** O nome que ele
  registrava no relatório era honesto; o problema era **ocupar o slot do 2.11**.
- **Ruling 44 — a Decisão E é implementável, e o ζ recuperado é de qualidade
  comparável:** das 61 amostras sem calibração, o ajuste adimensional converge em
  **61 (100 %)**, com a **ordem certa em 53**. MAPE de ζ: **2,93 %** (contra
  2,40 % do caminho físico nas que funcionam). **Não é resposta degradada; é
  resposta de qualidade comparável em amostras hoje descartadas.**

  **Ressalva que impede tratar os dois caminhos como equivalentes:** concordam no
  caso típico (1,33 % mediano) e **divergem muito na cauda (p95 de 50,5 %)**. O
  ponto fraco é o **zero estimado**. Logo o nível adimensional precisa ser
  reportado como **saída própria, com acurácia própria** — nunca fundido nos
  números do físico. *Esta é a razão de ser da estrutura de dois níveis.*
- **Ruling 45 — implementada, a 5 ms de custo.** Duas decisões de compatibilidade
  deliberadas: `params`/`order` no topo continuam sendo os do nível FÍSICO, e `ok`
  continua significando "há saída física", não "há resposta". **Nenhum critério
  físico se moveu um decimal** — a preservação da semântica de `ok` era o
  principal risco da mudança.

  **Decisão técnica que resolve a ressalva do 44:** quando a calibração fecha, o
  bloco adimensional é **derivado do ajuste físico**, não de um segundo ajuste.
  Economiza um ajuste **e** faz os dois níveis concordarem por construção — a
  divergência de 50,5 % deixa de existir. **Medido: 300/300 com bloco, 61/61 das
  sem calibração.**
- **Ruling 47 — o ζ adimensional entra no 2.6 como critério COMPANHEIRO**
  (`2.6-adim[zeta]`), não substituindo o físico. Fundir os dois apagaria a
  distinção que o `PLANO §1.7` existe para manter e quebraria a comparabilidade
  histórica. **141/300 amostras, sendo 31 sem calibração; +1,53 p.p. ✅.**

### 7.6 O extrator de polilinha: 19 tentativas, e o critério é que estava errado

- **Ruling 38 — Dice não pode ajudar: é identidade com IoU.** `max|Dice −
  2·IoU/(1+IoU)| = 1,11e-16`, transformação estritamente monótona, logo Spearman
  **idêntico**. Dice mede 0,7862 contra 0,6478 e **parece** melhor, mas é o mesmo
  número reparametrizado. **Trocar de métrica de área não escapa do problema de
  área.**
- **Ruling 39 — o IoU aqui mede ESPESSURA DE TRAÇO, não acurácia geométrica.**

  | espessura verdadeira | n | IoU | **centerline** |
  |---|---|---|---|
  | < 4 px | 98 | **0,468** | **1,00 px** |
  | 4–6 px | 60 | 0,609 | **1,00 px** |
  | 6–9 px | 54 | 0,710 | **1,00 px** |
  | ≥ 9 px | 88 | **0,782** | 1,08 px |

  **O erro geométrico é 1,00 px em TODAS as faixas — constante — e o IoU varia
  0,31.** Um modelo trivial (deslocamento numa faixa de largura `w`) prevê IoU
  0,667 contra 0,648 medido, correlação previsto×medido **+0,932**. **O alvo IoU
  ≥ 0,85 é provavelmente inalcançável por construção para traço fino: a meta foi
  fixada sem considerar a largura da linha.**
- **Ruling 40** — a rede acrescenta **+0,359 px** sobre um piso de 1,488 px que
  pertence ao extrator. Todas as correlações com a **degradação** de ζ são
  indistinguíveis de zero. *Ressalva de potência: com n=110 o Spearman só resolve
  |r| > ~0,19 — leia "o efeito, se existe, é pequeno", não "é zero".*
- **Ruling 41 — a falha do 2.2 é 100 % do extrator, 0 % da rede.** Com a máscara
  **VERDADEIRA** o p95 já reprova (6,701 px). **O 2.2 nunca foi problema de
  segmentação.**
- **Ruling 43 — a cauda do 2.2 é curva ÍNGREME.** As três hipóteses do bloco
  anterior estão **refutadas**: o traço `:` é o estrato com a **menor** fração na
  cauda (7,5 %). O que explica é a **inclinação máxima em px/px** (Spearman
  +0,868): nenhuma amostra abaixo de 6,45 px/px está na cauda; **70 %** das acima
  de 26,55 estão. **Lição: estrato que move a mediana não é o mesmo que estrato
  que produz a cauda.** E a inclinação **não vem dos parâmetros do sistema** — vem
  do aspecto do quadro (−0,600). **A cauda é estrato de RENDER, não de dinâmica.**
- **Ruling 46 — sete regras locais testadas, todas piores.** A mediana por coluna
  é a melhor das oito. **Mas o oráculo limitado à tinta PASSA** (1,186 / 3,947):
  **a informação ESTÁ na tinta e o teto não é a representação.**
- **Ruling 48 — a formulação global também perde, e o prior de SUAVIDADE é errado
  aqui.** Todas as seis variantes perdem e são **quase idênticas entre si**,
  indicando convergência para a mesma solução. **Numa resposta ao degrau a curva é
  genuinamente íngreme no transiente**, então penalizar curvatura briga com a
  verdade exatamente onde a cauda mora. Isso explica de uma vez as **14 tentativas
  que perderam** e por que só o oráculo ganha: ele não usa prior, usa a verdade.
- **Ruling 49 — e NÃO vale a mudança: um extrator perfeito não recupera
  acurácia.** O ganho em ζ tem **p=0,28**. E mesmo no teto ζ fica em 1,635 contra
  1,215 do oráculo de série: **a maior parte da degradação que o Ruling 42
  atribuiu ao extrator não está na redução coluna→ponto.**
- **Ruling 50 — a métrica PERPENDICULAR resolve 2.1 e 2.2: o pipeline é
  sub-pixel.**

  | | mediana | p95 |
  |---|---|---|
  | vertical (métrica do 2.2) | 1,488 px | **6,701 px** |
  | **perpendicular** | **0,614 px** | **1,135 px** |

  Correlação com a inclinação: vertical +0,869, **perpendicular +0,326**. **A
  reprovação do 2.2 é artefato de medir distância vertical em curva íngreme.**
  Isso explica retroativamente o Ruling 49: a redução "perfeita" não comprou
  acurácia **porque não havia erro geométrico a recuperar**.

  **De onde vem o limiar proposto, e por que não é mover a trave:** o 1,0 px sai
  do ORÇAMENTO do 2.6 (3 p.p.), não do resultado atual — com 0,800 px a
  contribuição da rede a ζ é +0,127 p.p., ~4 % do orçamento. E **não é vacuoso**:
  25 % de folga sobre o medido. **A legitimidade da troca vem de a MÉTRICA ter
  mudado para medir a grandeza certa, com prova independente de que a antiga era
  errada. Afrouxar o limiar da métrica ANTIGA seria mover a trave.**

  **Dois resguardos obrigatórios:** (1) continuar reportando IoU e vertical como
  **diagnóstico sem alvo**, para preservar comparabilidade; (2) **a perpendicular
  tem ponto cego** — não penaliza erro AO LONGO da curva, que num degrau é
  exatamente o θ. O par já existe: `2.6[theta]`.

### 7.7 Ruling 42 — a ablação que ordena as prioridades

Quatro cadeias, trocando um elo por sua versão perfeita, com **Wilcoxon pareado**
(n=207):

| elo | K | τ | ωₙ | **ζ** | θ |
|---|---|---|---|---|---|
| **extrator de polilinha** | +0,074✱ | +0,135✱ | +0,194✱ | **+0,368✱** | +0,065✱ |
| rede (U-Net) | +0,003 | +0,034 | −0,041 | +0,127 (p=0,056) | +0,050✱ |
| calibração / OCR | +0,059✱ | +0,000 | +0,200 | +0,120 | +0,111✱ |

Perda de **amostra**: a calibração responde por **56 das ~68** perdidas; a rede
custa **2**.

> **ARMADILHA METODOLÓGICA, registrada porque quase produziu conclusão errada:**
> a diferença de MEDIANAS por estágio sugere que a rede *melhora* ζ (−0,129 p.p.),
> porque cada estágio tem subconjunto de convergência diferente. O teste PAREADO
> refuta (+0,127, piora levemente). **Nunca decompor esta cadeia por mediana de
> estágio; usar sempre delta pareado por amostra.**

### 7.8 Rulings 51–53 — a revisão dos critérios

- **Ruling 36 — CORREÇÃO do Ruling 33: o OCR é ~92 % preciso, não 50 %.** O
  denominador usado era a saída de `detect_tick_pixels`, que conta ticks MENORES e
  que `read_tick_labels` explicitamente não consulta. Denominador certo (os ticks
  realmente rotulados, do meta): **91,9 % em x e 92,7 % em y**, com recall de
  89,4 % / 80,9 %. **Esta crença não nasceu no Bloco 7** — a docstring de
  `_equiespacados` já a afirmava desde o Bloco 2.
- **Ruling 37 / 51 / 52 — a poda, e os critérios que se contradizem.**
  - **2.4 SUBSUME 2.9**: `test_2_4` mede `rejeitadas/total`, que é exatamente
    `1 − cobertura` do 2.9. **Ter os dois como critérios independentes dá peso
    duplo à mesma grandeza.**
  - **2.3 e 2.5 são incompatíveis por construção**: 2.3 exige erro ≤ 1 % nas
    aceitas, 2.5 chama a rejeição de justificada quando o erro seria > 5 %. As
    amostras na faixa 1–5 % são **simultaneamente** "não deviam ter sido
    rejeitadas" e "ruins o bastante para estragar o 2.3". Prova aritmética: nenhum
    subconjunto fecha os dois. Um guardião por resíduo foi testado e **alterna
    direto** de (2.3 ✅, 2.5 ❌) para (2.5 ✅, 2.3 ❌) — não existe limiar onde
    ambos passam.
  - **A margem de rótulos NÃO conserta `ocr_insuficiente`:** sete configurações
    testadas, ganho máximo de 1,3 p.p., e na maioria o `calibration_failed`
    **sobe** (alargar captura número que não é rótulo). **É troca de motivo, não
    conserto.**
  - *(O Ruling 51 concluiu que a poda "nunca converte reprovação em aprovação".
    **Isso estava errado** — a análise não tinha o 2.5 no quadro; o Ruling 52
    mostra que a poda FECHA o 2.5, ao custo do 2.3.)*
- **Ruling 53 — a revisão foi ESCRITA no `PLANO.md` (§2.12) e IMPLEMENTADA:**

  | critério | antes | agora |
  |---|---|---|
  | **2.1** | IoU ≥ 0,85 | RMSE perpendicular ≤ 1,0 / 2,0 px; **IoU vira diagnóstico** |
  | **2.2** | vertical ≤ 2 / 5 px | perpendicular ≤ 1,0 / 2,0 px; **vertical vira diagnóstico** |
  | **2.7** | IoU por estrato | perpendicular por estrato |
  | **2.4** | rejeição < 5 % | **aposentado** — diagnóstico, unificado no 2.9 |
  | **2.5 / 2.3** | 5 % e 1 % | ambos por `ESCALA_TOL`, alinhamento por construção |

  **Falhas na suíte: 5 → 2.**

  > **Armadilha 8 — justificar o limiar com uma estatística e implementar com
  > outra.** Na primeira versão o 2.1 media a **mediana** do erro perpendicular
  > enquanto o limiar de 1,0 px foi derivado do **RMSE**. A mediana é mais
  > leniente (0,549 px) e deixava o critério mais fácil que a própria
  > justificativa. **Não basta a métrica ser a certa e o limiar ser derivado — a
  > *estatística* medida tem de ser a mesma da derivação.**

## Bloco 8 — o caso real, e o que oito imagens externas quebraram (28–31/08/2026)

### 8.1 Ruling 54 — OOD de aquisição: a fronteira é a ROTAÇÃO

120 amostras × 10 degradações que o gerador nunca produz (JPEG, reescala,
rotação, ruído de sensor).

| degradação | calibração ok | ordem ok | **MAPE ζ adimensional** |
|---|---|---|---|
| original | 78,3 % | 90,0 % | 2,55 % |
| JPEG q=30 | 45,0 % | 90,8 % | 3,13 % |
| reescala 0,33× | 38,3 % | 89,2 % | 3,24 % |
| **rotação 0,5°** | **3,3 %** | 93,3 % | **2,89 %** |
| **ruído σ=8** | **5,0 %** | 86,7 % | 2,60 % |
| **rotação 2°** | **0,0 %** | 88,3 % | **14,48 %** |

**Meio grau de rotação leva a calibração de 78,3 % a 3,3 %** — `detect_plot_bbox`
e `detect_tick_pixels` supõem moldura alinhada. Enquanto isso a **ordem acerta
86,7–93,3 % em TODAS as degradações**, e o ζ adimensional entrega **93,1 % das
amostras com acurácia intacta** onde a calibração entrega 3,3 %.

**A previsão do `PLANO §1.7` se confirma quantitativamente. A Decisão E deixou de
ser conveniência e passou a ser a característica que sustenta o sistema fora da
distribuição.** A fronteira é a rotação porque ela **torce a forma**, e ζ vem da
razão de overshoot, que é forma; JPEG, ruído e reescala degradam a máscara sem
torcer a geometria.

### 8.2 Ruling 55/56 — uma imagem real encontrou dois defeitos que 300 sintéticas não

`resposta_degrau.png`, ζ=0,5, ωn=2, **θ=0**. O pipeline devolveu `fopdt` para uma
curva com overshoot visível.

- **Defeito 1 — a máscara da U-Net.** Perdia os últimos 38 % da janela (onde a
  curva coincide com a reta de referência), capturava a reta e capturava glifos de
  título. Trocando **apenas a máscara**, com o mesmo estágio D: verdade → `second`
  ζ=0,5000; U-Net → `fopdt`; clássico → `second` ζ=0,3847. **A falha de ordem é da
  máscara, não do estágio D.**
- **Defeito 2 — o estimador de repouso da Decisão E.** `_FRAC_REPOUSO = 0.08`
  supõe prefixo plano por tempo morto. **Com θ=0 a curva sobe desde t=0**, a
  "mediana do repouso" fica em ~0,1, o patamar é subestimado em **3,8 %** — e
  3,8 % no patamar viram **12,6 % em ζ**. Corrigido para `_N_REPOUSO = 5` colunas
  fixas, escolhido por ganhar nas **duas** populações (real e sintética).
  **Por que o sintético nunca pegou isso: o gerador sorteia θ quase sempre
  positivo, então o prefixo plano existe.**
- **Segunda causa raiz, não prevista — a normalização do TEMPO.** `t` era
  normalizado pela extensão observada da polilinha. Com 38 % da largura sem tinta,
  isso **comprime o tempo e infla ωₙ na mesma proporção**: erro de **37,7 %**.
  Normalizar pela **moldura** leva a 1,0 %. **A mudança não é incondicional** — no
  corpus a moldura é 2,2 % a 11,7 % mais larga que a janela de dados, então usar a
  moldura sempre injetaria essa margem como viés. Ficou condicionada a
  `_COBERTURA_MIN_MOLDURA = 0.75`.

**Resultado:** o pipeline entrega **ζ = 0,5018** (erro 0,4 %) e **ωₙ = 2,0377**
(1,9 %), com `nrmse` 0,0029 — equivalente ao sintético. **A capacidade existia;
falharam dois elos consertáveis.**

### 8.3 O PONTO CEGO DE MEDIÇÃO — o achado mais transferível do projeto

> **A suíte inteira passava verde por uma regressão de escala de tempo.**
>
> Não existia critério medindo ωₙ no caminho adimensional. Havia só
> `2.6-adim[zeta]` — **e ζ é invariante à escala do tempo**. Ou seja: o caminho
> que existe justamente para dispensar a calibração media a **única grandeza cega
> ao defeito** que a escala de `t` produz. 36 testes, todos verdes.

Corrigido acrescentando `2.6-adim[wn_T]` e — na rodada seguinte — uma **segunda
linha**, `2.6-adim[wn_T/sem-calib]` (n=33), porque a primeira media n=143 e **110
dessas passam pelo caminho FÍSICO**, que nem executa `_serie_normalizada`: o
sinal ficava **diluído ~7×**.

Prova de sensibilidade, feita em cópia do repositório com a regressão
reintroduzida e **verificada de forma independente por quem revisou**:

| linha | produção | com a regressão | move |
|---|---|---|---|
| `2.6-adim[wn_T]` (corpus, n=143) | +1,04 | +1,82 | +0,78 p.p. |
| `2.6-adim[wn_T/sem-calib]` (n=33) | +0,63 | **+6,43** | **+5,80 p.p.** |

**A linha restrita à população certa reage 7,44× mais à mesma regressão.**

> **A lição:** um critério pode existir, passar, e ainda assim não cobrir nada —
> quando a grandeza que ele mede é **invariante** ao modo de falha do caminho que
> ele deveria vigiar, ou quando a **população** em que ele mede é majoritariamente
> de amostras que não passam por aquele caminho. **As duas falhas aconteceram
> aqui, na mesma linha**, e a segunda só foi vista porque alguém exigiu a prova de
> sensibilidade.

O padrão se repetiu duas vezes mais e ganhou os eixos que faltavam:
`2.6-adim[theta_T]` (a **origem** de `t`, que ζ e ωₙ·T não veem),
`2.6-adim[K_yrange]` (a escala de **y**) e `2.12-ordem` — este último porque a
ordem entrava só como **filtro de aceitação**, que é a forma clássica do ponto
cego: *uma regressão de ordem não piora mediana nenhuma, ela **encolhe** a
amostra, e as que somem são as difíceis, então as medianas que restam até
melhoram*. Controle positivo: trocando a U-Net pelo extrator clássico,
`2.12-ordem` cai de 91,3 % para 78,3 % — **13 p.p. que nenhum critério do
relatório registrava**.

E ganhou **guarda permanente**: `tests/part2/test_instrumentacao.py` sustenta uma
invariante — *nenhum critério declarado desaparece do relatório por causa de um
portão de `n`; se o `n` for insuficiente, a linha aparece dizendo isso, com o `n`
real*. (O padrão tinha reincidido **duas vezes no mesmo arquivo no mesmo dia**.)

### 8.4 Dois erros de MÉTODO, na mesma família, no mesmo bloco

- **Achatar uma distribuição num escalar.** Uma review pediu que
  `_COBERTURA_MIN_MOLDURA` subisse para o "ponto de empate ≈ 0,872". **A derivação
  estava correta e `c*` não é um número, é uma distribuição** (0,7912 a 0,9563;
  0,8716 é a **mediana** dela). Medido: afeta 3 das 179, **nenhum critério com
  meta se move**, e em `sample_00828` — para a qual o modelo do empate **prevê
  melhora** — o τ **PIORA de 4,9 % para 19,1 %**. **Decisão: recuar.**
- **Calibrar um limiar na população errada.** `_FALTA_ESQ_MAX_FRAC = 0.15`, um
  proxy geométrico, veio do deslocamento máximo de margem do matplotlib medido num
  corpus **cuja cobertura mínima é 0,8388 — nenhuma daquelas 299 amostras entra
  neste ramo**. Ele tinha **68 % de erro de ζ de um lado e nenhuma amostra do
  outro**:

  | falta à esquerda | decisão do 0,15 | erro de ζ |
  |---|---|---|
  | 0,1139 | aceita | 45,7 % |
  | **0,1327** | **aceita** | **68,0 %** |
  | 0,1501 | RECUSA | — |

  Substituído pelo **invariante direto**: `_PLANURA_MAX_FRAC = 0.03` exige que as
  primeiras colunas observadas sejam **planas**, que é literalmente a condição de
  que `_nivel_de_repouso` precisa, é independente da margem e **é transferível
  entre as populações**.

> **O padrão que a próxima pessoa precisa reconhecer:** *antes de calibrar um
> limiar, pergunte se a população em que ele foi medido é a população em que ele
> vai decidir.* Duas vezes no mesmo bloco, nas duas pontas — na constante e no
> diagnóstico.

**A decisão de RECUSAR, e por que ela foi medida:** 27 séries determinísticas com
o repouso já na subida — trocar pela moldura dá MAPE de ζ mediano de 27,6 %,
manter a extensão observada dá 14,2 %, **e as duas são piores que o defeito de
12,6 % que a correção existia para consertar**. **Um número errado que ninguém
distingue de um certo é pior que nenhum.**

### 8.5 Ruling 57 — o 2.9: três hipóteses refutadas, uma retratação

- **RETRATAÇÃO: "o gargalo é recall do OCR" saiu de um teste CIRCULAR.** Os pixels
  dos rótulos faltantes eram gerados a partir do PRÓPRIO ajuste afim, caindo
  perfeitamente sobre a reta. **O gate passava por construção; o teste não tinha
  como falhar.** Refeito de forma não circular: 716 → 715 amostras, **efeito
  nulo**. *O sinal de alerta era simples e foi ignorado: o teste não tinha um caso
  em que pudesse falhar.*
- **A anatomia real, em três camadas.** Segmentação de blobs **não é** o problema
  (recall mediano 100 %). Leitura tem perda real (86,9 % / 80,0 %) com um padrão
  contraintuitivo e diagnóstico — **o recall SOBE com o comprimento da string**
  (78,9 % para `0.2`, 93,5 % para `100`): é comportamento conhecido do LSTM com
  `--psm 7`, que é motor de LINHA. **Rótulo de eixo é exatamente o pior caso dele,
  e isso não é ajustável por parâmetro.** E é no **emparelhamento** que o gate
  morre: mediana de erro sub-pixel mas **desvio padrão de 10 a 27 px** nas formas
  curtas — não é imprecisão de centroide, é **valor lido certo atribuído ao pixel
  de OUTRO rótulo**.
- **Refutadas:** melhorar a precisão da detecção de ticks (espúrios 7→3 e `ok`
  **dígito a dígito igual** — os espúrios já eram inofensivos); ancorar o rótulo
  na marca de tick mais próxima (**piora monotonamente**, porque com 7 marcas
  espúrias o snap desloca rótulos BONS para marcas falsas); completar a rede de
  valores (o teste circular).
- **O que NÃO fazer: continuar ajustando `_equiespacados`.** 24 variantes varridas;
  **toda variante que sobe cobertura sobe falso positivo**, a ~1,3 por ponto
  percentual, com máximo de 84,8 % — abaixo da meta. **O gate não está mal
  calibrado; está mal condicionado por receber poucos pontos.**

### 8.6 Ruling 58 — a camada que faltava, achada por uma PORTA de aceitação

O item 1 do §36.5 (prior de origem) tinha critério de aceitação **fixado antes de
implementar**. A primeira porta exigia que o blob mais externo coincidisse com o
tick verdadeiro a ≤ 3 px em ≥ 99 % das amostras. **Medido: 76,1 % e 75,5 %.
Reprovada.**

**A causa, uma camada abaixo de tudo que o Ruling 57 mediu:** as MARCAS de tick
apontam para fora, então **estão dentro da faixa de rótulos**. Com
`BLOB_DILATE_X = 8` (16 px de alcance) elas se costuram numa barra horizontal
contínua que encosta nos rótulos. **Um blob só, cobrindo o eixo inteiro** — e
`read_tick_labels` usa **o centro do blob COMO O PIXEL DO TICK**. Numa amostra o
OCR leu literalmente `"1020304050"`.

Três mudanças (`BLOB_DILATE_X` 8→3; zerar a banda encostada na moldura antes de
dilatar; folga lateral de 20 px):

| | antes | depois |
|---|---|---|
| âncora x / y a ≤ 3 px | 76,09 % / 75,53 % | **99,89 % / 99,89 %** |
| `ok` | 79,56 % | **91,78 %** → **93,00 %** (com o teto de lote) |
| **falso positivo** | 55 (7,68 %) | **20 (2,42 %)** |
| `calibration_failed` | 76 | **8** |

**É o único ajuste medido em todo o bloco que sobe cobertura E desce falso
positivo.** E corrige o registro anterior: **o §36.4 apontou a hipótese errada** —
a consistência estava certa, ela rejeitava corretamente amostras cujas posições de
blob eram lixo; **com a entrada consertada, `calibration_failed` cai de 76 para 8
sem tocar uma linha do gate**.

> **A lição de método:** o item 1 foi para a fila em primeiro lugar por ser a
> hipótese mais forte, **e não foi implementado**. O que produziu o resultado foi a
> **PORTA de verificação** dele, fixada antes por exigência do plano, e a decisão
> de investigar por que ela reprovou em vez de descartar o item. **Uma porta de
> aceitação que reprova é informação sobre o sistema, não sobre a hipótese.**

**Suíte: 4 falhas → 1.** O 2.9 estava reprovado **desde o Bloco 5** e resistira a
24 variantes de `_equiespacados`, ao `snap` e ao `_sem_paralela`. O 2.5 fechou por
um caminho **diferente** do que a pista sugeria: não por rejeitar mais, e sim por
**deixar de produzir a leitura ruim que precisava ser rejeitada**.

### 8.7 Rulings 59–60 — as guardas de plausibilidade, e o estrato que funciona

- **59a — colapso do lote de OCR.** Quando a análise de LAYOUT decide que o
  mosaico não é uma linha de texto, o tesseract **não levanta exceção**: devolve
  zero palavras e o lote inteiro vira `None`. **Não é previsível pelo tamanho**:
  14 → 11 lidos, 16 → **0**, 18 → **0**, 20 → 17, 21 → **0**. Conserto: teto de 12
  recortes por mosaico + releitura individual de qualquer bloco que colapse.
- **59b — dois fundos na figura.** O fundo era a mediana do quadro inteiro, o que
  assume um fundo só; o `generator.py` garante isso, **o matplotlib real não**. Em
  tema escuro `detect_plot_bbox` devolveu o quadro inteiro. Conserto: `_fundo()` =
  **moda da borda** (que é fundo da FIGURA por construção). Com um fundo só,
  idêntico nas 900 amostras e bbox igual em **900/900**.
- **59c — uma guarda refutada, duas implementadas.**
  - **REFUTADA: descontinuidade da máscara.** Parecia óbvia (19,3 % de buraco na
    imagem que erra contra 5,5 % da segunda pior das outras sete), mas no corpus
    n=837 o Spearman é **+0,020 (p=0,57)**, e **o maior buraco do próprio corpus é
    MAIOR que o da imagem que erra**.

    > **O motivo é estrutural, e é a lição:** o corpus **não contém o modo de falha
    > que essa guarda existia para pegar**, então ele media só o **CUSTO** dela,
    > nunca o benefício. A ordem "guarda primeiro, estrato depois" estava errada.
    > A separação em n=8 era convincente e n=895 a desmente.
  - **IMPLEMENTADA: resíduo do ajuste** (`_NRMSE_MAX = 0.13`). Escapa do problema
    acima por **não depender do modo de falha**. Spearman +0,386 (p=4,5e-31),
    precisão 88,2 %, custo 2 boas em 837. Recall 19 % — **recusa o absurdo, não
    audita o aceitável**.
  - **IMPLEMENTADA: resposta inversa** (`_UNDERSHOOT_MAX = 0.10`). A primeira
    definição tinha falso positivo (marcava 0,147 numa imagem de ζ=0 cuja
    identificação estava CERTA, porque a série não assenta). Redefinida **pela
    física** — a excursão contrária só conta ANTES de a resposta arrancar.
    **ATENÇÃO: o gerador não produz fase não-mínima, então o corpus dá só o custo;
    o benefício está apoiado em n=1.**
- **Ruling 60 — estrato OOD que de fato reproduz o fenômeno.** Ablação pareada,
  n=60, com banda de acomodação e anotação com seta:

  | braço | IoU mediana | Δ vs base | p |
  |---|---|---|---|
  | base | 0,6121 | — | — |
  | só banda | 0,5352 | −0,0629 | 2,6e-10 |
  | só seta | 0,4969 | −0,1130 | 1,6e-11 |
  | **banda+seta** | **0,4381** | **−0,1898** | 1,6e-11 |

  **É o oposto da primeira tentativa** (o estrato `reta_no_patamar`, que **simulava
  o OBJETO e não o FENÔMENO**: em 30 seeds o ramo da moldura não disparava uma
  única vez e a cobertura até **subia**). **Limite honesto:** a IoU cai 31 % e o
  erro de identificação quase não se move — **o estrato prova que o Estágio A
  sofre, não que o Estágio D quebra**.

### 8.8 Ruling 61 — treze imagens externas medem o envelope

| resultado | n |
|---|---|
| identificadas corretamente | 10 |
| recusadas com motivo nomeado | 1 (fase não-mínima) |
| **erradas em silêncio** | **2** |

- **`Figure_f3` — não é bug, é identificabilidade.** Uma 2ª ordem criticamente
  amortecida com atraso saiu como FOPDT. O ajuste de 2ª ordem é melhor em SSE **e**
  em AIC e recupera θ com 1,6 % contra 5,9 %, mas é só **1,5 % melhor em SSE**, e
  com `n_eff = 21` o ganho é 0,308 contra o limiar 2,0. **Nem uma máscara perfeita
  criaria a informação que falta** — e a 2ª ordem também não recupera ζ=1 (devolve
  1,44). **O defeito não é a escolha, é o SILÊNCIO.**
- **`Figure_322` — instável, e a saída é sem sentido com `ok=true`.** `K = 1e+04`
  (**teto exato**) e `zeta = 0.001` (**piso exato**), com `nrmse` **0,03182**. O
  modelo não representa divergência, então o otimizador aproxima a exponencial com
  o primeiro quarto de período de uma senoide de ganho gigantesco. **O ajuste
  encaixa bem no trecho visível, por isso a guarda de resíduo não dispara.**
- **O sinal grátis que o sistema ignora: parâmetro cravado na borda da caixa não é
  medição — é o otimizador desistindo.** A informação já existe nos `*_BOUNDS` e
  custa uma comparação exata. **Mas há uma armadilha medida:** a `Figure_12`
  (ζ=0 verdadeiro) encosta no piso do ζ **legitimamente**. O critério não pode ser
  "algum parâmetro na borda" — a hipótese a testar é **K na borda**.

**Conclusão do bloco: o sistema não tem detector para a maior parte do que está
fora da família.** Há guarda para resposta inversa e resíduo alto. Não há para
instável, ordem superior, ganho negativo, zero no semiplano esquerdo, nem
múltiplas curvas.

### 8.9 Ruling 62 — os três sistemas do `rg.py`

Três imagens do próprio autor, verdade declarada na função de transferência.

**O que as três têm em comum, e por que o corpus não as encontrou:** `rg.py` usa
`plt.ylim(0, ...)`, o que encosta a curva no patamar de repouso a 2–3 px da
moldura (~0,9 % do span). **O gerador sorteia `y_margin_lo ~ U(0.03, 0.15)` e
nunca desce abaixo de 3 %** — a geometria inteira está fora da distribuição de
treino **e** de teste. **Uma causa, três defeitos.**

- **Sistema 3 — o eixo y era reprovado com os nove rótulos lidos CERTOS.** O
  rótulo extremo encostava na moldura, o strip decepava 3 px de cada ponta, e o
  centróide — que **É** o pixel do tick — entrava 1,5 px. `_equiespacados` usava
  `unit = min(d)`, o valor enviesado, inflando toda razão em 4,5 %: **reprovado por
  1,2 ponto percentual.** Correções: reconstruir o centro do blob cortado pela
  meia-altura mediana dos não cortados, e usar unidade **robusta** (`min(d)` é o
  estimador menos robusto possível). **Aumentar a folga foi testado e descartado**
  — perde uma amostra e não ganha nenhuma.
- **Sistema 1 — a ordem era decidida por DOIS pixels, de 406.** O polo extra do
  ajuste de 2ª ordem tinha constante de tempo de **3,6 px — a espessura do próprio
  traço no canto do degrau**. *A 2ª ordem não achou dinâmica; ajustou o
  antialiasing.* Conserto: `_polo_rapido_e_artefato` — se o escolhido é 2ª ordem
  **superamortecida** e um trecho **CONTÍGUO** de 3 % responde por ≥ 100 % da
  vantagem de SSE, devolve FOPDT.

  **"Contíguo" importa, e foi medido:** a primeira versão usava os pontos de maior
  ganho onde quer que estivessem e custava 2,6 p.p. no caminho ORÁCULO, que nem
  tem render — era a guarda pegando picos de **ruído**. *Um canto rasterizado é um
  acidente LOCAL; ruído de aquisição é disperso por construção.*

  **A restrição a ζ > 1 é estrutural:** com polos complexos não existe polo rápido
  separado a descartar — a oscilação é a assinatura inteira. **Custo real das
  rebaixadas: perde-se o rótulo, não a física** (τ a 0,1–1,5 % do dominante
  verdadeiro, K a menos de 1 %).

  **Efeito:** ordem correta no caminho imagem **88,89 % → 93,0 %**; 1ª ordem
  82,6 % → **92,1 %**. **A regressão que existe, declarada:** o caminho oráculo a
  20 dB caiu de 0,888 para 0,878 (6 amostras em 600), dentro de 1 sigma.
- **O defeito que SOBROU — e são DOIS, os dois exigindo RETREINO.** A U-Net dá
  `prob ≤ 0,004` em colunas onde há traço colorido, puro e **não ocluído**:

  - **Defeito A — curva rente à moldura inferior.** Curva dose-resposta: 1, 3 e
    5 px → **0/150**; 7 px → 107/150. **O degrau está em ~2 % do span, logo abaixo
    do piso de 3 % que o gerador sorteia.** Três hipóteses refutadas por ablação (a
    banda cinza, a COR do traço, a planura do trecho).
  - **Defeito B — trecho perfeitamente RETO.** Ondular ±1 px na mesma cauda
    recupera **79/85**. A probabilidade despenca exatamente onde a ondulação cai
    abaixo de 1 px. **O que ela suprime é a RETIDÃO, não a oclusão** — e a
    explicação é uma consequência **não intencional de um critério existente**: o
    G3b.2 ("sem reta de span completo") ensinou o modelo a rejeitar reta
    horizontal, **e uma resposta assentada É uma reta horizontal**.

  **A fragilidade é INTERMITENTE, e esse é o argumento para retreino e não para
  guarda:** os três sistemas têm a curva à mesma distância da moldura e o Sistema 2
  sobrevive enquanto o Sistema 1 colapsa a zero. **Não dá para prever qual imagem
  cai, então não dá para consertar com limiar. Precisa de dado.**

  **Registrado em código, não só em texto:** `xfail(strict=True)`, para que no dia
  em que o retreino consertar o teste vire XPASS e **reprove a suíte**, obrigando
  quem consertou a converter o defeito documentado em portão de regressão. **É o
  mecanismo que impede esta dívida de sumir do jeito que as oito imagens sumiram.**

## Bloco 9 — ganho negativo, e o prior de POSIÇÃO (02–04/09/2026)

### 9.1 Ruling 63 — o caminho C, e o detector de direção que custou TRÊS formulações

Degrau negativo era **estruturalmente inexprimível**: `K_BOUNDS = (1e-3, 1e4)`
trava K positivo, o ajuste saía com NRMSE 0,90–0,96, e as recusas eram **efeito
colateral, não detecção** (uma delas com diagnóstico **falso**: `resposta_inversa`
para o que é degrau negativo).

**Alargar `K_BOUNDS` foi DESCARTADO por razão estrutural, não de gosto:** põe
**K=0 dentro da caixa**, e K=0 é o modelo degenerado (resposta plana com τ e θ
livres). Cria um mínimo local trivial que hoje não existe **e destrói o sinal de
borda** que o Ruling 61 quer usar como guarda.

**O caminho C:** se a resposta DESCE, nega-se `y` antes do Estágio D, ajusta-se
com o código intocado, e o `K` devolvido troca de sinal. Equivalente a
parametrizar `K = s·|K|`, mas **não toca uma linha da matemática do módulo**;
`sse`, `nrmse` e `aic` são invariantes ao espelho, e com `s = +1` o caminho é byte
a byte o anterior.

*(O OCR também precisou de conserto: o matplotlib desenha o menos como U+2212 e o
tesseract devolve EM DASH — **8 de 9 rótulos viravam `None` com os dígitos lidos
CERTOS**.)*

**As três formulações do detector de direção, e por que as duas primeiras
quebravam em pontas OPOSTAS da série:**

1. **Mediana do primeiro decil contra a do último.** Morre quando o Estágio A come
   o platô inicial: o decil cai dentro do transitório, e numa subamortecida o
   transitório passa ALÉM do valor final, pelo lado oposto. **8 erros em 44**, e o
   defeito era **simétrico** — mordia o caminho positivo também. *Ajustar a fração
   não conserta, e é importante que fique registrado: na mesma imagem, 0,05 → −1;
   0,10 → +1; 0,20 → −1; 0,30 → +1.*
2. **Extremo mais distante do valor assentado.** Corrige (1) e mede 0 erros no
   sintético. Mas pressupõe que o último decil é o valor FINAL, e numa 2ª ordem
   muito subamortecida ele pousa no ringing e **o pico é eleito repouso**.
   Espelhava **2 das 900** do oráculo — todas com K > 0, logo duas amostras boas
   destruídas, visível como MAPE(K) saindo de 0,000 % para 0,239 %.
3. **O que ficou: o repouso é o extremo que aparece PRIMEIRO.** Se o máximo vem
   antes do mínimo, a série desceu. **A informação está no TEMPO, não no valor** —
   uma resposta ao degrau nunca cruza de volta o nível de onde partiu, então o
   repouso **é** um dos dois extremos, e o que o distingue do sobressinal é a
   ORDEM. **Ler a ordem dispensa saber onde a resposta assenta, que é justamente o
   que uma janela curta esconde.**

   **0 erros em 44** no sintético, **0 espelhos indevidos nas 900** do oráculo, e
   0 erros em 1600 séries com σ até 0,30. **Contrato e limite asseverados** com
   `xfail` estrito: vale enquanto o repouso ainda ESTIVER na série.

**O custo declarado — o extrator CLÁSSICO.** Sob ele, **32 das 295** séries são
lidas como descendentes e espelhadas, todas indevidamente (a U-Net erra 1 em 900).
**O que salva é o modo de falhar: as 32 são recusadas com `ajuste_inconsistente`,
nenhuma produz saída física.** O espelho errado não vira resposta confiante e
errada.

### 9.2 O estrato no gerador, e o alvo que media a coisa errada

`generate_sample(..., ganho_negativo=True)` — **opt-in e não sorteio**, porque
mexer em `sample_system` moveria toda amostra do corpus base e com ela todo número
histórico. Só o SINAL de K muda e `|K|` fica idêntico ao do mesmo seed, o que
torna o estrato comparável **amostra a amostra**. O sinal é aplicado ao SPEC e
nunca ao estilo — um traço que mudasse de cor por causa do sinal ensinaria a rede
a ler o sinal do RENDER.

> **O alvo de 95 % que o plano tinha escrito estava medindo a coisa errada.** Ele
> exigia K recuperado a 5 % em ≥ 95 % do estrato. O estrato mede 88,3 % — **mas o
> caminho POSITIVO, histórico e sem espelho nenhum, mede 90,0 % nos mesmos seeds.**
> O alvo estava capturando a dificuldade do estrato de janela truncada (RULING C),
> não o caminho C.

Trocado por três portões mais fortes: **equivalência exata** em série limpa (por
tolerância relativa 1e-9, porque a equivalência é MATEMÁTICA e não numérica —
ajustar `-y` acumula somas em ordem diferente, desvio ~1,5e-14), **recuperação
total** (60/60, porque alvo abaixo de 100 % aqui esconderia regressão) e
**paridade sob ruído** (o negativo acompanha o positivo dentro de 5 p.p.).

### 9.3 Ruling 63 (§40.7) — o prior de POSIÇÃO, isolado por ablação fatorial

Experimento pareado, n=150, mesmo seed com e sem `ganho_negativo`:

| métrica | positivo | negativo | Δ |
|---|---|---|---|
| IoU da máscara | 0,6217 | 0,5859 | −0,036 |
| cobertura de colunas | 0,9544 | 0,8663 | −0,088 |
| **cobertura do PLATÔ** | **0,9317** | **0,5523** | **−0,379** |

**O corpo da curva quase não sofre; o platô de repouso desaba.** E não é nenhuma
das duas causas já conhecidas: **não é "curva rente à moldura"** (o mesmo seed dá
o mesmo estilo, então o platô fica igualmente distante da SUA borda — o que muda é
QUAL borda) **nem planura em si** (o platô positivo é igualmente plano e sobrevive
em 93 % das colunas).

**Confundimento RESOLVIDO por ablação fatorial 2×2** (n=80 por célula, invertendo
o eixo y via `_axis_limits`, que alimenta imagem **e** máscara, então a verdade
fica consistente):

| sinal | eixo | platô em | cob. do platô |
|---|---|---|---|
| K>0 | normal | **rodapé** | **0,9443** |
| K<0 | invertido | **rodapé** | **0,9398** |
| K<0 | normal | **topo** | **0,5274** |
| K>0 | invertido | **topo** | **0,5238** |

> **Os resultados agrupam por POSIÇÃO e ignoram o SINAL. O sinal do ganho é
> IRRELEVANTE: o que a U-Net não sabe fazer é segmentar platô de repouso na metade
> de cima do quadro, qualquer que seja o sinal. O ganho negativo não é o defeito —
> é só o que expôs o defeito.**

Isso **amplia o escopo para além do Bloco 9**: qualquer figura de eixo y invertido
cai nele, mesmo com K > 0, e essa é convenção corrente em parte da engenharia.
Não há nenhuma amostra assim no corpus. E dá uma **validação independente** para o
retreino: treinar em negativo (platô no topo) e validar em positivo de eixo
invertido (condição **nunca treinada**).

**E refina o Ruling 62:** parte do que foi atribuído a "trecho reto" nas imagens
de ganho negativo é este prior de posição. **O retreino do defeito B e o retreino
do ganho negativo são provavelmente O MESMO retreino.**

### 9.4 O `IoU_val` é CEGO ao defeito — e é ele que seleciona o checkpoint

| conjunto de validação | n | IoU_val |
|---|---|---|
| `data/val` (todo K > 0, platô no rodapé) | 900 | 0,7814 |
| `data/val_kneg` (todo K < 0, platô no topo) | 300 | **0,7304** |

**O estrato que a rede não sabe segmentar custa 5 pontos de IoU, enquanto a
cobertura do platô nele desaba 42 pontos.** A razão é geométrica: o platô é uma
linha FINA, e o IoU é dominado pelo corpo da curva.

**Isso é um problema de PROCESSO, não de modelo** — `train_unet.py` seleciona por
`IoU_val`, a métrica quase cega ao defeito que o retreino existe para consertar.
**Retreinar sem resolver isto é gastar ~6 h numa loteria.** Saída adotada: salvar
**um checkpoint por época** e escolher depois (não toca métrica nenhuma e preserva
o `IoU_val` histórico para comparabilidade).

**LACUNA CRÍTICA identificada antes de treinar:** `data/val` é 100 % K > 0, logo
100 % platô no rodapé. Um estrato de validação com platô no topo era
**pré-requisito, não opcional**.

**O que o *smoke test* de 2 épocas encontrou, antes de comprometer ~6 h:**
- **`--batch 6`, não 8.** O log da rodada promovida não registra o batch, mas o
  **codifica**: 9450 amostras com 1575 passos só fecham com 6. O handoff tinha
  **previsto** que `base=32` não caberia com batch 8 e avisado que baixar o batch
  quebraria a comparabilidade — **o batch foi baixado e o registro disso se
  perdeu**, sobrou só a pegada na contagem de passos. **Os pilotos do Bloco 6 e o
  checkpoint promovido NÃO são comparáveis em batch.**
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` era necessário e estava
  ausente do script. **O alocador sozinho não salva o batch 8** (OOM faltando
  16 MiB); as duas coisas são necessárias.

### 9.5 Ruling 64 — o retreino: o prior caiu, e com ele o "defeito 4"

Rodada de 6h05, 25 épocas, `base=32`, batch 6, 10.950 amostras. **Promovido: época
13.**

| | topo treinado (K<0) | **topo NUNCA treinado** (K>0, eixo invertido) | rodapé (controle) |
|---|---|---|---|
| checkpoint anterior | 0,5433 | 0,5384 | 0,9299 |
| **época 13** | **0,9246** | **0,9174** | 0,9322 |

**A coluna do meio é a prova: condição que não existe em nenhuma amostra de
treino, e ela subiu junto. A rede aprendeu POSIÇÃO, não decorou o estrato.** O
rodapé não pagou nada.

- **O "defeito 4" era o MESMO defeito, e a atribuição anterior estava errada.**
  `caso_real_neg_fopdt.png` era recusada, e o Bloco 9 tinha concluído que a causa
  era "dois objetos de curva no mesmo quadro — envelope novo, spec própria".
  Depois do retreino ela **fecha** (K 0,15 %, τ 0,01 %, θ 0,02 %). **O mecanismo
  descrito estava certo; a CAUSA estava errada:** a rede não via o platô da
  resposta, que fica no topo, e a única linha visível naquela altura era a da
  entrada. **Um retreino fechou os dois defeitos porque eram um só.**

  > **Lição de método:** "dois objetos no mesmo quadro" era plausível e consistente
  > com o sintoma, **e passou porque ninguém pediu a ela que previsse mais nada**.
  > O prior de posição foi encontrado por ablação fatorial, não por inspeção.
- **Efeito no corpus:** 2.6 de +1,63 para **+1,08 p.p.**; 2.12-ordem de 92,3 % para
  **94,0 %**; 2.1 p95 de 1,703 para **1,489 px**; 2.6-aceitas de 254 para **260**.
- **A escolha da época, e por que nenhuma métrica decide sozinha.** A época 11 tem
  o melhor platô (0,9272) mas custa **−0,0092** no conjunto histórico; a 13
  preserva os números históricos e ainda ganha no conjunto novo. **Escolher pela
  métrica de platô sozinha era o mesmo erro de método do §40.9 com o sinal
  trocado.** O `IoU_val` "cego" **escolheu certo**, porque carrega exatamente a
  informação que a métrica de platô ignora.
- **O que o §40.9 NÃO tinha notado:** o `IoU_val` alimenta **dois** consumidores —
  a seleção do checkpoint **e** o `ReduceLROnPlateau`. Os ganhos desta fase vêm em
  passos de 0,001 a 0,007 e o `--lr-threshold` é **0,01 ABSOLUTO**, então o
  scheduler cortou o LR 10 vezes e as últimas 7 épocas não moveram nada. **O platô
  saturou na época 5** — uma rodada de ~12 épocas teria bastado, e **agora isso é
  medição**.
- **O custo:** a cauda assentada com reta de referência coincidente **piorou** —
  probabilidade mediana **0,0004** nas colunas perdidas (contra 0,1811 do anterior),
  supressão confiante, não limiar. Não afeta o resultado físico. Registrado como
  `xfail` estrito.
- **Uma falsa regressão, e o erro que ela expôs.** Um teste acusou 17 % de erro em
  ωₙ. **Não havia erro em ωₙ:** o teste lia `wn_T / 10`, onde `wn_T` é normalizado
  pelo SPAN DA SÉRIE e o 10 é a janela do eixo — supunha que a máscara cobre a
  janela inteira. Com o span caindo, a conta erra 18 %. **Duas coisas estavam
  desatualizadas: a premissa e a confusão entre COBERTURA e ACURÁCIA.**

### 9.6 Ruling 65 — `K` é `K_planta × U`, e isso não é conserto de software

Reportado como classificação errada: a pipeline devolveu `K = −1,997` onde a
função de transferência diz `K = 1`. **Não houve erro.** A planta é `2/(s+2)`
(ganho DC = 1) e o degrau aplicado tem amplitude **−2**. `STEP_AMPLITUDE = 1.0` é
convenção do projeto, então o `K` reportado é `K_planta × U = −2`. Verificado
reconstruindo a curva: **erro máximo 0,0039**.

**Não é limitação de implementação, é identificabilidade** — da curva de saída
sozinha, `K_planta` e `U` não são separáveis, só o produto é observável. **O
leitor humano acerta porque a figura desenha a entrada e ele divide; a pipeline
não lê a entrada.**

**Aberto, e agora plausível:** ler a amplitude do degrau da imagem. Era impossível
antes do retreino e **passou a ser plausível porque a máscara agora separa a
resposta da tracejada** — é exatamente o que fechou o antigo "defeito 4".

### 9.7 Ruling 66 — era a LEGENDA, e a atribuição errou duas vezes antes

`caso_real_neg_super.png` devolvia wn=2,95 (erro 26 %) e zeta=0,87 (erro 30 %). O
dono moveu a legenda de `lower left` para `upper right`, gerou a mesma figura de
novo, e a pipeline passou a devolver **wn=3,88 (2,97 %) e zeta=1,22 (2,23 %)**.

| faixa de t | original | legenda movida |
|---|---|---|
| platô inicial | 0,0077 | 0,0077 |
| transitório rápido | 0,0201 | 0,0205 |
| **acomodação (4,5–6,0)** | **0,1674** | **0,0090** |
| cauda assentada | 0,0073 | 0,0068 |

**Uma faixa mudou, 19×. Todas as outras são idênticas.** A polilinha segue a borda
da caixa da legenda e cria um patamar falso, antecipando a acomodação em ~0,7 s.
**O Estágio D está inocente, e isso foi medido:** o oráculo na MESMA grade de 593
pontos recupera wn=4,0000 e zeta=1,2500 com **NRMSE zero**.

**Duas atribuições erradas, escritas no repositório antes de serem refutadas:**
1. *"Perde a cauda assentada"* — herdado do Ruling 62, que era sobre OUTRA imagem.
   A cauda tem rms 0,0073, está perfeita.
2. *"Atração pela tracejada de entrada em −3"* — a polilinha era puxada para −2,93
   e "mais perto de −3" foi tratado como evidência. **Mas a caixa da legenda
   ocupava a MESMA vizinhança.**

> **A evidência disponível era compatível com as duas hipóteses, e uma foi
> escolhida sem o teste que as separa.** O que resolveu foi um experimento de uma
> variável: mover a legenda. **A pergunta que faltou nas duas vezes foi "qual
> observação distinguiria isto da alternativa?"**

**E o corpus já media isso, fraco demais para alguém agir.** O critério 2.7
estratificado por legenda está no relatório desde antes: `2.7-iou[legenda=False]`
0,6758 contra `legenda=True` **0,6148** — 6,1 pontos de IoU em quase metade do
corpus. **O número existia e nunca foi ligado a nada, porque IoU DILUI.** A imagem
real mostrou o mesmo dano em unidade de parâmetro físico: 26 % em ωₙ.

**O par controlado ficou versionado** (as duas variantes lado a lado, uma como
portão e a outra como `xfail` com a razão certa). **Enquanto os dois coexistirem,
nenhuma explicação alternativa sobrevive: uma variável muda, o resultado muda com
ela.**

---

# Estado ao fim do Bloco 9 (04/09/2026)

**Todos os critérios da Parte 2 com alvo estão APROVADOS.** `tests/part2/`:
131 passam, 9 `xfail`, zero falhas. Fonte de verdade:
`reports/part2_strata.md` (gerado — não editar).

| # | Critério | Alvo | Medido |
|---|---|---|---|
| 2.1 | erro perpendicular da máscara | ≤ 1,0 / 2,0 px | **0,804 / 1,489 px** ✅ |
| 2.2 | polilinha vs. máscara verdadeira | ≤ 1,0 / 2,0 px | **0,615 / 1,137 px** ✅ |
| 2.3 | erro relativo de escala | < 1 % em ≥ 95 % | **0,986** (n=280) ✅ |
| 2.4 | taxa de rejeição | *aposentado* (Ruling 52) | 0,067 ❓ |
| 2.5 | rejeições corretas | ≥ 90 % | **0,900** (n=20) ✅ |
| **2.6** | degradação end-to-end | ≤ 3 p.p. | **+1,08 p.p.** (n=260) ✅ |
| 2.6-adim[ζ] | degradação adimensional | ≤ 3 p.p. | **+0,95 p.p.** ✅ |
| 2.7 | perpendicular por estrato | ≤ 1,0 px | 0,672–0,964 px, todos ✅ |
| 2.8 | latência por imagem | < 500 ms | **168 ms** (p95 310) ✅ |
| 2.9 | cobertura da calibração | ≥ 90 % | **0,933** (n=300) ✅ |
| 2.11 | bloco `dimensionless` | 100 % | **300/300** ✅ |
| 2.12 | acerto de ordem | diagnóstico | 94,0 % ❓ |

Diagnósticos que continuam reportados **sem alvo, de propósito** (Ruling 50): IoU
da máscara (0,6473) e RMSE vertical (1,49 / 6,70 px). Removê-los esconderia;
mantê-los sem veredito preserva a comparabilidade com as rodadas 3 a 6.

---

# Os padrões de método que atravessam o projeto

Estes são o material de metodologia da monografia. Cada um foi pago com uma
rodada de trabalho.

1. **Limiar fixado sem calcular o máximo atingível.** Parte 1 (1.4, 1.5, duas
   vezes na sanidade da máscara), Ruling 39 (IoU ≥ 0,85 fixado sem considerar a
   largura da linha), Ruling 63.
2. **Calibrar um limiar na população errada.** `_FALTA_ESQ_MAX_FRAC` (68 % de erro
   de um lado, nenhuma amostra do outro) e o diagnóstico de ωₙ nascido diluído 7×.
   *Antes de calibrar, pergunte se a população em que ele foi medido é a população
   em que ele vai decidir.*
3. **Achatar uma distribuição num escalar.** O "ponto de empate ≈ 0,872" que era a
   mediana de uma distribuição de 0,79 a 0,96, usada para uma decisão tomada
   amostra a amostra.
4. **O critério que existe, passa, e não cobre nada.** Porque a grandeza é
   *invariante* ao modo de falha (ζ e a escala do tempo), ou porque a *população*
   é majoritariamente de amostras que não passam pelo caminho vigiado. Hoje há
   guarda permanente contra a segunda forma.
5. **O filtro que entra só como filtro.** Uma regressão de ordem não piora mediana
   nenhuma — ela **encolhe** a amostra, e as que somem são as difíceis, então as
   medianas que restam até melhoram.
6. **O teste circular.** Quando o experimento constrói a evidência a partir do
   modelo que ele deveria testar, o resultado é tautologia. *O sinal de alerta é
   simples: o teste não tem um caso em que possa falhar.*
7. **Hipótese consistente com o sintoma não é hipótese confirmada.** Duas vezes na
   mesma imagem (Ruling 66) e uma no "defeito 4" (Ruling 64). *A pergunta que
   falta é sempre "qual observação distinguiria isto da alternativa?".*
8. **Nunca concluir ausência de hardware a partir de uma biblioteca compilada sem
   suporte a ele** (Ruling 17). O teste era circular por construção.
9. **Uma porta de aceitação que reprova é informação sobre o sistema, não sobre a
   hipótese** (Ruling 58 — o maior ganho do Bloco 8 veio da porta, não do item).
10. **O corpus mede o CUSTO de uma guarda e não mede o BENEFÍCIO** quando ele não
    contém o modo de falha que a guarda existe para pegar. A ordem correta é
    estrato primeiro, guarda depois.
11. **Decompor uma cadeia por mediana de estágio produz conclusão invertida** —
    use delta pareado por amostra (Ruling 42).
12. **Um número errado que ninguém distingue de um certo é pior que nenhum.** É o
    que justifica recusar em vez de degradar.
13. **Registrar a dívida em `xfail(strict=True)`, não só em texto.** No dia em que
    o defeito for consertado, o teste vira XPASS e **reprova a suíte**, obrigando
    quem consertou a converter o defeito documentado em portão de regressão. É o
    que impede a dívida de sumir.
