# A frente de OCLUSÃO POR LEGENDA — o que foi medido e por que ela parou

Registro de 14/09/2026. **A frente não foi resolvida, e a conclusão é que ela
não se resolve onde estava sendo atacada.** O corpus que a atacava funciona — e
foi medido funcionando. O que sobrou é outro problema, em outro estágio.

Nada aqui foi removido do código: o estrato `legenda_oclusora` / `legenda_no_canto`
continua em `dataset/generator.py`, os corpora em `data/*_parleg*` e
`data/*_parleg2*`, e os checkpoints em `models/epocas_combinado/` e
`models/epocas_combinado2/`. Quem retomar encontra aqui o que já está
respondido.

## 1. O defeito

`tests/fixtures/caso_real_neg_super.png` é uma figura real com a caixa da
legenda sobre a curva. `caso_real_neg_super_legenda_movida.png` é a MESMA
figura com a legenda em outro canto — o par controlado que isola a variável.
Verdade: `K = -3`, `wn = 4`, `zeta = 1,25`, `theta = 3,5`.

| modelo | legenda OCLUINDO (wn / zeta) | legenda MOVIDA |
|---|---|---|
| promovido (`ad1253813caf5d3f`) | 1,3 % / 0,2 % | 3,3 % / 2,8 % |
| `render2` época 17 | 19,7 % / 18,1 % | 2,1 % / 1,5 % |

A coluna da direita boa e a da esquerda ruim é o que define o dano como sendo
de OCLUSÃO. O portão é `TOL = 0,06` em `tests/part2/test_caso_real_negativo.py`.

**Por que importa:** a época 17 do `render2` é a que entrega o objetivo do
`|K| < 1` — ESTRITO total de 77 % para 81 %, `|K| < 1` de 75 % para 82 %. Ela
foi promovida e REVERTIDA por causa destes portões. A oclusão é o que segura o
ganho do `|K| < 1`, não uma frente paralela.

**Estado em 14/09/2026.** Eram CINCO portões. Hoje são DOIS, e os dois são
estas duas asserções sobre esta imagem:

| portão | estado |
|---|---|
| `neg_super[wn]` | **aberto** — esta frente |
| `neg_super[zeta]` | **aberto** — esta frente, MESMA imagem |
| `sistema1_nao_inventa_polo_irresoluvel` | RESOLVIDO — `POLO_MIN_AMOSTRAS` |
| `sistema1_recupera_K_tau_theta` | RESOLVIDO — mesma causa |
| `estagio_a_cobre_a_janela_inteira[caso2-sistema3]` | não era quebra: XPASS(strict), a máscara CONSERTOU o defeito B do §39.3 e o teste fica vermelho porque a falha está registrada como esperada. Basta tirar a marca. |

Os dois do `sistema1` eram a MESMA degenerescência desta frente, do outro lado:
lá o ajuste INVENTAVA um polo rápido (ζ = 3,09, τ = 27 ms = 1,4 amostras) numa
planta de 1ª ordem; aqui ele PERDE o polo rápido legítimo (ζ = 1,25) quando a
caixa desloca `theta`. Um critério de resolubilidade do polo fechou o primeiro
lado (ver `POLO_MIN_AMOSTRAS` em `identify/classical.py`) e NÃO fecha o
segundo, porque aqui o polo rápido é resolúvel — 5,2 amostras — e o que se
perde é a posição de `theta`, não a estrutura.

## 2. Duas corridas de retreino, e o que elas mediram

**`retreino_combinado.sh`, primeira corrida** (corpus `parleg`, morreu na época
12 de 18 quando a sessão caiu e levou o grupo de processos). Dos 13
checkpoints, 1 passa o portão. Controle: a corrida `render2`, mesmo objetivo
SEM o estrato pareado, passa 3 de 18. **Fisher p = 0,62** — o par não mudou
nada.

A única candidata (época 10) entregava +1,7 pp de ESTRITO nos três lotes de
controle (230 → 235 de 300; 18 sobem, 13 descem; **p = 0,47**) e ficava no piso
exato da guarda (`val_nmp[:60]` = 90,0 %, folga zero). Não foi promovida.

**A causa da falha era um defeito de GERADOR, não de treino.** Medido nos 200
pares de `val_parleg_*`:

- a âncora era `theta + U(0,6; 3,0) · t_dom` — 0,6 a 3 constantes de tempo
  DEPOIS do início, o que é o patamar assentado e não a transição;
- o joelho caía DENTRO da caixa em **6 de 193 pares (3,1 %)**, e a 298 px dela
  (p50) nos outros;
- em 12 % das amostras a âncora era grampeada na borda do quadro para QUALQUER
  sorteio;
- a deformação da série na janela da transição dava **p50 = 0,0 px** e p90 de
  1,0 a 2,0 px nos sete checkpoints medidos, contra 6,8 px da figura real.

Sem dano no corpus não há o que aprender nem o que medir. É por isso que os
quatro instrumentos sintéticos de oclusão (recall sob a caixa, cauda do erro
pareado, patamar falso, invariância pareada) não ordenavam os checkpoints: os
dois últimos INVERTIAM.

**O erro de fundo era de UNIDADE.** `t_dom` é escala da DINÂMICA; a caixa é
objeto de LAYOUT, medido em fração do eixo. Na figura real `t_dom = 0,5 s` numa
janela de 10 s, então a caixa (0,43 do eixo) tem quase NOVE `t_dom` de largura e
qualquer deslocamento medido em `t_dom` some diante dela.

## 3. A geometria real, medida

Pela série extraída da metade de controle (`models/unet_stageA.pt`):

- repouso em `y = 72 px`, patamar em `y = 317 px`;
- o transitório inteiro cabe em **x 248 a 281 — 33 px de uma curva de 577**,
  ou 5,7 % da largura;
- a caixa vai de x 104 a 358 (**254 px**) e de y 331 a 406;
- ela cobre o tempo morto (144 px), o transitório todo e mais 77 px de patamar;
- a borda de CIMA corre **14 px abaixo do patamar** e se estende para trás
  cruzando a transição.

É essa borda longa que a rede segue como se fosse patamar — acomodação
antecipada. O `theta` sai tarde, o joelho em S some, e o ajuste vira FOPDT.

Medido nos 13 checkpoints da primeira corrida, o `theta` estimado é o pivô:

| `theta` estimado | o que sai |
|---|---|
| ≥ 3,57 s | `fopdt` |
| ≈ 3,50 s | correto, ζ 1,25–1,28 |
| ≤ 3,48 s | `second` com ζ 0,75–0,96, erro 26–40 % |

A faixa inteira entre os checkpoints é de **0,18 s — 1,8 % da janela**.

## 4. A correção do gerador

Três mudanças em `dataset/generator.py` (ver o bloco de `_OCLUSAO_CHEGADA`):

1. a referência passou a ser a **CHEGADA ao patamar** (`_OCLUSAO_CHEGADA = 0,95`
   da excursão), não `theta + k·t_dom`;
2. o deslocamento passou a ser em **meias-larguras da caixa**, medidas com
   `get_renderer()` depois do layout — `_OCLUSAO_FX_LIM`, que chutava 0,12 e
   0,88, deixou de existir;
3. a caixa ficou mais larga (`_OCLUSAO_N_TEXTOS` de 1–3 para 2–4 textos).

Medido em 150 pares, contra o corpus antigo e a figura real:

| | antigo | corrigido | figura real |
|---|---|---|---|
| curva tapada pela caixa (p50) | 0,051 | **0,080** | 0,1505 |
| deformação na chegada, p50 (promovido) | 3,9 px | **4,5 px** | 6,8 px |
| deformação na chegada, p50 (época 10) | 3,3 px | **4,4 px** | 8,0 px |

O teste do estrato foi trocado junto. O antigo
(`test_a_ancora_cobre_o_joelho_e_nao_o_repouso`) conferia só os NÚMEROS da
constante e ficou verde enquanto a caixa caía a 298 px do joelho em 96,9 % das
amostras — asseverava a intenção, não o desfecho. O novo
(`test_a_caixa_cai_sobre_a_transicao_e_se_estende_para_tras`) mede a pegada
exata da caixa contra o render sem legenda, e foi **verificado reprovando a
fórmula antiga**.

## 5. A segunda corrida, e o resultado que encerra a frente

`retreino_combinado.sh` com o corpus corrigido (`parleg2`), 18 épocas
completas, `models/epocas_combinado2/`.

**O corpus corrigido FUNCIONOU.** Deformação da série na janela da transição
(x 240–300) do par real:

| corrida | n | p50 | abaixo de 9 px |
|---|---|---|---|
| `combinado2` (corpus corrigido) | 18 | **6,97 px** | **12/18** |
| `combinado` (corpus antigo) | 13 | 19,70 px | 3/13 |
| `render2` (controle, sem o estrato) | 18 | 20,08 px | 2/18 |

Mann-Whitney: **p = 2,1e-03** contra o controle, **p = 2,0e-03** contra a
corrida anterior. As épocas tardias chegam a **2,0–2,5 px**.

**E o portão continuou fechado.** No critério real (`TOL = 0,06`), **1 de 18**
passa — a época 02. Os checkpoints de MENOR deformação são os que quebram:

| checkpoint | deformação | veredito |
|---|---|---|
| época 12 | 2,00 px | quebra |
| época 16 | 2,42 px | quebra |
| época 17 | 2,47 px | quebra |
| época 04 | 16,65 px | passa na varredura, reprova em ζ (7,9 % > 6 %) |

## 6. Por que os dois resultados convivem — e onde o gargalo está agora

| checkpoint | imagem | θ | ζ | wn | nrmse |
|---|---|---|---|---|---|
| `combinado2` ep. 17 | **ocluindo** | 3,471 | 1,026 | 3,212 | 0,0051 |
| `combinado2` ep. 17 | movida | 3,505 | 1,235 | 3,933 | 0,0017 |
| `combinado2` ep. 04 | **ocluindo** | 3,500 | 1,151 | 3,833 | 0,0045 |
| `combinado2` ep. 04 | movida | 3,503 | 1,221 | 3,884 | 0,0043 |

A época 17 tem 2,47 px de deformação e erra 19,7 %. Esses 2,5 px deslocam
`theta` em **0,034 s — 0,34 % da janela** — e o deslocamento leva ζ de 1,235
para 1,026. A época 04 desloca `theta` em 0,003 s, dez vezes menos, e chega
perto de passar.

E o `nrmse` do ajuste errado é **0,0051**: o ajuste errado descreve a curva tão
bem quanto o certo. Assinatura de problema mal condicionado — numa 2ª ordem
superamortecida (ζ = 1,25) muitos trios (θ, ζ, wn) explicam a mesma série.

**Conclusão: o gargalo saiu do Estágio A.** Era a robustez da máscara à
oclusão, e essa foi medida, atacada e reduzida em ~3× com significância. O que
resta é o CONDICIONAMENTO do ajuste de 2ª ordem, que converte 2,5 px de máscara
em 20 % de erro de parâmetro sem que o resíduo acuse nada. Um Estágio A
perfeito ainda teria de acertar `theta` dentro de ~0,003 s numa figura onde a
legenda cobre a transição.

**Mais retreino não fecha este portão.** Duas corridas, 31 checkpoints, dois
corpora, e a taxa de passagem não se moveu (3/18, 1/13, 3/18 no critério
frouxo; Fisher p = 1,00 e p = 0,62).

## 7. O que está respondido e o que não está

RESPONDIDO:

- o dano é de oclusão, não de 2ª ordem superamortecida em geral (par controlado);
- ele se concentra na janela da transição, não sob a caixa inteira;
- o corpus sintético ENSINA invariância quando a geometria está certa (p = 2e-03);
- e ensinar invariância NÃO basta, porque o ajuste amplifica 2,5 px em 20 %.

NÃO RESPONDIDO:

- se a informação para separar (θ, ζ, wn) está na imagem e o ajuste não a usa,
  ou se ela simplesmente não está;
- o portão real continua com **n = 1**. Uma figura. Todas as taxas de
  sobrevivência desta frente são sobre a mesma imagem.

## 8. Se retomar, por onde

1. **Condicionamento do ajuste** (`identify/classical.py`) — é onde o gargalo
   está, não precisa de GPU, e beneficia qualquer figura superamortecida.
   Detectar má determinação de `theta` e REPORTAR incerteza vale mais que um ζ
   confiante e errado.
2. **Figuras reais com legenda sobre a curva.** Com n = 1 não há como
   distinguir "o método funciona" de "esta imagem é assim". É a única coisa que
   transforma o portão em medida.
3. **Não mexer no corpus.** Ele está medido, corrigido e com teste de desfecho.
   O problema não é mais ele.
