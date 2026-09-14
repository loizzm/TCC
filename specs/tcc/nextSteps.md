# nextSteps — o que está aberto

**Documento vivo.** Reescrito sempre que algo fecha ou entra. O histórico do que
já foi decidido está em [`TIMELINE.md`](TIMELINE.md) — este arquivo só olha para
frente.

> Última revisão: **04/09/2026**, fim do Bloco 9 (`bc41fea`).
> Estado da suíte: `tests/part2/` **131 passam, 9 xfail, zero falhas**.
> **Todos os critérios da Parte 2 com alvo estão aprovados.**

---

## 0. A pergunta que decide a ordem

A Parte 2 fechou seus critérios. **O que ainda não existe é a Parte 3** — e ela é
que carrega o argumento do trabalho. Tudo abaixo de §3 é dívida da Parte 2 que
**não bloqueia** a Parte 3; tudo em §1 e §2 bloqueia ou dá alavancagem grande.

Regra herdada, e ela vale para cada item daqui: **fixe o critério de aceitação
ANTES de implementar, e inclua o falso positivo.** Qualquer coisa que suba
cobertura sem medir falso positivo repete um erro já pago.

---

## 1. Maior alavancagem — ampliar o acervo externo

**Este continua sendo o passo de maior retorno do TCC inteiro, e não só de um
critério.** Toda vez que uma imagem de fora entrou, ela achou um defeito que
centenas de amostras sintéticas não achavam:

| imagem | achou |
|---|---|
| `resposta_degrau.png` | ordem errada por máscara + o estimador de repouso com θ=0 (Rulings 55/56) |
| `Figure_322` | quatro números sem sentido com `ok=true` (Ruling 61) |
| três de `rg.py` | curva rente à moldura, trecho reto, blob de rótulo cortado (Ruling 62) |
| três de `rg_negativo.py` | degrau negativo inexprimível + o prior de POSIÇÃO (Rulings 63/64) |
| par de legenda movida | a oclusão pela legenda, com 26 % de erro em ωₙ (Ruling 66) |

### 1.1 Versionar as imagens que faltam — **dívida antiga e barata**

Das treze do Ruling 61, **só `caso_real_2ordem` virou fixture**. As outras doze
(`Figure_11`, `_12`, `_122`, `_15`, `_16`, `_21`, `_22`, `_f1`, `_f2`, `_f3`,
`_222`, `_322`) vieram por conversa e vivem em `/home/loizm/`. **Doze imagens que
já mediram coisas e não protegem nada em regressão.**

Hoje versionadas em `tests/fixtures/`: `caso_real_2ordem`, os três de `rg.py` e os
quatro de ganho negativo (incluindo o par controlado da legenda).

**É a tarefa mais barata desta lista: são arquivos, não código.**

### 1.2 Coletar o conjunto OOD da Parte 3 (~60 imagens)

MATLAB/Simulink, Python Control, figuras de livro (Ogata, Nise), planilhas e ~10
curvas de plantas reais. **Sem esse conjunto o trabalho demonstra apenas que o
sistema aprendeu a inverter o gerador.** O `PLANO` diz para começar no Dia 1, não
no Dia 17 — e o Bloco 8 mostrou por quê.

---

## 2. Parte 3 — nada dela existe ainda

A Parte 3 **não constrói componente novo** (o pipeline está completo ao fim da
Parte 2). Ela **mede**. Arquivos a criar: `tests/test_part3.py`,
`reports/final_report.md`, `ood/`, `e2e/`.

Os três critérios que carregam o argumento:

- **3.6 / 3.7 — NRMSE de reconstrução**, métrica primária, no sintético e no OOD.
- **3.10 — PID por IMC** sobre o modelo identificado vs. sobre o verdadeiro, em
  malha fechada simulada. **Fecha o círculo com o título do curso**: mostra que o
  erro paramétrico residual é irrelevante para a finalidade de controle.
- **3.9 — CNN fim-a-fim × pipeline em estágios.** Converte a Decisão B de
  argumento em medição, e testa a hipótese de vazamento pelo seu sintoma
  observável (generalizar pior fora da distribuição). Plano em
  `PLANO_CNN_FIM_A_FIM.md`.

**3.11 — latência com o `base=32`.** O alvo é < 2 s em CPU e o modelo cresceu
78 % desde a última medição. **Não foi remedido depois da promoção.** Barato, e
pode exigir decisão.

**3.12 — o gatilho que ressuscita o Estágio C.** Medir, sobre as séries
**extraídas** (não as do oráculo): taxa de convergência de `identify` ≥ 99 % e
NRMSE p95 ≤ 0,02. Se qualquer um falhar, a spec do Estágio C volta à mesa (está
preservada no histórico do git). **Reportar os dois números explicitamente mesmo
que passem folgado — um critério que ninguém mede é decoração.**

---

## 3. Estágio A — o que o retreino não consertou

### 3.1 Os dois defeitos do `rg.py` (Ruling 62) — `xfail(strict=True)` ativos

`tests/part2/test_caso_real_rg.py::test_estagio_a_cobre_a_janela_inteira`:

| | cobertura | causa |
|---|---|---|
| Sistema 1 | **65,3 %** | **defeito A** — curva rente à moldura inferior; perde o platô do tempo morto inteiro |
| Sistema 2 | 96,9 % | passa (controle negativo) |
| Sistema 3 | **82,2 %** | **defeito B** — trecho perfeitamente reto; perde a cauda assentada |

**Coerente com o prior de posição:** essas imagens são de ganho **positivo** com
`plt.ylim(0, ...)`, então o platô delas fica no **rodapé**, que a rede já dominava
— o retreino do Bloco 9 não as toca.

**O que o retreino precisa:**

1. **Estrato de margem inferior quase nula — NÃO EXISTE.** `y_margin_lo` sorteado
   em 0 a 1 % do span, contra o piso atual de 3 %. Ataca A.
2. **Estrato de cauda LONGA e assentada.** O `train_reta` ataca **oclusão**, não
   **retidão**. Falta janela larga o bastante para a resposta assentar de verdade e
   ficar reta por muitas colunas. *Nota: o corpus atual tem 442 de 600 amostras com
   `w < 3` — a maioria **nunca assenta**, e o caso que quebra é justamente o
   sub-representado.* Ataca B.
3. **Remedir o G3b.2 depois.** Ele é o que ensina a suprimir reta, e o defeito B é
   efeito colateral dele. **Retreinar sem remedi-lo pode trocar um defeito por
   outro**: reta de referência voltando para dentro da máscara.

### 3.2 A cauda assentada sob reta coincidente (Ruling 64) — `xfail` ativo

`tests/part2/test_caso_real.py::test_caso_real_cobre_a_cauda_assentada`. O
retreino trocou cobertura de cauda por platô no topo: perde 51 das 747 colunas com
probabilidade mediana **0,0004** — supressão confiante, não limiar.

**Não afeta o resultado físico** (ωₙ a ~1 % da verdade) nem nenhum critério do
corpus. Fica registrado porque **onde a curva e a reta coincidem pixel a pixel a
tarefa é mal-posta**: "manter a curva, suprimir a reta" não tem resposta única.

### 3.3 A oclusão pela legenda (Ruling 66) — `xfail` ativo, **com hipótese pronta**

`tests/part2/test_caso_real_negativo.py`. **Não precisa de retreino de cauda nem
de estrato novo de cauda** — o Estágio D está inocente (o oráculo na mesma grade
recupera com NRMSE zero). **Precisa que o Estágio A atravesse a legenda.**

**O candidato, agora com número:** um estrato de **legenda SOBREPOSTA AO TRECHO DE
ACOMODAÇÃO**, em vez de legenda em posição sorteada. O gerador já tem
`has_legend`; falta posicioná-la onde faz estrago. O corpus já mede o dano e
ninguém tinha ligado: `2.7-iou[legenda=False]` 0,6758 contra `legenda=True`
**0,6148**, em quase metade do corpus.

### 3.4 Ganho negativo: a U-Net viu o platô, mas o estrato não entrou no treino base

O estrato existe no gerador (`ganho_negativo=True`) e `data/train_kneg` entrou no
retreino do Bloco 9. **O que continua apoiado em poucos exemplos reais é o resto**
— e o Ruling 63 mostrou que **qualquer figura de eixo y invertido cai no mesmo
regime, mesmo com K > 0**, e não há nenhuma amostra assim no corpus.

### 3.5 Calibrar o estrato `reta_no_patamar` até reproduzir a TRUNCAGEM

Hoje ele simula o **objeto** (reta tracejada no patamar) e **não o fenômeno**
(fusão da máscara → polilinha truncada): em 30 seeds a cobertura mediana é 0,9403
e nenhuma amostra cruza `_COBERTURA_MIN_MOLDURA`. **Enquanto isso valer, o estrato
NÃO pode ser citado como validação da normalização pela moldura.**

Alavancas, todas em `dataset/generator.py`: aproximar a cor da reta da cor da
curva, engrossar o traço, subir a reta no `zorder`.
**Critério de aceitação, objetivo: a cobertura mediana do estrato tem de cair
abaixo de 0,75** (hoje 0,94).

⚠️ **Mexer no gerador é mexer no rótulo.** Qualquer alavanca tem de ser
reconferida contra a invariante de que `mask.png` é byte-idêntico entre
`reta_no_patamar=False` e `True` na mesma seed.

---

## 4. Detecção do que está FORA da família de modelos

O Ruling 61 fechou com o diagnóstico: **o sistema não tem detector para a maior
parte do que está fora da família.** Há guarda para resposta inversa e resíduo
alto. **Não há para instável, ordem superior, zero no semiplano esquerdo, nem
múltiplas curvas** — e a `Figure_322` mostra que **resíduo baixo não garante nada
quando o otimizador tem uma caixa grande para fugir**.

### 4.1 Guarda de parâmetro na borda — **fazer primeiro**

Ataca a pior saída possível: quatro números sem sentido com `ok=true`. **O sinal é
grátis** — a informação já existe em `K_BOUNDS`/`WN_BOUNDS`/`ZETA_BOUNDS`/
`TAU_BOUNDS` e custa uma comparação exata, **sem limiar calibrado contra corpus**.

**Critério de aceitação a fixar ANTES de implementar, e a primeira pergunta é o
CUSTO:** quantas das 900 amostras têm algum parâmetro na borda na estrutura
escolhida? Medir **separado** para K, τ, ωₙ e ζ.

⚠️ **A armadilha já está medida:** a `Figure_12` (ζ=0 verdadeiro) encosta no piso
do ζ **legitimamente**, e uma guarda ingênua sobre ζ a rejeitaria — ela está
certa. **A hipótese a testar é K na borda**, que não tem caso legítimo à vista.

### 4.2 Sinalização de ordem ambígua

Ataca a `Figure_f3`. Quando o ganho do teste de ordem fica muito abaixo do limiar
(0,308 contra 2,0), **a evidência para a estrutura mais complexa não existe — e
hoje a pipeline escolhe a mais simples em silêncio**. Uma faixa de indecisão
marcaria `ordem_incerta` e entregaria os parâmetros das **duas** estruturas,
deixando a escolha para quem tem o contexto físico.

**Validar contra `2.12-ordem` (94,0 %): a guarda não pode derrubá-lo.**

### 4.3 Detecção de múltiplas curvas

Ataca a `Figure_222`. **Escopo congelado no `PLANO §1.4`, então o objetivo NÃO é
identificar duas curvas — é RECUSAR em vez de mesclar em silêncio.** Sinal
candidato: componentes conexas com extensão horizontal comparável à da moldura, em
número maior que um, após o filtro de retas de span completo.

### 4.4 Ler a amplitude do degrau da imagem (Ruling 65)

Com `U` lido, `K_planta = K_reportado / U` sai de graça. Exige detectar e
**classificar** um segundo objeto de curva como "entrada" — **era impossível antes
do retreino do Bloco 9 e passou a ser plausível**, porque a máscara agora separa a
resposta da tracejada. Envelope próprio, spec própria.

**Versão barata do mesmo problema:** `identificar.py` imprime `K -1.997` sem dizer
que é ganho **por degrau unitário**. Um rótulo explícito na saída evitaria a
leitura errada sem tocar em nada do cálculo. *Decisão do dono.*

---

## 5. Instrumentação e limites conhecidos

### 5.1 A linha `A.0` descreve um modelo que não está em uso

`test_unet_tamanho_declarado` faz `sum(p.numel() for p in UNet().parameters())` —
instancia o **default `base=16`** em vez de ler o checkpoint. Reporta **1,94 M**
para um modelo que hoje tem **7.768.947**. Pior: o `assert 0.5e6 <= n <= 2.5e6`
**reprovaria** se algum dia medisse o modelo real.

### 5.2 `requirements.txt` ainda fixa `torch==2.13.0+cpu`

Quem reconstruir o ambiente ao pé da letra **volta para CPU e vai achar que a GPU
não funciona**. A linha para GPU é
`torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130`.

### 5.3 `2.6-adim[wn_T/sem-calib]` continua **sem meta**, com n pequeno

Está provado que detecta uma regressão **grande** (fator 7,44 de deslocamento).
**Não há evidência de que detecte uma pequena**, e o `n` não sustenta limiar.
Transformá-lo de diagnóstico em critério é trabalho de bloco próprio.

### 5.4 Ponto cego da guarda de planura: o corte profundo que RECOMEÇA plano

`_PLANURA_MAX_FRAC` **não distingue "parado no repouso" de "parado no patamar
assentado"**. Um corte à esquerda profundo o bastante para que a série remanescente
comece já assentada tem planura ~0 e passa. Medido em 82 séries: ocorre 3 vezes,
sempre com cobertura ≤ 0,45, com erro de ζ de 0,49 %, 6,43 % e **20,33 %**.

O segundo facet óbvio do mesmo invariante (exigir que o nível lido seja o extremo
de `y`) foi testado e **não fecha**. **Fechar exige distinguir repouso de patamar
sem proxy geométrico** — e o Bloco 8 registra por que NÃO se fecha isso com mais um
limiar de geometria. *Nenhuma amostra de nenhuma das duas populações reais está
nesse regime hoje.*

### 5.5 `_COBERTURA_MIN_MOLDURA = 0.75` só está examinado de UM lado

Para baixo é confortável (caso real em 0,617). **Para cima a folga é de 2,1 p.p.**
— o mínimo do corpus sintético é 0,7713. Uma amostra entre 0,75 e 0,77 passaria a
usar a moldura **sem nunca ter sido examinada**. Nenhuma existe hoje; **é o número
mais frágil do Bloco 8**.

### 5.6 A guarda de instrumentação vigia uma função só

`tests/part2/test_instrumentacao.py` cobre a região de relatório de
`test_2_6_degradacao_vs_oraculo`, que é onde as duas reincidências aconteceram. Os
outros portões numéricos foram conferidos um a um **e essa conferência não tem
teste que a mantenha** — um portão novo em OUTRA função nasceria sem guarda. E
**nada vigia o *conteúdo* dos critérios, só a presença deles.**

### 5.7 `test_1_4b` nunca exercita `reta_no_patamar=True`

Ela reconstrói o estilo **a partir da seed**, e `reta_no_patamar` é argumento de
*render*: não sai da seed, então o round-trip sempre reconstruiria `False`. Cobrir
exige uma fixture com o estrato e o teste ler `m["render"]["has_reference_line"]`,
no mesmo padrão que já usa para `snr_db`.

### 5.8 Limpezas menores, catalogadas e sem evidência de morder hoje

`_nivel_de_repouso` com série vazia · `np.median` em vez de `nanmedian` ·
`any(bbox_px)` duplicado · `bbox` invertido/degenerado zera a máscara e falha
seguro **mas silenciosamente** · `dentro.size == 0` em `polyline.py` é código
morto · `ZETA_BOUNDS` tem piso `1e-3`, então ζ=0 verdadeiro sai como 0,001 (não é
medição, é o limite da caixa).

### 5.9 A coluna logo depois de um vão largo

Pode errar **~80 px mesmo com o reset ativo**, por empate de distância entre blocos
resolvido pela ordem da lista. Peso maior do que parece: **"vão largo com objeto
concorrente por perto" é o regime REAL deste código**, não hipótese de
laboratório. Um erro grande numa coluna vira o `anterior` da coluna seguinte — há
**risco de propagação, NÃO MEDIDO**.

---

## 6. Dívida da Parte 1

- **A revisão final ampla da Parte 1 como entregável único NUNCA foi feita.** As
  três revisões por tarefa foram; a de conjunto, não.
- **1.4d não fala sobre ~25 % das amostras** (traço fino, sem pixel de interior
  puro). Não abre brecha para vazamento por `order` — esse atingiria todas —, mas
  abre para um que só existisse no estrato de traço fino. *Improvável a ponto de
  não valer mais complexidade.*
- **1.4e não fala sobre amostras com grade** (51 %), porque o `render` declara
  `has_grid` mas não quantas linhas. **Fechar exigiria mudança de contrato do
  gerador, não de teste.**
- **Igualdade de bytes depende do ambiente pinado** (Ruling 11): o critério 1.6
  compara sha256 de PNG e é válido **dentro** de uma máquina, não entre máquinas.

---

## 7. Coisas que NÃO fazer — refutadas com número

Cada linha custou uma rodada. **Não repetir sem evidência nova.**

| Não faça | Por quê |
|---|---|
| **Aumentar o dataset, augmentation, regularização** | reprovado por duas vias independentes: fatorial 2×2 do Bloco 6 (piora nos dois níveis de capacidade) e gap de +0,0105 (não há sobre-ajuste para combater). Rulings 13, 24 |
| **Treinar `base=32` para subir IoU** | o IoU não prediz o end-to-end (Ruling 25), mede espessura de traço e não acurácia (Ruling 39), e a rede contribui 0,36 px sobre um piso de 1,49 que não é dela (Ruling 40) |
| **Trocar IoU por Dice** | Dice **é** `2·IoU/(1+IoU)` — transformação monótona, Spearman idêntico. Trocar de métrica de área não escapa do problema de área |
| **Mexer no extrator de polilinha** | 14 heurísticas locais + 6 globais, todas piores. E o teto medido mostra que um extrator perfeito **não recupera acurácia** (ganho em ζ com p=0,28) |
| **Prior de suavidade na polilinha** | numa resposta ao degrau a curva é genuinamente íngreme no transiente — penalizar curvatura briga com a verdade exatamente onde a cauda mora |
| **Continuar ajustando `_equiespacados`** | 24 variantes; toda que sobe cobertura sobe falso positivo a ~1,3 por p.p., com teto de 84,8 %. **O gate nunca foi o problema — ele recebia lixo** (Ruling 58) |
| **Guarda por descontinuidade da máscara** | Spearman +0,020 (p=0,57) em n=837, e o maior buraco do corpus é MAIOR que o da imagem que erra |
| **Ancorar o rótulo na marca de tick mais próxima (`snap`)** | piora monotonamente com a tolerância — com marcas espúrias, o snap desloca rótulos BONS para marcas falsas |
| **Alargar `K_BOUNDS` para incluir negativos** | põe K=0 dentro da caixa (modelo degenerado, mínimo local trivial) e destrói o sinal de borda do §4.1 |
| **Alargar `MARGIN_Y_W`/`MARGIN_X_H`** | ganho máximo de 1,3 p.p., e na maioria das configurações o `calibration_failed` **sobe**. É troca de motivo, não conserto |
| **Trocar AIC por BIC** | 29 das 48 ainda escolheriam a ordem errada. O problema não é a constante da penalidade, é o `n` inflado |
| **Adotar "sempre 2ª ordem"** | arquivado: exigia redefinir o critério 2.6 fora de `r["order"] == m["order"]` e declarar isso na monografia. O `n_eff` entrega quase o mesmo sem tocar no critério |
| **Ativar a poda de pares no `_equiespacados`** | troca 2.5 por 2.3 um-por-um, sem ganho líquido em contagem de critérios — e desde o Ruling 58 os dois estão aprovados |
