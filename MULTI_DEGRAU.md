# A frente de MULTI-DEGRAU — o que foi feito, o que foi medido, por que saiu

Removida do código em 10/09/2026 (§68). Este arquivo é o registro: os dados,
os lotes e os checkpoints continuam versionados, e o código sai. Quem quiser
retomar encontra aqui o que já está respondido e o que não está.

O código removido está preservado em `docs_frente/` como texto, e o histórico
em `git log --oneline main..HEAD` — em especial `ad90610` (cabeça de contagem),
`8c46ce8` (truncagem) e `427c2bf` (estrato).

---

## 1. O que o sistema fazia com multi-degrau

**A convenção de verdade**, herdada do `rg_multidegrau.py`: `params` descreve o
PRIMEIRO degrau, porque é o único que a pipeline pode recuperar de um prefixo.
`K = K_planta × U1`.

**A truncagem** (`identify_com_truncagem`, `identify/classical.py`) varria 13
cortes candidatos de 0,35 a 0,95 da série e adotava o prefixo quando ele
ajustava decisivamente melhor. Dois portões em série: `_PISO_SUSPEITA`
(o resíduo da série inteira precisa ser suspeito o bastante para valer a
varredura) e `_GANHO_MIN` (o prefixo precisa ganhar o bastante para valer o
corte).

**A cabeça de contagem** (`identify/extract.py`) era uma segunda saída da
U-Net, pendurada no gargalo do encoder: um logit de "há mais de um degrau
nesta figura?".

---

## 2. Os números que sobrevivem à remoção

### 2.1 Assertividade do bloco (época 17, 100 figuras de render real)

| nível | multi-degrau | 1 degrau, K>0 | 1 degrau, K<0 |
|---|---|---|---|
| ESTRITO | 35 % | 89 % | 77 % |
| PRÁTICO | **46 %** | 93 % | 83 % |
| TOLERANTE | 50 % | 94 % | 87 % |

Erro de `K`: mediana **11,6 %**, p90 **107 %** — contra 0,2 % nos blocos de um
degrau. O multi-degrau era, por uma ordem de grandeza, o bloco que dominava o
erro do sistema.

### 2.2 O mecanismo do erro, identificado

Quando a truncagem NÃO dispara, o ajuste descreve a **excursão inteira** em vez
da do primeiro degrau. Medido: das multi não truncadas, o `K` estimado fica
mais perto do `K` TOTAL (soma dos degraus) em 27 de 29. Não é ruído — é o
ajuste respondendo uma pergunta diferente da que a verdade faz.

A truncagem disparava em 55 % das multi. Como detector de "mais de um degrau":
precisão 75 %, revocação 38 %.

### 2.3 As duas constantes, com as varreduras

`_PISO_SUSPEITA`, varrido em 100 figuras com `_GANHO_MIN = 0,60`:

| piso | TP | FN | FP | TN | precisão | revocação | F1 |
|---|---|---|---|---|---|---|---|
| 0,030 | 19 | 31 | 6 | 44 | 76,0 % | 38,0 % | 0,507 |
| 0,020 | 27 | 23 | 7 | 43 | 79,4 % | 54,0 % | 0,643 |
| 0,010 | 28 | 22 | 7 | 43 | 80,0 % | 56,0 % | 0,659 |
| **0,007** | 29 | 21 | 7 | 43 | 80,6 % | 58,0 % | **0,674** |
| 0,000 | 29 | 21 | 7 | 43 | 80,6 % | 58,0 % | 0,674 |

Satura em 0,007. O valor 0,030 original tinha sido calibrado quando a polilinha
ainda pulava para a linha de entrada: naquele regime o resíduo inflado pelo
distrator abria o portão **por acidente**.

### 2.4 A cabeça de contagem, e a lição que ela deixou

- BatchNorm na entrada da cabeça **não é enfeite**: sem ela, 2 de 5 sementes
  colapsam para AUC 0,50 (ReLU morta). Com ela, as 5 dão 0,965 a 0,979.
- **O número sintético MENTE.** A cabeça deu AUC **0,98 no sintético** e
  **0,63 no corpus real**, em três variantes de corpus seguidas. Selecionar
  checkpoint pelo número sintético escolhe o pior no real.
- Melhor AUC real alcançada: **0,6838** (época 08 do retreino multi), em
  `reports/selecao_multi.json`.

Essa lição **não é sobre multi-degrau** e vale para qualquer cabeça futura.

### 2.5 O teto do caminho clássico

Antes de retreinar, foi medido se dava para ler a linha de entrada sem rede,
separando-a por cor (`docs_frente/sonda_teto_cor.py`): teto de **F1 0,647**.
Foi o que justificou o retreino em vez de resolver no extrator clássico.

---

## 3. A truncagem FICA — e a razão mudou

Ela foi removida junto com a frente e **revertida no mesmo dia**, por medição.

**Ela nunca foi mecanismo de multi-degrau.** A docstring original já dizia que
dispara em duas situações que o código não distingue: segundo degrau e **cauda
de extração ruim**. Ao removê-la, **6 testes de imagem REAL de um degrau**
quebraram — e não por precisão:

```
test_caso_real_recupera_zeta_e_wn            zeta ausente; order='fopdt'
test_caso_real_acerta_a_ordem                'fopdt' != 'second'
test_neg_super_sem_oclusao[K/wn/zeta/theta]  'fopdt' != 'second'
```

Sem o corte de prefixo o ajuste vê a cauda ruim inteira e **degenera de 2ª
ordem para FOPDT**, perdendo `zeta` e `wn` por completo.

**Erro de método que vale registrar.** A medição que autorizou a remoção olhou
só **taxa de entrega** — 90→83 em `|K| < 1`, 95→94 em `|K| ≥ 1`, "8 figuras em
200". Ela não capturou o dano real: as figuras continuam entregando, **com a
estrutura errada**. Métrica de entrega é cega a degeneração de modelo.

**O relatório continua nomeando as DUAS causas.** Reescrevê-lo para citar só a
cauda ruim seria afirmar causa não verificada: remover o suporte a
multi-degrau não faz figuras de dois degraus deixarem de existir. O que mudou
é que o sistema declara não modelá-las.

## 4. O que fica pendente, se a frente for retomada

- **4 testes vermelhos** em `test_truncagem_corpus.py` (samples 00257, 00328,
  00341, 00357), causados pela guarda de continuidade. A mensagem do teste
  pede que a banda da spec §5.2.1 seja **remapeada, não contornada**. Os
  testes VOLTARAM junto com a truncagem, e as 4 falhas com eles. A questão
  de spec continua aberta.
- **A cabeça de contagem não chegou a ser usada pela pipeline.** Ela existia
  no modelo e nenhum caminho a consultava.
- **O checkpoint promovido foi regravado sem ela** (11 chaves, 186.562
  parâmetros). Pesos de segmentação conferidos tensor a tensor: idênticos.
  A versão com cabeça está em `models/unet_stageA_com_cabeca_backup.pt`.

## 5. O que continua no repositório

- `data/train_multi` (1500) e `data/val_multi` (400) — corpus intacto
- `reports/amostras_aleatorias/lote100*`, `balanceado*` — lotes com figuras
  multi-degrau e a verdade delas
- `logs/train_multi.log`, `reports/selecao_multi.json` — o retreino inteiro
- `docs_frente/` — o código removido, como texto
