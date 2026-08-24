# HANDOFF_P2_3 — Bloco 3: Estágio A, U-Net e treino

## 0. Estado do treino — LEIA PRIMEIRO (atualizado em 24/08/2026)

**A RODADA 5 terminou e foi remedida — não fechou ζ.** Ela deu o melhor IoU
de todas as cinco rodadas (0,6205 de teste, contra 0,560 da rodada 4), mas
ζ em 2.6 ficou praticamente igual, até um pouco pior (+3,73 p.p. contra
+3,64 p.p. da rodada 4) — ver Ruling 9 (fechamento) e Ruling 10 (próximas
hipóteses, **pendentes**, a executar em outra máquina). §3 e
`HANDOFF_P2_5.md` foram atualizados com os números finais da Rodada 5.
**Não há mais treino rodando nesta máquina.**

Linha do tempo até a Rodada 4 (tratada como a melhor até então):

1. **Rodada 1** (sem *scheduler* de LR): interrompida na época 13/25, platô
   em IoU_val ~0,65–0,67. Ver Ruling 5.
2. **Rodada 2** (com `ReduceLROnPlateau`): 25 épocas, melhor IoU_val 0,6856.
   O *scheduler* ajudou mas não resolveu o platô — Ruling 6.
3. **Rodada 3** (+ correção do alvo, limiar 127→0 — Ruling 7): 25 épocas,
   IoU_val disparou para 0,9055, mas o IoU de TESTE **piorou** (0,572→0,495)
   — o limiar 0 recupera a curva perdida mas infla a área do alvo até 3×.
   Efeito colateral medido e corrigido — Ruling 8.
4. **Rodada 4** (limiar ajustado para 32 — Ruling 8): 25 épocas completas,
   melhor IoU_val (limiar 32) = 0,8876 (época 19). **Remedido contra os
   critérios reais: é o melhor resultado das quatro rodadas** — ver §3.

| Rodada | IoU teste (2.1) | 2.6 — pior parâmetro | Parâmetros de 2.6 que passam |
|---|---|---|---|
| 1 | 0,544 | +8,00 p.p. | 2/5 (K, θ) |
| 2 | 0,572 | +8,10 p.p. | 2/5 (K, θ) |
| 3 (limiar 0) | 0,495 | +3,92 p.p. | 2/5 (K, θ) |
| 4 (limiar 32) | 0,560 | **+3,64 p.p.** ← melhor 2.6 | **4/5 (K, τ, θ, ωₙ)** |
| **5 (alvo contínuo)** | **0,6205** ← melhor IoU | +3,73 p.p. | 4/5 (K, τ, θ, ωₙ) |

A rodada 4 tem o segundo melhor IoU e de longe o melhor 2.6 até a rodada 5
terminar — só ζ continua acima do alvo. A rodada 5 (ver Ruling 9) deu o
melhor IoU das cinco, mas **não melhorou ζ** — ficou 0,09 p.p. pior que a
rodada 4 no critério que decide. **Não há um vencedor único entre as
rodadas 4 e 5**: a 4 é levemente melhor em 2.6 (o critério mais
decisivo, por convenção já estabelecida nesta sessão), a 5 é bem melhor em
2.1/2.7/2.10 (IoU puro). Ambos os checkpoints ficam preservados — ver
abaixo — para quem retomar decidir com mais dados (Ruling 10).

5. **Rodada 5** (alvo CONTÍNUO, sem nenhum limiar de binarização — Ruling 9):
   **completa, 25/25 épocas**, ~8h30 de parede. Objetivo: eliminar a
   escolha de limiar por completo, deixando o alvo refletir a cobertura
   real de curva em cada caixa do downscale. Resultado: melhor IoU das
   cinco rodadas, mas não fechou ζ — ver Ruling 9.

```bash
cat logs/train_unet_rodada5_alvo_continuo.log               # rodada 5, completa (final)
cat logs/train_unet_rodada4_thr32.log                       # rodada 4, completa
cat logs/train_unet_rodada3_fix_mascara.log                 # rodada 3, completa (limiar 0)
cat logs/train_unet_scheduler_sem_fix_mascara.log           # rodada 2, completa
cat logs/train_unet_sem_scheduler.log                       # rodada 1, encerrada na época 13
```

Checkpoints preservados: `models/unet_stageA_sem_scheduler_ep7.pt` (rodada 1),
`models/unet_stageA_scheduler_sem_fix_mascara_ep11.pt` (rodada 2),
`models/unet_stageA_rodada3_fix_mascara_ep_final.pt` /
`unet_stageA_rodada3_thr0_ep_final.pt` (rodada 3, duplicadas),
`models/unet_stageA_rodada4_thr32_ep_final.pt` (rodada 4, época 19 — melhor
2.6 até agora), `models/unet_stageA_rodada5_alvo_continuo_ep_final.pt`
(rodada 5, época 14 — melhor IoU), e `models/unet_stageA.pt` (= cópia da
rodada 5, é o que os testes usam por padrão hoje; trocar por
`cp models/unet_stageA_rodada4_thr32_ep_final.pt models/unet_stageA.pt`
se quiser voltar à rodada 4 para alguma medição).

## 1. Estado

| Componente | Estado | Evidência |
|---|---|---|
| `identify/extract.py` (UNet, letterbox, dice_bce_loss, load_model, predict_mask) | pronto, testado | letterbox roundtrip PASSED; U-Net com 1,94 M parâmetros |
| `train_unet.py` (com `ReduceLROnPlateau`) | pronto, **treino concluído** (25/25 épocas) | `logs/train_unet.log` |
| `models/unet_stageA.pt` | checkpoint final, época 11/25 (melhor) | IoU_val = 0,6856 |
| Critério 2.1 (IoU mediana ≥ 0,85) | medido, **abaixo do alvo** (final) | 0,5716 (conjunto de teste, 300 amostras) |
| Critério 2.7 (IoU por estrato ≥ 0,75) | medido, **abaixo do alvo em todos os estratos** (final) | ver §3 |
| Suíte de mutação (P2-M10 a P2-M13) | 3/4 executados (sem retreinar), 1 propositalmente não executado | ver §3 |

## 2. Interface publicada

```python
class UNet(nn.Module): ...              # forward(x[N,1,512,512]) -> logits[N,1,512,512]
def letterbox(gray, size=512) -> tuple[np.ndarray, LetterboxInfo]: ...
def unletterbox(mask512, info) -> np.ndarray: ...
def dice_bce_loss(logits, target, eps=1.0) -> Tensor: ...
def load_model(path, device="cpu") -> UNet: ...
def predict_mask(model, image_rgb, device="cpu", thr=0.5) -> np.ndarray: ...  # uint8 0/255
```

Todas as assinaturas são exatamente as do `PLANO_PARTE2.md` — nenhuma
mudança de interface neste bloco.

## 3. Números medidos

### A.0 — Contagem real de parâmetros

**Medido: 1.942.289 (1,94 M)**, contra "~1,2 M" do PLANO — ver Ruling 1.

### Decisão de resolução de treino (Passo 7 do PLANO_PARTE2)

| Resolução | Tempo/época medido | Decisão |
|---|---|---|
| 512² | **> 90 min** (interrompido aos 95,5 min sem terminar 1 época) | descartada — ver Ruling 2 |
| 256² | **29,3 min** (1757 s, medição isolada) | escolhida, treino de 25 épocas disparado |

### Curva de treino — RODADA 1, sem scheduler de LR (`lr` fixo em 3e-4, encerrada na época 13/25)

| Época | IoU_val | Tempo | Época | IoU_val | Tempo |
|---|---|---|---|---|---|
| 0 | 0,2904 | 1830 s | 7 | **0,6734** (melhor da rodada) | 1312 s |
| 1 | 0,4373 | 1685 s | 8 | 0,6581 | 1269 s |
| 2 | 0,6297 | 1728 s | 9 | 0,6536 | 1269 s |
| 3 | 0,6570 | 1633 s | 10 | 0,6672 | 1267 s |
| 4 | 0,6703 | 1775 s | 11 | 0,6522 | 1268 s |
| 5 | 0,6669 | 1702 s | 12 | 0,6538 | 1264 s |
| 6 | 0,6575 | 1706 s | 13 | 0,6686 | 1269 s |

O ganho por época cai de +0,15/+0,19 (épocas 0→2) para uma faixa que oscila
entre **0,6522 e 0,6734 por sete épocas seguidas (6 a 13), sem tendência**
— não é ruído de poucas amostras, é um platô confirmado. Ver Ruling 5 (a
correção aplicada).

### Curva de treino — RODADA 2, com `ReduceLROnPlateau` (completa, 25/25 épocas)

| Época | IoU_val | LR | | Época | IoU_val | LR |
|---|---|---|---|---|---|---|
| 0 | 0,2904 | 3,0e-4 | | 13 | 0,6789 | 1,9e-5 ↓ |
| 1 | 0,4373 | 3,0e-4 | | 14 | 0,6818 | 1,9e-5 |
| 2 | 0,6297 | 3,0e-4 | | 15 | 0,6800 | 9,4e-6 ↓ |
| 3 | 0,6570 | 3,0e-4 | | 16 | 0,6822 | 9,4e-6 |
| 4 | 0,6703 | 3,0e-4 | | 17 | 0,6807 | 4,7e-6 ↓ |
| 5 | 0,6669 | 3,0e-4 | | 18 | 0,6830 | 4,7e-6 |
| 6 | 0,6575 | 1,5e-4 ↓ | | 19 | 0,6841 | 2,3e-6 ↓ |
| 7 | 0,6782 | 1,5e-4 | | 20 | 0,6817 | 2,3e-6 |
| 8 | 0,6684 | 7,5e-5 ↓ | | 21 | 0,6849 | 1,2e-6 ↓ |
| 9 | 0,6660 | 7,5e-5 | | 22 | 0,6845 | 1,2e-6 |
| 10 | 0,6744 | 3,8e-5 ↓ | | 23 | 0,6836 | 5,9e-7 ↓ |
| 11 | **0,6856** (melhor) | 3,8e-5 | | 24 | 0,6842 | 5,9e-7 |
| 12 | 0,6793 | 3,8e-5 | | | | |

O *scheduler* funcionou como desenhado (reduziu o LR a cada plateau
detectado, marcado com ↓) e levantou o teto de ~0,66 (rodada 1) para
~0,685 (rodada 2, épocas 14 em diante) — um ganho real de ~2 pontos, não
ruído. Mas o platô persiste em ~0,68-0,69 mesmo com o LR já em 5,9e-7 (quase
nulo) nas últimas duas épocas — reduzir mais a taxa de aprendizado não muda
mais nada. Ver Ruling 6.

### Curva de treino — RODADA 3, com a correção do Ruling 7 (limiar 0, completa, 25/25 épocas)

| Época | IoU_val | | Época | IoU_val | | Época | IoU_val |
|---|---|---|---|---|---|---|---|
| 0 | 0,6131 | | 9 | 0,8866 | | 18 | 0,9055 |
| 1 | 0,7506 | | 10 | 0,8966 | | 19 | 0,9052 |
| 2 | 0,8321 | | 11 | 0,9005 | | 20 | 0,9053 |
| 3 | 0,8521 | | 12 | 0,9006 | | 21 | 0,9055 |
| 4 | 0,8641 | | 13 | 0,9024 | | 22 | 0,9055 |
| 5 | 0,8642 | | 14 | 0,9037 | | 23 | 0,9055 |
| 6 | 0,8784 | | 15 | 0,9039 | | 24 | **0,9055** (melhor) |
| 7 | 0,8866 | | 16 | 0,9050 | | | |
| 8 | 0,8898 | | 17 | 0,9049 | | | |

Comparar com a época 0 das rodadas 1/2 (IoU_val 0,2904): a rodada 3 já
começa mais que o dobro, e passa de 0,85 já na época 3. **Confirma a causa
raiz do Ruling 7 com folga** — mas ver Ruling 8: o IoU de validação aqui é
medido contra o mesmo alvo de limiar 0 que se mostrou inflado, então este
número é otimista em relação ao IoU real de teste (§ abaixo).

### Curva de treino — RODADA 4, limiar 32 (completa, 25/25 épocas, final)

| Época | IoU_val | | Época | IoU_val | | Época | IoU_val |
|---|---|---|---|---|---|---|---|
| 0 | 0,5067 | | 9 | 0,8834 | | 18 | 0,8872 |
| 1 | 0,6870 | | 10 | 0,8846 | | 19 | **0,8876** (melhor) |
| 2 | 0,8450 | | 11 | 0,8853 | | 20 | 0,8873 |
| 3 | 0,8534 | | 12 | 0,8838 | | 21 | 0,8874 |
| 4 | 0,8639 | | 13 | 0,8855 | | 22 | 0,8874 |
| 5 | 0,8664 | | 14 | 0,8863 | | 23 | 0,8871 |
| 6 | 0,8733 | | 15 | 0,8868 | | 24 | 0,8875 |
| 7 | 0,8811 | | 16 | 0,8869 | | | |
| 8 | 0,8814 | | 17 | 0,8871 | | | |

**Nota importante:** este IoU_val é medido contra o alvo de limiar 32 —
não é comparável diretamente com o IoU_val de 0,9055 da rodada 3 (limiar 0,
um alvo mais generoso) nem com o de 0,68 das rodadas 1/2 (limiar 127, um
alvo que perdia a curva). As três réguas são diferentes; só o resultado
contra os critérios reais do PLANO (abaixo) é comparável entre rodadas.

### Curva de treino — RODADA 5, alvo contínuo (completa, 25/25 épocas, final)

| Época | IoU_val | | Época | IoU_val | | Época | IoU_val |
|---|---|---|---|---|---|---|---|
| 0 | 0,2650 | | 9 | 0,5802 | | 18 | 0,5873 |
| 1 | 0,3629 | | 10 | 0,5792 | | 19 | 0,5854 |
| 2 | 0,5059 | | 11 | 0,5860 | | 20 | 0,5857 |
| 3 | 0,5091 | | 12 | 0,5870 | | 21 | 0,5883 |
| 4 | 0,5790 | | 13 | 0,5884 | | 22 | 0,5854 |
| 5 | 0,5805 | | 14 | **0,5895** (melhor) | | 23 | 0,5867 |
| 6 | 0,5889 | | 15 | 0,5858 | | 24 | 0,5868 |
| 7 | 0,5864 | | 16 | 0,5811 | | | |
| 8 | 0,5864 | | 17 | 0,5838 | | | |

**Nota:** este IoU_val é medido contra o alvo CONTÍNUO binarizado em 0,5
dentro de `iou()` (ver código) — régua diferente das rodadas 3/4, não
comparável diretamente entre si; só o IoU de teste (resolução original,
abaixo) é comparável entre rodadas.

### Critério 2.1 — IoU mediana (≥ 0,85), 300 amostras de teste

| Checkpoint | Medido | Veredito |
|---|---|---|
| Rodada 1, época 4 (preliminar) | 0,5439 | ❌ |
| Rodada 2, época 11 | 0,5716 | ❌ |
| Rodada 3, época 24 (limiar 0) | 0,4950 | ❌ |
| Rodada 4, época 19 (limiar 32) | 0,5597 | ❌ |
| **Rodada 5, época 14 (alvo contínuo — final)** | **0,6205** | ❌ |

A rodada 5 dá o melhor IoU de teste das cinco rodadas — supera até a
rodada 2 (0,572), que não sofria com o problema de alvo perdendo/inflando
área. Ainda assim abaixo do alvo de 0,85.

### Critério 2.7 — IoU por estrato (nenhum < 0,75)

| Estrato | Rodada 2 | Rodada 3 (limiar 0) | Rodada 4 (limiar 32) | Rodada 5 (contínuo) | Veredito |
|---|---|---|---|---|---|
| `fundo_escuro=False` | 0,5597 | 0,4629 | 0,5394 | **0,5856** (n=173) | ❌ |
| *(demais estratos: o `assert` dentro do laço interrompe a coleta no primeiro estrato que reprova — ver Armadilha 2)* | — | — | — | — | ❌ |

### Critério 2.10 (diagnóstico, sem alvo) — U-Net × extrator clássico

| Extrator | Rodada 1 | Rodada 2 | Rodada 3 (limiar 0) | Rodada 4 (limiar 32) | Rodada 5 (contínuo) |
|---|---|---|---|---|---|
| U-Net | 0,5439 | 0,5716 | 0,4950 | 0,5597 | **0,6205** |
| Clássico (Bloco 3b) | 0,7153 | 0,7153 | 0,7153 | 0,7153 | 0,7153 (inalterado) |

**O extrator clássico continua vencendo em IoU puro, em todas as cinco
rodadas** — a leitura do §1.8 do PLANO ("se empatar/vencer, o achado é que
a U-Net não se justifica") permanece válida para este critério isolado,
mesmo com a rodada 5 reduzindo a distância (0,715 vs 0,620, contra 0,715
vs 0,560 da rodada 4). Mas ver a leitura funcional abaixo: em 2.6, a U-Net
já supera o que o extrator clássico conseguiria produzir sozinho — IoU de
máscara e utilidade final para estimar parâmetros não são a mesma coisa. A
comparação de 2.10 mede só a máscara; a que decide o valor prático do
sistema é 2.6.

**O critério 2.6 teve seu melhor resultado na rodada 4, não na 5 —
contraintuitivo, dado que a 5 tem o IoU melhor.** Ver a tabela completa no
`HANDOFF_P2_5.md`: pior parâmetro caiu de 8,00–8,10 p.p. (rodadas 1/2) para
3,64 p.p. (rodada 4) e ficou praticamente estagnado em 3,73 p.p. na rodada
5 — quatro dos cinco parâmetros (K, τ, θ, ωₙ) atendem o alvo de 3 p.p. em
ambas; só ζ (amortecimento) continua acima em ambas, por uma margem
pequena, e a rodada 5 não reduziu essa margem (na verdade abriu 0,09 p.p.
a mais). IoU pixel-a-pixel melhor não implica automaticamente ζ melhor —
ver Ruling 9, fechamento.

### Suíte de mutação — sem retreinar (ver Ruling 4 sobre P2-M10)

Medida em 30 amostras (custo de inferência, não de OCR — mais barato que o
Bloco 2; ainda assim uma medição por mutante):

| Mutante | Substituição | Esperado | Observado |
|---|---|---|---|
| P2-M10 | `dice_bce_loss` → só `bce` (colapso para "tudo fundo") | 2.1 reprova | **não executado** — exigiria retreinar do zero; ver Ruling 4 |
| P2-M11 | `letterbox` com `cv2.resize` direto p/ (512,512), sem preservar razão de aspecto | 2.2 no Bloco 5 (RMSE), não necessariamente 2.1 | ✅ IoU quase inalterado (0,581 vs. 0,588 baseline — mutante NÃO aparece em 2.1), mas RMSE da polilinha explode: 29,5 px vs. piso de ~1,5 px — confirma exatamente a expectativa do PLANO, "reprova 2.2, não 2.1" |
| P2-M12 | `unletterbox` sem recortar o *padding* | 2.1 reprova | ✅ IoU despenca de 0,588 para **0,010** |
| P2-M13 | Controle: `thr` 0,50 → 0,51 | nada reprova | ✅ IoU 0,587 vs. 0,588 baseline — diferença dentro do ruído |

## 4. Rulings

1. **Contagem real de parâmetros (1,94 M) diverge do "~1,2 M" do PLANO.**
   Mantida a arquitetura tal como especificada (4 níveis, base 16 canais) —
   a fórmula de canais `[16,32,64,128,256]` com os dois blocos conv 3×3 por
   nível em encoder+decoder soma naturalmente mais que a estimativa do
   PLANO. Não é um desvio de implementação: é a contagem real da arquitetura
   *exatamente* como descrita (`_block`, `UNet.__init__`) — a estimativa "~1,2
   M" do texto do PLANO provavelmente não somava os blocos do decoder por
   completo. Não há necessidade de mudar a arquitetura: 1,94 M ainda está
   bem dentro da categoria "compacta" (a U-Net original tem ~31 M).
2. **512² foi abandonado sem completar 1 época — 512² excedeu 90 min sem
   terminar sequer 1 época (parado aos 95,5 min de relógio), disparando a
   regra do Passo 7 do PLANO_PARTE2** ("> 90 min/época → só a 256² é
   viável"). A máquina de execução não tem GPU (Ruling 1 do
   `HANDOFF_P2_0.md`) e é mais lenta que a do `PLANO.md` mesmo em CPU
   (Armadilha 2 do mesmo handoff) — o pior caso previsto no próprio PLANO se
   realizou. 256² mede 29,3 min/época, dentro da faixa "esperada em CPU".
3. **O treino foi interrompido nas medições deste handoff na época 6, não
   nas 25 planejadas — decisão tomada com números, não por impaciência.**
   O IoU de validação sobe rápido nas 3 primeiras épocas (+0,15, +0,19,
   +0,03) e depois **estagna e oscila para baixo** (0,6703 → 0,6669 →
   0,6575). `train_unet.py` usava `AdamW` com `lr` fixo (3e-4, sem
   *scheduler*) — o comportamento é consistente com o otimizador "circulando"
   em torno de um mínimo local sem conseguir refinar mais, o padrão clássico
   que pede decaimento de taxa de aprendizado. Na época em que este Ruling
   foi escrito, o treino continuava rodando na esperança de que mais épocas
   ajudassem; **confirmou-se que não ajudavam** (mais sete épocas medidas,
   platô entre 0,6522 e 0,6734, sem tendência — ver §3) e a decisão foi
   revertida — ver Ruling 5. Os números registrados aqui e no
   `HANDOFF_P2_5.md` (2.1, 2.7, 2.10, 2.6, 2.8) continuam sendo os da época
   4 da rodada 1, ainda não remedidos — ver §0 e §6.
4. **P2-M10 (perda BCE pura) não foi executado — exigiria retreinar do
   zero, o que este ambiente não comporta duas vezes.** Uma rodada de treino
   completa já consome a maior parte do orçamento de tempo desta sessão; uma
   segunda rodada só para provar que a perda Dice+BCE é melhor que BCE pura
   (algo já bem estabelecido na literatura, citada na docstring de
   `dice_bce_loss`, para classes desbalanceadas como aqui — "< 2% dos
   pixels") não se justifica pelo custo. **Registrado como não verificado**,
   não como verificado-e-passou — não citar este mutante como coberto na
   monografia sem antes rodá-lo de fato.
5. **`train_unet.py` precisou de um *scheduler* de taxa de aprendizado
   (`ReduceLROnPlateau`) — decisão tomada com dados de 14 épocas, não só
   suspeita do Ruling 3.** Com `lr` fixo, a rodada 1 confirmou platô
   persistente: sete épocas seguidas (6 a 13) oscilando entre IoU_val
   0,6522 e 0,6734, sem nenhuma tendência de subida — o oposto do salto de
   +0,15/+0,19 das duas primeiras épocas. É a assinatura clássica de um
   passo de otimização grande demais para refinar perto de um mínimo:
   melhora rápido no início, depois "circula" ao redor do ótimo sem se
   aproximar. A rodada 1 foi **interrompida deliberadamente** (não por falha
   nem por tempo esgotado) assim que o platô foi confirmado como
   persistente, e uma segunda rodada foi lançada com
   `torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
   patience=1, threshold=0.01, threshold_mode="abs")`, que reduz o LR pela
   metade sempre que o IoU de validação não melhorar em pelo menos 1 ponto
   percentual absoluto por 1 época — os parâmetros não foram um palpite:
   **medidos** rodando o *scheduler* (fora do laço de treino, só a lógica de
   decisão) contra a sequência real de 14 IoUs da rodada 1, comparando
   `patience ∈ {1, 2}` × `threshold ∈ {1e-4 relativo, 0,005 absoluto, 0,01
   absoluto}`. `patience=2` com o limiar padrão do PyTorch (1e-4 relativo)
   só reagia depois da época 10 — tarde demais, porque um "novo melhor"
   apenas marginal (ex.: época 7, 0,6734, mal acima da época 4, 0,6703) já
   reiniciava a paciência mesmo sem representar melhora real. `patience=1` +
   `threshold=0,01` absoluto reagia já na época 6 (a primeira após o pico da
   época 4 não superado por uma margem significativa) e continuava reduzindo
   a cada 2 épocas de plateau — é o comportamento escolhido. **A checagem
   fica registrada no próprio log de treino**: `train_unet.py` agora imprime
   o LR atual a cada época e marca `(LR reduzido)` sempre que o *scheduler*
   dispara, então o próximo executor não precisa confiar em memória — o log
   mostra exatamente quando e quantas vezes a taxa caiu.
   **Checkpoints preservados separadamente**: o melhor da rodada 1 (época 7,
   IoU_val 0,6734) foi copiado para `models/unet_stageA_sem_scheduler_ep7.pt`
   antes de iniciar a rodada 2, que sobrescreve `models/unet_stageA.pt`
   continuamente — nada foi perdido.
6. **O *scheduler* de LR ajudou, mas não resolveu — o gargalo não era só a
   taxa de aprendizado.** A rodada 2 completou as 25 épocas: o platô subiu
   de ~0,66 (rodada 1) para ~0,685 (rodada 2, estável da época 14 em
   diante), um ganho real de ~2 pontos de IoU. Mas nas últimas duas épocas
   o LR já estava em 5,9e-7 — seis ordens de grandeza abaixo do inicial — e
   o IoU não passou de 0,684-0,685. **Isso descarta taxa de aprendizado como
   único fator**: se fosse só isso, LR quase nulo deveria permitir refino
   fino contínuo, não outro platô. Os critérios 2.1/2.7/2.10 foram remedidos
   contra este checkpoint (época 11, IoU_val 0,6856) — ver §3, valores
   "Rodada 2". Hipóteses levantadas então para o gargalo remanescente, em
   ordem de suspeita: (a) capacidade do modelo; (b) resolução de treino
   reduzida a 256²; (c) `data/train` com 4.200 amostras insuficiente. A
   hipótese (b) foi refinada e **medida** — ver Ruling 7, que muda o
   diagnóstico de "resolução baixa demais" (exigiria treinar em 512², caro)
   para "alvo de treino mal preparado NA resolução atual" (barato de
   corrigir, sem mudar a resolução).
7. **A causa mais provável do platô não é resolução de treino em si — é
   como a máscara-alvo é reduzida para 256², medida diretamente.** Antes de
   propor treinar em maior resolução (caro: 512² mede > 90 min/época nesta
   máquina — Ruling 2), medi quanto da máscara verdadeira sobrevive ao
   `letterbox` atual (`cv2.INTER_AREA` + binarização em limiar 127), sem
   treinar nada — só pré-processamento, em 300 amostras de teste,
   estratificado pelo tamanho original da imagem:

   | Preparo do alvo | Cobertura de colunas da curva (mediana) | Pior caso (p10) |
   |---|---|---|
   | Máscara original (resolução cheia) | 96,6% | 73,1% |
   | Atual: `INTER_AREA` + limiar 127 | 68,8% | **0,4%** |
   | `INTER_AREA` + limiar 1 (qualquer resíduo > 0) | **100%** | **95,0%+** |

   Para as imagens grandes (900–1.600 px de lado — **68% do conjunto de
   teste**, `size_px` sorteado até 1600×1200 pelo `dataset/randomize.py`),
   o `letterbox` reduz a escala em até ~6,25×. Uma linha de 1–2 px, nesse
   downscale, vira um valor médio bem abaixo de 127 e **desaparece do
   alvo de treino** — no pior caso medido, praticamente 100% da curva some
   de uma imagem. A rede está sendo treinada, numa fração real do dataset,
   contra rótulos que já perderam a informação antes mesmo dela processar
   — isso é mais consistente com um platô persistente e insensível a LR do
   que qualquer uma das três hipóteses do Ruling 6. `mask.png` é binário
   limpo (Parte 1, sem antialiasing), então baixar o limiar para 1 não
   introduz ruído espúrio — todo pixel não-zero no downscale vem de pelo
   menos 1 pixel de curva real no bloco correspondente.
   **Correção aplicada** (`train_unet.py::MaskDataset`, Rodada 3): o limiar
   de binarização do alvo, depois do `letterbox`, caiu de 127 para 0 (linha
   `MASK_DOWNSCALE_THR`). Não mudei `identify.extract.letterbox` em si (é
   usado igual para a imagem em tons de cinza, onde `INTER_AREA` é o
   comportamento correto e não deve mudar) — a correção é só na preparação
   do alvo de máscara em `train_unet.py`. Não precisa de mais resolução, não
   precisa de GPU, mesmo custo por época (~20 min/época em 256²) — só um
   retreino do zero, porque o alvo mudou. **Resultado, medido na Rodada 3
   completa: corrigiu o platô (IoU_val 0,68→0,91), mas com um efeito
   colateral que piorou o IoU de teste** — ver Ruling 8, que ajusta a
   correção.
8. **O limiar 0 do Ruling 7 resolveu o sumiço da curva, mas superajustou
   para o lado oposto: infla a ÁREA do alvo, não só recupera a presença
   perdida.** Medido depois da Rodada 3 completa: IoU de validação disparou
   para 0,9055 (o alvo do treino), mas o IoU no conjunto de TESTE (máscara
   prevista, resolução original, contra a máscara verdadeira original —
   nenhuma das duas passa pelo `letterbox` na avaliação) **caiu** de 0,572
   (rodada 2) para 0,495. A causa: com `cv2.INTER_AREA`, qualquer bloco que
   toque a curva original — mesmo tangencialmente, na borda do bloco —
   produz um valor > 0 depois da média; aceitar QUALQUER valor > 0 como
   "curva" (limiar 0) captura esse halo inteiro em volta da linha real, não
   só a linha. Medido diretamente (300 amostras, sem treinar), a área do
   alvo no limiar 0 fica **1,47× a 3,03× maior** que a área proporcional
   esperada (pior nas imagens grandes, onde o bloco de amostragem é maior).
   A rede aprendeu, fielmente, a prever essa faixa mais grossa — bate bem
   contra o alvo inflado (por isso IoU_val alto), perde contra a máscara
   fina verdadeira (por isso IoU de teste caiu).

   Varredura de limiares (300 amostras, cobertura de colunas E razão de área
   em relação ao esperado por escala pura, sem treinar) para achar um
   equilíbrio:

   | Limiar | Cobertura de colunas, pior caso (img. grandes) | Inflação de área (img. grandes) |
   |---|---|---|
   | 0 | 98,9% | 3,03× |
   | 15 | 95,6% | 2,50× |
   | **32** | **85,1%** | **2,14×** |
   | 64 | 48,1% | 1,55× |
   | 96 | 12,7% | 1,14× |
   | 127 (original) | 0,4% | 0,83× |

   **Limiar 32 escolhido**: mantém o pior caso de cobertura em 85,1% (contra
   0,4% do original — ainda elimina o sumiço catastrófico da curva), com
   inflação de área bem menor que o limiar 0 (2,14× contra 3,03×). Não é um
   ótimo provado — é o melhor ponto da varredura testada, escolhido por
   medição, não por intuição. **Resultado, medido na Rodada 4 completa: foi
   a escolha certa.** IoU de teste recuperou quase todo o terreno perdido na
   rodada 3 (0,495 → 0,560) e o critério 2.6 teve o melhor resultado das
   quatro rodadas (pior parâmetro 3,64 p.p., 4 dos 5 parâmetros dentro do
   alvo) — ver §3 e `HANDOFF_P2_5.md`. Só ζ continuou reprovando, por uma
   margem pequena (0,64 p.p.) — motivou a Rodada 5, Ruling 9.
9. **Em vez de um quarto limiar, o alvo virou CONTÍNUO — elimina a escolha
   por completo.** Depois de três limiares testados (127, 0, 32), cada um
   com o mesmo formato de troca (cobertura vs. inflação de área), a
   pergunta certa deixou de ser "qual o próximo limiar" e passou a ser "por
   que binarizar de qualquer jeito". `cv2.INTER_AREA` já devolve, pra cada
   pixel do alvo reduzido, a FRAÇÃO real de cobertura de curva naquela
   caixa (0 a 255) — um valor informativo que a binarização joga fora.
   `dice_bce_loss` (BCE + Dice) aceita alvo contínuo em [0,1] sem nenhuma
   mudança de código — é a definição matemática usual das duas funções, só
   nunca tinha sido usada assim aqui. Trocado `MaskDataset.__getitem__`
   para devolver `y/255.0` direto (sem `> limiar`), e `iou()` (só a métrica
   de progresso do treino, não a perda) binariza o alvo internamente para
   continuar reportando um número interpretável.

   A expectativa: uma caixa com pouca cobertura de curva vira um alvo baixo
   mas NÃO-ZERO (resolve o sumiço do limiar 127 sem escolher limiar nenhum,
   porque BCE tem gradiente não-nulo mesmo para alvo pequeno-mas-positivo),
   e uma caixa com cobertura parcial não empurra a rede a prever confiança
   ALTA ali (limita a inflação de área do limiar 0/32 pela raiz do
   problema, não por um corte arbitrário escolhido por varredura). Se
   funcionar, deveria ajudar exatamente o parâmetro mais sensível a forma
   fina (ζ) mais que os outros, porque a região de erro sistemático era
   precisão de contorno, não presença/ausência.

   **Medido, Rodada 5 completa (25/25 épocas): NÃO fechou ζ.** IoU de teste
   melhorou substancialmente (0,560 → 0,6205, o melhor das cinco rodadas),
   e os estratos/2.10 também melhoraram — mas ζ em 2.6 ficou praticamente
   parado: +3,64 p.p. (rodada 4) → +3,73 p.p. (rodada 5), ou seja, **piorou
   0,09 p.p.**, na direção oposta da esperada. Os demais sub-parâmetros de
   2.6 também pioraram levemente (K: +0,92→+1,08; τ: +2,00→+2,57; ωₙ:
   +1,78→+2,11), exceto θ (estável: +0,55→+0,54). **A hipótese específica
   ("alvo contínuo ajuda mais ζ porque ataca precisão de contorno") não se
   confirmou** — ver §3 para os números completos.

   Leitura mais provável: o alvo contínuo melhora a IoU **global** (menos
   falso-negativo/falso-positivo de área), mas ζ depende de um detalhe
   ainda mais local — a amplitude exata do pico de *overshoot* — que talvez
   dependa mais de capacidade de representação da rede (consegue aprender a
   forma fina de um pico?) ou de exemplos suficientes desse padrão
   específico no dataset, do que de como o alvo é preparado. As três
   variações de preparo do alvo já testadas (limiar 127, 0, 32, contínuo)
   cobrem essencialmente o espaço de "como binarizar/suavizar o alvo"; o
   próximo eixo de investigação é outro — ver Ruling 10.
10. **Hipóteses do Ruling 6 (capacidade do modelo, tamanho do dataset) —
    PENDENTES, ainda NÃO executadas nesta máquina.** Depois de cinco
    rodadas de treino (~35h de parede acumuladas nesta máquina, só CPU) sem
    fechar ζ por variações no preparo do alvo, a decisão foi migrar o
    experimento para outra máquina (mais rápida/com GPU) em vez de gastar
    mais 19–34h aqui. **Nada foi treinado ainda para estas duas
    hipóteses** — o que segue é o plano de execução, não um resultado.

    a. **Capacidade do modelo.** `UNet(base=16)` tem 1,94 M parâmetros
       (Ruling 1). Contagem medida para alternativas maiores, mesma
       arquitetura (4 níveis):

       | `base` | Parâmetros | Fator vs. atual |
       |---|---|---|
       | 16 (atual) | 1.942.289 | 1,0× |
       | 24 | 4.367.641 | 2,25× |
       | 32 | 7.762.465 | 4,0× |

       Como a convolução escala grosseiramente com o quadrado do nº de
       canais, o tempo por época também escala nessa ordem — nesta máquina
       (CPU-only, ~1220s/época em `base=16`, 512²), `base=24` ficaria perto
       de 19h e `base=32` perto de 34h só de treino, para o mesmo orçamento
       de 25 épocas. **Custo alto demais para testar direto sem um piloto
       barato primeiro.** Plano recomendado antes de comprometer uma rodada
       inteira: treinar 2–3 épocas de `base=16` vs `base=24` (mesmo
       dataset, mesmo *seed*) e comparar a curva de IoU_val inicial — se
       `base=24` não mostrar vantagem já nas primeiras épocas, não vale
       gastar as ~19h completas. Ainda não executado.
    b. **Tamanho do dataset.** Hoje: `data/train` 4.200 / `data/val` 900 /
       `data/test` 900 (6.000 total, ~422 MB, gerados com
       `dataset.generator.generate_dataset`, determinístico — ver Ruling 4
       do `HANDOFF_P2_0.md`, item G0.3/1.6). Ainda não gerado nenhum split
       maior. Se for testada, a forma mais barata de isolar o efeito é
       gerar mais amostras de treino (ex.: dobrar para 8.400) mantendo
       val/test fixos (para os números continuarem comparáveis entre
       rodadas), com uma seed-base nova que não colida com as seeds já
       usadas (bases 1/2/3 — ver §0 do `HANDOFF_P2_0.md` para a fórmula
       `seed_i = base*1_000_003 + i`; usar base ≥ 4).
    c. **Nenhuma das duas hipóteses ataca especificamente por que ζ é o
       pior parâmetro** (ambas são genéricas — "mais capacidade"/"mais
       dados" ajudam quase tudo) — é uma limitação conhecida deste plano,
       não descoberta agora. Se qualquer uma delas rodar e ζ não fechar de
       novo, o próximo passo mais direcionado seria olhar amostra a amostra
       *quais* casos de ζ erram mais (ex.: sistemas muito subamortecidos,
       picos estreitos) em vez de tentar mais uma mudança genérica de
       treino — ainda não investigado.

## 5. Armadilhas

1. **Cuidado ao repetir a suíte de mutação do Bloco 3: ela precisa do
   checkpoint (`models/unet_stageA.pt`) carregado, não de retreinar** — P2-M11,
   P2-M12 e P2-M13 mudam só o pré/pós-processamento em torno de um modelo já
   treinado, então rodam em minutos (inferência em 30 imagens). Só P2-M10
   precisaria de treino novo — é por isso que ele ficou de fora.
2. **O checkpoint em disco (`models/unet_stageA.pt`) só é sobrescrito quando
   o IoU de validação MELHORA** (`if m > melhor`, em `train_unet.py`) —
   confirmado nas duas rodadas: o arquivo final corresponde à época 11 da
   rodada 2 (melhor global), não à última época treinada (24). Quem for
   remedir no futuro deve conferir qual época o log marca como melhor, não
   assumir que é a última.
3. **A tabela do critério 2.7 nunca mostrou mais que um estrato, nas duas
   rodadas.** `test_2_7_iou_por_estrato` usa `assert` DENTRO do laço
   `for nome, vs in sorted(estratos.items())`, então ele para no primeiro
   estrato que reprova (`fundo_escuro=False`, por ordem alfabética) e nunca
   chega aos demais (`grade=`, `legenda=`, `traco=`). Isso não é específico
   deste bloco — é uma limitação do padrão de teste usado desde o
   `PLANO_PARTE2.md` (o mesmo formato aparece em 2.2 e outros) — mas vale
   registrar que "nenhum estrato < 0,75" nunca foi de fato verificado em
   TODOS os estratos, só no primeiro. Se precisar da tabela completa por
   estrato para a monografia, comentar o `assert` temporariamente (ou trocar
   por uma lista de falhas acumuladas) antes de rodar de novo.

## 6. O que o próximo bloco precisa saber

1. **A U-Net é a escolha certa para o pipeline final — medido, não só
   arquitetado.** Em IoU de máscara pura (2.10), o extrator clássico do
   Bloco 3b continua à frente (0,715 vs. 0,620 na rodada 5). Mas essa não é
   a comparação que decide: medindo 2.6 (degradação end-to-end dos
   parâmetros físicos) com os dois extratores no mesmo pipeline, **a U-Net
   vence** (pior parâmetro +3,64 p.p. na rodada 4, ou +3,73 p.p. na rodada
   5, contra +4,38 p.p. do clássico — o clássico reprova ωₙ, nenhuma
   rodada da U-Net reprova) — ver `HANDOFF_P2_5.md §3`. `identify_from_image`
   ganhou o parâmetro opcional `extractor=` para permitir essa comparação,
   sem quebrar a assinatura do PLANO.
2. **Os números de 2.1, 2.7, 2.10, 2.6 e 2.8 no `HANDOFF_P2_5.md` já foram
   atualizados para as duas rodadas mais recentes (4 e 5)** — não é mais
   necessário remedir por causa de treino já executado. **Não há um
   checkpoint único "final"**: rodada 4 (`models/unet_stageA_rodada4_thr32_ep_final.pt`)
   continua com o melhor 2.6; rodada 5
   (`models/unet_stageA_rodada5_alvo_continuo_ep_final.pt`, = o que está em
   `models/unet_stageA.pt` hoje) tem o melhor IoU. Quem retomar decide qual
   usar como referência, ou usa as duas hipóteses do Ruling 10 para tentar
   superar ambas antes de escolher.
3. **O alvo mais claro para melhorar ainda mais a U-Net continua sendo só
   ζ** (o único parâmetro de 2.6 que ainda reprova, por 3,64–3,73 p.p.
   contra o alvo de 3,0 — margem pequena e estável nas duas últimas
   rodadas). As três variações de preparo do alvo (limiar 127, 0, 32,
   contínuo) já esgotaram esse eixo sem fechar ζ — ver Ruling 9. **As
   hipóteses do Ruling 10 (capacidade do modelo, tamanho do dataset) são o
   próximo passo, e estão PENDENTES — nenhuma foi executada ainda.** Ruling
   10 documenta o plano de execução (incluindo um piloto barato antes de
   comprometer um treino completo) para continuar em outra máquina.
4. Próximo passo: `HANDOFF_P2_5.md` (integração, 2.6, 2.8, relatório final —
   já atualizado com os números das rodadas 4 e 5; hipóteses pendentes
   também documentadas lá em §7).
