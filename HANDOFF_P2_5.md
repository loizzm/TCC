# HANDOFF_P2_5 — Bloco 5: Integração, degradação e relatório

## 0. Leia primeiro — mesma ressalva do `HANDOFF_P2_3.md`

**Atualização, 24/08/2026 — as cinco rodadas de treino terminaram; nenhum
critério numérico fecha, mas as duas últimas rodadas ficam bem perto de
2.6.** Histórico completo em `HANDOFF_P2_3.md §0`:

| Rodada | O que mudou | IoU teste (2.1) | 2.6 — pior parâmetro |
|---|---|---|---|
| 1 | LR fixo | 0,544 | +8,00 p.p. |
| 2 | + *scheduler* de LR | 0,572 | +8,10 p.p. |
| 3 | + alvo de treino, limiar 0 | 0,495 (piorou) | +3,92 p.p. |
| 4 | alvo de treino, limiar 32 | 0,560 | **+3,64 p.p.** ← melhor 2.6 |
| **5** | **alvo contínuo (sem limiar)** | **0,6205** ← melhor IoU | +3,73 p.p. |

**Não há um vencedor único entre as rodadas 4 e 5** — a 4 é levemente
melhor no critério mais decisivo (2.6), a 5 é bem melhor em IoU puro
(2.1/2.7/2.10). Os números abaixo trazem as duas rodadas lado a lado
quando fizer diferença; a tabela consolidada de §6 usa a rodada 4 como
referência para 2.6 (por ser o critério que decide, convenção já
estabelecida nesta sessão) e a rodada 5 para 2.1/2.7/2.10. **As hipóteses
de capacidade do modelo e tamanho do dataset (`HANDOFF_P2_3.md` Ruling 10)
ficam PENDENTES, a executar em outra máquina — ver §7.**

## 1. Estado

| Componente | Estado | Evidência |
|---|---|---|
| `identify/pipeline.py::identify_from_image` | pronto, testado ponta-a-ponta | roda sem exceção em 300/300 amostras (2.11) |
| Critério 2.6 (degradação end-to-end) | medido, **acima do alvo, mas por pouco** | pior parâmetro +3,64 p.p. (rodada 4) / +3,73 p.p. (rodada 5) — alvo ≤ 3 p.p.; 4 dos 5 parâmetros já passam nas duas |
| Critério 2.8 (latência) | medido, **acima do alvo** | rodada 4: mediana 2.104 ms, p95 5.659 ms; rodada 5: mediana 3.353 ms, p95 9.981 ms (alvo < 500 ms) — ver Armadilha 4 sobre a diferença entre as duas medições |
| `reports/part2_strata.md` | gerado automaticamente, mas **parcial por sessão** | ver Armadilha 1 |
| `reports/part1_metrics.md` | **intacto** durante toda a Parte 2 | hash idêntico verificado repetidas vezes |

## 2. Interface publicada

```python
def identify_from_image(image_rgb: np.ndarray, model, device: str = "cpu") -> dict:
    """{"order", "params", "ok", "reason", "latency_ms", "n_points"}. Nunca levanta."""
```

Assinatura exatamente a do `PLANO_PARTE2.md`. É a única porta de entrada que
a Parte 3 deve importar de `identify/`.

**Nota sobre a Decisão E (§1.7 do PLANO — saída em dois níveis, "dimensionless"
sempre preenchido).** `identify_from_image`, como implementada aqui (idêntica
ao Passo 1 do PLANO_PARTE2), devolve `params` no formato `FitResult.params`
de `identify.classical` (`K`, `tau`, `theta`, `wn`, `zeta` — já a saída
"física" da Parte 1) e **não** empacota separadamente um bloco
`dimensionless`/`physical` como o JSON de exemplo do `PLANO.md §1.7` mostra.
**Isto é uma lacuna real, não fechada nesta sessão** — ver Ruling 3.

## 3. Números medidos (Rodadas 4 e 5 — as duas mais recentes, sem vencedor único)

### Critério 2.6 — degradação end-to-end vs. oráculo (ΔMAPE ≤ 3 p.p.)

300 amostras avaliadas. Rodada 4: **168 comparáveis**; rodada 5: também
**168 comparáveis** (mesma ordem escolhida pelo oráculo e pelo pipeline
real, ambos convergem — poder estatístico suficiente, ≥ 100, nas duas).

| Parâmetro | ΔMAPE (Rodada 2) | ΔMAPE (Rodada 4) | ΔMAPE (Rodada 5) | Alvo | Veredito (4 / 5) |
|---|---|---|---|---|---|
| K | +1,86 p.p. | **+0,92 p.p.** | +1,08 p.p. | ≤ 3 p.p. | ✅ / ✅ |
| τ (tau) | +4,74 p.p. | **+2,00 p.p.** | +2,57 p.p. | ≤ 3 p.p. | ✅ / ✅ |
| θ (theta, NMAE/T_dom) | +0,78 p.p. | +0,55 p.p. | **+0,54 p.p.** | ≤ 3 p.p. | ✅ / ✅ |
| ωₙ (wn) | +3,60 p.p. | **+1,78 p.p.** | +2,11 p.p. | ≤ 3 p.p. | ✅ / ✅ |
| ζ (zeta) | +8,10 p.p. | **+3,64 p.p.** | +3,73 p.p. | ≤ 3 p.p. | ❌ / ❌ |

(Rodada 5 — MAPE bruto: K oráculo 0,11% / real 1,20%; τ 0,25% / 2,82%; θ
0,08% / 0,62%; ωₙ 0,90% / 3,01%; ζ 1,24% / 4,97%.)

**Pior parâmetro nas duas rodadas: ζ — reprova em ambas, por margem
pequena e estável** (0,64 p.p. acima do alvo na rodada 4, 0,73 p.p. na
rodada 5). **Quatro dos cinco parâmetros já atendem o alvo de 3 p.p. nas
duas rodadas.** Contraintuitivamente, a rodada 5 (melhor IoU de todas)
**piorou levemente** quatro dos cinco sub-parâmetros de 2.6 (só θ ficou
estável) — ver `HANDOFF_P2_3.md` Ruling 9 para a leitura completa.

Leitura: a correção do alvo de treino (Rulings 7/8) melhorou TODOS os
parâmetros da rodada 3 para a 4, inclusive τ (que tinha piorado na rodada
3). ζ (o parâmetro que depende mais de detalhe fino de forma — amplitude
do overshoot num sistema de 2ª ordem subamortecido) continua sendo o mais
sensível à qualidade da segmentação — a distância ao alvo caiu de ~5–7 p.p.
(rodadas 1/2) para ~0,6–0,7 p.p. (rodadas 4/5), mas as três variações
seguintes de preparo do alvo (limiar 32, contínuo) não reduziram mais essa
distância. As hipóteses de capacidade do modelo e tamanho do dataset
(`HANDOFF_P2_3.md §6`, Ruling 10) ficam como o próximo passo, ainda **não
executado**.

### Critério 2.8 — latência por imagem (< 500 ms)

| Rodada | Mediana | p95 |
|---|---|---|
| 2 | 2.073 ms | 5.307 ms |
| 4 | 2.104 ms | 5.659 ms |
| **5** | **3.353 ms** | **9.981 ms** |

❌ em todas. A latência **não deveria** depender de qual checkpoint está
carregado (mesma arquitetura, mesmo custo de forward pass) — o salto da
rodada 5 (quase 60% maior que a 4) é medido, mas não explicado por mudança
de código; ver Armadilha 4 (provável ruído de máquina, não efeito real do
treino). Decomposição por estágio (medição isolada da rodada 4, ainda
válida como referência de ordem de grandeza):

| Estágio | Tempo |
|---|---|
| `calibrate` (Bloco 2, OCR) | **3,12 s** |
| `predict_mask` (Bloco 3, U-Net 256², CPU) | 0,66 s |
| `mask_to_polyline` (Bloco 4) | 0,33 s |
| `identify` (Parte 1, estágio D) | 0,60 s |
| **total** | **~4,7 s** |

`calibrate` domina — é o custo de disparar um subprocesso `tesseract` por
blob de texto candidato (Ruling 1/Armadilha 1 do `HANDOFF_P2_2.md`).

### Critério 2.10 (consolidado das medições dos Blocos 3/3b)

| Extrator | Rodada 2 | Rodada 3 (limiar 0) | Rodada 4 | **Rodada 5 (melhor IoU)** |
|---|---|---|---|---|
| U-Net | 0,5716 | 0,4950 | 0,5597 | **0,6205** |
| Clássico (Bloco 3b) | 0,7153 | 0,7153 | 0,7153 | 0,7153 (inalterado) |

**Em IoU de máscara pura, o extrator clássico vence** em todas as cinco
rodadas de treino, mesmo com a rodada 5 reduzindo a distância. **Mas em
2.6 (o que efetivamente decide a qualidade do sistema), a U-Net vence em
qualquer uma das duas últimas rodadas** — comparação feita contra a rodada
4 (ver tabela abaixo; não repetida contra a rodada 5, que tem 2.6 pior que
a 4 mesmo com IoU melhor — não mudaria a conclusão qualitativa). IoU de
máscara e utilidade final para estimar parâmetros físicos **não são a
mesma coisa**, e este é o número que prova isso, não só a intuição.

`identify/pipeline.py::identify_from_image` ganhou o parâmetro `extractor=`
(opcional, não quebra a assinatura do `PLANO_PARTE2.md` quando omitido) para
permitir essa troca. Critério 2.6 medido, trocando `predict_mask`
(U-Net, rodada 4) por `extract_mask_classical` (Bloco 3b), mesmas 300
amostras, mesmo estágio D, mesmo oráculo:

| Parâmetro | ΔMAPE — U-Net (Rodada 4) | ΔMAPE — extrator clássico | Vencedor |
|---|---|---|---|
| K | +0,92 p.p. | +1,03 p.p. | U-Net (margem pequena) |
| τ | +2,00 p.p. | +1,85 p.p. | Clássico (margem pequena) |
| θ | +0,55 p.p. | +0,78 p.p. | **U-Net** |
| ωₙ | +1,78 p.p. | **+3,10 p.p.** (reprova) | **U-Net** |
| ζ | +3,64 p.p. | +4,38 p.p. | **U-Net** |
| **Pior parâmetro** | **+3,64 p.p.** | +4,38 p.p. | **U-Net** |

**A U-Net vence a comparação que importa**, mesmo perdendo em IoU de
máscara. Vence com folga em 3 dos 5 parâmetros (θ, ωₙ, ζ) — incluindo o
único caso em que o extrator clássico reprova um parâmetro que a U-Net não
reprova (ωₙ). Perde por margem pequena em K e τ (ambos os extratores dentro
do alvo de qualquer forma nesses dois). Leitura para a monografia: a
máscara mais grossa/menos precisa do extrator sem rede degrada mais o
detalhe fino (oscilação, frequência natural) do que a segmentação
aprendida da U-Net, mesmo com IoU pixel-a-pixel pior — **é a medição que
justifica a U-Net no trabalho**, exatamente o papel que o §1.8 do PLANO
previa para essa comparação (ela é que decide, não a intuição de qual
"deveria" ser melhor).

### Critério 2.11 — saída sempre presente (100%, sem exceção)

**Medido: 300/300 sem exceção**, tanto para `calibrate()` isoladamente
(Bloco 2) quanto para `identify_from_image` completo. **Mas ver Ruling 3**
— a checagem cobre "nunca levanta", não "o bloco `dimensionless` existe no
dicionário de retorno", porque esse bloco não existe na implementação atual.

## 4. Rulings

1. **Os números de 2.1/2.6/2.7/2.8/2.10 passaram por cinco remedições ao
   longo desta sessão**, uma por rodada de treino da U-Net — histórico
   completo em `HANDOFF_P2_3.md §0`. As rodadas 4 e 5 (as duas últimas) são
   as citadas como referência nas tabelas deste documento; rodadas
   anteriores aparecem só como comparação, nunca como o valor vigente. Ao
   contrário das remedições anteriores, a rodada 5 **não substitui** a
   rodada 4 como "número final" — nenhuma domina a outra (ver §0).
2. **A campanha de mutação consolidada (Passo 7 do Bloco 5 — os 17 mutantes
   de uma vez, sobre a suíte completa) não foi refeita aqui.** Cada mutante
   já foi validado individualmente no handoff do seu bloco de origem
   (`HANDOFF_P2_1.md`: P2-M01–M04; `HANDOFF_P2_2.md`: P2-M05–M09;
   `HANDOFF_P2_3.md`: P2-M11–M13, M10 não executado;
   `HANDOFF_P2_4.md`: P2-M14–M17), cada um com sua própria medição
   antes/depois. Repeti-los todos de uma vez sobre a suíte completa exigiria
   outra rodada de várias horas (o Bloco 2 sozinho já leva ~20-25 min por
   medição parcial, ~2h numa medição cheia).
3. **`identify_from_image` não separa `dimensionless`/`physical` no
   dicionário de retorno — lacuna real da Decisão E (§1.7 do PLANO), não
   fechada.** A implementação seguiu o Passo 1 do `PLANO_PARTE2.md`
   literalmente, que já não inclui essa separação no código de exemplo
   (só no JSON ilustrativo do `PLANO.md §1.7`). `params` vem de
   `FitResult.params` (`identify.classical`), que é só o nível físico; o
   nível adimensional (`order`, ζ, ωₙ·T, θ/T, K/y_faixa) **não é calculado
   nem devolvido separadamente** quando a calibração falha — o pipeline
   atual, quando `cal.ok=False`, devolve o dicionário vazio (`ok=False,
   reason=cal.reason`) sem tentar produzir nada adimensional. **Isto
   significa que o critério 2.11 do PLANO, na leitura mais estrita ("100%
   das amostras com `dimensionless` preenchido"), NÃO está fechado** — o que
   está fechado é uma leitura mais fraca ("nunca levanta exceção, `ok`/
   `reason` sempre bem formados"), registrada como tal na tabela de
   critérios. Fechar isto de verdade exigiria: (a) rodar o estágio D também
   sobre a série em pixels puros (sem calibração) para obter ζ, a ordem e as
   razões adimensionais, e (b) reestruturar o dicionário de retorno com os
   dois blocos. Não implementado por escopo/tempo — registrado como o item
   mais importante de dívida técnica desta Parte 2.
4. **A guarda do relatório da Parte 1 (`HANDOFF_P2_0.md` Ruling 3) segurou
   por toda a Parte 2**, confirmada repetidas vezes ao longo dos Blocos 1–5
   e das quatro rodadas de treino (hash `83e63912c32a6502af3744c16d3ba83e`
   idêntico antes e depois de cada sessão de teste, dezenas de
   verificações). Nenhuma delas rodou `pytest -q` completo (sem seleção)
   neste bloco — todas usaram seleção por `-k`, que é exatamente o caso que
   a guarda existe para proteger.

## 5. Armadilhas

1. **`reports/part2_strata.md` é regenerado do zero a cada processo `pytest`
   (o dicionário `RESULTS_P2` é de módulo, não persiste entre processos)** —
   então o arquivo no repositório neste momento reflete só a ÚLTIMA seleção
   de teste rodada, não a Parte 2 inteira. Os números de cada critério estão
   todos **documentados nos handoffs de cada bloco**, mas não
   simultaneamente no relatório em disco. Uma rodada `pytest tests/part2 -q`
   completa (sem `-k`) escreveria tudo de uma vez — não foi rodada por
   custo de tempo (estimativa: 3–4 horas, dominado pelo Bloco 2).
2. **A suíte completa da Parte 2 (`pytest tests/part2 -q`) nunca foi
   executada numa única invocação nesta sessão** — cada bloco foi validado
   com filtros `-k` isolados, por causa do custo de tempo de cada critério
   individualmente. Isso significa que o Passo 6 do Bloco 5 do
   `PLANO_PARTE2.md` ("rodar a suíte inteira das duas partes,
   `pytest -q`") **não foi executado como pedido**. O que FOI verificado
   repetidamente: (a) `pytest -q` roda 33/33 da Parte 1 sem quebrar (Bloco
   0); (b) `pytest tests/part2` isolado não corrompe o relatório da Parte 1
   (Bloco 0, e reconfirmado dezenas de vezes); (c) cada critério individual
   da Parte 2 foi medido e registrado. O que não foi confirmado: os testes
   de todos os blocos da Parte 2 juntos, numa única sessão, sem conflito de
   import ou efeito colateral entre eles. Risco julgado baixo (os módulos
   são independentes e os testes só leem `data/test`), mas não é o mesmo
   que verificado.
3. **O IoU de validação reportado DURANTE o treino não é comparável entre
   rodadas** (127, 0, 32, contínuo são réguas diferentes) — só os critérios
   medidos com a régua fixa e rigorosa da avaliação final (resolução
   original, limiar 127, mesma para todas as rodadas) são comparáveis — é o
   que as tabelas deste documento usam. Ver `HANDOFF_P2_3.md` Ruling 8 e a
   conversa que motivou essa descoberta (o IoU de validação da rodada 3,
   0,9055, media contra um alvo inflado — não era overfitting, era uma
   régua de treino desalinhada da régua de avaliação).
4. **A latência medida na rodada 5 (mediana 3.353 ms) é quase 60% maior que
   a da rodada 4 (2.104 ms), sem nenhuma mudança de código entre as duas
   medições** — só o checkpoint (pesos) mudou, e o forward pass da U-Net
   tem custo fixo independente dos valores dos pesos. **Isto não foi
   investigado a fundo**: a hipótese mais provável é ruído da máquina (a
   medição da rodada 5 rodou logo depois de ~8h30 de treino contínuo em
   CPU — possível *throttling* térmico ainda ativo, ou cache de disco/SO
   frio), não um efeito real do treino. Registrado como medido, não como
   explicado — se a latência for citada na monografia, remedir em condição
   de máquina ociosa antes de reportar como número definitivo.

## 6. Consolidado — todos os critérios do PLANO §PARTE 2 (números mais recentes: 2.6/2.8 da Rodada 4, 2.1/2.7/2.10 da Rodada 5 — ver §0 sobre não haver vencedor único)

| # | Critério | Alvo | Medido | Veredito | Bloco |
|---|---|---|---|---|---|
| 2.1 | IoU mediana (U-Net) | ≥ 0,85 | 0,6205 (rodada 5; 0,5597 na rodada 4) | ❌ | 3 |
| 2.2 | RMSE polilinha vs. máscara verdadeira | ≤ 2 px, p95 ≤ 5 px | RMSE 1,49 px / p95 6,70 px | ❌ (p95) | 4 |
| 2.3 | Erro relativo de escala | < 1% em ≥ 95% | 92,7% (n=232) | ❌ | 2 |
| 2.4 | Taxa de rejeição (falso alarme) | < 5% | 22,7% | ❌ | 2 |
| 2.5 | Rejeições corretas | ≥ 90%, n≥5 | 76,5% (n=68) | ❌ | 2 |
| 2.6 | Degradação end-to-end (pior parâmetro) | ≤ 3 p.p. | **+3,64 p.p. (ζ, rodada 4)** / +3,73 p.p. (rodada 5) — 4/5 parâmetros passam nas duas | ❌ (por pouco) | 5 |
| 2.7 | IoU por estrato | nenhum < 0,75 | 0,5856 no 1º estrato medido (rodada 5; 0,5394 na rodada 4) | ❌ | 3 |
| 2.8 | Latência por imagem | < 500 ms | mediana 2.104 ms (rodada 4) / 3.353 ms (rodada 5 — ver Armadilha 4) | ❌ | 5 |
| 2.9 | Cobertura da calibração | ≥ 90% | 77,3% | ❌ | 2 |
| 2.10 | U-Net × clássico (IoU de máscara) | sem alvo | U-Net 0,6205 (rodada 5) / clássico 0,7153 (clássico vence) | — (diagnóstico) | 3/3b |
| 2.10-func. | U-Net × clássico, medido em 2.6 (pior parâmetro) | sem alvo | U-Net +3,64 p.p. (rodada 4) / clássico +4,38 p.p. (**U-Net vence**) | — (diagnóstico) | 5 |
| 2.11 | Saída sempre presente, sem exceção | 100% | 100% nunca levanta (escopo parcial — Ruling 3) | ⚠️ | 2/5 |

**Nenhum critério numérico do PLANO §PARTE 2 fecha integralmente — mas 2.6
chegou muito perto (4 dos 5 sub-parâmetros já passam, o pior fica a
0,6–0,7 p.p. do alvo), depois de cinco rodadas de treino iterando sobre uma
causa raiz medida, não adivinhada.** Causa de cada critério:

- **2.1, 2.7, 2.10 (IoU de máscara):** mesmo depois de resolver o problema
  do alvo de treino perder a curva (Rulings 7/8) e trocar para alvo
  contínuo (Ruling 9), o IoU pixel-a-pixel fica no melhor caso em ~0,62 —
  abaixo do alvo de 0,85. Não foi mais investigado nesta sessão (cinco
  rodadas de treino, ~35h de parede acumuladas nesta máquina, já é o limite
  razoável aqui); as hipóteses do Ruling 10 do `HANDOFF_P2_3.md`
  (capacidade do modelo, tamanho do dataset) ficam como próximo passo
  **pendente**, a executar em outra máquina.
- **2.3, 2.4, 2.5, 2.9 (calibração/OCR):** cobertura de OCR abaixo do alvo,
  depois de seis correções medidas e documentadas (`HANDOFF_P2_2.md`) que já
  levaram o sistema de "quase não funciona" (2/30) a "funciona na maioria
  dos casos" (77%).
- **2.2 (p95):** limitação estrutural identificada (imagens de proporção
  extrema) — `HANDOFF_P2_4.md`.
- **2.6:** o mais próximo de fechar — só ζ falta, por margem pequena e
  estável nas duas últimas rodadas (0,64–0,73 p.p. acima do alvo). Três
  variações de preparo do alvo de treino (Rulings 7/8/9) não fecharam essa
  margem — hipóteses de capacidade/dataset (Ruling 10) pendentes.
- **2.8:** dominado pelo custo de OCR do Bloco 2 (subprocesso `tesseract`
  por rótulo).
- **2.11:** implementado com escopo mais estreito que o texto do PLANO —
  Ruling 3.

**O que FECHOU de fato, com número real e positivo:**
- G0.1–G0.4 (Bloco 0): ambiente, dataset, guarda do relatório.
- G1.1–G1.3 (Bloco 1): moldura e ticks geométricos, 99,7%/100%/98,3%.
- G3b.1–G3b.4 (Bloco 3b): extrator clássico, 0,7153 de IoU, latência 128 ms.
- 2.2 mediana (Bloco 4): 1,49 px, dentro do alvo (só o p95 falha).
- **2.6, quatro dos cinco sub-parâmetros** (K, τ, θ, ωₙ): dentro do alvo de
  3 p.p. na Rodada 4.
- **A U-Net vence o extrator clássico em 2.6** (pior parâmetro +3,64 p.p.
  contra +4,38 p.p.), mesmo perdendo em IoU de máscara (2.10) — a medição
  que justifica de fato a U-Net estar no trabalho, não só a arquitetura em
  si.
- A suíte de mutação confirma que os portões que passam **medem o que dizem
  medir** (controles corretos, mutantes reais detectados) em 15 dos 17
  mutantes do PLANO (mais 3 mutantes adicionais criados ao longo do
  caminho, todos detectados) — ver cada `HANDOFF_P2_*.md` para a tabela
  completa.

## 7. O que a Parte 3 (ou quem retomar) precisa saber

1. **`identify.pipeline.identify_from_image` é a porta de entrada**, pronta
   e testada quanto a nunca levantar exceção — mas devolve só o nível físico
   dos parâmetros (Ruling 3), não o par dimensionless/physical do PLANO
   §1.7. Se a Parte 3 depender dessa separação, ela precisa ser implementada
   primeiro.
2. **A Parte 2, como está, não está pronta para ser citada como "critérios
   fechados" na monografia** — está pronta para ser citada como "sistema
   funcional, com números reais medidos, e o critério mais importante (2.6)
   a menos de 1 ponto percentual de fechar", que é uma afirmação honesta e
   forte. A tabela da §6 é o material bruto para essa seção.
3. **A U-Net passou por CINCO rodadas de treino, cada uma motivada por uma
   causa medida, não por tentativa cega**: taxa de aprendizado → alvo de
   treino perdendo a curva → alvo de treino inflado demais → equilíbrio
   (limiar 32) → alvo contínuo (melhor IoU, mas não melhor ζ). Vale como
   material de metodologia para a monografia, não só como resultado
   numérico. **Dois checkpoints ficam preservados, sem um "final" único**:
   `models/unet_stageA_rodada4_thr32_ep_final.pt` (melhor 2.6) e
   `models/unet_stageA_rodada5_alvo_continuo_ep_final.pt` (melhor IoU, =
   `models/unet_stageA.pt` hoje).
4. **Comparação U-Net × extrator clássico em 2.6: feita** (ver §3, contra a
   rodada 4). A U-Net vence a comparação que importa (pior parâmetro
   +3,64 p.p. contra +4,38 p.p. do clássico), mesmo perdendo em IoU de
   máscara pura (2.10) — **é a medição que justifica usar a U-Net no
   pipeline final**, não só a intuição de que "deveria" ser melhor.
   `identify_from_image` ganhou o parâmetro opcional `extractor=` para
   permitir essa troca sem quebrar a assinatura do PLANO.
5. **Hipóteses para fechar ζ — TRIADAS em 25/08/2026, ver `HANDOFF_P2_6.md`.**
   Fatorial 2×2 de 4 pilotos: capacidade (`base=24`) APROVADA, tamanho do
   dataset REPROVADO. A rodada completa de 25 épocas segue não executada, com
   o procedimento em `HANDOFF_P2_6.md §7`. Linha de base remedida: **ζ = +3,65 p.p.**
   O texto abaixo é o plano original, preservado como registro.

   *(redação original:)* **PLANEJADAS, NÃO EXECUTADAS,
   documentadas para retomar em outra máquina** (`HANDOFF_P2_3.md` Ruling
   10, texto completo lá):
   - **(a) Capacidade do modelo.** `UNet(base=24)` (4,37 M parâmetros,
     2,25× o atual) ou `base=32` (7,76 M, 4×). Custo estimado nesta máquina
     (CPU-only): ~19h e ~34h de treino completo, respectivamente — caro
     demais para testar sem um piloto curto (2–3 épocas) primeiro
     comparando a curva de IoU_val inicial contra `base=16`.
   - **(b) Tamanho do dataset.** Dobrar `data/train` (4.200 → 8.400,
     mantendo val/test fixos em 900/900 para números continuarem
     comparáveis), usando uma seed-base nova (≥ 4; as bases 1/2/3 já estão
     em uso — ver `HANDOFF_P2_0.md`).
   - Nenhuma das duas foi treinada ainda. **Não é necessário regenerar
     `data/train`/`val`/`test` atuais para continuar em outra máquina** —
     ver a nota de reprodutibilidade abaixo.
6. **Reprodutibilidade do dataset em outra máquina.** `data/` é ignorado
   pelo git (`.gitignore`) — não viaja com o `git clone`/`pull`. Precisa
   ser regenerado do zero na máquina nova, o que é **barato e determinístico
   bit-a-bit** (confirmado no critério 1.6 da Parte 1 — mesma seed produz
   os mesmos bytes de `image.png`/`mask.png`): ~7 min para 6.000 amostras
   nesta máquina (medido, ver `reports/part1_metrics.md §7`; pode variar
   com o hardware, mas a ordem de grandeza é minutos, não horas). Comando
   exato usado para gerar os splits atuais (`seed_i = base·1_000_003 + i`,
   confirmado lendo `meta.json` das amostras existentes):
   ```bash
   .venv/bin/python -m dataset.generator data/train 4200 1
   .venv/bin/python -m dataset.generator data/val   900  2
   .venv/bin/python -m dataset.generator data/test  900  3
   ```
   Depois, `pytest tests/part2/test_env.py::test_g0_3_splits_disjuntos_e_completos`
   confirma 4.200/900/900 com seeds disjuntas antes de treinar. Se for
   testar a hipótese (b) do item 5 (mais dados), gerar um `data/train`
   maior com uma seed-base ≥ 4 (para não colidir com val/test), **sem**
   regenerar `data/val`/`data/test` — eles precisam continuar os mesmos
   para os números de 2.1/2.6/2.7/2.8/2.10 continuarem comparáveis com as
   cinco rodadas já medidas.
7. **Trabalho futuro de maior retorno, em ordem:** (a) fechar ζ em 2.6
   — rodada completa de `base=24`, única hipótese aprovada na triagem
   (`HANDOFF_P2_6.md §7.3`); (b) fechar a separação
   dimensionless/physical em `identify_from_image`; (c) investigar a cauda
   de falhas de OCR do Bloco 2 amostra a amostra; (d) `tesserocr` (engine
   persistente) para o critério 2.8.
