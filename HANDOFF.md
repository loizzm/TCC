# HANDOFF — TCC-2 (identificação de plantas a partir de imagens de resposta ao degrau)

Documento de retomada. Descreve o que está **pronto e verificado** na Parte 1 e o
que **falta** nas Partes 2 e 3. Escrito ao fim da Parte 1.

Ambiente: use sempre `/home/loizm/work/TCC-2/.venv/bin/python` (Python 3.11).
O `python3` do sistema é 3.14 e não tem as bibliotecas.

---

## 1. Estado geral

| Parte | Escopo | Estado |
|---|---|---|
| **Parte 1** | Gerador saneado + identificação clássica + prova de solubilidade com oráculo | **CONCLUÍDA — 33/33 testes verdes**, com o teste de mutação fechado (§3.5) |
| Parte 2 | Estágio A (extração da curva) + Estágio B (calibração dos eixos) | **executada — sistema funcional, nenhum critério numérico fechado integralmente (ver §4)** |
| Parte 3 | Validação OOD + PID/IMC + baseline fim-a-fim | **não iniciada** |

> **Revisão de arquitetura, 22/08/2026.** O **Estágio C (estimador neural 1D) foi
> medido e removido** do plano. O pipeline passou de quatro para três estágios:
> A → B → D. Duas decisões de robustez acompanham: OCR opcional e extrator clássico
> como contingência de GPU. Registro completo em `PLANO.md §1.3, §1.7, §1.8`; a
> alternativa fim-a-fim ficou planejada em `PLANO_CNN_FIM_A_FIM.md`; as referências
> que sustentam cada decisão em `REFERENCIAS.md`.

A Parte 1 fecha o portão de aceitação: os sete critérios 1.1–1.7 do PLANO estão
medidos, com os números na tabela mestre de `reports/part1_metrics.md`.

O **teste de mutação**, que era a única verificação em aberto, foi executado
(§3.5). Ele encontrou **três buracos de cobertura reais**, todos corrigidos na
suíte — nenhum era defeito do gerador. A suíte passou de 30 para 33 testes, e os
17 mutantes são hoje detectados 17/17.

---

## 2. O que existe hoje no repositório

```
TCC-2/
├── dataset/
│   ├── generator.py     (500 l)  sorteio do sistema, renderização, máscara, meta.json
│   └── randomize.py     (379 l)  sorteio do estilo visual, isolado do rótulo
├── identify/
│   └── classical.py     (893 l)  least_squares + AIC + 3 baselines clássicos
├── tests/
│   ├── conftest.py     (1063 l)  fixtures de sessão + gerador do relatório
│   ├── test_part1.py   (1121 l)  critérios 1.1, 1.2, 1.5, 1.6, 1.7, G + contrato
│   └── test_leakage.py  (549 l)  critérios 1.3 e 1.4 (a–e)
├── pytest.ini
├── reports/part1_metrics.md      GERADO pela suíte — não editar à mão
├── PLANO.md                      plano de execução das 3 partes
├── HANDOFF.md                    este arquivo
├── data/                         vazio, não versionado
└── img.py                        LEGADO DEFEITUOSO — NÃO TOCAR (ver §7)
```

### Como rodar

```
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest -q
```

Fecha em **~128 s** nos 16 threads com a máquina ociosa, **33 passed** (já medido
em 331 s numa execução com a máquina sob carga — o tempo total varia muito com
concorrência; o que não varia são os 33 verdes). Isso regenera
`reports/part1_metrics.md`. **Rode sempre sem `-m` / `-k`**: com filtro, o
relatório sai parcial (a suíte imprime um banner de aviso no topo do arquivo
quando isso acontece).

Para gerar um dataset fora dos testes:

```
.venv/bin/python -m dataset.generator data/train 6000 0
```

---

## 3. Parte 1 — o que foi feito

### 3.1 Gerador saneado (`dataset/`)

Substitui o `img.py` legado, que vazava o rótulo dentro da imagem. Duas
garantias estruturais:

- **Zero vazamento de rótulo.** `sample_style(rng)` **não recebe** o `SystemSpec`
  — ela fisicamente não pode ver o rótulo. Sistema e estilo vêm de streams de RNG
  independentes (`SeedSequence.spawn(3)`). A regra é estrutural, não uma promessa.
- **Determinismo bit-a-bit.** Mesma seed ⇒ mesmos bytes de `image.png` e
  `mask.png`, e independe do número de workers. Válido dentro do ambiente pinado
  do `requirements.txt`.

Cada amostra é um diretório com `image.png`, `mask.png` e `meta.json` (esquema
completo no contrato). A máscara é a curva isolada, renderizada numa segunda
figura de geometria idêntica — nunca extraída da imagem colorida.

### 3.2 Identificação clássica (`identify/classical.py`)

Estágio D do PLANO. `identify(t, y)` ajusta FOPDT e 2ª ordem por
`least_squares` com multi-start e escolhe por AIC. Inclui os três baselines
clássicos (tangente, Smith, Sundaresan–Krishnaswamy). Robusto a janela truncada
e a NaN; `identify`/`fit_*` nunca levantam exceção.

### 3.3 Suíte de aceitação (`tests/`)

Mede os sete critérios e **gera o relatório da monografia**. Resultado da última
execução completa:

| # | alvo | medido | veredito |
|---|---|---|---|
| 1.1 | MAPE < 1% em K, τ, θ, ωn, ζ (`w ≥ 3`, série limpa) | **0,0000%** nos cinco (n = 181) | PASSA |
| 1.2 | MAPE < 5% a 20 dB (`w ≥ 3`) | K 0,385% · τ 0,979% · θ 1,568% · ωn 3,111% · ζ 3,557% | PASSA |
| 1.2b | ζ ≥ 1,6: K e T_lento < 5%, NRMSE < 0,05 | K 0,351% · T_lento 1,798% · NRMSE 2,73e-03 | PASSA |
| 1.2c | idem, população dedicada (n = 256) | ζ<1,6: ωn 3,259%, ζ 3,789% · ζ≥1,6: K 0,307%, T_lento 1,479% | PASSA |
| 1.3 | acurácia do GBM só com `render` ≤ 0,55 | **0,4985** (n = 20000) | PASSA |
| 1.4a | \|ρ\| < 0,05 + Bonferroni (n = 20000) | max \|ρ\| = 0,0194; 0 significativos | PASSA |
| 1.4b | dataset renderizado (n = 600 cada) | clean 0,1133 (p 0,963) · noisy 0,1762 (p 0,129) | PASSA |
| 1.4c | round-trip exato do bloco `render` | 1200 amostras, todas idênticas | PASSA |
| 1.5 | máscara × `axis_affine` | RMSE do viés normal 0,1649 px (0,0298 sem marcador) | PASSA |
| 1.5c | controle negativo (3 px de erro ⇒ ≥ 10×) | 0,1649 → 2,2293 px (13,5×) | PASSA |
| 1.6 | determinismo bit-a-bit | 5 seeds × 2 gerações × {com, sem} ruído: idênticos | PASSA |
| 1.7 | extrapolado para 6000 < 15 min | 2,17 s/200 ⇒ **1,08 min** | PASSA |
| R | máscara não degenerada (Ruling S) | cobertura mín 0,9594; 0/1200 abaixo de 0,93 | PASSA |

Medidos **sem assertiva**, por decisão registrada: estrato truncado `w < 3`
(Ruling C), MAPE de ωn/ζ em ζ ≥ 1,6 (Ruling N), réplica do GBM em n = 600.

### 3.4 Resultados que são material direto da monografia

Estão todos em `reports/part1_metrics.md`, com as tabelas prontas:

1. **Não-identificabilidade prática em 2ª ordem superamortecida.** Em ζ ∈ [2,2;
   3,0], ωn e ζ erram ~85% enquanto K erra 0,28% e a constante do polo lento erra
   1,28%, com `corr(erro_ωn, erro_ζ)` = **+0,9997**. É deslizamento ao longo de
   uma curva de nível onde a dinâmica observável é a mesma — limite de
   informação, não deficiência do estimador. Reproduz de forma independente a
   tabela de evidência do contrato. **Isto justifica o NRMSE de reconstrução
   como métrica primária do trabalho** (PLANO §1.5).
2. **Contraprova do vazamento de rótulo.** Um GBM que só vê atributos visuais
   fica em 0,4985 de acurácia — o acaso. É a contraprova direta dos 93% do
   `img.py` legado.
3. **Limite de informação da janela.** Com `w < 3` a curva é cortada antes do
   regime permanente e o ganho deixa de ser identificável: MAPE(K) = 127,6% com
   mediana 2,32% — o erro vem de poucas amostras patológicas, não de degradação
   uniforme.
4. **Baselines clássicos × `identify`.** Os baselines não quebram em exatidão, e
   sim em **cobertura**: no estrato truncado Smith responde em 54,7% das séries e
   S–K em 24,2%, enquanto `identify` responde em 100%.
5. **Registro de metodologia — três denominadores.** A verificação de máscara não
   degenerada errou duas vezes pela mesma causa (limiar fixo contra grandeza com
   teto físico variável) antes de acertar. Está documentado no §5.2.1 do
   relatório; é material honesto sobre processo de verificação.

### 3.5 Status de verificação — o que foi auditado e o que não foi

Cada entregável passou por um revisor independente, que **não** confiou nos
relatórios e re-mediu por conta própria. Registro honesto do que ficou coberto:

| Alvo | Revisão | Resultado |
|---|---|---|
| `dataset/` (gerador) | completa | spec ✅, qualidade aprovada, 0 Critical, 0 Important, 9 Minor. Re-medido de forma independente: \|ρ\| máx 0,0087 sobre 154 pares rótulo×render em N=200k; determinismo bit-a-bit em 6 seeds novas e entre processos com `PYTHONHASHSEED` diferente; afim correta nos extremos (dpi 60/200, 1600×180 e 240×1200) com viés −0,0013 px; máscara sem contaminação com 3 distratores opacos + grade + textos; ≥3 ticks rotulados por eixo em 200/200 |
| `dataset/` — correção do `T_dom` | re-revisão escopada | 8/8 achados endereçados, 0 regressões |
| `identify/classical.py` | completa | conformidade ✅, qualidade boa, 0 Critical, 1 Important, 5 Minor. **Auditoria de Cramér–Rao SUSTENTADA** por teste oracle-start (ver §3.6) |
| `identify/classical.py` — correção | re-revisão escopada | todos os achados endereçados, 0 regressões; os 5 riscos da memoização nova checados um a um, nenhum se concretiza |
| `tests/` (suíte de aceitação) | **teste de mutação COMPLETO** | 17 mutantes + 1 controle; 3 buracos encontrados e corrigidos; hoje 17/17 detectados — ver §3.5.1 |
| Parte 1 como um todo | **NÃO FEITA** | revisão final ampla não chegou a ser despachada |

A revisão da suíte confirmou de forma independente que ela é **reproduzível** e
que `reports/part1_metrics.md` bate byte a byte com a execução do revisor, exceto
os números de tempo de parede. Isso continua valendo: na retomada de 16/08/2026 a
suíte reproduziu os mesmos números, em 127,9 s.

### 3.5.1 Teste de mutação — resultado

A pergunta que importa numa suíte de aceitação é *ela detectaria um defeito, ou
passa por vacuidade?* Método: para cada mutante, uma cópia do repositório em
`/tmp` com uma substituição exata, e a suíte **inteira** (sem `-x`, para levantar
quais testes pegam cada defeito). Um mutante-controle sem alteração roda junto e
tem de ficar verde — ele valida o harness.

| # | mutante | veredito | pego por |
|---|---|---|---|
| 00 | controle: nenhuma alteração | verde (valida o harness) | — |
| 01a | cor da curva por `order`, refletida no meta | detectado | 1.4c round-trip + 1.4b clean/noisy |
| 01b | idem, **só na imagem** (meta correto) | **passava — corrigido** | 1.4d (novo) |
| 02a | `axhline(K)`/`axvline(θ)` de volta **na imagem** | **passava — corrigido** | 1.4e (novo) |
| 02b | `axhline(K)` contaminando a máscara | detectado | 1.5 |
| 03 | distratores desenhados dentro da máscara | detectado | 1.5 |
| 04a | `axis_affine` deslocada em 1 px | detectado | 1.5 |
| 04b | `axis_affine` deslocada em 3 px | detectado | 1.5 |
| 05 | sinal de `sy` invertido | detectado | 1.5 + contrato do meta |
| 06 | `series.y` limpo com o ruído ligado | detectado | `test_series_is_what_was_drawn` |
| 07 | `np.random` global num sorteio de estilo | detectado | 1.6 (11 testes) |
| 08 | `generate_dataset` dependente dos workers | detectado | 1.6 workers |
| 09 | viés de +3% em K dentro do `identify` | detectado | 1.1 |
| 10 | `identify` sempre `order="fopdt"` | detectado | 1.1 e 1.2 |
| 11 | `T_dom` revertido para `1/(ζωn)` | detectado | 1.2 + tabela do RULING N |
| 12 | `_estimate_gain` voltando a `max(y)` | **passava — corrigido** | critério G (novo) |
| 13a | `series` com 256 pontos | detectado | contrato do meta + 1.2 |
| 13b | `series.t` não uniforme | detectado | contrato do meta |

**Os três buracos eram todos da suíte, nenhum do gerador** — exatamente o que a
regra do §9.2 previa. Os dois primeiros têm a mesma causa raiz e ela é o achado
metodológico da rodada:

> **Os critérios 1.3 e 1.4a–c nunca abriam a `image.png`.** Eles medem
> correlação entre o bloco `render` do meta e o rótulo. Um vazamento que existe
> só nos pixels — que é *literalmente* o defeito do `img.py` — era invisível para
> a suíte inteira. A garantia anti-vazamento era estrutural (`sample_style` não
> vê o rótulo) e o teste só confirmava o meta; faltava fechar o elo
> pixel → meta → estilo.

O terceiro é de outra natureza: `_estimate_gain` só alimenta os três baselines
clássicos, e a única assertiva sobre eles compara `identify` com o **melhor**
baseline — piorar todos de uma vez não quebra essa comparação. A tabela de
baselines da monografia estava, portanto, sem guarda nenhuma. Note que a versão
anterior deste documento afirmava que o mutante 12 "já tinha cobertura conhecida
por construção"; o teste de mutação mostrou que **não tinha** — foi encontrado
por medição ad hoc na Tarefa 2, não pela suíte.

**Correções (três testes novos, todos com limiar fixado depois de medir o teto):**

| critério | o que assere | falso positivo medido | poder medido |
|---|---|---|---|
| **1.4d** | cor modal da tinta na `image.png` == `render.line_color` | 0 em 900 (estrato de cor modal dominante) | 279/279 no mutante |
| **1.4e** | nenhuma reta de span completo além de `render.n_distractors` | 0 em 916 amostras sem grade | 143/157 (91%) |
| **G** | `_estimate_gain` em janela truncada, com controle positivo | 0 (MAPE 0,0000%, n = 191) | max(y) erra 30,5% no mesmo estrato |

O 1.4e custou três iterações de limiar, pela mesma razão de sempre neste projeto
(limiar fixo contra grandeza com teto variável): exigir continuidade descarta
linhas tracejadas; 1 px de folga em torno da máscara deixa a franja de
anti-aliasing de uma curva pontilhada virar "reta"; e extensão de ponta a ponta
sozinha marca uma linha que passa pela **legenda** num canto e por um **tick para
dentro** no outro. A regra final exige tinta em 7 das 8 faixas da linha. Cada uma
das três condições está documentada na docstring de `_spanning_rows` com o falso
positivo que a motivou — é material direto para a seção de metodologia.

**Reprodução:** o harness está em `/tmp/.../scratchpad/` (efêmero). Ele é ~120
linhas: copia o repo, aplica uma substituição exata (falha alto se a âncora não
casar exatamente uma vez), roda `pytest` e registra quais testes ficaram
vermelhos. Vale reescrevê-lo dentro do repositório se a Parte 2 for repetir o
exercício — a recomendação é repetir, sim, para os estágios A e B.

### 3.6 A auditoria de Cramér–Rao (por que o critério 1.2 é limite e não falha)

O implementador da identificação alegou que o erro residual a 20 dB era piso
estatístico, não deficiência do otimizador. Isso é o tipo de alegação que exige
ceticismo, então foi auditada de forma independente com um **teste oracle-start**:
partir `least_squares` dos parâmetros VERDADEIROS, com tolerância 1e-15 e 12
restarts, e ver se acha SSE menor que o do módulo.

Resultado: SSE estritamente menor em apenas **4/200** séries FOPDT e **1/200** de
2ª ordem. No pior estrato (ζ≥2,2, n=80), com 40 restarts: **0/80**, com MAPE
idêntico dígito a dígito. Mediana de `|erro|/desvio_CRLB` entre **0,59 e 0,74**,
contra 0,674 esperado de um estimador eficiente.

Conclusão: o estimador está no piso de informação. O que sobra de erro é do ruído,
não do método — e é isso que sustenta o item 1 da §3.4.

---

## 4. O que a Parte 2 entrega para a Parte 3

**Executada em 22/08/2026** (mesma sessão), nos seis blocos do
`PLANO_PARTE2.md` (0, 1, 2, 3b, 3, 4, 5). Handoffs completos, um por bloco:
`HANDOFF_P2_0.md` a `HANDOFF_P2_5.md` — cada um com os números medidos, os
Rulings (divergências do plano encontradas e corrigidas com medição) e as
armadilhas registradas. **Leia `HANDOFF_P2_5.md §6` primeiro** — tem a tabela
consolidada dos onze critérios.

**Resumo honesto: o sistema funciona (produz parâmetros físicos a partir de
imagem, sem exceção, na maioria dos casos), e o critério mais importante
(2.6, degradação end-to-end dos parâmetros físicos) chegou muito perto de
fechar — mas nenhum dos onze critérios numéricos do PLANO fecha
integralmente.** As causas são conhecidas e diagnosticadas, não um
mistério:

1. **A U-Net (Estágio A) passou por CINCO rodadas de treino, cada uma
   motivada por uma causa raiz medida** (não tentativa cega) — histórico
   completo no `HANDOFF_P2_3.md §0`. Rodada 1 (LR fixo): platô em
   IoU_val ~0,66. Rodada 2 (+ *scheduler* `ReduceLROnPlateau`): platô em
   ~0,685 — descartou taxa de aprendizado como único gargalo. Investigando
   mais fundo, mediu-se que o `letterbox` para 256×256 (256² foi a
   resolução viável nesta máquina sem GPU — 512² mede > 90 min/época)
   estava **apagando a curva do alvo de treino** em imagens grandes (68% do
   conjunto). Rodada 3 (alvo corrigido, limiar 0): resolveu o sumiço mas
   **inflou a máscara-alvo além do necessário**, piorando o IoU real de
   teste apesar do IoU de validação disparar — não era *overfitting* (a
   régua de treino e a régua de avaliação é que eram diferentes). Rodada 4
   (limiar recalibrado para 32, por medição): IoU de teste 0,56, e **o
   critério 2.6 chegou a 4 dos 5 parâmetros dentro do alvo**, faltando só
   ζ, por 0,64 ponto percentual — o melhor resultado em 2.6 até hoje.
   Rodada 5 (alvo contínuo, eliminando a escolha de limiar por completo):
   **melhor IoU de teste das cinco rodadas (0,62)**, mas ζ em 2.6 não
   melhorou — ficou 0,73 p.p. acima do alvo, levemente pior que a rodada 4.
   **Não há checkpoint único "final"**: rodada 4 continua melhor em 2.6
   (o critério que decide), rodada 5 é melhor em IoU puro — ambos
   preservados em disco. `identify/extract_classical.py` (Bloco 3b, §1.8
   do PLANO) é o extrator sem rede que tira a GPU do caminho crítico, e
   **continua com IoU de máscara melhor que a U-Net treinada** (0,72 vs.
   0,56–0,62) — mas medido em 2.6 (o que realmente decide a qualidade do
   sistema, não IoU de máscara isolado), **a U-Net vence** (pior parâmetro
   +3,64 p.p. contra +4,38 p.p. do clássico, que chega a reprovar ωₙ
   enquanto a U-Net não). É a medição que justifica a U-Net estar no
   trabalho — ver `HANDOFF_P2_5.md §3`. **Hipóteses de capacidade do
   modelo e tamanho do dataset ficam PENDENTES**, documentadas para
   continuar em outra máquina — ver `HANDOFF_P2_3.md` Ruling 10 e
   `HANDOFF_P2_5.md §7` item 5 (inclui o comando exato para regenerar
   `data/train`/`val`/`test`, já que `data/` não é versionado no git).
2. **A calibração de eixos (Estágio B, OCR) cobre ~77% das amostras**, não
   90%. Seis correções reais foram encontradas e aplicadas com medição
   (`HANDOFF_P2_2.md` Rulings 1–6): a ordem RANSAC→consistência (não o
   inverso), a whitelist do Tesseract que quebrava o OCR no engine LSTM,
   detecção de ticks bidirecional (dentro E fora da moldura), leitura de
   rótulo por blob de texto (não por marca de tick), tolerância a lacunas na
   checagem de equiespaçamento, e desempate do RANSAC por resíduo total. O
   sistema saiu de "quase não funciona" (2/30 amostras calibravam) para
   "funciona na maioria dos casos" (77%) — sem um próximo alvo óbvio de alto
   retorno para fechar os 13 pontos percentuais restantes.
3. **A latência (critério 2.8) é dominada pelo custo de disparar um
   subprocesso `tesseract` por rótulo candidato** — mediana medida entre
   2,1 s e 6,5 s (varia com a carga da máquina no momento da medição, não
   com o pipeline) contra o alvo de 500 ms. Caminho de correção conhecido e
   não implementado: um engine OCR persistente (`tesserocr`) em vez de
   `pytesseract`.

**Arquivos entregues** (todos com testes em `tests/part2/test_part2.py` e
suíte de mutação, ver cada `HANDOFF_P2_*.md`):
`identify/calibrate.py` (Estágio B — Blocos 1+2), `identify/extract.py`
(U-Net — Bloco 3), `identify/extract_classical.py` (extrator sem rede —
Bloco 3b), `identify/polyline.py` (máscara→polilinha — Bloco 4),
`identify/pipeline.py` (`identify_from_image`, a porta de entrada — Bloco 5),
`train_unet.py` (script de treino, raiz).

**Lacuna real, não fechada:** `identify_from_image` devolve só o nível
**físico** dos parâmetros — a separação `dimensionless`/`physical` da
Decisão E (`PLANO.md §1.7`) não está implementada no dicionário de retorno.
Ver `HANDOFF_P2_5.md` Ruling 3 antes de assumir que o critério 2.11 fecha na
leitura estrita do PLANO.

**O que a Parte 1 entregou para a Parte 2** (histórico, ainda válido):
- `mask.png` como verdade de terra de segmentação, validada contra a
  `axis_affine` a **0,03 px** de viés no estrato sem marcador;
- `plot_bbox_px`, `axis_affine` e `ticks` no meta como verdade de terra de
  calibração;
- o número que o Estágio A precisava saber de antemão: **quanto o extrator
  ingênuo "mediana por coluna" erra por estilo de traço** — 0,19 px em linha
  sólida contra 0,92 px em pontilhada, com **43% das colunas sem tinta** em
  `:` (§5.1 do relatório) — confirmado como o fator dominante também em
  `identify/polyline.py` (Bloco 4), que precisou de uma correção adicional
  (união de componentes conexas, não só a maior) para lidar com isso.

---

## 5. O que falta — Parte 3 (validação OOD, controle e baseline fim-a-fim)

> **Reescrita em 22/08/2026.** A versão anterior desta seção era "Estágios C e D + OOD".
> O Estágio C foi removido (`PLANO.md §1.3`) e o D já existe desde a Parte 1. A Parte 3
> **não constrói componente novo** — o pipeline está completo ao fim da Parte 2. Ela mede.

Arquivos a criar: `tests/test_part3.py`, `reports/final_report.md`, `ood/`, `e2e/`.
Arquivos que **deixaram** de ser necessários: `dataset/series.py`, `identify/estimator.py`.

- **Validação OOD (~60 imagens nunca vistas):** MATLAB/Simulink, Python Control,
  figuras de livro (Ogata, Nise), planilhas, e ~10 curvas de plantas reais (ou
  4ª ordem simulada). **Sem esse conjunto o trabalho demonstra apenas que o sistema
  aprendeu a inverter o gerador.** Comece a coletar no Dia 1, não no Dia 17.
- **Experimento de utilidade para controle:** PID por IMC sobre o modelo identificado
  vs. sobre o verdadeiro, comparados em malha fechada simulada. Critério 3.10.
- **Baseline fim-a-fim:** treinar a CNN 2D que a `§1.2` rejeitou, sobre o mesmo dataset,
  e comparar — sobretudo no OOD. Plano completo em `PLANO_CNN_FIM_A_FIM.md`. Critério 3.9.

Critérios 3.1 a 3.12. Os três que carregam o argumento:

- **3.6/3.7** — NRMSE de reconstrução, métrica primária, no sintético e no OOD;
- **3.10** — PID por IMC: mostra que o erro paramétrico residual é irrelevante para a
  finalidade de controle, o que fecha o círculo com o título do curso;
- **3.9** — CNN fim-a-fim × pipeline em estágios: converte a Decisão B de argumento em
  medição, e testa a hipótese de vazamento pelo seu sintoma observável (generalizar
  pior fora da distribuição).

### 5.1 O gatilho que pode ressuscitar o Estágio C

O critério **3.12** existe para isso, e é a condição de honestidade da Decisão C.
Meça, sobre as séries **extraídas** (não as do oráculo):

| medida | limite | consequência se violar |
|---|---|---|
| taxa de convergência de `identify` | ≥ 99% | abaixo disso, o chute inicial voltou a ser problema |
| NRMSE p95 de reconstrução | ≤ 0,02 | acima disso, idem |

Se qualquer um dos dois falhar, a especificação original do Estágio C — CNN 1D dilatada,
três cabeças, parametrização log/logit, degradação simulada — está preservada no
histórico do git (commit anterior a esta revisão) e volta à mesa. **Enquanto os dois
passarem, uma segunda rede é peso morto.** Reporte os dois números explicitamente no
`HANDOFF_P2_5.md`, mesmo que passem folgado: um critério que ninguém mede é decoração,
e isso é exatamente o que o §8 proíbe.

---

## 6. Decisões tomadas durante a execução (e o que custam se estiverem erradas)

Vários critérios do PLANO tiveram a **forma** alterada durante a execução, porque
o alvo literal era estatisticamente ou fisicamente impossível de verificar. Cada
uma dessas decisões foi tomada com medição, não por conveniência, e cada uma
precisa ser conferida por quem assina o trabalho. **Nenhum alvo numérico foi
afrouxado sem que a grandeza medida fosse trocada por uma bem posta.**

| # | O que o PLANO pedia | Problema | O que passou a valer | Custo se estiver errado |
|---|---|---|---|---|
| C | 1.1/1.2 com MAPE < 1% e < 5% sobre toda a distribuição | A janela é sorteada entre 0,5× e 6× a constante dominante; a 0,5× a curva não determina K a 1% nem com informação perfeita | Assertiva no estrato `w ≥ 3`; estrato truncado medido e reportado como resultado | O número de manchete cobre menos que o plano sugeria — mitigado por reportar os dois |
| J | MAPE de θ | θ é sorteado até 0,05·T_dom, então o MAPE é dominado pelos menores denominadores e mede o piso de sorteio | Erro normalizado `\|θ̂−θ\|/T_dom`; MAPE(θ) segue reportado ao lado. O próprio PLANO §3 já parametriza o Estágio C como `θ/T` | Viés absoluto grande em sistema muito lento passaria despercebido — mitigado pelo MAPE ao lado |
| K | `T_dom = 1/(ζωn)` (era do contrato de execução, não do PLANO) | Errado para ζ>1: a constante é a do polo lento. Para ζ=3, ωn=1 dava 0,333 s contra 5,83 s reais | `T_dom = (ζ+√(ζ²−1))/ωn` para ζ>1 | Janelas longas em ζ alto deslocam a distribuição visual |
| L | — (convenção de SNR não estava definida) | Dois agentes poderiam escolher convenções diferentes | Potência do sinal = **variância**, não média quadrática | Nenhum — é a convenção mais severa |
| N | 1.2 com MAPE < 5% em ωn e ζ | Não-identificabilidade prática comprovada em ζ≥1,6 | ωn/ζ assertados só em ζ<1,6; acima disso asserta-se K, polo lento e NRMSE de reconstrução | Se a não-identificabilidade fosse na verdade falha do estimador — descartado pela auditoria oracle-start (§3.6) |
| O | 1.4 com \|ρ\| < 0,05 (n=300) | Erro padrão de ρ com n=300 é 0,058, e a estatística é o **máximo sobre ~154 pares**. Sob independência perfeita, até \|ρ\|<0,20 é excedido em 60% das réplicas | Teste de permutação (p > 1e-3), que constrói o nulo correto para o máximo, mais round-trip exato do bloco `render` | Nenhum — o teste de permutação é estritamente mais poderoso |
| H→P | 1.5 por distância ponto→polilinha | A distância bruta é dominada pela meia-largura do traço (até 4 px), não pela calibração | Offset normal assinado, mais **controle negativo** (3 px de erro na afim ⇒ 13,5× de piora) | Deslocamento tangente à curva passaria — mitigado pela assertiva de viés no estrato sólido |
| I→R→S | 1.5 com fração mínima de pixels na máscara | Três denominadores tentados; os dois primeiros colidem com tetos físicos (ver §3.4 item 5) | Cobertura horizontal ≥ 0,93 da **projeção de `t_window`**, cujo teto é 1,0 por construção | O 0,93 é empírico, não derivado — reprovação futura deve ser lida primeiro como cauda de estilo |
| Q | `N_DATASET` = 300 | Subgrupo ζ<1,6 do 1.2 ficava com n=23 | 600 | Suíte ~10 s mais lenta |

Duas observações sobre esta tabela, que valem para a monografia:

- **O padrão dos erros é sempre o mesmo:** limiar fixado sem calcular antes o
  **máximo atingível** da grandeza medida. Aconteceu em 1.4 (ruído amostral do
  máximo), em 1.5 (espessura do traço) e duas vezes na sanidade da máscara (teto
  de tinta, depois teto de margens). Vale como seção de metodologia.
- **Três desses erros eram do plano de execução, não do PLANO.md**, e foram
  encontrados pelos próprios agentes implementadores medindo — inclusive o `T_dom`
  (K), que corrompia sistematicamente toda a população superamortecida sem que
  nenhum teste acusasse até alguém medir `y[-1]/K` por faixa de ζ.

---

## 7. Regras que precisam sobreviver ao handoff

1. **`img.py` é evidência, não código.** É o gerador legado defeituoso,
   preservado de propósito para a monografia (desenhava `axhline(K)`,
   `axvline(θ)` e coloria por tipo de amortecimento — por isso "acertava" 93%).
   **Não editar, não apagar, não importar.**
2. **`reports/*.md` são gerados.** Não editar à mão; regenerar com `pytest`.
3. **A suíte mede a realidade; ela não existe para passar.** Se um critério
   falhar, o número real vai para a tabela mestre — nunca se ajusta o limiar para
   passar. Quatro limiares deste projeto já se mostraram mal calibrados e foram
   corrigidos **com medição**, não com afrouxamento (Rulings O, R, S, mais as
   três condições do 1.4e — §3.5.1).
   Corolário do teste de mutação: **um critério que nunca falha em nenhum
   mutante não é um critério, é decoração.** Ao criar um teste novo, meça
   também o seu poder — contra um defeito injetado de propósito.
4. **Nunca `np.random` global**, nunca `time`/`uuid`/hash dependente de
   `PYTHONHASHSEED` — quebram o determinismo bit-a-bit.
5. **Paralelize com processos, nunca threads:** `identify.classical._estimate_gain`
   é memoizado em estado global de módulo.

---

## 8. Pontos abertos

- **FEITO: teste de mutação da suíte** (§3.5.1). 17/17 mutantes detectados após
  três correções na suíte.
- **PENDENTE: revisão final ampla da Parte 1** como entregável único. As três
  revisões por tarefa foram feitas; a revisão de conjunto não.
- **O critério 1.4d não fala sobre 25% das amostras** (as de traço fino ou curva
  muito recortada, onde não há pixel de interior puro). Isso não abre brecha para
  um vazamento de cor — um vazamento por `order` atinge todas as amostras e
  apareceria nas 900 assertadas —, mas abre para um vazamento que só existisse
  no estrato de traço fino. Improvável a ponto de não valer mais complexidade;
  fica registrado.
- **O critério 1.4e não fala sobre amostras com grade** (51% delas), porque o
  `render` declara `has_grid` mas não quantas linhas a grade tem. Fechar isso
  exigiria o gerador registrar as posições das linhas de grade no meta — mudança
  de contrato, não de teste. Registrado como limite conhecido.
- **Minor diferidos, nenhum bloqueante:** piso de `line_width` dependente do dpi
  empilha ~40% das amostras no piso em dpi 60–70 (julgado justificado — abaixo
  disso a máscara fura no limiar); `render_sample` tem ~158 linhas fazendo 4
  coisas; `render_sample` chamada na forma do contrato deixa `meta["seed"]` nulo
  (use `generate_sample`/`generate_dataset`).
- **Mensagem esporádica de LAPACK** (`DLASCL parameter 4`) vista uma vez em lote
  de 3000 séries, não reproduzida isoladamente — provável artefato de BLAS.
- **Igualdade de bytes depende do ambiente pinado.** O critério 1.6 compara
  sha256 de PNG; vale para matplotlib 3.11.1 / pillow 12.3.0. Outro ambiente pode
  reprovar 1.6 sem que o gerador tenha erro.
- **Subgrupos finos do portão.** 1.2b tem n = 32 e 1.2 com ζ < 1,6 tem n = 44.
  Decisão tomada: aceitar, porque a população dedicada do 1.2c cobre os mesmos
  subgrupos com n = 134 e n = 122 e confirma os valores.
- **O limiar de 0,93 do Ruling S é empírico, não derivado.** O canto analítico do
  espaço de estilos (figura estreita × dpi alto × traço grosso × pontilhado
  terminando em vão) admite déficits maiores que os 0,052 observados. Uma
  reprovação futura deve ser lida primeiro como cauda de estilo, não como defeito
  do gerador.
- **`tests/test_part1.py` importa `dataset.generator._apply_noise`** (privado),
  de propósito: reimplementar o ruído no teste duplicaria a convenção do Ruling L
  e correria o risco de divergir do pipeline. Se incomodar, promova a função.
- **`torch` ainda não está instalado** — será necessário na Parte 2.

---

## 9. Onde retomar

Na ordem:

1. Rode `pytest -q` para confirmar 33/33 e regenerar o relatório.
2. Confira a tabela de decisões da §6. Elas foram tomadas com medição, mas quem
   assina o trabalho precisa concordar com cada uma, porque várias mudam a forma
   de um critério do PLANO.
3. Leia `reports/part1_metrics.md` inteiro — ele é a linha de base contra a qual
   a Parte 2 vai medir degradação, e as tabelas dele já estão no formato da
   monografia.
4. Atualize o PLANO.md ou escreva o relatório do Ciclo 4 registrando os quatro
   pontos da §7 do PLANO, mais os achados desta execução: o `T_dom`, a
   não-identificabilidade prática medida, o registro dos três denominadores e o
   **teste de mutação** (§3.5.1) — este último é o que autoriza citar os demais
   números, porque é o que mostra que a suíte que os produziu tem dentes.
5. Só então comece a Parte 2: instale `torch`, gere o dataset de treino
   (`.venv/bin/python -m dataset.generator data/train 6000 0`) e ataque o
   **Estágio B** (determinístico, sem treino, feedback rápido) antes do
   Estágio A.
