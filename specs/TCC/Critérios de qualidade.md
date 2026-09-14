---
tags: [tcc, criterios, metricas]
aliases: [Critérios, Métricas de qualidade]
---

# Critérios de qualidade

> Como cada estágio é medido, **por que** com essa métrica, e de onde vem cada limiar.

O relatório é gerado automaticamente por `tests/part2/` (`pytest_sessionfinish` em `conftest.py`) em `reports/part2_strata.md`. **Não editar à mão.**

Estado atual: **69 testes, 0 falhas.**

---

## O princípio: o limiar vem do orçamento, não do resultado

Um limiar escolhido a partir do que foi medido não reprova nada — ele apenas descreve o presente. Os limiares deste projeto saem do **orçamento do critério a jusante**.

Exemplo canônico: `PERP_MED_MAX = 1.0 px` vem do orçamento do critério 2.6 (3 p.p. de degradação end-to-end). Com 0,800 px de erro perpendicular, a contribuição da rede ao erro de ζ é **+0,127 p.p.** — cerca de 4 % do orçamento. 1,0 px a mantém em ~5 %, com 25 % de folga sobre o medido: apertado o bastante para derrubar uma regressão real, largo o bastante para não reprovar ruído.

---

## 2.1 — o portão do Estágio A

**Alvo:** RMSE mediano ≤ 1,0 px, p95 ≤ 2,0 px · **Medido:** 0,799 px / 1,703 px (n=300) ✅

A distância **euclidiana mínima** de cada ponto da polilinha à curva verdadeira densificada, em coordenadas de pixel.

```python
s = np.linspace(float(cx.min()), float(cx.max()), max(4000, cx.size * 8))
cyi = np.interp(s, cx, cy)
for i in range(0, px.size, 128):
    dx = s[None, :] - px[i:i + 128, None]
    dy = cyi[None, :] - py[i:i + 128, None]
    e[i:i + 128] = np.sqrt((dx * dx + dy * dy).min(axis=1))
```

Em blocos de 128 para não materializar uma matriz de 630 × 4000 de uma vez.

### Por que não IoU

Esta é a decisão mais importante do bloco de métricas. Numa curva fina, a área é dominada pela **espessura do traço**, não pela posição dele:

| Correlação de Spearman | valor |
|---|---|
| IoU × tinta por coluna | **+0,860** |
| IoU × razão de espessura | **−0,879** |
| IoU × deslocamento geométrico real | +0,284 |

E a evidência que fecha o caso: em **todas** as faixas de espessura o erro de linha central ficou **constante em 1,00 px** enquanto o IoU variava de **0,468 a 0,782**. A métrica estava lendo a largura de linha que o gerador sorteia.

**Dice não ajudaria.** É `2·IoU/(1+IoU)` por identidade exata — mesma ordenação, mesma patologia.

### Por que perpendicular e não vertical

Num trecho de inclinação `m`, meio pixel de erro geométrico aparece como `m/2` px de erro vertical. A métrica vertical responde à **declividade do render**:

| Correlação com a inclinação | valor |
|---|---|
| erro vertical | **+0,869** |
| erro perpendicular | +0,326 |

### O ponto cego, declarado no código

> A distância perpendicular **não penaliza erro AO LONGO da curva**. Uma polilinha deslocada no tempo, mas *sobre* a curva, pontua zero.

Num degrau isso é exatamente o **θ** — e é por isso que `2.6[theta]` (+0,23 p.p.) é obrigatório ao lado desta métrica. Uma métrica com ponto cego declarado e coberto por outra é honesta; uma sem, não.

---

## 2.2 — o piso do extrator de polilinha

**Alvo:** RMSE ≤ 1,0 px, p95 ≤ 2,0 px · **Medido:** 0,615 px / 1,137 px (n=300) ✅

A mesma métrica, mas rodando `mask_to_polyline` sobre a máscara **ground-truth**.

**Para que serve.** Separa o erro da **rede** do erro do **extrator de polilinha**. Dos 0,799 px do 2.1, **0,615 px já existiriam com uma máscara perfeita**. A rede contribui com a diferença.

Sem esse piso, uma regressão no estágio C seria atribuída ao estágio A — e o retreino não resolveria nada.

O número vertical continua reportado como diagnóstico, sem alvo: RMSE 1,49 px, p95 6,70 px. A diferença entre 6,70 e 1,137 no p95 é a métrica antiga medindo declividade.

### A investigação que precedeu

Dezenove tentativas de melhorar a redução coluna→ponto ficaram **todas piores** que a atual, e um extrator oráculo que passa a métrica antiga **não recupera acurácia significativa**. Conclusão: o erro que o critério antigo penalizava **não existia**.

---

## 2.7 — a varredura por estrato

O mesmo limiar de 1,0 px aplicado **separadamente** a cada estrato. Uma mediana agregada esconde um estrato que desmorona.

| Estrato | erro perpendicular | IoU (diagnóstico) |
|---|---|---|
| `grade=False` | 0,792 px | 0,6423 |
| `grade=True` | 0,801 px | 0,6579 |
| `legenda=False` | 0,781 px | 0,6774 |
| `legenda=True` | 0,810 px | 0,6147 |
| `fundo_escuro=False` | 0,786 px | 0,6244 |
| `fundo_escuro=True` | 0,810 px | 0,6738 |
| `traco=-` | 0,684 px | 0,7064 |
| `traco=--` | 0,756 px | 0,6727 |
| `traco=-.` | 0,827 px | 0,6158 |
| **`traco=:`** | **0,981 px** | **0,5264** |

**O estrato de risco é o pontilhado**, a 2 % do teto. E repare que ele é também o de menor IoU — coerente com a métrica ler espessura, já que um traço pontilhado tem menos tinta por definição.

---

## 2.6 — degradação end-to-end

**Alvo:** ≤ 3 p.p. no pior parâmetro · **Medido:** +1,67 p.p. (n=240) ✅

Compara o MAPE do caminho real contra um **oráculo** que recebe a série verdadeira. A diferença é o que a extração custa.

| Parâmetro | oráculo | real | Δ |
|---|---|---|---|
| K | 0,12 % | 0,29 % | +0,16 p.p. |
| τ | 0,23 % | 0,44 % | +0,21 p.p. |
| θ | 0,06 % | 0,29 % | +0,23 p.p. |
| ωn | 0,73 % | 1,80 % | +1,06 p.p. |
| **ζ** | 1,17 % | 2,85 % | **+1,67 p.p.** |

**Este é o critério do qual todos os outros limiares derivam.** O orçamento de 3 p.p. é o que torna "1,0 px" um número e não um palpite.

A versão adimensional (`2.6-adim`) mede ζ pelo mesmo caminho, dispensando calibração: +1,66 p.p. — praticamente idêntico, o que confirma que ζ não depende de eixo.

---

## Calibração — 2.3, 2.5, 2.9

| Critério | Alvo | Medido |
|---|---|---|
| 2.3 · erro relativo de `sx`, `sy` | < 1 % em ≥ 95 % | ✅ |
| 2.5 · recusas corretas | — | 0,900 ✅ |
| 2.9 · cobertura da calibração | — | 0,933 ✅ |

> [!note] Os dois limiares tiveram de ser unificados
> Antes o 2.3 usava 1 % e o 2.5 usava 5 %, e as amostras na faixa intermediária eram **simultaneamente** "não deviam ter sido rejeitadas" (2.5) e "ruins o bastante para estragar o 2.3" — nenhum subconjunto satisfazia os dois.
>
> Escolhido 1 %, o valor **fisicamente motivado**: o erro de escala propaga direto para K e τ, e 5 % consumiria sozinho quase o dobro do orçamento do 2.6. A constante mora em `ESCALA_TOL`, não literal nos dois testes — o alinhamento é garantido por construção, não por coincidência de valor.

**A progressão da cobertura**, que é o resultado do bloco de calibração:

| Estado | ok | falso positivo |
|---|---|---|
| início | 79,56 % (716) | 55 |
| + conserto dos blobs | 91,78 % | — |
| + teto do lote de OCR | **93,00 %** | **20** |

Cobertura **e** precisão subindo juntas — o que nenhum ajuste de limiar tinha conseguido, porque o problema não era o limiar.

---

## 2.11 — o contrato da saída em dois níveis

**Alvo:** 100 % das amostras, sem exceção · **Medido:** 300/300 com bloco, 292/300 com valor, 20/20 das sem calibração ✅

Assevera a [[Decisões de projeto#D1 · Saída em dois níveis|Decisão E]]: `dimensionless` existe **sempre**, mesmo quando não há nada a preencher. As chaves existem, os valores são `None`. Nunca ausente, nunca exceção.

---

## Latência

| Critério | Alvo | Medido |
|---|---|---|
| 2.8 · pipeline completa | < 500 ms | mediana 160 ms, p95 298 ms ✅ |
| G3b.4 · extrator clássico | < 200 ms | mediana 12,2 ms, p95 29,0 ms ✅ |

> [!warning] Não remedido com o `base=32`
> O modelo cresceu **78 %** desde essa medição. O critério 3.11 da Parte 3 pede < 2 s em CPU, e nada disso foi verificado com o checkpoint atual.

---

## Diagnósticos sem alvo

Reportados para preservar comparabilidade histórica, sem veredito:

| Critério | Medido |
|---|---|
| 2.1-iou · IoU da máscara | 0,6482 |
| 2.10 · IoU U-Net vs. clássico | U-Net 0,6482 · clássico **0,7153** |
| 2.12-ordem · acerto de ordem | 89,0 % (267/300) |
| 2.12-ordem[sem-calib] | **100,0 %** (20/20) |
| G3b.1 · IoU do extrator clássico | 0,7153 |

---

## O 2.10 mede pela métrica que o projeto demitiu

O docstring do teste diz que ele é *"o resultado que justifica (ou não) a U-Net"*. Mas ele compara **só por IoU**:

```python
ious_unet.append(float(np.logical_and(p_unet, alvo).sum()) /
                 max(float(np.logical_or(p_unet, alvo).sum()), 1.0))
ious_classico.append(...)
med_unet, med_classico = float(np.median(ious_unet)), float(np.median(ious_classico))
record_p2("2.10", ..., "sem alvo", f"U-Net={med_unet:.4f}  clássico={med_classico:.4f}", None)
```

O clássico "ganha" (0,7153 × 0,6482) porque produz máscaras **mais grossas** — exatamente o que o Ruling 50 estabeleceu que o IoU mede. A comparação que decidiria a questão — **erro perpendicular** dos dois extratores no mesmo conjunto — não é calculada em lugar nenhum.

É a mesma classe de erro que o 2.1 corrigiu, sobrevivendo num critério vizinho. **Dívida aberta.**

---

## O critério A.0 não olha o modelo em produção

```python
def test_unet_tamanho_declarado():
    n = sum(p.numel() for p in UNet().parameters())    # base=16, in_ch=1 — os DEFAULTS
    record_p2("A.0", "Parâmetros da U-Net", "~1,2 M (PLANO)", f"{n/1e6:.2f} M", None)
    assert 0.5e6 <= n <= 2.5e6
```

Ele instancia os *defaults* do construtor, **não carrega o checkpoint**. Por isso o relatório mostra `A.0 … 1.94 M` enquanto o modelo promovido tem **7,76 M**.

Duas consequências:

1. Se o teste lesse o checkpoint de verdade, **falharia** — 7,76 M estoura o teto de 2,5 M.
2. O alvo declarado é "~1,2 M (PLANO)" contra 1,94 M medido, e como o veredito é `None`, isso nunca reprovou.

O único teste que mede o tamanho do modelo não mede o modelo que é enviado. **Dívida aberta.**

---

## O IoU ainda governa o treino

`train_unet.py` usa IoU de validação para **selecionar o checkpoint** e alimentar o `ReduceLROnPlateau`. A crítica do Ruling 50 vale aqui também: uma época que engrossa o traço ganha do scheduler.

Duas razões para a escolha: é barato (por lote, na GPU) e só precisa ser aproximadamente monótono para seleção de checkpoint. Mas é uma limitação do procedimento, não uma decisão defendida — e vale registrar que o ganho de `base=24` para `base=32` em IoU de validação foi de **+0,005**, enquanto o ganho real (acertar as duas imagens reais ao mesmo tempo) foi categórico.

---

Ver também: [[Arquitetura do projeto]] · [[Decisões de projeto]] · [[train_unet]] · [[identify.extract]]
