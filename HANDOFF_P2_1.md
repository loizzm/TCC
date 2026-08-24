# HANDOFF_P2_1 — Bloco 1: Estágio B, geometria (moldura e ticks, sem OCR)

## 1. Estado

| Componente | Estado | Evidência |
|---|---|---|
| `identify/calibrate.py::detect_plot_bbox` | pronto | G1.1 = 0,997; G1.3 ≥ 0,983 em todos os estratos |
| `identify/calibrate.py::detect_tick_pixels` | pronto | G1.2 = 1,000 (mediana) em x e y |
| `tests/part2/conftest.py` | pronto (fixture `test_samples`, `record_p2`, `to_gray`, geração de `reports/part2_strata.md`) | `pytest tests/part2 -q` roda sem erro |
| `tests/part2/test_part2.py` | G1.1, G1.2, G1.3 escritos e verdes | 8 passed |
| Suíte de mutação (P2-M01 a P2-M04) | 4/4 corretos (3 mutantes detectados, 1 controle não detectou nada) | ver §3 |

## 2. Interface publicada

```python
def detect_plot_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1), mesma convenção de meta["plot_bbox_px"]. None se
    nem o spine esquerdo nem o inferior forem achados."""

def detect_tick_pixels(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, list[float]]:
    """{"x": [px, ...], "y": [px, ...]}, coordenadas contínuas, origem no topo."""
```

Constantes calibradas (todas em `identify/calibrate.py`, ver §3 para as varreduras):
`INK_THR = 12.0`, `SPINE_COVER = 0.55`, `SPINE_MIN_FILL = 0.90`, `MIN_BBOX_PX = 8`,
`TICK_BAND = 6`, `TICK_PROM = 0.25`.

## 3. Números medidos

### G1.1 — Erro da moldura (≤ 2 px em ≥ 95%)

Varredura de `(INK_THR, SPINE_COVER, SPINE_MIN_FILL)` sobre as 300 amostras de
`data/test`. A varredura completa está registrada abaixo; **duas gerações de
algoritmo foram medidas e descartadas antes da terceira funcionar** — ver §5.

Varredura final (fração dentro de 2 px):

| INK_THR | SPINE_COVER | SPINE_MIN_FILL | frac ≤ 2px | none_rate |
|---|---|---|---|---|
| 12 | 0,50 | 0,80 | 0,997 | 0,003 |
| 12 | 0,50 | 0,90 | **0,997** | 0,003 |
| 12 | 0,50 | 0,95 | 0,987 | 0,010 |
| 12 | 0,55 | 0,90 | 0,997 | 0,003 |
| 12 | 0,60 | 0,90 | 0,997 | 0,003 |
| 18 | 0,55 | 0,90 | 0,990 | 0,010 |
| 25 | 0,55 | 0,90 | 0,987 | 0,013 |

Escolhido: **`INK_THR=12, SPINE_COVER=0.55, SPINE_MIN_FILL=0.90`** (margem
folgada acima de 0,95; `SPINE_COVER` não é sensível nessa faixa, mantido em
0,55 — meio da faixa testada).

**Resultado: 0,997 (299/300) ≥ 0,95 → ✅.**

### G1.2 — Recall de ticks (≥ 0,95 mediana, por eixo)

Varredura de `(TICK_BAND ∈ {4,6,8,12}, TICK_PROM ∈ {0,15; 0,25; 0,40})`:
**recall mediano = 1,000 em todas as 12 combinações**, em x e em y — o portão
é completamente insensível a esses dois parâmetros no regime testado (ticks
maiores do gerador são picos bem destacados na faixa periférica). Mantidos os
valores do plano (`TICK_BAND=6`, `TICK_PROM=0.25`) por não haver diferença
medida que justifique outro valor.

**Resultado: G1.2x = 1,000, G1.2y = 1,000 ≥ 0,95 → ✅.**

### G1.3 — Estratificação por `n_spines` (nenhum estrato < 0,90)

| n_spines | n | frac ≤ 2px |
|---|---|---|
| 2 | 85 | 1,000 |
| 3 | 156 | 1,000 |
| 4 | 59 | 0,983 |

**Resultado: mínimo 0,983 ≥ 0,90 → ✅.**

### Suíte de mutação

| Mutante | Substituição | Esperado | Observado |
|---|---|---|---|
| P2-M01 | `x1, y0 = x1_row, y0_col` → `x1, y0 = w - 1, 0` (ignora a extensão medida, sempre usa a borda da imagem) | G1.1 e G1.3 reprovam | ✅ G1.1 e G1.3 (n_spines=4) reprovaram |
| P2-M02 | `SPINE_MIN_FILL = 0.90` → `0.05` (aceita quase qualquer coisa como "spine", inclusive cruzamentos da curva) | G1.1 reprova | ✅ G1.1, G1.2 e G1.3 reprovaram (efeito em cascata: bbox errado quebra a busca de ticks também) |
| P2-M03 | banda de busca de ticks em x deslocada para fora da região real (`y1+1+TICK_BAND` a `y1+1+2*TICK_BAND` em vez de `y1+1` a `y1+1+TICK_BAND`) | G1.2 reprova | ✅ G1.2 reprovou |
| P2-M04 (controle) | tolerância de casamento de tick no teste, `≤ 3.0` → `≤ 50.0` | nada deve reprovar | ✅ nada reprovou |

## 4. Rulings

1. **A implementação de `detect_plot_bbox` não segue o esqueleto literal do
   Passo 4 do `PLANO_PARTE2.md` (`_edges` + `_long_lines` por gradiente).**
   Esse esqueleto mede coverage por **borda** (gradiente), o que só encontra o
   spine no lado onde ele existe — falha estruturalmente para `x1`/`y0` quando
   o lado correspondente (`right`/`top`) está ausente, porque nesse caso não
   há *nenhum* traço na borda perpendicular a detectar (a área de dados tem a
   mesma cor de fundo que a figura — `dataset/generator.py:200-202` — logo
   não há descontinuidade de cor a explorar). **Medido:** essa versão literal
   dava 13,3% de acerto em G1.1, estável entre `EDGE_Q ∈ {0,85; 0,90; 0,95}`
   e `SPINE_COVER ∈ {0,40 … 0,85}` — nenhuma combinação passava de 20%.
   **Causa raiz, medida diretamente nos pixels:** quando `right`/`top` estão
   ausentes, o retângulo `plot_bbox_px` de `ax.get_window_extent()` é um
   limite puramente geométrico (fração de `axes_rect`, sorteado independente
   de `left`/`bottom`) sem *nenhum* traço desenhado ali — nem spine, nem
   grade, nem ticks (o matplotlib só desenha ticks em `bottom`/`left` por
   padrão; `dataset/generator.py` não sobrescreve isso). A única informação
   recuperável é indireta: **o próprio traço do lado garantido (spine
   inferior/esquerdo) é recortado pelo matplotlib exatamente no retângulo dos
   eixos**, então a extensão horizontal do spine inferior já é `[x0, x1]`, e a
   extensão vertical do spine esquerdo já é `[y0, y1]` — não é preciso
   detectar nada no lado ausente. Reescrito para:
   varrer de baixo para cima a primeira linha "cheia" (span ≥ 0,55·largura
   **e** preenchimento ≥ 90% dentro do próprio intervalo, para não confundir
   com o cruzamento esparso de uma curva oscilante — ver Armadilha 1) e usar
   a extensão *dela mesma* como `(x0, x1)`; simetricamente para a coluna mais
   à esquerda dar `(y0, y1)`. **Medido: 99,7% (299/300), 98,3% no pior
   estrato (`n_spines=4`)** — a interface pública (`detect_plot_bbox(gray) ->
   (x0,y0,x1,y1)|None`) não mudou, só a implementação interna.
2. **`detect_plot_bbox` opera em tons de cinza (BT.601, conforme a assinatura
   publicada no `PLANO_PARTE2.md`), não em RGB.** Isso tem um custo real: o
   `dataset/randomize.py` garante contraste mínimo entre `axes_color` e
   `bg_color` usando **luminância relativa WCAG sobre sRGB linearizado**
   (`MIN_AXES_CONTRAST = 0.20`), não a luma BT.601 usada aqui. As duas métricas
   podem divergir para pares de cor com matiz muito diferente e luminância
   BT.601 parecida (ex.: magenta claro vs. verde, ambos com luma ≈ 200/255,
   mas contraste WCAG ≥ 0,20 graças à linearização gama) — foi exatamente o
   que aconteceu na única falha observada (`sample_00077`, §5). Manter a
   assinatura em tons de cinza é a decisão certa aqui: ela é o que o Bloco 2
   consome, a falha afeta 1/300 amostras (bem dentro da folga de 2,7 pontos
   percentuais até o alvo de 95%), e nada no plano pede robustez a cor —
   registrado como limite conhecido, não corrigido.

## 5. Armadilhas

1. **A maior armadilha do bloco: um cruzamento de curva pode imitar uma
   "linha longa cheia" se você só olhar o *alcance* (span) da tinta numa
   linha/coluna, sem olhar o *preenchimento*.** Uma resposta ao degrau
   subamortecida (com overshoot) cruza o mesmo nível de `y` duas ou mais
   vezes em `x` bem afastados — isso produz, para dezenas de linhas
   consecutivas, um "alcance" de quase 1000 px com **apenas ~1% de
   preenchimento** (só os 1-2 pixels onde a curva realmente passa). Uma
   primeira versão do detector (span-only, sem exigir preenchimento) dava
   45-70% em G1.1 — pior que a versão ingênua original em alguns casos,
   porque o "alcance" da curva é enganosamente parecido com o de um spine.
   **O preenchimento mínimo (`SPINE_MIN_FILL = 0.90`) é o que separa os dois
   casos** — um traço reto (spine/grade/distratora) tem preenchimento
   próximo de 100% no seu próprio intervalo; uma curva cruzando o nível tem
   perto de 0%. Qualquer implementação futura de detecção geométrica de
   linhas neste dataset **tem que** checar preenchimento, não só alcance —
   registrar isso explicitamente porque não é óbvio até medir.
2. **A falha isolada em G1.1 (`sample_00077`, erro `inf`, nenhuma detecção)**
   é exatamente o caso do Ruling 2: `bg_color = #85ff8d` (verde),
   `axes_color` com luma BT.601 ≈ 199,5 contra fundo ≈ 205,5 (diferença de
   0,024 em fração de 255) — abaixo de `INK_THR=12` em unidades de 0-255
   (equivale a ~0,047, mas a diferença absoluta medida foi 6,0/255 ≈ 0,024,
   portanto abaixo do limiar). WCAG garante contraste ≥ 0,20 na escala
   linearizada; BT.601 não linearizada não herda essa garantia. Não vale a
   pena perseguir esse 1/300 dado a folga do critério — mas se outro executor
   vir G1.1 cair abaixo de 95% no futuro, **olhe primeiro para pares de cor
   com matiz muito diferente e luma BT.601 parecida**, não para os parâmetros
   numéricos.
3. **`n_distractors` nunca é zero** (`dataset/randomize.py:317`,
   `rng.integers(1, 4)` → sempre 1 a 3): isso importa para o Bloco 2, porque
   pelo menos uma reta de referência (`axhline`/`axvline`) sempre existe,
   então há sempre pelo menos um traço extra recortado no retângulo dos
   eixos — mas nenhuma garantia de orientação (`h`/`v` é 50/50 por
   distratora), então não conte com isso para achar `x1`/`y0`
   especificamente; o algoritmo deste bloco não depende disso de qualquer
   forma (usa só os spines garantidos).

## 6. O que o próximo bloco precisa saber

1. **`detect_plot_bbox` e `detect_tick_pixels` estão prontos e testados** com
   as assinaturas exatas do `PLANO_PARTE2.md`. O Bloco 2 (`calibrate`,
   `Calibration`, `px_to_data`) pode importá-las diretamente de
   `identify.calibrate` sem mudança de interface.
2. **`to_gray` (BT.601) é a conversão usada em todo o Bloco 1** e está em
   `tests/part2/conftest.py`; se o Bloco 2 precisar de tons de cinza para
   recortar rótulos de tick antes do OCR, reusar essa mesma função para
   manter consistência com o que os detectores já viram.
3. **O ponto mais frágil herdado para o Bloco 2 é a cor**, não a geometria:
   G1.1/G1.3 já mostraram que o único modo de falha observado em 300 amostras
   foi um par de cores de baixo contraste sob luma BT.601. Se o Bloco 2 tiver
   problemas inesperados de OCR ou RANSAC concentrados num subconjunto
   pequeno de amostras, vale checar `bg_color`/`axes_color`/`line_color`
   antes de mexer em `LABEL_W`/`LABEL_H`/`RANSAC_TOL`.
4. `reports/part2_strata.md` já está sendo gerado por
   `tests/part2/conftest.py::pytest_sessionfinish` (chave nova neste bloco,
   não estava no esqueleto do Passo 1 do plano — necessária porque a tabela
   `Estrutura de arquivos` do `PLANO_PARTE2.md` atribui essa responsabilidade
   ao `conftest.py`). O Bloco 2 só precisa chamar `record_p2(...)`; o
   relatório se atualiza sozinho.
5. Próximo passo: **Bloco 2** — OCR opcional (Tesseract), RANSAC afim,
   consistência interna, critérios 2.3/2.4/2.5/2.9/2.11.
