# HANDOFF_P2_3b — Bloco 3b: Estágio A sem rede (extrator clássico)

**Este extrator é baseline e contingência, não a solução.** Ele existe para
tirar a GPU do caminho crítico (§1.8, Papel 1) e para dar à U-Net do Bloco 3
um número para justificar sua existência (§1.8, Papel 2). Se ele algum dia
empatar com a U-Net no critério 2.10, o achado correto é "a U-Net não se
justifica neste dataset" — não um problema a esconder.

## 1. Estado

| Componente | Estado | Evidência |
|---|---|---|
| `identify/extract_classical.py::extract_mask_classical` | pronto | G3b.1 = 0,7153 ≥ 0,70 |
| G3b.2 (sem reta de span completo na saída) | pronto | 0 violações em 300 |
| G3b.3 (não importa `torch`) | pronto, testado via subprocess com `torch` bloqueado de verdade | PASSED |
| G3b.4 (latência) | pronto | p95 = 128,5 ms < 200 ms |

## 2. Interface publicada

```python
def extract_mask_classical(image_rgb: np.ndarray,
                           bbox: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """uint8 0/255, mesma resolução de image_rgb. Nunca levanta exceção."""
```

Mesma assinatura de saída que `identify.extract.predict_mask` (Bloco 3) — é
o que permite os dois serem intercambiáveis em `identify/pipeline.py` no
Bloco 5, parametrizados por `extractor=...`.

Constantes calibradas em `identify/extract_classical.py`:
`QUANT=32`, `MIN_MODE_FRAC=0.001`, `BBOX_PAD=4`, `MIN_CURVE_PX=15`,
`MIN_EXTENT_FRAC=0.30`, `DASH_BRIDGE=25`; e, copiadas de
`tests/test_leakage.py::_spanning_rows` sem alteração: `INK_BG_TOL=12`,
`SPAN_FRAC=0.98`, `MIN_INK_FRAC=0.25`, `SPAN_BINS=8`, `SPAN_MIN_BINS=7`.

## 3. Números medidos

### G3b.1 — IoU mediana (≥ 0,70)

**Medido: 0,7153** (mediana, n=300) — ✅, com margem de 1,5 p.p.

Por estrato (diagnóstico, sem alvo — usado no critério 2.10 do Bloco 5):

| Estrato | IoU mediana | n |
|---|---|---|
| `fundo_escuro=False` | 0,6799 | 173 |
| `fundo_escuro=True` | 0,7382 | 127 |
| `grade=False` | 0,6928 | 150 |
| `grade=True` | 0,7301 | 150 |
| `n_distractors=1` | 0,7313 | 116 |
| `n_distractors=2` | 0,7153 | 92 |
| `n_distractors=3` | 0,6727 | 92 |

Leitura esperada e confirmada: o método piora com **mais distratoras** (mais
chance de uma reta pontilhada escapar da rejeição) — é exatamente onde a
Parte 1 previu que o método clássico cairia (§1.8) e a U-Net deveria ganhar.
Curiosamente ele **não** piora com grade — a rejeição de retas trata bem
linhas de grade (que raramente são muito pontilhadas com ciclo de trabalho
baixo, ao contrário de algumas distratoras).

### G3b.2 — Sem reta de span completo (0 violações)

**Medido: 0/300.** ✅

### G3b.3 — Não importa torch

**Medido: import bem-sucedido com `torch` bloqueado via `sys.meta_path` em
subprocess isolado.** ✅ (ver Armadilha 1 — por que subprocess e não leitura
de código).

### G3b.4 — Latência (< 200 ms)

**Medido: mediana 55,2 ms, p95 128,5 ms.** ✅. Ver Ruling 2 — a primeira
implementação estava em 1,9 s/amostra e 204 ms de p95 antes da otimização.

## 4. Rulings

1. **A escolha do candidato a curva não pode ser por maior EXTENSÃO
   horizontal isolada — precisa ser por ÁREA entre os candidatos que já
   cobrem uma fração mínima da largura.** Medido: com desempate só por
   extensão, uma reta distratora muito pontilhada (ciclo de trabalho baixo o
   bastante para escapar da rejeição de span, que exige ocupação ≥ 25% por
   linha) frequentemente tinha extensão marginalmente MAIOR que a curva de
   verdade (ex.: amostra `sample_00001`: modo da curva com 5097 px e
   extensão 1031 px perdia para um modo de 989 px e extensão 1075 px — a
   distratora "ganhava" por 44 px de extensão apesar de ter 5× menos tinta).
   Isso derrubava a mediana de IoU para **0,0** (mais da metade das 300
   amostras com interseção zero com a máscara verdadeira). Corrigido: entre
   candidatos com extensão ≥ 30% da largura útil, o desempate é por área
   (quantidade de tinta), não por extensão. **Medido, antes → depois:**
   mediana 0,0 → 0,708.
2. **Precisou de uma segunda correção: "fechar vãos" antes de decidir se uma
   linha/coluna é uma reta de span completo.** Mesmo com o desempate por
   área, ~14% das amostras ainda escolhiam a distratora pontilhada: seu
   ciclo de trabalho era baixo o bastante (< 25% de ocupação por linha) para
   nunca ser marcada como "span completo" pela função `_spanning_rows`
   (copiada da Parte 1) — ela sobrevivia inteira, sem nenhuma linha removida,
   e com isso `area` E `extensão` favoreciam ela (era mais "sólida" que
   restos fragmentados de outros modos). Corrigido com `_bridge_gaps_1d`
   (dilatação morfológica 1D, só para a DECISÃO de span, nunca aplicada à
   máscara de saída): fecha vãos de até `DASH_BRIDGE` px antes de checar
   `_spanning_rows`, então uma reta pontilhada "parece" contínua o bastante
   para ser rejeitada, mas a curva (que não é uma reta) continua livre.
   Varredura de `DASH_BRIDGE ∈ {5, 9, 15, 25}`: mediana idêntica em todos
   (0,7153 — a mediana já estava dominada por casos que o *bridging* não
   muda), mas o **p10 sobe de 0,068 para 0,384** — o parâmetro melhora
   consistentemente o pior caso, sem mexer no meio da distribuição. Escolhido
   `DASH_BRIDGE = 25`.
3. **A primeira implementação de `_color_modes`/cor de fundo usava
   `np.unique(flat, axis=0, ...)` sobre tuplas RGB — 1,9 s por amostra**,
   quase 10× o orçamento de latência do critério G3b.4 (< 200 ms). O custo
   é o `sort` por tupla dentro de `np.unique(axis=0)`, que não escala para
   ~1,9 M pixels (1600×1200). Reescrito para codificar RGB quantizado como um
   único inteiro (`_bucket_key`, 8³=512 baldes possíveis com `QUANT=32`) e
   usar `np.bincount`, que é O(n) sem comparação de tupla. **Medido:
   1,93 s → 0,07 s por amostra** (única imagem) — a mesma ideia depois
   aplicada ao `_bridge_gaps_1d` (troca de um laço Python de deslocamentos
   por `cv2.dilate` com elemento estruturante 1D) trouxe o p95 de volta de
   203,8 ms (falhando por 3,8 ms) para 128,5 ms.
4. **`_spanning_rows` foi COPIADO de `tests/test_leakage.py`, não
   importado**, apesar do texto do Bloco 3b dizer "reuse a implementação,
   não reescreva". Motivo: `tests/test_leakage.py` importa `pytest`,
   `sklearn.ensemble.GradientBoostingClassifier` e, no topo do módulo,
   `tests.conftest` inteiro (52 KB, com efeitos de import pesados) — nenhum
   desses é apropriado como dependência de produção de um módulo cujo ponto
   inteiro é ser leve e não precisar de `torch` nem de infraestrutura de
   teste. A cópia preserva a assinatura, a lógica e as constantes
   **exatamente** (nada foi "reescrito" no sentido que o plano queria evitar
   — só o mecanismo de reuso mudou de `import` para cópia literal com
   atribuição na docstring). Risco assumido: se `_spanning_rows` for
   corrigido na Parte 1 no futuro, esta cópia não acompanha automaticamente
   — registrado aqui para quem for mexer lá.

## 5. Armadilhas

1. **G3b.3 teria passado com um `import torch` escondido se testado por
   leitura de código.** O teste usa `subprocess` com um `sys.meta_path`
   customizado que levanta `ImportError` para qualquer `import torch`, rodado
   num interpretador Python **novo** (não o processo do pytest, que talvez já
   tenha `torch` carregado de outro teste da sessão) — é a única forma
   confiável, e o próprio `PLANO_PARTE2.md` avisa disso.
2. **O modo de falha residual (IoU baixo em ~10% das amostras) está
   concentrado em duas causas, ambas já diagnosticadas em blocos
   anteriores, não novas daqui:**
   - estilo de linha muito pontilhado (`:`) com ciclo de trabalho baixo o
     bastante para escapar até do `DASH_BRIDGE=25`;
   - a mesma armadilha de contraste do Bloco 1 (`sample_00077`: par de cor
     com luma BT.601 parecida apesar de contraste WCAG garantido) — quando
     `detect_plot_bbox` falha, `extract_mask_classical` devolve máscara
     vazia por construção (`bbox is None -> return out` vazio), então o erro
     se propaga em cascata do Bloco 1 para aqui. Não há nada a corrigir
     neste módulo — é o mesmo par de cores já registrado como limite
     conhecido.
3. **Não confundir `MIN_INK_FRAC` (copiado, 0,25) com `DASH_BRIDGE` (novo,
   25) só porque os números coincidem numericamente por acaso** — são
   parâmetros de naturezas diferentes (fração vs. pixels) que não têm
   relação entre si; a coincidência numérica é só isso, coincidência.

## 6. O que o próximo bloco precisa saber

1. **`extract_mask_classical` está pronto e cumpre a assinatura de
   `predict_mask`** — o Bloco 5 pode usá-lo como `extractor=` alternativo à
   U-Net sem mudança de interface.
2. **O critério 2.10 (U-Net × extrator clássico) já tem o lado clássico
   medido: IoU mediana 0,7153, com a tabela por estrato acima.** Quando o
   Bloco 3 terminar o treino, o Bloco 5 só precisa rodar a mesma bateria
   sobre `predict_mask` e comparar.
3. **A GPU está oficialmente fora do caminho crítico da Parte 2 a partir
   daqui** — mesmo que o Bloco 3 (treino da U-Net) não termine a tempo, a
   Parte 2 fecha com este extrator. O Bloco 3 virou melhoria mensurável, não
   pré-requisito (exatamente como o §1.8 previu).
4. `reports/part2_strata.md` já tem as linhas G3b.1 a G3b.4 (a sessão que
   gerou os números acima rodou só `-k g3b`; a versão final e completa do
   relatório é escrita no Bloco 5, quando a suíte inteira roda de uma vez).
5. Próximo passo: continuar o Bloco 3 (U-Net, já com o treino disparado em
   segundo plano — ver `HANDOFF_P2_3.md`) e o Bloco 2 (OCR/RANSAC).
