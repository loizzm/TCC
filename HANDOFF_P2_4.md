# HANDOFF_P2_4 — Bloco 4: Estágio A, pós-processamento até a polilinha

## 1. Estado

| Componente | Estado | Evidência |
|---|---|---|
| `identify/polyline.py::mask_to_polyline` | pronto | mediana 1,49 px ≤ alvo 2 px; p95 6,70 px **acima** do alvo 5 px — ver §3/§4 |
| `identify/polyline.py::polyline_to_series` | pronto (repassa para `px_to_data` do Bloco 2) | usado nos testes de estrato |
| Estratificação por `line_style`, `has_marker`, espessura | pronto, todos os sub-estratos ≤ 2 px | ver §3 |
| Suíte de mutação (P2-M14 a P2-M17) | 2/3 detectados, 1 não detectado e explicado, 1 controle correto | ver §3 |

## 2. Interface publicada

```python
def mask_to_polyline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x_px, y_px), uma amostra por coluna com tinta, vãos interpolados."""

def polyline_to_series(x_px: np.ndarray, y_px: np.ndarray, cal) -> tuple[np.ndarray, np.ndarray]:
    """(t, y) em unidades físicas, via identify.calibrate.px_to_data."""
```

Constantes: `MAX_GAP_FRAC = 0.15` (inalterada — varredura não mostrou efeito,
ver §3), `MIN_COMPONENT_PX = 2` (nova — ver Ruling 1).

## 3. Números medidos

### Critério 2.2 (piso, contra `mask.png` VERDADEIRA)

| Alvo | Medido | Veredito |
|---|---|---|
| RMSE mediana ≤ 2 px | **1,49 px** | ✅ |
| p95 ≤ 5 px | **6,70 px** | ❌ — ver §4, limitação estrutural |

Por estrato (todos ≤ 2 px, `test_2_2_estrato_marcador_e_estilo` passa):

| Estrato | RMSE mediana | n |
|---|---|---|
| `marcador=False` | 1,33 px | 214 |
| `marcador=True` | 1,91 px | 86 |
| `traco=-` | 1,08 px | 94 |
| `traco=--` | 1,52 px | 72 |
| `traco=-.` | 1,61 px | 67 |
| `traco=:` | 1,80 px | 67 |

Comparação com o extrator ingênuo medido na Parte 1 (`HANDOFF.md §4`: 0,19 px
em linha sólida, 0,92 px em pontilhada, **contra a série verdadeira
diretamente**, sem passar por máscara nem por coluna-mediana): os números
deste bloco são maiores porque medem uma etapa a mais do pipeline (máscara
→ esqueleto → coluna-mediana → polilinha, não só "coluna-mediana direta na
série"), então não são diretamente comparáveis número a número — mas a
ordem de grandeza (traço sólido bem melhor que pontilhado) é a mesma, e é o
esperado.

### Varredura de `MAX_GAP_FRAC`

`{0,05; 0,10; 0,15; 0,20; 0,30; 0,50}`: mediana e p95 **idênticos em todos os
valores** (1,488 px / 6,701 px). O parâmetro não tem efeito mensurável neste
regime — motivo, medido: depois do Ruling 1 (união de componentes), os vãos
que sobram entre fragmentos de um traço tracejado/pontilhado são pequenos
(a distância entre pontos do mesmo tracejado), não mais o vão gigante que
existia quando só a maior componente sobrevivia. Mantido em 0,15 (valor do
`PLANO_PARTE2.md`) por não haver medição que justifique outro.

## 4. Rulings

1. **`mask_to_polyline` não pode ficar só com a MAIOR componente conexa —
   union de todas acima de um piso mínimo.** O Passo 3 do `PLANO_PARTE2.md`
   usa `argmax(stats[1:, cv2.CC_STAT_AREA])` para escolher uma única
   componente. Isso quebra estruturalmente para `line_style` tracejado
   (`--`, `-.`, `:`): matplotlib desenha cada travessão/ponto como um traço
   geometricamente desconectado dos vizinhos, então a máscara — mesmo a
   **verdadeira**, sem ruído nenhum — tem o traço partido em dezenas de
   componentes pequenas. Manter só a maior descarta a curva quase inteira.
   **Medido, antes da correção:** 44/300 amostras (14,7%) com menos de 10
   pontos utilizáveis (a mediana de pixels utilizáveis nesses casos era de
   apenas 2 a 9!), todas com `line_style` tracejado/pontilhado; RMSE mediano
   do estrato `traco=:` em 2,43 px (acima do alvo). A mediana geral, com o
   `argmax`, já vinha em 2,01 px — reprovando por pouco. **Depois da
   correção** (união de todas as componentes com área ≥ `MIN_COMPONENT_PX =
   2`, que só descarta ruído de 1 px isolado — nunca existe na máscara
   verdadeira, então essa correção é livre de custo ali): **0 amostras**
   com menos de 10 pontos, mediana geral 1,49 px, estrato `traco=:` em
   1,80 px. Isso também eliminou o `NaN` que aparecia no p95 do critério
   piso (`np.percentile` interpolando entre dois `+inf` — sintoma
   secundário da causa raiz, não um bug à parte).
2. **O p95 do critério 2.2-piso (6,70 px) fica acima do alvo (5 px) por uma
   causa estrutural identificada, não corrigida.** Os piores casos (16,5 px,
   16,4 px, 12,3 px...) são todos imagens com proporção extrema
   altura≫largura (ex.: 260×1158, 284×1023, 569×1105 px) — poucas colunas
   disponíveis para uma curva que, nessas proporções, tem trechos quase
   verticais. O método "uma mediana de linha por coluna" perde informação
   estrutural exatamente onde a curva deixa de ser função de x localmente
   — é uma limitação conhecida de qualquer extrator coluna-a-coluna, não um
   defeito de implementação. Corrigi-lo exigiria acompanhar a curva pelo
   eixo medial (traçado 2D, não coluna-a-coluna), fora do escopo deste
   bloco. **Não ajustado o alvo** — registrado como número real medido, e
   como candidato a limitação a citar na monografia (junto da discussão de
   `t_window`/resolução do `HANDOFF.md`).
3. **A suíte de mutação precisou de um estrato novo (`espessura`) para ter
   poder contra P2-M14.** Os estratos originais do plano (`traco`,
   `marcador`) não isolam espessura de linha; P2-M14 (pular
   `skeletonize`) só piora sistematicamente traços GROSSOS (uma linha
   grossa sem esqueletonizar tem uma "banda" larga de tinta por coluna, cuja
   mediana desvia mais do centro real do que o traço de uma linha fina).
   Acrescentado `espessura={fina,grossa}` (limiar 3 px renderizados,
   `line_width * dpi / 72`) — é adição de diagnóstico, não afrouxamento,
   registrada aqui como o `PLANO_PARTE2.md` pede.

## 5. Armadilhas

1. **P2-M15 (remover a interpolação de vãos) não é mais detectado pela
   suíte, e o motivo é uma interação real com o Ruling 1, não um teste
   fraco por descuido.** Antes da união de componentes, a interpolação
   bridgeava vãos GRANDES (o buraco inteiro entre a maior componente e o
   resto do traço); depois da união, os vãos que sobram são só os
   intervalos naturais dentro de um mesmo tracejado — pequenos o bastante
   para o `MAX_GAP_FRAC` não fazer diferença mensurável contra a máscara
   VERDADEIRA (medido: mediana 1,4847 px sem interpolação vs. 1,4880 px
   com — diferença dentro do ruído). **Isso pode não valer para a máscara
   PREDITA do Bloco 3/3b**, que tem vãos reais (segmentação imperfeita, não
   só o padrão do tracejado) — a interpolação deve voltar a importar lá.
   Reavaliar a mutação P2-M15 no Bloco 5, contra `predict_mask` e
   `extract_mask_classical`, antes de considerar o parâmetro decorativo.
2. **`mask_to_polyline` está testado só contra `mask.png` verdadeira** (sem
   ruído algum, por contrato da Parte 1). O critério 2.2 "de operação"
   (contra a máscara PREDITA) é do Bloco 5, e pode se comportar diferente —
   em especial porque uma máscara predita pode ter componentes espúrias
   pequenas (ruído do modelo) que a união do Ruling 1 agora INCLUI em vez
   de descartar (o antigo comportamento, "só a maior", acidentalmente
   filtrava ruído disperso; o novo é mais vulnerável a isso). Se o Bloco 5
   medir IoU alto mas RMSE de polilinha ruim, olhar primeiro para
   componentes espúrias pequenas sobrevivendo ao piso de `MIN_COMPONENT_PX`.

## 6. O que o próximo bloco precisa saber

1. **`mask_to_polyline`/`polyline_to_series` estão prontos** com as
   assinaturas exatas do `PLANO_PARTE2.md`. O Bloco 5 (`identify_from_image`)
   pode importá-los diretamente.
2. **O critério 2.2 "de operação"** (contra a máscara predita, no Bloco 5)
   é o que decide o veredito oficial do PLANO — o "piso" medido aqui
   (mediana 1,49 px) é a referência de melhor caso possível: se o número do
   Bloco 5 ficar muito acima disso, a degradação vem da segmentação (IoU),
   não da polilinha.
3. **Reavaliar P2-M15 (interpolação de vãos) contra máscaras preditas** —
   ver Armadilha 1. Pode voltar a ser um mutante detectável lá mesmo não
   sendo aqui.
4. Próximo passo: terminar o Bloco 3 (treino da U-Net, rodando em segundo
   plano — ver `HANDOFF_P2_3.md`) e fechar o Bloco 2 (números finais de
   300 amostras — ver `HANDOFF_P2_2.md`), depois integrar tudo no Bloco 5.
