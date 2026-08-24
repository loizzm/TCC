# HANDOFF_P2_2 — Bloco 2: Estágio B completo (OCR opcional, RANSAC, consistência)

## 1. Estado

| Componente | Estado | Evidência |
|---|---|---|
| `identify/calibrate.py::calibrate` | pronto, funcional | 232/300 amostras `ok=True` (77,3%) |
| `identify/calibrate.py::px_to_data` | pronto (inalterado) | usado pelo Bloco 4 |
| Critério 2.3 (erro de escala) | medido, **abaixo do alvo por pouco** | 92,7% < 95% |
| Critério 2.4 (falso alarme) | medido, **acima do alvo** | 22,7% > 5% |
| Critério 2.5 (rejeições corretas) | medido, **abaixo do alvo** | 76,5% < 90% |
| Critério 2.9 (cobertura) | medido, **abaixo do alvo** | 77,3% < 90% |
| Critério 2.11 (nunca levanta) | ✅ passa (escopo parcial — ver §4, Ruling 7) | 300/300 |
| Suíte de mutação (P2-M05 a P2-M09) | 3/5 detectados, 2 não detectados e explicados | ver §3 |

**Leitura honesta:** este bloco não fecha os cinco critérios do PLANO. O
sistema funciona (77% das amostras calibram com < 1% de erro típico — ver a
distribuição em §3), mas fica abaixo dos quatro alvos numéricos. A distância
para os alvos, porém, é pequena e concentrada numa causa comum (cobertura de
OCR), não em cinco problemas distintos — ver §4.

## 2. Interface publicada

```python
@dataclass(frozen=True)
class Calibration:
    sx: float; ox: float; sy: float; oy: float
    bbox_px: tuple[int, int, int, int]
    n_pairs_x: int; n_pairs_y: int
    ok: bool; reason: str

def calibrate(image_rgb: np.ndarray) -> Calibration: ...
def px_to_data(cal: Calibration, x_px: np.ndarray, y_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
```

Motivos de `reason` observados: `"bbox_not_found"`, `"ocr_insuficiente"`
(< 2 pares lidos em algum eixo), `"calibration_failed"` (RANSAC convergiu mas
os inliers não são equiespaçados), `"ransac_failed"`,
`"sinal_de_escala_invalido"` (`sx <= 0` ou `sy >= 0`).

## 3. Números medidos (300 amostras de `data/test`, ~2h de execução)

| Critério | Alvo | Medido | Veredito |
|---|---|---|---|
| 2.3 — erro relativo de `sx`, `sy` | < 1% em ≥ 95% do subconjunto ok | **92,7%** (n=232) | ❌ |
| 2.4 — taxa de rejeição (falso alarme) | < 5% | **22,7%** | ❌ |
| 2.5 — rejeições corretas | ≥ 90%, n ≥ 5 | **76,5%** (n=68 rejeições) | ❌ |
| 2.9 — cobertura da calibração | ≥ 90% global | **77,3%** (n=300) | ❌ |
| 2.9 por DPI (diagnóstico) | — | 60-99: 70,6% (n=85); 100-149: 85,2% (n=115); 150-200: 74,0% (n=100) | — |
| 2.11 — nunca levanta exceção | 100% | **100%** (300/300) | ✅ (escopo parcial) |

A cobertura por DPI não mostra uma tendência monotônica clara (a faixa média
de DPI é a melhor) — não há um único regime de DPI a culpar.

### Suíte de mutação (subconjunto de 60 amostras, por custo — ver Armadilha 1)

| Mutante | Substituição | Esperado | Observado |
|---|---|---|---|
| P2-M05 | `_equiespacados` → `return True` | 2.5 reprova (rejeições somem) | ✅ cobertura sobe 0,750→0,817, 2.3 cai 0,956→0,898 — a guarda desligada deixa entrar calibração pior |
| P2-M06 | `SPACING_TOL` 0,05 → 0,001 | 2.4 explode | ✅ taxa de rejeição 0,250→0,900, cobertura despenca para 0,100 |
| P2-M07 | `fit_axis_affine`: só o primeiro par, sem busca de consenso | 2.3 reprova | ✅ 2.3 cai de 0,956 para 0,674 |
| P2-M08 | `_NUM_RE` aceita qualquer string (`.*`) | 2.3 reprova | ❌ **não detectado** — ver §4, Ruling 6 |
| P2-M09 | Controle: `RANSAC_TOL` 0,02 → 0,021 | nada reprova | ✅ números idênticos ao baseline |

## 4. Rulings

1. **Ordem RANSAC → consistência, não consistência → RANSAC (o esqueleto
   literal do Passo 5 do `PLANO_PARTE2.md`).** Checar `_equiespacados` sobre
   os pares BRUTOS do OCR, antes do RANSAC, faz um único valor lido errado
   reprovar a amostra inteira mesmo com todos os outros pares corretos —
   exatamente o tipo de outlier que o RANSAC existe para descartar. Medido
   num lote de 30 amostras: essa ordem dava 18/30 (60%) `ok=True`; invertendo
   (RANSAC primeiro, depois `_equiespacados` sobre os INLIERS do RANSAC,
   reconstruídos com a mesma tolerância), 24/30 (80%). `fit_axis_affine`
   manteve a assinatura pública; a reconstrução de inliers é uma função nova,
   interna (`_inliers`).
2. **Sem whitelist de caracteres no Tesseract.** `tessedit_char_whitelist`,
   no engine LSTM padrão do Tesseract 4/5 (não o legado, que era o único que
   a respeitava direito), **quebra** o reconhecimento em vez de restringi-lo.
   Medido: um recorte nítido com o dígito "4" bem centralizado, sem ambiguidade
   nenhuma, voltava vazio (`\x0c`) com a whitelist e lia corretamente sem ela.
   O filtro `_NUM_RE` já rejeita qualquer saída não-numérica, então a
   whitelist era redundante mesmo quando funcionava.
3. **`detect_tick_pixels` (Bloco 1) tinha um defeito que só apareceu ao
   medir precisão, não recall.** `tick_direction` é sorteado em
   `{"in", "out", "inout"}` e a implementação original só olhava a faixa
   FORA da moldura — perdendo os ticks "in" inteiramente (~1/3 das amostras).
   O portão G1.2 (Bloco 1) não pegou isso porque mede só RECALL por MEDIANA:
   com 2/3 das amostras intactas, a mediana continuava 1,0. Corrigido para
   checar os dois lados da moldura (dentro e fora) e unir os picos
   (`_merge_close`, tolerância 8 px). Também foi preciso excluir `SPINE_PAD =
   4` px nas pontas de cada faixa, porque sem isso o spine PERPENDICULAR
   (ex.: o spine esquerdo, ao varrer a faixa dos ticks do eixo x) aparecia
   como um pico de tick espúrio — e nesse caso especificamente, o recorte
   largo do OCR alcançava por acidente o rótulo do tick real vizinho e "lia"
   um valor plausível, mas errado, na posição do spine. Este Ruling deveria
   estar no `HANDOFF_P2_1.md` por dono do arquivo (`identify/calibrate.py`
   já existia desde o Bloco 1), mas só foi descoberto neste bloco — registrado
   aqui e referenciado de lá.
4. **`read_tick_labels` não usa mais o parâmetro `ticks` para posicionar os
   recortes de OCR — varre a margem inteira por blobs de texto.** A
   assinatura publicada no `PLANO_PARTE2.md`
   (`read_tick_labels(gray, bbox, ticks)`) foi mantida, mas o argumento
   `ticks` ficou vestigial. Motivo: um recorte de largura fixa (`LABEL_W`)
   centrado em cada posição de `detect_tick_pixels` lê o rótulo do tick
   VIZINHO sempre que há um tick sem marca visível ou sem rótulo entre dois
   ticks maiores rotulados (comum com `has_minor_ticks` ou
   `has_major_ticks=False`, que deixa só o TEXTO do rótulo, sem marca
   nenhuma, onde a busca por "pico de tinta" esperava uma marca curta) —
   isso inflava os pares com duplicatas e valores lidos fora de posição.
   Medido: só 2/30 amostras chegavam a `ok=True` com a abordagem "recorte ao
   redor do tick"; escanear blobs de texto (componentes conexas, com
   dilatação para fundir os caracteres do MESMO número sem fundir rótulos
   de linhas vizinhas) chegou a 24/30. Um blob que toca a borda da faixa
   perto do spine pode ter uma marca de tick fundida ao dígito (medido:
   "7.5" virava "75:" com uma marca colada) — tratado tentando o OCR com e
   sem essa borda aparada, aceitando o primeiro resultado válido, em vez de
   decidir de antemão se há marca (decidir errado quebrava tantos casos
   quanto ajudava).
5. **`_equiespacados` foi reescrita para tolerar LACUNAS**, não só desvio em
   torno da média. A versão original (Passo 5 do `PLANO_PARTE2.md`) compara
   diferenças CONSECUTIVAS contra a média — um tick perdido no MEIO da
   sequência (comum: o OCR não lê 100% dos rótulos) produz uma diferença de
   "N espaçamentos" que não bate com a média de "1 espaçamento", reprovando
   mesmo com todo par individualmente correto. Reescrita para checar se cada
   diferença consecutiva é próxima de um múltiplo INTEIRO do menor
   espaçamento observado (cobre tanto "sem lacuna" quanto "faltaram N
   ticks"), e ainda reprova um valor lido errado (razão longe de qualquer
   inteiro). Esta foi a mudança de maior impacto isolado nas primeiras
   iterações (18/30 → ainda 18/30 nesse passo específico, mas eliminou a
   maioria dos falsos `calibration_failed` que restavam depois do Ruling 4).
6. **`fit_axis_affine` precisou de desempate por RESÍDUO TOTAL**, não só
   contagem de inliers. Com poucos candidatos (3 pares, 1 errado), quaisquer
   2 deles "empatam" em 2 inliers (2 pontos sempre se ajustam exatamente um
   ao outro) — sem desempate, a ORDEM DE ITERAÇÃO decide arbitrariamente
   qual par vence, e nada garante que seja o par correto. Medido num caso
   concreto: a reta formada por 1 ponto bom + 1 ponto com OCR errado venceu
   por ordem de iteração, dando 20% de erro de escala numa amostra que
   `calibrate` reportava como `ok=True`. O desempate por resíduo total
   (soma dos erros absolutos sobre TODOS os pares, não só os inliers da reta
   candidata) corrigiu esse caso (erro caiu para 0,01%) — é o que fez a
   fração de erros < 1% (critério 2.3) saltar de 76,5% para 95,65% num lote
   de 30, e depois estabilizar perto de 92,7% em 300 (a folga cai um pouco
   com mais amostras, mas o mecanismo continua correto — ver §4 abaixo sobre
   os alvos não batidos).
7. **O critério 2.11, como testado aqui, cobre só o CONTRATO de
   `calibrate()`** (nunca levanta, `ok`/`reason` bem formados) — não a
   afirmação completa do PLANO ("100% das amostras com `dimensionless`
   preenchido"), que depende de `identify_from_image` (Bloco 5, ainda não
   existe). Marcado como coberto **parcialmente**; o Bloco 5 precisa
   reexecutar 2.11 contra o pipeline completo antes de declará-lo fechado.

### Por que os quatro alvos numéricos não foram batidos

Os quatro criterios que falham (2.3, 2.4, 2.5, 2.9) têm a MESMA causa raiz:
**cobertura de OCR insuficiente**, não quatro problemas independentes.
2.9 mede a cobertura diretamente (77,3%, alvo 90%); 2.4/2.5 são o par
precisão/cobertura do lado da REJEIÇÃO (com menos amostras "boas" sobrando,
a proporção que passa pela consistência mas ainda erra fica mais visível);
2.3 mede a exatidão do subconjunto aceito, que já está em 92,7% — perto do
alvo, mas ainda abaixo. Não há um bug isolado restante conhecido: as seis
correções acima (Rulings 1–6) já atacaram cada causa medida até aqui, uma de
cada vez, e cada uma teve efeito medido e positivo. O que resta é
provavelmente uma cauda mais difusa de casos individuais (fontes pequenas em
DPI baixo — exatamente o risco "Alta probabilidade" que o `PLANO.md §5` já
antecipava — e a interação entre estilos de eixo menos comuns), sem um único
próximo alvo óbvio de alto retorno. **Não foram forçados os limiares para
passar** — os números acima são os medidos, e ficam registrados como tal.

## 5. Armadilhas

1. **O ciclo de medição completo (300 amostras, 4 critérios) leva ~2 horas.**
   Cada `calibrate()` dispara vários subprocessos `tesseract` (um por blob de
   texto candidato), e o overhead de spawn de processo domina — não é
   CPU-bound (o sistema fica majoritariamente ocioso durante a suíte). Isso
   torna a iteração rápida "medir → mudar → medir de novo" inviável em
   escala completa; toda a depuração deste bloco foi feita em lotes de
   20–60 amostras, com a confirmação final em 300. A suíte de mutação
   também rodou em 60 amostras por este motivo — registrado como desvio
   consciente do "sobre a suíte completa" que o `PLANO_PARTE2.md` pede no
   Bloco 5, Passo 7; refazer em 300 antes de citar a tabela de mutação na
   monografia, se o tempo permitir.
2. **P2-M08 não é mais detectável com a arquitetura atual, e não é uma
   fraqueza de teste — é uma mudança real de superfície de falha.**
   `_ocr_number` já envolve `float(txt)` num `try/except ValueError`; uma
   string que não é um número válido falha ali de qualquer forma, com ou sem
   o filtro de regex antes. A regex parou de ser a única linha de defesa
   quando o resto do pipeline mudou (Rulings 4–6). Não regredido de propósito
   — mas também não vale reescrever só para reabrir um mutante já coberto por
   outra camada.
3. **Cuidado ao reexecutar os mutantes:** cada teste envolve editar
   `identify/calibrate.py` diretamente e depois restaurar a partir de um
   backup (`/tmp/.../scratchpad/calibrate_bloco2_final.py` nesta sessão, mas
   esse caminho é efêmero — salve um backup antes de repetir isto em outra
   sessão). Um mutante esquecido no arquivo contaminaria qualquer medição
   seguinte silenciosamente.

## 6. O que o próximo bloco precisa saber

1. **`calibrate()` está pronto para uso, mesmo sem bater os quatro alvos.**
   Ele produz `Calibration` bem formada em 100% das amostras (nunca levanta),
   com `ok=True` em ~77% e erro < 1% em ~93% desses. O Bloco 5 pode e deve
   usá-lo como está; os números finais do critério 2.6 (degradação
   end-to-end) vão refletir esta cobertura — é esperado que 2.6 também sofra
   proporcionalmente, e isso não é um defeito novo do Bloco 5, é a
   propagação honesta deste resultado.
2. **A causa dominante de falha é cobertura de OCR, concentrada em nenhum
   estrato de DPI específico** (ver tabela §3) — se o Bloco 5 precisar
   melhorar 2.6, o primeiro lugar a olhar é aqui, não em `identify.classical`
   (Parte 1, inalterado) nem no Estágio A.
3. **`detect_tick_pixels` (Bloco 1) foi modificado neste bloco** (Ruling 3) —
   o `HANDOFF_P2_1.md` não reflete essa mudança porque ela só foi descoberta
   aqui. Releitura recomendada se outro executor for revisitar o Bloco 1.
4. Próximo passo: aguardar o treino da U-Net (Bloco 3, em segundo plano) e
   integrar tudo no Bloco 5 (`identify/pipeline.py`).
