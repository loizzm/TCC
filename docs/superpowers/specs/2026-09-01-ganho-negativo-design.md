# Ganho negativo (K < 0) — design

**Data:** 2026-09-01 · **Bloco:** Parte 2 / Bloco 9 · **Caminho escolhido:** C (espelhamento)

## Problema

Três imagens externas produzidas pelo `rg.py` (degrau negativo) são recusadas
pela pipeline, e pelos motivos errados:

| imagem | verdade declarada | hoje |
|---|---|---|
| `Figure_dn.png` | FOPDT, K=−2, θ=1 s, degrau em t=2 s | `resposta_inversa` |
| `Figure_dn2.png` | 2ª ordem subamortecida, K=−1, θ=2 s | `ajuste_inconsistente` |
| `Figure_dl3.png` | 2ª ordem superamortecida, K=−3, θ=0,5 s, degrau em t=3 s | `ajuste_inconsistente` |

`resposta_inversa` é diagnóstico FALSO: não é fase não-mínima, é degrau
negativo. As recusas são efeito colateral, não detecção.

## Causa raiz — quatro defeitos independentes, medidos

1. **`classical.py:41` — `K_BOUNDS = (1e-3, 1e4)`.** Ganho negativo é
   inexprimível. O ajuste trava K no piso `0,001` e sai com NRMSE 0,90–0,96.
   `_profiled_sse` (`classical.py:221`) também faz `np.clip(K, K_BOUNDS)`.
2. **`generator.py:122` — `K = _loguniform(rng, 0.2, 20.0)`.** O corpus tem
   ZERO sistemas de ganho negativo: sem treino, sem teste, sem critério.
3. **`calibrate.py` — o OCR perde o sinal.** O tesseract lê os DÍGITOS
   corretamente e devolve o sinal como em-dash: `'—0.2'` (U+2014) onde o
   matplotlib desenhou U+2212. O `_NUM_RE` rejeita, e o rótulo inteiro vira
   `None`. Em `dn2`, 1 rótulo lido de 9. O blob e a dilatação estão CERTOS —
   o glifo entra no recorte (37 px contra 26 px do `0.0`).
4. **A polilinha mistura ENTRADA com saída.** Estas figuras desenham o degrau
   de entrada (tracejada branca) junto da resposta. Em `dn` a polilinha pula
   entre os dois objetos e o resíduo infla.

## Escopo

**DENTRO:** defeitos 1, 2 e 3.
**FORA, declarado:** defeito 4 (entrada plotada) e o §39.3 (Estágio A perde
trecho plano — exige retreino). Os dois são extensões de envelope próprias.

## Decisão de projeto — caminho C, espelhamento

Se a resposta DESCE, nega `y` antes do Estágio D, ajusta com o código atual
intocado, e devolve `K` com o sinal invertido.

É exatamente equivalente a parametrizar `K = s·|K|` — o modelo é linear em K e
a base (`_fopdt_basis`, `_second_basis`) é livre de sinal — mas não toca uma
linha da matemática de `classical.py`.

**Alternativas descartadas.**

- **A, alargar a caixa para `(-1e4, 1e4)`.** Põe K=0 DENTRO da caixa, e K=0 é o
  modelo degenerado (resposta plana com τ e θ livres): cria um mínimo local
  trivial que hoje não existe. Também destrói o sinal de borda que o §38.5
  quer usar como guarda ("K colado no piso").
- **B, `K = s·|K|` dentro do otimizador.** Correta, mas espalha o sinal por
  `fit_fopdt`, `fit_second` e `_profiled_sse` — três lugares onde hoje não há
  sinal nenhum.

C preserva os números históricos POR CONSTRUÇÃO: para `s = +1` o caminho é
byte a byte o de hoje. Nenhum critério precisa ser re-argumentado, só remedido
para confirmar que não mudou.

## Onde o sinal é decidido

`s = −1` quando o MÁXIMO da série aparece antes do MÍNIMO; `+1` caso
contrário. O repouso é o extremo que vem primeiro.

A informação está no TEMPO, não no valor. Uma resposta ao degrau nunca cruza de
volta o nível de onde partiu (o sobressinal de 1ª/2ª ordem nunca ultrapassa
100 % do salto), então o repouso É um dos dois extremos da série — e o que o
distingue do sobressinal é a ORDEM em que aparecem. Ler a ordem dispensa saber
onde a resposta assenta, que é o que uma janela curta esconde.

> **Corrigido em 2026-09-02, depois de DUAS formulações refutadas por
> medição.** (1) Mediana do primeiro decil como proxy do repouso: com o platô
> inicial comido pelo §39.3 o decil cai no transitório e a leitura inverte —
> 8 erros em 44 casos, todos ζ ≤ 0,4 sem cabeça, **nos dois sinais de ganho**;
> era a causa de a `dn2` não fechar. (2) Extremo mais distante do assentado:
> pressupõe que o último decil é o valor final, o que a janela curta quebra —
> **espelhava 2 das 900 séries do oráculo**, todas com K > 0, e movia o MAPE(K)
> do estrato limpo w<3 de 0,000 % para 0,239 %. Cada uma quebrava numa ponta
> diferente da série. Detalhe no plano, Task 2.

**Limite do contrato, medido.** Vale enquanto o repouso ainda estiver na série.
Cortada a cabeça além dele, o primeiro extremo que sobra é um sobressinal e a
regra inverte — a direção passa a viver só no envelope decadente, que exige
ajuste. Em ζ=0,2 e θ=2 s: corte em t0 ≤ 2,2 s lê certo; t0 ∈ [2,3; 2,7] lê
errado. Asseverado com `xfail` estrito.

**Não reusar `pipeline._nivel_de_repouso`.** Ele lê as 5 PRIMEIRAS colunas e
supõe a curva parada ali — uma suposição que o próprio `pipeline.py:96-104`
documenta como frágil sob truncagem à esquerda, que é exatamente o defeito
§39.3. Para DIREÇÃO isso é mais tolerante que para nível, mas o estimador novo
usa decis em vez de 5 colunas e vive em `classical.py`, que é a camada de
baixo. `pipeline.py` não muda.

**Empate (série plana):** devolve `+1`. Uma série sem excursão não tem sinal a
recuperar, e `+1` mantém o comportamento atual.

## Resultado esperado, medido em sonda

| imagem | depois do caminho C |
|---|---|
| `dn2` | **fecha**: K=−0,998, ωₙ=5,002, ζ=0,201, θ=2,005, NRMSE 0,0053 — **CONFIRMADO fim a fim em 2026-09-02** (K=−0,9983, ωₙ=5,002, ζ=0,2011, θ=2,005; erro máximo 0,55 %) |
| `dn` | K=−1,997, τ=0,5, θ=2,998 — **todos certos** — mas NRMSE 0,173 > 0,13 e a guarda recusa (defeito 4). **Medido:** o motivo que sai é `resposta_inversa`, não `ajuste_inconsistente` — `_undershoot`=0,973 e `_implausivel` testa a inversa ANTES do NRMSE. Os dois motivos são verdadeiros; a spec só errou qual vence. |
| `dl3` | K=−2,979 certo, θ=2,576 e ζ=0,759 ERRADOS; passa a guarda |

## O custo que esta mudança introduz, declarado

**A `dl3` passa de recusa a ERRO CONFIANTE.** Hoje ela é rejeitada por
`ajuste_inconsistente`; depois do caminho C ela responde com K certo e θ/ζ
errados. A causa é o §39.3 (o Estágio A come o platô e a cauda, cobertura de
62,5 %), não o caminho C — mas quem paga o preço é o usuário da pipeline.

**Isto NÃO será remediado com uma guarda de cobertura de máscara.** O §37.11 já
testou e REFUTOU exatamente essa guarda no corpus: Spearman buraco × erro =
+0,020 (p = 0,57), e o maior buraco do corpus é MAIOR que o da imagem que erra.
Adicionar agora seria repetir um experimento já feito.

A decisão é **aceitar e registrar**: `dl3` entra como fixture com `xfail`
estrito em θ e ζ, apontando para o §39.3. Quando o retreino fechar aquele
defeito, o teste vira XPASS e reprova a suíte, forçando a conversão em portão.

## Estrato do gerador

Ganho negativo entra como estrato **opt-in**, no molde do `reta_no_patamar`
(§34.5): `sample_spec` continua sorteando `K > 0` e o corpus base fica byte a
byte idêntico. O sinal é aplicado por um parâmetro de `generate_sample`, então
nenhum número histórico se move.

## Critérios de aceitação

1. `_texto_para_numero` lê `'—0.2'`, `'−0.2'`, `'–0.2'` e `'--0.8'`; segue
   rejeitando `'=1.2'` e todo texto não numérico.
2. Séries sintéticas de K negativo (FOPDT e 2ª ordem) recuperam K, τ/ωₙ, ζ e θ
   com erro ≤ 1 % no caminho oráculo.
3. **Não regressão:** `data/test` (n=900) mantém calibração aceita em 837,
   acerto de ordem em 93,0 % e o critério 2.6 ≤ 3 p.p. Os relatórios da Parte 1
   e da Parte 2 não mudam de veredito em nenhum critério.
4. `Figure_dn2.png` fecha fim a fim com erro ≤ 6 % em K, ωₙ, ζ e θ.
5. `Figure_dn.png` e `Figure_dl3.png` entram como fixture com `xfail` estrito
   nomeando o defeito que as bloqueia.
6. Estrato de ganho negativo gerável e determinístico, com o corpus base
   inalterado byte a byte.

## Fora de escopo, e a consequência

- **Defeito 4:** `dn` segue recusada. Fechar exige distinguir dois objetos de
  curva no mesmo quadro — envelope novo, spec própria.
- **§39.3:** `dl3` segue com θ e ζ errados. Exige retreino do Estágio A.
