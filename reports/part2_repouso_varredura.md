sintetico sem calibracao, 2a ordem: n=33

candidato               REAL zeta    erro  SINT MAPE  ordem ok
  mediana 8% (ATUAL)       0.4851    3.0%      2.92%    97.0%
  mediana 3%               0.5045    0.9%      3.02%   100.0%
  mediana 5 colunas        0.5057    1.1%      1.34%   100.0%
  mediana 3 colunas        0.5061    1.2%      2.72%   100.0%
  percentil 99             0.5052    1.0%      3.64%   100.0%

## Fix round 1 — achado adicional: normalização do tempo (t) em `_serie_normalizada`

Com o estimador de repouso já trocado (mediana de 5 colunas), o teste
`test_caso_real_recupera_zeta_e_wn` continuava falhando — não em zeta (que já
passava, 1,1% de erro), mas em ωₙ (37,7% de erro, tolerância 5%). Diagnóstico:
a polilinha do caso real cobre só 61,7% da largura da moldura do gráfico
(462/746 px) porque o trecho assentado da curva se sobrepõe à reta de
referência tracejada e a máscara não separa as duas ali. `_serie_normalizada`
normalizava `t` pela extensão OBSERVADA da polilinha (`x[-1]-x[0]`), não pela
janela real — com a polilinha truncada, isso comprime o tempo e infla ωₙ na
mesma proporção.

### B2 — cobertura no corpus sintético (900 amostras de `data/test`, sem
calibração, 2ª ordem, com zeta): min=0,8937, p1=0,8967, p5=0,9011,
mediana=0,9299, max=0,9714. Caso real: 0,6166. A moldura é sempre MAIS LARGA
que a janela de dados no sintético (razão bbox/janela: min=1,03, mediana=1,08,
max=1,12) — margem do matplotlib (`x_margin_lo`/`x_margin_hi`, U(0,01,0,06)
cada), não falta de cobertura da polilinha. Ou seja: a moldura não é a janela
real do gráfico, é uma aproximação POR EXCESSO dela.

Consequência de usar a moldura SEMPRE (sem condicionar à truncagem), medido no
mesmo corpus (n=75, sem-calibração e zeta presente, ordem correta):

| modo                  | MAPE zeta | ordem_ok | \|erro θ_T\| mediana | \|erro θ_T\| máx | MAPE ωₙ·T |
|------------------------|-----------|----------|------------------|-------------|-----------|
| extensão observada (antigo) | 2,86% | 96,0% (72/75) | 0,0024 | 0,0818 | (referência) |
| moldura SEMPRE         | 2,86%     | 96,0% (72/75) | 0,0199 | 0,0527 | — |
| moldura CONDICIONAL (limiar 0,75) | 2,86% | 96,0% (72/75) | 0,0024 | 0,0818 | — |

E na medição feita depois, dentro de `tests/part2/test_part2.py`
(`2.6-adim[wn_T]`, n=143, amostras de `test_samples` — as 300 primeiras de
`data/test`): MAPE de ωₙ·T real sobe quando a moldura é usada SEMPRE — piora
que não aparece em nenhum critério existente porque só havia `2.6-adim[zeta]`,
e ζ é invariante à escala de `t` (motivo do item B3 abaixo).

> **Correção da fix round 2 (C1).** O par "2,60% → 7,20% em n=143" que esta
> seção afirmava na rodada 1 mistura duas populações e NÃO reproduz. Medido de
> novo (ver "Fix round 2" no fim deste arquivo): em n=143 (corpus daquele
> diagnóstico) o real vai de **1,76% para 2,54%**; no subconjunto **sem
> calibração** daquele mesmo diagnóstico (n=33), que é a população que
> `_serie_normalizada` de fato serve, vai de **1,35% para 7,15%**. O "7,20%"
> era parente do 7,15% do subconjunto de n=33; o "2,60%" era de outra medição
> ainda (n=75, sem calibração e 2ª ordem, verdade = janela de dados) e não é a
> linha de base de nenhuma das duas.

Escolha: **condicionar** o uso da moldura à cobertura medida
(`extensão observada / largura da moldura`). Limiar escolhido: **0,75** — a
meio caminho entre os dois extremos medidos (caso real 0,6166 e mínimo do
sintético 0,8937), com ~13 p.p. de folga para cada lado. *(Corrigido na fix
round 2, C2: aquele 0,8937 é o mínimo só das 75 de 2ª ordem. Sobre TODAS as
179 sem calibração o mínimo é 0,7713 e a folga real é de 2,1 p.p.)* Abaixo do
limiar
(polilinha truncada, caso do real): usa a moldura, porque a extensão observada
erra muito mais (37,7% em ωₙ) do que o viés de margem da moldura (~7-12%).
Acima (regime do sintético): usa a extensão observada, porque ali ela já é a
melhor referência e a moldura só importaria o viés de margem.

### B1 — origem e escala precisam do MESMO referencial

Quando a moldura é usada para a escala (`span`), a origem de `t` também
precisa vir da moldura (`bbox_px[0]`), não do primeiro ponto da polilinha
(`x[0]`) — os dois valores vêm de fontes diferentes e a diferença entre eles
vaza como viés ADITIVO em θ_T. Medido no mesmo corpus (n=75, moldura sempre
usada para isolar o efeito da origem): viés do offset de origem em % de T —
mediana 3,43%, máx 6,03%, p90 4,94% (n=75, sem calibração e 2ª ordem).
*(Remedido na fix round 2 sobre as 179 sem calibração, todas as ordens:
mediana 3,1%, p95 5,5%, máximo 11,4% da largura da moldura.)* Custo de
corrigir a origem: zero
nas três métricas medidas (MAPE ωₙ·T, ordem_ok, e o próprio |erro θ_T| não
piora — ver tabela acima, linha "moldura CONDICIONAL" == linha "extensão
observada"). No caso real o offset é de apenas 1 px (bbox_x0=75, x_px.min()=76),
por isso B1 não aparecia nele.

### B3 — critério ausente que deixou B2 passar

Não existe (e não é criado agora) nenhum critério `2.6-adim[wn]`/`2.6-adim[wn_T]`
com meta de aprovação — só `2.6-adim[zeta]`, e ζ é invariante à escala de `t`.
Isso é o motivo estrutural de a regressão de ωₙ do "moldura sempre" (números
corrigidos na fix round 2: 1,76% → 2,54% em n=143, e 1,35% → 7,15% nas n=33
sem calibração) não ter sido pega por nenhum teste. `tests/part2/test_part2.py`
ganhou `2.6-adim[wn_T]` como **diagnóstico** (sem meta, sem assert) dentro de
`test_2_6_degradacao_vs_oraculo`, ao lado de `2.6-adim[zeta]`, para tornar o
número visível em toda rodada futura — ver seção "Fix round 1" do relatório
da task para os números medidos com a correção condicional aplicada.

## Fix round 2 — C1 a C4 (tudo remedido nesta rodada)

Ambiente: `.venv/bin/python`, `models/unet_stageA.pt`, CUDA. As polilinhas, as
molduras e o veredito da calibração das **900** amostras de `data/test` foram
cacheados uma vez e reusados em todas as medições abaixo (scripts no
scratchpad, fora da árvore). Toda linha diz a que população pertence.

### Populações usadas (contadas nesta rodada)

| população | n | o que é |
|---|---|---|
| `data/test` inteiro | 900 | corpus sintético completo |
| sem calibração | 184 | onde `_serie_normalizada` roda |
| sem calibração, polilinha ≥ 10 pts e moldura válida | 179 | domínio real da constante de cobertura |
| … dessas, 1ª ordem | 104 | fopdt |
| … dessas, 2ª ordem | 75 | o único subconjunto medido na fix round 1 |
| primeiras 300 de `data/test` | 300 | população de `test_samples` (`tests/part2`) |
| … aceitas em `2.6-adim[wn_T]` | 143 | corpus do diagnóstico |
| … dessas, sem calibração | 33 | subconjunto que o diagnóstico deveria vigiar |
| caso real | 1 | `tests/fixtures/caso_real_2ordem.png` |

### C1 — números citados fora da população

Medido com o limiar em `1.01` (moldura incondicional) contra o valor de
produção (`0.75`), replicando o portão de `2.6-adim[wn_T]`:

| população | MAPE ωₙ·T real, condicional | MAPE ωₙ·T real, moldura sempre | Δ da linha do relatório |
|---|---|---|---|
| 300 primeiras, aceitas (n=143) | 1,76% | 2,54% | +1,04 → +1,82 p.p. |
| … só sem calibração (n=33) | 1,35% | 7,15% | +0,63 → +6,43 p.p. |
| 900 inteiro, aceitas (n=418) | 2,06% | 2,64% | +1,38 → +1,97 p.p. |
| … só sem calibração (n=76) | 2,59% | 6,82% | +1,85 → +6,08 p.p. |

O par "2,60% → 7,20% em n=143" da fix round 1 não existe em nenhuma dessas
linhas. Corrigido no código, neste arquivo e no relatório da task.

### C2 — distribuição de cobertura sobre TODAS as sem calibração (n=179)

```
min=0.7713  p1=0.8512  p5=0.9009  p25=0.9162  mediana=0.9332
p75=0.9460  p95=0.9667  max=0.9873
abaixo de 0,75: 0     abaixo de 0,80: 1     abaixo de 0,872: 3
1a ordem (n=104): min=0.7713  mediana=0.9346  max=0.9873
2a ordem (n= 75): min=0.8937  mediana=0.9299  max=0.9714
```

O mínimo é `sample_00639` (fopdt, 0,7713). **A folga do limiar 0,75 até o
mínimo observado é de 2,1 p.p.**, não os ~13 p.p. documentados na fix round 1
— que vinham de olhar só as 75 de 2ª ordem. O ramo da moldura continua sem
disparar em nenhuma amostra do sintético; o que muda é a margem, não o
comportamento.

Margem do matplotlib nessas mesmas 179 (`largura da moldura / largura da
janela de dados − 1`, pela `axis_affine` verdadeira): min 0,0223, mediana
0,0686, p95 0,1052, max 0,1166 — isto é, a moldura é de +2,2% a +11,7% mais
larga que a janela de dados.

### C3 — o ponto de empate, derivado e medido (decisão: MANTER 0,75)

Álgebra refeita aqui. Com `c` = cobertura e `m` = soma das duas margens, a
moldura vale `(1+m)·T` em pixels e a extensão observada vale `c(1+m)·T`. O erro
de ESCALA de `t` é `|c(1+m) − 1|` usando a extensão observada e `m` usando a
moldura. Igualando: **c\* = (1−m)/(1+m)**. Como `m` varia por amostra, c\*
também varia — medido nas 179: min 0,7912, mediana **0,8716**, p95 0,9379,
max 0,9563. Não é uma constante, e o "0,872" é a mediana dela.

Subir o limiar para 0,872 foi medido:

- Passa a trocar de referência em **3 das 179** (0,7713, 0,8505, 0,8514 —
  todas fopdt). A amostra mais próxima acima é 0,8937, ou seja o limiar cai
  dentro de um vão empírico de 4,2 p.p. (0,8514 → 0,8937).
- `2.6-adim[zeta]` e `2.6-adim[wn_T]` (corpus e sem calibração) ficam
  **idênticos dígito a dígito** a 0,75, nas 300 e nas 900 — porque as três
  amostras afetadas são de 1ª ordem e não entram em nenhuma métrica de ωₙ/ζ.
- Nas grandezas que elas movem (sem calibração, n=184; aceitas: 75 para τ/T e
  151 para θ/T), o saldo é **misto**:

```
limiar   ordem_ok   MAPE tau_T med   tau_T p95   |dtheta_T| med   p95
0.750      85.9%          0.87%        10.37%         0.0028      0.0164
0.872      85.9%          0.87%         6.42%         0.0028      0.0181
1.010      85.9%          5.84%        11.51%         0.0197      0.0487
```

Por amostra, entre as três que cruzariam:

```
sample_00639 fopdt  tau_T 24.9% -> 3.6%   theta_T 17.7% -> 70.9%
sample_00828 fopdt  tau_T  4.9% -> 19.1%  theta_T 13.4% -> 40.4%
sample_00446 fopdt  tau_T  9.9% -> 6.4%   theta_T 90.1% -> 73.7%
```

**Por que não fecha.** O empate acima só considera a ESCALA. Trocar para a
moldura move junto a ORIGEM de `t` para `bbox_px[0]` (exigência da fix round 1,
B1), e quando a truncagem é à DIREITA — o regime do caso real — a origem
observada já estava certa: a troca importa a margem esquerda como viés aditivo
em θ. Visível em `sample_00639`: θ/T sai de 0,0333 para 0,0484 com alvo
0,0283, enquanto τ/T melhora de 24,9% para 3,6%. Em 1 das 3 (`sample_00828`) a
escala também piora, apesar de o modelo de empate prever melhora — sinal de
que o erro de escala não é o único termo em jogo.

**Decisão: recuar e manter 0,75**, com o empate registrado no comentário do
código. Não há ganho líquido medido: nenhuma métrica com meta se move, uma
cauda melhora (τ/T p95) e outra piora (|Δθ/T| p95), e o limiar mais alto passa
a expor amostras não truncadas ao viés de margem que a fix round 1 existia
para evitar.

### C4 — o diagnóstico passou a medir a população que vigia

`tests/part2/test_part2.py` agora registra DUAS linhas: `2.6-adim[wn_T]`
(corpus, n=143, para comparar com `2.6-adim[zeta]`) e
`2.6-adim[wn_T/sem-calib]` (n=33, o subconjunto onde `_serie_normalizada`
roda), esta sem exigência de n ≥ 100.

**Prova de sensibilidade** (cópia do repositório em `/tmp`, só
`_COBERTURA_MIN_MOLDURA = 1.01`, `pytest tests/part2/test_part2.py::test_2_6_degradacao_vs_oraculo`):

```
2.6-adim[wn_T]            +1.82 p.p. (oráculo 0.72%, real 2.54%, n=143)
2.6-adim[wn_T/sem-calib]  +6.43 p.p. (oráculo 0.72%, real 7.15%, n=33)
```

contra `+1,04` e `+0,63 p.p.` da árvore de produção: a linha do corpus se move
**+0,78 p.p.**, a nova se move **+5,80 p.p.** — o fator ~7 de diluição
eliminado.
