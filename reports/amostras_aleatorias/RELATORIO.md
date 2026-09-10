# Assertividade em 99 figuras aleatorias (33 por gerador)

Gerado por `rg_aleatorio.py` (figuras + verdade), medido por `identificar.py`
(U-Net `models/unet_stageA.pt`) e apurado por `analisa_aleatorias.py`.
Numeros brutos em `analise.txt`, `analise.json`, `verdade.json`,
`resultado.json`. Semente 20260908 — a geracao inteira e reproduzivel.

## 0. O que foi gerado

33 figuras por familia, cada familia preservando a assinatura de render do
gerador que imita e sorteando todo o resto (planta, degraus, cor, legenda e
sua posicao, tipo de linha, largura de linha, preenchimento, grade, dpi,
figsize, limites de eixo):

| familia | imita | tema | degraus | ganho |
|---|---|---|---|---|
| `rg` | `rg.py` | seaborn-v0_8-darkgrid, entrada acumulada tracejada | 2 | qualquer sinal |
| `neg` | `rg_negativo.py` | dark_background, `plt.step` + `axvspan` | 1 | sempre negativo |
| `multi` | `rg_multidegrau.py` | seaborn darkgrid, gravado via Agg | 1 a 3 | qualquer sinal |

As plantas ficam dentro do dominio de treino de
`dataset/generator.py:sample_system`. O que sai do dominio, de proposito, e o
RENDER. Assim a medida isola o efeito do render e nao o de extrapolar fisica.

**Convencoes da verdade** (ARQUITETURA.md, "O que K significa"): `K = K_planta
x U`, e `theta` e o instante de partida (`instante do degrau + tempo morto`).
Em figura com mais de um degrau a verdade declarada e a do PRIMEIRO degrau.

## 1. Um defeito do GERADOR, achado antes de virar conclusao errada

A primeira geracao mediu 12,1 % de calibracao na familia `neg` contra 97 % nas
outras, com `bbox_not_found` e `ocr_insuficiente`. A conclusao obvia — "fundo
escuro quebra a calibracao" — estava errada: as quatro fixtures reais de
`rg_negativo.py`, que tambem sao `dark_background`, calibram todas.

Causa real, medida: `plt.style.use` so sobrescreve as chaves que o estilo
declara. `seaborn-v0_8-darkgrid` zera `axes.linewidth`, `dark_background` nao
a restaura, e como as familias rodavam na mesma sessao as figuras `neg` sairam
**sem as molduras dos eixos** — exatamente o que `detect_plot_bbox` procura.
Medido nos pixels: no `neg_00` a melhor linha candidata a spine tinha
preenchimento 0,858 contra os 0,90 exigidos; na fixture real, 1,000.

Corrigido com `rcdefaults()` antes de cada estilo (`_estilo()` em
`rg_aleatorio.py`). Depois da correcao: 0 de 33 sem moldura, e a familia `neg`
foi de 12,1 % para 93,9 % de entrega fisica. **Todos os numeros abaixo sao os
de depois da correcao.**

## 2. Numeros de topo

| medida | resultado |
|---|---|
| entregou parametro fisico | **95/99 (96,0 %)** |
| calibrou os dois eixos | 98/99 (99,0 %) |
| sinal de `K` correto | 95/95 (100 %) |
| rotulo de estrutura correto | 78/96 (81,2 %) |
| curva reconstruida com NRMSE <= 5 % | **74/99 (74,7 %)** |
| curva reconstruida com NRMSE <= 2 % | 54/99 (54,5 %) |
| latencia | mediana 732 ms, p90 1369 ms |

As 4 recusas: 3 `ajuste_inconsistente` e 1 `sinal_de_escala_invalido`. Nenhuma
delas devolveu numero errado em silencio.

O erro de `theta` e pequeno em toda a populacao: `|dtheta|/T` mediana 0,0035,
p90 0,0544. `theta` e a grandeza mais robusta do sistema.

## 3. O que de fato explica os erros

Nao e o render. E **a truncagem — se ela dispara ou nao.**

### 3.1 Truncagem como detector de "mais de um degrau"

Precisao 65,8 %, revocacao 43,1 % (TP=25, FN=33, FP=13, TN=24).

| regime | n | NRMSE mediano | \|erro de K\| mediano |
|---|---|---|---|
| 1 degrau, sem truncagem | 24 | 0,0026 | **0,26 %** |
| 1 degrau, truncagem espuria | 13 | 0,0582 | 14,3 % |
| 2+ degraus, truncou | 25 | 0,0054 | **1,4 %** |
| 2+ degraus, nao truncou | 33 | 0,0377 | **50,2 %** |

Quando a truncagem acerta, o sistema e quase exato. Quando erra, nos dois
sentidos, o erro de `K` sobe uma ou duas ordens de grandeza.

### 3.2 O que o ajuste nao-truncado esta descrevendo

Nas 33 multi-degrau que nao truncaram, `K` esta a 50,2 % da verdade do 1o
degrau — mas a **11,7 % da excursao TOTAL** (soma dos degraus), e mais perto do
total em **31 de 33**. O ajuste nao devolve lixo: responde a outra pergunta —
o ganho da entrada acumulada inteira. E exatamente o mal-entendido do Ruling 65
que `identificar.py` ja documenta no bloco "Serie truncada".

### 3.3 Onde a truncagem acerta o lugar

Nas que truncaram uma figura multi-degrau, o corte caiu a menos de 20 % da
janela do 2o degrau em **15/18** (2 degraus) e **7/7** (3 degraus). O problema
da truncagem e de disparo, nao de pontaria.

### 3.4 Tamanho da janela nao e a causa (confundimento desfeito)

Agregado, janela efetiva < 2 t_dom parecia catastrofica (\|erro de K\| 39 %).
Separando por numero de degraus, o efeito some:

| janela efetiva | 1 degrau | 2+ degraus | 2+ degraus truncadas |
|---|---|---|---|
| 0–2 t_dom | 3,5 % (n=9) | 46,4 % (n=35) | 2,0 % (n=12) |
| 2–3 t_dom | 5,8 % (n=10) | 6,0 % (n=23) | 0,8 % (n=13) |
| 3–4,6 t_dom | 0,4 % (n=14) | — | — |
| >= 4,6 t_dom | 0,7 % (n=4) | — | — |

Em figura de um degrau so, o modelo extrapola o patamar corretamente mesmo com
2 t_dom de janela. A janela curta so machuca quando ha um segundo degrau que a
truncagem deixou passar.

### 3.5 Rotulo de estrutura

Matriz de confusao: fopdt->fopdt 35, fopdt->second 13, second->fopdt 5,
second->second 43. Das 18 trocas, 5 sao dinamicamente inocuas (t_dom a menos de
15 % — um polo lento de 2a ordem superamortecida E um `tau` de FOPDT). As
outras 13 se concentram no mesmo lugar dos erros de `K`:

| regime | taxa de troca de rotulo |
|---|---|
| 2+ degraus, nao truncou | 11/33 = 33,3 % |
| 2+ degraus, truncou | 4/25 = 16,0 % |
| 1 degrau, limpa | 2/24 = 8,3 % (as duas com NRMSE <= 0,022) |

### 3.6 Degenerescencia K x amortecimento

6 de 95 ajustes (6,3 %) terminaram com `zeta` **encostado no limite da caixa**
(`ZETA_BOUNDS = (1e-3, 10.0)`, `identify/classical.py:70`). Nao e um valor: e o
otimizador dizendo que nao ha minimo interior. Entre eles, NRMSE mediano 0,1870
e \|erro de K\| 0,4996; entre os demais, 0,0135 e 0,0602. Casos extremos:
`neg_11` saiu com `zeta=10,0` e `K=-569,6` para uma verdade de `zeta=1,06`,
`K=-10,1`; `rg_04` e `rg_06` sairam com `zeta=0,001`.

**Este e um sintoma sem guarda.** A pipeline recusa por residuo alto
(`ajuste_inconsistente`), mas nao por parametro encostado no limite — e nesses
6 casos o residuo era baixo, entao a recusa nao dispara.

## 4. O que o render NAO estragou

| corte | entrega fisica | NRMSE mediano |
|---|---|---|
| linha solida `-` (n=30) | 96,7 % | 0,0043 |
| tracejada `--` (n=25) | 100 % | 0,0190 |
| traco-ponto `-.` (n=26) | 92,3 % | 0,0239 |
| pontilhada `:` (n=18) | 94,4 % | 0,0156 |
| fundo claro (n=66) | 97,0 % | 0,0191 |
| fundo escuro (n=33) | 93,9 % | 0,0055 |
| sem preenchimento (n=34) | 94,1 % | 0,0115 |
| `axvspan` no atraso (n=17) | 100 % | 0,0188 |
| `fill_between` sob a curva (n=17) | 100 % | 0,0054 |
| `axhspan` de referencia (n=13) | 92,3 % | 0,0254 |

Tipo de linha, cor, legenda, preenchimento, grade, dpi e tema **nao movem a
agulha**. A U-Net do estagio A esta robusta a variacao de estilo — o que faz
sentido, porque e exatamente o que `dataset/randomize.py` sorteia no treino.

## 5. Veredito

O que o sistema entrega hoje, com uma frase de escopo:

- **Figura de um degrau, sem truncagem espuria (n=26, 26 % do corpus):**
  entrega 92,3 %, estrutura 22/24, NRMSE <= 5 % em 88,5 %, erro de `K` mediano
  0,26 %, `|dtheta|/T` mediano 0,0015. Isto e um resultado forte.
- **Figura multi-degrau em que a truncagem disparou (n=25):** NRMSE <= 5 % em
  96 %, erro de `K` mediano 1,4 %. Quando o mecanismo funciona, funciona.
- **Figura multi-degrau em que a truncagem nao disparou (n=33):** os
  parametros descrevem a excursao total, nao o 1o degrau. Nao e ruido, e outra
  pergunta — mas a saida nao avisa.

Ordem de ataque, por retorno medido:

1. **Revocacao da truncagem** (43,1 %). E o maior item isolado: leva 33
   amostras de 50 % de erro de `K` para ~1,4 %.
2. **Truncagem espuria em figura de um degrau** (13/37 = 35 %). Custa 55x no
   erro de `K` (0,26 % -> 14,3 %) justamente no regime em que o sistema e
   melhor.
3. **Guarda de parametro no limite da caixa** (6/95). Barata: `zeta` a 1e-3 ou
   a 10 com residuo baixo deveria virar recusa nomeada ou aviso, nao um `K` de
   -569.

Nada nesta rodada aponta para o estagio A (extracao) nem para a calibracao: 99
% calibraram e o estilo do render nao correlaciona com erro. O gargalo esta no
estagio D, na decisao de truncar.

---

# Adendo: onde esta o problema — sinal de K ou multiplos degraus?

Fonte: `cruza_sinal_degraus.py`, saida bruta em `cruzamento.txt`. n = 299.

## 6. Por que precisou de mais dois lotes

O lote de 33 por familia **nao responde a pergunta**: `neg` e 1 degrau negativo
por construcao e `rg` e 2 degraus, entao a celula (K>0, 1 degrau) fica com
**n=2** e "o problema e o ganho negativo" seria indistinguivel de "o problema e
multi-degrau". Foram gerados mais dois lotes em fatorial 2x2 — (sinal de K) x
(1 ou 2 degraus), 25 por celula, com a assinatura de render sorteada entre as
tres familias — justamente para desfazer esse confundimento.

| lote | n | como |
|---|---|---|
| `amostras_aleatorias/` | 99 | 33 por familia de gerador |
| `amostras_aleatorias/balanceado/` | 100 | fatorial 2x2, semente 20260908 |
| `amostras_aleatorias/balanceado2/` | 100 | fatorial 2x2, semente 20260909 |

## 7. A tabela cruzada (n=299)

| celula | n | entrega fisica | estrutura | NRMSE med | \|erro de K\| med | \|erro K\|>50% |
|---|---|---|---|---|---|---|
| K>0 · 1 degrau | 52 | 73,1 % | 37/38 | 0,0021 | **0,32 %** | 4/38 |
| K<0 · 1 degrau | 87 | 89,7 % | 72/78 | 0,0035 | **0,42 %** | 9/78 |
| K>0 · 2+ degraus | 80 | 95,0 % | 58/77 | 0,0194 | **17,90 %** | 26/76 |
| K<0 · 2+ degraus | 80 | 91,2 % | 56/74 | 0,0198 | **20,76 %** | 21/73 |

Marginais:

| corte | n | \|erro de K\| med | acerto de estrutura |
|---|---|---|---|
| K>0 | 132 | 5,37 % | 95/115 = 82,6 % |
| K<0 | 167 | 3,43 % | 128/152 = 84,2 % |
| **1 degrau** | 139 | **0,35 %** | 109/116 = **94,0 %** |
| **2+ degraus** | 160 | **20,74 %** | 114/151 = **75,5 %** |

## 8. Veredito: **e multiplos degraus, e nao e o sinal**

**O sinal de K nao tem efeito.** Testado com controle, nao apenas na marginal:

| teste | resultado |
|---|---|
| sinal -> recusa, controlando `K_planta<1` (CMH, 4 estratos) | OR=1,38 · **p=0,523** |
| sinal -> \|erro de K\|, 1 degrau (Mann-Whitney) | 0,32 % vs 0,42 % · **p=0,493** |
| sinal -> \|erro de K\|, 2+ degraus | 17,9 % vs 20,8 % · **p=0,769** |
| sinal -> NRMSE, 1 degrau / 2+ degraus | **p=0,168 / p=0,854** |
| sinal -> revocacao da truncagem | 48,7 % vs 47,9 % · **p=1,000** |

A diferenca de recusa que aparece na marginal (K>0 1 degrau 26,9 % contra K<0
1 degrau 10,3 %, Fisher p=0,017) **desaparece ao controlar**: foi um
desbalanceio do sorteio, com `K_planta<1` mais frequente entre as amostras de
K>0 (Mann-Whitney p=0,013 no proprio `K_planta`). O retreino de ganho negativo
(`retreino_kneg.sh`, §41) fez o trabalho dele — nao ha divida em K<0.

**Multiplos degraus e o problema, e por uma margem grande.** Erro de `K`
mediano vai de 0,35 % para 20,74 % — **59x** — e o acerto de estrutura cai de
94,0 % para 75,5 %. O efeito aparece igual nos dois sinais (17,9 % e 20,8 %),
o que confirma que a variavel e o numero de degraus.

O mecanismo esta na truncagem, e ele tambem e indiferente ao sinal:

| | K>0 | K<0 |
|---|---|---|
| revocacao (multi que truncou) | 37/76 = 48,7 % | 35/73 = 47,9 % |
| precisao | 86,0 % | 58,3 % |
| erro de K quando truncou | 1,5 % | 1,5 % |
| erro de K quando **nao** truncou | 64,2 % | 45,2 % |
| ...mas contra o K da excursao TOTAL | 17,7 % | 10,8 % |
| ...mais perto do total em | 35/39 | 34/38 |

Ou seja: **metade das figuras multi-degrau nao e truncada, e nessas o `K`
reportado descreve a excursao acumulada inteira** (mais perto do total em
69 de 77). Nao e ruido — e a resposta a outra pergunta, entregue sem aviso.

## 9. O segundo problema, que nao e nenhum dos dois: resposta menor que o degrau desenhado

Sai do cruzamento um terceiro fator, mais forte que o sinal:

**Quando `K_planta < 1` — a resposta assenta ABAIXO da linha de entrada que os
tres geradores desenham no mesmo quadro — a recusa quadruplica.**

| estrato | `K_planta<1` | `K_planta>=1` |
|---|---|---|
| K>0 · 1 degrau | 12/24 = 50,0 % | 2/28 = 7,1 % |
| K>0 · 2+ degraus | 2/24 = 8,3 % | 2/56 = 3,6 % |
| K<0 · 1 degrau | 6/25 = 24,0 % | 3/62 = 4,8 % |
| K<0 · 2+ degraus | 1/20 = 5,0 % | 6/60 = 10,0 % |

**CMH sobre os 4 estratos: OR = 3,91, p = 0,0007.**

Todas as 23 recusas de 1 degrau sao `ajuste_inconsistente` (1 e
`resposta_inversa`) — o codigo ja diz o que e: *"a mascara provavelmente saltou
para um distrator"*. E o **defeito 4** que `rg_negativo.py` documenta na
docstring, agora medido: quanto menor a resposta em relacao ao degrau plotado,
mais a polilinha pula entre os dois objetos. Vale notar que a guarda **esta
funcionando**: entre as que passam, o erro nao e pior (NRMSE mediano 0,0070 com
`K_planta<1` contra 0,0158 com `K_planta>=1`). O custo e cobertura, nao
exatidao.

## 10. Ordem de ataque, revisada

1. **Revocacao da truncagem — 48 %.** Vale 77 amostras das 299 saindo de ~50 %
   de erro em `K` para ~1,5 %. Indiferente ao sinal, entao e uma correcao so.
2. **Separar resposta de linha de entrada quando `K_planta < 1`** (OR=3,91,
   p=0,0007). Custa 23 recusas em 139 figuras de um degrau. E o envelope que o
   ARQUITETURA.md ja prevê ("recuperar `K_planta` exige ler a amplitude do
   degrau da imagem").
3. **Truncagem espuria em 1 degrau** — 31/116 = 26,7 %. Por sinal, 15,8 % (K>0)
   contra 32,1 % (K<0), Fisher **p=0,076: nao significativo**, e o corte por
   tema sugere que o confundidor e o render escuro da familia `neg`
   (27,3 %/37,5 % no escuro contra 11,1 %/26,3 % no claro), nao o sinal.
4. **Guarda de `zeta` no limite da caixa** — 6 casos, barata.

**Nao entra na lista: ganho negativo.** Em toda medida testada, com controle,
K<0 e indistinguivel de K>0.

---

# Adendo 2: mecanismo de cada um dos quatro erros

Fontes: `varre_piso_truncagem.py` (varredura do piso + cache `series.npz`),
`cruza_sinal_degraus.py`, e as series extraidas das 297 figuras que calibraram.
Cada afirmacao abaixo aponta a linha que decide.

## 11. Erro 1 — revocacao da truncagem: quem barra e o PISO, nao o ganho

`identify_com_truncagem` (`identify/classical.py:1085`) tem **dois portoes em
serie**:

```python
if full.nrmse <= _PISO_SUSPEITA:      # 0.030, linha 1036
    return ...                        # nem varre cortes
...
if max(ganhos) < _GANHO_MIN:          # 0.60, linha 1066
    return ...                        # varreu, nao aceitou
```

Medido nas 158 figuras multi-degrau que entregaram fisica: **das que nao foram
truncadas, 80,7 % morreram no portao 1** — o ajuste de degrau unico aplicado a
curva de dois degraus tem residuo **abaixo** de 0,030, entao a varredura de
cortes nem roda. So 19,3 % chegaram ao portao 2.

A razao e geometrica: dois degraus **de mesmo sinal** produzem uma curva que
continua monotona e em forma de S. Um FOPDT com `K` maior e `tau` maior a
reproduz com ~2 % de RMS. **O residuo nao carrega a informacao de "ha dois
degraus aqui"** quando os degraus tem o mesmo sinal. Confirmado pela razao
entre degraus: `|U2/U1|` mediano 0,535 nos nao detectados contra 0,771 nos
detectados — quanto menor o segundo degrau, menor a perturbacao, menor o
residuo, mais invisivel.

### Varredura do piso (297 amostras, `_GANHO_MIN` fixo em 0,60)

| piso | TP | FN | FP | TN | revocacao | precisao | varre cortes |
|---|---|---|---|---|---|---|---|
| **0,030** (producao) | 74 | 84 | 31 | 108 | **46,8 %** | 70,5 % | 155/297 |
| 0,020 | 98 | 60 | 31 | 108 | 62,0 % | 76,0 % | 184/297 |
| **0,010** | 110 | 48 | 31 | 108 | **69,6 %** | **78,0 %** | 211/297 |
| 0,007 | 111 | 47 | 31 | 108 | 70,3 % | 78,2 % | 221/297 |
| 0,000 | 111 | 47 | 31 | 108 | 70,3 % | 78,2 % | 297/297 |

**O `FP` nao se mexe: 31 em todos os pisos.** Os falsos positivos ja tem
residuo bem acima de 0,030 (mediana 0,131), entao baixar o piso nao cria nenhum
novo. Neste corpus o piso **so custa revocacao**, sem oferecer protecao.

Baixar para 0,010 levaria 36 figuras multi-degrau de ~50 % de erro em `K` para
~1,5 %, com precisao SUBINDO de 70,5 % para 78,0 %. O custo e de tempo: 13
ajustes extras em 211 imagens em vez de 155.

**Ressalva obrigatoria.** O piso foi calibrado em `data/test` (p98 = 0,0246,
dispara em 1,91 %), nao aqui. A tabela acima e do corpus de render REAL destes
tres geradores. Antes de mexer em producao, remedir em `data/test` — e o que a
propria docstring da constante exige.

Os 47 que sobram com piso = 0 sao irredutiveis por limiar: `melhor_ganho`
negativo (o `rg_04` da −0,054). Nenhum prefixo ajusta melhor, porque a curva de
dois degraus e genuinamente indistinguivel de uma de um degrau so. Para esses,
so lendo a linha de entrada do grafico.

## 12. Erros 2 e 3 sao O MESMO DEFEITO em duas severidades

Este e o achado que reorganiza a lista. Medido nas 116 figuras de **um degrau**
que entregaram fisica, cruzando o erro da serie EXTRAIDA contra a verdade
analitica:

| extracao | n | truncagem espuria |
|---|---|---|
| limpa (NRMSE < 0,05) | 79 | **0/79 = 0,0 %** |
| suja (NRMSE >= 0,05) | 37 | **31/37 = 83,8 %** |

Separacao perfeita. **A truncagem espuria nao e um defeito da truncagem** — ela
esta reagindo corretamente a uma serie corrompida. O defeito esta antes.

### Para onde a polilinha pula

Das 60 figuras de um degrau com extracao suja, classificando o nivel do trecho
corrompido:

| para onde pula | n | recusou | truncou | passou |
|---|---|---|---|---|
| nivel da ENTRADA (patamar `U`) | 38 | 20 | 14 | 4 |
| entre os dois objetos (mediana da coluna) | 12 | 0 | 10 | 2 |
| nivel ZERO (entrada antes do degrau) | 8 | 1 | 7 | 0 |
| perto da resposta (ruido) | 2 | 2 | 0 | 0 |

**58 de 60 — 97 % — sao a linha de entrada desenhada.** Nao ha um "defeito da
recusa" e um "defeito da truncagem": ha UM defeito, e o desfecho depende de
quanto da serie ele estraga.

- estraga muito -> residuo > `_NRMSE_MAX = 0,13` -> `ajuste_inconsistente`
  (`identify/pipeline.py:463`). Sao as **23 recusas**.
- estraga um trecho curto e tardio (posicao mediana 0,88 da janela, pico de
  0,875 da faixa, so 3,6 % das colunas) -> residuo entre 0,030 e 0,13 -> abre o
  portao 1 da truncagem, um prefixo que exclui o pulo ganha ~0,95 -> **trunca**.
  Sao os **31 falsos positivos**, e o `K` sai de um prefixo que nao assentou.

### Por que `K_planta < 1` e o gatilho

Os tres geradores plotam a entrada no mesmo quadro, e os limites de `y` sao
fixados pelo maior dos dois objetos. Quando `K_planta < 1` a resposta assenta
**abaixo** do patamar da entrada e ocupa so uma fracao vertical do quadro.
Chamando `frac = |K| / max(|K|, |U|)`:

| `frac` | n | polilinha vai parar na ENTRADA | residuo p50 | recusas |
|---|---|---|---|---|
| 0 – 0,5 | 22 | **8/22 = 36,4 %** | 0,1270 | 9 |
| 0,5 – 0,8 | 19 | 6/19 = 31,6 % | 0,1084 | 7 |
| 0,8 – 1,0 | 8 | 1/8 = 12,5 % | 0,0037 | 2 |
| = 1,0 | 90 | **7/90 = 7,8 %** | 0,0043 | 5 |

Gradiente monotono, 4,7x entre as pontas. Spearman `frac` x residuo =
**−0,328 (p = 7,9e-5)**; contra a altura da resposta em pixels, −0,327.

O mecanismo esta em `mask_to_polyline` (`identify/polyline.py:110-129`): quando
uma coluna tem **um** bloco de tinta, o codigo usa a **mediana de todas as
linhas** daquela coluna; a desambiguacao por ramos so entra quando ha dois
blocos separados por mais de `3x` a espessura mediana. O degrau de entrada
desenhado tem um **flanco vertical continuo** que liga os dois objetos: naquela
coluna existe UM bloco so, indo de `0` ate `U`, e a mediana cai no meio do
caminho. O ponto `anterior` fica entao mais perto da entrada do que da
resposta, e a partir dali a heuristica de "seguir o bloco mais proximo" segue o
objeto errado com confianca.

Isto e o **defeito 4** que a docstring do `rg_negativo.py` ja nomeia — agora
medido, com dose-resposta.

O corte por tema confirma que o sinal de K nao e a variavel: fundo escuro
18/51 = 35,3 % de truncagem espuria contra fundo claro 13/65 = 20,0 %, com
NRMSE de extracao mediano 0,0095 contra 0,0041. A familia `neg` desenha a
entrada com `plt.step` **mais** um `axvspan` sobre a faixa do atraso — mais
tinta estranha no quadro, mais chance de pulo.

**A guarda esta fazendo o trabalho dela.** Entre as que passam com `frac < 0,5`
o erro nao e pior; o custo do defeito e cobertura, nao numero errado em
silencio.

## 13. Erro 4 — `zeta` no limite da caixa e sintoma de `K` nao observavel

15 de 265 entregas (5,7 %) terminam com `zeta` encostado em
`ZETA_BOUNDS = (1e-3, 10.0)` (`identify/classical.py:70`): 11 no piso, 4 no
teto. Encostar no limite nao e um valor — e o otimizador dizendo que nao existe
minimo interior.

Os quatro do teto sao os piores da amostra inteira:

| amostra | `zeta` | `K` ajustado | `K` verdadeiro | erro | nrmse | truncou |
|---|---|---|---|---|---|---|
| `bal_K-_1deg_15` | 10 | −734,3 | −5,05 | −144x | 0,0137 | sim |
| `bal_K-_2deg_19` | 10 | −139,6 | −1,26 | −110x | 0,0121 | sim |
| `neg_11` | 10 | −569,6 | −10,12 | −55x | 0,0182 | sim |
| `bal_K+_1deg_11` | 1e-3 | 3344 | 0,559 | +5984x | 0,0381 | sim |

**Os 15 tem residuo mediano de 0,0182 — os ajustes parecem otimos.** Nenhum
chega perto do `_NRMSE_MAX = 0,13`, entao a guarda existente nunca dispara.
E esse e o ponto: um modelo degenerado casa com um trecho parcial de curva
*perfeitamente*. Residuo baixo nao e evidencia de `K` correto.

### O sinal que separa os 15 sem ambiguidade

Definindo **cobertura** = `(janela ajustada − theta) / t_dom DO MODELO
AJUSTADO` — quantas constantes de tempo do proprio modelo cabem na janela que
ele viu:

| | p10 | p50 | p90 |
|---|---|---|---|
| `zeta` no limite (n=15) | 0,002 | **0,003** | 0,007 |
| `zeta` livre (n=250) | 0,624 | **1,890** | 4,438 |

Tres ordens de grandeza de separacao, sem nenhuma sobreposicao. O modelo esta
dizendo "sou 300x mais lento do que a janela que voce me deu" — ou seja, o
patamar nunca aparece e `K` foi **extrapolado**, nao medido.

| cobertura | n | \|erro K\| med | \|erro K\|>50 % | nrmse med | `zeta` no limite |
|---|---|---|---|---|---|
| 0 – 0,5 | 35 | **1,3384** | 22/35 | 0,0193 | 15/35 |
| 0,5 – 1 | 25 | 0,1567 | 8/25 | 0,0083 | 0/25 |
| 1 – 2 | 86 | 0,0148 | 14/86 | 0,0080 | 0/86 |
| 2 – 4,6 | 100 | 0,0050 | 13/100 | 0,0058 | 0/100 |
| >= 4,6 | 19 | 0,0497 | 3/19 | 0,0045 | 0/19 |

Uma guarda `cobertura < 0,5` marcaria 35 de 265 entregas (13,2 %), das quais 22
tem erro de `K` acima de 50 % — **precisao 63 %**, contra os 5,7 % que a
deteccao por "`zeta` no limite" alcanca. E mais geral que olhar o limite da
caixa: pega tambem os casos de `zeta` interior com `K` extrapolado.

## 14. A lista, reescrita pelo mecanismo

| # | defeito | onde | evidencia | efeito |
|---|---|---|---|---|
| 1 | `_PISO_SUSPEITA = 0.030` fecha o portao antes da varredura | `classical.py:1036` | 80,7 % dos nao-truncados; FP invariante na varredura | 0,010 leva a revocacao de 46,8 % a 69,6 % **e** a precisao de 70,5 % a 78,0 % |
| 2 | polilinha pula para a linha de entrada desenhada | `polyline.py:110-129` | 58/60 das extracoes sujas; dose-resposta em `frac`, rho=−0,328 | 23 recusas **e** 31 truncagens espurias — o mesmo defeito |
| 3 | sem guarda de `K` extrapolado | `pipeline.py:459` | cobertura 0,003 contra 1,89, sem sobreposicao | 15 ajustes com residuo otimo e `K` ate 5984x errado |
| 4 | curva de dois degraus de mesmo sinal e indistinguivel de um degrau | estrutural | `melhor_ganho` negativo em 47 casos com piso = 0 | irredutivel por limiar; exige ler a entrada |

O que era "quatro problemas" e, no mecanismo, **tres**: um limiar mal
posicionado, **um** defeito de extracao com duas caras, e uma guarda que falta.
O quarto e um limite de informacao, nao um bug.

---

# Adendo 3: sondagem — dá para ler a linha de entrada sem retreinar?

Fontes: `sonda_linha_entrada.py` (detector heuristico), `sonda_teto_cor.py`
(oraculo). Saidas em `sondagem_entrada.{txt,json}` e `sondagem_teto.{txt,json}`.

## 15. Por que a sondagem

A proposta de ler o degrau de entrada dependia de uma premissa que **estava
errada** e foi corrigida por medicao antes de virar spec: eu havia afirmado que
"a mascara ja ve essa linha, so esta fundida com a resposta". Medido nas 297
figuras que calibram:

| | p10 | p50 | p90 | >50 % |
|---|---|---|---|---|
| colunas da linha de entrada com mascara acesa | 0,014 | **0,079** | 0,314 | 11/297 |
| tinta da mascara que cai sobre a linha de entrada | 0,013 | **0,081** | 0,254 | 43/297 |

A mascara **suprime** a linha de entrada, e o retreino do §41 funcionou. O que
sobra e um vazamento de ~8 % das colunas — suficiente para a polilinha pular
(o trecho corrompido sao 3,6 % das colunas), longe de ser segmentacao.

Logo a informacao nao esta disponivel de graca. A pergunta virou: **da para
extrai-la da imagem RGB por visao classica, sem retreinar?**

## 16. Detector heuristico de escada

Criterio estrutural, nao de cor: `u(t)` assume exatamente `n_degraus + 1`
valores. Se a mediana por coluna de um objeto e explicada por poucos NIVEIS
PLANOS, o objeto e uma escada. Tres refinamentos, cada um motivado pelo erro
medido do anterior:

| versao | precisao | revocacao | F1 | achou escada |
|---|---|---|---|---|
| niveis planos + cobertura | 56,5 % | 43,8 % | 0,493 | 62,2 % |
| \+ nitidez (transicao vertical) | 52,8 % | 41,2 % | 0,463 | 51,2 % |
| \+ niveis disjuntos em x (tira a legenda) | 63,6 % | 21,9 % | 0,326 | 27,1 % |
| **referencia: gate de ganho, sem piso** | **78,2 %** | **70,3 %** | **0,740** | — |

Nenhuma versao chega perto do gate que ja existe. O padrao de erro dominante
era um vies de **+1 nivel** (33 figuras de 1 degrau lidas como 2, 27 de 2 lidas
como 3): a AMOSTRA DE LINHA DA LEGENDA tem a mesma cor da entrada, fica num y
proprio, e entra como um degrau a mais. Exigir niveis disjuntos em x corrige o
vies e melhora a precisao — mas derruba a deteccao para 27 %.

## 17. O teto: a cor isola a linha de entrada?

Um detector fraco nao prova que o sinal nao esta la. Este oraculo mede o TETO:
a trajetoria verdadeira da entrada vem de `verdade.json` + calibracao, e para
cada cor da figura se mede quanto dela cai sobre a trajetoria. Nenhuma
heuristica de cor pode superar isso.

| métrica | p10 | p50 | p90 |
|---|---|---|---|
| F1 (melhor cor contra a linha real) | 0,356 | **0,647** | 0,810 |
| precisão (tinta da cor sobre o alvo) | 0,438 | 0,653 | 0,887 |
| revocação (colunas cobertas) | 0,282 | 0,636 | 1,000 |

| limiar | figuras com uma cor que isola a entrada |
|---|---|
| F1 >= 0,95 | **1/297 (0,3 %)** |
| F1 >= 0,90 | 10/297 (3,4 %) |
| F1 >= 0,80 | 32/297 (10,8 %) |

E nao e artefato da quantizacao. Refinando de `//32` ate a **cor exata**
(n=120):

| quantização | F1 p50 | >= 0,90 |
|---|---|---|
| //32 | 0,651 | 3,3 % |
| //8 | 0,676 | 9,2 % |
| //4 | 0,677 | 9,2 % |
| **//1 (cor exata)** | **0,675** | **9,2 %** |

O teto nao se move. **O limite nao e a granularidade da cor — e a cor nao
separar os objetos.** Duas causas, as duas estruturais:

1. **A entrada divide a cor com a moldura do grafico.** Preto no tema claro e
   branco no escuro sao tambem o texto da legenda, os rotulos, os spines e a
   grade. Precisao mediana 0,653 — um terco da tinta daquela cor nao e a linha.
2. **A entrada e tracejada nas TRES familias** — `rg.py:64`, `rg_negativo.py:104`,
   `rg_multidegrau.py:95`, todas com `linestyle='--'`. Uma tracejada nao existe
   nos vaos, o que limita a revocacao por construcao (mediana 0,636). Isso nao e
   um estresse que a geracao aleatoria inventou: e como os geradores reais
   desenham.

## 18. Veredito da sondagem: **o retreino e necessario**

A separacao classica por cor tem teto de F1 ~0,65 na SEGMENTACAO — antes mesmo
de contar degraus. O melhor detector heuristico construido sobre ela chega a
F1 0,326-0,493 na contagem, contra 0,740 do gate de ganho que ja existe.
Investir engenharia nessa direcao seria construir sobre um teto baixo e medido.

**O que a sondagem custou e o que ela poupou.** Custou tres iteracoes de um
detector e um oraculo. Poupou a construcao do corpus multi-degrau, o head de 2
canais e o retreino, que teriam sido feitos com a justificativa errada ("a
mascara ja ve, e so separar"). A justificativa correta e outra e e mais forte:
a linha de entrada **nao e separavel por cor**, entao o unico jeito de
recupera-la e ensinar a rede a segmenta-la — que e exatamente o que um segundo
canal de saida faz.

Ordem de ataque, com a sondagem incorporada:

1. **Baixar `_PISO_SUSPEITA` de 0,030 para ~0,010.** Uma constante. Revocacao
   46,8 % -> 69,6 % e precisao 70,5 % -> 78,0 %, FP invariante. Remedir em
   `data/test` antes.
2. **Guarda de cobertura (`< 0,5 t_dom`)** para `K` extrapolado. Precisao 63 %,
   pega os 15 casos de `zeta` no limite e mais.
3. **Head de 2 canais + corpus multi-degrau.** Agora com justificativa medida.
   E o unico caminho para o defeito de extracao, para `K_planta` e para a
   contagem exata de degraus — e nao ha atalho classico.

**O item 3 e o unico que resolve o defeito de extracao, e ele exige retreino.**
Os itens 1 e 2 sao ajustes de constante e valem ser feitos antes, porque sao
baratos e independentes.

---

# Adendo 4: e as figuras que NÃO desenham o degrau de entrada?

Fonte: `rg_aleatorio.py --entrada {omite,omite_fit}` (par controlado) e
`reports/amostras_aleatorias/sem_omite*/`.

## 19. O que a rede entregaria

Com o head de 2 canais, numa figura sem a entrada desenhada o **canal 1 sai
vazio** — que e a saida correta, nao uma falha. A pipeline cai no caminho de
hoje: a truncagem por ganho de prefixo decide sozinha se houve mais de um
degrau.

O que **nao muda**: o canal 0 e a resposta, alvo identico ao de hoje. Estrutura,
`K`, `tau`/`wn`/`zeta` e `theta` saem exatamente como saem agora. A mudanca e
estritamente **aditiva** — nao ha regressao possivel nessa populacao.

O que **nao se ganha**, e continua irrecuperavel: contagem de degraus pela
entrada, `U`, e portanto `K_planta`; e a separacao entre tempo morto e instante
do degrau. `K` segue sendo `K_planta x U` por convencao e `theta` segue sendo o
instante de partida, como o ARQUITETURA.md ja declara.

## 20. Medido: a figura SEM entrada é a mais fácil

Par controlado — mesmas plantas, mesmos estilos, mesmas sementes; a unica
variavel e a linha estar desenhada.

| lote | n | entrega física | estrutura | NRMSE p50 | \|erro K\| p50 | recusas |
|---|---|---|---|---|---|---|
| COM entrada (como hoje) | 100 | 83,0 % | 88,1 % | 0,0073 | **0,0288** | 17 |
| SEM entrada, mesmos eixos | 100 | **91,0 %** | 90,2 % | 0,0049 | **0,0082** | **9** |
| SEM entrada, reenquadrada | 100 | 91,0 % | 88,0 % | 0,0033 | 0,0090 | 9 |

Na atribuicao limpa (eixos identicos): **11 figuras passam de recusada a
aceita**, 3 fazem o contrario, e **6 truncagens espurias somem**. O erro de `K`
cai 3,5x. Isso confirma o defeito nº 2 por um par controlado, e nao so por
correlacao: a linha de entrada e o distrator, e tira-la resolve.

As 3 que pioram sao instrutivas — `resposta_inversa` sobe de 1 para 3. Sem a
tracejada a faixa de `y` da serie extraida encolhe, e ela e o DENOMINADOR de
`_undershoot`; o mesmo undershoot absoluto vira uma fracao maior e cruza
`_UNDERSHOOT_MAX`. E exatamente o modo de falha que a docstring da constante
ja registra ("a máscara melhor captura mais faixa de y, que é o DENOMINADOR
desta métrica"), reaparecendo por outra porta.

## 21. A interação que muda o plano de implantação

| | precisão | revocação |
|---|---|---|
| COM entrada | 70,3 % | **56,5 %** |
| SEM entrada, mesmos eixos | 78,3 % | **40,0 %** |
| SEM entrada, reenquadrada | 77,3 % | 38,6 % |

A precisao sobe, como esperado — mas a **revocacao CAI de 56,5 % para 40,0 %**.

O motivo e mecanico: a linha de entrada inflava o residuo do ajuste inteiro, e
era esse residuo inflado que abria o portao `_PISO_SUSPEITA`. O distrator
estava, por acidente, ajudando a detectar multi-degrau. Limpar a extracao tira
essa ajuda.

**Consequencia direta:** corrigir o defeito de extracao SOZINHO piora a
deteccao de multi-degrau. A etapa 1 do retreino tem de ser implantada
**junto** com a baixa do `_PISO_SUSPEITA` (item 1 da lista), nao antes nem
depois. As duas mudancas nao sao independentes — sao uma so.

---

# Adendo 5: corrigir a truncagem espúria SEM retreino

Fontes: `testa_guarda_continuidade.py`, `compara_guarda_e2e.py`. Saidas em
`guarda_continuidade.json`, `e2e_guarda.json`, `guarda_e2e.txt`.

## 22. A causa raiz, numa linha

`identify/polyline.py:114`, no caminho de RAMO UNICO:

```python
if not multi_ramo or anterior is None:
    v = float(np.median(linhas))     # aceita o bloco, esteja onde estiver
```

Numa coluna com um bloco so o codigo aceita incondicionalmente, por mais longe
que o bloco esteja de `anterior`. Quando a curva de saida e TRACEJADA, no vao
do tracejado a unica tinta da coluna e a linha de entrada: o bloco unico e o
distrator, ele vira o novo `anterior`, e da coluna seguinte em diante a logica
de "siga o bloco mais proximo" segue o objeto errado com confianca.

Evidencia (n=139 de um degrau): curva SOLIDA 8/43 ruins (18,6 %), tracejada ou
pontilhada 46/96 (47,9 %). **Fisher OR=0,248, p=0,0012.** Se o mecanismo fosse
"a mascara confunde os objetos por proximidade", o estilo da linha nao
importaria.

## 23. A guarda

Numa coluna de ramo unico, se o bloco esta longe demais de `anterior`, **pule a
coluna e preserve `anterior`**. O vao deixa de corromper a referencia, a
interpolacao final (que ja existe) cobre o buraco, e quando a tinta da curva
volta a referencia ainda esta correta. Limiar em multiplos da espessura mediana
— a mesma escala que o resto da funcao ja usa.

Qualidade da serie extraida, n=297:

| limiar (× espessura) | NRMSE p50 | extrações sujas (>= 0,05) |
|---|---|---|
| **sem guarda (produção)** | 0,0062 | **115/297 (38,7 %)** |
| 12 | 0,0038 | 48/297 (16,2 %) |
| **8** | 0,0038 | 48/297 (16,2 %) |
| 5 | 0,0037 | 47/297 (15,8 %) |
| 3 | 0,0037 | 47/297 (15,8 %) |

A reducao se concentra onde o mecanismo exige — `-` 16/81 -> 13/81 (−19 %),
`--` 35/77 -> 12/77 (−66 %), `-.` 31/70 -> 11/70 (−65 %), `:` 33/69 -> 11/69
(−67 %). A solida quase nao muda: ela nao tem vaos.

**Custo:** 113 figuras melhoram, 5 pioram, nenhuma perde a polilinha. As 5 que
pioram JA ESTAVAM SUJAS (0,1013 -> 0,3057; 0,3577 -> 0,3990; 0,5199 -> 0,5458)
— a guarda nao quebra nenhuma extracao que estava limpa.

**Banda:** 3 vs 5 diferem em 42/297; 5 vs 8 em 9; 8 vs 12 em apenas 4. O salto
que a guarda rejeita e ordens de grandeza maior que o limiar, entao existe um
plato estavel de ~5 a ~12. **8 fica no centro**, pela mesma disciplina que o
`_GANHO_MIN` documenta.

## 24. Efeito ponta a ponta (limiar 8), n=299

| | entrega física | recusas | \|erro K\| p50 | trunc. espúria |
|---|---|---|---|---|
| produção | 88,6 % | 34 | 0,0354 | 31/139 |
| **com guarda** | **95,7 %** | **13** | 0,0243 | **17/139** |

So a populacao de UM degrau: entrega **83,5 % -> 95,7 %**, recusas **23 -> 6**,
`ajuste_inconsistente` **28 -> 7** no total.

Figura a figura: **22 recusadas viram aceitas**, 1 vira recusada
(`bal_K-_1deg_13`), **20 truncagens espurias somem**, **nenhuma nova aparece**.

O resultado que mais importa:

| núcleo (1 degrau, sem truncagem espúria) | n | \|erro K\| p50 |
|---|---|---|
| produção | 85 | 0,0024 |
| **com guarda** | **116** | **0,0024** |

**31 figuras a mais entram no nucleo bom, com a MESMA exatidao.** Nao e um
ganho medio diluido: e populacao nova chegando no mesmo padrao.

## 25. O custo, e por que ele obriga a mudanca combinada

| | precisão | revocação | F1 |
|---|---|---|---|
| produção | 69,9 % | **48,3 %** | 0,571 |
| com guarda | 76,4 % | **35,9 %** | 0,489 |

A revocacao de multi-degrau CAI, e o erro de `K` nas multi-degrau piora de
0,2074 para 0,3311. E exatamente o efeito que o par controlado do Adendo 4 ja
previa: o distrator inflava o residuo, e era esse residuo inflado que abria o
portao `_PISO_SUSPEITA`. Limpar a extracao tira essa ajuda acidental.

Medido: com a guarda, o piso barra **87,8 %** dos multi-degrau nao detectados
(era 80,7 % em producao). O portao ficou ainda mais dominante.

**Conclusao operacional: a guarda e a baixa do `_PISO_SUSPEITA` sao UMA
mudanca, nao duas.** Implantar a guarda sozinha troca 21 recusas e 20
truncagens espurias por 21 multi-degrau a mais passando batido. Implantar as
duas juntas deveria ficar com os dois ganhos — mas isso ainda NAO foi medido:
exige refazer a varredura do piso sobre a serie ja guardada.

## 26. O que a guarda NAO resolve

A curva solida mantem 13/81 extracoes sujas mesmo com a guarda. Existe um
segundo mecanismo, menor, que ela nao cobre — provavelmente colunas em que os
dois objetos se cruzam de fato e a mediana do bloco unico cai entre eles. Para
esse residuo, so a segmentacao em dois canais.

---

# Adendo 6: plano de ataque do retreino, e o que ele custa à base

Levantado a partir de `dataset/generator.py`, `train_unet.py`,
`retreino_kneg.sh` e `seleciona_checkpoint.py` — o precedente do retreino de
ganho negativo (§40-41) é o molde.

## 27. O padrão que o repositório já impõe

Todo estrato novo (`reta_no_patamar`, `janela_assentada`, `anotacao_com_seta`,
`banda_de_acomodacao`, `ganho_negativo`) segue a mesma disciplina, e o
multi-degrau tem de segui-la:

1. **Flag booleana, padrão `False`**, passada por `generate_sample` ->
   `render_sample`, ou aplicada ao `spec` DEPOIS de `sample_style`.
2. **Corpus base byte a byte idêntico** — `test_o_padrao_nao_muda_um_byte`
   (`tests/part2/test_estrato_ganho_negativo.py:36`) compara os PNG e o
   `meta.json`. Mexer no sorteio de `sample_system` moveria toda amostra do
   corpus e com ela todo número histórico das Partes 1 e 2.
3. **Diretório próprio** `data/train_<nome>`, entrando na lista de
   `--train-dir`; `data/val_<nome>` no `--val-dir`.
4. **Não promover no fim do treino.** O `train_unet.py` guarda o melhor por
   `IoU_val`, e o `IoU_val` é quase cego a defeito localizado — no caso do
   platô, custava 5 pontos de IoU enquanto a cobertura do platô desabava 42.
   A escolha é feita depois, por um script que mede o defeito-alvo.
5. **Anti-vazamento:** `sample_style` não pode ver o `spec`. Um estilo que
   mudasse com o número de degraus ensinaria a rede a ler o rótulo do render.

Custo do precedente: 10.950 amostras, `base=32`, `in_ch=3`, 25 épocas,
`--batch 6` (batch 8 estoura VRAM), ~5,7 h.

## 28. Três descobertas que mudam o desenho

**(a) O corpus não tem como ensinar a linha de entrada hoje — e não é
descuido, é estrutural.** No `SystemSpec` o degrau é aplicado em `t=0` e
`theta` é o tempo morto. Se você desenhar a entrada desse corpus, sai uma
**reta horizontal**, não uma escada: o degrau vertical fica fora do quadro, na
borda esquerda. Os três geradores reais aplicam o degrau DENTRO da janela
(`rg_negativo.py` em t=2 s de uma janela 0-15 s). Logo, para treinar o canal de
entrada é preciso **separar o instante do degrau do tempo morto** — um campo
novo no `SystemSpec`. E essa é exatamente a grandeza que o canal 1 recuperaria.

**(b) Multi-degrau quebra o contrato do `meta.json`.** `params` tem UM `K` e UM
`theta`; `step_amplitude` é escalar. Uma amostra de dois degraus precisa de
lista. Isso é `SCHEMA_VERSION: 1 -> 2`, e os testes-oráculo da Parte 1, que
comparam parâmetros ajustados contra o `meta`, precisam ou pular as amostras
multi-degrau ou comparar só contra o PRIMEIRO degrau (a convenção que o
`rg_multidegrau.py` já usa).

**(c) A guarda de continuidade tornou obsoleta a justificativa original.** O
defeito de extração se resolve sem retreino (Adendo 5). O que sobra para o
retreino é a DETECÇÃO de degraus, e ela é agora o resíduo dominante: 98/153 =
64,1 % dos multi-degrau passam batidos mesmo com a guarda.

## 29. O plano, em etapas independentes

### Etapa 0 — sem retreino (fazer primeiro)

Guarda de continuidade em `polyline.py:114` **junto com** a baixa do
`_PISO_SUSPEITA`. Medidas em separado; a combinação ainda não. Remedir as duas
em `data/test` antes de promover.

Impacto na base: **nenhum**. Impacto na U-Net: **nenhum**.

### Etapa 1 — estrato multi-degrau + cabeça de contagem

O núcleo do retreino, e o mais barato dos dois que envolvem a rede.

**Base de dados:**

| item | mudança |
|---|---|
| `SystemSpec` | campo `degraus: tuple[(amplitude, instante), ...]`, default um degrau em `t=0` (compatível) |
| `sample_system` | inalterada no caminho padrão; sorteio de degraus só sob a flag |
| `generate_sample` | flag `multi_degrau: bool = False` |
| `render_sample` | superposição na série; `mask.png` continua sendo só a resposta |
| `meta.json` | `SCHEMA_VERSION` 1 -> 2; `params` passa a descrever o 1º degrau; lista `degraus` nova |
| corpus novo | `data/train_multi` (~3000-4500, balanceado 1/2/3 degraus), `data/val_multi`, **`data/test_multi`** |
| testes | Parte 1 pula ou reinterpreta multi-degrau; `test_o_padrao_nao_muda_um_byte` cobre a flag nova |

`data/test_multi` não é opcional: sem ele não há como medir a contagem.

**U-Net:** o encoder e o decoder não mudam. Entra uma cabeça de classificação
sobre o gargalo (`self.bott`): pooling global -> linear -> logits de
`n_degraus`. A perda vira `dice_bce_loss` + λ·cross-entropy. São poucos
milhares de parâmetros e nenhuma mudança na saída de segmentação.

**Seleção:** o `seleciona_checkpoint.py` mede cobertura de platô — não serve.
Precisa de um análogo que meça acerto de contagem em `data/test_multi`, e que
verifique que a segmentação **não regrediu** (compartilhar encoder entre
tarefas pode ajudar ou atrapalhar; é empírico).

**Evidência de que vale:** o detector de derivada feito à mão chega a F1 0,923
na curva ideal, contra 0,740 do teto da heurística de resíduo. O sinal está na
forma da curva.

**Risco medido:** o subconjunto subamortecido (ζ < 0,9) é o difícil — o mesmo
detector cai para revocação 58,3 % lá, porque oscilação imita re-aceleração. É
o primeiro corte a olhar na avaliação.

### Etapa 2 — canal de segmentação da entrada

Só se `K_planta`, `U` e a separação tempo morto / instante do degrau forem
requisito. É o mais caro e o menos urgente.

**Base de dados**, além de tudo da Etapa 1:

| item | mudança |
|---|---|
| `SystemSpec` | instante do degrau **separado** do tempo morto (ver 28a) |
| `render_sample` | desenha a entrada acumulada (`drawstyle="steps-post"`), cor/estilo/espessura sorteados |
| novo arquivo | `mask_input.png` — mesmo truque da segunda figura de `generator.py:598-609`, só a escada em branco sobre preto |
| `load_sample` | devolve a máscara nova |
| `MaskDataset` | empilha as duas máscaras num alvo de 2 canais |

**U-Net:** `identify/extract.py:76`, `nn.Conv2d(chs[0], 1, 1)` -> `2`. ~150
parâmetros. `dice_bce_loss` soma os canais.

**Pipeline:** caminho "canal 1 vazio -> cai na truncagem por ganho", porque a
maioria das figuras do mundo não plota a entrada — e `data/test` inteiro não
plota.

## 30. O que quebra, e o que tem de ser remedido

- **`SCHEMA_VERSION` 1 -> 2** e os testes-oráculo da Parte 1.
- **Toda constante a jusante foi calibrada contra o estágio A atual**, e as
  docstrings dizem isso explicitamente: `_PISO_SUSPEITA`, `_GANHO_MIN`,
  `_NRMSE_MAX`, `_UNDERSHOOT_MAX`, `_COBERTURA_MIN_MOLDURA`, `_N_REPOUSO`.
  Qualquer promoção de checkpoint obriga a remedir as seis.
- **VRAM.** O precedente já roda com folga de 0,73 GB e `--batch 6`; uma cabeça
  a mais é barata, mas o alvo de 2 canais da Etapa 2 dobra a memória do alvo.
  Testar com smoke antes de disparar 8 h de treino.

## 31. Ordem recomendada

1. **Etapa 0** — guarda + piso. Sem retreino, ganho medido (entrega 88,6 % ->
   95,7 %, recusas 34 -> 13), risco baixo.
2. **Medir a combinação** guarda + piso, que é o número que falta.
3. **Etapa 1** — estrato multi-degrau + cabeça de contagem. Ataca o resíduo
   dominante (64,1 % dos multi-degrau passam batidos) com evidência de teto
   alto (F1 0,923 na curva ideal).
4. **Etapa 2** — canal da entrada, só se `K_planta` virar requisito.

O que mudou desde a primeira versão deste plano: a Etapa 2 era a proposta
principal, justificada por consertar a extração. A guarda consertou a extração
de graça, e a medição do teto de separação por cor (Adendo 3) mostrou que a
Etapa 2 exige retreino de qualquer forma. Ela desceu para último — e a cabeça
de contagem, que era um adendo, virou o centro.

---

# Adendo 7: preparação do retreino, e a medição que o dispensa

## 32. O que foi implementado

| arquivo | mudança |
|---|---|
| `dataset/generator.py` | `SystemSpec.degraus`, superposição em `step_response`, `entrada_acumulada()`, `sorteia_degraus()`, flag `multi_degrau`, `SCHEMA_VERSION` 1→2 com `degraus`/`n_degraus`/`u_final` |
| `identify/extract.py` | `UNet(com_contagem=)`, `forward(x, com_contagem=)`, `gargalo()`, `conta_de_gargalo()`, `load_model` infere a cabeça do checkpoint |
| `train_unet.py` | rótulo binário no `MaskDataset`, flags `--contagem`/`--lambda-contagem`, perda conjunta, `acerto_contagem` por época |
| `retreino_multi.sh` | receita, herdando `--batch 6` e `expandable_segments` do precedente |
| `sonda_cabeca_contagem.py` | a sondagem de encoder congelado |

## 33. Verificações de integridade

**O corpus base não mudou um byte.** Regenerei amostras de `data/train`,
`data/val`, `data/test`, `data/train_kneg` e `data/train_reta` a partir da
seed gravada no próprio `meta.json` e comparei com o que está em disco:
`image.png` e `mask.png` **idênticos por SHA-256** em todos os casos. A única
divergência de `meta` é pré-existente e aditiva (`has_annotation_arrow`,
`has_reference_line`, `has_settling_band`, de estratos posteriores ao corpus).
`data/train_reta` reproduz com `reta_no_patamar=True, janela_assentada=True` —
a diferença restante era o `sample_id`, que é o nome do diretório.

Os 11 testes de `tests/part2/test_estrato_ganho_negativo.py` passam, incluindo
`test_o_padrao_nao_muda_um_byte`.

**Compatibilidade da rede.** `forward` sem o flag devolve tensor, não tupla —
`predict_mask` e o laço de treino ficam intactos. `load_model` carrega o
checkpoint promovido sem a cabeça (`load_state_dict` continua ESTRITO). O
treino sem `--contagem` reporta 1.942.577 parâmetros e o mesmo `IoU_val` de
antes.

**Estrato validado.** 60 amostras de fumaça: distribuição 14/28/18 para 1/2/3
degraus, `|U2/U1|` mediana 0,586 (a faixa difícil real é 0,535), 2º degrau
visível na janela em 46/46, posição mediana 0,45 da janela.

## 34. A sondagem: o encoder congelado já carrega o sinal

Congelei `models/unet_stageA.pt`, extraí a ativação do gargalo de 1.900
amostras uma vez, e treinei **só a cabeça** sobre as features cacheadas.

**AUC = 0,9755**, com o encoder que **nunca viu uma figura multi-degrau**.

| ponto de operação | precisão | revocação | F1 |
|---|---|---|---|
| melhor F1 | 95,9 % | 95,9 % | **0,959** |
| precisão >= 95 % | 95,0 % | 96,2 % | 0,956 |
| precisão >= 98 % | 98,0 % | 93,7 % | 0,958 |
| — gate de resíduo (produção) | 69,9 % | 48,3 % | 0,571 |
| — gate de resíduo (sem piso) | 78,2 % | 70,3 % | 0,740 |
| — contagem PERFEITA como portão | 100 % | 70,3 % | 0,825 |

## 35. Dois defeitos do próprio experimento, achados antes de virar veredito

**(a) Acurácia é cega em população desbalanceada.** A primeira versão da
sondagem media acerto e concluiu *"as features NÃO carregam o sinal"*. Errado:
o estrato tem 81 % de multi-degrau, a cabeça colapsou na classe maioritária
(revocação 100 %, precisão 79,5 %) e o acerto ficou colado na taxa-base sem
dizer nada. Corrigido com `pos_weight` e AUC.

**(b) A cabeça precisa de normalização na entrada.** Depois de (a), o script
ainda dava AUC 0,555 enquanto uma medição avulsa dava 0,882. Bissecção com 5
sementes:

| | AUC por semente |
|---|---|
| sem normalizar | 0,864 · 0,857 · 0,883 · **0,500** · **0,500** |
| com `BatchNorm1d` na entrada | 0,965 · 0,966 · 0,967 · 0,979 · 0,975 |

Duas de cinco sementes colapsavam (ReLU morta) — as ativações do gargalo têm
escala por canal muito desigual e a cabeça é rasa demais para absorver isso. O
`BatchNorm1d` entrou na cabeça do modelo, não só na sondagem.

Sem a bissecção, o projeto teria disparado 6,5 h de GPU com a conclusão
invertida.

## 36. Consequência para o plano

**O treino conjunto pode ser dispensável para a contagem.** Com o encoder
congelado a cabeça já entrega F1 0,959, contra 0,740 do melhor gate atual e
acima até dos 0,825 da contagem perfeita como portão. E com o encoder
congelado **a segmentação não pode regredir** — os pesos não se movem.

Ressalva: os 0,9755 são **em distribuição**, medidos em `data/val_multi`, do
mesmo gerador do treino. Generalização para figura real é outra pergunta, e o
corpus real (`reports/amostras_aleatorias`) é o lugar de respondê-la.

Ordem revisada:

1. Guarda de continuidade + `_PISO_SUSPEITA` (sem retreino).
2. **Cabeça de contagem com encoder CONGELADO** — minutos de treino, risco zero
   para a segmentação. Validar no corpus real antes de ligar na pipeline.
3. Treino conjunto (`retreino_multi.sh`) — só se (2) não generalizar.
4. Canal de segmentação da entrada — só se `K_planta` virar requisito.

---

# Adendo 8: a cabeça congelada NÃO generaliza, e o estrato tinha um vazamento

Fonte: `valida_cabeca_no_real.py`, saída em `validacao_cabeca_real.txt`.

## 37. O teste, e o protocolo

A sondagem do Adendo 7 mediu AUC 0,9755 — mas **em distribuição**, em
`data/val_multi`, do mesmo gerador do treino. Este teste aplica a mesma cabeça
congelada às 299 figuras dos três geradores reais.

O limiar de decisão é escolhido em `data/val_multi` (sintético) e só depois
aplicado ao real. Escolhê-lo no próprio conjunto real seria selecionar no teste.

## 38. Resultado: não generaliza — antes e DEPOIS de corrigir o vazamento

| corpus de treino | AUC sintético | AUC real | F1 real |
|---|---|---|---|
| com o vazamento da §39 | 0,9794 | **0,6020** | 0,556 |
| **sem o vazamento (definitivo)** | **0,9919** | **0,6540** | **0,698** |
| — referência: gate de resíduo (produção) | — | — | 0,571 |
| — referência: gate de resíduo (sem piso) | — | — | **0,740** |
| — referência: contagem PERFEITA | — | — | 0,825 |

Corrigir o vazamento melhorou o real (0,602 -> 0,654) mas **não resolveu**: o
fosso entre 0,99 in-distribution e 0,65 no real continua enorme, e o F1 de
0,698 fica ABAIXO do gate de ganho sem piso (0,740) — uma heurística sem
aprendizado nenhum. **A cabeça congelada não é utilizável.**

No ponto de operação do melhor F1 sintético, a matriz real é TP=134 FP=90
FN=26 TN=49: revocação alta (83,8 %) comprada com precisão baixa (59,8 %). Ela
dispara demais.

Estratificado, o padrão não corresponde a nenhuma hipótese simples: `rg` 0,657,
`neg` 0,700, `multi` 0,677; e por tipo de linha vai de 0,502 (`--`) a 0,767
(`:`). Não é o tema nem o traço — é o domínio como um todo.

**A linha de entrada não é a causa.** Os lotes reais sem ela dão o mesmo:

| lote | n | AUC |
|---|---|---|
| real COM entrada | 100 | 0,549 |
| real SEM entrada, mesmos eixos | 100 | 0,550 |
| real SEM entrada, reenquadrada | 100 | 0,568 |

## 39. O vazamento que eu introduzi no estrato

Procurando a causa, encontrei um defeito no corpus que eu mesmo tinha acabado
de construir. A primeira versão de `sorteia_degraus` sorteava as separações e
depois **esticava `t_end`** para caber o último degrau:

```python
t_end = max(t_end, theta + ultimo + 1.5 * t_dom)   # <- o atalho
```

Efeito medido em 1900 amostras: **`janela/t_dom` sozinha separava 1 de 2+
degraus com AUC 0,822.** Uma cabeça treinada nisso aprende a ler "quantas
constantes de tempo cabem no quadro" em vez de "há uma re-aceleração".

É o mesmo tipo de defeito que a regra anti-vazamento de `randomize.py` existe
para impedir (`sample_style` não pode ver o `spec`), só que pelo eixo do TEMPO
em vez do estilo — e por isso a regra existente não o pegou.

**Correção:** a janela é sorteada PRIMEIRO, da mesma distribuição para qualquer
número de degraus, e os degraus são colocados dentro dela.

| candidato a atalho | AUC antes | AUC depois |
|---|---|---|
| **janela/t_dom** | **0,822** | **0,486** |
| t_dom | 0,508 | 0,491 |
| t_end | 0,592 | 0,488 |
| \|K\| | 0,518 | 0,491 |

Nota: o corpus REAL tem a mesma correlação (AUC 0,843) — é propriedade do
fenômeno, não defeito. Mas um modelo que se apoie nela é frágil, e o estrato de
treino não deve oferecê-la de graça.

## 40. Dois defeitos de experimento em sequência, e o que eles custaram

Nesta etapa, três medições seguidas estavam erradas antes de a quarta valer:

1. **Acurácia em população desbalanceada** → veredito "não há sinal", falso.
2. **Cabeça sem normalização** → 2 de 5 sementes colapsavam em AUC 0,50.
3. **Vazamento de janela no estrato** → AUC 0,98 in-distribution, 0,60 no real.

Nenhum dos três apareceria numa validação in-distribution. Os três foram
pegos por: medir com a métrica certa (AUC), repetir com várias sementes, e
**validar fora da distribuição de treino**.

Se o projeto tivesse ido direto ao `retreino_multi.sh`, teria gasto ~6,5 h de
GPU para produzir um modelo com validação excelente e comportamento de acaso
nas figuras reais — e o vazamento provavelmente teria sido creditado como
sucesso.

## 41. Consequência para o plano

- A etapa "cabeça com encoder congelado" está **descartada** pelo número real:
  F1 0,698 contra 0,740 de uma heurística sem aprendizado.
- O **treino conjunto** volta a ser necessário — agora sobre um estrato sem
  vazamento, que é o que ficou preparado.
- Mas ele NÃO deve ser disparado ainda. A queda no real não é explicada pela
  linha de entrada (medido) nem pelo tema nem pelo traço (medido). Falta
  identificar o componente de domínio responsável, e disparar 6,5 h de GPU
  antes disso é apostar. O próximo experimento barato: gerar um lote com a
  física do `rg_aleatorio.py` mas o RENDER de `sample_style` — se a AUC subir,
  o salto é de estilo e a correção é ampliar `randomize.py`; se não subir, é da
  física do estrato (instante do primeiro degrau em t=0, razões de amplitude) e
  a correção é em `sorteia_degraus`.

**O que esta etapa entrega, então:** o retreino inteiramente preparado e
verificado (corpus base intacto, compatibilidade de checkpoint, laço de treino,
receita, script de sondagem), um vazamento de corpus encontrado e corrigido
antes de custar GPU, e a evidência de que o atalho barato não funciona. O que
ela NÃO entrega é sinal verde para o treino — e essa é a conclusão correta a
partir do que foi medido.

---

# Adendo 9: por que a cabeça não generaliza — estilo ou física?

Fontes: `isola_estilo_vs_fisica.py`, `estilo_vs_fisica.txt`.

## 42. O par controlado

Reconstruí o `SystemSpec` **exato** de cada uma das 299 figuras reais (mesma
planta, mesmos degraus, mesma janela) e renderizei com `sample_style` — o
render do treino. A física fica idêntica à real; só o desenho muda.

| lote | AUC |
|---|---|
| sintético (física treino + render treino) | 0,9919 |
| **ESTE (física REAL + render treino)** | **0,7091** |
| real (física REAL + render real) | 0,6540 |

**Trocar o render recupera pouco: 0,654 -> 0,709.** O grosso do fosso até 0,99
é da FÍSICA do estrato, não do desenho. Ampliar `randomize.py` não é a
correção principal.

## 43. Qual dimensão da física está descoberta

| grandeza | sintético p10/p50/p90 | real p10/p50/p90 |
|---|---|---|
| janela/t_dom | 2,81 / 4,85 / 7,88 | 1,98 / 3,96 / 5,88 |
| **separação/t_dom** | **1,12 / 2,22 / 4,26** | **0,89 / 1,72 / 2,53** |
| \|U2/U1\| | 0,29 / 0,62 / 1,24 | 0,23 / 0,75 / 2,69 |
| posição do 2º degrau | 0,33 / 0,51 / 0,74 | 0,29 / 0,42 / 0,60 |
| theta/t_dom | 0,07 / 0,23 / 0,75 | 0,00 / 0,12 / 0,63 |

O estrato é **sistematicamente mais fácil**: separações maiores e janelas mais
longas. A AUC da cabeça no real, cortada por separação, confirma:

| separação/t_dom | n multi | AUC |
|---|---|---|
| 0,0 – 1,2 | 39 | **0,518** (acaso) |
| 1,2 – 2,0 | 68 | 0,666 |
| 2,0 – 3,5 | 53 | 0,739 |

Monotônico. Separação apertada é o caso difícil — o segundo degrau entra antes
de o primeiro transitório terminar — e é onde o corpus tinha menos amostras.

Por razão de amplitude o gradiente é bem mais fraco (0,608 / 0,666 / 0,705 /
0,666), então não é ela.

## 44. A correção de separação foi aplicada — e REFUTADA pela medição

Duas tentativas, porque a primeira não pegou a constante certa.

**(a) `_MULTI_SEP` de `(0.8, 3.0)` para `(0.4, 3.0)` não mudou nada.** A
distribuição saiu idêntica (p10 1,12 / p50 2,22). Motivo: `_MULTI_SEP[0]` só
agia como piso de desempate; quem amarrava era `lo = 0.25 * janela`, e com
janela mediana de 4,85 t_dom isso dá exatamente os 1,2 t_dom medidos.

**(b) Ancorar `lo` em `_MULTI_SEP[0]` funcionou — no corpus.** A distribuição
passou a casar com a real:

| | p10 | p50 | p90 | < 1,2 t_dom |
|---|---|---|---|---|
| sintético antes | 1,12 | 2,22 | 4,26 | 12,1 % |
| **sintético depois** | **0,61** | **1,73** | **3,85** | **33,1 %** |
| real | 0,89 | 1,72 | 2,53 | 24,4 % |

**Mas não transferiu.** Com o corpus corrigido:

| corpus de treino | AUC sintético | AUC real |
|---|---|---|
| com vazamento | 0,9794 | 0,6020 |
| sem vazamento | 0,9919 | 0,6540 |
| **sem vazamento + separação coberta** | 0,9733 | **0,5913** |

A AUC real **piorou**. A hipótese "o estrato é fácil demais em separação" está
refutada: cobrir a faixa difícil não recuperou nada.

## 44b. Três hipóteses testadas, três refutadas

| hipótese | teste | resultado |
|---|---|---|
| a linha de entrada atrapalha | lotes reais sem ela | 0,549 / 0,550 / 0,568 — sem efeito |
| é o estilo do render | física real + render de treino | 0,654 -> 0,656 — sem efeito |
| separação é fácil demais | corpus com a faixa coberta | 0,654 -> 0,591 — piorou |

Em três variantes do corpus a AUC real fica entre 0,59 e 0,65, enquanto a
in-distribution fica entre 0,97 e 0,99. A constância dos dois lados é o dado
mais informativo: **não é um detalhe do corpus, é o método.**

A explicação que sobra, e que é consistente com tudo: **o encoder congelado não
foi treinado para preservar informação de contagem.** Suas features otimizam
segmentação. Num mapa de 512x32 com 1500 amostras de treino há capacidade de
sobra para a cabeça achar um correlato que funciona naquela distribuição sem
ser o sinal causal — e correlato não transfere. A sonda LINEAR chegando a AUC
0,81 in-distribution reforça isso: o "sinal" e barato demais para ser a coisa
certa.

**Conclusão desta linha: o atalho de encoder congelado está morto.** A contagem
exige treinar o encoder para ela — treino conjunto —, e a validação desse
treino tem de ser no corpus REAL, nunca in-distribution.

## 45. Estado final desta linha de trabalho

O que ficou pronto e verificado:

- estrato multi-degrau, **sem o vazamento de janela** e com separações
  cobrindo a faixa difícil;
- `SCHEMA_VERSION` 2, corpus base **intacto byte a byte** (SHA-256 conferido
  contra o disco em 5 diretórios);
- cabeça de contagem na U-Net, com `BatchNorm` na entrada (sem ela, 2 de 5
  sementes colapsam) e compatibilidade estrita de checkpoint;
- laço de treino conjunto, `retreino_multi.sh`, e três scripts de validação.

O que **não** ficou: sinal verde para o treino. A cabeça congelada dá F1 0,698
no real contra 0,740 de uma heurística sem aprendizado, e a causa do fosso está
identificada só em parte.

Recomendação: **não disparar `retreino_multi.sh` ainda.** Regenerar o estrato
com o `_MULTI_SEP` corrigido, repetir a sondagem congelada (minutos), e só
considerar o treino conjunto se a AUC real subir de forma material.

---

# Adendo 10: o retreino conjunto rodou — e não justifica promoção

Fontes: `logs/train_multi.log`, `reports/selecao_multi.json`,
`seleciona_checkpoint_contagem.py`.

## 46. A rodada

25 épocas, 12.450 amostras, `base=32 in_ch=3 batch=6 size=512`, `--contagem
--lambda-contagem 0.2`, **`--lr-threshold 0.002`** (a única divergência
deliberada da receita do `retreino_kneg.sh`, documentada no script). ~6,4 h,
sem OOM e sem interrupção.

| | valor | época |
|---|---|---|
| melhor `IoU_val` | 0,7719 | 18 |
| melhor `acerto_contagem` (sintético) | 0,9845 | 14 |
| **melhor AUC de contagem no REAL** | **0,6838** | **08** |

A troca entre tarefas que se temia no encoder compartilhado **não aconteceu**:
20 dos 25 checkpoints passam a guarda de IoU, e o melhor (0,7667) fica ACIMA
da referência promovida (0,7618). A segmentação não regrediu.

## 47. O veredito: não promover

| | AUC real / F1 |
|---|---|
| cabeça com encoder CONGELADO | AUC 0,59 – 0,65 |
| **cabeça com treino CONJUNTO** | **AUC 0,684** |
| gate de resíduo (produção) | F1 0,571 |
| **gate de resíduo (sem piso)** | **F1 0,740** |

O treino conjunto ganhou ~0,03 de AUC sobre o encoder congelado — marginal — e
continua **abaixo do critério de 0,75** que a spec fixou de antemão. A
heurística de ganho, sem aprendizado nenhum, segue sendo a melhor opção para o
portão de truncagem.

## 48. O achado que sobrevive ao resultado negativo

A AUC real **pica na época 08 e depois CAI**, enquanto a métrica sintética
continua subindo:

| | da época 08 à 24 |
|---|---|
| `acerto_contagem` (sintético) | 0,9805 → 0,9845 (**+0,0011**) |
| AUC de contagem (real) | 0,6838 → 0,6403 (**−0,0435**) |

Spearman entre as duas, no corpus inteiro: +0,466. **Depois da época 08:
−0,401.** Ou seja, passado o pico real, as duas métricas se movem em sentidos
OPOSTOS — treinar mais melhora o número que se olha e piora o que importa.

É sobreajuste à distribuição sintética, medido diretamente. E tem consequência
prática imediata: **`train_unet.py` teria promovido a época 18** (melhor
`IoU_val`), cuja AUC real é 0,6232 — pior que a época 08 em 6 pontos. Selecionar
pelo instrumento errado custaria isso.

## 49. O que fica

- **Não promover** `models/unet_stageA_multi.pt`. O checkpoint promovido
  (`unet_stageA.pt`) continua sendo o de produção.
- **A guarda de continuidade do Adendo 5 continua sendo o melhor ganho
  disponível**, e é a única mudança desta linha de trabalho que se justifica
  por medição: entrega 88,6 % → 95,7 %, recusas 34 → 13, sem retreino.
- O estrato multi-degrau, o `SCHEMA_VERSION` 2, a cabeça de contagem e os
  scripts de seleção ficam versionados e funcionando. São insumo para uma
  próxima tentativa, não trabalho perdido — mas a próxima tentativa precisa
  atacar o fosso de domínio, não repetir a receita.
- **Nunca selecionar checkpoint por métrica in-distribution nesta tarefa.** O
  §48 dá o número que justifica a regra.

---

# Adendo 11: guarda aplicada e validada em 100 figuras novas

`identify/polyline.py` agora traz a guarda de continuidade em definitivo
(`SALTO_MAX_ESPESSURA = 8.0`), com a medição documentada no próprio código.
Verificado que o código aplicado reproduz bit a bit o que foi medido: 49/49
figuras com divergência zero contra `guarda_continuidade.json`.

Lote novo em `reports/amostras_aleatorias/lote100/`, semente 20260909 — nenhuma
destas figuras foi vista no ajuste da guarda nem no retreino. Fatorial 2x2, 25
por célula: sinal de K x (1 ou 2 degraus), render sorteado entre as três
famílias.

## 50. Assertividade geral (n=100)

| | com a guarda | sem a guarda (299 anteriores) |
|---|---|---|
| calibrou | **100/100 (100 %)** | 99,0 % |
| entregou parâmetro físico | **96/100 (96 %)** | 88,6 % |
| estrutura correta | 79/96 (82,3 %) | 82,3 % |
| sinal de K correto | 96/96 (100 %) | 100 % |
| \|erro de K\| mediano | **0,0139** | 0,0354 |
| NRMSE <= 5 % | 79/100 | 74,7 % |
| latência mediana | 270 ms | 732 ms |

Só 4 recusas: 2 `ajuste_inconsistente` e 2 `resposta_inversa`. Nenhuma devolveu
número errado em silêncio.

## 51. Cruzado: sinal x degraus

| célula | físico | estrutura | NRMSE p50 | \|erro K\| p50 | \|errK\|>50 % |
|---|---|---|---|---|---|
| K>0 · 1 degrau | 96 % | 24/24 | 0,0016 | **0,0026** | 0/24 |
| K<0 · 1 degrau | 100 % | 24/25 | 0,0022 | **0,0026** | 1/25 |
| K>0 · 2 degraus | 96 % | 16/24 | 0,0285 | **0,4539** | 12/24 |
| K<0 · 2 degraus | 92 % | 15/23 | 0,0262 | **0,2563** | 8/23 |

Marginais: **K>0 0,0139 contra K<0 0,0188** (o sinal não importa, como as três
rodadas anteriores já haviam medido); **1 degrau 0,0026 contra 2 degraus 0,2861
— 110x**.

## 52. Os problemas, em ordem de tamanho

**1. Multi-degrau não detectado — de longe o maior.** A truncagem tem
revocação de **38,3 %** (TP=18, FN=29) e precisão 75 %.

| | n | NRMSE p50 | \|erro K\| p50 |
|---|---|---|---|
| 1 degrau, sem truncagem | 43 | 0,0018 | **0,0023** |
| 2 degraus, **truncou** | 18 | 0,0051 | **0,0205** |
| 2 degraus, **não truncou** | 29 | 0,0480 | **0,5844** |

E as 29 não truncadas não erram por acaso: o `K` delas está a 18 % da
**excursão TOTAL** contra 58 % da do primeiro degrau, e mais perto do total em
**27 de 29**. Respondem a outra pergunta, sem avisar.

**2. `zeta` encostado no limite da caixa.** 6 de 96 (6,2 %), com \|erro de K\|
mediano de **1,046** contra 0,012 dos demais. Segue sem guarda: o resíduo
desses ajustes é baixo, então `_NRMSE_MAX` não dispara. É o mesmo defeito do
Adendo 2 §13, inalterado.

**3. Tema escuro entrega menos.** 87,0 % contra 98,7 % no claro — as 3 recusas
da família `neg` concentram aí. Mas a exatidão entre as aceitas não é pior, e a
guarda já reduziu bastante esse corte.

**4. Estrutura em multi-degrau.** 82,3 % no geral, mas 31/47 nas de 2 degraus
contra 48/49 nas de 1. É consequência do item 1, não defeito próprio.

**Não é problema:** sinal de K (0,0139 vs 0,0188), tipo de linha (89,5 % a
100 % de entrega), preenchimento, ordem verdadeira. A guarda removeu o tipo de
linha como fator — antes a sólida entregava 97,7 % contra 70,4 % da tracejada.

## 53. O que a guarda entregou, medido fora da amostra de calibração

Entrega física **88,6 % -> 96 %**, erro de K mediano **0,0354 -> 0,0139**,
latência mediana **732 ms -> 270 ms** (menos polilinha corrompida, menos
trabalho no otimizador). O ganho se sustenta em população nova.

O que ela **não** resolve continua sendo o que o Adendo 5 §25 previu: a
detecção de multi-degrau. A revocação de 38,3 % aqui é coerente com os 35,9 %
medidos lá, e confirma que **baixar o `_PISO_SUSPEITA` continua pendente** — é
a única correção barata que resta.

---

# Adendo 12: `_PISO_SUSPEITA` 0,030 -> 0,005, medido ponta a ponta

`identify/classical.py` agora traz o piso revisado, com a varredura documentada
no código. Medido nas MESMAS 100 figuras do Adendo 11 (lote `lote100_piso`),
com a guarda de continuidade ativa nos dois lados — uma variável isolada.

## 54. Por que o piso antigo virou obsoleto

0,030 foi calibrado quando a polilinha ainda pulava para a linha de entrada, e
nesse regime o resíduo inflado pelo distrator abria o portão **por acidente**.
Com a extração limpa esse empurrão some e o piso passa a barrar quase tudo:
**48 % das multi-degrau ficam com `nrmse_full` abaixo de 0,030** e nem chegam à
varredura de cortes.

| piso | TP | FN | FP | precisão | revocação | F1 |
|---|---|---|---|---|---|---|
| **0,030** (antigo) | 19 | 31 | 6 | 76,0 % | **38,0 %** | 0,507 |
| 0,020 | 27 | 23 | 7 | 79,4 % | 54,0 % | 0,643 |
| 0,010 | 28 | 22 | 7 | 80,0 % | 56,0 % | 0,659 |
| **0,007** | 29 | 21 | 7 | 80,6 % | **58,0 %** | **0,674** (satura) |
| 0,000 | 29 | 21 | 7 | 80,6 % | 58,0 % | 0,674 |

**Os falsos positivos praticamente não se movem (6 -> 7).** Os que existem já
têm resíduo bem acima de qualquer piso testado, então baixar o piso não cria
detecção espúria — ele só custava revocação. A precisão até SOBE, porque os
verdadeiros positivos recuperados diluem os mesmos falsos.

0,005 fica na região já saturada sem zerar o portão, que continua poupando a
varredura de 13 ajustes nas figuras de resíduo muito baixo.

## 55. O efeito, ponta a ponta

**Acerto conjuntivo** — todas as métricas certas na mesma figura:

| nível | antes | **depois** | 1 degrau | 2 degraus (antes -> depois) |
|---|---|---|---|---|
| ESTRITO | 59 % | **67 %** | 94 % | 24 % -> **40 %** |
| PRÁTICO | 60 % | **70 %** | 94 % | 28 % -> **46 %** |
| TOLERANTE | 66 % | **75 %** | 96 % | 38 % -> **54 %** |

**+10 pontos no total, +18 nas multi-degrau, e nada perdido no caso de um
degrau** (94 % nos dois).

Marginais:

| | antes | depois |
|---|---|---|
| entrega física | 96 % | 96 % |
| \|erro de K\| mediano | 0,0139 | **0,0088** |
| \|erro de t_dom\| mediano | 0,0347 | **0,0187** |
| revocação da truncagem | 38,0 % | **59,6 %** |
| precisão da truncagem | 76,0 % | **80,0 %** |
| **latência mediana** | 270 ms | **776 ms** |

## 56. O custo, e o que ainda não resolve

**A latência quase triplicou** (270 -> 776 ms). É o preço direto do portão mais
aberto: mais figuras passam pela varredura de 13 ajustes de prefixo. Para uso
interativo em imagem única é irrelevante; para lote grande, é 3x o tempo.

E ainda sobram **19 multi-degrau não detectadas**. Elas continuam com o mesmo
comportamento: `K` a 62 % da verdade do 1º degrau mas a 18 % da excursão TOTAL,
e mais perto do total em **18 de 19**. São os casos em que nenhum prefixo
ajusta decisivamente melhor — o limite de informação do Adendo 2 §11, que
limiar nenhum remove.

## 57. Estado das correções sem retreino

| correção | ganho medido | estado |
|---|---|---|
| guarda de continuidade (`polyline.py`) | entrega 88,6 % -> 96 % | **aplicada** |
| `_PISO_SUSPEITA` 0,030 -> 0,005 | conjuntivo 60 % -> 70 % | **aplicada** |
| guarda de cobertura para `K` extrapolado | 6 casos, \|errK\| 1,046 | pendente |

As duas aplicadas juntas levam o acerto conjuntivo de um baseline que era
**~50 %** (antes da guarda, estimado) para **70 %**, sem tocar na rede.

---

# Adendo 13: remedição de `_UNDERSHOOT_MAX` — não mexer, e por quê

Fonte: `remede_undershoot.py`, `remedicao_undershoot.txt`.

## 58. A hipótese que motivou a remedição estava ERRADA

Eu havia atribuído o salto de `resposta_inversa` (1 -> 6 recusas em 100) à
guarda de continuidade, pelo argumento de que ela encolhe a faixa de `y` e
essa faixa é o DENOMINADOR de `_undershoot`. Testado diretamente nas 6
recusas, com e sem a guarda:

| figura | com guarda | sem guarda | muda o veredito? |
|---|---|---|---|
| bal_K-_1deg_08 | 0,6094 | 0,6094 | não |
| bal_K-_1deg_15 | 0,2685 | 0,2685 | não |
| bal_K+_1deg_17 | 0,5000 | 0,9968 | não |
| bal_K+_1deg_21 | 0,1425 | 0,1425 | não |
| bal_K+_1deg_33 | 0,1172 | 0,1172 | não |
| bal_K-_1deg_44 | 0,1053 | 0,0985 | não |

**0 de 6.** A guarda não causa essas recusas. A diferença entre os dois lotes
(1 contra 6) é de composição e amostragem, não da mudança que eu fiz.

## 59. O lado do BENEFÍCIO não é mensurável com a máscara atual

Gerei 40 figuras de fase não-mínima, `G(s) = K(1 - a s)/(tau s + 1)`, com
`a/tau` de 0,15 a 2,0 — o undershoot IDEAL delas vai de 0,13 a 0,67, muito
acima do limiar de 0,08. Medido na série extraída:

| | undershoot |
|---|---|
| ideal (da física) | p50 = **0,331** |
| medido na série extraída | p50 = **0,012** |
| correlação entre os dois | **−0,110** |

Detecção a 0,08: **3 de 40**. A causa não é o limiar nem a guarda — é a
MÁSCARA. Inspecionado `nmp_09` (ideal 0,645, medido 0,0013): a série extraída
começa em 3,427, ou seja, **o platô de repouso em y=0 não está na máscara** —
zero de 60 colunas com tinta no início da moldura. Sem o nível de repouso,
`_nivel_de_repouso` mede o pico, e todo o resto do cálculo desmorona.

É o mesmo defeito de prior de posição do §40.7 ("a rede não via o platô de
repouso da resposta"), reaparecendo numa geometria que o retreino daquela
seção não cobriu.

## 60. Decisão: NÃO alterar o limiar

Alterar `_UNDERSHOOT_MAX` agora seria trocar um custo medido por um benefício
**não mensurável** — exatamente o que a docstring da constante adverte desde
que foi escrita ("o corpus dá só o CUSTO; o benefício segue apoiado em n=1").
A remedição não mudou essa situação: a n=1 continua, porque as 40 figuras que
eu gerei não conseguem exercitar a guarda com a máscara atual.

O custo, medido em **499 figuras** (todas de fase mínima, então toda detecção é
falso positivo): **12 recusas = 2,4 %.** Perfil delas:

| corte | distribuição |
|---|---|
| família | `neg` 8, `multi` 3, `rg` 1 |
| tema | escuro 8, claro 4 |
| traço | `--` 4, `:` 4, `-.` 4, sólido **0** |
| sinal de K | K>0 6, K<0 6 |

Nenhum sólido entre os 12, e 8 de 12 no tema escuro — o mesmo perfil da
extração difícil. **São falhas de extração que vazam como diagnóstico de
física**, não um limiar mal posto.

## 61. O que fazer no lugar

1. **Deixar `_UNDERSHOOT_MAX = 0.08`.** Mexer nele mascara o sintoma.
2. O alvo real é a máscara perder o platô de repouso em geometrias novas — o
   mesmo eixo do §40.7. É trabalho de corpus e retreino, não de constante.
3. As 40 figuras de fase não-mínima ficam versionadas em
   `reports/amostras_aleatorias/fase_nao_minima/`. Elas são o primeiro
   conjunto de POSITIVOS que o projeto tem para este fenômeno, e servem de
   teste de regressão assim que a máscara aprender a segmentá-las.

---

# Adendo 14: preparar o corpus do platô — duas hipóteses refutadas e a causa vista

## 62. O gap medido, e o estrato construído

O corpus nunca mostra o nível de repouso longe da borda do quadro:

| corpus | p05 | p50 | p95 | no meio (0,25–0,75) |
|---|---|---|---|---|
| `data/train` (K>0) | 0,038 | 0,090 | 0,137 | **0,0 %** |
| `data/train_kneg` (K<0) | 0,864 | 0,914 | 0,963 | **0,0 %** |

Construí o estrato `plato_no_meio` (`_ylim_plato_no_meio` em
`dataset/generator.py`): estica o quadro assimetricamente para o repouso cair
em U(0,25 · 0,75). Só o QUADRO muda — série, spec e estilo ficam idênticos aos
da mesma seed sem o flag, comparável amostra a amostra como o `ganho_negativo`.

Gerados `data/train_plato` (1200), `data/train_plato_kneg` (600),
`data/val_plato` (300). **100 % das amostras com o platô na faixa do meio.**
Corpus base intacto (8/8 por SHA-256; 11 testes do estrato passam).

## 63. Mas o estrato não ataca a causa — duas hipóteses refutadas

Medindo a **cobertura do platô de repouso** (`mede_plato_repouso.py`):

| população | render | física | p50 | < 50 % |
|---|---|---|---|---|
| `data/val` | treino | mínima | 0,884 | 8,6 % |
| **`data/val_plato`** | treino | mínima, platô no meio | **0,861** | 12,5 % |
| `lote100_1deg` | rg_aleatorio | mínima | 0,678 | 24,5 % |
| **`fase_nao_minima`** | rg_aleatorio | **não-mínima** | **0,000** | **100 %** |

**Hipótese 1 (platô no meio) — REFUTADA.** 0,861 contra 0,884: praticamente
igual. O gap de cobertura existia, mas não causava dano.

**Hipótese 2 (subida quase vertical) — REFUTADA.** O corpus sorteia
`t_end = theta + loguniform(0,5, 6,0)·t_dom`, então a janela sempre escala com
a dinâmica e a subida nunca é vertical. Testado: Spearman janela/t_dom x
cobertura = **+0,212** — janela longa dá cobertura MELHOR (1,000 na faixa
12–25 t_dom), não pior.

E o render explica só uma parte: 0,884 -> 0,678 (~20 pp). A física explica o
resto: 0,678 -> 0,000, **com o mesmo render**.

## 64. A causa, vista na máscara

Sobrepondo a máscara à figura `nmp_09`: ela cobre **apenas o trecho monótono
de decaimento**, do pico até a acomodação. Não cobre o platô de repouso, nem o
salto. E pinta parte do TÍTULO — falso positivo em texto, defeito próprio.

O mecanismo não é posição nem inclinação: **a curva inteira não parece uma
resposta ao degrau**, e a rede segmenta a sub-parte que parece, descartando o
resto. Ela aprendeu a FORMA "aproximação monótona a um patamar", e o que não
se encaixa nela é tratado como distrator.

## 65. Consequência: o estrato que falta é fora da família

Em FOPDT/2ª ordem a resposta é contínua em `t = theta` — o valor ali é 0, igual
ao repouso. **Não existe descontinuidade dentro da família**, então nenhum
estrato in-family produz a geometria que a rede não conhece.

Ensinar a máscara a achar o platô nessas curvas exige colocar curvas FORA DA
FAMÍLIA no corpus de treino — fase não-mínima é a escolha natural. Isso é
coerente com o desenho da pipeline: `resposta_inversa` existe justamente para
RECUSAR essas curvas, e para recusá-las com diagnóstico correto a máscara
precisa primeiro ACHAR a curva. Hoje ela não acha, e a guarda dispara pelos
motivos errados (12 falsos positivos em 499 figuras de fase mínima).

Implicações de contrato, que precisam de decisão antes de gerar:

- `order` teria de admitir um valor fora de {`fopdt`, `second`}, ou o meta
  ganhar um campo `fora_da_familia: true`;
- os testes-oráculo da Parte 1 comparam parâmetros ajustados contra o meta e
  teriam de pular essas amostras;
- o alvo da máscara continua sendo a curva inteira — é só isso que se ensina.

## 66. O que fica pronto

| artefato | estado |
|---|---|
| `_ylim_plato_no_meio` + flag `plato_no_meio` | implementado, corpus base intacto |
| `data/train_plato` (1200), `_kneg` (600), `val_plato` (300) | gerados e verificados |
| `mede_plato_repouso.py` | medidor direto do defeito, com baseline registrado |
| `reports/amostras_aleatorias/fase_nao_minima/` (40) | os primeiros POSITIVOS do projeto para fase não-mínima |

O estrato `plato_no_meio` fica como cobertura de um gap real (0 % -> 100 % na
faixa do meio) e custa pouco, mas **não é a correção** — e seria erro registrá-lo
como tal. A correção pede a decisão de contrato do §65.

---

# Adendo 15 — o estrato fora da família, construído e medido

As três decisões do §65 foram tomadas: `fora_da_familia: true` como campo
aditivo (não um terceiro valor em `order`), oráculo **invertido** (não pulado),
e corpus out-of-family admitido no Estágio A a ~11 % do lote. O que segue é o
que a construção mediu — inclusive onde ela **derrubou a minha própria
recomendação**.

## 67. O que se sorteia é o mergulho, não o zero

A planta vira `(1 - a·s)·G_polos(s)` e a resposta ao degrau é
`y0(t) - a·y0'(t)`, exata porque derivar é linear (derivada analítica, validada
contra diferença finita: erro relativo máximo 1,2e-09 em 300 sistemas).

Duas correções de rota durante a construção, ambas por medição:

**Sortear `a` em constantes de tempo não funciona.** Com o mesmo `a/t_dom`,
`fopdt` mergulha `K·a/τ` (grau relativo zero — há salto em `t = θ`) e `second`
mergulha muito menos (grau relativo 1 — sai de zero com derivada finita). Em 60
amostras, **todas as 9 abaixo do limiar da guarda eram de 2ª ordem**. Passei a
sortear a profundidade do mergulho e resolver `a` por bisecção; as duas ordens
ficam na mesma escala de severidade.

**Janela curta produz amostras que não são fase não-mínima.** Com a janela
padrão, 2 de 24 acabam com a curva ainda mergulhada na última coluna: ela
desce, cruza o zero, e o quadro fecha antes de assentar. Nessas, `_undershoot`
lê 0,000 e **acerta** — o que está desenhado é um decaimento monótono. Daí
`_NMP_T_DOM_MIN = 5,5` (0,6 do cruzamento + 4,6 de acomodação). Depois disso,
60/60 acima do limiar, mínimo 0,127 = 1,59× o 0,08.

## 68. O defeito, medido com o render de treino

| corpus | cobertura do platô | recall da máscara | veredito |
|---|---|---|---|
| `data/val` | 0,884 (8,6 % < ½) | 0,784 | 92 % ok |
| `data/val_nmp` | **0,286 (60,7 % < ½)** | 0,729 | **55 % recusa, 41 % ok** |

**Isto corrige o Adendo 14 em magnitude.** Lá eu registrei 0,000 de cobertura
nas figuras de fase não-mínima e concluí que "a rede segmenta a sub-parte que
parece". Com o render de treino a queda é 0,884 → 0,286, não → 0,000: o
fenômeno é real, mas boa parte do 0,000 era o render do `rg_aleatorio`, não a
física. O recall global mal se mexe (0,784 → 0,729) porque o platô é uma fração
pequena dos pixels — é por isso que a cobertura do platô é o detector sensível
e o recall não é.

O número operacional é o outro: **41 % das figuras genuinamente de fase
não-mínima saem como resposta confiante e errada.**

## 69. Duas hipóteses minhas, testadas e refutadas

**A guarda de continuidade apagaria o salto.** Era a suspeita natural — 73 %
das que escapam são `fopdt`, justamente as que têm descontinuidade, e
`SALTO_MAX_ESPESSURA` descarta coluna que salta mais de 8 espessuras.
Desligando a guarda: **1 veredito muda em 120.** Refutada.

**O limiar seria a causa.** Não é. Nas 49 que escaparam, o undershoot da série
*extraída* fica **abaixo de 0,02 em 34 (69 %)** — o mergulho não chega na
guarda, e nenhum limiar o recupera. Só 15 estão na faixa que um limiar mais
baixo pegaria.

## 70. O limiar fica em 0,08 — e agora está justificado

Primeira varredura da história do projeto com **conjunto positivo real**
(200 × 200), nas séries que a guarda de fato recebe:

| limiar | pega NMP | recusa base | F1 |
|---|---|---|---|
| 0,01 | 168/200 | 70/200 | 0,767 |
| 0,02 | 144/200 | 29/200 | 0,772 |
| 0,04 | 128/200 | 7/200 | 0,764 |
| **0,08** | **116/200** | **1/200** | **0,732** |
| 0,10 | 113/200 | 0/200 | 0,722 |
| 0,15 | 89/200 | 0/200 | 0,616 |

Descer de 0,08 para 0,02 troca 28 recusas certas por 28 recusas erradas — 1:1
em contagem bruta, e piora clara em uso real, onde a população dentro da
família é ordens de grandeza maior.

**Isto contradiz a recomendação que eu dei ao apresentar as três decisões**, de
"endurecer o `_UNDERSHOOT_MAX` no mesmo passo, não depois". A medição diz o
contrário: mexer nele agora é pagar em falso positivo por um defeito que não é
dele. A ordem certa é retreinar e só então remedir. O que a constante ganhou
foi a justificativa — a ressalva "o benefício segue apoiado em n=1" saiu do
código.

## 71. Regressão pendente, fora do escopo deste trabalho

`tests/part2/test_truncagem_corpus.py` tem **4 falhas** em
`test_amostras_auditadas_nao_pioram` (samples 00328, 00341, 00357 e mais uma):
amostras auditadas que deixaram de truncar. Isolado por bissecção no working
tree: some ao reverter `identify/polyline.py` + `identify/extract.py`, persiste
ao reverter `identify/classical.py`. **A causa é a guarda de continuidade**,
aplicada mais cedo nesta sessão — um custo dela que eu não tinha medido. Fica
registrado, não corrigido: a banda da spec §5.2.1 precisa ser remapeada, e
isso é decisão de spec.

## 72. O que fica pronto

| artefato | estado |
|---|---|
| `fora_da_familia` + `params.a`, chaves **condicionais** | corpus base byte a byte intacto (imagens e máscaras conferidas) |
| `SCHEMA_VERSION_FORA_DA_FAMILIA = 3`, por amostra | a amostra declara o schema a que ela obedece; o base continua v2 |
| `data/train_nmp` (1200), `_kneg` (600), `val_nmp` (300) | gerados, 0 colisões de seed com o resto do corpus |
| `train_unet.py` — peso 0 na perda de contagem | verificado: métrica se cala com val 100 % OOD, reaparece com val misto |
| `tests/part2/test_estrato_fora_da_familia.py` | 9 testes, oráculo invertido com piso de `n_points`, todos passam |
| `tests/test_part1.py::test_meta_contract` | consertado (faltavam as chaves v2 desde o bump) + invariante in-family asseverado |
| `mede_fora_da_familia.py` | recall da máscara + distribuição de veredito |
| `retreino_nmp.sh` | receita com os dois estratos novos, alvo medível declarado |

## 73. Pré-voo do retreino

Verificação antes de gastar as horas de GPU. Duas coisas estavam quebradas e
foram consertadas; uma decisão fica aberta.

**Quebrado — `seleciona_checkpoint_contagem.py:82` desempacotava 3 elementos.**
`MaskDataset` passou a devolver 4 (o peso da perda de contagem). O seletor é
passo obrigatório do retreino e teria estourado com `ValueError` **depois** das
8 h de treino. Corrigido, e o script foi re-executado ponta a ponta contra
`models/epocas_multi` para provar isso.

**Quebrado — o seletor escrevia em `reports/selecao_multi.json` fixo**, e
rodá-lo num segundo retreino sobrescreveria o registro do primeiro em silêncio.
Agora o nome deriva do diretório de checkpoints.

**Faltando — nenhum seletor tinha o objetivo deste retreino.**
`seleciona_checkpoint.py` mede platô em `data/val` (retreino de ganho
negativo); `seleciona_checkpoint_contagem.py` mede AUC de contagem
(multi-degrau). Promover por qualquer um deles seria o erro do §40.9 outra vez.
Escrito `seleciona_checkpoint_nmp.py`: objetivo é cobertura do platô em
`data/val_nmp`, com três guardas — platô em `data/val`, IoU em `val`+`val_multi`,
e AUC de contagem ≥ **0,6838** (barra **absoluta**, porque o modelo promovido
não tem cabeça de contagem; 0,6838 é o melhor apto do retreino multi, época 08).

**Verificado e limpo:**

| item | resultado |
|---|---|
| integridade dos 4200 samples novos | 0 problemas; metas, schema 3, `a > 0` |
| colisão de seed com o resto do corpus | 0 |
| mergulho em todo o estrato (n=2100) | 0,115 a 0,453; **2100/2100** voltam acima do repouso |
| anti-vazamento: render muda com o flag? | **nenhuma chave**, nos quatro estratos |
| todos os 21 diretórios da receita existem | sim — 16.050 treino / 2.500 validação |
| aridade do dataset nos outros consumidores | varrido; só o seletor estava quebrado |
| `tests/part2/test_train_checkpoints.py` | 4 passam |
| VRAM / disco | 5,5 GB livres de 5,6; 202 GB livres, 760 MB por 25 checkpoints |
| baseline pré-retreino | `reports/baseline_pre_nmp.txt` |

**Diluição da classe positiva da contagem: 9,61 % -> 8,39 %** (1,21 p.p.). A
cabeça vê 14.250 amostras em vez de 16.050 — as 1.800 fora da família entram
com peso 0, como projetado. `binary_cross_entropy_with_logits` roda sem
`pos_weight` aqui; como a seleção é por AUC, que é invariante a limiar, a
diluição afeta calibração e não ranqueamento. Registrado, não corrigido.

**Decisão aberta: orçamento de épocas.** No retreino multi, o melhor apto foi a
**época 08**, o IoU platôou na 13 (0,7703 -> 0,7704 da 13 à 23) e o LR caiu a
9,37e-06 na 17. As épocas 14–24 não produziram nada que fosse selecionado.
Escalando 922 s/época pelo corpus maior: **~1.190 s/época**, 25 épocas = **8,3 h**,
18 épocas = **6,0 h**. A receita está em 25.

**Pressão de memória, medida.** A máquina estava com **swap em 7,0 de 8,0 GiB**
e 189 MiB de RAM livre — e tarefas de fundo desta própria sessão já tinham sido
mortas por falta de memória. Não é hipótese. Medido o pico de RSS da árvore de
treino com a receita cheia (base 32, batch 6, size 512):

| workers | pico de RSS |
|---|---|
| 4 + 2 (padrão) | **3,6 GB** |
| 2 + 1 (`WORKERS=2`) | **2,6 GB** |

Contra 6,0 GB disponíveis, cabe nos dois casos. O número de workers **não muda
resultado nenhum** — a ordem das amostras vem do sampler no processo principal,
com `torch.manual_seed(20260817)` fixo —, então baixá-lo custa no máximo
throughput, e o run é limitado por GPU (0,44 s/passo). `train_unet.py` ganhou
`--workers` (padrão 4, nada muda por omissão) e a receita aceita `WORKERS=` e
`EPOCAS=` por ambiente. O impacto de `--workers 2` na velocidade **não foi
medido**; a economia de 1 GB foi.
