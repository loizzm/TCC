# TCC — Identificação automática de plantas a partir de imagens de resposta ao degrau

Dado **um gráfico de resposta ao degrau como imagem** (um PNG qualquer, com
resolução, cores, grade, legendas e ruído arbitrários), recuperar os
**parâmetros da planta** que o gerou — ganho `K`, constante de tempo `τ`, atraso
`θ` para modelos FOPDT, ou `ωn` e `ζ` para modelos de segunda ordem.

Este repositório contém o código, os testes de aceitação e o planejamento do
trabalho. Ele **não é uma biblioteca de uso geral**: é o artefato de um TCC, e a
suíte de testes é, ao mesmo tempo, o portão de aprovação e a fábrica dos números
que vão para a monografia.

---

## O pipeline em quatro estágios

```
image.png
   │
   ├─[A]─► extração da curva            U-Net → polilinha em pixels
   │
   ├─[B]─► calibração dos eixos         moldura + ticks + OCR + RANSAC
   │            │
   │            ▼
   │      série y(t) em unidades físicas
   │            │
   ├─[C]─► estimador neural 1D          ordem do sistema + chute inicial
   │            │
   └─[D]─► refinamento least_squares    ──► K, τ, θ, ωn, ζ
```

A separação entre **C** e **D** é o argumento central do trabalho. Otimização
não-linear atinge erro sub-1%, mas falha de dois modos clássicos: escolher a
estrutura errada e cair em mínimo local por inicialização ruim. Rede neural é
boa exatamente nessas duas coisas e medíocre em precisão numérica; o otimizador
é o oposto. Isso define qual é a contribuição da rede — sem esse enquadramento,
"por que não usar só mínimos quadrados?" não tem resposta boa.

### Estado do trabalho

| | escopo | estado |
|---|---|---|
| **Parte 1** | gerador de dados + estágio **D** + prova de solubilidade com oráculo | **concluída**, 33/33 testes verdes |
| **Parte 2** | estágios **A** e **B** — substituir o oráculo por percepção real | planejada (`PLANO_PARTE2.md`), não iniciada |
| **Parte 3** | estágio **C** + integração + validação fora da distribuição | não iniciada |

A Parte 1 curto-circuita A, B e C: lê a série direto do gabarito (o *oráculo*) e
roda só o estágio D. Serve para provar que o problema é solúvel e para medir o
teto de desempenho contra o qual a percepção real será comparada.

---

## Mapa do repositório

```
TCC/
├── dataset/                    [1] PRODUZ OS DADOS
│   ├── randomize.py     (379)  sorteia o estilo visual — não vê o rótulo
│   └── generator.py     (500)  sorteia o sistema, renderiza, calibra, escreve o meta
│
├── identify/                   [2] CONSOME AS SÉRIES
│   └── classical.py     (893)  estágio D: least_squares + AIC + 3 baselines clássicos
│
├── tests/                      [3] MEDE E DOCUMENTA
│   ├── conftest.py     (1063)  fixtures de sessão + gerador do relatório
│   ├── test_part1.py   (1121)  critérios 1.1, 1.2, 1.5, 1.6, 1.7, G + contrato
│   └── test_leakage.py  (549)  critérios 1.3 e 1.4 (a–e) — anti-vazamento
│
├── reports/
│   └── part1_metrics.md        GERADO pela suíte — não editar à mão
│
├── PLANO.md                    plano das 3 partes, decisões e critérios de aceitação
├── PLANO_PARTE2.md             plano de execução da Parte 2, em 6 blocos
├── HANDOFF.md                  estado atual, decisões tomadas e pontos abertos
├── ARQUITETURA.md              mapa detalhado, fluxo de dados e glossário
├── img.py               (264)  LEGADO DEFEITUOSO — evidência, não código
├── pytest.ini                  testpaths e marcador `slow`
├── requirements.txt            ambiente PINADO (um critério compara sha256 de PNG)
├── data/                       datasets gerados; fora do versionamento
└── relatorio_acompanhamento_pfc1_*.pdf   relatórios de acompanhamento entregues
```

**Os três módulos não formam uma cadeia de importação.** `identify/` não importa
nada de `dataset/`; `dataset/` não sabe que `identify/` existe. Quem os costura é
a suíte de testes. Isso é deliberado — ver "verificação cruzada" abaixo.

---

## Os módulos

### `dataset/` — o gerador

Produz amostras sintéticas de gráficos de resposta ao degrau, com a verdade de
terra ao lado. Substitui o `img.py` legado, que vazava o rótulo dentro da imagem.

**`randomize.py`** — `RenderStyle` (dataclass com todos os atributos visuais) e
`sample_style(rng)`. A assinatura é a garantia anti-vazamento:

```python
def sample_style(rng: np.random.Generator) -> RenderStyle:
```

Ela **não recebe o `SystemSpec`**. Não é uma promessa de que ninguém vai
correlacionar estilo com rótulo — é impossibilidade: a função não tem acesso ao
rótulo. Um teste trava a assinatura para que ninguém acrescente o parâmetro por
descuido. Sorteia resolução (240–1600 × 180–1200), dpi, paleta com contraste
mínimo garantido, espessura, marcadores, grade, ticks, spines, margens, fontes,
textos semanticamente vazios, 1 a 3 distratores, SNR e quantização.

**`generator.py`** — o núcleo da geração:

| função | papel |
|---|---|
| `dominant_time_constant()` | fonte única de `T_dom` |
| `sample_system(rng)` | sorteia ordem, K, τ ou (ωn, ζ), θ e a janela temporal |
| `step_response(spec, t)` | resposta analítica, três ramos (ζ<1, ζ=1, ζ>1) |
| `_apply_noise()` | ruído gaussiano por SNR sobre a variância + quantização |
| `render_sample()` | desenha, calcula a calibração, escreve os três arquivos |
| `generate_sample/dataset()` | API de alto nível, com seeds independentes |
| `load_sample()` | caminho inverso: meta + PNGs como arrays |

Três decisões de renderização sustentam o resto:

- **API orientada a objetos do matplotlib, nunca `pyplot`.** `Figure` +
  `FigureCanvasAgg` instanciados à mão, backend `Agg`. `pyplot` mantém registro
  global de figuras, e estado global é incompatível com determinismo bit-a-bit e
  com `ProcessPoolExecutor`.
- **A máscara é renderizada de novo, não segmentada.** São desenhadas duas
  figuras de geometria idêntica: a imagem colorida com tudo, e uma segunda com
  fundo preto e só a curva em branco. Quem desenha a curva sabe onde ela está —
  o custo marginal de emitir a verdade de terra é ~zero e a qualidade é perfeita
  por construção. Há verificação em tempo de execução se as geometrias divergirem.
- **A calibração é lida do matplotlib, não estimada.** A afim `axis_affine` sai
  de `ax.transData.transform()` — a mesma transformação usada para desenhar.

**Determinismo**, via três streams independentes de uma seed:

```
seed ──► SeedSequence.spawn(3) ──┬── children[0] → rng_sys    → sample_system
                                 ├── children[1] → rng_style  → sample_style
                                 └── children[2] → rng_noise  → _apply_noise
```

Como a seed de cada amostra é função do **índice**, e não da ordem de execução,
`generate_dataset` dá o mesmo resultado com 1 ou 16 workers.

### `identify/classical.py` — o estágio D

Recebe `(t, y)` e devolve parâmetros. **Não importa `dataset/`**:
`model_response` é uma reimplementação *independente* dos mesmos modelos. Não é
duplicação por descuido, é **verificação cruzada** — `test_model_cross_check`
confronta as duas implementações em 200 conjuntos de parâmetros, incluindo a
vizinhança exata de ζ=1. Se uma importasse da outra, um erro comum passaria
despercebido nas duas.

```
model_response()                      modelos analíticos
        ↑
_integral_guess_* / _grid_guess_*     chutes iniciais (regressão integral,
_overshoot_guess / _overdamped_guess  grade com K perfilado)
        ↑
initial_guess_fopdt / _second         consolida e sanitiza
        ↑
_multistart() → _run_lsq()            least_squares em duas passadas (grossa, fina)
        ↑
fit_fopdt() / fit_second()            → FitResult(order, params, aic, nrmse, sse, success)
        ↑
identify_both() → identify()          ajusta AS DUAS estruturas, escolhe por AIC
```

À parte, os **baselines clássicos** (`baseline_tangent`, `baseline_smith`,
`baseline_sundaresan_krishnaswamy`), para a tabela comparativa da monografia.

Duas armadilhas de uso, herdadas do desenho:

- `_estimate_gain` é **memoizado em estado global de módulo** ⇒ paralelize com
  **processos**, nunca threads.
- `identify` e `fit_*` **nunca levantam exceção** — devolvem `success=False` ou
  `nan`. Um estimador que inventa número é pior que um que se cala.

Latência medida, 1 thread, 512 pontos: mediana 55 ms, p95 143 ms, máx 267 ms.

### `tests/` — o portão de aceitação e a fábrica do relatório

Duas funções que normalmente não andam juntas: decidir aprovação e gerar o
documento da monografia.

**`conftest.py`** carrega as fixtures de sessão (dados gerados uma vez e
compartilhados) e o acumulador do relatório:

| fixture | o que é |
|---|---|
| `clean_dataset` | 600 amostras sem ruído |
| `noisy_dataset` | 600 amostras com SNR fixo em 20 dB |
| `fits_clean` / `fits_noisy` | `identify` sobre cada série |
| `sampling_population` | 20.000 sorteios sem renderizar nada |

A última merece explicação: os testes de vazamento não precisam de imagem — a
independência entre estilo e rótulo é gerada no **sorteio**. Rodar só o sorteio
20.000 vezes dá poder estatístico impossível de obter renderizando.

Cada teste, além de assertar, **registra o que mediu** (`record_criterion`,
`record_block`, `record_gate_n`). No fim da sessão, o hook `pytest_sessionfinish`
monta `reports/part1_metrics.md` a partir do acumulado.

Daí a regra central: **a suíte mede a realidade; ela não existe para passar.** Se
um critério falha, o número real vai para a tabela com veredito FALHA — nunca se
ajusta o limiar.

**`test_leakage.py`** cobre os critérios anti-vazamento em cinco camadas de força
crescente:

| | o que faz | poder |
|---|---|---|
| 1.3 | GBM treinado só com atributos visuais tenta prever `order` | n=20.000 → 0,4985, o acaso |
| 1.4a | Spearman estilo × parâmetro no sorteio | n=20.000, com Bonferroni |
| 1.4b | idem no dataset renderizado, com nulo de permutação | assertiva estatisticamente correta |
| 1.4c | round-trip exato: re-deriva o estilo da seed e exige igualdade | falso positivo e falso negativo zero |
| 1.4d / 1.4e | lê os **pixels** da `image.png` | fecha o elo pixel → meta → estilo |

As duas últimas nasceram do teste de mutação: antes delas toda a verificação
passava pelo `meta.json`, e um gerador que pintasse a curva por `order` sem mexer
no meta passava com 30/30 — que é *literalmente* o defeito do `img.py`.

### `img.py` — o legado defeituoso

Gerador original, mantido no repositório como **evidência, não como código**.
Ele vazava o rótulo dentro da imagem: um classificador que só olhava atributos
visuais acertava a ordem do sistema em 93% dos casos. A contraprova está no
critério 1.3, onde o gerador saneado leva o mesmo classificador a 0,4985 — o
acaso. Não importe nada dele.

---

## O contrato de dados

A fronteira entre os módulos. Cada amostra é um diretório:

```
sample_XXXXX/
  image.png     figura completa (resolução, cores, estilo, textos aleatórios)
  mask.png      uint8, mesma resolução: 255 onde há curva, 0 caso contrário
  meta.json
```

```jsonc
{
  "schema_version": 1,
  "sample_id": "sample_00000",
  "seed": 12345,
  "order": "fopdt" | "second",
  "params": {"K":…, "tau":…, "theta":…, "wn":…, "zeta":…},
  "step_amplitude": 1.0,
  "t_window": [t0, t1],
  "plot_bbox_px": [x0, y0, x1, y1],                 // moldura da área de dados
  "axis_affine": {"sx":…, "ox":…, "sy":…, "oy":…},  // t = sx·px + ox ; y = sy·py + oy
  "ticks": {"x": [[px, val], …], "y": [[px, val], …]},
  "series": {"t": […512], "y": […512]},             // O QUE FOI DESENHADO
  "noise": {"enabled":…, "snr_db":…, "quantization_levels":…},
  "render": {…}                                     // só estratificação
}
```

Dois pontos que costumam gerar erro:

- **`series.y` é o que foi desenhado**, com ruído se houver — não a curva ideal.
- **`render` nunca entra em modelo nenhum.** Existe para estratificar métricas
  ("o IoU cai quando há grade?"). Usá-lo como feature seria reintroduzir o
  vazamento pela porta dos fundos.

---

## Como rodar

Requer **Python 3.11**. As versões em `requirements.txt` estão pinadas de
propósito: um dos critérios de aceitação compara o `sha256` dos PNGs gerados, e
qualquer mudança de versão do matplotlib altera a renderização.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Rodar a suíte de aceitação da Parte 1:

```bash
.venv/bin/python -m pytest -q
```

Fecha em ~130 s com 16 threads e a máquina ociosa, **33 passed**. Isso regenera
`reports/part1_metrics.md`. **Rode sempre sem `-m` / `-k`**: com filtro o
relatório sai parcial (a suíte imprime um banner de aviso no topo do arquivo
quando isso acontece).

Gerar um dataset fora dos testes:

```bash
.venv/bin/python -m dataset.generator data/train 6000 0
#                                      <saída>   <n>  <seed>
```

---

## Os documentos

| pergunta | documento |
|---|---|
| *por que assim?* — decisões de arquitetura, critérios de aceitação | `PLANO.md` |
| *como está hoje?* — estado, decisões da execução, pontos abertos | `HANDOFF.md` |
| *o que cada peça é e como se encaixam?* — mapa detalhado e glossário | `ARQUITETURA.md` |
| *o que vem agora?* — Parte 2 em 6 blocos, com critérios e handoffs | `PLANO_PARTE2.md` |
| *quanto deu?* — todos os números medidos, com a metodologia de cada limiar | `reports/part1_metrics.md` (**gerado**) |

Comece por `ARQUITETURA.md` se o objetivo é entender o código; por `HANDOFF.md`
se o objetivo é continuar o trabalho de onde parou.

---

## O que a Parte 1 já mostrou

Todos os números estão em `reports/part1_metrics.md`, com as tabelas prontas.

1. **Não-identificabilidade prática em 2ª ordem superamortecida.** Em ζ ∈ [2,2;
   3,0], ωn e ζ erram ~85% enquanto K erra 0,28% e a constante do polo lento erra
   1,28%, com `corr(erro_ωn, erro_ζ)` = **+0,9997**. É deslizamento ao longo de
   uma curva de nível onde a dinâmica observável é a mesma — limite de
   informação, não deficiência do estimador. É o que justifica adotar o NRMSE de
   reconstrução como métrica primária do trabalho.
2. **Contraprova do vazamento de rótulo.** Um GBM que só vê atributos visuais
   fica em 0,4985 de acurácia — o acaso. Contraprova direta dos 93% do `img.py`.
3. **Limite de informação da janela.** Com janela `w < 3` a curva é cortada antes
   do regime permanente e o ganho deixa de ser identificável.
4. **Baselines clássicos × `identify`.** Os baselines não quebram em exatidão, e
   sim em **cobertura**: no estrato truncado, Smith responde em 54,7% das séries
   e Sundaresan–Krishnaswamy em 24,2%, enquanto `identify` responde em 100%.
