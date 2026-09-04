# ARQUITETURA — mapa dos módulos, fluxo de dados e glossário

Documento de orientação. Responde **o que cada arquivo faz e como eles se
conectam**, e define os termos que aparecem no código e no relatório sem
explicação.

Ele não substitui os outros três:

| pergunta | documento |
|---|---|
| *por que assim?* — decisões de arquitetura, critérios de aceitação | `PLANO.md` |
| *como está hoje?* — estado, decisões tomadas na execução, pontos abertos | `HANDOFF.md` |
| *quanto deu?* — todos os números medidos, com a metodologia de cada limiar | `reports/part1_metrics.md` (**gerado**) |
| *o que cada peça é e como se encaixam?* | este arquivo |
| *em que literatura isso se apoia?* | `REFERENCIAS.md` |
| *e se fosse fim-a-fim?* | `PLANO_CNN_FIM_A_FIM.md` |

Ambiente: sempre `/home/loizm/work/TCC-2/.venv/bin/python` (3.11). O `python3` do
sistema é 3.14 e não tem as bibliotecas.

---

## 1. O sistema em uma tela

O sistema final identifica os parâmetros de uma planta a partir de **uma imagem**
de resposta ao degrau, em três estágios:

```
image.png
   │
   ├─[A]─► extração da curva          U-Net (ou extrator clássico) → polilinha
   │
   ├─[B]─► calibração dos eixos       moldura + ticks + OCR opcional + RANSAC
   │            │
   │            ▼
   │      série y(t)  ──► adimensional sempre; física quando a calibração fecha
   │            │
   └─[D]─► identificação least_squares   multi-start + AIC  ──► K, τ, θ, ωn, ζ
```

**Havia um quarto estágio.** O plano original punha um `[C] estimador neural 1D`
entre B e D, para escolher a estrutura e dar o chute inicial ao otimizador. Ele foi
**medido e removido** em 22/08/2026 (`PLANO.md §1.3`): a arbitragem por AIC acerta
**100%** onde a distinção entre FOPDT e 2ª ordem é observável (ζ < 1,6), e onde erra
o custo é 9e-04 de NRMSE — porque ali as duas estruturas são indistinguíveis e
nenhuma rede pode superar a ausência de informação no dado. O multi-start clássico
convergiu em 400/400 amostras, e o estágio D reduz 72% o NRMSE do chute inicial,
contra o alvo de 40% que o critério original pedia da rede.

Resta **um** modelo treinado: a U-Net do estágio A. Segmentar uma curva de um
gráfico com resolução, paleta, grade, legenda e distratores arbitrários é o problema
de aprendizado profundo deste trabalho — e é onde a contribuição está.

E a pergunta de banca "por que não usar só mínimos quadrados?" tem hoje uma resposta
melhor que a original: **é exatamente o que se usa, e aqui está a estratificação que
mostra onde isso basta e onde nada bastaria.**

### Onde cada Parte do trabalho entra

| | escopo | estado |
|---|---|---|
| **Parte 1** | gerador de dados + **D** + prova de solubilidade com oráculo | concluída, 33/33 verdes |
| **Parte 2** | **A** e **B** — substituir o oráculo por percepção real | não iniciada |
| **Parte 3** | validação fora da distribuição + PID/IMC + baseline fim-a-fim | não iniciada |

A Parte 1 curto-circuita A e B: lê a série direto do gabarito e roda só o D.
Ver §7 (glossário, *oráculo*).

A Parte 3 não constrói componente novo — o pipeline está completo ao fim da Parte 2.
Ela **mede**: fora da distribuição, utilidade para controle, e a CNN fim-a-fim como
baseline (`PLANO_CNN_FIM_A_FIM.md`).

---

## 2. Mapa do repositório

```
TCC-2/
├── dataset/                    [1] PRODUZ OS DADOS
│   ├── __init__.py        (6)
│   ├── randomize.py     (379)  sorteia o estilo visual — não vê o rótulo
│   └── generator.py     (500)  sorteia o sistema, renderiza, calibra, escreve o meta
│
├── identify/                   [2] CONSOME AS SÉRIES
│   ├── __init__.py        (3)
│   └── classical.py     (893)  Estágio D: least_squares + AIC + 3 baselines
│
├── tests/                      [3] MEDE E DOCUMENTA
│   ├── __init__.py        (3)
│   ├── conftest.py     (1063)  fixtures de sessão + gerador do relatório
│   ├── test_part1.py   (1121)  critérios 1.1, 1.2, 1.5, 1.6, 1.7, G + contrato
│   └── test_leakage.py  (549)  critérios 1.3 e 1.4 (a–e) — anti-vazamento
│
├── reports/
│   └── part1_metrics.md        GERADO pela suíte. Não editar à mão
│
├── PLANO.md                    plano das 3 partes e critérios
├── HANDOFF.md                  estado, decisões e regras
├── ARQUITETURA.md              este arquivo
├── pytest.ini                  testpaths, marcador `slow`
├── requirements.txt            ambiente PINADO (o critério 1.6 compara sha256 de PNG)
├── img.py               (264)  LEGADO DEFEITUOSO — evidência, não código (§6)
├── data/                       vazio; datasets grandes ficam fora do versionamento
└── .venv/                      Python 3.11
```

**Os três módulos não formam uma cadeia de importação.** `identify/` não importa
nada de `dataset/`; `dataset/` não sabe que `identify/` existe. Quem os costura é
a suíte. Isso é deliberado — ver §3.2.

---

## 3. Os módulos

### 3.1 `dataset/` — o gerador

Substitui o `img.py` legado, que vazava o rótulo dentro da imagem.

#### `randomize.py`

`RenderStyle` (dataclass com todos os atributos visuais) e `sample_style(rng)`.

A assinatura é a garantia anti-vazamento:

```python
def sample_style(rng: np.random.Generator) -> RenderStyle:
```

**Não recebe o `SystemSpec`.** Não é promessa de que ninguém vai correlacionar
estilo com rótulo — é impossibilidade: a função não tem acesso ao rótulo. Há um
teste que trava a assinatura (`test_sample_style_signature_cannot_see_the_label`)
para que ninguém acrescente o parâmetro por descuido.

Sorteia: resolução (240–1600 × 180–1200), dpi (60–200), paleta com **contraste
mínimo garantido**, espessura, estilo de traço, marcadores, grade, ticks, spines,
margens, fontes, títulos/legendas/anotações com **texto semanticamente vazio**,
1 a 3 **distratores**, SNR e quantização.

Dois pisos que existem por medição, não por gosto:

- `MIN_LINE_PX = 1.5` — abaixo disso o anti-aliasing fura a máscara no limiar 127
- `font_size` escalado pelo tamanho físico da figura — senão o texto vira borrão

`to_meta()` serializa o bloco `render`, que **nunca entra em modelo nenhum**: só
serve para estratificar métricas ("o IoU cai quando há grade?").

#### `generator.py`

| função | papel |
|---|---|
| `dominant_time_constant()` | **fonte única** de `T_dom`. Foi aqui que morava o bug do ζ>1 |
| `sample_system(rng)` | sorteia ordem, K, τ ou (ωn, ζ), θ e a janela temporal |
| `step_response(spec, t)` | resposta analítica, três ramos (ζ<1, ζ=1, ζ>1) |
| `_apply_noise()` | ruído gaussiano por SNR **sobre a variância** + quantização |
| `render_sample()` | desenha, calcula a calibração, escreve os três arquivos |
| `generate_sample/dataset()` | API de alto nível, com as seeds independentes |
| `load_sample()` | caminho inverso: meta + PNGs como arrays |

Três decisões de renderização que sustentam tudo o mais:

**API orientada a objetos, nunca `pyplot`.** `Figure` + `FigureCanvasAgg`
instanciados à mão, backend `Agg` (headless). `pyplot` mantém registro global de
figuras, e estado global é incompatível com determinismo bit-a-bit e com
`ProcessPoolExecutor`.

**A máscara é renderizada de novo, não segmentada.** `render_sample` desenha
**duas figuras de geometria idêntica**: a imagem colorida com tudo, e uma segunda
com fundo preto, só a curva em branco, eixos desligados. Quem desenha a curva
sabe onde ela está — o custo marginal de emitir a verdade de terra é ~zero e a
qualidade é perfeita por construção. Há verificação em tempo de execução que
levanta exceção se as geometrias divergirem.

**A calibração é lida do matplotlib, não estimada.** A afim `axis_affine` sai de
`ax.transData.transform()` — a mesma transformação usada para desenhar — com as
duas conversões de convenção: origem inferior→superior (`h_px - p[0,1]`) e centro
do pixel *i* em *i+0,5* (`- 0.5`). Meio pixel de erro aqui só apareceria na
Parte 2 como viés sistemático; o critério 1.5 existe para trancar isso.

**Determinismo**, via três streams independentes de uma seed:

```
seed ──► SeedSequence.spawn(3) ──┬── children[0] → rng_sys    → sample_system
                                 ├── children[1] → rng_style  → sample_style
                                 └── children[2] → rng_noise  → _apply_noise
```

Como a seed de cada amostra é função do **índice**, e não da ordem de execução,
`generate_dataset` dá o mesmo resultado com 1 ou 16 workers.

### 3.2 `identify/classical.py` — o Estágio D

Recebe `(t, y)` e devolve parâmetros. **Não importa `dataset/`**: `model_response`
é uma reimplementação *independente* dos mesmos modelos.

Isso não é duplicação por descuido — é **verificação cruzada**.
`test_model_cross_check` confronta `dataset.step_response` com
`identify.model_response` em 200 conjuntos de parâmetros, incluindo a vizinhança
exata de ζ=1. Dois códigos escritos separadamente que concordam numericamente são
evidência forte de que ambos estão certos; se um importasse do outro, um erro
comum passaria despercebido nos dois.

Camadas internas:

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
`baseline_sundaresan_krishnaswamy`), para a tabela comparativa da monografia. Os
três obtêm o ganho de `_estimate_gain` — por isso essa função tem guarda própria
(critério G).

Duas armadilhas de uso, herdadas do desenho:

- `_estimate_gain` é **memoizado em estado global de módulo** ⇒ paralelize com
  **processos**, nunca threads
- `identify` e `fit_*` **nunca levantam exceção** — devolvem `success=False` ou
  `nan`. Um estimador que inventa número é pior que um que se cala

Latência medida, 1 thread, 512 pontos: mediana 55 ms, p95 143 ms, máx 267 ms.

### 3.3 `tests/` — o portão de aceitação **e** a fábrica do relatório

Duas funções que normalmente não andam juntas: decidir aprovação e gerar o
documento da monografia.

#### `conftest.py`

**(a) Fixtures de sessão** — os dados compartilhados, gerados uma vez:

| fixture | o que é | usado por |
|---|---|---|
| `clean_dataset` | 600 amostras **sem ruído** | 1.1, 1.5, contrato, baselines |
| `noisy_dataset` | 600 amostras com **SNR fixo em 20 dB** | 1.2 |
| `fits_clean` / `fits_noisy` | `identify` sobre cada série | 1.1, 1.2 |
| `sampling_population` | **20.000 sorteios sem renderizar nada** | 1.3, 1.4a |

A última merece explicação: os testes de vazamento não precisam de imagem — a
independência entre estilo e rótulo é gerada no **sorteio**. Rodar só o sorteio
20.000 vezes dá poder estatístico impossível de obter renderizando.

**(b) Workers de processo** — `_fit_worker`, `_sample_worker`, `_gen_fixed_snr`,
com `ProcessPoolExecutor` e contexto `fork`, 16 workers.

**(c) O acumulador do relatório:**

```python
RESULTS = {"criteria": {}, "blocks": {}}

record_criterion(id, nome, alvo, medido, ok)   # uma linha da tabela mestre
record_block(chave, valor)                     # dados brutos de uma seção
record_gate_n(criterio, subgrupo, n)           # tamanho de amostra efetivo
```

Cada teste, além de assertar, **registra o que mediu**. No fim da sessão o hook
`pytest_sessionfinish` chama `_write_report()`, que monta
`reports/part1_metrics.md` a partir do acumulado.

Daí a regra central: **a suíte mede a realidade; ela não existe para passar.** Se
um critério falha, o número real vai para a tabela com veredito FALHA — nunca se
ajusta o limiar. E há salvaguarda: rodar com `-k`/`-m` põe um banner de aviso no
topo do relatório dizendo que ele está parcial.

#### `test_part1.py` e `test_leakage.py`

Os critérios anti-vazamento, em cinco camadas de força crescente:

| | o que faz | poder |
|---|---|---|
| 1.3 | GBM treinado só com atributos visuais tenta prever `order` | n=20.000 → 0,4985, o acaso |
| 1.4a | Spearman estilo × parâmetro no sorteio | n=20.000, com Bonferroni |
| 1.4b | idem no dataset renderizado, com nulo de permutação | assertiva estatisticamente correta |
| 1.4c | **round-trip exato**: re-deriva o estilo da seed e exige igualdade | falso positivo e falso negativo **zero** |
| 1.4d / 1.4e | lê os **pixels** da `image.png` | fecha o elo pixel → meta → estilo |

As duas últimas nasceram do teste de mutação (HANDOFF §3.5.1): antes delas toda a
verificação passava pelo `meta.json`, e um gerador que pintasse a curva por
`order` sem mexer no meta passava com 30/30 — que é *literalmente* o defeito do
`img.py`.

O padrão de qualidade que se repete: quase todo critério vem com um **controle**.
O 1.5 não se contenta em medir 0,16 px de erro — desloca a afim em 3 px de
propósito e exige que piore ≥10× (deu 13,5×). O critério G tem controle positivo:
se `max(y)` deixar de errar ≥10% no estrato, o teste avisa que perdeu o poder de
separar. Sem isso, uma métrica que sempre dá zero pode significar "está perfeito"
ou "não estou medindo nada", e não há como distinguir.

---

## 4. O contrato de dados

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
  Um mutante que gravava a série limpa com ruído ligado foi pego por causa dessa
  distinção.
- **`render` nunca entra em modelo nenhum.** Existe para estratificar métricas.
  Usá-lo como feature seria reintroduzir o vazamento pela porta dos fundos.

### O que `K` significa, e o que ele NÃO significa

`K` é a **excursão da saída por unidade de entrada** — o produto `K_planta × U`,
onde `U` é a amplitude do degrau realmente aplicado. `STEP_AMPLITUDE = 1.0`
(`identify/classical.py`) é uma **convenção**, não uma medição: a pipeline nunca
vê a entrada, só a resposta.

Isso importa porque `K` **não** é o ganho DC da planta, a menos que `U = 1`. E a
diferença não é corrigível por software: da curva de saída sozinha, `K_planta` e
`U` não são separáveis, só o produto é observável. Estas três situações geram a
mesma curva ponto a ponto:

| `K_planta` | `U` | curva observada |
|---|---|---|
| 1,997 | −1 | idêntica |
| 0,999 | −2 | idêntica |
| 0,499 | −4 | idêntica |

**O caso concreto que expôs isso.** `Figure_dn.png` (do `rg_negativo.py`) tem
planta `2/(s+2)`, ganho DC = 1, com degrau de amplitude **−2**. A pipeline
devolve `K = −1,997`, que é `1 × (−2)` e está correto: a curva reconstruída a
partir dos parâmetros reportados bate com a verdade analítica com erro máximo de
0,0039 e RMSE 0,0026, e `tau` e `theta` saem a 0,01 % e 0,02 %. Mesmo assim, quem
comparar esse `−1,997` com o `K = 1` escrito na função de transferência conclui
que houve falha de ajuste. Não houve.

Recuperar `K_planta` exige **ler a amplitude do degrau da imagem**, o que só é
possível quando a entrada está plotada no mesmo quadro — como nas figuras do
`rg_negativo.py`, que desenham o degrau como tracejada. É envelope próprio, com
spec própria, e ficou plausível só depois do retreino do §41, que ensinou a
máscara a separar a resposta da tracejada. Ver `HANDOFF_P2_7.md` §42.

> **Nota:** o código cita um `contract.md` em ~12 lugares (`contrato §1`, `§2`,
> `§4`, `§5`, `§6`) que **não está no repositório**. Era o contrato de execução
> passado aos implementadores. Pelas citações: §1 = modelos analíticos, `T_dom` e
> amplitude do degrau; §2 = regra anti-vazamento; §4 = esquema do `meta.json`;
> §5 = determinismo e independência de workers; §6 = "nunca levante exceção".
> Reconstruí-lo é item aberto.

---

## 5. Os dois fluxos

**Geração:**

```
seed ──► SeedSequence.spawn(3)
           │
           ├──► sample_system  ──► SystemSpec ──┐
           ├──► sample_style   ──► RenderStyle ─┤  (não se cruzam antes daqui)
           └──► rng_noise ──────────────────────┤
                                                ▼
                                        render_sample()
                                                │
                     ┌──────────────────────────┼──────────────────────────┐
                     ▼                          ▼                          ▼
                image.png                   mask.png                  meta.json
             (figura completa)      (2ª figura, só a curva)            (contrato)
```

**Verificação — o pipeline-oráculo da Parte 1:**

```
meta.json["series"]  ──►  identify.identify(t, y)  ──►  parâmetros estimados
       │                                                        │
       └────────── params verdadeiros ──────────────────────────┘
                                  │
                                  ▼
                     MAPE por parâmetro, por estrato
                                  │
                   record_criterion(...) ──► reports/part1_metrics.md
```

---

## 6. Invariantes que atravessam o projeto

1. **`img.py` é evidência, não código.** Gerador legado defeituoso, preservado de
   propósito: desenhava `axhline(K)`, `axvline(θ)` e coloria por tipo de
   amortecimento — por isso "acertava" 93%. Não editar, não apagar, não importar.
2. **`reports/*.md` são gerados.** Regenerar com `pytest`, nunca editar à mão.
3. **A suíte mede a realidade; não existe para passar.** Limiar mal calibrado se
   corrige **com medição**, nunca com afrouxamento.
4. **Nunca `np.random` global**, nunca `time`/`uuid`/hash dependente de
   `PYTHONHASHSEED` — quebram o determinismo bit-a-bit.
5. **Paralelize com processos, nunca threads** (`_estimate_gain` é memoizado em
   estado global).
6. **Nada de estado global no gerador** — nem `pyplot`, nem layout automático.
7. **A verdade de terra é emitida, nunca inferida.** O gerador jamais tenta
   *descobrir* algo que ele já sabe.
8. **Conferências que quebram alto.** Geometria da máscara, tamanho do PNG. Um
   dataset silenciosamente errado foi exatamente o modo de falha do `img.py`.
9. **Um critério que nunca falha em nenhum mutante não é critério, é decoração.**
   Ao criar teste novo, meça também o seu poder — contra defeito injetado.

---

## 7. Glossário

**Oráculo.** Substituir um estágio pelo seu valor de verdade, para isolar o
resto. Na Parte 1, em vez de extrair a curva da imagem (A) e ler os eixos (B), a
suíte pega `series` direto do `meta.json`; e em vez da rede (C), usa os chutes
clássicos do próprio `identify`. Responde à pergunta mais básica: *com informação
perfeita, o problema é solúvel e os rótulos estão certos?* Se a resposta fosse
não, nada mais importaria. Como foi sim, vira a linha de base contra a qual a
Parte 2 mede degradação (critério 2.6: ΔMAPE ≤ 3 p.p.).

**MAPE** (*Mean Absolute Percentage Error*). Erro percentual absoluto médio:

```
MAPE = (100 / n) · Σ |p̂ᵢ − pᵢ| / |pᵢ|
```

No código são duas linhas (`tests/test_part1.py`):

```python
def _rel(a, b):    return abs(a - b) / abs(b)
def _mape(vals):   return 100.0 * np.mean(vals)
```

É **relativo** por necessidade: K é sorteado log-uniformemente entre 0,2 e 20 e τ
entre 0,05 e 50 s. Erro absoluto médio seria dominado pelas amostras de maior
magnitude. A patologia do MAPE é o denominador perto de zero — foi o que motivou
o Ruling J (ver *NMAE*).

**NMAE/T_dom.** Erro absoluto normalizado pela constante de tempo dominante, usado
para θ (Ruling J). θ é sorteado a partir de 0,05·T_dom, então o MAPE de θ mede o
piso do sorteio, não o estimador. Normalizar por `T_dom` responde à pergunta
certa: *o atraso estimado erra que fração da dinâmica do sistema?*

**NRMSE de reconstrução.** RMSE entre a resposta do modelo identificado e a curva
verdadeira, normalizado pela faixa de y. É a **métrica primária do trabalho**
(PLANO §1.5): para uso em controle, o que importa é o modelo reproduzir a
dinâmica, não carregar o rótulo "certo" numa região onde o rótulo é ambíguo.

**`T_dom`.** Constante de tempo dominante. `τ` para FOPDT; `1/(ζωn)` para ζ ≤ 1;
`(ζ + √(ζ²−1))/ωn` para ζ > 1 — a do polo lento (Ruling K). A forma
racionalizada evita cancelamento catastrófico para ζ grande. Definir isso errado
corrompeu silenciosamente toda a população superamortecida até alguém medir
`y[-1]/K` por faixa de ζ.

**`w`.** Largura da janela em múltiplos de `T_dom`, contada a partir do degrau:
`w = (t_end − θ)/T_dom`. O estrato assertado nos critérios 1.1/1.2 é `w ≥ 3`
(Ruling C) — abaixo disso a curva é cortada antes do regime permanente e K não é
identificável nem com informação perfeita.

**FOPDT.** *First-Order Plus Dead Time*: `G(s) = K·e^{−θs}/(τs+1)`. É o modelo de
referência da sintonia PID industrial (Ziegler–Nichols, Cohen–Coon, IMC).

**2ª ordem canônica.** `G(s) = Kωn²/(s² + 2ζωn·s + ωn²)`, cobrindo ζ
subamortecido, crítico e superamortecido.

**AIC.** Critério de informação de Akaike — resíduo penalizado pelo número de
parâmetros. `identify` ajusta **as duas** estruturas e escolhe pelo menor AIC, em
vez de decidir por regra dura, porque a distinção entre FOPDT e 2ª ordem
superamortecida é genuinamente ambígua.

**Não-identificabilidade prática.** Em ζ ≥ 1,6, ωn e ζ deslizam ao longo de uma
curva de nível produzindo a mesma dinâmica observável — medido em ζ ∈ [2,2; 3,0]:
erram ~85% enquanto K erra 0,28%, com `corr(erro_ωn, erro_ζ) = +0,9997`. É limite
de informação, não deficiência do estimador (auditado por teste oracle-start
contra o limite de Cramér–Rao). Daí o Ruling N.

**Vazamento de rótulo** (*label leakage*). Um atributo da entrada correlacionado
ao gabarito, que permite acertar sem aprender o fenômeno. O `img.py` desenhava
`axhline(K)` e coloria por amortecimento; uma CNN treinada nisso aprende a
localizar linhas tracejadas e ler cor — não dinâmica.

**Distrator.** Reta de referência em posição sorteada, **descorrelacionada** dos
parâmetros. Substitui as linhas que vazavam e ainda torna o modelo robusto a
figuras reais que têm marcações.

**SNR.** Relação sinal-ruído em dB, com **potência do sinal = variância** (não
média quadrática) — Ruling L. É a convenção mais severa das duas.

**Ruling.** Decisão tomada durante a execução, quando um critério do PLANO se
mostrou impossível de verificar como escrito. Cada um está na tabela do
`HANDOFF.md` §6, com o custo caso esteja errado. Nenhum alvo numérico foi
afrouxado — a grandeza medida é que foi trocada por uma bem posta.

**IoU** (*Intersection over Union*). Métrica de segmentação da Parte 2: área da
interseção entre máscara predita e verdadeira, dividida pela área da união.

---

## 8. O que a Parte 1 entrega para as próximas

| entrega | uso |
|---|---|
| `mask.png` | Parte 2, Estágio A — verdade de terra de segmentação, validada contra a `axis_affine` a **0,03 px** de viés no estrato sem marcador |
| `plot_bbox_px`, `axis_affine`, `ticks` | Parte 2, Estágio B — verdade de terra de calibração |
| `render` | os estratos do critério 2.7, já registrados amostra a amostra |
| MAPE do oráculo | o denominador do critério 2.6 |
| `series` | Parte 1, oráculo do estágio D; Parte 2, verdade de terra do critério 2.2 |
| erro do extrator ingênuo por estilo de traço | dado de projeto do Estágio A: 0,184 px em linha sólida contra **0,896 px** em pontilhada, com **43,8% das colunas sem tinta** em `:` (§5.1 do relatório) — diz de antemão quanto será preciso interpolar |

E o modo oráculo continua sendo o instrumento de medida: a Parte 2 não pergunta
"meu extrator é bom?", e sim "**quanto** o pipeline degrada quando troco o
oráculo por percepção real?" — com o mesmo Estágio D, nas mesmas amostras. É por
isso que o desacoplamento entre os módulos importa: dá para substituir a origem
de `(t, y)` sem tocar em nada do resto.
