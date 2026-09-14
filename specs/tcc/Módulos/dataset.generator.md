---
tags: [modulo, dataset]
aliases: [generator.py, dataset/generator.py]
---

# `dataset/generator.py`

> **709 linhas.** Sintetiza o corpus: sorteia um sistema, resolve a resposta ao degrau, renderiza a figura com matplotlib e produz a máscara de verdade.

Não é ferramenta auxiliar. É **parte da arquitetura** — ver [[Arquitetura do projeto#7. O gerador é parte da arquitetura, não uma ferramenta auxiliar]].

---

## Por que sintetizar

Treinar a U-Net exige pares imagem/máscara com verdade **exata**. Nenhum acervo real oferece isso. Aqui a máscara sai de um **segundo render contendo só a curva** — a verdade é exata por construção, não anotada por ninguém.

---

## Estruturas

```python
@dataclass(frozen=True)
class SystemSpec:
    order: str                       # "fopdt" | "second"
    K: float
    tau: float | None                # FOPDT
    theta: float
    wn: float | None                 # 2ª ordem
    zeta: float | None
    t_start: float
    t_end: float
    step_amplitude: float

    @property
    def t_dom(self) -> float:        # constante de tempo dominante
        return dominant_time_constant(self.order, self.tau, self.wn, self.zeta)
```

---

## Funções principais

### `sample_system(rng) -> SystemSpec`

```python
def sample_system(rng: np.random.Generator) -> SystemSpec:
    order = ORDERS[int(rng.integers(0, 2))]
    K = _loguniform(rng, 0.2, 20.0)
    if order == "fopdt":
        tau = _loguniform(rng, 0.05, 50.0);  wn = zeta = None
    else:
        tau = None
        wn = _loguniform(rng, 0.02, 20.0)
        zeta = float(rng.uniform(0.10, 3.00))
    t_dom = dominant_time_constant(order, tau, wn, zeta)
    theta = _loguniform(rng, 0.05, 1.0) * t_dom
    t_end = theta + _loguniform(rng, 0.5, 6.0) * t_dom
    return SystemSpec(...)
```

**Este é o envelope de validade do projeto inteiro.** Tudo o que o sistema sabe identificar bem está dentro dessas faixas:

| Parâmetro | Faixa | Distribuição |
|---|---|---|
| K | 0,2 – 20 | log-uniforme |
| τ | 0,05 – 50 s | log-uniforme |
| ωn | 0,02 – 20 rad/s | log-uniforme |
| ζ | 0,10 – 3,00 | uniforme |
| θ | 0,05 – 1,0 × `t_dom` | log-uniforme |
| janela | θ + 0,5 – 6,0 × `t_dom` | log-uniforme |

Log-uniforme para K, τ e ωn porque são grandezas de escala: erro relativo é o que importa, e uma amostragem uniforme concentraria tudo na década de cima. ζ é uniforme porque **não** é de escala — 0,1 e 3,0 são regimes qualitativamente diferentes, não um múltiplo do outro.

> Consequência medida: com a janela em `0,5 a 6 × t_dom`, o corpus **praticamente nunca mostra o patamar** — a fração final já assentada tem mediana de 0,98 % da janela, e zero de 60 amostras chegam a 30 %. Foi por isso que o estrato `janela_assentada` teve de existir.

### `render_sample(spec, style, out_dir, ...) -> dict`

```python
def render_sample(
    spec: SystemSpec, style: RenderStyle, out_dir,
    add_noise: bool = True, rng=None, *, seed=None,
    reta_no_patamar: bool = False,
    anotacao_com_seta: bool = False,
    banda_de_acomodacao: bool = False,
) -> dict:
    """Escreve image.png, mask.png e meta.json em out_dir. Devolve o dict do meta."""
```

Os estratos são marcados numa **cópia** do estilo, nunca mutando o objeto recebido:

```python
style = replace(style, has_reference_line=bool(reta_no_patamar),
                has_annotation_arrow=bool(anotacao_com_seta),
                has_settling_band=bool(banda_de_acomodacao))
```

`render_sample` não é dona desse objeto, e mutar um argumento é efeito colateral observável para quem chamou.

### `generate_sample(out_dir, seed, ...) -> dict`

O anti-vazamento acontece aqui:

```python
ss = np.random.SeedSequence(int(seed))
children = ss.spawn(3)
rng_sys   = np.random.default_rng(children[0])
rng_style = np.random.default_rng(children[1])
rng_noise = np.random.default_rng(children[2])

spec  = sample_system(rng_sys)
style = sample_style(rng_style)      # nao ve o spec: anti-vazamento estrutural
```

**Três streams independentes.** Acrescentar um sorteio no estilo não desloca o sorteio do sistema — sem isso, nenhuma comparação entre rodadas de treino seria interpretável.

### `generate_dataset(out_dir, n, seed=0, workers=None, ...) -> list[str]`

```python
jobs = [(str(root / f"sample_{i:05d}"), int(seed) * 1_000_003 + i, ...) for i in range(n)]
with ProcessPoolExecutor(max_workers=workers) as ex:
    return list(ex.map(_generate_one, jobs, chunksize=4))
```

A semente de cada amostra é **derivada deterministicamente do índice**, não do estado de um RNG compartilhado. Por isso o resultado independe do número de workers.

### `load_sample(sample_dir) -> dict`

```python
def load_sample(sample_dir) -> dict:
    meta = json.load(open(d / "meta.json"))
    meta["series"] = {"t": np.asarray(...), "y": np.asarray(...)}
    meta["image"] = np.asarray(Image.open(d / "image.png").convert("RGB"), dtype=np.uint8)
    meta["mask"]  = np.asarray(Image.open(d / "mask.png").convert("L"),   dtype=np.uint8)
    return meta
```

---

## O que uma amostra contém

```
sample_00000/
├── image.png     figura completa: moldura, grade, legenda, distratores, ruído
├── mask.png      SÓ a curva, do segundo render
└── meta.json
```

Chaves do `meta.json`:

| Chave | Conteúdo |
|---|---|
| `schema_version`, `sample_id`, `seed` | rastreabilidade |
| `order`, `params` | a verdade: `{K, tau, theta, wn, zeta}` |
| `step_amplitude`, `t_window` | o degrau e a janela |
| `plot_bbox_px` | moldura verdadeira, em pixels |
| `axis_affine` | `{sx, ox, sy, oy}` — a verdade que o Estágio B tenta recuperar |
| `ticks` | pares `(pixel, valor)` verdadeiros, por eixo |
| `series` | `t[]` e `y[]` verdadeiros, antes do render |
| `noise` | `{enabled, snr_db, quantization_levels}` |
| `render` | todos os atributos de [[dataset.randomize]], para estratificar |

O bloco `render` é o que permite estratificar as métricas por grade, legenda, fundo escuro e estilo de traço no critério 2.7.

---

## Estratos opt-in

Cada defeito que uma imagem real expôs virou um estrato, **com o caminho padrão preservado byte a byte**.

| Estrato | O que renderiza | Degradação medida |
|---|---|---|
| `reta_no_patamar` | Reta de referência coincidente com o patamar, em cor de luminância colidente | motivou a mudança para RGB |
| `janela_assentada` | Estende `t_end` para ≥ `θ + 20·t_dom` | eixo separado, permite ablação |
| `banda_de_acomodacao` | `ax.axhspan` de ±5 % em torno do setpoint | IoU −0,063 (p = 2,6e−10) |
| `anotacao_com_seta` | Caixa de texto ligada por seta ao pico | IoU −0,113 (p = 1,6e−11) |

```python
if banda_de_acomodacao:
    alvo = float(spec.K * spec.step_amplitude)
    meia = 0.05 * abs(alvo) if abs(alvo) > 1e-9 else 0.05 * (ylim[1] - ylim[0])
    ax.axhspan(alvo - meia, alvo + meia,
               color=_cor_colidente(style.line_color), alpha=0.55, zorder=2, linewidth=0.0)
```

`_cor_colidente` escolhe uma cor de **luminância igual** à da curva — é o que torna o estrato adversarial contra um modelo de 1 canal.

> [!important] O critério para acrescentar um estrato
> Ele precisa **degradar o modelo atual de forma mensurável**. Um estrato que não move a métrica não reproduz o fenômeno, e treinar com ele seria desperdício. Foi o que aconteceu na primeira tentativa do estrato de reta coincidente, cuja versão inicial não degradava nada — e a investigação revelou que a janela curta era a causa, o que originou o `janela_assentada`.

`_T_DOM_ESTRATO = 20.0` foi escolhido por varredura medida (n=40 por ponto), com alvo no maior vão de colunas sem tinta das imagens reais. Acima de 20 o máximo vai a 0,83–0,89, **fora** da faixa real, e a mediana não sobe.

---

## Linha de comando

```bash
.venv/bin/python -m dataset.generator data/train 6000 0
#                                      saída     n    semente-base
```

---

Ver também: [[dataset.randomize]] · [[train_unet]] · [[Decisões de projeto#D7 · O estilo não pode ver o sistema]]
