---
tags: [tcc, estagio-d, aic, selecao-de-modelo]
aliases: [AIC, Seleção de ordem, n efetivo]
---

# AIC e seleção de ordem

> Como o Estágio D decide entre **FOPDT** e **2ª ordem**, por que o AIC clássico dava a resposta errada, e o que a correção muda.

Implementa: [[Módulos/identify.classical|identify.classical]] · Decisão: [[Decisões de projeto#D6 · AIC com n efetivo]]

---

## O problema

Duas estruturas são ajustadas à mesma série:

```
FOPDT      G(s) = K·e^(−θs) / (τs + 1)                    k = 3   (K, τ, θ)
2ª ordem   G(s) = K·ωn²·e^(−θs) / (s² + 2ζωn·s + ωn²)     k = 4   (K, ωn, ζ, θ)
```

**A de 2ª ordem sempre ajusta melhor ou igual.** Não é acaso: com ζ → ∞ ela degenera em duas constantes de tempo, e a família de 1ª ordem está *contida* na de 2ª. Um parâmetro a mais nunca piora o SSE.

Então comparar SSE é inútil — a resposta seria "2ª ordem" sempre. É preciso um critério que **cobre pelo parâmetro extra**.

---

## O que o AIC é

Akaike (1974) derivou, da divergência de Kullback–Leibler, que a quantidade a minimizar é

```
AIC = −2·ln(verossimilhança máxima) + 2k
```

Sob resíduo gaussiano de variância desconhecida, isso vira o que está no código:

```python
def _metrics(y, resid, k) -> tuple[float, float, float]:
    n = int(resid.size)
    sse = float(resid @ resid)
    sse_safe = max(sse / max(n, 1), 1e-300)
    aic = float(n * np.log(sse_safe) + 2 * k)
    return sse, nrmse, aic
```

Dois termos, e a tensão entre eles é o critério inteiro:

| termo | o que faz |
|---|---|
| `n·ln(SSE/n)` | **prêmio pelo ajuste** — cai quando o modelo descreve melhor |
| `+2k` | **preço da complexidade** — sobe 2 por parâmetro |

A interpretação: cada parâmetro extra precisa "pagar" 2 unidades de log-verossimilhança. Se não pagar, o modelo mais simples vence.

> [!note] Por que 2 e não outro número
> O 2 não é arbitrário nem ajustável. Sai da correção assintótica do viés do estimador de máxima verossimilhança da divergência KL — a esperança do log-verossimilhança na amostra superestima a da população por aproximadamente `k`, e o fator 2 vem da convenção `−2·ln L`.
>
> O **BIC** troca `2k` por `ln(n)·k`, penalizando mais quando `n` é grande. Isso não resolve o problema deste projeto — ver adiante.

---

## A regra de decisão

```python
def identify(t, y) -> FitResult:
    r1, r2 = identify_both(t, y)            # FOPDT, 2ª ordem
    ...
    n_eff = _n_efetivo(tc, yc, r2)
    ganho = n_eff * np.log(max(r1.sse, 1e-300) / max(r2.sse, 1e-300))
    return r2 if ganho > 2.0 * (r2.n_params - r1.n_params) else r1
```

É **exatamente** `AIC₂ < AIC₁`, reescrito. A álgebra:

```
AIC₁ − AIC₂ = n·ln(SSE₁/n) + 2k₁ − n·ln(SSE₂/n) − 2k₂
            = n·ln(SSE₁/SSE₂) − 2(k₂ − k₁)
```

Escolher a 2ª ordem quando `AIC₂ < AIC₁` é escolher quando `n·ln(SSE₁/SSE₂) > 2(k₂−k₁)`. Com `k₂−k₁ = 1`, o limiar é **2,0**.

Isolando, a 2ª ordem vence quando

```
SSE₁ / SSE₂  >  exp(2/n)
```

**E aqui está o problema inteiro, numa fórmula.** O limiar depende de `n`, e depende *forte*:

| n | melhora de SSE exigida |
|---:|---:|
| 50 | 4,08 % |
| **112** | **1,80 %** |
| 300 | 0,67 % |
| **738** | **0,27 %** |
| 2 000 | 0,10 % |

Quanto mais pontos, mais barato fica comprar o parâmetro extra. Faz sentido: mais observações são mais evidência. **Desde que sejam mesmo observações independentes.**

---

## Por que o AIC clássico errava aqui

A série não vem de um experimento. Vem de uma **polilinha extraída de imagem**, com uma amostra por coluna de pixel. E pixels vizinhos carregam **erro de extração correlacionado** — a máscara não erra aleatoriamente coluna a coluna, ela erra em trechos.

Medido em `data/test`:

| | |
|---|---:|
| autocorrelação de defasagem 1 do resíduo (ρ) | **≈ 0,71** |
| `n` mediano | 738 |
| `n_eff` mediano | **112** |

O AIC tratava **738 pontos correlacionados como 738 evidências independentes**.

### A consequência, e por que ela é estrutural

Com n = 738, bastava **0,27 %** de melhora de SSE para a 2ª ordem vencer. E ela consegue esse 0,27 % — não descrevendo melhor a planta, mas **ajustando o próprio artefato de extração** com o polo extra.

A evidência que fecha o argumento: o NRMSE das duas estruturas **empatava** — 0,00353 contra 0,00351. As duas descreviam a curva igualmente bem. A de 2ª ordem só tinha um grau de liberdade a mais para absorver o ruído de extração, e o AIC inflado o comprava barato.

| | plantas de 1ª ordem classificadas como 2ª |
|---|---:|
| pipeline real (série extraída) | **32 %** |
| oráculo (série verdadeira) | **6 %** |

A diferença entre 32 % e 6 % **é** o erro de extração sendo lido como estrutura.

---

## A correção

```python
def _rho1(resid: np.ndarray) -> float:
    """Autocorrelação de defasagem 1 do resíduo, saturada em [0, 0.99]."""
    r = resid - resid.mean()
    d = float(r @ r)
    if not np.isfinite(d) or d <= 0.0:
        return 0.0
    return float(np.clip(float(r[:-1] @ r[1:]) / d, 0.0, 0.99))


def _n_efetivo(t, y, fit: FitResult) -> float:
    resid = y - model_response(fit.order, fit.params, t)
    rho = _rho1(resid)
    n_eff = float(t.size) * (1.0 - rho) / (1.0 + rho)
    return max(n_eff, float(fit.n_params) + 2.0)
```

A fórmula `n·(1−ρ)/(1+ρ)` é o **tamanho amostral efetivo** de uma série AR(1): quantas observações independentes carregariam a mesma informação. Com ρ = 0,71, cada 6,6 pontos valem 1.

Com `n_eff = 112`, a exigência sobe de 0,27 % para **1,80 %** — quase 7× mais cara. Um ganho de 0,27 % deixa de ser suficiente.

### Três detalhes que não são acidentais

**ρ sai do resíduo da estrutura mais flexível.** `_n_efetivo` recebe `r2` (2ª ordem), não `r1`. Num modelo subespecificado, a correlação do resíduo mistura ruído de extração com **erro de estrutura** — a curva sistemática que o modelo simples não consegue seguir. Isso superestimaria ρ, e portanto a correção.

**ρ é saturado em [0, 0.99].** O piso em 0 impede que anticorrelação (ρ < 0) *infle* `n_eff` acima de `n`. O teto em 0,99 impede a divisão explodir.

**`n_eff` tem piso em `k+2`.** Com menos pontos efetivos que parâmetros, o critério perde sentido — o piso mantém a comparação bem definida.

### Por que BIC não resolveria

Testado: corrige **só 40 %** dos casos.

O BIC troca `2k` por `ln(n)·k`, mexendo no **lado direito** da desigualdade. Mas o defeito está no **lado esquerdo** — no `n` que multiplica o logaritmo. Uma penalidade maior atenua o sintoma sem tocar na causa: continua tratando pontos correlacionados como independentes, só cobra mais caro por parâmetro.

> A escolha certa não é penalizar mais. É **contar direito**.

---

## O que o `.aic` do `FitResult` ainda é

```python
@dataclass
class FitResult:
    aic: float = float("inf")        # AIC CLÁSSICO, e NÃO mudou
```

O campo continua sendo o AIC de Akaike com `n` cru. A correção vive **só** em `identify()`, no momento da decisão. Motivo: `tests/conftest` reporta esses valores na Parte 1, e mudá-los quebraria a comparabilidade histórica.

A docstring diz o que isso significa: o critério *"equivale ao AIC quando o resíduo é branco; difere dele exatamente na medida em que a polilinha extraída é autocorrelacionada."*

---

## Onde a seleção ainda falha, e por que não é bug

Uma 2ª ordem **criticamente amortecida** (ζ ≈ 1) com atraso é quase indistinguível de uma FOPDT com atraso maior.

Num caso real medido (`Figure_f3`):

| | |
|---|---|
| SSE da 2ª ordem | **1,5 % melhor** |
| ganho no teste | 0,308 |
| limiar | 2,0 |
| escolhida | **FOPDT** |
| θ recuperado — FOPDT | erro de 5,9 % |
| θ recuperado — 2ª ordem | erro de 1,6 % |

A 2ª ordem teria sido melhor. Mas repare no que ela devolveria: **ζ = 1,44**, não o ζ = 1,0 verdadeiro.

> **Escolher a estrutura certa não teria dado o parâmetro certo.**

Isso é o que separa *identificabilidade* de *bug*. A informação que distingue as duas estruturas quase não existe nos dados — e nem uma máscara perfeita a criaria. Burnham & Anderson tratam exatamente disso: ΔAIC pequeno significa **modelos equivalentes na evidência**, não classificador ruim.

Registro relacionado, sob a mesma leitura: `2.12-ordem` mede **89,0 %** de acerto de ordem no corpus, e **100,0 %** no subconjunto sem calibração (n=20).

### Dívida aberta

Quando o ganho fica muito abaixo do limiar, a evidência para a estrutura complexa simplesmente **não existe**. Hoje a pipeline escolhe a mais simples **sem avisar** — deveria declarar a ordem como incerta. Ver [[Decisões de projeto#Dívidas conhecidas]].

---

## Referências

| | |
|---|---|
| AIC | **Akaike, H.** A new look at the statistical model identification. *IEEE TAC* 19(6), p. 716–723, 1974 |
| ΔAIC, AICc, modelos equivalentes | **Burnham, K. P.; Anderson, D. R.** *Model Selection and Multimodel Inference*. 2. ed. Springer, 2002 |
| Seleção de estrutura em identificação | **Ljung, L.** *System Identification: Theory for the User*. 2. ed., 1999 |
| Identificabilidade prática | **Raue, A. et al.** *Bioinformatics* 25(15), 2009 |

> [!warning] A correção `n_eff` está sem citação
> É a peça algorítmica mais citável do projeto e a única sem fundamentação bibliográfica registrada em `REFERENCIAS.md`. A ideia de tamanho amostral efetivo é clássica — Bartlett (1935), Bayley & Hammersley (1946) — mas a aplicação a critérios de informação sobre série extraída de imagem precisa de busca dirigida.
>
> Detalhe em [[Proveniência#1. A correção `n_eff` está sem citação]].
