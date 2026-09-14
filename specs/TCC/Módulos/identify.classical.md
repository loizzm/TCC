---
tags: [modulo, estagio-d]
aliases: [classical.py, identify/classical.py]
---

# `identify/classical.py`

> **950 linhas · Estágio D.** Da série aos parâmetros da planta: mínimos quadrados não-lineares, seleção de estrutura e os baselines clássicos.

O maior módulo do projeto. Depende só de `numpy` e `scipy`.

---

## O que faz

Recebe `t[]` e `y[]` e devolve **qual estrutura de modelo** descreve a curva e **com que parâmetros**. Duas estruturas são ajustadas e comparadas:

```
FOPDT      G(s) = K·e^(−θs) / (τs + 1)                       3 parâmetros
2ª ordem   G(s) = K·ωn²·e^(−θs) / (s² + 2ζωn·s + ωn²)        4 parâmetros
```

A família é deliberadamente pequena. Ela cobre a maioria das plantas industriais que um curso de controle trata e — mais importante — é **identificável** a partir de uma única resposta ao degrau.

---

## Assinatura principal

```python
def identify(t, y) -> FitResult:
    """Ajusta FOPDT e 2ª ordem e escolhe pela verossimilhança penalizada
    com nº de pontos EFETIVO."""
```

### Recebe
Dois `float64[N]` em unidade física (ou no quadro normalizado, se a calibração falhou — o módulo não sabe a diferença).

### Devolve

```python
@dataclass
class FitResult:
    order: str                       # "fopdt" | "second"
    params: dict = field(default_factory=dict)
    aic: float = float("inf")        # AIC CLÁSSICO, mantido para a Parte 1
    nrmse: float = float("nan")      # resíduo normalizado — usado pela guarda
    sse: float = float("nan")
    success: bool = False
    n_params: int = 0
```

`params` é `{K, tau, theta}` para FOPDT, `{K, wn, zeta, theta}` para 2ª ordem.

### Caixa de parâmetros

```python
K_BOUNDS    = (1e-3, 1e4)
TAU_BOUNDS  = (1e-4, 1e4)
WN_BOUNDS   = (1e-4, 1e3)
ZETA_BOUNDS = (1e-3, 10.0)
```

> [!warning] Um parâmetro na borda não é uma medição
> É o otimizador desistindo. Numa planta instável (polo em s=+1,5) a saída veio com `K` no **teto exato** e `ζ` no **piso exato**, `nrmse` de 0,032 e `ok=true`. Não há guarda para isso hoje. Ver [[Arquitetura do projeto#Fora do envelope]].

---

## Como faz

### 1. Chute inicial por integrais, não por tentativa

```python
def _integral_guess_fopdt(t, y) -> dict | None:
def _integral_guess_second(t, y) -> dict | None:
```

Momentos da curva — áreas acumuladas — dão estimativas **em forma fechada** de K, τ e θ sem otimização nenhuma. É barato e coloca o otimizador perto do mínimo certo. Complementados por uma grade grossa (`_grid_guess_fopdt`, `_grid_guess_second`) e por heurísticas de forma (`_overshoot_guess` para subamortecido, `_overdamped_guess` para o resto).

### 2. SSE perfilado

```python
def _profiled_sse(basis: np.ndarray, y: np.ndarray, yy: float):
```

`K` entra **linearmente** no modelo. Em vez de otimizá-lo junto com os não-lineares, ele é resolvido em forma fechada para cada combinação dos demais. Isso reduz a dimensão do problema não-linear e **elimina uma direção inteira de mínimos locais**.

É por isso que existem `_fopdt_basis` e `_second_basis`: elas devolvem a base do modelo *sem* o ganho, e o ganho sai por projeção.

### 3. Multistart com refino em dois níveis

```python
_LSQ_COARSE = dict(method="trf", x_scale="jac", ftol=1e-9,  xtol=1e-9,  ...)
_LSQ_FINE   = dict(method="trf", x_scale="jac", ftol=1e-15, xtol=1e-15, ...)

def _multistart(fun, jac, starts, lo, hi, n_coarse=5, n_fine=2):
```

Vários pontos de partida são avaliados **grosseiramente** e só os melhores recebem refino fino. A superfície de erro com atraso é notoriamente multimodal — θ e τ trocam de papel com facilidade — e um único ponto de partida cai em mínimo local com frequência.

`_dedupe` descarta pontos de partida que já convergiriam para o mesmo lugar, e `_sanitize_fopdt` / `_sanitize_second` garantem que nenhum chute saia da caixa antes de entrar no otimizador.

---

## A correção do AIC — o coração do módulo

Escolher entre FOPDT e 2ª ordem é seleção de modelo, e o instrumento natural é o AIC. Mas o AIC clássico usa **n cru**, o que pressupõe observações independentes.

```python
def _rho1(resid: np.ndarray) -> float:
    """Autocorrelação de defasagem 1 do resíduo, saturada em [0, 0.99]."""
    r = resid - resid.mean()
    d = float(r @ r)
    return float(np.clip(float(r[:-1] @ r[1:]) / d, 0.0, 0.99))


def _n_efetivo(t, y, fit: FitResult) -> float:
    resid = y - model_response(fit.order, fit.params, t)
    rho = _rho1(resid)
    n_eff = float(t.size) * (1.0 - rho) / (1.0 + rho)
    return max(n_eff, float(fit.n_params) + 2.0)
```

```python
def identify(t, y) -> FitResult:
    r1, r2 = identify_both(t, y)
    ...
    n_eff = _n_efetivo(tc, yc, r2)
    ganho = n_eff * np.log(max(r1.sse, 1e-300) / max(r2.sse, 1e-300))
    return r2 if ganho > 2.0 * (r2.n_params - r1.n_params) else r1
```

> [!important] Por que a correção existe
> A série vem de uma polilinha extraída de imagem, onde pixels vizinhos carregam erro de extração **correlacionado**. Medido em `data/test`: o resíduo tem ρ ≈ 0,71, e `n` mediano 738 contra `n_eff` mediano **112**. O AIC tratava 738 pontos correlacionados como 738 evidências independentes.
>
> A consequência era estrutural: a 2ª ordem vence quando `SSE₁/SSE₂ > exp(2/n)`, e com n=806 bastava **0,234 %** de ganho de SSE. Ela consegue isso ajustando o *próprio artefato de extração* com o polo extra — o NRMSE das duas estruturas empatava (0,00353 × 0,00351).
>
> Resultado: **32 % das plantas de 1ª ordem** classificadas como 2ª ordem, contra 6 % quando o mesmo estágio recebe a série verdadeira.
>
> Trocar AIC por BIC **não resolve** (corrige só 40 % dos casos): o problema não é a constante da penalidade, é o *n* inflado.

Dois detalhes que não são acidentais:

- **ρ sai do resíduo da estrutura mais flexível** (2ª ordem), de propósito. Num modelo subespecificado a correlação do resíduo mistura ruído de extração com erro de estrutura, e superestimaria a correção.
- **`.aic` dos dois `FitResult` continua sendo o AIC clássico** e não mudou — `tests/conftest` o reporta na Parte 1.

### Onde a seleção ainda falha

Uma 2ª ordem **criticamente amortecida** (ζ = 1) com atraso é quase indistinguível de uma FOPDT com atraso maior. Num caso real medido, a 2ª ordem tinha SSE apenas 1,5 % melhor — o ganho ficou em 0,308 contra limiar 2,0, e a estrutura mais simples venceu. E a 2ª ordem teria devolvido ζ=1,44, não 1,0: **escolher a estrutura certa não teria dado o parâmetro certo**. É identificabilidade, não bug.

---

## Baselines clássicos

```python
def baseline_tangent(t, y) -> dict:
def baseline_smith(t, y) -> dict:
def baseline_sundaresan_krishnaswamy(t, y) -> dict:
```

Só FOPDT. **Não competem com o ajuste** — existem como referência de comparação no texto do TCC. São os métodos que um engenheiro faria à mão sobre o mesmo gráfico, e ter os três implementados é o que permite dizer quanto o ajuste numérico ganha.

Auxiliares: `_estimate_gain` (com cache LRU de 8 entradas, porque os três baselines pedem o mesmo ganho da mesma série) e `_crossing_time`.

---

## Função utilitária pública

```python
def model_response(order: str, params: dict, t: np.ndarray) -> np.ndarray:
```

Avalia qualquer das duas estruturas. Usada pela própria correção do AIC e pelos testes.

---

Ver também: [[identify.pipeline]] (quem chama) · [[Decisões de projeto#D6 · AIC com n efetivo]]
