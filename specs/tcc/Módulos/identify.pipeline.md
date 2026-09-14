---
tags: [modulo, estagio-cola]
aliases: [pipeline.py, identify/pipeline.py]
---

# `identify/pipeline.py`

> **572 linhas · a única porta de entrada do sistema.** Cola os estágios A, B, C e D, aplica as guardas de plausibilidade e monta a saída em dois níveis.

Depende de: [[identify.calibrate]] · [[identify.extract]] · [[identify.polyline]] · [[identify.classical]]
Nenhum estágio conhece outro — só este módulo conhece todos.

---

## O que faz

Recebe uma imagem e devolve os parâmetros da planta. Entre as duas pontas:

1. Chama a **calibração** (estágio B) — independente da máscara.
2. Chama o **extrator** (estágio A) — U-Net por padrão, injetável.
3. Reduz a máscara a uma **polilinha** (estágio C), recortada à moldura que B achou.
4. Ajusta e escolhe a estrutura (estágio D), em unidade física se B fechou, no quadro normalizado se não.
5. Aplica as **guardas** e monta o dicionário de saída.

**Nunca levanta exceção.** Toda falha vira `ok=False` com um `reason` que a nomeia.

---

## Assinatura

```python
def identify_from_image(image_rgb: np.ndarray, model, device: str = "cpu",
                        extractor=None) -> dict:
```

### Recebe

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `image_rgb` | `uint8[H,W,3]` | RGB puro, sem canal alfa |
| `model` | `UNet` \| `None` | Checkpoint carregado por [[identify.extract]]`.load_model`. Ignorado se `extractor` for dado |
| `device` | `str` | `"cpu"` ou `"cuda"` |
| `extractor` | `callable` \| `None` | `f(image_rgb) -> uint8[H,W]` 0/255. Troca a U-Net por outro estágio A com o mesmo contrato de saída |

O parâmetro `extractor` é como o extrator clássico entra na suíte **sem código condicional espalhado** — é o contrato estreito de A → C que torna isso possível.

### Devolve

```python
{
  "order":      "fopdt" | "second" | "",     # estrutura escolhida
  "params":     {...} | {},                  # bloco FÍSICO (vazio se não houver)
  "ok":         bool,                        # "há saída física", não "há resposta"
  "reason":     str,                         # nomeia a falha; "" se ok
  "dimensionless": {...},                    # NUNCA nulo — critério 2.11
  "physical":   {...} | None,                # None exatamente quando a calibração falha
  "physical_parcial": {...},                 # o que cada eixo permitiu, aditivo
  "calibration": {"ok", "reason", "ok_x", "ok_y",
                  "T_s", "y_faixa", "n_pairs_x", "n_pairs_y"},
  "latency_ms": float,
  "n_points":   int,
}
```

O bloco `dimensionless` tem sempre as seis chaves — `zeta`, `wn_T`, `tau_T`, `theta_T`, `theta_tau`, `K_yrange`. Quando não há nada a preencher, os valores são `None`, **nunca a chave ausente e nunca exceção**:

```python
def _vazio_adimensional() -> dict:
    return {"zeta": None, "wn_T": None, "tau_T": None,
            "theta_T": None, "theta_tau": None, "K_yrange": None}
```

---

## Como faz

### O fluxo, no essencial

```python
cal = calibrate(image_rgb)
mask = extractor(image_rgb) if extractor is not None else predict_mask(model, image_rgb, device)
x_px, y_px = mask_to_polyline(mask, bbox=cal.bbox_px if any(cal.bbox_px) else None)

if x_px.size < 10:
    return _saida("", {}, False, "polilinha_curta", _vazio_adimensional(), cal, x_px.size)

if cal.ok:
    t, y = polyline_to_series(x_px, y_px, cal)
    ordem = np.argsort(t); t, y = t[ordem], y[ordem]
    fit = identify(t, y)
    ...
```

Repare que **a moldura de B é usada mesmo quando B falha**: `cal.bbox_px` continua válido porque a falha é no mapeamento de unidades dos eixos, não na detecção do retângulo. Isso recorta título, rótulo de eixo e legenda externa, que vivem fora do quadro e não são a curva.

### Dois caminhos, um só ajuste

Quando a calibração fecha, o bloco adimensional é **derivado do ajuste físico** em vez de sair de um segundo ajuste. Isso evita um ajuste a mais por imagem e faz os dois níveis concordarem por construção — medindo antes, ajustes independentes divergiam em 50,5 % no p95.

Quando não fecha, a série vai para o **quadro normalizado** (T = 1, faixa de y = 1), e `_adimensional` recebe as duas como 1:

```python
tn, yn = _serie_normalizada(x_px, y_px, bbox_px=cal.bbox_px)
g = identify(tn, yn)
dim = _adimensional(g.params, 1.0, 1.0) if g.success else _vazio_adimensional()
```

`order` sai preenchido nesse caminho — a estrutura é adimensional. `params` fica vazio, porque ele é por contrato o bloco físico.

### O nível de repouso é uma janela FIXA, não uma fração

```python
_N_REPOUSO = 5

def _nivel_de_repouso(y: np.ndarray) -> float:
    n = int(min(_N_REPOUSO, y.size))
    return float(np.median(y[:max(1, n)]))
```

Antes era 8 % da largura, o que supõe prefixo plano por tempo morto. Com θ = 0 a curva já subiu ~28 % dentro dessa janela, o patamar sai 3,8 % baixo, e isso vira **12,6 % de erro em ζ** porque ζ vem da razão de overshoot. Uma janela pequena e fixa é correta nos dois regimes. Medido: MAPE de ζ cai de 2,92 % para 1,34 %.

### Físico parcial: o que cada eixo permite

```python
def _escalas_por_eixo(cal) -> tuple[float | None, float | None]:
    x0, y0, x1, y1 = cal.bbox_px
    T = abs(float(cal.sx)) * float(x1 - x0) if cal.ok_x and ... else None
    yf = abs(float(cal.sy)) * float(y1 - y0) if cal.ok_y and ... else None
    return T, yf
```

`zeta` nunca depende de eixo. `wn`, `tau` e `theta` dependem **só do eixo X**; `K` **só do eixo Y**. Onde o eixo não calibrou, a chave sai `None` — nunca um número que finge unidade que não existe.

---

## As guardas

```python
_NRMSE_MAX = 0.13        # p98 do corpus (n=837)
_UNDERSHOOT_MAX = 0.08   # recalibrado na promoção do base 32

def _implausivel(y: np.ndarray, nrmse: float) -> str:
    if _undershoot(y) > _UNDERSHOOT_MAX:
        return "resposta_inversa"
    if not np.isfinite(nrmse) or nrmse > _NRMSE_MAX:
        return "ajuste_inconsistente"
    return ""
```

O `_undershoot` mede a excursão **contrária à do degrau, antes de a resposta arrancar** — a definição vem da física da fase não-mínima, e foi assim que ela deixou de acusar falso positivo numa curva de ζ=0 legítima (0,147 → 0,0044):

```python
def _undershoot(y: np.ndarray, frac_alvo: float = 0.10) -> float:
    y0 = _nivel_de_repouso(y)
    faixa = float(np.ptp(y))
    d = np.sign(float(np.median(y[y.size // 2:])) - y0)     # direção do degrau
    subiu = np.flatnonzero((y - y0) * d > frac_alvo * faixa)
    fim = int(subiu[0]) if subiu.size else y.size           # onde a resposta arranca
    return max(float(np.max((y0 - y[:fim]) * d)), 0.0) / faixa
```

Quando uma guarda dispara, a saída **não devolve nem o nível adimensional** — os dois níveis saem do mesmo ajuste, então devolver um seria trocar um número errado por outro.

---

## Códigos de recusa

| `reason` | Significado |
|---|---|
| `polilinha_curta` | Menos de 10 pontos extraídos — a curva não foi encontrada |
| `ajuste_falhou` | O otimizador não convergiu para nenhuma das duas estruturas |
| `ajuste_inconsistente` | Convergiu, mas o resíduo é alto demais |
| `resposta_inversa` | Fase não-mínima — fora da família de modelos |
| `bbox_not_found` | A moldura não foi localizada |
| `ocr_insuficiente` | Menos de 2 rótulos por eixo |
| `ransac_failed` | Os rótulos não formam reta consistente |
| `calibration_failed` | Ticks não equiespaçados |
| `sinal_de_escala_invalido` | Escala com sinal impossível |

Ver também: [[Arquitetura do projeto#6. Guardas: recusar é melhor que errar com confiança]] · [[Decisões de projeto#D1 · Saída em dois níveis]]
