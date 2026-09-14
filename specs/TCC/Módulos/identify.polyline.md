---
tags: [modulo, estagio-c]
aliases: [polyline.py, identify/polyline.py]
---

# `identify/polyline.py`

> **155 linhas · Estágio C.** Máscara de pixels → função amostrada `y(t)`. Determinístico, **sem torch**.

Não importar `torch` é deliberado: permite testar o estágio C contra máscaras *verdadeiras*, sem GPU nenhuma. Foi essa separação que permitiu descobrir que a métrica de erro do estágio estava medindo declividade e não geometria.

---

## O que faz

A máscara é uma região de largura variável; o estágio D precisa de um valor por instante.

```
uint8[H,W] 0/255  ──►  componentes conexas  ──►  esqueleto  ──►  mediana por coluna  ──►  interpolação  ──►  x_px[], y_px[]
```

---

## Assinatura

```python
def mask_to_polyline(mask: np.ndarray,
                     bbox: tuple[int, int, int, int] | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
```

### Recebe

| Parâmetro | Tipo | Descrição |
|---|---|---|
| `mask` | `uint8[H,W]` | Binarizada internamente com `> 127` |
| `bbox` | tupla \| `None` | Moldura da área de dados. `None` preserva o comportamento anterior byte a byte |

### Devolve
Dois `float64[N]` — coordenadas de pixel, **uma amostra por coluna**, já ordenadas em x. Devolve dois arrays vazios quando não há nada utilizável (`< 2` pontos).

O recorte à moldura existe porque título, rótulo de eixo e legenda externa vivem **fora** do quadro e não são a curva. Num caso real a polilinha ia de y=21 a 551 com a moldura em 39..503.

---

## Como faz

### 1. União das componentes, não a maior

```python
n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
uniao = np.zeros(binary.shape, dtype=bool)
for k in range(1, n):
    if stats[k, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX:
        uniao |= (lab == k)
skel = skeletonize(uniao)
```

> [!important] Por que não a maior componente
> Uma curva tracejada ou pontilhada é, **por construção**, uma sequência de componentes desconectadas — cada travessão é a sua própria componente. Manter só a maior descarta a curva quase inteira nesses estilos.
>
> Medido contra a máscara verdadeira: **40 de 300** amostras ficavam com menos de 10 pontos utilizáveis, e o RMSE do estrato `traco=:` estourava o alvo (2,43 px contra 2 px).
>
> O limiar `MIN_COMPONENT_PX = 2` ainda descarta ruído isolado de 1 px numa máscara *predita*.

### 2. A espessura do traço é a escala de referência

```python
espessuras_coluna = uniao.sum(axis=0)
espessuras_coluna = espessuras_coluna[espessuras_coluna > 0]
espessura_mediana = float(np.median(espessuras_coluna)) if espessuras_coluna.size else 1.0
VAO_MIN_FRAC = 3.0
```

Medida na máscara **antes** da esqueletização. Ela separa um vão real entre dois objetos (curva × legenda, curva × reta de referência) de uma quebra espúria dentro do *mesmo* objeto — anti-aliasing, ou o próprio padrão de um traço pontilhado.

### 3. Desambiguação por coluna

```python
for x in range(skel.shape[1]):
    linhas = np.flatnonzero(skel[:, x])
    if not linhas.size: continue

    if anterior is not None and (x - ultimo_x) > VAO_MIN_FRAC * espessura_mediana:
        anterior = None          # referência velha demais: descarta

    blocos = _blocos(skel[:, x])
    multi_ramo = len(blocos) > 1 and vao_maximo > VAO_MIN_FRAC * espessura_mediana

    if not multi_ramo or anterior is None:
        v = float(np.median(linhas))                # ramo único: mediana de tudo
    else:
        a, b = min(blocos, key=lambda t: ...)       # bloco mais próximo do ponto anterior
        dentro = linhas[(linhas >= a) & (linhas <= b)]
        v = float(np.median(dentro))
```

Três casos, e a ordem importa:

- **Ramo único** — mediana de todas as linhas. Idêntico ao comportamento antigo; mexer aqui foi medido e **piora** o sintético.
- **Ramo múltiplo com referência válida** — segue o bloco mais próximo do ponto anterior. É o que separa a curva da amostra de linha da legenda.
- **Vão largo demais desde o último ponto** — descarta a referência e cai no caso seguro. A curva pode ter se deslocado no meio do vão o bastante para que "o bloco mais próximo" escolha o bloco *errado*; melhor não arriscar com confiança falsa.

### 4. Interpolação de vãos, com corte

```python
MAX_GAP_FRAC = 0.15

x_full = np.arange(int(x_arr[0]), int(x_arr[-1]) + 1, dtype=float)
y_full = np.interp(x_full, x_arr, y_arr)

largura = x_arr[-1] - x_arr[0]
vaos = np.diff(x_arr)
for i in np.flatnonzero(vaos > MAX_GAP_FRAC * largura):
    corte = (x_full > x_arr[i]) & (x_full < x_arr[i + 1])
    y_full[corte] = np.nan            # vão longo demais não é traço: é ausência de dado
ok = ~np.isnan(y_full)
return x_full[ok], y_full[ok]
```

O estilo pontilhado deixa **43 % das colunas sem tinta nenhuma**. Sem interpolar, quase metade do domínio temporal some e o ajuste recebe uma série com buracos sistemáticos. Mas interpolar cegamente fabricaria dado — daí o teto de 15 % da largura.

---

## Função auxiliar exportada

```python
def polyline_to_series(x_px, y_px, cal) -> tuple[np.ndarray, np.ndarray]:
    """Pixels -> unidades físicas, com a afim estimada pelo Estágio B."""
    from identify.calibrate import px_to_data
    return px_to_data(cal, x_px, y_px)
```

O import é local de propósito — mantém o módulo leve no caminho em que só a geometria importa.

---

Ver também: [[Decisões de projeto#D5 · União das componentes conexas]] · [[Critérios de qualidade#2.2 — o piso do extrator de polilinha]]
