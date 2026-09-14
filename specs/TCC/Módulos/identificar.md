---
tags: [modulo, cli, ferramenta]
aliases: [identificar.py, CLI]
---

# `identificar.py`

> **204 linhas · casca de linha de comando.** Coloca uma imagem, recebe a resposta.

**Toda a lógica vive em [[identify.pipeline]]`.identify_from_image`.** Nada aqui muda o resultado — se mudasse, o número que aparece no terminal não seria o mesmo que a suíte mede.

---

## Uso

```bash
.venv/bin/python identificar.py grafico.png
.venv/bin/python identificar.py *.png              # várias de uma vez
.venv/bin/python identificar.py grafico.png --json # saída para script
.venv/bin/python identificar.py grafico.png --classico  # sem torch, sem GPU
```

| Flag | Efeito |
|---|---|
| `--json` | Saída em JSON, para encadear em script |
| `--classico` | Usa [[identify.extract_classical]], dispensa `torch` e GPU |
| `--modelo PATH` | Outro checkpoint (padrão: `models/unet_stageA.pt`) |
| `--cpu` | Força CPU mesmo com GPU disponível |

Aceita **várias imagens** numa chamada — o modelo é carregado uma vez só.

---

## O que faz

### 1. Carrega a imagem no contrato de entrada

```python
def _carrega_imagem(caminho: Path) -> np.ndarray:
    """RGB puro, sem alfa — o contrato de entrada do estagio A."""
    with Image.open(caminho) as im:
        return np.asarray(im.convert("RGB"))
```

O `.convert("RGB")` não é cosmético: um PNG com canal alfa quebraria o contrato `uint8[H,W,3]`.

### 2. Escolhe o extrator

```python
if a.classico:
    from identify.extract_classical import extract_mask_classical
    extrator = extract_mask_classical
else:
    import torch
    from identify.extract import load_model
    dev = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = load_model(str(a.modelo), dev)
```

O `import torch` está **dentro do ramo**: com `--classico` o script roda numa máquina que não tem torch instalado.

### 3. Chama a pipeline e traduz

```python
r = identify_from_image(img, modelo, dev, extractor=extrator)
```

---

## A tradução dos motivos de recusa

É a única coisa que este arquivo acrescenta de substantivo:

```python
MOTIVOS = {
    "polilinha_curta":
        "a curva nao foi encontrada na imagem (menos de 10 pontos extraidos)",
    "ajuste_inconsistente":
        "o ajuste convergiu mas o residuo e alto demais — a serie extraida nao "
        "sustenta modelo nenhum (mascara provavelmente saltou para um distrator)",
    "resposta_inversa":
        "a curva desce antes de subir. Isso e fase nao-minima (zero no semiplano "
        "direito), que NAO pertence a familia de modelos deste sistema",
    "ocr_insuficiente":
        "menos de 2 rotulos numericos lidos por eixo",
    ...
}
```

A pipeline **nomeia a causa** em vez de devolver número errado em silêncio. Traduzir isso aqui é o que separa "não funcionou" de "não funcionou **por isto**" — e o código bruto também é impresso, para o relatório continuar rastreável ao código.

---

## As três formas de saída

### Sucesso com calibração

```
Estrutura      1a ordem com atraso (FOPDT)

  Parametros fisicos
    K            0.9867
    tau          1.21 s
    theta        0.3402 s

  Janela lida    12.5 s   ·   faixa de y  1.184
  Rotulos        7 no eixo x, 5 no eixo y

  738 pontos extraidos · 214 ms
```

### Sucesso sem calibração — o nível adimensional

```
  Sem parametros fisicos — menos de 2 rotulos numericos lidos por eixo
  Eixos aprovados: x=sim, y=nao

  Adimensional (independe da escala dos eixos)
    zeta         0.4213
    wn·T         8.44     (multiplique por 1/T para ter rad/s)

  Parcial (do eixo que foi aprovado)
    wn           0.6752 rad/s
    theta        0.4100 s
```

Esse bloco existe por causa da [[Decisões de projeto#D1 · Saída em dois níveis|Decisão E]]: **amortecimento se lê da forma da curva, não da escala dos eixos**. A calibração dá a *unidade*; ωn em rad/s precisa dela, ζ não.

### Recusa

```
  SEM RESPOSTA — a curva desce antes de subir. Isso e fase nao-minima
  (zero no semiplano direito), que NAO pertence a familia de modelos
  (codigo: resposta_inversa)
```

---

## Formatação

```python
def _fmt(v, casas=4, unidade=""):
    if v is None:                                    return "—"
    if isinstance(v, float) and abs(v) < 1e-12:      return f"0{unidade}"
    return f"{v:.{casas}g}{unidade}"
```

`None` vira travessão, nunca `0` nem `nan` — o consumidor precisa distinguir "vale zero" de "não foi medido".

---

Ver também: [[identify.pipeline]] · [[Arquitetura do projeto]]
