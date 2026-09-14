---
tags: [tcc, indice, moc]
aliases: [Mapa de módulos]
---

# Módulos

> Um arquivo por módulo: **o que faz**, **como faz** (com trechos do código real), **o que recebe** e **o que devolve**.

---

## Cadeia de produção

```
                    ┌─────────────────────────┐
        imagem ────►│ identify/pipeline.py    │◄──── única porta de entrada
                    └───────────┬─────────────┘
              ┌─────────────┬───┴────┬──────────────┐
              ▼             ▼        ▼              ▼
      calibrate.py    extract.py  polyline.py  classical.py
         (B)         (A)  │          (C)           (D)
                          │
                   extract_classical.py
                   (A alternativo, sem torch)
```

| Módulo | Estágio | Linhas | Papel | Depende de |
|---|---|---:|---|---|
| [[identify.pipeline]] | cola | 572 | Única porta de entrada. Guardas e saída em dois níveis | todos abaixo |
| [[identify.extract]] | **A** | 139 | U-Net compacta, letterbox, inferência | torch |
| [[identify.extract_classical]] | **A** | 184 | Alternativa sem rede, e a linha de base | cv2 · *nunca torch* |
| [[identify.calibrate]] | **B** | 746 | Moldura, blobs, OCR, afim por RANSAC | cv2, pytesseract |
| [[identify.polyline]] | **C** | 155 | Máscara → série amostrada | cv2, skimage |
| [[identify.classical]] | **D** | 950 | Ajuste LSQ e seleção de estrutura | numpy, scipy |

## Referência visual

[[Anatomia da U-Net]] — o U desenhado, com canais, resolução, parâmetros e custo por bloco.

## Corpus e treino

| Módulo | Linhas | Papel | Depende de |
|---|---:|---|---|
| [[dataset.generator]] | 709 | Síntese do corpus e da verdade exata | matplotlib |
| [[dataset.randomize]] | 396 | Sorteio do estilo visual, **cego ao sistema** | numpy |
| [[train_unet]] | 172 | Treino do Estágio A, estratos acumuláveis por CLI | torch |

## Ferramenta

| Módulo | Linhas | Papel |
|---|---:|---|
| [[identificar]] | 204 | Casca de linha de comando para uso avulso |

---

## A direção das dependências é intencional

**`pipeline.py` conhece todos; nenhum estágio conhece outro.**

Duas consequências que se pagaram:

- **`polyline.py` não importa `torch`.** Permite testar o estágio C contra máscaras *verdadeiras*, sem GPU. Foi essa separação que permitiu descobrir que a métrica de erro do estágio media declividade e não geometria — se C só pudesse ser exercitado através de A, os dois erros estariam misturados.
- **`extract_classical.py` nunca importa `torch`**, e um teste assevera isso. É o que torna a linha de base honesta: se ela dependesse do mesmo ecossistema, "funciona sem GPU" seria afirmação não verificada.

## O contrato estreito é o que permite trocar A

```python
identify_from_image(img, model, device, extractor=...)
```

Qualquer função `f(image_rgb) -> uint8[H,W]` 0/255 serve. É assim que o extrator clássico entra na suíte **sem código condicional espalhado**.

---

Ver também: [[Arquitetura do projeto]] · [[Decisões de projeto]] · [[Critérios de qualidade]]
