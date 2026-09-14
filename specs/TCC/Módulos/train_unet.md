---
tags: [modulo, treino]
aliases: [train_unet.py, treino]
---

# `train_unet.py`

> **172 linhas.** Treino do Estágio A. **Determinístico**: seed fixa, sem RNG global.

---

## O que faz

Monta o dataset a partir de diretórios de amostras, treina a [[identify.extract|U-Net]] com `dice + BCE`, avalia por IoU a cada época e **salva só quando melhora**.

```bash
.venv/bin/python train_unet.py \
    --train-dir data/train --train-dir data/train_reta_banda_seta \
    --val-dir data/val     --val-dir data/val_reta_banda \
    --base 32 --in-ch 3 --batch 6 --epochs 25 \
    --out models/unet_stageA_base32.pt
```

`--train-dir` e `--val-dir` são **acumuláveis**: cada repetição soma um split. É assim que os estratos entram no treino sem tocar em `data/train`, `data/val` e `data/test`, que ficam fixos para os números continuarem comparáveis com as rodadas anteriores.

---

## Argumentos

| Flag | Padrão | Para que serve |
|---|---|---|
| `--epochs` | 25 | |
| `--batch` | 8 | com `base=32` a 512×512 cabe 6 em 6 GB |
| `--size` | 512 | lado do letterbox |
| `--lr` | 3e-4 | |
| `--base` | 16 | canais da primeira camada (16, 24, 32) — hipótese (a) do Ruling 10 |
| `--in-ch` | 1 | 1 = cinza (comportamento anterior), 3 = RGB |
| `--train-dir` | `data/train` | repetível |
| `--val-dir` | `data/val` | repetível |
| `--batches-per-epoch` | 0 | limita os passos de gradiente por época |
| `--lr-patience` | 1 | épocas sem melhora antes de reduzir o LR |
| `--lr-factor` | 0,5 | |
| `--lr-threshold` | 0,01 | ganho mínimo de IoU_val que conta como melhora |

> [!note] Por que `--batches-per-epoch` existe
> Comparar 4 200 com 8 400 amostras mudaria **duas** variáveis de uma vez: diversidade de dados *e* número de passos de gradiente por época. Fixando os passos, a única diferença entre as duas curvas de IoU é a diversidade — que era exatamente a hipótese (b) que se queria testar.

---

## `MaskDataset`

```python
class MaskDataset(Dataset):
    def __init__(self, root: str | list[str], size: int = 512, in_ch: int = 1):
        roots = [root] if isinstance(root, str) else list(root)
        self.dirs = [d for r in roots for d in sorted(Path(r).glob("sample_*"))]
```

**Recebe** um ou vários diretórios · **Devolve** por item um par `(x, y)` de tensores `float32` normalizados 0–1.

### O alvo é CONTÍNUO

Esta é a decisão mais consequente do módulo, e tem histórico documentado no próprio código:

| Versão do alvo | Resultado medido |
|---|---|
| limiar 127 | perdia a curva quase toda em imagens grandes — cobertura de colunas caía a **0,4 %** |
| limiar 0 | recuperava a presença, mas inflava a área do alvo até **3,03×**; IoU de teste piorou de 0,572 para 0,495 apesar do IoU de validação subir para 0,91 |
| limiar 32 | melhor dos três: cobertura 85,1 %, inflação 2,14×, IoU de teste 0,560 — mas ζ ainda ficava 0,64 p.p. acima do alvo |
| **contínuo** | escolhido |

```python
y, _ = letterbox(m["mask"], self.size)
return xt, torch.from_numpy(y.astype(np.float32) / 255.0)[None]
```

Em vez de caçar um **quarto** limiar mágico, o alvo usa o valor real que sai do `cv2.INTER_AREA` (0–255, normalizado), **sem nenhuma binarização**. `dice_bce_loss` aceita alvo contínuo nativamente — é a definição matemática usual dos dois, sem mudança de código lá.

A vantagem sobre qualquer limiar fixo:
- uma caixa do downscale com pouca cobertura de curva vira alvo **baixo mas não-zero**, preservando o gradiente de BCE — resolve o sumiço do limiar 127 sem escolher limiar nenhum;
- uma caixa com cobertura parcial **não** empurra a rede a prever confiança alta ali — limita a inflação de área pela raiz, não por um corte arbitrário.

### O caminho de 3 canais

```python
if self.in_ch == 3:
    ent = np.ascontiguousarray(m["image"][..., :3])
    x, _ = letterbox(ent, self.size)
    xt = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1)
else:
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = (m["image"].astype(np.float32) @ w).round().astype(np.uint8)
    ...
```

O ramo de 1 canal ficou **idêntico ao anterior**, para que os checkpoints existentes não mudem de resultado.

---

## A métrica de validação

```python
def iou(logits, target, thr: float = 0.5) -> float:
    p = (torch.sigmoid(logits) >= thr).float()
    t = (target >= thr).float()          # binariza AQUI: o treino não vê isso
    inter = (p * t).sum(dim=(1, 2, 3))
    union = ((p + t) >= 1).float().sum(dim=(1, 2, 3))
    return float((inter / union.clamp(min=1.0)).mean())
```

O alvo do treino é contínuo, mas o IoU de **validação** binariza dentro da função — para continuar interpretável como métrica de progresso e comparável com as rodadas anteriores.

> [!caution] Limitação conhecida
> IoU numa curva fina é dominado por **espessura de traço**, não por posição — foi por isso que ele foi demitido como critério de aceitação (ver [[Critérios de qualidade]]). Aqui ele ainda governa a seleção de checkpoint e o scheduler. Uma época que engrossa o traço ganha do scheduler. É uma limitação do procedimento de treino, não uma escolha defendida.

---

## O laço

```python
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="max", factor=a.lr_factor, patience=a.lr_patience,
    threshold=a.lr_threshold, threshold_mode="abs")

for ep in range(a.epochs):
    model.train()
    for nb, (x, y) in enumerate(tr):
        loss = dice_bce_loss(model(x), y)
        loss.backward(); opt.step()
    model.eval()
    m = float(np.mean([iou(model(x), y) for x, y in va]))
    sched.step(m)
    print(f"epoca {ep:02d}  IoU_val={m:.4f}  lr={lr_depois:.2e}{marca}  ...")
    if m > melhor:
        melhor = m
        torch.save(model.state_dict(), a.out)
```

> [!note] Por que `ReduceLROnPlateau`
> Sem scheduler, o treino sobe rápido nas 3 primeiras épocas e depois **oscila em torno de um platô** (IoU_val 0,65–0,67 por 10+ épocas, sem tendência de melhora), porque o passo de otimização fica grande demais para refinar perto do mínimo.
>
> `ReduceLROnPlateau` reage à métrica que de fato importa, reduzindo o LR só quando ela para de melhorar — não segue um cronograma fixo que teria de ser adivinhado de antemão.

O otimizador é `AdamW`, e a semente é fixa (`torch.manual_seed(20260817)`) — sem RNG global.

---

## Determinismo

`torch.manual_seed` fixo + `DataLoader` com `shuffle=True` só no treino + o corpus sendo determinístico bit a bit ([[dataset.generator]]) significa que **duas rodadas com os mesmos argumentos são comparáveis**. Sem isso, uma diferença medida entre modelos poderia vir do dado e não da arquitetura — que é exatamente a confusão que o teste de saturação (`base=24` → `base=32`) precisava evitar.

---

Ver também: [[identify.extract]] · [[dataset.generator]] · [[Critérios de qualidade]]
