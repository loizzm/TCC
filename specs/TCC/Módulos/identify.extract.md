---
tags: [modulo, estagio-a]
aliases: [extract.py, identify/extract.py, U-Net]
---

# `identify/extract.py`

> **139 linhas · Estágio A.** A U-Net compacta, o pré-processamento em letterbox, a perda composta e a inferência.

Único módulo do `identify/` que importa `torch`.

---

## O que faz

Transforma a figura inteira — moldura, grade, legenda, título, distratores — numa máscara binária que contém **apenas os pixels da curva**.

```
uint8[H,W,3]  ──letterbox──►  512×512  ──U-Net──►  logits  ──σ, limiar──►  ──unletterbox──►  uint8[H,W] 0/255
```

---

## A rede

```python
class UNet(nn.Module):
    """4 níveis, base 16 canais. Saída = logits, mesma resolução da entrada."""

    def __init__(self, base: int = 16, levels: int = 4, in_ch: int = 1):
        chs = [base * 2 ** i for i in range(levels + 1)]     # [32,64,128,256,512] em produção
        self.enc  = nn.ModuleList()        # 4 blocos, com pooling entre eles
        self.bott = _block(chs[-2], chs[-1])
        self.up   = nn.ModuleList()        # ConvTranspose2d, 4
        self.dec  = nn.ModuleList()        # 4 blocos, entrada = chs[i]*2 (skip concatenado)
        self.head = nn.Conv2d(chs[0], 1, 1)
```

O bloco básico são duas convoluções 3×3 **sem bias** — o BatchNorm seguinte já tem seu próprio deslocamento:

```python
def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )
```

O `forward` é o padrão U-Net: guarda cada saída de encoder, e no decoder concatena o skip com o upsample.

```python
def forward(self, x):
    skips = []
    for e in self.enc:
        x = e(x); skips.append(x); x = self.pool(x)
    x = self.bott(x)
    for up, dec, s in zip(self.up, self.dec, reversed(skips)):
        x = dec(torch.cat([up(x), s], dim=1))
    return self.head(x)
```

### Contagem de parâmetros

O modelo em produção é `UNet(base=32, levels=4, in_ch=3)`:

| grupo | parâmetros | fração |
|---|---:|---:|
| encoder | 1 173 216 | 15,1 % |
| **gargalo** | **3 540 992** | **45,6 %** |
| ConvTranspose2d | 696 800 | 9,0 % |
| decoder | 2 352 000 | 30,3 % |
| cabeça 1×1 | 33 | ~0 % |
| **total** | **7 763 041** | |

Contados com `sum(p.numel() for p in model.parameters())` — que **exclui** os buffers de BatchNorm. O arquivo do checkpoint carrega 7 768 947 escalares; a diferença de 5 906 são `running_mean` + `running_var` (2×2944) mais 18 `num_batches_tracked`.

Escalonamento é quadrático em `base`: `base=16, in_ch=1` → 1 942 289; `base=32, in_ch=3` → 7 763 041 (razão 3,996).

---

## Letterbox

```python
def letterbox(img: np.ndarray, size: int = 512) -> tuple[np.ndarray, LetterboxInfo]:
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size) if img.ndim == 2 else (size, size, img.shape[2]), dtype=img.dtype)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, LetterboxInfo(px, py, nw, nh, w, h, size)
```

**Recebe** imagem 2D (cinza) ou 3D (H,W,C) · **Devolve** o quadrado 512×512 e um `LetterboxInfo` frozen com o preenchimento, para desfazer depois.

Escala **isotrópica** mais preenchimento, nunca esticamento. Distorção anisotrópica mudaria a inclinação local da curva, e a inclinação é o que o estágio D lê como constante de tempo.

`unletterbox` desfaz com `INTER_NEAREST` — a máscara é categórica, interpolar valores intermediários criaria pixels que não são nem curva nem fundo.

---

## A perda

```python
def dice_bce_loss(logits, target, eps: float = 1.0):
    """BCE pura colapsa: a classe positiva ocupa < 2% dos pixels (PLANO)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1.0 - num / den).mean()
```

Com ~1 % de pixels positivos, o mínimo trivial "tudo fundo" já acerta 98 % — BCE sozinha converge para ele. O termo Dice normaliza pela área da classe positiva. Ambos aceitam alvo **contínuo** nativamente, o que é aproveitado no treino (ver [[train_unet]]).

---

## Carregamento e inferência

```python
def load_model(path, device: str = "cpu") -> UNet:
    state = torch.load(path, map_location=device)
    base   = int(state["enc.0.0.weight"].shape[0])
    levels = sum(1 for k in state if k.startswith("enc.") and k.endswith(".0.weight"))
    in_ch  = int(state["enc.0.0.weight"].shape[1])
    model = UNet(base=base, levels=levels, in_ch=in_ch)
    model.load_state_dict(state); model.to(device).eval()
    return model
```

A arquitetura é **inferida do próprio checkpoint**. Um `UNet()` fixo aqui recusaria carregar as rodadas de capacidade (`base=24`, `base=32`), e um modelo de 1 canal convive com um de 3 sem o chamador saber.

```python
@torch.no_grad()
def predict_mask(model, image_rgb, device="cpu", thr=0.5) -> np.ndarray:
    if getattr(model, "in_ch", 1) == 3:
        small, info = letterbox(np.ascontiguousarray(image_rgb[..., :3]))
        x = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    else:
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = (image_rgb.astype(np.float32) @ w).round().astype(np.uint8)
        small, info = letterbox(gray)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(device)
    p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return unletterbox(np.where(p >= thr, 255, 0).astype(np.uint8), info)
```

**Recebe** `uint8[H,W,3]` · **Devolve** `uint8[H,W]` com valores 0 ou 255, **mesma resolução da entrada**.

O caminho de 1 canal ficou byte a byte idêntico ao anterior, de propósito: os checkpoints antigos não mudam de resultado. O caminho de 3 canais existe porque a projeção em luminância é destrutiva — ver [[Decisões de projeto#D2 · RGB em vez de luminância]].

---

Ver também: [[identify.extract_classical]] (a alternativa sem rede) · [[train_unet]] · [[Critérios de qualidade]]
