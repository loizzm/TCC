"""Estágio A — segmentação da curva. U-Net compacta (PLANO §PARTE 2).

Preenchimento preserva a razão de aspecto: distorção anisotrópica alteraria a
geometria da curva e envenenaria o critério 2.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LetterboxInfo:
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    src_w: int
    src_h: int
    size: int


def letterbox(img: np.ndarray, size: int = 512) -> tuple[np.ndarray, LetterboxInfo]:
    """Aceita 2D (cinza) e 3D (H, W, C). O caminho 2D e byte a byte o de antes."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    forma = (size, size) if img.ndim == 2 else (size, size, img.shape[2])
    out = np.zeros(forma, dtype=img.dtype)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, LetterboxInfo(px, py, nw, nh, w, h, size)


def unletterbox(mask512: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    crop = mask512[info.pad_y:info.pad_y + info.new_h,
                   info.pad_x:info.pad_x + info.new_w]
    return cv2.resize(crop, (info.src_w, info.src_h),
                      interpolation=cv2.INTER_NEAREST)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4 níveis, base 16 canais. Saída = logits, mesma resolução da entrada."""

    def __init__(self, base: int = 16, levels: int = 4, in_ch: int = 1):
        super().__init__()
        self.in_ch = int(in_ch)
        chs = [base * 2 ** i for i in range(levels + 1)]
        self.enc = nn.ModuleList()
        cin = self.in_ch
        for c in chs[:-1]:
            self.enc.append(_block(cin, c))
            cin = c
        self.bott = _block(chs[-2], chs[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(levels - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(_block(chs[i] * 2, chs[i]))
        self.head = nn.Conv2d(chs[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bott(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], dim=1))
        return self.head(x)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor,
                  eps: float = 1.0) -> torch.Tensor:
    """BCE pura colapsa: a classe positiva ocupa < 2% dos pixels (PLANO)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1.0 - num / den).mean()


def load_model(path: str | Path, device: str = "cpu") -> UNet:
    # `base`/`levels` saem do proprio checkpoint: as rodadas de capacidade
    # (HANDOFF_P2_3 Ruling 10a) salvam UNet(base=24)/UNet(base=32) com a mesma
    # arquitetura, e um `UNet()` fixo aqui recusaria carrega-los.
    state = torch.load(path, map_location=device)
    base = int(state["enc.0.0.weight"].shape[0])
    levels = sum(1 for k in state if k.startswith("enc.") and k.endswith(".0.weight"))
    # `in_ch` tambem sai do checkpoint: um modelo de 1 canal (cinza) e um de 3
    # (RGB) convivem, e trocar de um para o outro nao exige mexer no chamador.
    in_ch = int(state["enc.0.0.weight"].shape[1])
    model = UNet(base=base, levels=levels, in_ch=in_ch)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_mask(model: UNet, image_rgb: np.ndarray,
                 device: str = "cpu", thr: float = 0.5) -> np.ndarray:
    # A conversao para cinza e DESTRUTIVA: `0.299R + 0.587G + 0.114B` projeta
    # R^3 em R^1, e dois objetos de luminancia igual chegam a rede como o MESMO
    # byte. Medido nas duas imagens reais do Ruling 55: curva (44,160,44) e reta
    # de referencia (230,61,61) viram ambas 112 — separar uma da outra deixa de
    # ser dificil e passa a ser impossivel. Um modelo de 3 canais recebe RGB e
    # nao perde essa informacao. O caminho de 1 canal fica byte a byte igual ao
    # anterior, para que os checkpoints existentes nao mudem de resultado.
    if getattr(model, "in_ch", 1) == 3:
        entrada = np.ascontiguousarray(image_rgb[..., :3])
        small, info = letterbox(entrada)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(device)
    else:
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = (image_rgb.astype(np.float32) @ w).round().astype(np.uint8)
        small, info = letterbox(gray)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(device)
    p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    m512 = np.where(p >= thr, 255, 0).astype(np.uint8)
    return unletterbox(m512, info)
