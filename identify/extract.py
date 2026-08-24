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


def letterbox(gray: np.ndarray, size: int = 512) -> tuple[np.ndarray, LetterboxInfo]:
    h, w = gray.shape
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), dtype=gray.dtype)
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

    def __init__(self, base: int = 16, levels: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(levels + 1)]
        self.enc = nn.ModuleList()
        cin = 1
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
    model = UNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_mask(model: UNet, image_rgb: np.ndarray,
                 device: str = "cpu", thr: float = 0.5) -> np.ndarray:
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = (image_rgb.astype(np.float32) @ w).round().astype(np.uint8)
    small, info = letterbox(gray)
    x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(device)
    p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    m512 = np.where(p >= thr, 255, 0).astype(np.uint8)
    return unletterbox(m512, info)
