#!/usr/bin/env python3
"""Escolhe o checkpoint do retreino pela COBERTURA DO PLATO, nao pelo IoU_val.

Por que este script existe (HANDOFF_P2_7 secao 40.9). O `train_unet.py` guarda
o melhor checkpoint segundo o `IoU_val`, e essa metrica e quase cega ao defeito
que o retreino existe para consertar: o estrato de plato no topo custa 5 pontos
de IoU (0,7814 -> 0,7304) enquanto a cobertura do plato nele desaba 42 pontos
(0,944 -> 0,527). O plato e uma linha FINA, poucos pixels contra o corpo da
curva. Selecionar por IoU_val e selecionar pelo instrumento errado.

Uso:
    .venv/bin/python seleciona_checkpoint.py CKPT_DIR [--n 120] [--base 32]

Mede, para cada `epoca_NN.pt` do diretorio, tres numeros sobre amostras de
plato no TOPO e de plato no RODAPE, e reporta os dois lado a lado. O
checkpoint bom e o que sobe o TOPO sem derrubar o RODAPE — subir um as custas
do outro nao e aprender posicao, e trocar de vies.

As amostras de topo saem de `ganho_negativo=True` (estrato treinado) E de
positivo com eixo y INVERTIDO (condicao NUNCA treinada). A segunda e a
validacao independente: se o retreino aprendeu POSICAO em vez de decorar o
estrato, as duas sobem juntas. Se so a treinada subir, decorou.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dataset.generator as G
from dataset.generator import generate_sample
from identify.extract import UNet, predict_mask

_orig_limits = G._axis_limits


def _limites_invertidos(t, y, style):
    """Inverte o eixo y. Alimenta a figura da imagem E a da mascara
    (`generator.py` linhas 381 e 569), entao a verdade fica consistente."""
    xlim, ylim = _orig_limits(t, y, style)
    return xlim, (ylim[1], ylim[0])


def _metricas(d: Path, model, dev: str) -> tuple[float, float, float]:
    """(IoU, cobertura de colunas, cobertura do PLATO) de uma amostra."""
    meta = json.loads((d / "meta.json").read_text())
    img = np.asarray(Image.open(d / "image.png").convert("RGB"), dtype=np.uint8)
    verd = np.asarray(Image.open(d / "mask.png").convert("L"), dtype=np.uint8) > 127
    pred = predict_mask(model, img, dev) > 127

    uniao = int((pred | verd).sum())
    iou = int((pred & verd).sum()) / uniao if uniao else np.nan
    cv, cp = verd.any(axis=0), pred.any(axis=0)
    cob = float((cv & cp).sum() / cv.sum()) if cv.sum() else np.nan
    aff, theta = meta["axis_affine"], meta["params"]["theta"]
    x_dados = aff["sx"] * np.arange(verd.shape[1]) + aff["ox"]
    plato = (x_dados < theta) & cv
    cob_p = float((plato & cp).sum() / plato.sum()) if plato.sum() >= 5 else np.nan
    return iou, cob, cob_p


def _celula(model, dev, seeds, neg: bool, invertido: bool, tmp: str):
    G._axis_limits = _limites_invertidos if invertido else _orig_limits
    try:
        vals = []
        for seed in seeds:
            d = Path(tmp) / f"s{seed}_{int(neg)}_{int(invertido)}"
            if not d.exists():
                generate_sample(str(d), seed=seed, ganho_negativo=neg, add_noise=True)
            vals.append(_metricas(d, model, dev))
    finally:
        G._axis_limits = _orig_limits
    arr = np.array(vals, dtype=float)
    return [float(np.nanmean(arr[:, j])) for j in range(3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", help="diretorio com epoca_NN.pt")
    ap.add_argument("--n", type=int, default=120, help="amostras por celula")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=6000,
                    help="base dos seeds de AVALIACAO. Tem de ser disjunta do "
                         "treino: 90004 (train_kneg) e 90005 (val_kneg).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tambem", action="append", default=None,
                    help="checkpoint avulso para comparar (repetivel), ex. o promovido")
    a = ap.parse_args()

    seeds = list(range(a.seed0, a.seed0 + a.n))
    alvos = sorted(Path(a.ckpt_dir).glob("epoca_*.pt")) if a.ckpt_dir != "-" else []
    alvos += [Path(x) for x in (a.tambem or [])]
    if not alvos:
        raise SystemExit("nenhum checkpoint encontrado")

    print(f"device={a.device}  n={a.n} por celula  seeds {seeds[0]}..{seeds[-1]}")
    print()
    cab = (f"{'checkpoint':<22} | {'TOPO (K<0, treinado)':^26} | "
           f"{'TOPO (K>0 eixo inv., NAO treinado)':^36} | {'RODAPE (K>0, controle)':^26}")
    print(cab)
    print(f"{'':<22} | {'plato':>8}{'colunas':>9}{'IoU':>9} | "
          f"{'plato':>12}{'colunas':>12}{'IoU':>12} | {'plato':>8}{'colunas':>9}{'IoU':>9}")
    print("-" * len(cab))

    with tempfile.TemporaryDirectory() as tmp:
        for ck in alvos:
            model = UNet(base=a.base, in_ch=a.in_ch).to(a.device)
            model.load_state_dict(torch.load(ck, map_location=a.device))
            model.eval()
            topo_tr = _celula(model, a.device, seeds, neg=True, invertido=False, tmp=tmp)
            topo_ood = _celula(model, a.device, seeds, neg=False, invertido=True, tmp=tmp)
            rodape = _celula(model, a.device, seeds, neg=False, invertido=False, tmp=tmp)
            print(f"{ck.name:<22} | {topo_tr[2]:>8.4f}{topo_tr[1]:>9.4f}{topo_tr[0]:>9.4f} | "
                  f"{topo_ood[2]:>12.4f}{topo_ood[1]:>12.4f}{topo_ood[0]:>12.4f} | "
                  f"{rodape[2]:>8.4f}{rodape[1]:>9.4f}{rodape[0]:>9.4f}", flush=True)

    print()
    print("COMO LER: o checkpoint bom sobe o TOPO sem derrubar o RODAPE.")
    print("  Baseline do promovido, medido no Bloco 9: topo 0,527 / rodape 0,944.")
    print("  Se so a coluna TREINADA subir e a NAO TREINADA ficar, a rede decorou")
    print("  o estrato em vez de aprender posicao — e o defeito volta em qualquer")
    print("  figura de eixo invertido, que o corpus nao contem.")


if __name__ == "__main__":
    main()
