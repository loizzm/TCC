"""INVARIANCIA A OCLUSAO POR LEGENDA — o instrumento que faltava (§74).

O PROBLEMA QUE ELE RESOLVE. O dano de oclusao so tinha UM caso real
(`caso_real_neg_super` contra `_legenda_movida`: `wn`/`zeta` a 32 % com a caixa
sobre a curva e a 2,6 % com ela movida). Tres metricas sinteticas foram
tentadas — recall sob a caixa, cauda do erro pareado, patamar falso na mascara
— e as TRES ordenaram os checkpoints ao CONTRARIO do par real. Sem instrumento,
treinar para consertar aquele portao e depois verificar nele e' ajustar ao
teste.

A IDEIA. Nao medir "quao bem vai com a caixa", e sim a SOBREVIVENCIA: das
figuras que a rede acerta SEM a caixa, quantas ela continua acertando COM ela.
O lado "sem" e' conhecidamente bom, entao qualquer perda isola a oclusao e nada
mais — a mesma logica do par real, com centenas de pares em vez de um.

O PAR. `data/*_parleg_canto` e `data/*_parleg_ocl` tem as MESMAS sementes: mesma
planta, mesma serie, mesma legenda, mesmo rotulo — so a POSICAO da caixa muda.
A mascara e' identica nos dois (o gerador a desenha numa figura sem legenda),
entao o alvo nao se mexe.

RESSALVA MEDIDA: em ~17 % dos pares a caixa oclusiva e' ancorada perto da borda
e sai parcialmente do quadro, ficando menor que a do canto. O vies e'
conservador (subestima a oclusao), mas o par nao e' exato ali.

Uso:
    .venv/bin/python mede_invariancia.py [--modelo models/unet_stageA.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from identify.extract import load_model, predict_mask
from identify.pipeline import identify_from_image

RAIZ = Path(__file__).resolve().parent
NIVEIS = {"ESTRITO": dict(K=0.05, theta_T=0.02, estrutura=True),
          "PRATICO": dict(K=0.10, theta_T=0.05, estrutura=False)}


def _t_dom(order, tau, wn, zeta):
    if order == "fopdt":
        return float(tau)
    wn, zeta = float(wn), float(zeta)
    return 1.0 / (zeta * wn) if zeta < 1 else 1.0 / float(wn * (zeta - np.sqrt(zeta * zeta - 1)))


def _passa(r, meta, nivel) -> bool:
    if not r["ok"]:
        return False
    c = NIVEIS[nivel]
    p, v = r["params"], meta["params"]
    K = float(v["K"]) * float(meta["step_amplitude"])
    T = float(meta["t_window"][1]) - float(meta["t_window"][0])
    if c["estrutura"] and r["order"] != meta["order"]:
        return False
    if not c["estrutura"]:
        td = _t_dom(meta["order"], v.get("tau"), v.get("wn"), v.get("zeta"))
        tdh = _t_dom(r["order"], p.get("tau"), p.get("wn"), p.get("zeta"))
        if td > 0 and abs(tdh - td) / td > 0.15:
            return False
    if abs(K) > 1e-9 and abs(float(p["K"]) - K) / abs(K) > c["K"]:
        return False
    return abs(float(p["theta"]) - float(v["theta"])) / T <= c["theta_T"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="models/unet_stageA.pt")
    ap.add_argument("--par", default="val_parleg")
    ap.add_argument("--lim", type=int, default=300)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / a.modelo), dev)
    modelo.eval()

    A = RAIZ / "data" / f"{a.par}_canto"
    B = RAIZ / "data" / f"{a.par}_ocl"
    res = {n: {"n": 0, "sobrevive": 0} for n in NIVEIS}
    cob = []
    for s in sorted(A.glob("sample_*"))[:a.lim]:
        t = B / s.name
        if not t.exists():
            continue
        meta = json.loads((s / "meta.json").read_text())
        alvo = np.asarray(Image.open(s / "mask.png").convert("L")) > 127
        ia = np.asarray(Image.open(s / "image.png").convert("RGB"))
        ib = np.asarray(Image.open(t / "image.png").convert("RGB"))
        ra, rb = identify_from_image(ia, modelo, dev), identify_from_image(ib, modelo, dev)
        for n in NIVEIS:
            if _passa(ra, meta, n):
                res[n]["n"] += 1
                res[n]["sobrevive"] += int(_passa(rb, meta, n))
        if alvo.any():
            ma = float((predict_mask(modelo, ia, dev) > 127)[alvo].mean())
            mb = float((predict_mask(modelo, ib, dev) > 127)[alvo].mean())
            cob.append((ma, mb))

    print(f"INVARIANCIA A OCLUSAO — {Path(a.modelo).name}   par `{a.par}`\n")
    print(f"  {'nivel':<10}{'acerta SEM a caixa':>22}{'sobrevive COM':>16}{'taxa':>9}")
    for n in NIVEIS:
        d = res[n]
        if not d["n"]:
            continue
        print(f"  {n:<10}{d['n']:>22}{d['sobrevive']:>16}"
              f"{d['sobrevive']/d['n']:>9.0%}")
    if cob:
        c = np.array(cob)
        print(f"\n  RECALL da mascara sobre a curva verdadeira (n={len(c)})")
        print(f"    canto p50    = {np.median(c[:,0]):.3f}")
        print(f"    oclusora p50 = {np.median(c[:,1]):.3f}"
              f"   (queda {np.median(c[:,0])-np.median(c[:,1]):+.3f})")
    print("\n  100 % = a caixa nao custa nada. O ALVO do retreino e' subir a taxa")
    print("  SEM derrubar a coluna 'acerta SEM a caixa' — subir uma as custas da")
    print("  outra e' trocar de vies, nao aprender invariancia.")


if __name__ == "__main__":
    main()
