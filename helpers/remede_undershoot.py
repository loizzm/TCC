"""Remede `_UNDERSHOOT_MAX` contra o Estagio A COM a guarda de continuidade.

POR QUE REMEDIR. O limiar de 0,08 foi calibrado contra a mascara anterior, e a
propria docstring dele avisa: "qualquer mudanca no Estagio A exige remedir este
numero". A guarda de continuidade (`polyline.py`) e mudanca no Estagio A, e ela
mexe exatamente no DENOMINADOR desta metrica — `_undershoot` normaliza pela
FAIXA de y da serie extraida, e a guarda impede a polilinha de saltar para a
linha de entrada, o que ENCOLHE essa faixa. O mesmo undershoot absoluto vira
uma fracao maior e cruza o limiar.

Medido: num lote de 100 figuras de um degrau, `resposta_inversa` passou de 1
para 6 recusas depois da guarda, e virou o motivo dominante.

O QUE ESTE ARQUIVO ACRESCENTA. A docstring registra que "o gerador nao produz
fase nao-minima, entao o corpus da so o CUSTO; o beneficio segue apoiado em
n=1" — e a imagem daquele n=1 nao esta versionada. Aqui os dois lados sao
medidos:

  CUSTO     — figuras de fase MINIMA (as 100 do `lote100_1deg`, cuja verdade
              garante que nenhuma tem zero no semiplano direito). Toda deteccao
              ali e falso positivo.
  BENEFICIO — figuras de fase NAO-MINIMA geradas de proposito,
              `G(s) = K(1 - a s)/(tau s + 1)`, com o mesmo render aleatorio.
              Toda NAO-deteccao ali e falso negativo.

Uso:
    .venv/bin/python helpers/remede_undershoot.py [--n 40]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import signal

import rg_aleatorio as RG
from identify.calibrate import calibrate
from identify.extract import load_model, predict_mask
from identify.pipeline import _undershoot
from identify.polyline import mask_to_polyline, polyline_to_series

RAIZ = Path(__file__).resolve().parent.parent
MINIMA = RAIZ / "reports" / "amostras_aleatorias" / "lote100_1deg"
NAOMIN = RAIZ / "reports" / "amostras_aleatorias" / "fase_nao_minima"


def gera_nao_minima(n: int) -> list[dict]:
    """`G(s) = K (1 - a s)/(tau s + 1)`: a resposta DESCE antes de subir.

    `a` (o zero no semiplano direito) fica em fracoes de `tau`. Quanto maior
    `a/tau`, mais fundo o undershoot — a faixa sorteada cobre desde o caso
    sutil ate o obvio, para que o limiar seja escolhido contra o dificil e nao
    contra o facil.
    """
    NAOMIN.mkdir(parents=True, exist_ok=True)
    verdades = []
    for i in range(n):
        rng = np.random.default_rng([424242, i])
        tau = float(np.exp(rng.uniform(np.log(0.2), np.log(10.0))))
        K = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))
        a = tau * float(np.exp(rng.uniform(np.log(0.15), np.log(2.0))))
        theta = 0.0 if rng.random() < 0.3 else float(rng.uniform(0.05, 1.0) * tau)
        t_fim = theta + float(rng.uniform(3.0, 8.0) * tau)
        sinal_k = 1 if rng.random() < 0.5 else -1
        t = np.linspace(0.0, t_fim, int(rng.integers(800, 2001)))
        sys_ = signal.TransferFunction([-K * a, K], [tau, 1.0])
        y = np.zeros_like(t)
        m = t >= theta
        if m.any():
            _, ya = signal.step(sys_, T=t[m] - theta)
            y[m] = ya * sinal_k
        escuro = bool(rng.random() < 0.35)
        st = RG.sorteia_estilo(rng, escuro=escuro)
        # Sem a linha de entrada: isola o fenomeno de fase nao-minima da
        # interferencia do distrator, que ja tem medicao propria.
        RG.MODO_ENTRADA = "omite_fit"
        p = {"theta_sistema": theta, "ordem": "fopdt"}
        nome = f"nmp_{i:02d}.png"
        render = RG.figura_neg if escuro else RG.figura_rg
        render(NAOMIN / nome, p, [(1.0, theta)], t, y, np.zeros_like(t), st)
        verdades.append({"arquivo": nome, "K": K * sinal_k, "tau": tau,
                         "a": a, "a_sobre_tau": a / tau, "theta": theta,
                         "t_fim": t_fim})
    (NAOMIN / "verdade.json").write_text(json.dumps(verdades, indent=2))
    return verdades


def undershoots(dirs_e_pngs, modelo, dev) -> list[float]:
    out = []
    for p in dirs_e_pngs:
        img = np.asarray(Image.open(p).convert("RGB"))
        cal = calibrate(img)
        mk = predict_mask(modelo, img, dev)
        x, y = mask_to_polyline(mk, bbox=cal.bbox_px if any(cal.bbox_px) else None)
        if x.size < 10:
            continue
        if cal.ok:
            t, ys = polyline_to_series(x, y, cal)
            o = np.argsort(t)
            ys = ys[o]
        else:
            ys = y[np.argsort(x)]
        out.append(float(_undershoot(ys)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / "models" / "unet_stageA.pt"), dev)
    modelo.eval()

    if not (NAOMIN / "verdade.json").exists():
        print(f"  gerando {a.n} figuras de fase NAO-MINIMA...")
        gera_nao_minima(a.n)
    vnm = json.loads((NAOMIN / "verdade.json").read_text())

    print("  medindo undershoot nas de fase MINIMA (custo)...")
    u_min = undershoots(sorted(MINIMA.glob("*.png")), modelo, dev)
    print("  medindo undershoot nas de fase NAO-MINIMA (beneficio)...")
    u_nmp = undershoots(sorted(NAOMIN.glob("*.png")), modelo, dev)

    a_min, a_nmp = np.array(u_min), np.array(u_nmp)
    L = []
    def P(s=""):
        L.append(s)
    P("REMEDICAO DE _UNDERSHOOT_MAX — Estagio A COM guarda de continuidade")
    P("=" * 74)
    P(f"  fase MINIMA (nenhuma tem zero no SPD): n={a_min.size}")
    P(f"    p50={np.median(a_min):.4f}  p90={np.percentile(a_min,90):.4f}  "
      f"p99={np.percentile(a_min,99):.4f}  max={a_min.max():.4f}")
    P(f"  fase NAO-MINIMA (todas tem):          n={a_nmp.size}")
    P(f"    p10={np.percentile(a_nmp,10):.4f}  p50={np.median(a_nmp):.4f}  "
      f"min={a_nmp.min():.4f}")
    P()
    P(f"  {'limiar':>8}{'falsos pos.':>14}{'detectados':>13}{'F1':>8}")
    melhor = None
    for lim in (0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
        fp = int((a_min > lim).sum())
        tp = int((a_nmp > lim).sum())
        fn = a_nmp.size - tp
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        marca = "  <- producao" if abs(lim - 0.08) < 1e-9 else ""
        P(f"  {lim:>8.2f}{fp:>8}/{a_min.size:<5}{tp:>8}/{a_nmp.size:<5}{f1:>8.3f}{marca}")
        if melhor is None or f1 > melhor[0]:
            melhor = (f1, lim)
    P()
    P(f"  melhor F1: limiar={melhor[1]:.2f}  (F1={melhor[0]:.3f})")
    P()
    P("  ATENCAO ao que continua sem medicao: as figuras de fase nao-minima")
    P("  sao geradas por este arquivo, nao vem do corpus nem do mundo real. O")
    P("  beneficio deixa de estar apoiado em n=1, mas passa a estar apoiado num")
    P("  gerador — que reproduz a FISICA certa e um render plausivel, e nada")
    P("  garante que cobre a variedade real.")
    txt = "\n".join(L)
    (RAIZ / "reports" / "amostras_aleatorias" / "remedicao_undershoot.txt").write_text(
        txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
