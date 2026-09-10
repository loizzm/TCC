"""Diagnostico: QUAL atributo de render faz a fase nao-minima escapar da guarda.

No render de TREINO (`data/val_nmp`) a rede promovida recusa 96,7 %. No render
do `rg_aleatorio` recusa 50 %. A fisica e a mesma; a diferenca e o desenho.
Este arquivo mede QUAL parte do desenho, para o conserto de corpus ser dirigido
por evidencia em vez de por palpite.

Gera N figuras de fase nao-minima com o render aleatorio COMPLETO, guarda o
dicionario de estilo inteiro na verdade, roda a pipeline e cruza cada atributo
contra o desfecho. `ok` aqui e ERRO SILENCIOSO: a curva esta fora da familia e
o sistema devolveu parametro assim mesmo.

Uso:
    .venv/bin/python sonda_render_nmp.py [--n 200] [--modelo models/unet_stageA.pt]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image
from scipy import signal, stats

import rg_aleatorio as RG

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "reports" / "amostras_aleatorias" / "nmp_render"


def gera(n: int, seed0: int) -> list[dict]:
    """Mesma fisica de `remede_undershoot.gera_nao_minima`, mas guardando o
    ESTILO inteiro — sem ele nao ha o que cruzar."""
    SAIDA.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        rng = np.random.default_rng([seed0, i])
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
        RG.MODO_ENTRADA = "omite_fit"
        nome = f"nmp_{i:03d}.png"
        render = RG.figura_neg if escuro else RG.figura_rg
        render(SAIDA / nome, {"theta_sistema": theta, "ordem": "fopdt"},
               [(1.0, theta)], t, y, np.zeros_like(t), st)
        v = {"arquivo": nome, "K": K * sinal_k, "tau": tau, "a": a,
             "a_sobre_tau": a / tau, "theta": theta, "t_fim": t_fim,
             "escuro": escuro, "familia": "neg" if escuro else "rg",
             "janela_em_tau": (t_fim - theta) / tau,
             "n_amostras": int(t.size)}
        v.update({k: (list(x) if isinstance(x, tuple) else x)
                  for k, x in st.items()})
        out.append(v)
    (SAIDA / "verdade.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=909090)
    ap.add_argument("--modelo", default="models/unet_stageA.pt")
    ap.add_argument("--pular-geracao", action="store_true")
    a = ap.parse_args()

    v = (json.loads((SAIDA / "verdade.json").read_text())
         if a.pular_geracao else gera(a.n, a.seed))
    print(f"{len(v)} figuras de fase nao-minima, render aleatorio completo\n")

    from identify.extract import load_model
    from identify.pipeline import identify_from_image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = load_model(str(RAIZ / a.modelo), dev)
    mod.eval()

    for e in v:
        img = np.asarray(Image.open(SAIDA / e["arquivo"]).convert("RGB"))
        r = identify_from_image(img, mod, dev)
        e["reason"] = r["reason"] or ""
        e["silencioso"] = bool(r["ok"])      # ok = errado, por construcao
        e["n_points"] = int(r["n_points"])
    (SAIDA / "verdade.json").write_text(json.dumps(v, indent=2))

    c = Counter(("ok (ERRO SILENCIOSO)" if e["silencioso"] else (e["reason"] or "?"))
                for e in v)
    n = len(v)
    for k, x in c.most_common():
        print(f"  {k:<26}{x:>4}  ({x/n:.0%})")

    y = np.array([e["silencioso"] for e in v], float)
    print(f"\nCATEGORICOS — taxa de erro silencioso por nivel (Fisher vs resto)")
    for campo in ("familia", "linestyle", "preenchimento", "legenda_loc",
                  "grade_style", "cor"):
        niveis = sorted({str(e[campo]) for e in v})
        if len(niveis) > 8:
            continue
        print(f"  {campo}")
        for lv in niveis:
            m = np.array([str(e[campo]) == lv for e in v])
            if m.sum() < 5:
                continue
            tab = [[int((y[m] == 1).sum()), int((y[m] == 0).sum())],
                   [int((y[~m] == 1).sum()), int((y[~m] == 0).sum())]]
            p = stats.fisher_exact(tab)[1]
            marca = "  <<<" if p < 0.05 else ""
            print(f"    {lv:<16} n={int(m.sum()):<4} silencioso={y[m].mean():.0%}"
                  f"  p={p:.3f}{marca}")

    print(f"\nNUMERICOS — Spearman contra erro silencioso")
    for campo in ("a_sobre_tau", "linewidth", "dpi", "alpha_preench",
                  "margem_y", "janela_em_tau", "fontsize_titulo", "n_points"):
        x = np.array([float(e[campo]) for e in v])
        rho, p = stats.spearmanr(x, y)
        marca = "  <<<" if p < 0.05 else ""
        print(f"  {campo:<18} rho={rho:+.3f}  p={p:.4f}{marca}")


if __name__ == "__main__":
    main()
