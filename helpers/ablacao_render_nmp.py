"""ABLACAO: qual COMPONENTE do render do `rg_aleatorio` faz a fase nao-minima
escapar da guarda.

A sondagem por atributo (`helpers/sonda_render_nmp.py`) deu NULO: nenhum estilo
sorteado dentro do lote prevê o erro silencioso. Logo a diferenca esta no que e
CONSTANTE no `rg_aleatorio` e ausente do treino. Duas candidatas medidas:

  FOLHA DE ESTILO — `rg_aleatorio` faz `plt.style.use("seaborn-v0_8-darkgrid")`
  ou `"dark_background"`. O gerador de treino NUNCA usa folha de estilo: mexe
  em rcParams soltos. Folha muda dezenas de chaves correlacionadas de uma vez
  (fundo dos eixos, grade, ciclo de cores, ticks, moldura, fonte) — e' uma
  regiao inteira do espaco de render que a rede nunca visitou.

  DENSIDADE DA POLILINHA — treino tem 512 pontos FIXOS; teste, 803 a 1996. Em
  figura de 600 a 1400 px isso e' a diferenca entre traco visivelmente
  poligonal nas curvas fechadas e traco liso. A regiao de maior curvatura de
  uma resposta de fase nao-minima e' exatamente o mergulho.

Mesma fisica e mesma seed nas quatro variantes: so o componente muda.

Uso:
    .venv/bin/python helpers/ablacao_render_nmp.py [--n 150]
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
from scipy import signal, stats

import rg_aleatorio as RG

RAIZ = Path(__file__).resolve().parent.parent
BASE = RAIZ / "reports" / "amostras_aleatorias" / "ablacao_nmp"

VARIANTES = {
    "A_como_e":        dict(folha=True,  n_pts=None),
    "B_sem_folha":     dict(folha=False, n_pts=None),
    "C_512_pontos":    dict(folha=True,  n_pts=512),
    "D_ambos":         dict(folha=False, n_pts=512),
}
_ESTILO_ORIG = RG._estilo


def _sem_folha(nome: str) -> None:
    """Substitui `plt.style.use(nome)` por so o reset — mata a folha de estilo
    preservando tudo o mais que o `rg_aleatorio` faz."""
    plt.rcdefaults()


def gera(variante: str, cfg: dict, n: int, seed0: int) -> None:
    out = BASE / variante
    out.mkdir(parents=True, exist_ok=True)
    RG._estilo = _ESTILO_ORIG if cfg["folha"] else _sem_folha
    try:
        for i in range(n):
            rng = np.random.default_rng([seed0, i])   # MESMA fisica em toda variante
            tau = float(np.exp(rng.uniform(np.log(0.2), np.log(10.0))))
            K = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))
            a = tau * float(np.exp(rng.uniform(np.log(0.15), np.log(2.0))))
            theta = 0.0 if rng.random() < 0.3 else float(rng.uniform(0.05, 1.0) * tau)
            t_fim = theta + float(rng.uniform(3.0, 8.0) * tau)
            sinal_k = 1 if rng.random() < 0.5 else -1
            n_nat = int(rng.integers(800, 2001))
            t = np.linspace(0.0, t_fim, cfg["n_pts"] or n_nat)
            sys_ = signal.TransferFunction([-K * a, K], [tau, 1.0])
            y = np.zeros_like(t)
            m = t >= theta
            if m.any():
                _, ya = signal.step(sys_, T=t[m] - theta)
                y[m] = ya * sinal_k
            escuro = bool(rng.random() < 0.35)
            st = RG.sorteia_estilo(rng, escuro=escuro)
            RG.MODO_ENTRADA = "omite_fit"
            render = RG.figura_neg if escuro else RG.figura_rg
            render(out / f"nmp_{i:03d}.png",
                   {"theta_sistema": theta, "ordem": "fopdt"},
                   [(1.0, theta)], t, y, np.zeros_like(t), st)
    finally:
        RG._estilo = _ESTILO_ORIG


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=717171)
    ap.add_argument("--modelo", default="models/unet_stageA.pt")
    a = ap.parse_args()

    from identify.extract import load_model
    from identify.pipeline import identify_from_image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = load_model(str(RAIZ / a.modelo), dev)
    mod.eval()

    res = {}
    for nome, cfg in VARIANTES.items():
        gera(nome, cfg, a.n, a.seed)
        sil = []
        for i in range(a.n):
            img = np.asarray(Image.open(BASE / nome / f"nmp_{i:03d}.png").convert("RGB"))
            r = identify_from_image(img, mod, dev)
            sil.append(bool(r["ok"]))
        res[nome] = np.array(sil)
        print(f"  {nome:<16} erro silencioso {res[nome].mean():>6.1%}  "
              f"({int(res[nome].sum())}/{a.n})", flush=True)

    print("\n  contrastes (McNemar pareado, mesma fisica):")
    base = res["A_como_e"]
    for nome in ("B_sem_folha", "C_512_pontos", "D_ambos"):
        v = res[nome]
        b = int((base & ~v).sum())      # era silencioso, deixou de ser
        c = int((~base & v).sum())      # virou silencioso
        p = stats.binomtest(c, b + c, 0.5).pvalue if b + c else 1.0
        print(f"    A -> {nome:<14} melhora={b} piora={c}  "
              f"delta={v.mean()-base.mean():+.1%}  p={p:.4f}")
    (BASE / "resultado.json").write_text(json.dumps(
        {k: v.tolist() for k, v in res.items()}))


if __name__ == "__main__":
    main()
