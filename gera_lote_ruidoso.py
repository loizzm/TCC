#!/usr/bin/env python3
"""Lote de avaliacao com RUIDO DE MEDICAO, estratificado por ganho e por SNR.

POR QUE ELE EXISTE. `rg_aleatorio.py` nao tem ruido — as curvas dele sao a
resposta analitica limpa. O corpus de TREINO tem (`_apply_noise`, SNR em dB
sobre a variancia), mas o de AVALIACAO nunca teve, entao toda assertividade
medida ate aqui vale para figura limpa. Dado real de processo nao e limpo: a
referencia que motivou este arquivo e uma tela de trending com a PV visivelmente
ruidosa em volta do modelo FOPDT ajustado.

O DESENHO e fatorial 2 x 5: dois niveis de ganho (|K| < 1 e |K| >= 1) por cinco
niveis de SNR (30, 25, 20, 15, 10 dB), 10 figuras por celula = 100. Sortear o
SNR continuamente dava ~4 figuras por faixa de 1 dB e nenhuma leitura por
nivel; com niveis discretos cada celula tem n=10 e da para ver ONDE quebra, que
e a pergunta util. O ruido entra na SERIE, antes do render — e ruido de
medicao, nao artefato de imagem.

A VERDADE guardada e a da planta LIMPA: o que se pede do sistema e recuperar o
processo, nao a realizacao do ruido.

Uso:
    .venv/bin/python gera_lote_ruidoso.py [--por-celula 10] [--seed 20260911]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

import rg_aleatorio as RG

RAIZ = Path(__file__).resolve().parent
SNRS = (30.0, 25.0, 20.0, 15.0, 10.0)


def _ruido(y: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """Gaussiano aditivo, SNR em dB sobre a VARIANCIA do sinal — mesma
    definicao de `dataset/generator._apply_noise`, para que os numeros deste
    lote sejam comparaveis com os do corpus de treino."""
    y = np.asarray(y, dtype=float)
    var = float(np.var(y))
    if not (var > 0.0 and np.isfinite(var)):
        return y.copy()
    sigma = float(np.sqrt(var / (10.0 ** (snr_db / 10.0))))
    return y + sigma * rng.standard_normal(y.size)


def gera(por_celula: int, seed0: int, out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    verdades, i = [], 0
    for faixa, (lo, hi) in (("Kmenor1", (0.0, 1.0)), ("Kmaior1", (1.0, np.inf))):
        for snr in SNRS:
            feitas = 0
            tent = 0
            while feitas < por_celula:
                rng = np.random.default_rng([seed0, i, tent])
                tent += 1
                # rejeicao na FAIXA DE GANHO: o gerador nao sabe sortear |K|
                # numa faixa, e mexer nele criaria caminho de codigo novo no
                # gerador de avaliacao. Rejeitar preserva a distribuicao dele.
                p = RG.sorteia_planta(rng)
                sinal = 1 if rng.random() < 0.5 else -1
                degraus, t_fim = RG.sorteia_degraus(rng, p, 1, sinal)
                K = p["K_planta"] * degraus[0][0]
                if not (lo <= abs(K) < hi):
                    continue
                sistema = RG.planta(p["ordem"], p["K_planta"], p["tau"],
                                    p["wn"], p["zeta"])
                t = np.linspace(0.0, t_fim, int(rng.integers(800, 2001)))
                y, u = RG.resposta_multi_degrau(sistema, p["theta_sistema"],
                                                degraus, t)
                y_ruid = _ruido(y, snr, rng)
                escuro = bool(rng.random() < 0.3)
                st = RG.sorteia_estilo(rng, escuro)
                RG.MODO_ENTRADA = "desenha"
                nome = f"ruido_{faixa}_{int(snr):02d}dB_{feitas:02d}.png"
                render = RG.figura_neg if escuro else RG.figura_rg
                render(out / nome, p, degraus, t, y_ruid, u, st)
                reg = RG._registro(nome, "neg" if escuro else "rg", p, degraus,
                                   t_fim, st, escuro, celula=f"{faixa}_{int(snr)}dB")
                reg.update(snr_db=float(snr), faixa_ganho=faixa,
                           sigma_rel=float(np.std(y_ruid - y) / max(np.ptp(y), 1e-12)))
                verdades.append(reg)
                feitas += 1
                i += 1
    (out / "verdade.json").write_text(json.dumps(verdades, indent=2))
    return verdades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-celula", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--out", type=Path,
                    default=RAIZ / "reports" / "amostras_aleatorias" / "lote_ruido")
    a = ap.parse_args()
    v = gera(a.por_celula, a.seed, a.out)
    ak = np.abs([e["K"] for e in v])
    print(f"{len(v)} figuras em {a.out}")
    print(f"  |K| < 1: {int((ak < 1).sum())}   |K| >= 1: {int((ak >= 1).sum())}")
    for s in SNRS:
        g = [e for e in v if e["snr_db"] == s]
        sr = np.array([e["sigma_rel"] for e in g])
        print(f"  SNR {int(s):>2} dB: n={len(g):<4} desvio do ruido / faixa de y: "
              f"mediana {np.median(sr):.3f}")


if __name__ == "__main__":
    main()
