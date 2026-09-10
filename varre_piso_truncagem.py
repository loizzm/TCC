"""Extrai a serie de cada figura UMA vez e varre `_PISO_SUSPEITA` offline.

Por que existe. Medido nos tres lotes, **80,7 % das figuras multi-degrau que a
pipeline nao truncou foram barradas pelo gate 1** — `_PISO_SUSPEITA = 0.030`
(`identify/classical.py:1036`), o piso de residuo abaixo do qual a varredura de
cortes nem roda. So 19,3 % chegaram ao gate 2 (`_GANHO_MIN = 0.60`) e foram
reprovadas la. Saber isso muda a correcao: nao adianta mexer no `_GANHO_MIN`.

O que este arquivo NAO faz: propor um valor. O piso e um gatilho de CUSTO e a
docstring dele registra que as duas populacoes (degrau unico e multi-degrau) se
SOBREPOEM em ganho — nao existe classificador. O que a varredura mede e o
preco: quantas multi-degrau a mais entram, e quantas de um degrau so passam a
ser truncadas por engano, a cada piso.

A serie extraida e cacheada em `series.npz`; a varredura em si nao chama a rede.

CUIDADO com a chave: `balanceado/` e `balanceado2/` usam os MESMOS nomes de
arquivo (`bal_K+_1deg_00.png` existe nos dois). Indexar por `png.name` faz o
segundo lote sobrescrever o primeiro em silencio — 100 amostras somem e a
verdade fica trocada. A chave aqui e `"<lote>/<arquivo>"`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent / "reports" / "amostras_aleatorias"
LOTES = (RAIZ, RAIZ / "balanceado", RAIZ / "balanceado2")
CACHE = RAIZ / "series.npz"


def extrai_series() -> dict[str, np.ndarray]:
    """Repete os passos de `identify_from_image` ate a serie calibrada."""
    import torch
    from PIL import Image
    from identify.calibrate import calibrate
    from identify.extract import load_model, predict_mask
    from identify.polyline import mask_to_polyline, polyline_to_series

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(Path(__file__).resolve().parent / "models" /
                            "unet_stageA.pt"), dev)
    out = {}
    for d in LOTES:
        for png in sorted(d.glob("*.png")):
            img = np.asarray(Image.open(png).convert("RGB"))
            cal = calibrate(img)
            if not cal.ok:
                continue
            mask = predict_mask(modelo, img, dev)
            xp, yp = mask_to_polyline(mask, bbox=cal.bbox_px if any(cal.bbox_px) else None)
            if xp.size < 10:
                continue
            t, y = polyline_to_series(xp, yp, cal)
            o = np.argsort(t)
            out[f"{d.name}/{png.name}"] = np.stack([t[o], y[o]])
            print(f"  serie: {png.name} ({t.size} pts)")
    return out


def main() -> None:
    from identify.classical import (_FRACS_CORTE, _GANHO_MIN, _N_MIN_PREFIXO,
                                    _PISO_SUSPEITA, identify)

    if CACHE.exists():
        series = {k: v for k, v in np.load(CACHE).items()}
        print(f"series lidas do cache: {len(series)}")
    else:
        series = extrai_series()
        np.savez_compressed(CACHE, **series)
        print(f"series gravadas em {CACHE}: {len(series)}")

    verdades = {}
    for d in LOTES:
        for v in json.loads((d / "verdade.json").read_text()):
            verdades[f"{d.name}/{v['arquivo']}"] = v

    # Um unico calculo por amostra: nrmse do ajuste inteiro e o MELHOR ganho
    # entre os cortes candidatos. Com esses dois numeros qualquer par
    # (piso, ganho_min) se avalia sem refazer ajuste nenhum.
    linhas = []
    for nome, ty in series.items():
        t, y = ty[0], ty[1]
        full = identify(t, y)
        if not (full.success and np.isfinite(full.nrmse)):
            continue
        melhor_ganho, corte = -np.inf, None
        for fr in _FRACS_CORTE:
            k = int(fr * t.size)
            if k < _N_MIN_PREFIXO:
                continue
            r = identify(t[:k], y[:k])
            if r.success and np.isfinite(r.nrmse):
                g = 1.0 - r.nrmse / full.nrmse
                if g > melhor_ganho:
                    melhor_ganho, corte = g, k
        linhas.append({"arquivo": nome, "nrmse_full": float(full.nrmse),
                       "melhor_ganho": float(melhor_ganho),
                       "n_degraus": verdades[nome]["n_degraus"]})
        print(f"  {nome:<24} nrmse_full={full.nrmse:.4f} melhor_ganho={melhor_ganho:.4f}")

    (RAIZ / "varredura_piso.json").write_text(
        json.dumps(linhas, indent=2), encoding="utf-8")

    print()
    print(f"VARREDURA DE _PISO_SUSPEITA (com _GANHO_MIN fixo em {_GANHO_MIN})")
    print(f"  valor de producao hoje: {_PISO_SUSPEITA}")
    print()
    print(f"  {'piso':>8}{'TP':>5}{'FN':>5}{'FP':>5}{'TN':>5}"
          f"{'revocacao':>12}{'precisao':>11}{'varre cortes':>15}")
    for piso in (0.030, 0.020, 0.015, 0.010, 0.007, 0.005, 0.003, 0.001, 0.0):
        TP = FN = FP = TN = varre = 0
        for x in linhas:
            trunca = x["nrmse_full"] > piso and x["melhor_ganho"] >= _GANHO_MIN
            varre += x["nrmse_full"] > piso
            multi = x["n_degraus"] > 1
            TP += multi and trunca
            FN += multi and not trunca
            FP += (not multi) and trunca
            TN += (not multi) and not trunca
        print(f"  {piso:>8.3f}{TP:>5}{FN:>5}{FP:>5}{TN:>5}"
              f"{TP/max(1,TP+FN):>11.1%}{TP/max(1,TP+FP):>11.1%}"
              f"{varre:>10}/{len(linhas):<4}")
    print()
    print(f"  E se o piso caisse de vez e so o ganho decidisse (piso=0)?")
    print(f"  Sobra o gate 2. Distribuicao do melhor ganho:")
    for rot, cond in (("1 degrau ", lambda x: x["n_degraus"] == 1),
                      ("2+ degraus", lambda x: x["n_degraus"] > 1)):
        g = np.array([x["melhor_ganho"] for x in linhas if cond(x)])
        print(f"    {rot} n={g.size:<4d} p10={np.percentile(g,10):7.4f} "
              f"p50={np.median(g):7.4f} p90={np.percentile(g,90):7.4f} "
              f"acima de {_GANHO_MIN}: {(g>=_GANHO_MIN).sum()}/{g.size}")


if __name__ == "__main__":
    main()
