"""MASCARA ou CALIBRACAO? Separa as duas metades do 'custo da extracao'.

O oraculo (`oraculo_ruido.py --dir .../lote_k_menor1`) mediu que o Estagio D
acerta 100/100 com a serie analitica e a pipeline entrega 75 — ou seja, os 25
pontos que faltam sao INTEIRAMENTE extracao. Mas "extracao" junta dois
trabalhos diferentes, que se consertam de formas diferentes:

  MASCARA (Estagio A, U-Net) — quais pixels sao a curva. Conserta-se com
    retreino e corpus.
  CALIBRACAO (Estagio B, moldura + ticks + OCR) — quantas unidades vale um
    pixel. Conserta-se com codigo, nao com GPU.

Somar os dois num numero so' faz o retreino parecer mais promissor do que e'.

QUATRO CONDICOES sobre as MESMAS amostras, trocando uma peca de cada vez:
  PIPELINE        mascara predita  + calibracao estimada   (o que se entrega)
  AFIM VERDADEIRA mascara predita  + afim do `meta.json`
  MASCARA VERDADEIRA  `mask.png`   + calibracao estimada
  ORACULO         a serie do `meta.json`, direto no Estagio D

A distancia de PIPELINE ate cada oraculo parcial e' o custo daquela peca.

POPULACAO: `data/val` filtrado por |K*degrau| < 1, que e' o estrato que o
`lote_k_menor1` amostra. ATENCAO: o RENDER e' o do corpus de treino, nao o do
`rg_aleatorio.py` — as distribuicoes de estilo diferem, entao o nivel absoluto
nao e' comparavel com o do lote; o que transfere e a REPARTICAO entre as duas
pecas.

Uso:
    .venv/bin/python separa_extracao.py [--n 120]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from identify.calibrate import calibrate
from identify.classical import identify as estagio_d
from identify.classical import model_response
from identify.extract import load_model, predict_mask
from identify.polyline import mask_to_polyline

RAIZ = Path(__file__).resolve().parent
NIVEIS = {"ESTRITO": dict(K=0.05, theta_T=0.02, nrmse=0.02, estrutura=True),
          "PRATICO": dict(K=0.10, theta_T=0.05, nrmse=0.05, estrutura=False)}


def t_dom(order, tau, wn, zeta):
    if order == "fopdt":
        return float(tau)
    wn, zeta = float(wn), float(zeta)
    if zeta < 1.0:
        return 1.0 / (zeta * wn)
    return 1.0 / float(wn * (zeta - np.sqrt(zeta * zeta - 1.0)))


def avalia(r, meta, nivel):
    if r is None:
        return False
    c = NIVEIS[nivel]
    p, v = r.params, meta["params"]
    K, th = float(v["K"]) * float(meta["step_amplitude"]), float(v["theta"])
    T = float(meta["t_window"][1]) - float(meta["t_window"][0])
    if c["estrutura"] and r.order != meta["order"]:
        return False
    if not c["estrutura"]:
        td = t_dom(meta["order"], v.get("tau"), v.get("wn"), v.get("zeta"))
        tdh = t_dom(r.order, p.get("tau"), p.get("wn"), p.get("zeta"))
        if td > 0 and abs(tdh - td) / td > 0.15:
            return False
    if abs(K) > 1e-9 and abs(float(p["K"]) - K) / abs(K) > c["K"]:
        return False
    if abs(float(p["theta"]) - th) / T > c["theta_T"]:
        return False
    t = np.asarray(meta["series"]["t"], float)
    y0 = np.asarray(meta["series"]["y"], float)
    faixa = float(np.ptp(y0))
    if faixa > 0 and float(np.sqrt(np.mean((model_response(r.order, p, t) - y0) ** 2))
                           ) / faixa > c["nrmse"]:
        return False
    return True


def _px_para_dados(x_px, y_px, sx, ox, sy, oy):
    return sx * np.asarray(x_px, float) + ox, sy * np.asarray(y_px, float) + oy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--corpus", default="data/val")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / "models/unet_stageA.pt"), dev)
    modelo.eval()

    res = {k: [] for k in ("pipeline", "afim", "mascara", "oraculo")}
    n = 0
    for s in sorted((RAIZ / a.corpus).glob("sample_*")):
        if n >= a.n:
            break
        meta = json.loads((s / "meta.json").read_text())
        if abs(float(meta["params"]["K"]) * float(meta["step_amplitude"])) >= 1.0:
            continue
        img = np.asarray(Image.open(s / "image.png").convert("RGB"))
        alvo = np.asarray(Image.open(s / "mask.png").convert("L"))
        pred = predict_mask(modelo, img, dev)
        cal = calibrate(img)
        aa = meta["axis_affine"]
        n += 1

        def ajusta(mask, usar_afim_verdadeira):
            bbox = cal.bbox_px if any(cal.bbox_px) else None
            x_px, y_px = mask_to_polyline(mask, bbox=bbox)
            if x_px.size < 10:
                return None
            if usar_afim_verdadeira:
                t, y = _px_para_dados(x_px, y_px, aa["sx"], aa["ox"], aa["sy"], aa["oy"])
            else:
                if not cal.ok:
                    return None
                from identify.calibrate import px_to_data
                t, y = px_to_data(cal, x_px, y_px)
            o = np.argsort(t)
            try:
                return estagio_d(t[o], y[o])
            except Exception:
                return None

        res["pipeline"].append(ajusta(pred, False))
        res["afim"].append(ajusta(pred, True))
        res["mascara"].append(ajusta(alvo, False))
        try:
            res["oraculo"].append(estagio_d(np.asarray(meta["series"]["t"], float),
                                            np.asarray(meta["series"]["y"], float)))
        except Exception:
            res["oraculo"].append(None)
        res.setdefault("_meta", []).append(meta)

    metas = res.pop("_meta")
    print(f"SEPARACAO DO CUSTO DA EXTRACAO — |K*degrau| < 1 em {a.corpus}   n={n}\n")
    print(f"  {'condicao':<34}{'ESTRITO':>10}{'PRATICO':>10}")
    rot = {"pipeline": "PIPELINE (o que se entrega)",
           "afim": "+ AFIM verdadeira (Estagio B ok)",
           "mascara": "+ MASCARA verdadeira (Estagio A ok)",
           "oraculo": "ORACULO (serie verdadeira)"}
    tab = {}
    for k in ("pipeline", "afim", "mascara", "oraculo"):
        tab[k] = {niv: float(np.mean([avalia(r, m, niv)
                                      for r, m in zip(res[k], metas)]))
                  for niv in NIVEIS}
        print(f"  {rot[k]:<34}{tab[k]['ESTRITO']:>10.0%}{tab[k]['PRATICO']:>10.0%}")
    print()
    print("  QUANTO CADA PECA CUSTA (pontos percentuais sobre a PIPELINE)")
    print(f"    {'':<34}{'ESTRITO':>10}{'PRATICO':>10}")
    for k, nome in (("afim", "CALIBRACAO (Estagio B)"),
                    ("mascara", "MASCARA (Estagio A, U-Net)")):
        print(f"    {nome:<34}"
              + "".join(f"{tab[k][niv]-tab['pipeline'][niv]:>+10.0%}" for niv in NIVEIS))
    print(f"    {'as duas juntas (= oraculo)':<34}"
          + "".join(f"{tab['oraculo'][niv]-tab['pipeline'][niv]:>+10.0%}" for niv in NIVEIS))
    print()
    print("  Se a soma das duas pecas for MENOR que o total, elas se sobrepoem:")
    print("  ha figuras em que consertar uma so nao salva, porque a outra tambem")
    print("  esta errada. Se for MAIOR, uma delas mascara a outra.")


if __name__ == "__main__":
    main()
