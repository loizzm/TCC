#!/usr/bin/env python3
"""Escolhe o checkpoint do retreino FORA DA FAMILIA (§65) pelo objetivo certo.

POR QUE NAO REUSAR OS OUTROS DOIS SELETORES. `seleciona_checkpoint.py` mede
cobertura do plato em `data/val` — a metrica do retreino de ganho negativo.
`seleciona_checkpoint_contagem.py` mede AUC de contagem no corpus real — a
metrica do retreino multi-degrau. Nenhum dos dois olha para o defeito que ESTE
retreino existe para consertar, e promover pelo instrumento do retreino
anterior e' o erro que o §40.9 ja registrou uma vez.

O OBJETIVO e' a cobertura do plato de repouso em `data/val_nmp`: das colunas em
que a figura desenha a curva em repouso, quantas a mascara acende. Baseline
medido com `models/unet_stageA.pt`: 0,286 (contra 0,884 em `data/val`).

AS GUARDAS sao tres, e cada uma protege um ganho que ja foi pago:
  * cobertura do plato em `data/val` — o retreino de ganho negativo;
  * IoU em `data/val` + `data/val_multi` — a segmentacao em geral;
  * AUC de contagem nas 299 figuras reais — o retreino multi-degrau.
Um checkpoint que sobe o objetivo derrubando qualquer uma delas nao aprendeu a
tarefa nova: trocou uma pela outra. Nao promover.

NAO SELECIONE PELO `IoU_val` DO LOG. Ele mistura seis populacoes e ja
anticorrelacionou com a metrica real neste projeto (Spearman -0,401 depois da
epoca 08 do retreino multi): `train_unet.py` teria escolhido o checkpoint pior.

Uso:
    .venv/bin/python seleciona_checkpoint_nmp.py models/epocas_nmp \\
        [--referencia models/unet_stageA.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from identify.extract import load_model
from mede_plato_repouso import cobertura_do_plato
from seleciona_checkpoint_contagem import (auc_contagem_real, figuras_reais,
                                           iou_val)

RAIZ = Path(__file__).resolve().parent


def avalia(modelo, itens, dev, a) -> dict:
    """Objetivo + as tres guardas, num checkpoint."""
    modelo.eval()
    alvo = cobertura_do_plato(RAIZ / "data" / "val_nmp", modelo, dev, lim=a.lim_plato)
    base = cobertura_do_plato(RAIZ / "data" / "val", modelo, dev, lim=a.lim_plato)
    return {
        "plato_nmp": float(np.median(alvo)) if alvo.size else float("nan"),
        "plato_val": float(np.median(base)) if base.size else float("nan"),
        "iou": iou_val(modelo, a.val_dir, dev, limite=a.batches_iou),
        "auc_real": auc_contagem_real(modelo, itens, dev),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--referencia", type=Path,
                    default=RAIZ / "models" / "unet_stageA.pt")
    ap.add_argument("--val-dir", action="append", default=None,
                    help="para o IoU. default: data/val + data/val_multi")
    ap.add_argument("--lim-plato", type=int, default=120,
                    help="amostras por corpus na cobertura do plato")
    ap.add_argument("--batches-iou", type=int, default=40)
    ap.add_argument("--limite-iou", type=float, default=0.02,
                    help="queda de IoU tolerada contra a referencia")
    ap.add_argument("--limite-plato", type=float, default=0.03,
                    help="queda tolerada na cobertura do plato de data/val")
    ap.add_argument("--limite-auc", type=float, default=0.03,
                    help="queda tolerada na AUC de contagem real. So vale "
                         "quando a REFERENCIA tem cabeca de contagem.")
    ap.add_argument("--auc-minima", type=float, default=None,
                    help="barra ABSOLUTA para a AUC de contagem real, para "
                         "quando a referencia nao tem cabeca — que e' o caso "
                         "de `models/unet_stageA.pt`, o modelo promovido. Sem "
                         "ela a guarda de contagem fica DESLIGADA e o ganho do "
                         "retreino multi-degrau nao esta protegido. O valor a "
                         "usar e' o melhor apto daquele retreino: 0,6838 "
                         "(epoca_08, IoU 0,756), em reports/selecao_multi.json.")
    a = ap.parse_args()
    a.val_dir = a.val_dir or ["data/val", "data/val_multi"]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    itens = figuras_reais()
    print(f"  {len(itens)} figuras reais para a AUC de contagem")

    ref = avalia(load_model(str(a.referencia), dev), itens, dev, a)
    tem_auc = bool(np.isfinite(ref["auc_real"]))
    # Barra de contagem: relativa a referencia quando ela tem cabeca, absoluta
    # quando nao tem. Uma das duas, nunca as duas.
    barra_auc = (ref["auc_real"] - a.limite_auc) if tem_auc else a.auc_minima
    print(f"  referencia {a.referencia.name}: "
          f"plato_nmp={ref['plato_nmp']:.3f}  plato_val={ref['plato_val']:.3f}  "
          f"IoU={ref['iou']:.4f}  AUC_real="
          + (f"{ref['auc_real']:.4f}" if tem_auc else "— (sem cabeca)"))
    if barra_auc is None:
        print("  guarda de AUC DESLIGADA: a referencia nao tem cabeca de "
              "contagem e nenhuma --auc-minima foi dada. O ganho do retreino "
              "multi-degrau NAO esta protegido nesta selecao.")
    else:
        print(f"  barra de AUC de contagem: {barra_auc:.4f}"
              + ("  (relativa a referencia)" if tem_auc else "  (absoluta)"))
    print()

    linhas = []
    for ck in sorted(a.ckpt_dir.glob("epoca_*.pt")):
        m = avalia(load_model(str(ck), dev), itens, dev, a)
        g_iou = m["iou"] >= ref["iou"] - a.limite_iou
        g_plato = m["plato_val"] >= ref["plato_val"] - a.limite_plato
        g_auc = (barra_auc is None) or (np.isfinite(m["auc_real"])
                                        and m["auc_real"] >= barra_auc)
        m.update(ckpt=ck.name, guarda_iou=g_iou, guarda_plato=g_plato,
                 guarda_auc=g_auc, apto=bool(g_iou and g_plato and g_auc))
        linhas.append(m)
        falhou = [n for n, ok in (("IoU", g_iou), ("plato_val", g_plato),
                                  ("AUC", g_auc)) if not ok]
        print(f"  {ck.name}  plato_nmp={m['plato_nmp']:.3f}  "
              f"plato_val={m['plato_val']:.3f}  IoU={m['iou']:.4f}  "
              f"AUC={m['auc_real']:.4f}"
              + (f"  <-- REGREDIU: {', '.join(falhou)}" if falhou else ""),
              flush=True)

    (RAIZ / "reports" / f"selecao_{a.ckpt_dir.name}.json").write_text(
        json.dumps({"referencia": ref, "checkpoints": linhas}, indent=2))

    aptos = [x for x in linhas if x["apto"]]
    print()
    if not aptos:
        print("  NENHUM checkpoint passa as guardas. NAO PROMOVER — o treino")
        print("  trocou um ganho por outro em vez de somar.")
        return
    melhor = max(aptos, key=lambda x: x["plato_nmp"])
    print(f"  VENCEDOR: {melhor['ckpt']}  plato_nmp={melhor['plato_nmp']:.3f} "
          f"(referencia {ref['plato_nmp']:.3f})  "
          f"plato_val={melhor['plato_val']:.3f}  IoU={melhor['iou']:.4f}")
    print(f"  ({len(aptos)}/{len(linhas)} passaram as guardas)")
    if melhor["plato_nmp"] <= ref["plato_nmp"] + 0.05:
        print()
        print("  ATENCAO: o objetivo mal se moveu. O estrato entrou no treino e")
        print("  a mascara continua largando a curva fora da familia — antes de")
        print("  promover, e' o caso de duvidar do estrato, nao do checkpoint.")


if __name__ == "__main__":
    main()
