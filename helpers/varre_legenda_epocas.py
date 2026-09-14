"""Varre TODOS os checkpoints de uma pasta de epocas no par real de oclusao.

A PERGUNTA. O retreino de ruido (`helpers/retreino_ruido.sh`) melhorou as tres batidas
de controle e quebrou `caso_real_neg_super.png`: `wn` e `zeta` saltaram de
1,3 %/0,2 % no modelo promovido para 32,1 %/32,2 % no epoca_17. Sondar cinco
candidatos nao responde se o dano e INTRINSECO ao estrato de ruido ou se e uma
regiao ruim do run — para isso e' preciso varrer a corrida inteira.

O CUSTO e' duas imagens por checkpoint, entao nao ha razao para amostrar.

E O PAR E' O CONTROLE. `caso_real_neg_super_legenda_movida.png` e' a MESMA
figura com a legenda em outro canto. Um checkpoint que erra nas duas esta ruim
em 2a ordem superamortecida e nao tem nada a ver com oclusao; so a DIFERENCA
entre as duas colunas isola a variavel.

Uso:
    .venv/bin/python helpers/varre_legenda_epocas.py [models/epocas_ruido] \
        [--referencia models/unet_stageA.pt]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from identify.extract import load_model
from identify.pipeline import identify_from_image

RAIZ = Path(__file__).resolve().parent.parent
FIX = RAIZ / "tests" / "fixtures"
VERDADE = {"K": -3.0, "wn": 4.0, "zeta": 1.25, "theta": 3.5}


def erros(modelo, dev, png: str) -> dict[str, float] | None:
    img = np.asarray(Image.open(FIX / png).convert("RGB"))
    r = identify_from_image(img, modelo, dev)
    if not r["ok"] or r["order"] != "second":
        return None
    return {k: abs(r["params"][k] - v) / abs(v) for k, v in VERDADE.items()}


def linha(nome: str, modelo, dev) -> None:
    ocl = erros(modelo, dev, "caso_real_neg_super.png")
    sem = erros(modelo, dev, "caso_real_neg_super_legenda_movida.png")
    def f(e):
        return "  sem 2a ordem  " if e is None else (
            f"{e['wn']:>7.1%}{e['zeta']:>8.1%}")
    pior = (max(ocl["wn"], ocl["zeta"]) if ocl else float("nan"))
    marca = "  <-- quebra" if (ocl is None or pior > 0.10) else ""
    print(f"  {nome:<16}{f(ocl)}   |{f(sem)}{marca}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta", nargs="?", default="models/epocas_ruido")
    ap.add_argument("--referencia", default="models/unet_stageA.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("PAR REAL DE OCLUSAO POR LEGENDA — erro de wn e zeta\n")
    print(f"  {'checkpoint':<16}{'legenda OCLUINDO':^17}   |"
          f"{'legenda MOVIDA':^17}")
    print(f"  {'':<16}{'wn':>7}{'zeta':>8}   {'':<3}{'wn':>7}{'zeta':>8}")

    m = load_model(str(RAIZ / a.referencia), dev)
    m.eval()
    linha(Path(a.referencia).stem, m, dev)
    print()

    pts = sorted((RAIZ / a.pasta).glob("epoca_*.pt"),
                 key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    for p in pts:
        m = load_model(str(p), dev)
        m.eval()
        linha(p.stem, m, dev)
    print()
    print("  'quebra' = pior de wn/zeta acima de 10 % COM a legenda em cima.")
    print("  Se a coluna da direita estiver boa e a da esquerda ruim, o dano e")
    print("  de OCLUSAO. Se as duas estiverem ruins, o checkpoint e ruim em 2a")
    print("  ordem superamortecida e a oclusao nao explica nada.")


if __name__ == "__main__":
    main()
