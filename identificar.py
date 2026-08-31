#!/usr/bin/env python3
"""Identifica a planta a partir de UMA imagem de resposta ao degrau.

Uso:
    python identificar.py grafico.png
    python identificar.py *.png                    # varias de uma vez
    python identificar.py grafico.png --json       # saida para script
    python identificar.py grafico.png --classico   # extrator sem rede (sem torch)

Devolve a estrutura do modelo (FOPDT ou 2a ordem) e os parametros. Quando a
calibracao dos eixos falha, ainda devolve o bloco ADIMENSIONAL — ordem, zeta e
as grandezas normalizadas pela janela —, porque amortecimento se le da FORMA da
curva e nao da escala dos eixos (Decisao E do PLANO §1.7).

Este arquivo e so uma casca de linha de comando: toda a logica vive em
`identify.pipeline.identify_from_image`, que e a unica porta de entrada do
sistema. Nada aqui muda o resultado — se mudasse, o numero que voce ve aqui nao
seria o mesmo que a suite mede.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MODELO_PADRAO = Path(__file__).resolve().parent / "models" / "unet_stageA.pt"

# Motivos de recusa, em portugues corrente. A pipeline nomeia a causa em vez de
# devolver numero errado em silencio; traduzir isso aqui e o que separa "nao
# funcionou" de "nao funcionou POR ISTO".
MOTIVOS = {
    "polilinha_curta":
        "a curva nao foi encontrada na imagem (menos de 10 pontos extraidos)",
    "ajuste_falhou":
        "o otimizador nao convergiu para nenhuma das duas estruturas",
    "ajuste_inconsistente":
        "o ajuste convergiu mas o residuo e alto demais — a serie extraida nao "
        "sustenta modelo nenhum (mascara provavelmente saltou para um distrator)",
    "resposta_inversa":
        "a curva desce antes de subir. Isso e fase nao-minima (zero no semiplano "
        "direito), que NAO pertence a familia de modelos deste sistema",
    "bbox_not_found":
        "a moldura da area de dados nao foi localizada",
    "ocr_insuficiente":
        "menos de 2 rotulos numericos lidos por eixo",
    "ransac_failed":
        "os rotulos lidos nao formam uma reta consistente",
    "calibration_failed":
        "os ticks aceitos nao ficaram equiespacados — leitura provavelmente errada",
    "sinal_de_escala_invalido":
        "a escala saiu com sinal impossivel (x tem de crescer para a direita)",
}


def _fmt(v, casas=4, unidade=""):
    if v is None:
        return "—"
    if isinstance(v, float) and abs(v) < 1e-12:
        return f"0{unidade}"
    return f"{v:.{casas}g}{unidade}"


def _carrega_imagem(caminho: Path) -> np.ndarray:
    """RGB puro, sem alfa — o contrato de entrada do estagio A."""
    with Image.open(caminho) as im:
        return np.asarray(im.convert("RGB"))


def _relatorio(caminho: Path, r: dict) -> str:
    L = []
    cal = r["calibration"]
    p = r["params"] or {}
    dim = r["dimensionless"] or {}

    L.append(f"\n\033[1m{caminho.name}\033[0m")
    L.append("─" * max(len(caminho.name), 46))

    if not r["order"] and not r["ok"]:
        motivo = MOTIVOS.get(r["reason"], r["reason"] or "desconhecido")
        L.append(f"  SEM RESPOSTA — {motivo}")
        L.append(f"  (codigo: {r['reason']})")
        return "\n".join(L)

    nome = {"fopdt": "1a ordem com atraso (FOPDT)",
            "second": "2a ordem com atraso"}.get(r["order"], r["order"])
    L.append(f"  Estrutura      {nome}")

    if r["ok"]:
        L.append("")
        L.append("  \033[1mParametros fisicos\033[0m")
        if r["order"] == "fopdt":
            L.append(f"    K            {_fmt(p.get('K'))}")
            L.append(f"    tau          {_fmt(p.get('tau'), unidade=' s')}")
        else:
            L.append(f"    K            {_fmt(p.get('K'))}")
            L.append(f"    wn           {_fmt(p.get('wn'), unidade=' rad/s')}")
            L.append(f"    zeta         {_fmt(p.get('zeta'))}")
        L.append(f"    theta        {_fmt(p.get('theta'), unidade=' s')}")
        L.append("")
        L.append(f"  Janela lida    {_fmt(cal.get('T_s'), unidade=' s')}"
                 f"   ·   faixa de y  {_fmt(cal.get('y_faixa'))}")
        L.append(f"  Rotulos        {cal['n_pairs_x']} no eixo x, "
                 f"{cal['n_pairs_y']} no eixo y")
    else:
        motivo = MOTIVOS.get(cal["reason"], cal["reason"] or "desconhecido")
        L.append("")
        L.append(f"  \033[1mSem parametros fisicos\033[0m — {motivo}")
        L.append(f"  Eixos aprovados: x={'sim' if cal['ok_x'] else 'nao'}, "
                 f"y={'sim' if cal['ok_y'] else 'nao'}")
        L.append("")
        L.append("  \033[1mAdimensional\033[0m (independe da escala dos eixos)")
        if dim.get("zeta") is not None:
            L.append(f"    zeta         {_fmt(dim['zeta'])}")
        if dim.get("wn_T") is not None:
            L.append(f"    wn·T         {_fmt(dim['wn_T'])}"
                     "     (multiplique por 1/T para ter rad/s)")
        if dim.get("tau_T") is not None:
            L.append(f"    tau/T        {_fmt(dim['tau_T'])}")
        if dim.get("theta_T") is not None:
            L.append(f"    theta/T      {_fmt(dim['theta_T'])}")
        if dim.get("K_yrange") is not None:
            L.append(f"    K/faixa_y    {_fmt(dim['K_yrange'])}")

        parcial = r.get("physical_parcial") or {}
        uteis = {k: v for k, v in parcial.items() if v is not None}
        if uteis and not r["ok"]:
            L.append("")
            L.append("  \033[1mParcial\033[0m (do eixo que foi aprovado)")
            for k, v in uteis.items():
                un = " s" if k in ("tau", "theta") else (
                     " rad/s" if k == "wn" else "")
                L.append(f"    {k:<12} {_fmt(v, unidade=un)}")

    L.append("")
    L.append(f"  {r['n_points']} pontos extraidos · {r['latency_ms']:.0f} ms")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Identifica a planta a partir de uma imagem de resposta ao degrau.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Uso:")[1].split("Devolve")[0].strip())
    ap.add_argument("imagens", nargs="+", type=Path, help="uma ou mais imagens")
    ap.add_argument("--json", action="store_true",
                    help="saida em JSON, para encadear em script")
    ap.add_argument("--classico", action="store_true",
                    help="usa o extrator sem rede (dispensa torch e GPU)")
    ap.add_argument("--modelo", type=Path, default=MODELO_PADRAO,
                    help=f"checkpoint da U-Net (padrao: {MODELO_PADRAO.name})")
    ap.add_argument("--cpu", action="store_true", help="forca CPU mesmo com GPU disponivel")
    a = ap.parse_args()

    faltando = [p for p in a.imagens if not p.exists()]
    if faltando:
        for p in faltando:
            print(f"erro: nao encontrei {p}", file=sys.stderr)
        return 2

    from identify.pipeline import identify_from_image

    modelo = None
    extrator = None
    dev = "cpu"
    if a.classico:
        from identify.extract_classical import extract_mask_classical
        extrator = extract_mask_classical
    else:
        import torch
        from identify.extract import load_model
        if not a.modelo.exists():
            print(f"erro: checkpoint ausente em {a.modelo}", file=sys.stderr)
            print("      use --classico para rodar sem rede", file=sys.stderr)
            return 2
        dev = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        modelo = load_model(str(a.modelo), dev)

    saidas = []
    for caminho in a.imagens:
        try:
            img = _carrega_imagem(caminho)
        except Exception as e:                      # imagem corrompida, formato exotico
            print(f"erro: nao consegui abrir {caminho.name}: {e}", file=sys.stderr)
            continue
        r = identify_from_image(img, modelo, dev, extractor=extrator)
        if a.json:
            saidas.append({"imagem": str(caminho), **r})
        else:
            print(_relatorio(caminho, r))

    if a.json:
        print(json.dumps(saidas if len(saidas) != 1 else saidas[0],
                         indent=2, ensure_ascii=False, default=str))
    elif not a.json:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
