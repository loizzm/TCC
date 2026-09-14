"""ONDE o ruido quebra: na EXTRACAO ou no AJUSTE?

O PROBLEMA. A assertividade cai de 82 % (>=25 dB) para 52 % (<=20 dB) num
DEGRAU, nao numa rampa: dentro do bloco sujo, 20 dB e 10 dB empatam (10/20 nos
dois, Fisher p=1,000). Isso acusa a borda da distribuicao de treino
(`snr_db` em U(20, 60)) e nao a fisica do ruido. Mas acusar a borda do TREINO
so faz sentido se o gargalo estiver no ESTAGIO A — se for o ajustador que nao
aguenta serie ruidosa, retreinar a U-Net nao muda nada, e o retreino de ruido
que ja rodou (que NAO moveu o bloco, McNemar p=0,73) teria uma explicacao.

O EXPERIMENTO. Tres condicoes sobre as MESMAS plantas do lote:

  PIPELINE  — imagem -> mascara -> polilinha -> calibracao -> Estagio D.
              E o que o sistema entrega.
  ORACULO RUIDOSO — a serie analitica da planta MAIS ruido no mesmo `sigma_rel`
              que a figura usou, direto no Estagio D. Pula extracao e
              calibracao inteiras, mas o ajustador ve o MESMO nivel de ruido.
  ORACULO LIMPO — a serie analitica sem ruido, direto no Estagio D. Teto do
              ajustador com dado perfeito.

COMO LER:
  ORACULO RUIDOSO ~ ORACULO LIMPO, e os dois >> PIPELINE
      -> o ajustador aguenta o ruido; quem perde e a EXTRACAO. Retreinar o
         Estagio A e' o caminho certo, e o retreino anterior falhou por outro
         motivo (composicao, selecao de epoca) e nao por ser a frente errada.
  ORACULO RUIDOSO cai junto com o PIPELINE
      -> o gargalo e' o AJUSTE. Nenhum retreino da U-Net resolve, e a frente
         certa passa a ser `identify/classical.py`.

A realizacao do ruido do oraculo NAO e' a mesma da figura — `verdade.json`
guarda `sigma_rel`, nao a serie desenhada. Isso e' de proposito: a pergunta e'
"o ajustador aguenta ESTE NIVEL de ruido?", que e' sobre a distribuicao e nao
sobre uma realizacao. A semente e derivada do nome do arquivo, entao o
resultado e reprodutivel.

O numero de pontos e' casado com o que a pipeline extraiu naquela figura
(`n_points` do `resultado.json`), para o oraculo nao ganhar de graca por
receber uma grade mais densa.

Uso:
    .venv/bin/python oraculo_ruido.py [--dir reports/amostras_aleatorias/lote_ruido]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import signal

from identify.classical import identify as estagio_d
from identify.classical import model_response

RAIZ = Path(__file__).resolve().parent
# MESMOS criterios do `acerto_conjuntivo.py`, inclusive o NRMSE da curva — as
# tres colunas tem de ser julgadas pela mesma regua, senao o oraculo ganha de
# graca por nao ser cobrado num criterio.
NIVEIS = {"ESTRITO": dict(K=0.05, theta_T=0.02, nrmse=0.02, t_dom=None),
          "PRATICO": dict(K=0.10, theta_T=0.05, nrmse=0.05, t_dom=0.15)}


def t_dom(order, tau, wn, zeta) -> float:
    if order == "fopdt":
        return float(tau)
    wn, zeta = float(wn), float(zeta)
    if zeta < 1.0:
        return 1.0 / (zeta * wn)
    p = wn * (zeta - np.sqrt(zeta * zeta - 1.0))
    return 1.0 / float(p)


def serie(v: dict, t: np.ndarray) -> np.ndarray:
    """Resposta analitica LIMPA da planta declarada. MESMA aritmetica do
    `analisa_aleatorias.resposta` — o trecho antes de `theta` fica zerado e a
    `scipy.signal.step` recebe uma grade uniforme comecando em zero, que e' o
    que ela exige."""
    K = float(v["K"])
    theta = float(v["theta"])
    y = np.zeros_like(t)
    m = t >= theta
    ta = t[m] - theta
    if ta.size == 0:
        return y
    if v["order"] == "fopdt":
        y[m] = K * (1.0 - np.exp(-ta / float(v["tau"])))
    else:
        wn, z = float(v["wn"]), float(v["zeta"])
        sys_ = signal.TransferFunction([K * wn * wn], [1.0, 2.0 * z * wn, wn * wn])
        _, ya = signal.step(sys_, T=ta)
        y[m] = ya
    return y


def avalia(r, v, t: np.ndarray, y0: np.ndarray, nivel: str) -> bool:
    """Reprova pelos mesmos criterios do `acerto_conjuntivo.py`, menos os que
    dependem de imagem (estrutura exata entra so no ESTRITO, como la)."""
    c = NIVEIS[nivel]
    if r is None:
        return False
    K_hat = float(r.params["K"])
    if v["K"] != 0 and abs(K_hat - v["K"]) / abs(v["K"]) > c["K"]:
        return False
    if np.sign(K_hat) != np.sign(v["K"]):
        return False
    T = float(v["t_fim"])
    if abs(float(r.params["theta"]) - float(v["theta"])) / T > c["theta_T"]:
        return False
    if c["t_dom"] is None:
        if r.order != v["order"]:
            return False
    else:
        td_hat = t_dom(r.order, r.params.get("tau"), r.params.get("wn"),
                       r.params.get("zeta"))
        td = float(v["t_dom"])
        if td > 0 and abs(td_hat - td) / td > c["t_dom"]:
            return False
    faixa = float(np.ptp(y0))
    if faixa > 0:
        y_hat = model_response(r.order, r.params, t)
        if float(np.sqrt(np.mean((y_hat - y0) ** 2)) / faixa) > c["nrmse"]:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=RAIZ / "reports/amostras_aleatorias/lote_ruido")
    a = ap.parse_args()
    V = {x["arquivo"]: x for x in json.loads((a.dir / "verdade.json").read_text())}
    A = {Path(x["imagem"]).name: x
         for x in json.loads((a.dir / "resultado.json").read_text())}
    P = {x["arquivo"]: x for x in json.loads((a.dir / "analise.json").read_text())}

    from acerto_conjuntivo import NIVEIS as NC, avalia as avalia_pipe

    linhas = []
    for nome, v in V.items():
        n = int(A[nome]["n_points"]) if A.get(nome, {}).get("n_points") else 400
        n = int(np.clip(n, 50, 2000))
        t = np.linspace(0.0, float(v["t_fim"]), n)
        y0 = serie(v, t)
        sd = int(hashlib.sha256(nome.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(sd)
        # Lote sem ruido de medicao (`rg_aleatorio.py`) nao tem `sigma_rel`:
        # ali as duas colunas de oraculo coincidem, e a distancia ate a
        # PIPELINE e' inteiramente extracao + calibracao.
        sigma = float(v.get("sigma_rel") or 0.0) * float(np.ptp(y0))
        yr = y0 + sigma * rng.standard_normal(y0.size)
        try:
            r_lim = estagio_d(t, y0)
        except Exception:
            r_lim = None
        try:
            r_rui = estagio_d(t, yr)
        except Exception:
            r_rui = None
        linhas.append({
            "snr": (int(m.group(1)) if (m := re.search(r"_(\d+)dB_", nome)) else None),
            "limpo": {k: avalia(r_lim, v, t, y0, k) for k in NIVEIS},
            "ruidoso": {k: avalia(r_rui, v, t, y0, k) for k in NIVEIS},
            "pipe": {k: not avalia_pipe(P[nome], NC[k]) for k in NIVEIS},
        })

    faixas = sorted({x["snr"] for x in linhas if x["snr"] is not None}, reverse=True)
    print(f"ORACULO DO RUIDO — onde o ruido quebra   n={len(linhas)}\n")
    for nivel in NIVEIS:
        print(f"  {nivel}")
        print(f"    {'SNR':>7}{'n':>5}{'PIPELINE':>12}{'ORAC.RUIDOSO':>15}"
              f"{'ORAC.LIMPO':>13}")
        agregados = (["<=20", ">=25"] if faixas else []) + ["TODAS"]
        for s in faixas + agregados:
            if s == "<=20":
                sub = [x for x in linhas if x["snr"] is not None and x["snr"] <= 20]
            elif s == ">=25":
                sub = [x for x in linhas if x["snr"] is not None and x["snr"] >= 25]
            elif s == "TODAS":
                sub = linhas
            else:
                sub = [x for x in linhas if x["snr"] == s]
            if not sub:
                continue
            rot = f"{s} dB" if isinstance(s, int) else str(s)
            print(f"    {rot:>7}{len(sub):>5}"
                  f"{np.mean([x['pipe'][nivel] for x in sub]):>11.0%} "
                  f"{np.mean([x['ruidoso'][nivel] for x in sub]):>14.0%} "
                  f"{np.mean([x['limpo'][nivel] for x in sub]):>12.0%}")
        print()
    print("  ORAC.RUIDOSO ~ ORAC.LIMPO e ambos >> PIPELINE -> gargalo na EXTRACAO.")
    print("  ORAC.RUIDOSO cai junto com PIPELINE           -> gargalo no AJUSTE.")
    print()
    sujo = [x for x in linhas if x["snr"] is not None and x["snr"] <= 20]
    rotulo = "bloco <=20 dB" if sujo else "lote inteiro (sem ruido de medicao)"
    print(f"  DE QUEM E' O BURACO ({rotulo})")
    print(f"    {'':<28}{'ESTRITO':>10}{'PRATICO':>10}")
    sub = sujo or linhas
    for rot, f in (("teto com dado perfeito", lambda k: np.mean([x["limpo"][k] for x in sub])),
                   ("- custo do RUIDO no ajuste",
                    lambda k: np.mean([x["limpo"][k] for x in sub])
                            - np.mean([x["ruidoso"][k] for x in sub])),
                   ("- custo da EXTRACAO",
                    lambda k: np.mean([x["ruidoso"][k] for x in sub])
                            - np.mean([x["pipe"][k] for x in sub])),
                   ("= o que a pipeline entrega",
                    lambda k: np.mean([x["pipe"][k] for x in sub]))):
        print(f"    {rot:<28}{f('ESTRITO'):>9.0%} {f('PRATICO'):>9.0%}")
    print()
    print("  O 'custo da EXTRACAO' e' TODO o espaco que um retreino do Estagio A")
    print("  pode recuperar. O 'custo do RUIDO no ajuste' so cai mexendo em")
    print("  `identify/classical.py` — nenhuma mascara melhor o alcanca. A")
    print("  pipeline NAO suaviza a serie antes do Estagio D (verificado: nao ha")
    print("  savgol nem filtro nenhum em `pipeline.py`), entao a comparacao com")
    print("  o oraculo, que tambem nao suaviza, e' justa.")


if __name__ == "__main__":
    main()
