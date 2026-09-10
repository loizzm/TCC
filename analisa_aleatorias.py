"""Confronta a saida de `identificar.py` com a verdade de `rg_aleatorio.py`.

Tres niveis de veredito, do mais frouxo ao mais exigente:

  1. RESPOSTA      — a pipeline devolveu alguma coisa (nao recusou).
  2. PARAMETRO     — cada grandeza contra a verdade declarada. Como a estrutura
                     escolhida pode diferir da verdadeira sem que a dinamica
                     mude (um polo lento de 2a ordem superamortecida E um `tau`
                     de FOPDT), a grandeza que vale para as duas estruturas e a
                     CONSTANTE DE TEMPO DOMINANTE `t_dom`, medida sempre.
  3. CURVA         — reconstroi a resposta a partir dos parametros reportados e
                     compara com a verdade analitica do PRIMEIRO degrau, no
                     trecho em que so o primeiro degrau age. E o unico veredito
                     que nao depende de qual rotulo de estrutura saiu.

O erro de theta sai em duas formas: relativo (|dtheta|/theta) e normalizado
pela janela (|dtheta|/T), porque theta pequeno faz o relativo explodir sem que
o ajuste tenha piorado.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal

import argparse

_RAIZ = Path(__file__).resolve().parent / "reports" / "amostras_aleatorias"
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--dir", type=Path, default=_RAIZ)
BASE = _ap.parse_known_args()[0].dir


def t_dom(order, tau, wn, zeta):
    if order == "fopdt":
        return float(tau)
    if zeta <= 1.0:
        return float(1.0 / (zeta * wn))
    return float((zeta + np.sqrt(zeta * zeta - 1.0)) / wn)


def resposta(order, K, tau, wn, zeta, theta, t):
    """Resposta ao degrau unitario do modelo, ja escalada por K e atrasada."""
    y = np.zeros_like(t)
    m = t >= theta
    ta = t[m] - theta
    if ta.size == 0:
        return y
    if order == "fopdt":
        y[m] = K * (1.0 - np.exp(-ta / tau))
    else:
        sys_ = signal.TransferFunction([K * wn * wn], [1.0, 2.0 * zeta * wn, wn * wn])
        _, ya = signal.step(sys_, T=ta)
        y[m] = ya
    return y


def erro_rel(hat, ref):
    if hat is None or ref is None or ref == 0:
        return None
    return float((hat - ref) / abs(ref))


def resumo(vals, nome):
    v = np.array([abs(x) for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return f"  {nome:<26} n=0"
    return (f"  {nome:<26} n={v.size:<4d} mediana={np.median(v):8.4f}  "
            f"media={v.mean():8.4f}  p90={np.percentile(v, 90):8.4f}  "
            f"max={v.max():8.4f}")


def main() -> None:
    verdades = {v["arquivo"]: v for v in json.loads((BASE / "verdade.json").read_text())}
    saidas = json.loads((BASE / "resultado.json").read_text())

    linhas = []
    for s in saidas:
        nome = Path(s["imagem"]).name
        v = verdades[nome]
        p = s.get("params") or {}
        ordem_hat = s.get("order")

        reg = {
            "arquivo": nome, "familia": v["familia"],
            "n_degraus": v["n_degraus"], "linestyle": v["linestyle"],
            "preenchimento": v["preenchimento"], "fundo_escuro": v["fundo_escuro"],
            "legenda_loc": v["legenda_loc"], "dpi": v["dpi"],
            "order_true": v["order"], "order_hat": ordem_hat,
            "respondeu": bool(ordem_hat), "ok": bool(s.get("ok")),
            "reason": s.get("reason"),
            "cal_ok": bool(s["calibration"]["ok"]),
            "cal_reason": s["calibration"]["reason"],
            "truncado_em": s.get("truncado_em"),
            "n_points": s.get("n_points"), "latency_ms": s.get("latency_ms"),
            "K_true": v["K"], "theta_true": v["theta"], "T_true": v["t_fim"],
            "t_dom_true": v["t_dom"],
        }

        if s.get("ok") and p:
            reg["K_hat"] = p.get("K")
            reg["theta_hat"] = p.get("theta")
            reg["err_K"] = erro_rel(p.get("K"), v["K"])
            # sinal de K: acertar o SENTIDO da resposta e pre-requisito de tudo
            reg["sinal_K_ok"] = bool(p.get("K") is not None
                                     and np.sign(p["K"]) == np.sign(v["K"]))
            dth = None if p.get("theta") is None else p["theta"] - v["theta"]
            reg["err_theta_rel"] = (None if dth is None or v["theta"] == 0
                                    else float(dth / v["theta"]))
            reg["err_theta_T"] = None if dth is None else float(dth / v["t_fim"])

            td_hat = t_dom(ordem_hat, p.get("tau"), p.get("wn"), p.get("zeta"))
            reg["t_dom_hat"] = td_hat
            reg["err_t_dom"] = erro_rel(td_hat, v["t_dom"])

            # so quando as duas estruturas coincidem os parametros proprios
            # sao comparaveis um a um
            if ordem_hat == v["order"]:
                if ordem_hat == "fopdt":
                    reg["err_tau"] = erro_rel(p.get("tau"), v["tau"])
                else:
                    reg["err_wn"] = erro_rel(p.get("wn"), v["wn"])
                    reg["err_zeta"] = erro_rel(p.get("zeta"), v["zeta"])

            # --- nivel 3: erro de CURVA no trecho do 1o degrau
            # limite superior do trecho valido: onde o 2o degrau entra (ou o
            # fim da janela, quando so ha um degrau)
            if v["n_degraus"] > 1:
                t_lim = float(v["degraus"][1][1]) + v["theta_sistema"]
            else:
                t_lim = float(v["t_fim"])
            t = np.linspace(0.0, t_lim, 600)
            y_true = resposta(v["order"], v["K"], v["tau"], v["wn"], v["zeta"],
                              v["theta"], t)
            y_hat = resposta(ordem_hat, p.get("K"), p.get("tau"), p.get("wn"),
                             p.get("zeta"), p.get("theta"), t)
            faixa = float(np.ptp(y_true))
            reg["nrmse_curva"] = (None if faixa <= 0 else
                                  float(np.sqrt(np.mean((y_hat - y_true) ** 2)) / faixa))
            reg["t_lim_curva"] = t_lim
            # JANELA EFETIVA DO 1o DEGRAU, em constantes de tempo dominantes.
            # E o quanto da resposta do primeiro degrau esta de fato desenhado
            # antes de o proximo entrar. Assentar a 1 % leva ~4,6 t_dom; abaixo
            # disso o patamar nao aparece e K deixa de ser observavel.
            reg["janela_ef"] = float((t_lim - v["theta"]) / v["t_dom"])
            # ZETA_BOUNDS = (1e-3, 10.0) em identify/classical.py. Encostar no
            # limite nao e um valor, e o otimizador dizendo que nao ha minimo
            # interior — sintoma da degenerescencia K x amortecimento.
            z = p.get("zeta")
            reg["zeta_no_limite"] = bool(z is not None and
                                         (z <= 1.001e-3 or z >= 9.99))
        else:
            d = s.get("dimensionless") or {}
            reg["zeta_adim"] = d.get("zeta")
            if d.get("zeta") is not None and v["zeta"] is not None:
                reg["err_zeta_adim"] = erro_rel(d["zeta"], v["zeta"])

        linhas.append(reg)

    (BASE / "analise.json").write_text(
        json.dumps(linhas, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------- relatorio
    out = []
    def P(s=""):
        out.append(s)

    n = len(linhas)
    P(f"AMOSTRAS: {n}")
    P()

    P("=" * 78)
    P("1. TAXA DE RESPOSTA")
    P("=" * 78)
    for fam in ("rg", "neg", "multi", "TODAS"):
        sub = linhas if fam == "TODAS" else [x for x in linhas if x["familia"] == fam]
        nr = sum(x["respondeu"] for x in sub)
        nok = sum(x["ok"] for x in sub)
        ncal = sum(x["cal_ok"] for x in sub)
        P(f"  {fam:<8} n={len(sub):<4d} respondeu={nr:>3d} ({nr/len(sub):6.1%})  "
          f"calibrou={ncal:>3d} ({ncal/len(sub):6.1%})  "
          f"fisico(ok)={nok:>3d} ({nok/len(sub):6.1%})")
    P()
    mot = defaultdict(int)
    for x in linhas:
        if not x["ok"]:
            mot[x["reason"] or x["cal_reason"] or "?"] += 1
    P("  motivos de nao entregar parametro fisico:")
    for k, c in sorted(mot.items(), key=lambda kv: -kv[1]):
        P(f"    {k:<28} {c}")
    P()

    P("=" * 78)
    P("2. ESTRUTURA (so entre as que responderam)")
    P("=" * 78)
    resp = [x for x in linhas if x["respondeu"]]
    acerto = sum(x["order_hat"] == x["order_true"] for x in resp)
    P(f"  acerto estrito de rotulo: {acerto}/{len(resp)} ({acerto/len(resp):.1%})")
    mat = defaultdict(int)
    for x in resp:
        mat[(x["order_true"], x["order_hat"])] += 1
    P("  matriz de confusao (verdade -> predito):")
    for k, c in sorted(mat.items()):
        P(f"    {k[0]:<8} -> {k[1]:<8} {c}")
    # quantas trocas de rotulo sao DINAMICAMENTE inocuas (t_dom bate a <15%)
    trocas = [x for x in resp if x["ok"] and x["order_hat"] != x["order_true"]]
    inocuas = [x for x in trocas
               if x.get("err_t_dom") is not None and abs(x["err_t_dom"]) < 0.15]
    if trocas:
        P(f"  trocas de rotulo com t_dom a menos de 15%: "
          f"{len(inocuas)}/{len(trocas)} — a dinamica nao muda, so o nome")
    P()

    P("=" * 78)
    P("3. PARAMETROS FISICOS (so as ok=True)")
    P("=" * 78)
    oks = [x for x in linhas if x["ok"]]
    P(f"  n = {len(oks)}")
    sig = sum(1 for x in oks if x.get("sinal_K_ok"))
    P(f"  sinal de K correto: {sig}/{len(oks)} ({sig/max(1,len(oks)):.1%})")
    P()
    P("  erro relativo (|.|):")
    P(resumo([x.get("err_K") for x in oks], "K"))
    P(resumo([x.get("err_t_dom") for x in oks], "t_dom (unifica ordens)"))
    P(resumo([x.get("err_tau") for x in oks], "tau (mesma estrutura)"))
    P(resumo([x.get("err_wn") for x in oks], "wn (mesma estrutura)"))
    P(resumo([x.get("err_zeta") for x in oks], "zeta (mesma estrutura)"))
    P(resumo([x.get("err_theta_rel") for x in oks], "theta (relativo)"))
    P(resumo([x.get("err_theta_T") for x in oks], "theta (|dtheta|/T janela)"))
    P()
    P("  erro de CURVA reconstruida (NRMSE no trecho do 1o degrau):")
    P(resumo([x.get("nrmse_curva") for x in oks], "nrmse"))
    for lim in (0.02, 0.05, 0.10, 0.20):
        c = sum(1 for x in oks if x.get("nrmse_curva") is not None
                and x["nrmse_curva"] <= lim)
        P(f"    NRMSE <= {lim:.2f}: {c}/{len(oks)} ({c/max(1,len(oks)):.1%})  "
          f"[{c}/{n} do total = {c/n:.1%}]")
    P()

    P("=" * 78)
    P("4. ESTRATIFICACAO")
    P("=" * 78)

    def estrato(chave, titulo):
        P(f"  por {titulo}:")
        grupos = defaultdict(list)
        for x in linhas:
            grupos[x[chave]].append(x)
        for k in sorted(grupos, key=lambda z: str(z)):
            g = grupos[k]
            nok = sum(y["ok"] for y in g)
            nr = sum(y["respondeu"] for y in g)
            nrm = [y["nrmse_curva"] for y in g
                   if y.get("nrmse_curva") is not None]
            med = f"{np.median(nrm):.4f}" if nrm else "—"
            bom = sum(1 for z in nrm if z <= 0.05)
            P(f"    {str(k):<16} n={len(g):<4d} respondeu={nr/len(g):6.1%}  "
              f"fisico={nok/len(g):6.1%}  NRMSE med={med:>8}  "
              f"NRMSE<=0.05: {bom}/{len(g)}")
        P()

    estrato("familia", "familia")
    estrato("linestyle", "tipo de linha da curva")
    estrato("n_degraus", "numero de degraus")
    estrato("preenchimento", "preenchimento")
    estrato("fundo_escuro", "fundo escuro")
    estrato("order_true", "ordem verdadeira")

    P("=" * 78)
    P("5. TRUNCAGEM (o mecanismo de multi-degrau)")
    P("=" * 78)
    for nd in sorted({x["n_degraus"] for x in linhas}):
        g = [x for x in linhas if x["n_degraus"] == nd]
        tr = [x for x in g if x["truncado_em"] is not None]
        P(f"  {nd} degrau(s): n={len(g)}  truncou={len(tr)} ({len(tr)/len(g):.1%})")
        if nd > 1 and tr:
            # onde DEVERIA truncar: no instante em que o 2o degrau age
            certos = 0
            for x in tr:
                v = verdades[x["arquivo"]]
                alvo = float(v["degraus"][1][1]) + v["theta_sistema"]
                if abs(x["truncado_em"] - alvo) <= 0.20 * v["t_fim"]:
                    certos += 1
            P(f"      truncou a menos de 20% da janela do 2o degrau: "
              f"{certos}/{len(tr)}")
    P()

    P("=" * 78)
    P("5b. TRUNCAGEM COMO DETECTOR DE 'MAIS DE UM DEGRAU'")
    P("=" * 78)
    TP = sum(1 for x in oks if x["n_degraus"] > 1 and x["truncado_em"] is not None)
    FN = sum(1 for x in oks if x["n_degraus"] > 1 and x["truncado_em"] is None)
    FP = sum(1 for x in oks if x["n_degraus"] == 1 and x["truncado_em"] is not None)
    TN = sum(1 for x in oks if x["n_degraus"] == 1 and x["truncado_em"] is None)
    P(f"  TP={TP}  FN={FN}  FP={FP}  TN={TN}")
    P(f"  precisao={TP/max(1,TP+FP):.1%}   revocacao={TP/max(1,TP+FN):.1%}")
    for rot, g in (("1 degrau  ", [x for x in oks if x["n_degraus"] == 1]),
                   ("2+ degraus", [x for x in oks if x["n_degraus"] > 1])):
        for sub, gg in (("truncou   ", [x for x in g if x["truncado_em"] is not None]),
                        ("nao trunc ", [x for x in g if x["truncado_em"] is None])):
            if not gg:
                continue
            nr = [x["nrmse_curva"] for x in gg if x.get("nrmse_curva") is not None]
            ek = [abs(x["err_K"]) for x in gg if x.get("err_K") is not None]
            P(f"    {rot} {sub} n={len(gg):<3d} NRMSE med={np.median(nr):7.4f}  "
              f"|errK| med={np.median(ek):7.4f}")
    # O que o ajuste NAO-truncado de uma figura multi-degrau esta descrevendo:
    # a excursao TOTAL (soma dos degraus), nao a do primeiro. Nao e um numero
    # errado, e a resposta a outra pergunta — o mal-entendido do Ruling 65.
    fn = [x for x in oks if x["n_degraus"] > 1 and x["truncado_em"] is None]
    if fn:
        e1, eT = [], []
        for x in fn:
            v = verdades[x["arquivo"]]
            k_tot = v["K_planta"] * sum(u for u, _ in v["degraus"])
            e1.append(abs(x["K_hat"] - v["K"]) / abs(v["K"]))
            eT.append(abs(x["K_hat"] - k_tot) / abs(k_tot))
        P(f"  as {len(fn)} multi-degrau NAO truncadas, contra que K elas batem:")
        P(f"    contra K do 1o degrau: |erro| mediana = {np.median(e1):.4f}")
        P(f"    contra K TOTAL (soma): |erro| mediana = {np.median(eT):.4f}")
        P(f"    mais perto do TOTAL em {sum(1 for a, b in zip(e1, eT) if b < a)}"
          f"/{len(fn)} — o ajuste descreve a excursao inteira, nao a do 1o degrau")
    P()

    P("=" * 78)
    P("5c. JANELA EFETIVA DO 1o DEGRAU (em constantes de tempo dominantes)")
    P("=" * 78)
    P("  assentar a 1% leva ~4,6 t_dom; abaixo disso o patamar nao esta na figura")
    P("  ATENCAO ao confundimento: nas figuras de 1 degrau a janela curta NAO")
    P("  estraga o ajuste (|errK| 3,5% em 0-2 t_dom); nas de 2+ degraus estraga")
    P("  (46%) — e cai para 2% quando a truncagem dispara. O que pesa e a")
    P("  truncagem, nao o tamanho da janela. Ver o corte por n_degraus abaixo.")
    faixas = [(0, 2), (2, 3), (3, 4.6), (4.6, 8), (8, 1e9)]
    P(f"  {'faixa (t_dom)':<16}{'n':>4}{'NRMSE med':>12}{'|errK| med':>12}"
      f"{'|errK|>50%':>12}{'zeta no lim':>13}")
    for lo, hi in faixas:
        g = [x for x in oks if x.get("janela_ef") is not None
             and lo <= x["janela_ef"] < hi]
        if not g:
            continue
        nr = [x["nrmse_curva"] for x in g if x.get("nrmse_curva") is not None]
        ek = [abs(x["err_K"]) for x in g if x.get("err_K") is not None]
        ruim = sum(1 for e in ek if e > 0.5)
        lim = sum(1 for x in g if x.get("zeta_no_limite"))
        rot = f"{lo:g}-{hi:g}" if hi < 1e9 else f">= {lo:g}"
        P(f"  {rot:<16}{len(g):>4}{np.median(nr):>12.4f}{np.median(ek):>12.4f}"
          f"{ruim:>8}/{len(g):<3}{lim:>9}/{len(g):<3}")
    P()
    for rotulo, sub in (("so 1 degrau", [x for x in oks if x["n_degraus"] == 1]),
                        ("2+ degraus", [x for x in oks if x["n_degraus"] > 1]),
                        ("2+ degraus truncadas",
                         [x for x in oks if x["n_degraus"] > 1
                          and x["truncado_em"] is not None])):
        P(f"  -- {rotulo}")
        for lo, hi in faixas:
            g = [x for x in sub if lo <= x["janela_ef"] < hi]
            if not g:
                continue
            rot = f"{lo:g}-{hi:g}" if hi < 1e9 else f">= {lo:g}"
            P(f"     {rot:<12} n={len(g):<3d} NRMSE med="
              f"{np.median([x['nrmse_curva'] for x in g]):7.4f}  |errK| med="
              f"{np.median([abs(x['err_K']) for x in g]):7.4f}")
    P()
    pin = [x for x in oks if x.get("zeta_no_limite")]
    P(f"  ajustes com zeta ENCOSTADO no limite da caixa (1e-3 ou 10): "
      f"{len(pin)}/{len(oks)} ({len(pin)/len(oks):.1%})")
    if pin:
        nr = [x["nrmse_curva"] for x in pin if x.get("nrmse_curva") is not None]
        ek = [abs(x["err_K"]) for x in pin if x.get("err_K") is not None]
        P(f"    entre eles: NRMSE med={np.median(nr):.4f}  |errK| med={np.median(ek):.4f}")
        livres = [x for x in oks if not x.get("zeta_no_limite")]
        nr2 = [x["nrmse_curva"] for x in livres if x.get("nrmse_curva") is not None]
        ek2 = [abs(x["err_K"]) for x in livres if x.get("err_K") is not None]
        P(f"    os demais:  NRMSE med={np.median(nr2):.4f}  |errK| med={np.median(ek2):.4f}")
    P()

    P("=" * 78)
    P("6. LATENCIA")
    P("=" * 78)
    lat = np.array([x["latency_ms"] for x in linhas if x["latency_ms"]], float)
    P(f"  mediana={np.median(lat):.0f} ms  p90={np.percentile(lat,90):.0f} ms  "
      f"max={lat.max():.0f} ms")
    P()

    P("=" * 78)
    P("7. PIORES CASOS (por NRMSE de curva)")
    P("=" * 78)
    piores = sorted([x for x in oks if x.get("nrmse_curva") is not None],
                    key=lambda z: -z["nrmse_curva"])[:12]
    P(f"  {'arquivo':<16}{'ordem v->p':<20}{'NRMSE':>9}{'errK':>9}"
      f"{'err t_dom':>11}{'dth/T':>9}  trunc")
    for x in piores:
        P(f"  {x['arquivo']:<16}"
          f"{x['order_true']+' -> '+str(x['order_hat']):<20}"
          f"{x['nrmse_curva']:>9.4f}"
          f"{(x.get('err_K') or 0):>9.3f}"
          f"{(x.get('err_t_dom') or 0):>11.3f}"
          f"{(x.get('err_theta_T') or 0):>9.3f}  "
          f"{'sim' if x['truncado_em'] is not None else 'nao'}")
    P()
    P("  RECUSADAS / sem fisica:")
    for x in linhas:
        if not x["ok"]:
            P(f"    {x['arquivo']:<16} reason={str(x['reason']):<24}"
              f" cal={str(x['cal_reason']):<22} ordem_v={x['order_true']}")

    texto = "\n".join(out)
    (BASE / "analise.txt").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    sys.exit(main())
