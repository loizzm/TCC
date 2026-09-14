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
            "linestyle": v["linestyle"],
            "preenchimento": v["preenchimento"], "fundo_escuro": v["fundo_escuro"],
            "legenda_loc": v["legenda_loc"], "dpi": v["dpi"],
            "order_true": v["order"], "order_hat": ordem_hat,
            "respondeu": bool(ordem_hat), "ok": bool(s.get("ok")),
            "reason": s.get("reason"),
            "cal_ok": bool(s["calibration"]["ok"]),
            "cal_reason": s["calibration"]["reason"],
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

            # --- nivel 3: erro de CURVA na janela inteira
            # Era o trecho ate o 2o degrau enquanto a frente de multi-degrau
            # existia (§68); sem ela, a janela desenhada e a janela valida.
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
            # JANELA EFETIVA, em constantes de tempo dominantes. Assentar a
            # 1 % leva ~4,6 t_dom; abaixo disso o patamar nao aparece e K
            # deixa de ser observavel.
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
        if not sub:
            # Lote sem esta familia. Acontece desde que o multi-degrau saiu do
            # gerador (ver MULTI_DEGRAU.md): `gera_lote_ruidoso.py` nao produz
            # amostra `multi` nenhuma, e a divisao por zero derrubava o
            # relatorio inteiro DEPOIS de `analise.json` ja ter sido escrito —
            # entao o numero conjuntivo saia e o texto nao.
            continue
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
    estrato("preenchimento", "preenchimento")
    estrato("fundo_escuro", "fundo escuro")
    estrato("order_true", "ordem verdadeira")

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
      f"{'err t_dom':>11}{'dth/T':>9}")
    for x in piores:
        P(f"  {x['arquivo']:<16}"
          f"{x['order_true']+' -> '+str(x['order_hat']):<20}"
          f"{x['nrmse_curva']:>9.4f}"
          f"{(x.get('err_K') or 0):>9.3f}"
          f"{(x.get('err_t_dom') or 0):>11.3f}"
          f"{(x.get('err_theta_T') or 0):>9.3f}")
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
