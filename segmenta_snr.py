"""Acerto conjuntivo do lote ruidoso SEGMENTADO POR SNR.

O numero agregado do lote (64 % ESTRITO / 75 % PRATICO) mistura cinco
populacoes com relacao sinal-ruido muito diferente, e a media esconde
exatamente o que interessa: ONDE fica o joelho. O corpus de treino base sorteia
`snr_db` em U(20, 60) e nunca desce de 20 dB — se a queda for um joelho em 20 e
nao uma rampa, a causa e a borda da distribuicao de treino, nao "ruido" em
geral.

Alem do acerto, reporta o CRITERIO BINDING por faixa: o agregado ja mostrou que
o gargalo no ruidoso e `t_dom`, mas a pergunta e se ele domina em toda a faixa
ou so na ponta suja.

Uso:
    .venv/bin/python segmenta_snr.py [--dir reports/amostras_aleatorias/lote_ruido]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from acerto_conjuntivo import NIVEIS, avalia

RAIZ = Path(__file__).resolve().parent


def snr_do_nome(nome: str) -> int | None:
    m = re.search(r"_(\d+)dB_", nome)
    return int(m.group(1)) if m else None


def ganho_do_nome(nome: str) -> str:
    if "Kmaior1" in nome:
        return "|K|>1"
    if "Kmenor1" in nome:
        return "|K|<1"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=RAIZ / "reports/amostras_aleatorias/lote_ruido")
    a = ap.parse_args()
    A = json.loads((a.dir / "analise.json").read_text())
    for x in A:
        x["_snr"] = snr_do_nome(x["arquivo"])
        x["_K"] = ganho_do_nome(x["arquivo"])
    faixas = sorted({x["_snr"] for x in A if x["_snr"] is not None}, reverse=True)

    print(f"ACERTO CONJUNTIVO POR SNR — {a.dir.name}  n={len(A)}\n")
    print(f"  {'SNR':>7}{'n':>5}" + "".join(f"{k:>12}" for k in NIVEIS)
          + f"{'entregou':>11}")
    for s in faixas:
        sub = [x for x in A if x["_snr"] == s]
        linha = f"  {str(s)+' dB':>7}{len(sub):>5}"
        for nivel in NIVEIS:
            c = NIVEIS[nivel]
            bons = sum(1 for x in sub if not avalia(x, c))
            linha += f"{bons/len(sub):>11.0%} "
        ent = sum(1 for x in sub if x["ok"])
        print(linha + f"{ent/len(sub):>10.0%}")
    sub = A
    linha = f"  {'TODAS':>7}{len(sub):>5}"
    for nivel in NIVEIS:
        bons = sum(1 for x in sub if not avalia(x, NIVEIS[nivel]))
        linha += f"{bons/len(sub):>11.0%} "
    print(linha + f"{sum(1 for x in sub if x['ok'])/len(sub):>10.0%}")

    print("\n  O treino base para em 20 dB. Um JOELHO ali acusa a borda da")
    print("  distribuicao; uma rampa suave acusa o ruido em si.\n")

    print("  POR SINAL DO GANHO (nivel PRATICO)")
    print(f"  {'SNR':>7}" + "".join(f"{k:>10}" for k in ("|K|>1", "|K|<1")))
    for s in faixas:
        linha = f"  {str(s)+' dB':>7}"
        for g in ("|K|>1", "|K|<1"):
            sub = [x for x in A if x["_snr"] == s and x["_K"] == g]
            if not sub:
                linha += f"{'-':>10}"
                continue
            bons = sum(1 for x in sub if not avalia(x, NIVEIS["PRATICO"]))
            linha += f"{bons/len(sub):>9.0%} "
        print(linha)

    print("\n  CRITERIO BINDING por faixa (nivel PRATICO, cada reprovacao conta)")
    c = NIVEIS["PRATICO"]
    chaves: list[str] = []
    tab = {}
    for s in faixas:
        sub = [x for x in A if x["_snr"] == s]
        cont = Counter()
        for x in sub:
            for k in avalia(x, c):
                k = k.split(" (")[0]
                cont[k] += 1
                if k not in chaves:
                    chaves.append(k)
        tab[s] = (cont, len(sub))
    chaves.sort(key=lambda k: -sum(tab[s][0].get(k, 0) for s in faixas))
    print(f"  {'SNR':>7}" + "".join(f"{k:>12}" for k in chaves))
    for s in faixas:
        cont, n = tab[s]
        print(f"  {str(s)+' dB':>7}"
              + "".join(f"{cont.get(k,0):>6}/{n:<5}" for k in chaves))
    print("\n    'reprova' conta cada criterio reprovado; uma figura pode")
    print("    aparecer em mais de uma coluna.")

    # DEGRAU OU RAMPA. Com n=20 por faixa o erro padrao de cada celula passa de
    # 10 pp, entao ler a tabela linha a linha convida a ver tendencia onde ha
    # ruido amostral. As duas comparacoes abaixo respondem a pergunta que
    # importa com a amostra agrupada: existe um salto na BORDA do treino
    # (20 dB), e existe degradacao ADICIONAL abaixo dela?
    from scipy.stats import fisher_exact
    print("\n  DEGRAU NA BORDA DO TREINO? (Fisher exato)")
    for nivel in ("ESTRITO", "PRATICO"):
        c = NIVEIS[nivel]
        alto = [x for x in A if (x["_snr"] or 0) >= 25]
        baixo = [x for x in A if (x["_snr"] or 0) <= 20]
        if not alto or not baixo:
            continue
        a = sum(1 for x in alto if not avalia(x, c))
        b = sum(1 for x in baixo if not avalia(x, c))
        orr, p = fisher_exact([[a, len(alto) - a], [b, len(baixo) - b]])
        print(f"    {nivel:<9} >=25 dB {a}/{len(alto)} = {a/len(alto):.0%}"
              f"   <=20 dB {b}/{len(baixo)} = {b/len(baixo):.0%}"
              f"   OR={orr:.2f}  p={p:.4f}")
        g20 = [x for x in A if x["_snr"] == 20]
        g10 = [x for x in A if x["_snr"] == 10]
        if g20 and g10:
            a2 = sum(1 for x in g20 if not avalia(x, c))
            b2 = sum(1 for x in g10 if not avalia(x, c))
            orr2, p2 = fisher_exact([[a2, len(g20) - a2], [b2, len(g10) - b2]])
            print(f"    {'':<9} 20 dB {a2}/{len(g20)} x 10 dB {b2}/{len(g10)}"
                  f"        OR={orr2:.2f}  p={p2:.3f}   (piora ALEM da borda?)")


if __name__ == "__main__":
    main()
