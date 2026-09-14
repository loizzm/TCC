"""SONDAGEM: da para achar a linha de entrada na IMAGEM, sem retreinar a rede?

A pergunta que isto responde. O detector de 2o degrau mais robusto seria ler o
DEGRAU DE ENTRADA que as tres familias desenham no mesmo quadro — a contagem de
degraus esta literalmente tracada. Medido antes desta sondagem, a mascara da
U-Net NAO serve para isso: ela foi treinada (§41) para SUPRIMIR a tracejada e em
geral consegue — acende em so 7,9 % das colunas da linha de entrada (mediana,
n=297). Ou seja, a informacao nao esta na mascara; separar os dois objetos
exigiria um segundo canal de saida e, com ele, um corpus novo (o gerador de
treino desenha distratores como `axhline`/`axvline`, retas isoladas, e NUNCA
uma escada) e um retreino.

Antes de pagar esse preco, esta sondagem testa a hipotese barata: a linha de
entrada e CONSTANTE POR PARTES e de cor distinta, entao talvez saia da imagem
RGB por visao computacional classica, sem rede nenhuma.

O criterio e estrutural, nao de cor: `u(t)` assume exatamente `n_degraus + 1`
valores distintos. Se a mediana por coluna de um objeto e explicada por poucos
NIVEIS PLANOS, o objeto e uma escada, e o numero de niveis menos um e o numero
de degraus. Uma curva de resposta nao passa nesse teste — ela varre valores
continuamente entre o repouso e o patamar.

Saida: `sondagem_entrada.txt` e `sondagem_entrada.json`.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from identify.calibrate import calibrate

RAIZ = Path(__file__).resolve().parent / "reports" / "amostras_aleatorias"
LOTES = (RAIZ, RAIZ / "balanceado", RAIZ / "balanceado2")

COBERTURA_MIN = 0.50     # fracao da largura que o objeto tem de ocupar
ESPESSURA_MAX = 10       # px por coluna: acima disso e preenchimento, nao linha
NIVEIS_MAX = 5           # 4 degraus ja e mais do que qualquer figura tem
FRAC_NIVEL_MIN = 0.04    # um nivel tem de segurar 4 % das colunas para contar
QUALIDADE_MIN = 0.90     # fracao das colunas explicada pelos niveis planos


def _objetos_por_cor(img, bbox, margem=3):
    """Mascaras candidatas: uma por cor dominante dentro da moldura.

    Quantiza em baldes de 32 para colapsar o antialiasing, e descarta o fundo
    (o balde mais frequente) e tudo que for raro demais para ser uma linha.
    """
    x0, y0, x1, y1 = (int(v) for v in bbox)
    sub = img[y0 + margem:y1 - margem + 1, x0 + margem:x1 - margem + 1]
    if sub.size == 0:
        return [], sub.shape[:2]
    q = (sub.astype(np.int32) // 32)
    chave = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    cont = Counter(chave.ravel())
    fundo = cont.most_common(1)[0][0]
    h, w = chave.shape
    saida = []
    for cor, n in cont.most_common(12):
        if cor == fundo or n < 0.3 * w:      # raro demais para cruzar o quadro
            continue
        saida.append((cor, chave == cor))
    return saida, (h, w)


def _escada(m, larg_trans_max=0.02):
    """(n_niveis, qualidade, cobertura, nitidez) do objeto, ou None.

    `nitidez` e o que separa a ENTRADA da RESPOSTA, e sem ela a sondagem nao
    funciona: as duas sao "constantes por partes" no fim das contas — uma
    resposta assentada tambem passa a maior parte das colunas num patamar. O
    que so a entrada tem e a TRANSICAO VERTICAL: ela troca de nivel em uma ou
    duas colunas, enquanto a resposta leva uma constante de tempo inteira. Sem
    este criterio a sondagem escolhia a curva de resposta e errava a contagem
    para mais (medido: 30 figuras de 1 degrau lidas como 2, 24 de 2 lidas
    como 3).
    """
    h, w = m.shape
    cols = np.flatnonzero(m.any(axis=0))
    if cols.size < COBERTURA_MIN * w:
        return None
    esp = float(np.median(m.sum(axis=0)[cols]))
    if esp > ESPESSURA_MAX or esp <= 0:
        return None
    yc = np.array([float(np.median(np.flatnonzero(m[:, c]))) for c in cols])

    # Tolerancia de "mesmo nivel": a propria espessura do traco, com um piso.
    tol = max(2.0 * esp, 3.0)
    ordem = np.argsort(yc)
    ys = yc[ordem]
    # agrupamento 1-D por lacuna: quebra onde o salto passa da tolerancia
    grupos, atual = [], [ys[0]]
    for a, b in zip(ys, ys[1:]):
        if b - a > tol:
            grupos.append(atual); atual = []
        atual.append(b)
    grupos.append(atual)
    niveis = [g for g in grupos if len(g) >= FRAC_NIVEL_MIN * cols.size]
    if not niveis or len(niveis) > NIVEIS_MAX:
        return None
    centros = np.array([np.median(g) for g in niveis])

    # NIVEIS DISJUNTOS EM X. Uma escada percorre os niveis em SEQUENCIA: cada
    # um ocupa um intervalo de colunas, e os intervalos nao se cruzam. A
    # AMOSTRA DE LINHA DA LEGENDA quebra exatamente isso — ela tem a mesma cor
    # da entrada, fica num y proprio e no MEIO do intervalo da linha
    # principal, entao entra como um nivel a mais. Era a causa do vies de +1
    # medido antes deste criterio (33 figuras de 1 degrau lidas como 2, 27 de
    # 2 lidas como 3). Um nivel cujo intervalo de x esta contido no de outro
    # nao e degrau: e legenda, e sai.
    atrib = np.abs(yc[:, None] - centros[None, :]).argmin(axis=1)
    faixas = []
    for i in range(len(centros)):
        xi = cols[atrib == i]
        if xi.size:
            faixas.append((i, float(xi.min()), float(xi.max()), xi.size))
    manter = []
    for i, lo_i, hi_i, n_i in faixas:
        contido = any(
            lo_j <= lo_i and hi_i <= hi_j and n_j > n_i
            for j, lo_j, hi_j, n_j in faixas if j != i)
        if not contido:
            manter.append(i)
    if not manter:
        return None
    centros = centros[sorted(manter)]
    niveis = [niveis[i] for i in sorted(manter)]
    dist = np.abs(yc[:, None] - centros[None, :]).min(axis=1)
    dentro = dist <= tol
    qualidade = float(dentro.mean())

    # NITIDEZ: as colunas FORA de qualquer nivel sao a transicao. Numa escada
    # elas sao poucas e agrupadas; numa resposta sao o transitorio inteiro.
    # Mede-se o maior bloco contiguo de colunas fora de nivel, em fracao da
    # largura — uma resposta lenta tem um bloco enorme, um degrau nao tem
    # quase nenhum.
    fora = ~dentro
    maior = atual_n = 0
    for f in fora:
        atual_n = atual_n + 1 if f else 0
        maior = max(maior, atual_n)
    nitidez = 1.0 - maior / max(cols.size, 1)
    if (1.0 - nitidez) > larg_trans_max:
        return None
    return len(niveis), qualidade, float(cols.size / w), nitidez


def sonda(caminho: Path):
    img = np.asarray(Image.open(caminho).convert("RGB"))
    cal = calibrate(img)
    bbox = cal.bbox_px if any(cal.bbox_px) else (0, 0, img.shape[1] - 1, img.shape[0] - 1)
    objs, _ = _objetos_por_cor(img, bbox)
    cands = []
    for cor, m in objs:
        r = _escada(m)
        if r is None:
            continue
        n, q, cob, nit = r
        # Uma entrada tem PELO MENOS dois niveis (repouso e o pos-degrau); um
        # objeto de nivel unico e uma reta de grade ou de referencia.
        if q >= QUALIDADE_MIN and n >= 2:
            cands.append({"cor": int(cor), "niveis": n, "qualidade": q,
                          "cobertura": cob, "nitidez": nit})
    if not cands:
        return None, cands
    # Melhor escada: a de maior qualidade; empate resolve por cobertura.
    # Mais nitida primeiro: entre dois objetos constantes por partes, o que
    # troca de nivel mais abruptamente e a entrada.
    melhor = max(cands, key=lambda c: (round(c["nitidez"], 3),
                                       round(c["qualidade"], 3), c["cobertura"]))
    return melhor, cands


def main() -> None:
    linhas = []
    for d in LOTES:
        ver = {v["arquivo"]: v for v in json.loads((d / "verdade.json").read_text())}
        for nome, v in ver.items():
            p = d / nome
            if not p.exists():
                continue
            melhor, cands = sonda(p)
            linhas.append({
                "arquivo": f"{d.name}/{nome}", "familia": v["familia"],
                "n_degraus": v["n_degraus"], "fundo_escuro": v["fundo_escuro"],
                "achou": melhor is not None,
                "niveis": None if melhor is None else melhor["niveis"],
                "n_hat": None if melhor is None else melhor["niveis"] - 1,
                "qualidade": None if melhor is None else melhor["qualidade"],
                "nitidez": None if melhor is None else melhor["nitidez"],
                "cobertura": None if melhor is None else melhor["cobertura"],
                "n_candidatos": len(cands),
            })
    (RAIZ / "sondagem_entrada.json").write_text(
        json.dumps(linhas, indent=2), encoding="utf-8")

    L = []
    def P(s=""):
        L.append(s)

    n = len(linhas)
    achou = [x for x in linhas if x["achou"]]
    P(f"SONDAGEM DA LINHA DE ENTRADA — deteccao classica, sem rede   n={n}")
    P("=" * 74)
    P(f"  achou alguma escada: {len(achou)}/{n} ({len(achou)/n:.1%})")
    P()
    P("  CONTAGEM DE DEGRAUS (niveis - 1) contra a verdade:")
    P(f"    {'verdade':>9}{'n':>5}{'achou':>8}{'exato':>8}{'>=2 quando e >=2':>19}")
    for nd in sorted({x["n_degraus"] for x in linhas}):
        g = [x for x in linhas if x["n_degraus"] == nd]
        a = [x for x in g if x["achou"]]
        ex = sum(1 for x in a if x["n_hat"] == nd)
        mm = sum(1 for x in a if (x["n_hat"] >= 2) == (nd >= 2))
        P(f"    {nd:>9}{len(g):>5}{len(a):>8}{ex:>8}{mm:>19}")
    P()
    ex = sum(1 for x in achou if x["n_hat"] == x["n_degraus"])
    P(f"  contagem exata: {ex}/{len(achou)} das detectadas ({ex/max(1,len(achou)):.1%}),"
      f" {ex}/{n} do total ({ex/n:.1%})")
    P()

    P("  COMO DETECTOR DE MULTI-DEGRAU (a comparacao que importa):")
    TP = sum(1 for x in linhas if x["n_degraus"] > 1 and x["achou"] and x["n_hat"] > 1)
    FN = sum(1 for x in linhas if x["n_degraus"] > 1 and not (x["achou"] and x["n_hat"] > 1))
    FP = sum(1 for x in linhas if x["n_degraus"] == 1 and x["achou"] and x["n_hat"] > 1)
    TN = sum(1 for x in linhas if x["n_degraus"] == 1 and not (x["achou"] and x["n_hat"] > 1))
    pr = TP / max(1, TP + FP); rv = TP / max(1, TP + FN)
    P(f"    TP={TP} FN={FN} FP={FP} TN={TN}")
    P(f"    precisao={pr:.1%}  revocacao={rv:.1%}  F1={2*pr*rv/max(1e-9,pr+rv):.3f}")
    P(f"    referencia (gate de ganho, sem piso): precisao 78,2 %  revocacao 70,3 %  F1 0,740")
    P(f"    referencia (producao, piso 0,030):    precisao 70,5 %  revocacao 46,8 %  F1 0,563")
    P()
    P("  por familia e por tema:")
    for chave, rot in (("familia", "familia"), ("fundo_escuro", "fundo escuro")):
        for k in sorted({x[chave] for x in linhas}, key=str):
            g = [x for x in linhas if x[chave] == k]
            a = [x for x in g if x["achou"]]
            ex2 = sum(1 for x in a if x["n_hat"] == x["n_degraus"])
            P(f"    {rot}={str(k):<8} n={len(g):<4d} achou={len(a)/len(g):6.1%}"
              f"  contagem exata={ex2}/{len(g)} ({ex2/len(g):5.1%})")
    P()
    P("  onde erra, quando acha:")
    err = Counter((x["n_degraus"], x["n_hat"]) for x in achou if x["n_hat"] != x["n_degraus"])
    for (v_, h_), c in err.most_common(8):
        P(f"    verdade {v_} -> detectou {h_}:  {c}")

    txt = "\n".join(L)
    (RAIZ / "sondagem_entrada.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
