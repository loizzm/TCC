"""O começo da polilinha ainda é o REPOUSO? — a invariante que a review pediu.

Motivo de existir. A Task 4 do bloco do caso real entregou DOIS remédios que
supunham coisas incompatíveis e nenhum dos dois verificava a sua:

  * `_nivel_de_repouso` lê as `_N_REPOUSO` PRIMEIRAS colunas observadas e supõe
    a curva PARADA ali;
  * `_COBERTURA_MIN_MOLDURA` dispara em QUALQUER truncagem, porque a condição é
    a cobertura TOTAL (`span_observado / span_moldura`), que não diz nada sobre
    o começo da série.

Numa truncagem à ESQUERDA — a U-Net perdendo o trecho plano inicial, plausível
porque um trecho horizontal colado ao eixo ou a uma linha de grade é justamente
onde a rede perde tinta — a origem passava a vir de `bbox_px[0]`, correta, mas as
primeiras colunas observadas já estavam NA SUBIDA. Isto é o defeito de 12,6 % em
ζ que a Task 4 acabou de corrigir, reintroduzido pelo outro remédio da MESMA
task. A docstring de `identify/pipeline.py` chegou a escrever a suposição
("quando a truncagem é à DIREITA, o regime do caso real") sem que o código a
asseverasse.

HISTÓRICO IMPORTANTE (HANDOFF_P2_7 §35.9.1). A primeira versão desta guarda
condicionava por um PROXY geométrico — `(x[0] − bbox[0]) / largura ≤ 0,15` — e o
limiar estava no lugar errado da curva de dano: admitia 68 % de erro em ζ logo
antes dele. O proxy foi trocado pelo invariante DIRETO: **planura** das
`_N_REPOUSO` primeiras colunas. Estes testes acompanham a troca.

O defeito é GEOMÉTRICO, não estatístico: não precisa de rede nem de `data/`.
As máscaras aqui são desenhadas por aritmética, determinísticas, e passam pelo
`mask_to_polyline` de produção antes de chegar ao `_serie_normalizada`.

Números que ancoram o limiar (medidos, populações escritas):
  * caso real (n=1): `bbox=(75,39,821,503)`, `x[0]=76`, `x[-1]=536`, cobertura
    0,617, **planura 0,0039**.
  * corpus, 300 primeiras de `data/test` com a cadeia de produção (n=299):
    cobertura mínima 0,8388 — o ramo NUNCA dispara; planura mediana 0,0044,
    p95 0,0316, p99 0,0485, máximo 0,0722.
  * curva de dano, 82 séries determinísticas: planura 0,0064 -> 0,08 % de erro
    em ζ; 0,0423 -> 4,25 %; 0,0613 -> 13,09 %; 0,0860 -> 27,34 %;
    0,1171 -> 45,44 %; 0,1587 -> 67,82 %.
"""
from __future__ import annotations

import numpy as np

from identify.pipeline import (_COBERTURA_MIN_MOLDURA, _N_REPOUSO,
                               _PLANURA_MAX_FRAC, _serie_normalizada)
from identify.polyline import mask_to_polyline

# Moldura do caso real, para que a aritmética dos testes seja a mesma da
# fixture que motivou a correção.
BBOX = (75, 39, 821, 503)
SPAN_MOLDURA = float(BBOX[2] - BBOX[0])
MARGEM = 0.04          # margem do matplotlib de cada lado, dentro da moldura

# Planura medida na fixture real (n=1), com a cadeia de produção. Guardada como
# constante para que o piso do limiar seja testável sem carregar a U-Net.
PLANURA_CASO_REAL = 0.0039


def _resposta_2a_ordem(zeta: float, wn: float, T: float, n: int = 2000):
    t = np.linspace(0.0, T, n)
    wd = wn * np.sqrt(1.0 - zeta ** 2)
    y = 1.0 - np.exp(-zeta * wn * t) * (
        np.cos(wd * t) + zeta / np.sqrt(1.0 - zeta ** 2) * np.sin(wd * t))
    return t, y


def _mascara(zeta: float = 0.5, wn: float = 2.0, T: float = 10.0,
             corte_esq: float = 0.0, corte_dir: float = 0.0) -> np.ndarray:
    """Máscara 0/255 com uma resposta de 2ª ordem desenhada dentro de `BBOX`.

    `corte_esq`/`corte_dir` são as frações da JANELA DE DADOS apagadas em cada
    ponta — é assim que se simula a rede perdendo tinta de um lado só, sem
    mexer na moldura (que continua sendo o referencial correto).
    """
    x0, y0, x1, y1 = BBOX
    mask = np.zeros((y1 + 60, x1 + 60), dtype=np.uint8)
    largura = x1 - x0
    xa = x0 + MARGEM * largura
    xb = x1 - MARGEM * largura
    t, y = _resposta_2a_ordem(zeta, wn, T)
    xp = xa + (t - t[0]) / (t[-1] - t[0]) * (xb - xa)
    ya = y1 - 0.08 * (y1 - y0)
    yb = y0 + 0.08 * (y1 - y0)
    yp = ya + (y - y.min()) / (y.max() - y.min()) * (yb - ya)   # pixel p/ baixo
    manter = ((xp >= xa + corte_esq * (xb - xa))
              & (xp <= xb - corte_dir * (xb - xa)))
    for cx, cy in zip(xp[manter], yp[manter]):
        c, r = int(round(cx)), int(round(cy))
        mask[r - 1:r + 2, c] = 255        # traço de 3 px, sólido
    return mask


def _polilinha(**kw):
    x, y = mask_to_polyline(_mascara(**kw), bbox=BBOX)
    assert x.size >= 10, "máscara sintética degenerada — o teste não mede nada"
    k = np.argsort(x)
    return np.asarray(x, dtype=float)[k], np.asarray(y, dtype=float)[k]


def _cobertura(x: np.ndarray) -> float:
    return float(x[-1] - x[0]) / SPAN_MOLDURA


def _falta_esq(x: np.ndarray) -> float:
    return (float(x[0]) - float(BBOX[0])) / SPAN_MOLDURA


def _planura(y: np.ndarray) -> float:
    """A MESMA grandeza que `_serie_normalizada` calcula, reimplementada aqui.

    Reimplementada de propósito: se a produção mudar a definição, estes testes
    têm de acusar em vez de acompanhar em silêncio.
    """
    n = int(min(_N_REPOUSO, y.size))
    faixa = float(np.ptp(y))
    return float(np.ptp(y[:max(1, n)])) / faixa if faixa > 0 else 0.0


def test_serie_intacta_ancora_na_moldura():
    """Sem truncagem a âncora TAMBÉM é a moldura — mudança deliberada.

    Até a promoção do modelo RGB, a moldura só entrava abaixo de
    `_COBERTURA_MIN_MOLDURA`, e sem truncagem `t` ia de 0 a 1 sobre a extensão
    observada. Duas medições derrubaram esse desenho:

    1. Recuperando ωₙ FÍSICO com `T = sx * largura_da_moldura` (75 amostras de
       2ª ordem sem calibração), a extensão observada dá MAPE 13,51 % e mediana
       8,08 %; a moldura dá 9,81 % e 1,80 %. No caso real `resposta_degrau.png`
       o erro cai de 2,8 % para 0,3 %.
    2. O gatilho por cobertura deixou de discriminar: com a máscara RGB as duas
       imagens reais cobrem 0,9638 e 0,9775, ACIMA da mediana do corpus
       (0,9325). Nenhum limiar as separa das amostras normais.

    O que sobra do `_COBERTURA_MIN_MOLDURA` é o portão da guarda de PLANURA
    (proteger `_nivel_de_repouso` sob truncagem), não a escolha da âncora.
    """
    x, y = _polilinha()
    assert _cobertura(x) >= _COBERTURA_MIN_MOLDURA, (
        f"máscara sem truncagem com cobertura {_cobertura(x):.4f} — o teste "
        "perdeu o regime que queria exercitar")
    t, _ = _serie_normalizada(x, y, bbox_px=BBOX)
    assert t is not None
    # origem e escala vêm as duas da moldura: o primeiro ponto cai sobre a
    # falta à esquerda e o último sobre falta + cobertura.
    assert np.isclose(t[0], _falta_esq(x)), (
        f"origem não veio da moldura: t[0]={t[0]!r}, "
        f"falta à esquerda={_falta_esq(x)!r}")
    assert np.isclose(t[-1], _falta_esq(x) + _cobertura(x)), (
        f"escala não veio da moldura: t[-1]={t[-1]!r}, esperado "
        f"{_falta_esq(x) + _cobertura(x)!r}")


def test_truncagem_a_direita_troca_pela_moldura():
    """O REGIME DO CASO REAL: falta tinta só à direita, o começo da curva
    continua lá (colunas iniciais PLANAS) -> escala e origem da moldura."""
    x, y = _polilinha(corte_dir=0.40)
    assert _cobertura(x) < _COBERTURA_MIN_MOLDURA
    assert _planura(y) <= _PLANURA_MAX_FRAC, (
        f"planura {_planura(y):.4f} — o teste perdeu o regime aceito")
    t, yn = _serie_normalizada(x, y, bbox_px=BBOX)
    assert t is not None and yn is not None, (
        "truncagem à direita é o caso que a correção da Task 4 serve — recusar "
        "aqui quebraria o caso real")
    # origem = bbox_px[0] e escala = largura da moldura, os dois no MESMO
    # referencial: o primeiro ponto cai exatamente sobre a falta à esquerda.
    assert np.isclose(t[0], _falta_esq(x))
    assert np.isclose(t[-1], _falta_esq(x) + _cobertura(x))


def test_truncagem_a_esquerda_recusa_a_serie():
    """A INVARIANTE (C3). A polilinha começa NA SUBIDA: o nível de repouso é
    inválido, e a série é RECUSADA em vez de virar um número que ninguém
    distingue de um certo.

    É este teste que falha contra o código anterior à correção, que olhava só a
    cobertura total e trocava escala e origem do mesmo jeito.
    """
    x, y = _polilinha(corte_esq=0.10, corte_dir=0.35)
    assert _cobertura(x) < _COBERTURA_MIN_MOLDURA, (
        f"cobertura {_cobertura(x):.4f} não entra no ramo — teste sem poder")
    assert _planura(y) > _PLANURA_MAX_FRAC, (
        f"planura {_planura(y):.4f} não caracteriza começo na subida")
    t, yn = _serie_normalizada(x, y, bbox_px=BBOX)
    assert t is None and yn is None, (
        "polilinha começando na subida produziu série adimensional: o nível de "
        "repouso vem das primeiras colunas OBSERVADAS. Neste ponto exato da "
        "curva de dano o erro de ζ medido é 67,8 % — cinco vezes o defeito de "
        "12,6 % que a Task 4 corrigiu.")


def test_truncagem_simetrica_recusa_a_serie():
    """Faltando dos DOIS lados vale o mesmo argumento: sem patamar inicial não
    há nível de repouso, e a cobertura total não sabia disso."""
    x, y = _polilinha(corte_esq=0.06, corte_dir=0.35)
    assert _cobertura(x) < _COBERTURA_MIN_MOLDURA
    assert _planura(y) > _PLANURA_MAX_FRAC
    t, yn = _serie_normalizada(x, y, bbox_px=BBOX)
    assert t is None and yn is None, (
        "truncagem simétrica produziu série adimensional apesar de a curva já "
        "estar subindo na primeira coluna observada")


def test_caso_real_continua_no_ramo_da_moldura():
    """Guarda dos números da fixture real (n=1), sem carregar a imagem.

    `bbox=(75,39,821,503)`, `x[0]=76`, `x[-1]=536`: cobertura 0,617 (dispara o
    ramo) e planura 0,0039 (bem dentro do limiar). Se alguém apertar
    `_PLANURA_MAX_FRAC` a ponto de pegar o caso real, é aqui que aparece — e
    barato, sem U-Net.
    """
    cobertura = (536.0 - 76.0) / SPAN_MOLDURA
    assert cobertura < _COBERTURA_MIN_MOLDURA, f"cobertura real {cobertura:.4f}"
    assert PLANURA_CASO_REAL <= _PLANURA_MAX_FRAC, (
        f"planura do caso real {PLANURA_CASO_REAL} acima do limiar "
        f"{_PLANURA_MAX_FRAC} — a única amostra que usa este ramo seria "
        "recusada")


def test_limiar_de_planura_esta_entre_as_duas_populacoes_medidas():
    """O limiar tem de ficar ENTRE as duas populações, e as duas foram medidas.

    Este teste substitui `test_limiar_da_esquerda_acima_da_margem_do_matplotlib`,
    cuja premissa não se sustentava: ele exigia que o limiar do proxy ficasse
    acima do deslocamento máximo de MARGEM (0,1227) medido num corpus cuja
    cobertura mínima é 0,8388 — ou seja, numa população que nunca entra neste
    ramo. Protegia contra um falso positivo impossível ali, e deixava passar
    68 % de erro em ζ. Ver HANDOFF_P2_7 §35.9.1.

    Os dois lados agora são medidos na grandeza que o código de fato usa:

    * PISO — a planura não é zero nem sem truncagem (com θ = 0 e ωₙ alto a
      curva já se move dentro de 5 colunas). Corpus, n=299: mediana 0,0044,
      p95 0,0316. O limiar não pode recusar a movimentação normal de início de
      curva, nem o caso real (0,0039).
    * TETO — na curva de dano medida (82 séries), planura 0,0613 já custa
      13,09 % de erro em ζ, acima do defeito de 12,6 % que a Task 4 corrigiu.
    """
    assert _PLANURA_MAX_FRAC >= 4.0 * PLANURA_CASO_REAL, (
        f"limiar {_PLANURA_MAX_FRAC} sem folga sobre o caso real "
        f"({PLANURA_CASO_REAL}) — a única amostra que usa o ramo")
    assert _PLANURA_MAX_FRAC >= 0.0044, (
        "limiar abaixo da planura MEDIANA do corpus (0,0044, n=299): recusaria "
        "séries pela movimentação normal de início de curva")
    assert _PLANURA_MAX_FRAC < 0.0613, (
        "limiar acima do ponto em que o dano medido (13,09 % de erro em ζ) já "
        "passa o defeito de 12,6 % que a Task 4 corrigiu")
