"""GUARDA DE EMENDA: a polilinha não atravessa um vão para outro objeto.

O defeito que ela existe para pegar, isolado em `resposta_3.png` (matplotlib
puro, um degrau, 2ª ordem sobreamortecida, figura visualmente perfeita):

  1. o Estágio A perde a cauda assentada a partir de x = 828 px — trecho
     perfeitamente reto, o `Defeito B` que `test_caso_real_rg.py` já registra;
  2. 45 colunas adiante ele acende a TRACEJADA DA ENTRADA `u(t) = 1`;
  3. `mask_to_polyline` emenda os dois, e a série ganha uma rampa de 0,80 para
     1,00 que o gráfico não desenha.

O vão tem 45 px numa curva de 823 px de largura — 5,5 %, bem abaixo do
`MAX_GAP_FRAC = 0.15`, que só olha a DISTÂNCIA HORIZONTAL. O que denuncia a
emenda é o salto VERTICAL: o segmento da direita está a ~117 px de onde a
continuação do da esquerda o colocaria.

Estes testes são sintéticos de propósito — a geometria é o enunciado, e uma
fixture PNG amarraria a guarda à máscara de hoje.
"""
import numpy as np

from identify.polyline import mask_to_polyline


def _mascara(h=300, w=400):
    return np.zeros((h, w), dtype=np.uint8)


def _traco(m, x0, x1, y0, y1, esp=3, passo=1):
    """Segmento de (x0,y0) a (x1,y1). `passo > 1` desenha tracejado."""
    for i, x in enumerate(range(x0, x1 + 1)):
        if passo > 1 and (i // 4) % passo:
            continue
        y = int(round(y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1)))
        m[y:y + esp, x] = 255
    return m


def test_nao_emenda_curva_com_reta_de_referencia():
    """O caso real: platô em y=200, vão de 45 px, tracejada da entrada em y=40."""
    m = _traco(_mascara(), 20, 200, 200, 200)
    m = _traco(m, 245, 330, 40, 40)
    xs, ys = mask_to_polyline(m)
    assert xs.size >= 10, "a guarda não pode zerar a polilinha"
    assert ys.max() < 60 or ys.min() > 180, "a polilinha misturou os dois objetos"
    assert abs(np.median(ys) - 200) < 5, (
        f"o segmento mantido deveria ser o MAIOR (o platô em y=200), "
        f"e a mediana saiu {np.median(ys):.1f}")
    assert xs.max() <= 205, f"a polilinha passou do fim do platô (x={xs.max()})"


def test_curva_tracejada_ingreme_continua_inteira():
    """Guarda de custo: subida íngreme desenhada TRACEJADA não pode ser cortada.

    É o caso que separa "vão entre dois objetos" de "vão dentro do mesmo
    traço": aqui o salto vertical através de cada vão é grande, mas é
    exatamente o que a inclinação local prevê.
    """
    m = _traco(_mascara(), 20, 300, 260, 60, passo=2)
    xs, ys = mask_to_polyline(m)
    assert xs.size >= 10
    assert xs.max() - xs.min() >= 250, (
        f"a guarda cortou uma curva legítima: span {xs.max() - xs.min():.0f} px "
        f"de 280 desenhados")
    assert ys.min() < 80 and ys.max() > 240, "a curva perdeu uma das pontas"


def test_curva_solida_sem_vao_e_intocada():
    """Sem vão não há emenda para julgar: o caminho tem de ser o de antes."""
    m = _traco(_mascara(), 20, 350, 250, 80)
    xs, ys = mask_to_polyline(m)
    assert xs.min() == 20 and xs.max() == 350
    assert ys.min() < 90 and ys.max() > 240


def test_nao_corta_a_cabeca_mesmo_quando_ela_e_o_segmento_menor():
    """A guarda só tira CAUDA. Ver a justificativa em `mask_to_polyline`.

    Geometria: um trecho CURTO em y=60 (x=20..70), vão, e um trecho LONGO em
    y=200 (x=110..380). "Fique com o maior segmento" jogaria a cabeça fora; a
    restrição manda a guarda não mexer, porque a cabeça carrega o repouso e a
    partida da resposta — é dela que saem `theta`, o sinal do degrau e a
    detecção de `resposta_inversa`.

    Medido em `data/val_nmp[:60]` com a regra irrestrita: cabeça cortada em 21
    das 60, cauda em nenhuma, e a recusa por `resposta_inversa` caindo de 90 %
    para 58,3 %.
    """
    m = _traco(_mascara(w=420), 20, 70, 60, 60)
    m = _traco(m, 110, 380, 200, 200)
    xs, ys = mask_to_polyline(m)
    assert xs.min() <= 21, (
        f"a guarda cortou a cabeça: polilinha começa em x={xs.min():.0f}, "
        f"e o desenho começa em x=20")
    assert ys.min() < 70, "o trecho da cabeça (y=60) sumiu da polilinha"
