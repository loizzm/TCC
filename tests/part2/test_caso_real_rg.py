"""Regressão dos TRÊS sistemas do `rg.py` (HANDOFF_P2_7 §39, caso real do Bloco 8).

Os três foram produzidos FORA do gerador do projeto, por `rg.py`, com a verdade
declarada na própria função de transferência — não há estimativa envolvida:

    Sistema 1 (`caso_real_rg_fopdt_atraso.png`)
        signal.TransferFunction([5], [1, 1]), theta = 4.0
        -> FOPDT, K = 5, tau = 1, theta = 4        (atraso DOMINANTE: theta > tau)

    Sistema 2 (`caso_real_rg_superamortecido.png`)
        signal.TransferFunction([8], [1, 6, 4]), theta = 2.0
        -> 2a ordem, K = 2, wn = 2, zeta = 1.5, theta = 2   (SUPERamortecida)

    Sistema 3 (`caso_real_rg_subamortecido.png`)
        signal.TransferFunction([25], [1, 1.5, 25]), theta = 1.0
        -> 2a ordem, K = 1, wn = 5, zeta = 0.15, theta = 1  (SUBamortecida)

Os três compartilham uma geometria que o corpus sintético NÃO produz: `rg.py`
usa `plt.ylim(0, ...)`, que encosta a curva e o rótulo extremo na moldura
inferior (2 a 3 px, ~0,9 % do span do eixo y), enquanto o gerador sorteia
`y_margin_lo ~ U(0.03, 0.15)` e nunca desce abaixo de 3 %. Dois defeitos saíram
daí, e um terceiro continua ABERTO por exigir retreino (ver §39.3).

  - Sistema 1: o Estágio D escolhia `second` para uma planta de 1a ordem. O polo
    extra ficava em tau = 45 ms = 2,35 amostras = 3,6 px — a espessura do próprio
    traço no canto do degrau. 107,5 % do ganho de SSE do 2a ordem sobre o FOPDT
    vinha de DOIS pontos. Consertado por `_polo_rapido_e_artefato`.
  - Sistema 3: `_equiespacados` reprovava o eixo y com os NOVE rótulos lidos
    corretamente, porque o recorte da faixa cortava os glifos dos rótulos
    extremos e enviesava o centróide deles em 1,5 px para dentro. Consertado por
    `_centros_y_corrigidos` mais a unidade robusta em `_equiespacados`.
  - Sistema 2 JÁ PASSAVA antes das duas correções, e está aqui por duas razões:
    é o CONTROLE NEGATIVO da guarda do Sistema 1 (uma 2a ordem genuinamente
    superamortecida, zeta = 1,5, que a guarda não pode rebaixar), e é a prova de
    que a fragilidade do Estágio A na geometria rente à moldura é INTERMITENTE —
    mesma distância de 3 px, e a máscara sobrevive aqui e colapsa no Sistema 1.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIX = Path(__file__).resolve().parents[1] / "fixtures"

SISTEMA_1 = {"path": FIX / "caso_real_rg_fopdt_atraso.png",
             "order": "fopdt", "K": 5.0, "tau": 1.0, "theta": 4.0}
SISTEMA_2 = {"path": FIX / "caso_real_rg_superamortecido.png",
             "order": "second", "K": 2.0, "wn": 2.0, "zeta": 1.5, "theta": 2.0}
SISTEMA_3 = {"path": FIX / "caso_real_rg_subamortecido.png",
             "order": "second", "K": 1.0, "wn": 5.0, "zeta": 0.15, "theta": 1.0}

# Folgado sobre o que a cadeia consertada entrega e apertado o bastante para
# reprovar o defeito. Mesma lógica de `test_caso_real.py`.
TOL = 0.06


@pytest.fixture(scope="module")
def modelo():
    import torch
    from identify.extract import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model("models/unet_stageA.pt", dev), dev


def _roda(caso, modelo):
    from identify.pipeline import identify_from_image
    m, dev = modelo
    img = np.asarray(Image.open(caso["path"]).convert("RGB"))
    return identify_from_image(img, m, dev)


def _erro(medido, esperado):
    return abs(medido - esperado) / abs(esperado)


# --------------------------------------------------------------------------- #
# Sistema 1 — seleção de ordem
# --------------------------------------------------------------------------- #

def test_sistema1_nao_inventa_polo_irresoluvel(modelo):
    """A planta é de 1a ordem e o ajuste FOPDT já a descreve (NRMSE 0,0020).

    Aceitar `second` aqui é reportar como dinâmica um polo cuja constante de
    tempo cabe em 2,4 amostras da série extraída.
    """
    r = _roda(SISTEMA_1, modelo)
    assert r["order"] == "fopdt", (
        f"ordem {r['order']!r}, esperada 'fopdt'; params={r['params']}")


def test_sistema1_recupera_K_tau_theta(modelo):
    r = _roda(SISTEMA_1, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r} cal={r['calibration']}"
    p = r["params"]
    for nome in ("K", "tau", "theta"):
        e = _erro(p[nome], SISTEMA_1[nome])
        assert e <= TOL, (f"{nome} = {p[nome]:.4f}, esperado {SISTEMA_1[nome]} "
                          f"(erro {e:.1%}, tolerância {TOL:.0%})")


# --------------------------------------------------------------------------- #
# Sistema 3 — calibração do eixo y
# --------------------------------------------------------------------------- #

def test_sistema3_calibra_o_eixo_y():
    """Os nove rótulos do eixo y são lidos CERTOS; o eixo tem de ser aceito.

    Sem isso a pipeline devolve o nível adimensional e K fica indisponível —
    perda de informação causada por um artefato de recorte, não por leitura ruim.
    """
    from identify.calibrate import calibrate
    img = np.asarray(Image.open(SISTEMA_3["path"]).convert("RGB"))
    cal = calibrate(img)
    assert cal.ok_y, f"eixo y reprovado: reason={cal.reason!r}, n_pairs_y={cal.n_pairs_y}"
    assert cal.ok, f"calibração reprovada: reason={cal.reason!r}"


def test_sistema3_recupera_K_wn_zeta_theta(modelo):
    r = _roda(SISTEMA_3, modelo)
    assert r["order"] == "second", f"ordem {r['order']!r}, esperada 'second'"
    assert r["ok"], f"sem físico: reason={r['reason']!r} cal={r['calibration']}"
    p = r["params"]
    for nome in ("K", "wn", "zeta", "theta"):
        e = _erro(p[nome], SISTEMA_3[nome])
        assert e <= TOL, (f"{nome} = {p[nome]:.4f}, esperado {SISTEMA_3[nome]} "
                          f"(erro {e:.1%}, tolerância {TOL:.0%})")


# --------------------------------------------------------------------------- #
# Sistema 2 — controle negativo da guarda de polo-artefato
# --------------------------------------------------------------------------- #

def test_sistema2_guarda_nao_rebaixa_2a_ordem_legitima(modelo):
    """zeta = 1,5 é 2a ordem SUPERamortecida de verdade — o regime exato em que
    `_polo_rapido_e_artefato` tem licença para agir.

    Sem este teste a guarda do Sistema 1 poderia ser apertada até rebaixar
    qualquer superamortecida, e o corpus sintético sozinho não protegeria contra
    isso nesta geometria (curva rente à moldura), que ele não produz. Margem
    medida: o estatístico do trecho dá 0,639 aqui contra o limiar de 1,0, e
    1,013 no Sistema 1.
    """
    r = _roda(SISTEMA_2, modelo)
    assert r["order"] == "second", (
        f"ordem {r['order']!r}, esperada 'second' — a guarda de polo-artefato "
        f"rebaixou uma 2a ordem legítima; params={r['params']}")


def test_sistema2_recupera_K_wn_zeta_theta(modelo):
    r = _roda(SISTEMA_2, modelo)
    assert r["ok"], f"sem físico: reason={r['reason']!r} cal={r['calibration']}"
    p = r["params"]
    for nome in ("K", "wn", "zeta", "theta"):
        e = _erro(p[nome], SISTEMA_2[nome])
        assert e <= TOL, (f"{nome} = {p[nome]:.4f}, esperado {SISTEMA_2[nome]} "
                          f"(erro {e:.1%}, tolerância {TOL:.0%})")


def test_os_tres_encostam_na_moldura_como_o_corpus_nunca_faz():
    """Documenta em CÓDIGO a razão de estas três imagens existirem como fixture.

    O defeito aberto do §39.3 (Estágio A perde o traço rente à moldura) só é
    alcançável nesta geometria, e o gerador não a produz: `y_margin_lo` é
    sorteado em U(0.03, 0.15), então a curva nunca fica a menos de 3 % do span
    do eixo y da moldura inferior. Se um retreino com estrato novo mudar isso, é
    aqui que a premissa deixa de valer e o teste avisa.
    """
    from identify.calibrate import calibrate

    for caso in (SISTEMA_1, SISTEMA_2, SISTEMA_3):
        img = np.asarray(Image.open(caso["path"]).convert("RGB"))
        cal = calibrate(img)
        x0, y0, x1, y1 = cal.bbox_px
        span = y1 - y0
        # linha mais baixa com tinta CROMÁTICA (a curva; a moldura é acinzentada)
        c = img[y0:y1, x0:x1, :].astype(int)
        sat = c.max(2) - c.min(2)
        linhas = np.flatnonzero((sat > 60).any(axis=1))
        assert linhas.size, f"curva não localizada em {caso['path'].name}"
        folga = span - int(linhas[-1])
        assert folga / span < 0.03, (
            f"{caso['path'].name}: curva a {100*folga/span:.1f}% do span da moldura "
            f"— acima do piso de 3 % do gerador, então esta fixture deixou de "
            f"cobrir a geometria fora da distribuição de treino (§39.3)")


# --------------------------------------------------------------------------- #
# Defeito ABERTO — exige retreino (§39.3)
# --------------------------------------------------------------------------- #
#
# Marcado `xfail(strict=True)`: hoje FALHA de propósito, e no dia em que um
# retreino com o estrato novo consertar o Estágio A ele vira XPASS, que em modo
# estrito REPROVA a suíte. Isso é o alarme: quem consertar é obrigado a vir aqui,
# tirar a marca e transformar o defeito documentado em portão de regressão.
# Registrar em código, e não só no HANDOFF, é o que impede a dívida de sumir.

COBERTURA_MIN = 0.90
# Fração das colunas da moldura em que o Estágio A precisa entregar ponto. Os
# três sistemas têm curva contínua atravessando a janela inteira, então o valor
# honesto é ~1,0; 0,90 dá folga para as bordas e para a legenda. Medido hoje:
#   Sistema 1 ..... 65,3 %   (perde t=[0, 3.95] s inteiro — o platô do atraso)
#   Sistema 2 ..... 96,9 %   (passa)
#   Sistema 3 ..... 82,2 %   (perde a cauda assentada sobre a tracejada K=1)


@pytest.mark.parametrize("caso, rotulo", [
    pytest.param(SISTEMA_1, "sistema1", marks=pytest.mark.xfail(
        strict=True, reason="§39.3 defeito A: curva rente à moldura — máscara "
                            "perde o platô do tempo morto, 65,3% de cobertura")),
    pytest.param(SISTEMA_2, "sistema2"),
    pytest.param(SISTEMA_3, "sistema3", marks=pytest.mark.xfail(
        strict=True, reason="§39.3 defeito B: trecho perfeitamente reto — máscara "
                            "perde a cauda assentada, 82,2% de cobertura")),
])
def test_estagio_a_cobre_a_janela_inteira(caso, rotulo, modelo):
    """O Estágio A tem de enxergar a curva no platô e na cauda assentada.

    São DOIS defeitos, não um, e a ablação separa (deslocando só o trecho
    perdido, resto do render idêntico):

        ablação                          Sistema 1   Sistema 3
        original                            0/200        0/85
        ondulação de +-1 px, mesma altura   0/200       79/85
        12 px acima da moldura, ainda reto 98/200          --

    **Defeito A (Sistema 1) — curva rente à moldura inferior.** Dose-resposta
    no platô (span do eixo = 318 px): 5 px da moldura -> 0/150 colunas; 7 px ->
    107/150; 13 px -> 122/150. O degrau está em ~2 % do span, logo ABAIXO do
    piso de 3 % que `dataset/randomize.py` sorteia em `y_margin_lo`. Refutadas
    por ablação: a banda cinza do `axvspan` (0/150 -> 0/150), a COR do traço
    (recolorir para o rosa do Sistema 2, que funciona, mantém 0/150) e a
    planura (ondular na mesma altura mantém 0/200).

    **Defeito B (Sistema 3) — trecho perfeitamente RETO.** Não é oclusão:
    remover a tracejada de referência recupera só 7/85, e afastar a cauda dela
    25 px recupera 0/85. Ondular +-1 px recupera 79/85. A rede vê amarelo puro
    não ocluído em x=640 e responde 0,047 — a probabilidade cai de 0,954 em
    x=634 para 0,027 em x=636, exatamente onde a ondulação da curva fica
    sub-pixel. O modelo aprendeu a suprimir reta horizontal de span completo
    (critério G3b.2, que existe para os distratores), e uma resposta assentada
    É uma reta horizontal.

    Nenhum dos dois tem conserto em código: são regiões fora da distribuição de
    treino. Ver §39.3 para o que o retreino precisa cobrir.
    """
    from identify.calibrate import calibrate
    from identify.extract import predict_mask
    from identify.polyline import mask_to_polyline

    m, dev = modelo
    img = np.asarray(Image.open(caso["path"]).convert("RGB"))
    cal = calibrate(img)
    x0, _, x1, _ = cal.bbox_px
    xs, _ = mask_to_polyline(predict_mask(m, img, dev), bbox=cal.bbox_px)
    cols = np.unique(np.round(xs).astype(int))
    cob = cols[(cols >= x0) & (cols <= x1)].size / (x1 - x0)
    assert cob >= COBERTURA_MIN, (
        f"{rotulo}: máscara cobre {100*cob:.1f}% das colunas da moldura, "
        f"mínimo {100*COBERTURA_MIN:.0f}%")
