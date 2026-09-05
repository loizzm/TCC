"""Custo da truncagem no corpus de degrau único — medido em DANO, não em taxa.

As quatro amostras que disparam foram auditadas uma a uma contra a verdade do
`meta.json` na spec §5.2. Nenhuma cruza a TOL de 6 % de boa para ruim, e numa
delas a truncagem é um conserto grande (40,0 % -> 0,9 %). Os dois gates que
controlam o disparo (`_PISO_SUSPEITA` e `_GANHO_MIN`) são uma CONJUNÇÃO: as
quatro que disparam passam nos dois (`nrmse_full` de 0,3085, 0,0444, 0,0418 e
0,1137). O `sample_00193` é o VIZINHO mais próximo do limiar, mas é excluído
pelo PISO, não pelo ganho — seu `nrmse_full` é 0,0056, abaixo de
`_PISO_SUSPEITA = 0,030`, então ele nunca chega a entrar no scan; o ganho de
0,4325 que ele exibiria vem de um scan forçado que ignora o piso. Baixar
`_GANHO_MIN` sozinho não o admitiria — seria preciso também baixar
`_PISO_SUSPEITA` em 5,4x. Ainda assim ele é a amostra mais próxima do dano no
corpus e segue na lista parametrizada como vizinho a observar.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.part2.conftest import record_p2

ROOT = Path(__file__).resolve().parents[2]

# id -> (dispara?, erro esperado do ajuste TRUNCADO contra a verdade)
# Medidos na spec §5.2. `sample_00328` está quebrado nos dois casos por razão
# alheia a esta mudança (erro silencioso preexistente, nrmse 0,1137 sob o
# limiar de recusa de 0,13) e por isso entra só como "dispara", sem alvo de erro.
AUDITADAS = {
    "sample_00341": (True, 0.06),
    "sample_00257": (True, 0.06),
    "sample_00357": (True, 0.06),
    "sample_00328": (True, None),
    "sample_00193": (False, None),
}


@pytest.fixture(scope="module")
def modelo():
    import torch
    from identify.extract import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model("models/unet_stageA.pt", dev), dev


def _resultado(sid, modelo):
    from identify.pipeline import identify_from_image
    d = ROOT / "data" / "test" / sid
    assert d.is_dir(), f"{sid} ausente: rode o Passo 12 do Bloco 0"
    meta = json.loads((d / "meta.json").read_text())
    img = np.asarray(Image.open(d / "image.png").convert("RGB"))
    m, dev = modelo
    return meta, identify_from_image(img, m, dev)


@pytest.mark.parametrize("sid", sorted(AUDITADAS))
def test_amostras_auditadas_nao_pioram(sid, modelo):
    """Portão de DANO. Fixa nominalmente as quatro que disparam mais a vizinha
    do limiar, e falha se alguma degradar além da TOL."""
    dispara_esperado, erro_max = AUDITADAS[sid]
    meta, r = _resultado(sid, modelo)

    disparou = r["truncado_em"] is not None
    assert disparou == dispara_esperado, (
        f"{sid}: truncou={disparou}, esperado {dispara_esperado} "
        f"(ganho {r['ganho_truncagem']}) — a banda vazia da spec §5.2.1 mudou, "
        f"e ela precisa ser REMAPEADA, não contornada")

    if erro_max is None or not r["ok"]:
        return
    verd = meta["params"]
    chaves = ("K", "tau") if meta["order"] == "fopdt" else ("K", "wn", "zeta")
    for k in chaves:
        if verd.get(k) in (None, 0):
            continue
        e = abs(r["params"][k] - verd[k]) / abs(verd[k])
        assert e <= erro_max, (
            f"{sid}: {k} = {r['params'][k]:.4f}, verdade {verd[k]:.4f} "
            f"(erro {e:.1%}) — a truncagem PIOROU esta amostra")


@pytest.mark.slow
def test_taxa_de_disparo_no_corpus_e_reportada(modelo):
    """Diagnóstico SEM alvo, de propósito (Ruling 50). A taxa medida na spec é
    1,06 % (4/378) sobre OUTRA população (o corpus inteiro auditado, n=378);
    este teste anda só nas 300 primeiras amostras de `data/test`, então a taxa
    aqui é sobre n=300 e as duas não são comparáveis ponto a ponto — só uma das
    quatro amostras que disparam (spec §5.2) tem índice abaixo de 300. O nome
    do critério abaixo declara a população para quem for comparar contra a spec.

    **Latência (`2.8-trunc`) é DIAGNÓSTICO, sem veredito — e é assim de
    propósito, não por preguiça.** Este teste aquece o modelo antes de
    cronometrar (mesmo mecanismo do 2.8 original em `test_part2.py`, para não
    deixar o custo de carregar o modelo na primeira imagem entrar na amostra),
    mas mesmo assim um único run de parede numa estação de trabalho contendida
    não sustenta veredito em NENHUMA direção. Medido diretamente (fora deste
    teste, isolando a variável truncagem): 120 imagens do corpus em que ZERO
    truncaram deram media 424,8 ms / p95 706,1 ms nesta mesma máquina — 2,28x
    o baseline histórico do `reports/part2_strata.md` (mediana 168 ms, p95
    310 ms) — sem a truncagem entrar em jogo nenhuma vez. Ou seja: o número
    absoluto de parede aqui mede a MÁQUINA, não o código. Além disso a
    truncagem dispara em ~1 de 300 amostras (Ruling 50) e por isso NÃO PODE
    mover o p95 estruturalmente — o índice do p95 (285 de 300) cai bem dentro
    da maioria que não trunca, então o custo do scan nem aparece nessa
    estatística. Por essas duas razões o critério 2.8 (a linha em
    `test_part2.py`, intocada por este arquivo) precisa de um re-run numa
    máquina ociosa antes de sua linha na tabela poder carregar veredito de
    novo — decisão do dono do branch, não deste teste."""
    from identify.pipeline import identify_from_image

    m, dev = modelo
    dirs = sorted((ROOT / "data" / "test").glob("sample_*"))[:300]
    assert dirs, "rode o Passo 12 do Bloco 0: data/test está vazio"

    img0 = np.asarray(Image.open(dirs[0] / "image.png").convert("RGB"))
    identify_from_image(img0, m, dev)   # aquecimento

    n_trunc, lat = 0, []
    for d in dirs:
        img = np.asarray(Image.open(d / "image.png").convert("RGB"))
        r = identify_from_image(img, m, dev)
        n_trunc += r["truncado_em"] is not None
        lat.append(r["latency_ms"])

    taxa = n_trunc / len(dirs)
    p95 = float(np.percentile(lat, 95))
    record_p2("2.13", "taxa de truncagem (300 primeiras de data/test)",
              "diagnostico, sem alvo", f"{100*taxa:.2f}% ({n_trunc}/{len(dirs)})",
              None)
    record_p2("2.8-trunc", "latencia por imagem COM a truncagem ligada",
              "diagnostico, sem alvo (ver docstring)",
              f"media {np.mean(lat):.0f} ms, p95 {p95:.0f} ms", None)
