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
TOL = 0.06

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
    1,06 % (4/378); ela é reportada para que uma mudança grande fique visível,
    não para reprovar a suíte."""
    from identify.pipeline import identify_from_image

    m, dev = modelo
    dirs = sorted((ROOT / "data" / "test").glob("sample_*"))[:300]
    assert dirs, "rode o Passo 12 do Bloco 0: data/test está vazio"

    n_trunc, lat = 0, []
    for d in dirs:
        img = np.asarray(Image.open(d / "image.png").convert("RGB"))
        r = identify_from_image(img, m, dev)
        n_trunc += r["truncado_em"] is not None
        lat.append(r["latency_ms"])

    taxa = n_trunc / len(dirs)
    p95 = float(np.percentile(lat, 95))
    record_p2("2.13", "taxa de truncagem no corpus de degrau unico",
              "diagnostico, sem alvo", f"{100*taxa:.2f}% ({n_trunc}/{len(dirs)})",
              None)
    record_p2("2.8-trunc", "latencia por imagem COM a truncagem ligada",
              "< 500 ms", f"{np.mean(lat):.0f} ms (p95 {p95:.0f})",
              bool(np.mean(lat) < 500.0))
    assert np.mean(lat) < 500.0, (
        f"latência média {np.mean(lat):.0f} ms furou o critério 2.8; o scan "
        f"custa ~2,8 s e só se paga porque dispara em ~2 % das imagens")
