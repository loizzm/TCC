"""PAR CONTROLADO: a cabeca nao generaliza por causa do ESTILO ou da FISICA?

A cabeca de contagem com encoder congelado da AUC 0,99 no sintetico e 0,65 no
corpus real (`valida_cabeca_no_real.py`). Duas explicacoes competem, e elas
pedem correcoes opostas:

  (a) ESTILO — o corpus de treino renderiza com `dataset/randomize.py`; as
      figuras reais usam paletas escolhidas a mao, temas seaborn/dark,
      preenchimentos e legendas fora dessa distribuicao. Correcao: ampliar
      `randomize.py`.
  (b) FISICA — as distribuicoes de razao de amplitude, separacao entre degraus
      e janela do estrato nao cobrem as do corpus real. Correcao:
      `sorteia_degraus`.

Este arquivo separa as duas com UMA variavel. Le a verdade de cada figura real,
reconstroi o `SystemSpec` EXATO (mesma planta, mesmos degraus, mesma janela) e
renderiza com `sample_style` — o render do treino. A fisica fica identica a
real; so o desenho muda.

  AUC sobe perto do sintetico -> a fisica ja esta certa, o problema e (a).
  AUC continua baixa          -> o render nao era o problema, e (b).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from dataset.generator import SystemSpec, render_sample
from dataset.randomize import sample_style
from identify.extract import UNet, letterbox, load_model

RAIZ = Path(__file__).resolve().parent
REAL = RAIZ / "reports" / "amostras_aleatorias"
LOTES = (REAL, REAL / "balanceado", REAL / "balanceado2")
SAIDA = RAIZ / "data" / "real_estilo_treino"


def spec_da_verdade(v: dict) -> SystemSpec:
    """`SystemSpec` que reproduz EXATAMENTE a fisica de uma figura real."""
    degraus = tuple((float(a), float(t)) for a, t in v["degraus"])
    return SystemSpec(
        order=v["order"],
        K=float(v["K_planta"]),
        tau=None if v["tau"] is None else float(v["tau"]),
        theta=float(v["theta_sistema"]),
        wn=None if v["wn"] is None else float(v["wn"]),
        zeta=None if v["zeta"] is None else float(v["zeta"]),
        t_start=0.0,
        t_end=float(v["t_fim"]),
        step_amplitude=float(degraus[0][0]),
        degraus=degraus,
    )


def gera() -> list[dict]:
    if SAIDA.exists():
        shutil.rmtree(SAIDA)
    SAIDA.mkdir(parents=True)
    rotulos = []
    i = 0
    for d in LOTES:
        for v in json.loads((d / "verdade.json").read_text()):
            spec = spec_da_verdade(v)
            # O estilo NAO pode ver o spec (anti-vazamento de randomize.py):
            # stream propria, semente so do indice.
            style = sample_style(np.random.default_rng([31337, i]))
            out = SAIDA / f"sample_{i:05d}"
            render_sample(spec, style, out, add_noise=False,
                          rng=np.random.default_rng([777, i]), seed=i)
            rotulos.append({"dir": str(out), "n_degraus": v["n_degraus"],
                            "origem": f"{d.name}/{v['arquivo']}"})
            i += 1
    (SAIDA / "rotulos.json").write_text(json.dumps(rotulos, indent=2))
    return rotulos


@torch.no_grad()
def avalia(rotulos, modelo, cabeca, dev):
    esc, y = [], []
    for j, r in enumerate(rotulos):
        img = np.asarray(Image.open(Path(r["dir"]) / "image.png").convert("RGB"))
        small, _ = letterbox(np.ascontiguousarray(img[..., :3]), 512)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(dev)
        esc.append(float(cabeca(modelo.gargalo(x).mean(dim=2))[0, 0]))
        y.append(1.0 if r["n_degraus"] > 1 else 0.0)
        if j % 100 == 0:
            print(f"    {j}/{len(rotulos)}", flush=True)
    return np.array(y), np.array(esc)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / "models" / "unet_stageA.pt"), dev)
    modelo.eval()
    cabeca = UNet(base=32, levels=4, in_ch=3, com_contagem=True).cabeca_conta.to(dev)
    cabeca.load_state_dict(torch.load(RAIZ / "models" / "cabeca_contagem_congelada.pt",
                                      map_location=dev))
    cabeca.eval()

    marca = SAIDA / "rotulos.json"
    rotulos = json.loads(marca.read_text()) if marca.exists() else gera()
    print(f"  {len(rotulos)} figuras: fisica REAL, render do TREINO\n")
    y, esc = avalia(rotulos, modelo, cabeca, dev)

    L = []
    def P(s=""):
        L.append(s)
    P("PAR CONTROLADO — estilo x fisica")
    P("=" * 66)
    P(f"  n={len(y)}  ({y.mean():.1%} multi-degrau)")
    P()
    P(f"  {'lote':<40}{'AUC':>8}")
    P(f"  {'sintetico (fisica treino + render treino)':<40}{0.9919:>8.4f}")
    P(f"  {'ESTE (fisica REAL + render treino)':<40}{roc_auc_score(y, esc):>8.4f}")
    P(f"  {'real (fisica REAL + render real)':<40}{0.6540:>8.4f}")
    P()
    a = roc_auc_score(y, esc)
    if a > 0.85:
        P("  VEREDITO: com o render do treino a AUC volta. A fisica do estrato")
        P("  ja cobre o caso real — o que falta e AMPLITUDE DE ESTILO em")
        P("  `dataset/randomize.py` (paletas, temas, preenchimentos).")
    elif a > 0.75:
        P("  VEREDITO: o estilo explica boa parte, mas nao tudo. As duas")
        P("  correcoes sao necessarias.")
    else:
        P("  VEREDITO: trocar o render NAO recupera a AUC. O problema esta na")
        P("  FISICA do estrato, nao no desenho — corrigir `sorteia_degraus`.")
    txt = "\n".join(L)
    (REAL / "estilo_vs_fisica.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
