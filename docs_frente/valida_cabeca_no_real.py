"""Valida a cabeca de contagem, com encoder CONGELADO, nas figuras REAIS.

A sondagem (`sonda_cabeca_contagem.py`) mediu AUC 0,9755 — mas EM DISTRIBUICAO,
em `data/val_multi`, do mesmo gerador do treino. Este arquivo faz a pergunta
que decide se a cabeca serve: ela generaliza para as figuras dos tres
geradores reais?

O salto de dominio e grande e deliberado. O corpus sintetico:
  - NAO desenha o degrau de entrada; as reais desenham, nas tres familias;
  - aplica o primeiro degrau em t=0; as reais aplicam dentro da janela;
  - nao tem tema escuro, `axvspan` sobre a resposta, nem legenda ocluindo.

PROTOCOLO. O limiar de decisao e escolhido em `data/val_multi` (sintetico) e
so DEPOIS aplicado ao real. Escolher o limiar no proprio conjunto real seria
selecionar no teste e inflaria o numero.

Uso:
    .venv/bin/python valida_cabeca_no_real.py [--epocas 60]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_auc_score

from identify.extract import UNet, letterbox, load_model

RAIZ = Path(__file__).resolve().parent
REAL = RAIZ / "reports" / "amostras_aleatorias"
LOTES_REAIS = (REAL, REAL / "balanceado", REAL / "balanceado2")


@torch.no_grad()
def features(modelo, imagens, device, size=512):
    """Gargalo com o eixo Y colapsado — o que a cabeca consome."""
    X = []
    for i, p in enumerate(imagens):
        img = np.asarray(Image.open(p).convert("RGB"))
        small, _ = letterbox(np.ascontiguousarray(img[..., :3]), size)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(device)
        X.append(modelo.gargalo(x).mean(dim=2)[0].cpu().numpy().astype(np.float32))
        if i % 100 == 0:
            print(f"    {i}/{len(imagens)}", flush=True)
    return np.stack(X)


def metricas(y, escore, limiar):
    pred = (escore > limiar).astype(float)
    TP = float(((pred == 1) & (y == 1)).sum())
    FP = float(((pred == 1) & (y == 0)).sum())
    FN = float(((pred == 0) & (y == 1)).sum())
    TN = float(((pred == 0) & (y == 0)).sum())
    p = TP / max(TP + FP, 1); r = TP / max(TP + FN, 1)
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, prec=p, rev=r,
                f1=2 * p * r / max(p + r, 1e-9), acc=(TP + TN) / len(y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = load_model(str(RAIZ / "models" / "unet_stageA.pt"), dev)
    for p in modelo.parameters():
        p.requires_grad = False
    modelo.eval()

    # ---- features sinteticas (cache da sondagem)
    cache_sin = RAIZ / "reports" / "sonda_contagem_features.npz"
    if cache_sin.exists():
        z = np.load(cache_sin)
        Xtr, ytr, Xva, yva = z["Xtr"], z["ytr"], z["Xva"], z["yva"]
    else:
        def _com_rotulo(dirs):
            am = [d for r in dirs for d in sorted(Path(r).glob("sample_*"))]
            X = features(modelo, [d / "image.png" for d in am], dev)
            y = np.array([1.0 if json.loads((d / "meta.json").read_text())
                          .get("n_degraus", 1) > 1 else 0.0 for d in am],
                         dtype=np.float32)
            return X, y
        print("  extraindo features sinteticas...")
        Xtr, ytr = _com_rotulo(["data/train_multi"])
        Xva, yva = _com_rotulo(["data/val_multi"])
        cache_sin.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_sin, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)

    # ---- features reais
    cache = REAL / "features_reais.npz"
    verdades = {}
    for d in LOTES_REAIS:
        for v in json.loads((d / "verdade.json").read_text()):
            verdades[str(d / v["arquivo"])] = v
    caminhos = sorted(verdades)
    yre = np.array([1.0 if verdades[k]["n_degraus"] > 1 else 0.0
                    for k in caminhos], dtype=np.float32)
    if cache.exists():
        Xre = np.load(cache)["X"]
        print(f"features reais do cache: {Xre.shape}")
    else:
        print("  extraindo features das figuras reais...")
        Xre = features(modelo, [Path(k) for k in caminhos], dev)
        np.savez_compressed(cache, X=Xre)

    # ---- treina a cabeca (encoder congelado)
    torch.manual_seed(0)
    C = Xtr.shape[1]
    cabeca = UNet(base=C // 16, levels=4, in_ch=3,
                  com_contagem=True).cabeca_conta.to(dev)
    opt = torch.optim.AdamW(cabeca.parameters(), lr=a.lr)
    Xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(ytr)[:, None].to(dev)
    Xv = torch.from_numpy(Xva).to(dev)
    pw = torch.tensor([float((1 - ytr.mean()) / ytr.mean())], device=dev)
    melhor = {"auc": -1.0}
    for ep in range(a.epocas):
        cabeca.train()
        perm = torch.randperm(len(yt), device=dev)
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(
                cabeca(Xt[idx]), yt[idx], pos_weight=pw).backward()
            opt.step()
        cabeca.eval()
        with torch.no_grad():
            lo = cabeca(Xv).cpu().numpy().ravel()
        auc = float(roc_auc_score(yva, lo))
        if auc > melhor["auc"]:
            melhor = {"auc": auc, "ep": ep,
                      "estado": {k: v.clone() for k, v in cabeca.state_dict().items()}}
    cabeca.load_state_dict(melhor["estado"])
    torch.save(cabeca.state_dict(), RAIZ / "models" / "cabeca_contagem_congelada.pt")
    print(f"\n  cabeca treinada: AUC sintetico = {melhor['auc']:.4f} "
          f"(epoca {melhor['ep']})")

    # ---- limiar escolhido NO SINTETICO
    cabeca.eval()
    with torch.no_grad():
        lo_va = cabeca(torch.from_numpy(Xva).to(dev)).cpu().numpy().ravel()
        lo_re = cabeca(torch.from_numpy(Xre).to(dev)).cpu().numpy().ravel()
    pr, rc, th = precision_recall_curve(yva, lo_va)
    f1v = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    lim_f1 = float(th[int(np.nanargmax(f1v[:-1]))])
    k = np.flatnonzero(pr[:-1] >= 0.95)
    lim_p95 = float(th[k[int(np.argmax(rc[:-1][k]))]]) if k.size else lim_f1

    L = []
    def P(s=""):
        L.append(s)
    P("VALIDACAO NO REAL — cabeca de contagem, encoder CONGELADO")
    P("=" * 76)
    P(f"  treinada em data/train_multi (sintetico, n={len(ytr)})")
    P(f"  avaliada em {len(yre)} figuras dos tres geradores reais "
      f"({yre.mean():.1%} multi-degrau)")
    P(f"  limiar escolhido no SINTETICO, aplicado ao real (nao selecionado nele)")
    P()
    P(f"  AUC sintetico (data/val_multi) = {melhor['auc']:.4f}")
    P(f"  AUC REAL                       = {roc_auc_score(yre, lo_re):.4f}")
    P()
    P(f"  {'limiar':<26}{'precisao':>10}{'revocacao':>11}{'F1':>8}{'acerto':>9}")
    for rot, lim in (("melhor F1 no sintetico", lim_f1),
                     ("precisao>=95% no sintetico", lim_p95)):
        m = metricas(yre, lo_re, lim)
        P(f"  {rot:<26}{m['prec']:>10.1%}{m['rev']:>11.1%}{m['f1']:>8.3f}"
          f"{m['acc']:>9.1%}")
    P()
    P("  Referencias medidas nas MESMAS 299 figuras:")
    P(f"  {'gate de residuo (producao)':<26}{'69,9%':>10}{'48,3%':>11}{0.571:>8.3f}")
    P(f"  {'gate de residuo (sem piso)':<26}{'78,2%':>10}{'70,3%':>11}{0.740:>8.3f}")
    P(f"  {'contagem PERFEITA':<26}{'100,0%':>10}{'70,3%':>11}{0.825:>8.3f}")
    P()
    m = metricas(yre, lo_re, lim_f1)
    P(f"  matriz no limiar de melhor F1: TP={m['TP']:.0f} FP={m['FP']:.0f} "
      f"FN={m['FN']:.0f} TN={m['TN']:.0f}")
    P()
    P("  estratificado (limiar de melhor F1):")
    for chave, rot in (("familia", "familia"), ("fundo_escuro", "tema escuro"),
                       ("linestyle", "tipo de linha")):
        P(f"    por {rot}:")
        grupos = {}
        for i, k2 in enumerate(caminhos):
            grupos.setdefault(verdades[k2][chave], []).append(i)
        for g in sorted(grupos, key=str):
            idx = np.array(grupos[g])
            mm = metricas(yre[idx], lo_re[idx], lim_f1)
            auc_g = (roc_auc_score(yre[idx], lo_re[idx])
                     if len(set(yre[idx])) > 1 else float("nan"))
            P(f"      {str(g):<14} n={len(idx):<4d} AUC={auc_g:6.3f}  "
              f"prec={mm['prec']:6.1%} rev={mm['rev']:6.1%} acerto={mm['acc']:6.1%}")
    txt = "\n".join(L)
    (REAL / "validacao_cabeca_real.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
