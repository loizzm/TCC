# Correção da máscara e do nível de repouso — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corrigir os dois defeitos que a imagem real expôs — a máscara ingerir conteúdo fora da moldura e escolher o ramo de tinta errado, e o estimador de nível de repouso enviesado quando θ ≈ 0 — sem regredir nenhum número do conjunto sintético.

**Architecture:** Três mudanças cirúrgicas, cada uma validada contra **as duas** frentes (a imagem real e as 300 amostras sintéticas), mais um estrato novo no gerador que torna mensurável o terceiro defeito, que não é consertável nos mesmos termos. Nenhuma mudança entra se melhorar o real e regredir o sintético.

**Tech Stack:** Python 3.11 (`.venv`), numpy, OpenCV, scikit-image, torch 2.13.0+cu130 (GPU), pytest.

**Spec:** `HANDOFF_P2_7.md` §34 (Ruling 55 — diagnóstico completo do caso real), com apoio em §33 (Ruling 54, OOD de aquisição) e §23 (Ruling 45, a Decisão E que introduziu o estimador defeituoso).

## Global Constraints

- **Nunca quebrar o contrato físico de `identify_from_image`**: as chaves `order`, `params`, `ok` no topo continuam sendo do nível FÍSICO, porque `tests/part2` e a Parte 3 as consomem. `ok` continua significando "há saída física".
- **Toda mudança é validada nas DUAS frentes.** Melhorar o caso real e regredir o sintético = não entra. Os números sintéticos a preservar, medidos com `models/unet_stageA.pt` (rodada 6, `base=24`) sobre as 300 amostras de `data/test`:
  - 2.1 erro perpendicular: **0,800 px mediana / 1,699 px p95**
  - 2.2 piso perpendicular: **0,614 px / 1,135 px**
  - 2.7: nenhum estrato acima de **1,0 px** (o mais apertado é `traco=:` em 0,956)
  - 2.3: **0,958** · 2.6: **+0,99 p.p.** · 2.6-adim[zeta]: **+1,53 p.p.** · 2.9: **0,797**
  - 2.8 latência: **183 ms** mediana (alvo < 500 ms)
  - suíte completa: **2 failed / 57 passed**
- **Não regenerar `data/train|val|test`.** Os números acima dependem desses splits exatos. O estrato novo do gerador entra **opt-in**, com o comportamento padrão idêntico ao atual (Task 5).
- **`git commit` só com autorização explícita do autor no momento.** Os passos de commit deste plano são pontos de parada para pedir autorização, não permissão prévia.
- Verdade do caso real, para asserções: **ζ = 0,5 · ωₙ = 2,0 · K = 1,0 · θ = 0 · janela T = 10 s**, segunda ordem.

---

## File Structure

| Arquivo | Responsabilidade | Task |
|---|---|---|
| `tests/fixtures/caso_real_2ordem.png` | Ativo de teste: a imagem real, congelada no repositório | 1 |
| `tests/part2/test_caso_real.py` | **Criar.** Teste de regressão do caso real, com a verdade declarada | 1, 2, 3, 4 |
| `identify/polyline.py` | `mask_to_polyline` passa a aceitar moldura e a desambiguar colunas multi-ramo | 2, 3 |
| `identify/pipeline.py` | Passa a moldura à polilinha; troca o estimador de nível de repouso | 2, 4 |
| `dataset/randomize.py` | Estrato novo, opt-in: reta de referência no patamar | 5 |
| `dataset/generator.py` | Propaga o estrato para o meta | 5 |
| `HANDOFF_P2_7.md` | Registro dos resultados (§35) | 6 |

Decomposição: `polyline.py` recebe as duas mudanças de extração porque ambas são
sobre "como a máscara vira uma sequência de pontos" — mudam juntas e são testadas
pelo mesmo critério (2.2 perpendicular). O estimador de repouso fica em
`pipeline.py` porque pertence à normalização da Decisão E, não à extração.

---

### Task 1: Fixture do caso real e teste de regressão que falha

Congela a imagem real no repositório e escreve o teste que documenta o defeito.
O teste **deve falhar** ao fim desta task — é o alvo das tasks 2 a 4.

**Files:**
- Create: `tests/fixtures/caso_real_2ordem.png` (cópia de `/home/loizm/Downloads/resposta_degrau.png`, 58 KB)
- Create: `tests/part2/test_caso_real.py`

**Interfaces:**
- Consumes: `identify.pipeline.identify_from_image`, `identify.extract.load_model`
- Produces: `CASO_REAL` (dict com o caminho do fixture e a verdade declarada) e `caso_real()` (fixture pytest que devolve a imagem como `np.ndarray` RGB), consumidos pelas tasks 2, 3 e 4.

- [ ] **Step 1: Copiar a imagem para o repositório**

```bash
cd /home/loizm/work/TCC-2
cp /home/loizm/Downloads/resposta_degrau.png tests/fixtures/caso_real_2ordem.png
ls -la tests/fixtures/caso_real_2ordem.png
```

Esperado: arquivo de ~58 KB. Não está coberto pelo `.gitignore` (que ignora `data/`, não `tests/`).

- [ ] **Step 2: Escrever o teste de regressão**

Criar `tests/part2/test_caso_real.py`:

```python
"""Regressão do CASO REAL (HANDOFF_P2_7 §34, Ruling 55).

Imagem produzida FORA do gerador do projeto, com a verdade declarada pelo autor
do sistema: num = [wn**2], den = [1, 2*zeta*wn, wn**2], zeta = 0.5, wn = 2.0,
degrau unitário, janela de 10 s, theta = 0.

Este teste existe porque UMA imagem real encontrou dois defeitos que 300 amostras
sintéticas não encontraram. Ele não substitui a suíte sintética: guarda o eixo que
ela não cobre.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "caso_real_2ordem.png"

CASO_REAL = {
    "path": FIXTURE,
    "order": "second",
    "zeta": 0.5,
    "wn": 2.0,
    "K": 1.0,
    "theta": 0.0,
    "T": 10.0,          # janela em segundos, lida do eixo x da figura
}

# Tolerâncias. O caso real é mais difícil que o sintético (Ruling 55): a máscara
# enfrenta reta de referência coincidente com o patamar, legenda dentro da moldura
# e texto fora dela. 5 % é folgado sobre os 0,4 % que a cadeia consertada entrega
# e apertado o bastante para reprovar os 12,6 % do estimador defeituoso.
TOL_ZETA = 0.05
TOL_WN = 0.05


@pytest.fixture(scope="module")
def caso_real() -> np.ndarray:
    assert FIXTURE.exists(), f"fixture ausente: {FIXTURE}"
    return np.asarray(Image.open(FIXTURE).convert("RGB"))


def test_caso_real_acerta_a_ordem(caso_real):
    """A ordem é o portão: com ordem errada, zeta e wn_T saem nulos."""
    import torch
    from identify.extract import load_model
    from identify.pipeline import identify_from_image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    r = identify_from_image(caso_real, model, dev)
    assert r["order"] == CASO_REAL["order"], (
        f"ordem {r['order']!r}, esperada {CASO_REAL['order']!r}; "
        f"calibração: {r['calibration']}"
    )


def test_caso_real_recupera_zeta_e_wn(caso_real):
    """zeta e wn*T pelo nível ADIMENSIONAL — a calibração falha nesta imagem
    (reta de referência + legenda), e é exatamente o cenário da Decisão E."""
    import torch
    from identify.extract import load_model
    from identify.pipeline import identify_from_image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    r = identify_from_image(caso_real, model, dev)
    d = r["dimensionless"]

    assert d["zeta"] is not None, f"zeta ausente; order={r['order']!r}"
    e_zeta = abs(d["zeta"] - CASO_REAL["zeta"]) / CASO_REAL["zeta"]
    assert e_zeta <= TOL_ZETA, (
        f"zeta = {d['zeta']:.4f}, esperado {CASO_REAL['zeta']:.4f} "
        f"(erro {e_zeta:.1%}, tolerância {TOL_ZETA:.0%})"
    )

    assert d["wn_T"] is not None, "wn_T ausente"
    wn = d["wn_T"] / CASO_REAL["T"]
    e_wn = abs(wn - CASO_REAL["wn"]) / CASO_REAL["wn"]
    assert e_wn <= TOL_WN, (
        f"wn = {wn:.4f}, esperado {CASO_REAL['wn']:.4f} "
        f"(erro {e_wn:.1%}, tolerância {TOL_WN:.0%})"
    )
```

- [ ] **Step 3: Rodar e confirmar que FALHA**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_caso_real.py -v
```

Esperado: **as duas asserções falham.** A primeira com `ordem 'fopdt', esperada 'second'`; a segunda com `zeta ausente; order='fopdt'`. Se alguma passar, o diagnóstico do §34 está errado e o plano precisa ser revisto antes de continuar.

- [ ] **Step 4: Pedir autorização e commitar**

Perguntar ao autor antes de commitar (regra do projeto). Com autorização:

```bash
cd /home/loizm/work/TCC-2
git add tests/fixtures/caso_real_2ordem.png tests/part2/test_caso_real.py
git commit -m "test: fixture e regressão do caso real (Ruling 55)

Imagem produzida fora do gerador, com verdade declarada zeta=0.5 wn=2.0.
As duas asserções FALHAM por ora: o pipeline classifica como fopdt e nao
entrega zeta. E o alvo das correcoes seguintes."
```

---

### Task 2: A polilinha respeita a moldura do gráfico

Primeiro defeito do §34.2: a máscara captura título e rótulo de eixo, e a
polilinha vai de y = 21 a 551 quando a moldura é 39 a 503. Conteúdo fora da
moldura não é a curva, por definição.

**Files:**
- Modify: `identify/polyline.py:18` (assinatura e corpo de `mask_to_polyline`)
- Modify: `identify/pipeline.py:139` (passa a moldura)
- Test: `tests/part2/test_caso_real.py` (novo teste), `tests/part2/test_part2.py` (não-regressão)

**Interfaces:**
- Consumes: `CASO_REAL`, `caso_real()` da Task 1.
- Produces: `mask_to_polyline(mask, bbox=None)` — `bbox` é `(x0, y0, x1, y1)` inclusivo, no formato que `identify.calibrate.detect_plot_bbox` devolve. `None` preserva o comportamento atual, byte a byte.

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar em `tests/part2/test_caso_real.py`:

```python
def test_polilinha_nao_sai_da_moldura(caso_real):
    """Título e rótulo de eixo ficam FORA da moldura e não são a curva."""
    import torch
    from identify.calibrate import detect_plot_bbox
    from identify.extract import load_model, predict_mask
    from identify.polyline import mask_to_polyline
    from tests.part2.conftest import to_gray

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    bbox = detect_plot_bbox(to_gray(caso_real))
    assert bbox is not None
    x0, y0, x1, y1 = bbox

    mask = predict_mask(model, caso_real, dev)
    xp, yp = mask_to_polyline(mask, bbox=bbox)
    assert xp.size >= 10, "polilinha curta"
    assert yp.min() >= y0, f"y mínimo {yp.min():.0f} acima da moldura (y0={y0})"
    assert yp.max() <= y1, f"y máximo {yp.max():.0f} abaixo da moldura (y1={y1})"
    assert xp.min() >= x0 and xp.max() <= x1, "x fora da moldura"
```

- [ ] **Step 2: Rodar e confirmar que falha**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_caso_real.py::test_polilinha_nao_sai_da_moldura -v
```

Esperado: `TypeError: mask_to_polyline() got an unexpected keyword argument 'bbox'`.

- [ ] **Step 3: Implementar o recorte**

Em `identify/polyline.py`, trocar a assinatura e acrescentar o recorte logo no início do corpo:

```python
def mask_to_polyline(mask: np.ndarray,
                     bbox: tuple[int, int, int, int] | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
```

E, imediatamente após o docstring existente, antes de `binary = (mask > 127)...`:

```python
    # Recorte à moldura (HANDOFF_P2_7 §34.2). Título, rótulo de eixo e legenda
    # externa vivem FORA do quadro e não são a curva; no caso real a polilinha ia
    # de y=21 a 551 com a moldura em 39..503. `bbox=None` preserva o
    # comportamento anterior byte a byte, porque `tests/part2` compara números
    # medidos sem moldura contra o histórico das rodadas 3 a 6.
    if bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        fora = np.ones(mask.shape, dtype=bool)
        fora[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = False
        mask = mask.copy()
        mask[fora] = 0
```

Em `identify/pipeline.py:139`, trocar a chamada por:

```python
    x_px, y_px = mask_to_polyline(mask, bbox=cal.bbox_px if any(cal.bbox_px) else None)
```

O `any(...)` protege o caso `bbox_not_found`, em que `Calibration` traz o default `(0, 0, 0, 0)`.

- [ ] **Step 4: Rodar o teste e confirmar que passa**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_caso_real.py::test_polilinha_nao_sai_da_moldura -v
```

Esperado: PASS.

- [ ] **Step 5: Confirmar não-regressão no sintético**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_part2.py -q 2>&1 | tail -8
grep -E '^\| (2\.1 |2\.2-piso |2\.6 |2\.8 )' reports/part2_strata.md
```

Esperado, **sem nenhuma mudança de dígito**: 2.1 = `0.800 px / 1.699 px`; 2.2-piso = `RMSE=0.614 px, p95=1.135 px`; 2.6 = `+0.99 p.p.`; 2.8 mediana em torno de 183 ms. A máscara VERDADEIRA do sintético não tem nada fora da moldura, então o recorte é no-op ali; qualquer mudança nesses números indica que o recorte está cortando curva e a task precisa ser revista.

- [ ] **Step 6: Pedir autorização e commitar**

```bash
cd /home/loizm/work/TCC-2
git add identify/polyline.py identify/pipeline.py tests/part2/test_caso_real.py reports/part2_strata.md
git commit -m "fix: polilinha recortada a moldura do grafico (Ruling 55, defeito 1a)

Titulo e rotulo de eixo eram ingeridos como curva: no caso real a polilinha ia
de y=21 a 551 com a moldura em 39..503. bbox=None preserva o comportamento
anterior, e os numeros do sintetico ficam identicos."
```

---

### Task 3: Desambiguar colunas com mais de um ramo de tinta

Segundo pedaço do §34.2: a legenda tem uma amostra de linha da mesma cor da curva,
**dentro** da moldura. 37 colunas têm dois ramos (em x=536, `(169,172)` é a curva e
`(447,449)` é a legenda), e a mediana por coluna mistura os dois.

O mesmo mecanismo cobre a **reta de referência** onde ela passa perto da curva
(§34.2, trecho t ≈ 1,2 a 3): também é uma coluna com dois blocos de tinta, e a
escolha por proximidade ao ponto anterior segue a curva. Não cobre o trecho em que
a curva **coincide** com a reta — ali não há dois blocos, há um só, e é o defeito
da Task 5.

**A restrição que define esta task:** o Ruling 46 mediu que continuidade aplicada a
**todas** as colunas PIORA o sintético (p95 de 6,701 para 9,078 na métrica vertical).
Aquela variante trocava o valor também nas colunas de ramo único. Aqui a mudança
atinge **somente** colunas multi-ramo, que são 2,4 % do sintético (`frac_multi = 0,024`
na máscara verdadeira) e 22,3 % no caso real.

**Files:**
- Modify: `identify/polyline.py` (corpo de `mask_to_polyline`, o laço por coluna)
- Test: `tests/part2/test_caso_real.py`, `tests/part2/test_part2.py` (não-regressão)

**Interfaces:**
- Consumes: `mask_to_polyline(mask, bbox=None)` da Task 2.
- Produces: nenhuma assinatura nova. Comportamento novo: em coluna com ≥ 2 blocos contíguos de tinta, escolhe o bloco mais próximo do ponto anterior e usa a mediana **daquele bloco**; coluna de ramo único continua usando a mediana de todas as linhas, exatamente como antes.

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar em `tests/part2/test_caso_real.py`:

```python
def test_polilinha_ignora_amostra_de_linha_da_legenda(caso_real):
    """A legenda tem uma amostra da MESMA cor da curva, dentro da moldura.

    Na figura, a curva assenta em y = 1,0 (pixel ~172) e a amostra da legenda
    está em ~0,15 (pixel ~448). Sem desambiguação a mediana por coluna cai para
    o meio dos dois. O teste fixa o efeito no resultado: a cauda da polilinha
    tem de ficar no patamar, não no meio do caminho.
    """
    import torch
    from identify.calibrate import detect_plot_bbox
    from identify.extract import load_model, predict_mask
    from identify.polyline import mask_to_polyline
    from tests.part2.conftest import to_gray

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    bbox = detect_plot_bbox(to_gray(caso_real))
    x0, y0, x1, y1 = bbox
    mask = predict_mask(model, caso_real, dev)
    xp, yp = mask_to_polyline(mask, bbox=bbox)

    # patamar em y = 1,0 de um eixo 0,0..1,4: fração 0,2857 do topo da moldura
    y_patamar = y0 + (1.0 - 1.4) / (0.0 - 1.4) * (y1 - y0)
    cauda = yp[xp >= x0 + 0.55 * (x1 - x0)]
    assert cauda.size >= 10, "cauda curta"
    desvio = float(np.median(np.abs(cauda - y_patamar)))
    assert desvio <= 0.05 * (y1 - y0), (
        f"cauda a {desvio:.0f} px do patamar (limite {0.05 * (y1 - y0):.0f} px) — "
        "a amostra da legenda provavelmente está sendo misturada"
    )
```

- [ ] **Step 2: Rodar e confirmar que falha**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_caso_real.py::test_polilinha_ignora_amostra_de_linha_da_legenda -v
```

Esperado: FAIL, com a cauda longe do patamar.

- [ ] **Step 3: Implementar a desambiguação**

Em `identify/polyline.py`, acrescentar o helper antes de `mask_to_polyline`:

```python
def _blocos(coluna: np.ndarray) -> list[tuple[int, int]]:
    """Blocos contíguos de tinta numa coluna -> [(linha_inicial, linha_final)]."""
    idx = np.flatnonzero(coluna)
    if idx.size == 0:
        return []
    cortes = np.flatnonzero(np.diff(idx) > 1)
    blocos, ini = [], 0
    for c in cortes:
        blocos.append((int(idx[ini]), int(idx[c])))
        ini = c + 1
    blocos.append((int(idx[ini]), int(idx[-1])))
    return blocos
```

E substituir o laço por coluna:

```python
    xs, ys = [], []
    for x in range(skel.shape[1]):
        linhas = np.flatnonzero(skel[:, x])
        if linhas.size:
            xs.append(float(x))
            ys.append(float(np.median(linhas)))
```

por:

```python
    xs, ys = [], []
    anterior = None
    for x in range(skel.shape[1]):
        coluna = skel[:, x]
        linhas = np.flatnonzero(coluna)
        if not linhas.size:
            continue
        blocos = _blocos(coluna)
        if len(blocos) == 1 or anterior is None:
            # Ramo único: mediana de TODAS as linhas, idêntico ao comportamento
            # anterior. O Ruling 46 mediu que mexer aqui PIORA o sintético.
            v = float(np.median(linhas))
        else:
            # Ramo múltiplo (HANDOFF_P2_7 §34.2): a coluna tem mais de um objeto —
            # no caso real, a curva e a amostra de linha da legenda. Segue o bloco
            # mais próximo do ponto anterior e usa a mediana DAQUELE bloco.
            a, b = min(blocos,
                       key=lambda t: 0.0 if t[0] <= anterior <= t[1]
                       else min(abs(t[0] - anterior), abs(t[1] - anterior)))
            dentro = linhas[(linhas >= a) & (linhas <= b)]
            v = float(np.median(dentro)) if dentro.size else float(np.median(linhas))
        xs.append(float(x))
        ys.append(v)
        anterior = v
```

- [ ] **Step 4: Rodar o teste e confirmar que passa**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_caso_real.py -v
```

Esperado: `test_polilinha_ignora_amostra_de_linha_da_legenda` PASSA. Os dois testes da Task 1 podem continuar falhando — eles dependem também da Task 4.

- [ ] **Step 5: Confirmar não-regressão no sintético**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_part2.py -q 2>&1 | tail -8
grep -E '^\| (2\.1 |2\.2-piso |2\.2-piso-vertical|2\.6 )' reports/part2_strata.md
```

Esperado: 2.1 e 2.2-piso dentro de **±0,02 px** dos valores de referência (0,800/1,699 e 0,614/1,135), e 2.6 dentro de ±0,05 p.p. de +0,99. Colunas multi-ramo são 2,4 % do sintético, então o efeito tem de ser pequeno. **Se o 2.2-piso subir acima de 0,64 px, esta task recria a regressão do Ruling 46 e deve ser revertida** — nesse caso, restringir a desambiguação a colunas cujos blocos estejam separados por mais de `3 * espessura_mediana` px, e remedir.

- [ ] **Step 6: Pedir autorização e commitar**

```bash
cd /home/loizm/work/TCC-2
git add identify/polyline.py tests/part2/test_caso_real.py reports/part2_strata.md
git commit -m "fix: desambiguacao de colunas multi-ramo na polilinha (Ruling 55, defeito 1b)

Coluna com 2+ blocos de tinta segue o bloco mais proximo do ponto anterior. No
caso real a legenda tem amostra da mesma cor da curva, dentro da moldura, e 37
colunas tinham dois ramos. Coluna de ramo unico continua usando a mediana de
todas as linhas, porque o Ruling 46 mediu que mexer ali piora o sintetico."
```

---

### Task 4: Estimador de nível de repouso robusto a θ = 0

Segundo defeito, §34.3, e é meu: `_FRAC_REPOUSO = 0.08` supõe prefixo plano por
tempo morto. Com θ = 0 a curva já subiu 28 % dentro da janela de estimativa, o
patamar sai 3,8 % baixo, e isso vira 12,6 % de erro em ζ.

A escolha do estimador é **medida**, não arbitrada: o Step 1 varre candidatos nas
duas frentes e o Step 3 implementa o vencedor.

**Files:**
- Modify: `identify/pipeline.py` (`_FRAC_REPOUSO` e `_serie_normalizada`)
- Create: `reports/part2_repouso_varredura.md` (resultado da varredura)
- Test: `tests/part2/test_caso_real.py` (os dois testes da Task 1)

**Interfaces:**
- Consumes: `mask_to_polyline` das Tasks 2 e 3.
- Produces: `_nivel_de_repouso(y: np.ndarray) -> float` em `identify/pipeline.py`, usado por `_serie_normalizada`. Recebe o vetor `y` em pixels **já ordenado por x**; devolve o nível de repouso em pixels.

- [ ] **Step 1: Varrer candidatos nas DUAS frentes**

Criar `/tmp/varredura_repouso.py` e rodar:

```python
"""Escolhe o estimador de repouso medindo no CASO REAL e no SINTETICO."""
import numpy as np, torch
from pathlib import Path
from PIL import Image
from dataset.generator import load_sample
from identify.calibrate import calibrate, detect_plot_bbox
from identify.classical import identify as estagio_d
from identify.extract import load_model, predict_mask
from identify.polyline import mask_to_polyline
from tests.part2.conftest import to_gray

CAND = {
    "mediana 8% (ATUAL)": lambda y: float(np.median(y[:max(3, int(0.08 * y.size))])),
    "mediana 3%":         lambda y: float(np.median(y[:max(3, int(0.03 * y.size))])),
    "mediana 5 colunas":  lambda y: float(np.median(y[:min(5, y.size)])),
    "mediana 3 colunas":  lambda y: float(np.median(y[:min(3, y.size)])),
    "percentil 99":       lambda y: float(np.percentile(y, 99)),
}

def normaliza(x, y, f):
    k = np.argsort(x); x, y = np.asarray(x, float)[k], np.asarray(y, float)[k]
    span = float(x[-1] - x[0])
    if span <= 0: return None, None
    dev = f(y) - y
    esc = float(np.max(np.abs(dev)))
    if esc <= 0: return None, None
    return (x - x[0]) / span, dev / esc

dev_ = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("models/unet_stageA.pt", dev_)

# --- frente 1: caso real
img = np.asarray(Image.open("tests/fixtures/caso_real_2ordem.png").convert("RGB"))
bb = detect_plot_bbox(to_gray(img))
xr, yr = mask_to_polyline(predict_mask(model, img, dev_), bbox=bb)

# --- frente 2: sintetico, so as amostras SEM calibracao (onde a Decisao E atua)
sint = []
for d in sorted(Path("data/test").glob("sample_*"))[:300]:
    m = load_sample(d)
    if m["params"].get("zeta") is None: continue
    if calibrate(m["image"]).ok: continue
    bb2 = detect_plot_bbox(to_gray(m["image"]))
    xs, ys = mask_to_polyline(predict_mask(model, m["image"], dev_), bbox=bb2)
    if xs.size >= 10: sint.append((xs, ys, m["params"]["zeta"]))
print(f"sintetico sem calibracao, 2a ordem: n={len(sint)}\n")

print(f"{'candidato':22s}{'REAL zeta':>11s}{'erro':>8s}{'SINT MAPE':>11s}{'ordem ok':>10s}")
for nome, f in CAND.items():
    t, y = normaliza(xr, yr, f)
    fr = estagio_d(t, y) if t is not None else None
    zr = fr.params.get("zeta") if fr and fr.order == "second" else None
    real = f"{zr:.4f}" if zr else "—"
    err = f"{abs(zr - 0.5) / 0.5 * 100:.1f}%" if zr else "—"
    es, ordem = [], 0
    for xs, ys, zt in sint:
        ts, yss = normaliza(xs, ys, f)
        if ts is None: continue
        g = estagio_d(ts, yss)
        if g.order == "second":
            ordem += 1
            if g.params.get("zeta") is not None:
                es.append(abs(g.params["zeta"] - zt) / max(abs(zt), 1e-12) * 100)
    mape = f"{np.median(es):.2f}%" if es else "—"
    print(f"  {nome:20s}{real:>11s}{err:>8s}{mape:>11s}{ordem / max(len(sint),1):9.1%}")
```

```bash
cd /home/loizm/work/TCC-2
PYTHONPATH=. .venv/bin/python /tmp/varredura_repouso.py | tee reports/part2_repouso_varredura.md
```

**Critério de escolha, nesta ordem:** (a) acerta a ordem no caso real; (b) erro de ζ no caso real ≤ 5 %; (c) MAPE de ζ no sintético **não pior** que o do estimador atual. Um candidato que melhore o real e piore o sintético não é escolhido.

- [ ] **Step 2: Rodar os testes da Task 1 e confirmar que ainda falham**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_caso_real.py -v 2>&1 | tail -12
```

Esperado: `test_caso_real_acerta_a_ordem` e `test_caso_real_recupera_zeta_e_wn` continuam falhando — as Tasks 2 e 3 limparam a máscara, mas o estimador ainda está enviesado.

- [ ] **Step 3: Implementar o estimador vencedor**

Em `identify/pipeline.py`, substituir a constante:

```python
_FRAC_REPOUSO = 0.08
```

por (ajustar o corpo ao vencedor do Step 1 — o exemplo abaixo é a **mediana das
primeiras 5 colunas**, candidato mais provável por ser correto com θ = 0 e com
θ ≫ 0, e robusto a um pixel espúrio):

```python
# Nº de colunas iniciais usadas para estimar o nível de repouso.
# Antes era uma FRAÇÃO (8 %) da largura, o que supõe prefixo plano por tempo
# morto. Com θ = 0 a curva já subiu ~28 % dentro dessa janela, o patamar sai
# 3,8 % baixo e isso vira 12,6 % de erro em ζ, porque ζ vem da razão de
# overshoot (HANDOFF_P2_7 §34.3). Uma janela pequena e FIXA é correta nos dois
# regimes: com θ grande o prefixo é plano e 5 colunas caem nele; com θ = 0 as 5
# primeiras colunas ainda estão praticamente no repouso. A mediana (e não o
# primeiro valor) dá robustez a um pixel espúrio.
_N_REPOUSO = 5


def _nivel_de_repouso(y: np.ndarray) -> float:
    """Nível de repouso em pixels, a partir do início da série ordenada por x."""
    n = int(min(_N_REPOUSO, y.size))
    return float(np.median(y[:max(1, n)]))
```

E em `_serie_normalizada`, trocar:

```python
    n0 = max(3, int(_FRAC_REPOUSO * x.size))
    repouso = float(np.median(y[:n0]))
```

por:

```python
    repouso = _nivel_de_repouso(y)
```

- [ ] **Step 4: Rodar os testes do caso real e confirmar que passam**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_caso_real.py -v
```

Esperado: **os quatro testes passam.** ζ na casa de 0,50 com erro ≤ 5 % e ωₙ ≤ 5 %.

- [ ] **Step 5: Confirmar o sintético, com atenção ao 2.6-adim**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_part2.py -q 2>&1 | tail -8
grep -E '^\| 2\.6-adim' reports/part2_strata.md
```

Esperado: `2.6-adim[zeta]` **≤ +1,53 p.p.** (o valor commitado) e ainda aprovado. Uma melhora é o resultado desejado — o estimador antigo estava enviesado em toda amostra de θ pequeno. **Se piorar, reverter e escolher o segundo colocado do Step 1.**

- [ ] **Step 6: Pedir autorização e commitar**

```bash
cd /home/loizm/work/TCC-2
git add identify/pipeline.py tests/part2/test_caso_real.py reports/part2_repouso_varredura.md reports/part2_strata.md
git commit -m "fix: nivel de repouso robusto a theta=0 (Ruling 55, defeito 2)

_FRAC_REPOUSO = 0.08 supunha prefixo plano por tempo morto. Com theta=0 a curva
ja subiu ~28% dentro da janela, o patamar saia 3,8% baixo e isso virava 12,6% de
erro em zeta. Janela pequena e fixa e correta nos dois regimes. Estimador
escolhido por varredura nas duas frentes, nao por arbitrio."
```

---

### Task 5: Estrato novo no gerador — reta de referência no patamar

§34.5: o terceiro defeito (a U-Net perder a curva onde ela coincide com a reta de
referência) **não é consertável na polilinha** — não há tinta nas colunas perdidas.
Esta task não o conserta: torna-o **mensurável e treinável**, que é o pré-requisito.

`dataset/randomize.py:317-332` sorteia 1 a 3 distratores com `frac ~ U(0.03, 0.97)`,
"sem relação com o rótulo". Uma reta **no patamar** — o caso real corriqueiro do
*setpoint* marcado — quase nunca sai por acaso.

**Duas restrições duras, e a segunda invalida o desenho ingênuo:**

1. O estrato entra **opt-in**, com padrão idêntico ao atual, para que
   `data/train|val|test` regenerem byte a byte iguais e todos os números
   commitados continuem válidos.
2. **O parâmetro NÃO pode entrar em `sample_style`.** `tests/test_part1.py:1115`
   (`test_sample_style_signature_cannot_see_the_label`) afirma
   `names == ["rng"]` — é a guarda anti-vazamento estrutural do projeto, que
   garante que o sorteio de estilo não vê o `SystemSpec`. Acrescentar argumento
   ali reprova esse teste, e com razão.

   Logo o estrato entra no **render**, onde `render_sample` já tem `spec` e
   `style` em mãos. E é o lugar semanticamente correto: a reta de referência é
   fixada no **patamar da curva**, portanto depende do sistema — exatamente o que
   `sample_style` tem de continuar sem poder ver. Não há vazamento novo: a
   posição do patamar já é visível na imagem, é a assíntota da própria curva.

**Files:**
- Modify: `dataset/generator.py:266-274` (`render_sample` ganha o parâmetro), `:296-306` (render da reta), `:437-448` (`generate_sample` repassa), `:457-476` (`generate_dataset` repassa e entra no job)
- Test: `tests/part2/test_estrato_referencia.py` (criar)

**Interfaces:**
- Consumes: nada das tasks anteriores.
- Produces:
  - `render_sample(spec, style, out_dir, add_noise=True, rng=None, *, seed=None, reta_no_patamar=False)`
  - `generate_sample(out_dir, seed, add_noise=True, reta_no_patamar=False)`
  - `generate_dataset(out_dir, n, seed=0, workers=None, add_noise=True, reta_no_patamar=False)`
  - meta ganha `render.has_reference_line: bool`
  - `sample_style` fica **inalterada**.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/part2/test_estrato_referencia.py`:

```python
"""Estrato OOD: reta de referência COINCIDENTE com o patamar (Ruling 55 §34.5).

O gerador sorteia distratores em posição uniforme; uma reta no patamar — o caso
real do setpoint marcado — quase nunca sai por acaso, e é exatamente onde a U-Net
perde a curva. Este teste fixa o contrato do estrato novo E a invariante
anti-vazamento que ele não pode violar.
"""
import inspect
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import generator, randomize


def test_sample_style_continua_cega_ao_sistema():
    """Regressão da guarda de tests/test_part1.py:1115. O estrato entra no
    RENDER, não no sorteio de estilo."""
    nomes = list(inspect.signature(randomize.sample_style).parameters)
    assert nomes == ["rng"], f"assinatura de sample_style mudou: {nomes}"


def test_padrao_nao_gera_reta_de_referencia():
    """Sem opt-in, o meta declara ausência da reta."""
    tmp = Path(tempfile.mkdtemp())
    try:
        meta = generator.generate_sample(tmp / "s", seed=7)
        assert meta["render"]["has_reference_line"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_opt_in_gera_reta_e_declara_no_meta():
    tmp = Path(tempfile.mkdtemp())
    try:
        meta = generator.generate_sample(tmp / "s", seed=7, reta_no_patamar=True)
        assert meta["render"]["has_reference_line"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_padrao_reproduz_bit_a_bit():
    """O opt-in não pode alterar UM BYTE do caminho padrão: os splits
    data/train|val|test e todos os números medidos dependem disso."""
    tmp = Path(tempfile.mkdtemp())
    try:
        generator.generate_sample(tmp / "a", seed=99)
        generator.generate_sample(tmp / "b", seed=99, reta_no_patamar=False)
        for nome in ("image.png", "mask.png"):
            assert (tmp / "a" / nome).read_bytes() == (tmp / "b" / nome).read_bytes(), \
                f"{nome} mudou com reta_no_patamar=False"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reta_fica_no_patamar_da_curva():
    """A reta tem de coincidir com o valor final da curva, não com posição
    uniforme — é essa coincidência que produz o defeito do §34.5."""
    tmp = Path(tempfile.mkdtemp())
    try:
        meta = generator.generate_sample(tmp / "s", seed=11, reta_no_patamar=True)
        y = np.asarray(meta["series"]["y"], dtype=float)
        a = meta["axis_affine"]
        # linha de pixels do patamar, e a linha com mais tinta NAO-curva na imagem
        with Image.open(tmp / "s" / "image.png") as im:
            img = np.asarray(im.convert("L"), dtype=np.uint8)
        py_patamar = (float(y[-1]) - a["oy"]) / a["sy"]
        # a reta de span completo produz uma linha com tinta em quase toda a largura
        escura = (img < 200).sum(axis=1)
        linha = int(np.argmax(escura))
        assert abs(linha - py_patamar) <= 4, (
            f"linha mais cheia em y={linha}, patamar em y={py_patamar:.0f}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

- [ ] **Step 2: Rodar e confirmar que falha**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_estrato_referencia.py -v
```

Esperado: `test_sample_style_continua_cega_ao_sistema` e `test_padrao_reproduz_bit_a_bit` **passam** (nada mudou ainda); os três restantes falham — dois com `TypeError: generate_sample() got an unexpected keyword argument 'reta_no_patamar'` e `test_padrao_nao_gera_reta_de_referencia` com `KeyError: 'has_reference_line'`.

- [ ] **Step 3: Implementar o estrato no RENDER**

Em `dataset/randomize.py`, no dataclass `RenderStyle` (linha ~203, junto de `distractors`):

```python
    has_reference_line: bool = False
```

E na `to_meta` (linha ~245, dentro do dict retornado):

```python
            "has_reference_line": bool(self.has_reference_line),
```

`sample_style` **não é tocada** — o default `False` do dataclass cobre o caminho padrão.

Em `dataset/generator.py`, na assinatura de `render_sample` (linha 266-274), acrescentar
ao bloco keyword-only:

```python
def render_sample(
    spec: SystemSpec,
    style: RenderStyle,
    out_dir: str | Path,
    add_noise: bool = True,
    rng: np.random.Generator | None = None,
    *,
    seed: int | None = None,
    reta_no_patamar: bool = False,
) -> dict:
```

Logo depois de `xlim, ylim = _axis_limits(t, y_draw, style)` (linha 285), acrescentar:

```python
    # Estrato OOD, opt-in (HANDOFF_P2_7 §34.5). O laço de distratores abaixo
    # sorteia posição uniforme; aqui se acrescenta uma reta horizontal FIXADA no
    # patamar da curva, que é o caso real do setpoint marcado e onde a U-Net perde
    # a curva. Entra no RENDER e não em `sample_style`, porque a posição depende do
    # sistema e `sample_style` tem de continuar cega ao spec
    # (tests/test_part1.py:1115). Sob `if`, para que o caminho padrão não mude um byte.
    if reta_no_patamar:
        style.has_reference_line = True
        style.distractors = list(style.distractors) + [
            {"orient": "h", "frac": None, "no_patamar": True,
             "color": "#d62728", "line_style": "--",
             "line_width": 1.5, "alpha": 0.9}
        ]
```

No laço de distratores (linha 296-306), tratar `frac=None`:

```python
    for d in style.distractors:
        if d["orient"] == "h":
            if d.get("no_patamar"):
                val = float(y_draw[-1])     # o patamar da própria curva
            else:
                val = ylim[0] + d["frac"] * (ylim[1] - ylim[0])
            ax.axhline(
```

Em `generate_sample` (linha 437-448):

```python
def generate_sample(out_dir: str | Path, seed: int, add_noise: bool = True,
                    reta_no_patamar: bool = False) -> dict:
```

e na chamada a `render_sample` (linha 447-448):

```python
    return render_sample(spec, style, out_dir, add_noise=add_noise, rng=rng_noise,
                         seed=int(seed), reta_no_patamar=reta_no_patamar)
```

Em `generate_dataset` (linha 457-476), acrescentar o parâmetro e propagá-lo pelo job.
`_generate_one` recebe uma tupla, então a tupla ganha um campo:

```python
def generate_dataset(
    out_dir: str | Path,
    n: int,
    seed: int = 0,
    workers: int | None = None,
    add_noise: bool = True,
    reta_no_patamar: bool = False,
) -> list[str]:
    """Gera n amostras em paralelo. Resultado independe do numero de workers."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(root / f"sample_{i:05d}"), int(seed) * 1_000_003 + i, bool(add_noise),
         bool(reta_no_patamar))
        for i in range(int(n))
    ]
```

E `_generate_one` (linha 451-454) passa a desempacotar quatro campos:

```python
def _generate_one(args: tuple) -> str:
    out_dir, seed, add_noise, reta = args
    generate_sample(out_dir, seed, add_noise=add_noise, reta_no_patamar=reta)
    return str(out_dir)
```

- [ ] **Step 4: Rodar os testes e confirmar que passam**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/part2/test_estrato_referencia.py -v
```

Esperado: os cinco PASSAM, com destaque para `test_padrao_reproduz_bit_a_bit` e
`test_sample_style_continua_cega_ao_sistema`.

- [ ] **Step 5: Provar que a Parte 1 e os splits existentes não mudaram**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest tests/test_part1.py tests/test_leakage.py -q 2>&1 | tail -5
.venv/bin/python -m pytest tests/part2/test_env.py -q
```

Esperado: Parte 1 e vazamento **sem nenhuma falha** (inclui a guarda
`test_sample_style_signature_cannot_see_the_label`) e os 5 portões do Bloco 0
passando. **Este é o passo que protege todo o histórico de medições.**

- [ ] **Step 6: Gerar o split OOD e quantificar o defeito**

```bash
cd /home/loizm/work/TCC-2
PYTHONPATH=. .venv/bin/python -c "
from dataset.generator import generate_dataset
generate_dataset('data/ood_referencia', 200, 7, reta_no_patamar=True)
print('gerado')
"
PYTHONPATH=. .venv/bin/python -c "
import numpy as np, torch
from pathlib import Path
from dataset.generator import load_sample
from identify.extract import load_model, predict_mask
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model('models/unet_stageA.pt', dev)
cob = []
for d in sorted(Path('data/ood_referencia').glob('sample_*'))[:200]:
    m = load_sample(d)
    p = predict_mask(model, m['image'], dev) > 127
    t = m['mask'] > 127
    cob.append(np.flatnonzero(p.sum(0)).size / max(np.flatnonzero(t.sum(0)).size, 1))
print(f'cobertura de colunas, split OOD: mediana={np.median(cob):.3f} p10={np.percentile(cob,10):.3f}')
"
```

Esperado: cobertura **abaixo de 1,0** — no caso real foi 502/746 = 0,67. É a
métrica que quantifica o defeito e que uma rodada de treino futura tem de melhorar.
Se a cobertura vier perto de 1,0, o estrato **não** reproduziu o caso real e a
Task 5 precisa ser revista antes de servir de base para treino.

- [ ] **Step 7: Pedir autorização e commitar**

`data/` é ignorado pelo git, então o split OOD não entra no commit.

```bash
cd /home/loizm/work/TCC-2
git add dataset/randomize.py dataset/generator.py tests/part2/test_estrato_referencia.py
git commit -m "feat: estrato OOD opt-in — reta de referencia no patamar (Ruling 55 §34.5)

O gerador sorteava distratores em posicao uniforme; uma reta no patamar — o caso
real do setpoint marcado, onde a U-Net perde a curva — quase nunca saia por
acaso.

Entra no RENDER, nao em sample_style: tests/test_part1.py:1115 afirma que
sample_style recebe SO o rng, e essa e a guarda anti-vazamento estrutural do
projeto. A posicao da reta depende do sistema (e o patamar da curva), logo e
decisao de render por construcao. Nao ha vazamento novo: o patamar ja e visivel
na imagem, e a assintota da propria curva.

reta_no_patamar=False por padrao e sob if, para que data/train|val|test
reproduzam byte a byte. Nao conserta o defeito: torna-o mensuravel e treinavel."
```

---

### Task 6: Revalidação completa e registro

**Files:**
- Modify: `HANDOFF_P2_7.md` (acrescentar §35)
- Modify: `reports/part2_strata.md` (regenerado pela suíte)

**Interfaces:**
- Consumes: todas as tasks anteriores.
- Produces: nada de código.

- [ ] **Step 1: Rodar a suíte inteira, sem filtros**

```bash
cd /home/loizm/work/TCC-2
.venv/bin/python -m pytest -q 2>&1 | tail -12
cp reports/part2_strata.md reports/part2_strata_pos_caso_real.md
```

Esperado: **no máximo 2 failed** (2.5 e 2.9, os que já reprovavam), **57+ passed**, mais os testes novos do caso real e do estrato. Qualquer falha nova é regressão e bloqueia a task.

- [ ] **Step 2: Comparar cada número contra a referência**

```bash
cd /home/loizm/work/TCC-2
diff <(sed 's/|$//' reports/part2_strata_26adim.md) <(sed 's/|$//' reports/part2_strata_pos_caso_real.md) \
  | grep -E '^[<>]' | sed 's/^</ANTES /;s/^>/AGORA /'
```

Esperado: mudanças **apenas** em `2.6-adim[zeta]` (melhora, ou igual) e possivelmente ±0,02 px em 2.1/2.2. Qualquer outro critério que se mova precisa de explicação escrita antes do commit.

- [ ] **Step 3: Escrever o §35 no handoff**

Acrescentar em `HANDOFF_P2_7.md` uma seção `## 35. Ruling 56 — as correções do caso real` contendo, com os números reais medidos: o antes/depois de ζ e ωₙ no caso real; o antes/depois de `2.6-adim[zeta]`; o estimador de repouso escolhido e por quê (tabela do Step 1 da Task 4); a cobertura de colunas no split OOD; e a declaração explícita de que a perda de cauda sob reta coincidente **continua não consertada**, apenas mensurável.

- [ ] **Step 4: Pedir autorização e commitar**

```bash
cd /home/loizm/work/TCC-2
git add HANDOFF_P2_7.md reports/part2_strata.md reports/part2_strata_pos_caso_real.md
git commit -m "Parte 2 / Bloco 7: registro das correcoes do caso real (Ruling 56)"
```

---

## Notas de execução

**O que este plano NÃO faz, e é deliberado:**

1. **Não retreina a U-Net.** A perda de cauda sob reta coincidente é limitação de
   máscara; consertá-la exige uma rodada com o estrato novo, que só faz sentido
   depois de a Task 5 tornar o estrato disponível. Fica para um bloco próprio.
2. **Não mexe no `mask_to_polyline` para colunas de ramo único.** O Ruling 46 mediu
   19 variantes e todas pioraram; a Task 3 é a única mudança que não toca esse
   caminho.
3. **Não altera critério nenhum.** As metas revisadas do §2.12 do `PLANO.md`
   permanecem como estão; este plano conserta implementação, não régua.

**Ordem de dependência:** 1 → 2 → 3 → 4 são sequenciais (cada uma limpa um elo que a
seguinte precisa). A Task 5 é independente e pode ser executada em paralelo. A
Task 6 exige todas.
