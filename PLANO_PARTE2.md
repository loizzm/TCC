# Parte 2 — Plano de execução em blocos

> **Para executores (humanos ou agentes):** este documento **não altera** o `PLANO.md`.
> Ele decompõe a Parte 2 em blocos entregáveis, cada um com portão numérico
> próprio e um **handoff** escrito ao final. Os alvos dos critérios 2.1 a 2.11 são
> citados verbatim do `PLANO.md §PARTE 2` e **não podem ser negociados aqui**.
> Os passos usam caixas (`- [ ]`) para acompanhamento.
>
> **Revisão de 22/08/2026 — leia antes de executar.** O `PLANO.md` foi revisado e
> três decisões novas mudam este plano:
> - **§1.3 (Decisão C):** o Estágio C foi medido e **removido**. O pipeline tem três
>   estágios, A → B → D. Nada neste documento muda por isso (a Parte 2 nunca
>   construiu o C), mas o handoff final do Bloco 5 muda — ver Bloco 5, Passo 8.
> - **§1.7 (Decisão E):** o OCR passa a ser **opcional**, não estrutural. Afeta o
>   Bloco 2 e acrescenta os critérios **2.9** e **2.11**.
> - **§1.8 (Decisão F):** um extrator clássico sem rede entra como Plano B do risco
>   de GPU e como baseline do Estágio A. Acrescenta o **Bloco 3b** e o critério **2.10**.

**Objetivo:** substituir o oráculo da Parte 1 por percepção real (Estágios A e B) e
medir quanto o pipeline degrada, com o critério 2.6 (ΔMAPE ≤ 3 p.p.) como veredito.

**Arquitetura:** `imagem → [B] calibração dos eixos (determinístico + OCR opcional) → [A]
segmentação da curva (U-Net, ou extrator clássico) → polilinha → série y(t) → [D]
`identify()` da Parte 1, inalterado`. O Estágio B vem antes do A na ordem de
execução porque é determinístico, não treina e dá feedback em segundos — é o
conselho registrado em `HANDOFF.md §9, item 5`.

**Não há Estágio C.** O pipeline entregue pela Parte 2 já é o pipeline final:
`identify()` consome a série extraída diretamente. Ver `PLANO.md §1.3`.

**Stack:** Python 3.11 (`.venv`), PyTorch (CPU ou CUDA), OpenCV, scikit-image,
Tesseract via `pytesseract`, NumPy/SciPy, pytest.

**Spec:** `PLANO.md` §PARTE 2 (critérios 2.1–2.11) e §1.1–1.3, §1.7, §1.8
(decisões A, B, C, E, F); `HANDOFF.md` §4 (o que a Parte 1 entrega) e §9 (ordem
de retomada).
O plano argumenta a partir da spec — leia os dois.

---

## Restrições globais

Valem para **todos** os blocos. Copiadas do `PLANO.md`, do `HANDOFF.md §8` e das
convenções verificadas no código da Parte 1.

- **Python 3.11 no `.venv` do repositório.** Todo comando é `.venv/bin/python -m ...`.
- **Nunca `np.random` global.** Sempre `np.random.default_rng(seed)` ou
  `SeedSequence(seed).spawn(n)`. Vale também para inicialização de pesos e
  embaralhamento de lote no PyTorch (`torch.manual_seed`).
- **Nada de `time`, `uuid` ou hash dependente de `PYTHONHASHSEED`** em código que
  afete um resultado medido. Tempo só é permitido para medir latência (critério 2.8).
- **Nenhuma função do pipeline levanta exceção em amostra malformada.** Falha vira
  bandeira no resultado (`ok=False`, `reason=...`), nunca `raise`. Regra do
  `contract.md §6`, citada em `identify/classical.py`.
- **Convenção de pixel:** origem no canto **superior** esquerdo; o centro do pixel
  de índice `i` fica na coordenada contínua `i + 0.5`.
- **Convenção da afim** (`meta["axis_affine"]`), validada geometricamente na Parte 1
  a 0,03 px de viés — **não reinventar**:
  `x_dados = sx * x_px + ox` e `y_dados = sy * y_px + oy`, com **`sy < 0`**.
- **`meta["plot_bbox_px"] = [x0, y0, x1, y1]`**, índices de pixel, origem no topo,
  medidos no **centro da spine** desenhada.
- **`meta["ticks"]["x"] = [[coord_px, valor], ...]`** — coordenada horizontal para o
  eixo x, vertical (origem no topo) para o eixo y.
- **`left` e `bottom` sempre existem; `right` e `top` aparecem com p = 0,45 cada**
  (`dataset/randomize.py:289`). Logo `n_spines ∈ {2, 3, 4}` e a moldura **nunca**
  está completamente ausente — mas o detector não pode assumir 4 lados.
- **O repositório não é um git repo.** Não existe `git commit` a executar. O passo de
  "commit" de cada tarefa é substituído por **rodar a suíte e escrever o handoff**.
  Não rodar `git init`/`git commit`/`git push` sem pedir autorização explícita.
- **Não editar `PLANO.md`.** Divergências viram *Ruling* registrado no handoff do
  bloco, no formato já usado em `HANDOFF.md §6`.
- **Todo número citado vem de execução real**, registrado via `record_criterion(...)`
  e impresso no relatório. Nada de valor estimado em prosa.
- **Princípio herdado da Parte 1 (`HANDOFF.md §8`):** *um critério que não reprova
  nenhum mutante não é um critério, é decoração.* Todo portão novo deste plano é
  validado contra um defeito injetado antes de ser considerado pronto.

---

## Mapa dos blocos

| Bloco | Entregável | Critérios do PLANO | Precisa de GPU? | Trabalho | Parede |
|---|---|---|---|---|---|
| **0** | Ambiente + dataset em disco + guarda do relatório | — (portões próprios G0.1–G0.4) | decide | 0,5–1 dia | +3 h de timebox |
| **1** | Estágio B, geometria: moldura e ticks, **sem OCR** | — (portões G1.1–G1.3) | não | 1 dia | minutos |
| **2** | Estágio B completo: OCR **opcional** + RANSAC + consistência + saída em dois níveis | **2.3, 2.4, 2.5, 2.9, 2.11** | não | 1–1,5 dia | minutos |
| **3b** | Estágio A **sem rede**: extrator clássico (§1.8) | **2.10** (metade) | **não** | 0,5–1 dia | minutos |
| **3** | Estágio A: U-Net, perda Dice+BCE, treino | **2.1, 2.7, 2.10** | **sim, muda o prazo — não mais o escopo** | 0,5 dia | 20–35 h/rodada em CPU; 0,5–1 h com GPU |
| **4** | Estágio A: pós-processamento → polilinha | **2.2** | não | 1 dia | minutos |
| **5** | Integração A+B+D, degradação, relatório | **2.6, 2.8** | não | 1,5–2 dias | ~10 min de suíte |

**Grafo de dependência** — o que pode andar em paralelo:

```
Bloco 0 ──┬─> Bloco 1 ──> Bloco 2 ──────┐
          │                              │
          ├─> Bloco 3b (clássico) ───────┤
          │        │                     ├─> Bloco 5
          ├─> Bloco 3  (U-Net, treino) ──┤
          └─> Bloco 4 ───────────────────┘
             (contra mask.png verdadeira,
              não espera o treino do Bloco 3)
```

**O Bloco 3b é o que tira a GPU do caminho crítico.** Ele produz uma máscara com a
mesma assinatura de `predict_mask`, então o Bloco 5 fecha — e a Parte 2 entrega —
**mesmo que o Bloco 3 nunca termine**. A U-Net deixa de ser pré-requisito e passa a
ser melhoria mensurável (critério 2.10). Se o driver NVIDIA não subir no timebox do
Bloco 0, execute 3b e siga; o Bloco 3 vira trabalho de fundo.

O ganho de escalonamento está no Bloco 4: ele se desenvolve contra a
`mask.png` **verdadeira** da Parte 1, que é a segmentação ideal. Então ele pode
ser escrito e medido **enquanto a U-Net do Bloco 3 treina a noite inteira**. Não
serialize os dois.

---

## Convenções de handoff

Cada bloco termina escrevendo `HANDOFF_P2_<n>.md` na raiz, com **exatamente** estas
seções — o formato espelha o `HANDOFF.md` da Parte 1, que já provou funcionar para
retomar trabalho em outra sessão:

1. **Estado** — o que ficou pronto, em uma tabela `componente | estado | evidência`.
2. **Interface publicada** — assinaturas exatas que o próximo bloco vai consumir,
   copiadas do código, não parafraseadas.
3. **Números medidos** — cada portão do bloco com alvo, valor medido e veredito.
   Nunca um número que não saiu de uma execução.
4. **Rulings** — toda decisão que divergiu do `PLANO.md`, com a justificativa e a
   medição que a motivou.
5. **Armadilhas** — o que quebrou no caminho e o que o próximo executor não deve
   repetir.
6. **O que o próximo bloco precisa saber** — em imperativo, no formato do `§9`.

Regra: **o handoff é escrito antes de o bloco ser declarado pronto**, não depois.
Se ele não puder ser escrito sem inventar um número, o bloco não terminou.

---

## Estrutura de arquivos

| Arquivo | Responsabilidade | Bloco |
|---|---|---|
| `identify/calibrate.py` | Estágio B inteiro: moldura, ticks, OCR opcional, RANSAC, consistência | 1, 2 |
| `identify/extract_classical.py` | Estágio A sem rede: cor modal, componentes, rejeição de retas (§1.8) | 3b |
| `identify/extract.py` | Estágio A: U-Net, letterbox, perda, inferência de máscara | 3 |
| `identify/polyline.py` | Máscara → polilinha → série física (pós-processamento puro) | 4 |
| `identify/pipeline.py` | Cola A+B+D; a única função que a Parte 3 vai importar. Saída em dois níveis (§1.7) | 5 |
| `train_unet.py` (raiz) | Script de treino, fora do pacote (não é biblioteca) | 3 |
| `tests/part2/conftest.py` | Fixtures da Parte 2 + geração de `reports/part2_strata.md` | 1→5 |
| `tests/part2/test_part2.py` | Critérios 2.1 a 2.11 | 1→5 |
| `tests/conftest.py` | **modificar**: guarda contra sobrescrever o relatório da Parte 1 | 0 |

Motivo de `polyline.py` separado de `extract.py`: o pós-processamento é
determinístico, não importa `torch` e é testável contra a máscara verdadeira sem
modelo nenhum. Fundi-los amarraria o Bloco 4 ao término do treino do Bloco 3, que é
exatamente a serialização que o grafo acima evita.

---

# Bloco 0 — Infraestrutura, dataset em disco e guarda do relatório

**Por que primeiro:** é a dívida do Dia 1 do cronograma que a Parte 1 nunca pagou.
`data/` está vazio, `torch`, `cv2` e `tesseract` não existem no ambiente, e
`nvidia-smi` não existe na máquina. Nada da Parte 2 anda sem isto.

> **Revisão de 22/08/2026:** o timebox de 3 h do driver deixou de ser crítico. Se
> estourar, siga para o Bloco 3b (extrator clássico, `PLANO.md §1.8`) e trate o
> Bloco 3 como trabalho de fundo. `torch` continua na lista de dependências, mas
> **não** é bloqueio para fechar a Parte 2. O portão G0.2 continua valendo: a
> decisão de dispositivo tem de ficar registrada por escrito de todo modo.

**Arquivos:**
- Modificar: `requirements.txt`
- Modificar: `tests/conftest.py:1049-1062` (guarda do `pytest_sessionfinish`)
- Criar: `data/train/`, `data/val/`, `data/test/` (artefatos, não versionados)
- Criar: `HANDOFF_P2_0.md`

**Interfaces:**
- Consome: `dataset.generator.generate_dataset(out_dir, n, seed, workers, add_noise)`
- Produz: três diretórios de amostras com seeds disjuntos; a variável de ambiente
  de decisão `TCC_DEVICE ∈ {"cuda", "cpu"}` registrada no handoff.

### Portões do bloco

| # | Portão | Alvo |
|---|---|---|
| G0.1 | `import torch, cv2, skimage, pytesseract` sem erro | passa |
| G0.2 | Decisão de dispositivo tomada e registrada | `cuda` ou `cpu` **por escrito** |
| G0.3 | 6.000 amostras em disco, splits disjuntos | 4.200/900/900, nenhum `sample_id` repetido |
| G0.4 | Rodar só a Parte 2 **não** corrompe `reports/part1_metrics.md` | arquivo intacto byte a byte |

- [ ] **Passo 1: timebox de 3 h para o driver NVIDIA**

O `PLANO.md §4, Dia 1` prevê exatamente isto, e o §5 registra o plano B. Fedora 43
com Secure Boot exige assinatura MOK; o `akmod-nvidia` do RPM Fusion cuida disso.
Este passo precisa de root — peça ao usuário rodar na sessão:

```
! sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
```

Depois reiniciar e verificar. **Encerre o timebox em 3 h de relógio, sem exceção** —
o §5 do PLANO já autorizou a desistência.

- [ ] **Passo 2: verificar a decisão de dispositivo**

Run: `nvidia-smi -L && .venv/bin/python -c "import torch; print(torch.cuda.is_available())"`
Esperado: ou `True` (siga por `cuda`), ou o comando falha (siga por `cpu`).

Escreva a decisão no handoff com a data. Ela muda a estimativa do Bloco 3 em
cerca de quatro dias e **não pode ficar implícita**.

- [ ] **Passo 3: instalar as dependências**

Se `cuda` disponível:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Se `cpu`:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Em ambos os casos:

```bash
.venv/bin/pip install opencv-python-headless scikit-image pytesseract
```

E o binário do Tesseract, que é pacote do sistema (peça ao usuário):

```
! sudo dnf install -y tesseract tesseract-langpack-eng
```

- [ ] **Passo 4: congelar o ambiente**

```bash
.venv/bin/pip freeze > requirements.txt
```

- [ ] **Passo 5: escrever o teste de fumaça do ambiente**

Crie `tests/part2/__init__.py` (vazio) e `tests/part2/test_env.py`:

```python
"""Portão G0.1/G0.2: o ambiente da Parte 2 existe e o dispositivo é conhecido."""
import shutil

import pytest


def test_g0_1_dependencias_importam():
    import cv2            # noqa: F401
    import pytesseract    # noqa: F401
    import skimage        # noqa: F401
    import torch          # noqa: F401


def test_g0_1_binario_tesseract_no_path():
    assert shutil.which("tesseract") is not None, (
        "instale o pacote de sistema: sudo dnf install tesseract"
    )


def test_g0_2_dispositivo_declarado():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev in ("cuda", "cpu")
    print(f"TCC_DEVICE={dev}")
```

- [ ] **Passo 6: rodar o teste de fumaça**

Run: `.venv/bin/python -m pytest tests/part2/test_env.py -v -s`
Esperado: 3 passed, e a linha `TCC_DEVICE=...` no stdout.

- [ ] **Passo 7: fechar o buraco que corrompe o relatório da Parte 1**

**Este é um defeito real, encontrado ao ler o código, e ele morde já no Bloco 1.**
`tests/conftest.py:1049` escreve `reports/part1_metrics.md` em **toda**
`pytest_sessionfinish`. O banner de "relatório parcial" só dispara com `-m`/`-k`
(`tests/conftest.py:1052-1059`) — **não** com seleção por caminho. Então
`pytest tests/part2` hoje sobrescreve silenciosamente o relatório da Parte 1 com
um relatório vazio, sem aviso. Esse relatório é a linha de base contra a qual o
critério 2.6 mede degradação: perdê-lo custa 128 s de re-execução na melhor
hipótese, e a comparação inteira na pior.

Escreva primeiro o teste que reprova, em `tests/part2/test_env.py`:

```python
def test_g0_4_rodar_parte2_nao_apaga_relatorio_da_parte1(tmp_path):
    """Portão G0.4: sessão sem critérios da Parte 1 não reescreve o relatório."""
    import tests.conftest as c1

    saved_criteria = dict(c1.RESULTS["criteria"])
    c1.RESULTS["criteria"] = {}
    try:
        antes = c1.REPORT_PATH.read_bytes() if c1.REPORT_PATH.exists() else None
        c1.pytest_sessionfinish(_FakeSession(), 0)
        depois = c1.REPORT_PATH.read_bytes() if c1.REPORT_PATH.exists() else None
        assert antes == depois, "o relatório da Parte 1 foi alterado"
    finally:
        c1.RESULTS["criteria"] = saved_criteria


class _FakeSession:
    class config:
        class option:
            markexpr = ""
            keyword = ""
        args = ["tests/part2"]
```

- [ ] **Passo 8: rodar o teste para confirmar que ele reprova**

Run: `.venv/bin/python -m pytest tests/part2/test_env.py::test_g0_4_rodar_parte2_nao_apaga_relatorio_da_parte1 -v`
Esperado: FAIL — o relatório é reescrito e os bytes diferem.

- [ ] **Passo 9: aplicar a guarda mínima**

Em `tests/conftest.py`, dentro de `pytest_sessionfinish`, antes do `try` que chama
`_write_report()`:

```python
    # Sessão que não mediu nenhum critério da Parte 1 (ex.: `pytest tests/part2`)
    # NÃO pode reescrever o relatório: ele é a linha de base do critério 2.6.
    if not RESULTS["criteria"]:
        return
```

E estenda a detecção de seleção parcial para incluir caminho, logo abaixo do
cálculo de `sel` existente:

```python
    args = [a for a in getattr(session.config, "args", []) if not a.startswith("-")]
    if args and args != ["tests"]:
        sel = (sel + " " if sel else "") + f"paths {args!r}"
```

- [ ] **Passo 10: rodar os testes para confirmar que passam**

Run: `.venv/bin/python -m pytest tests/part2/test_env.py -v`
Esperado: 4 passed.

- [ ] **Passo 11: confirmar que a Parte 1 continua verde**

Run: `.venv/bin/python -m pytest -q`
Esperado: **33 passed** em ~130 s, e `reports/part1_metrics.md` regenerado sem
banner de parcialidade. Se der 34, você contou o `tests/part2` — confira que os
testes novos estão sob `tests/part2/`.

- [ ] **Passo 12: gerar os três splits**

Seeds disjuntos por construção: `generate_dataset` deriva
`seed * 1_000_003 + i` (`dataset/generator.py:461`), então basta separar as bases.

```bash
.venv/bin/python -m dataset.generator data/train 4200 1
.venv/bin/python -m dataset.generator data/val    900 2
.venv/bin/python -m dataset.generator data/test   900 3
```

Esperado: ~30 min no total (critério 1.7 do PLANO, medido em 6.000 imagens),
~2 GB em disco. Você tem 305 GB livres.

- [ ] **Passo 13: escrever e rodar o teste de disjunção dos splits**

Em `tests/part2/test_env.py`:

```python
def test_g0_3_splits_disjuntos_e_completos():
    """Portão G0.3: 4200/900/900 e nenhuma seed compartilhada entre splits."""
    from pathlib import Path
    import json

    esperado = {"train": 4200, "val": 900, "test": 900}
    seeds: dict[str, set[int]] = {}
    for split, n in esperado.items():
        root = Path("data") / split
        dirs = sorted(root.glob("sample_*"))
        assert len(dirs) == n, f"{split}: {len(dirs)} != {n}"
        seeds[split] = {
            json.loads((d / "meta.json").read_text(encoding="utf-8"))["seed"]
            for d in dirs
        }
    assert not seeds["train"] & seeds["val"]
    assert not seeds["train"] & seeds["test"]
    assert not seeds["val"] & seeds["test"]
```

Run: `.venv/bin/python -m pytest tests/part2/test_env.py -v`
Esperado: 5 passed.

- [ ] **Passo 14: escrever `HANDOFF_P2_0.md`**

Obrigatório registrar: a decisão `cuda`/`cpu` **com a data e o motivo**, a versão
do torch instalada, a versão do Tesseract (`tesseract --version`), o tempo real de
geração dos splits, o espaço em disco ocupado, e o Ruling da guarda do relatório
(divergência: `tests/conftest.py` foi tocado, o que a Parte 1 dava por fechado).

---

# Bloco 1 — Estágio B, geometria: moldura e ticks, sem OCR

**Por que separado do OCR:** a detecção geométrica tem verdade de terra exata no
meta (`plot_bbox_px`, `ticks`) e roda em milissegundos. Isolá-la significa que,
quando o critério 2.3 falhar no Bloco 2, você já sabe que não é a geometria — é o
OCR. Sem esse corte, os dois erros se confundem e o diagnóstico some.

**Arquivos:**
- Criar: `identify/calibrate.py`
- Criar: `tests/part2/conftest.py`
- Modificar: `tests/part2/test_part2.py` (criar)
- Criar: `HANDOFF_P2_1.md`

**Interfaces:**
- Consome: `dataset.generator.load_sample(sample_dir) -> dict` com chaves
  `image` (uint8 HxWx3), `mask` (uint8 HxW), `plot_bbox_px`, `ticks`, `axis_affine`.
- Produz, para o Bloco 2:
  - `detect_plot_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None`
    — `(x0, y0, x1, y1)`, mesma convenção de `meta["plot_bbox_px"]`.
  - `detect_tick_pixels(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, list[float]]`
    — `{"x": [px, ...], "y": [py, ...]}`, coordenadas contínuas, origem no topo.

### Portões do bloco

| # | Portão | Alvo |
|---|---|---|
| G1.1 | Erro da moldura vs. `plot_bbox_px`, por lado | ≤ 2 px em ≥ 95% das amostras |
| G1.2 | *Recall* de ticks maiores (casados a ≤ 3 px do verdadeiro) | ≥ 0,95 por eixo, mediana |
| G1.3 | Estratificação de G1.1 por `n_spines ∈ {2, 3, 4}` | nenhum estrato < 0,90 |

G1.3 existe porque `right` e `top` só aparecem com p = 0,45 (`randomize.py:289`).
Um detector que assume moldura fechada passa em ~20% das amostras e falha no resto —
e sem o estrato isso vira uma média que esconde o defeito.

- [ ] **Passo 1: escrever a fixture compartilhada da Parte 2**

`tests/part2/conftest.py`:

```python
"""Fixtures da Parte 2. NÃO reescreve o relatório da Parte 1 (ver G0.4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset.generator import load_sample

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "part2_strata.md"
N_EVAL = 300           # amostras de `data/test` usadas nos portões geométricos
RESULTS_P2: dict = {"criteria": {}, "blocks": {}}


def record_p2(cid: str, name: str, target: str, measured: str, ok: bool | None) -> None:
    RESULTS_P2["criteria"][cid] = {
        "name": name, "target": target, "measured": measured, "ok": ok,
    }


def to_gray(image: np.ndarray) -> np.ndarray:
    """RGB uint8 -> cinza uint8. Luma ITU-R BT.601, sem depender do OpenCV."""
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (image.astype(np.float32) @ w).round().astype(np.uint8)


@pytest.fixture(scope="session")
def test_samples() -> list[dict]:
    root = ROOT / "data" / "test"
    dirs = sorted(root.glob("sample_*"))[:N_EVAL]
    assert dirs, "rode o Passo 12 do Bloco 0: data/test está vazio"
    return [load_sample(d) for d in dirs]
```

- [ ] **Passo 2: escrever o teste que reprova o portão G1.1**

`tests/part2/test_part2.py`:

```python
"""Critérios 2.1 a 2.11 do PLANO §PARTE 2, mais os portões internos G1/G2/G3b."""
from __future__ import annotations

import numpy as np

from identify.calibrate import detect_plot_bbox
from tests.part2.conftest import record_p2, to_gray


def test_g1_1_moldura_dentro_de_2px(test_samples):
    erros = []
    for m in test_samples:
        got = detect_plot_bbox(to_gray(m["image"]))
        if got is None:
            erros.append(np.inf)
            continue
        exp = m["plot_bbox_px"]
        erros.append(max(abs(g - e) for g, e in zip(got, exp)))
    erros = np.asarray(erros, dtype=float)
    frac = float(np.mean(erros <= 2.0))
    record_p2("G1.1", "Erro da moldura", "≤ 2 px em ≥ 95%", f"{frac:.3f}", frac >= 0.95)
    assert frac >= 0.95, f"apenas {frac:.1%} das molduras dentro de 2 px"
```

- [ ] **Passo 3: rodar o teste para confirmar que ele reprova**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_g1_1_moldura_dentro_de_2px -v`
Esperado: FAIL com `ModuleNotFoundError: No module named 'identify.calibrate'`.

- [ ] **Passo 4: implementar a detecção da moldura**

`identify/calibrate.py`. A chave está nas restrições globais: `left` e `bottom`
**sempre existem**; `right` e `top` não. Então detecte os dois lados garantidos por
projeção de gradiente, e infira os ausentes pela extensão dos ticks e da tinta.

```python
"""Estágio B — calibração dos eixos. Determinístico; o OCR entra no Bloco 2.

Convenções (ver PLANO_PARTE2.md, Restrições globais): origem de pixel no canto
superior esquerdo, centro do pixel i em i+0.5; bbox = [x0, y0, x1, y1] no centro
da spine; x_dados = sx*x_px + ox, y_dados = sy*y_px + oy com sy < 0.

Nenhuma função aqui levanta exceção: falha devolve None ou ok=False.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fração da altura (largura) que uma coluna (linha) precisa cobrir para ser
# considerada spine. Calibrado no Bloco 1 contra `plot_bbox_px`; ver handoff.
SPINE_COVER = 0.55
EDGE_Q = 0.90          # quantil do gradiente que define "borda"


def _edges(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Magnitudes de gradiente horizontal e vertical, normalizadas em [0, 1]."""
    g = gray.astype(np.float32)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    for a in (gx, gy):
        m = float(a.max())
        if m > 0:
            a /= m
    return gx, gy


def _long_lines(mag: np.ndarray, axis: int, cover: float) -> np.ndarray:
    """Índices cuja projeção ao longo de `axis` cobre `cover` da extensão."""
    thr = float(np.quantile(mag, EDGE_Q))
    hits = (mag >= max(thr, 1e-6)).sum(axis=axis)
    need = cover * mag.shape[axis]
    return np.flatnonzero(hits >= need)


def detect_plot_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Retângulo da área de dados. None quando nem left/bottom são achados."""
    h, w = gray.shape
    gx, gy = _edges(gray)
    cols = _long_lines(gx, 0, SPINE_COVER)   # colunas = spines verticais
    rows = _long_lines(gy, 1, SPINE_COVER)   # linhas  = spines horizontais
    if cols.size == 0 or rows.size == 0:
        return None
    x0, y1 = int(cols[0]), int(rows[-1])     # left e bottom: sempre presentes
    x1 = int(cols[-1]) if cols[-1] > x0 else w - 1
    y0 = int(rows[0]) if rows[0] < y1 else 0
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)
```

- [ ] **Passo 5: rodar o teste e calibrar `SPINE_COVER`**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_g1_1_moldura_dentro_de_2px -v`

Se reprovar, **meça antes de mexer**: escreva um script descartável em
`/tmp/claude-*/scratchpad/` que varra `SPINE_COVER ∈ {0,40; 0,50; 0,55; 0,65; 0,75}`
e imprima a fração dentro de 2 px por valor. É o mesmo método que a Parte 1 usou
para calibrar `SPAN_BINS` no `test_leakage.py`. Escolha o valor pela curva medida e
**registre a varredura inteira no handoff**, não só o vencedor.

Armadilha conhecida: com `has_grid=True` as linhas de grade também são longas. Elas
ficam **dentro** da moldura, então pegar o extremo (`cols[0]`, `cols[-1]`) já as
ignora — mas se a grade for mais escura que a spine, o quantil `EDGE_Q` pode
suprimir a spine. Se isso aparecer na varredura, estratifique por `has_grid` antes
de concluir qualquer coisa.

- [ ] **Passo 6: escrever o teste do portão G1.2 (ticks)**

```python
def test_g1_2_recall_de_ticks(test_samples):
    from identify.calibrate import detect_tick_pixels

    rec = {"x": [], "y": []}
    for m in test_samples:
        bbox = detect_plot_bbox(to_gray(m["image"]))
        if bbox is None:
            rec["x"].append(0.0); rec["y"].append(0.0)
            continue
        got = detect_tick_pixels(to_gray(m["image"]), bbox)
        for eixo in ("x", "y"):
            verdade = [p for p, _ in m["ticks"][eixo]]
            if not verdade:
                continue
            achados = np.asarray(got[eixo], dtype=float)
            if achados.size == 0:
                rec[eixo].append(0.0); continue
            ok = sum(bool(np.min(np.abs(achados - v)) <= 3.0) for v in verdade)
            rec[eixo].append(ok / len(verdade))
    for eixo in ("x", "y"):
        med = float(np.median(rec[eixo]))
        record_p2(f"G1.2{eixo}", f"Recall de ticks ({eixo})", "≥ 0,95 mediana",
                  f"{med:.3f}", med >= 0.95)
        assert med >= 0.95, f"recall mediano de ticks em {eixo}: {med:.3f}"
```

- [ ] **Passo 7: rodar para confirmar que reprova**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_g1_2_recall_de_ticks -v`
Esperado: FAIL com `ImportError: cannot import name 'detect_tick_pixels'`.

- [ ] **Passo 8: implementar a detecção de ticks**

Acrescente a `identify/calibrate.py`:

```python
TICK_BAND = 6          # px inspecionados fora da moldura, do lado dos rótulos
TICK_PROM = 0.25       # proeminência mínima do pico, relativa ao máximo da faixa


def _peaks(sig: np.ndarray, prom: float) -> list[float]:
    """Picos locais simples acima de `prom * max`, com refino por centroide."""
    if sig.size < 3:
        return []
    thr = prom * float(sig.max())
    out: list[float] = []
    i = 1
    while i < sig.size - 1:
        if sig[i] >= thr and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
            j = i
            while j + 1 < sig.size and sig[j + 1] == sig[i]:
                j += 1
            lo, hi = max(i - 1, 0), min(j + 2, sig.size)
            w = sig[lo:hi]
            idx = np.arange(lo, hi, dtype=float)
            out.append(float((w * idx).sum() / max(w.sum(), 1e-9)))
            i = j + 1
        else:
            i += 1
    return out


def detect_tick_pixels(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, list[float]]:
    """Ticks maiores por picos de tinta na faixa imediatamente FORA da moldura."""
    x0, y0, x1, y1 = bbox
    h, w = gray.shape
    g = gray.astype(np.float32)
    fundo = float(np.median(g))
    tinta = np.abs(g - fundo)

    faixa_x = tinta[min(y1 + 1, h - 1):min(y1 + 1 + TICK_BAND, h), x0:x1 + 1]
    faixa_y = tinta[y0:y1 + 1, max(x0 - TICK_BAND, 0):max(x0, 1)]
    px = [x0 + p for p in _peaks(faixa_x.sum(axis=0), TICK_PROM)] if faixa_x.size else []
    py = [y0 + p for p in _peaks(faixa_y.sum(axis=1), TICK_PROM)] if faixa_y.size else []
    return {"x": px, "y": py}
```

- [ ] **Passo 9: rodar e calibrar `TICK_BAND` / `TICK_PROM`**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py -v -k "g1_"`

Mesma disciplina do Passo 5: varra `TICK_BAND ∈ {4, 6, 8, 12}` e
`TICK_PROM ∈ {0,15; 0,25; 0,40}`, meça, escolha e registre a varredura. O `dpi`
varia de 60 a 200, então uma banda fixa em pixels **vai** ser estreita demais no
dpi alto — se a varredura mostrar isso, torne `TICK_BAND` proporcional ao `dpi`
lido do meta e anote como Ruling.

- [ ] **Passo 10: escrever e rodar o teste do portão G1.3 (estratos)**

```python
def test_g1_3_moldura_por_n_spines(test_samples):
    por_estrato: dict[int, list[float]] = {}
    for m in test_samples:
        got = detect_plot_bbox(to_gray(m["image"]))
        exp = m["plot_bbox_px"]
        err = np.inf if got is None else max(abs(g - e) for g, e in zip(got, exp))
        por_estrato.setdefault(int(m["render"]["n_spines"]), []).append(float(err))
    for n_spines, errs in sorted(por_estrato.items()):
        frac = float(np.mean(np.asarray(errs) <= 2.0))
        record_p2(f"G1.3-{n_spines}", f"Moldura, n_spines={n_spines}",
                  "≥ 0,90", f"{frac:.3f} (n={len(errs)})", frac >= 0.90)
        assert frac >= 0.90, f"n_spines={n_spines}: {frac:.1%} dentro de 2 px"
```

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py -v -k "g1_"`
Esperado: todos os G1 passam.

- [ ] **Passo 11: validar os portões contra defeitos injetados**

Princípio do `HANDOFF.md §8`. Para cada mutante abaixo, copie o repositório para
`/tmp/claude-*/scratchpad/mut_<id>/`, aplique **uma** substituição e rode
`pytest tests/part2 -q` inteiro (sem `-x`):

| Mutante | Substituição em `identify/calibrate.py` | Deve reprovar |
|---|---|---|
| P2-M01 | `x1 = int(cols[-1])` → `x1 = w - 1` (sempre a borda) | G1.1 e G1.3 (n_spines=4) |
| P2-M02 | `y0 = int(rows[0])` → `y0 = 0` | G1.1 |
| P2-M03 | `_peaks(..., TICK_PROM)` → `_peaks(..., 0.90)` | G1.2 |
| P2-M04 | `<= 3.0` no casamento de ticks → `<= 50.0` (controle: afrouxa o teste) | **nada** deve reprovar |

O P2-M04 é o controle: se ele reprovar alguma coisa, o teste está medindo outra
coisa que não o recall. Registre a tabela preenchida no handoff.

- [ ] **Passo 12: escrever `HANDOFF_P2_1.md`**

Obrigatório: as varreduras de calibração completas (não só o valor escolhido), a
tabela de mutantes preenchida, os valores medidos de G1.1/G1.2/G1.3 por estrato, e
as assinaturas exatas de `detect_plot_bbox` e `detect_tick_pixels` para o Bloco 2.

---

# Bloco 2 — Estágio B completo: OCR opcional, RANSAC e consistência

**Critérios do PLANO fechados aqui: 2.3, 2.4, 2.5, 2.9, 2.11.**

> **Decisão E (`PLANO.md §1.7`) — o OCR não é estrutural.** `ok = False` **não**
> descarta a amostra: a saída adimensional (`order`, ζ, ωₙ·T, θ/T, K/y_faixa) não
> depende de calibração nenhuma e tem de ser produzida de qualquer forma. Dois
> critérios novos guardam isso: **2.9** (cobertura ≥ 90 %, estratificada por DPI —
> impede que 2.3 e 2.4 sejam satisfeitos rejeitando tudo) e **2.11** (a saída
> adimensional existe em 100 % das amostras, sem exceção levantada).
> Acrescente aos campos de `Calibration` nada novo — `ok` e `reason` já bastam —
> mas o **`px_to_data` não pode ser o único caminho de saída** do pipeline.

**Arquivos:**
- Modificar: `identify/calibrate.py`
- Modificar: `tests/part2/test_part2.py`
- Criar: `HANDOFF_P2_2.md`

**Interfaces:**
- Consome: `detect_plot_bbox`, `detect_tick_pixels` (Bloco 1).
- Produz, para os Blocos 4 e 5:
  - `@dataclass(frozen=True) Calibration` com campos
    `sx: float`, `ox: float`, `sy: float`, `oy: float`,
    `bbox_px: tuple[int, int, int, int]`, `n_pairs_x: int`, `n_pairs_y: int`,
    `ok: bool`, `reason: str` (vazio quando `ok`).
  - `calibrate(image_rgb: np.ndarray) -> Calibration`
  - `px_to_data(cal: Calibration, x_px: np.ndarray, y_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]`

### Critérios

| # | Critério (verbatim do PLANO) | Alvo |
|---|---|---|
| 2.3 | Erro relativo das escalas `sx`, `sy` | < 1% em ≥ 95% das amostras |
| 2.4 | Taxa de rejeição por consistência (falso alarme) | < 5% |
| 2.5 | Rejeições corretas: quando rejeita, o erro de escala seria de fato > 5% | ≥ 90% das rejeições |

Leia 2.4 e 2.5 juntos: eles são um par precisão/cobertura. Um detector que nunca
rejeita passa em 2.4 e **vacuamente** em 2.5 (zero rejeições, zero erradas). Por
isso o teste de 2.5 abaixo exige `n_rejeicoes ≥ 5` para ter direito de asseverar —
sem isso o critério é decoração, exatamente o que o `HANDOFF.md §8` proíbe.

- [ ] **Passo 1: escrever o teste do critério 2.3 (reprova)**

```python
def test_2_3_erro_das_escalas(test_samples):
    from identify.calibrate import calibrate

    erros = []
    for m in test_samples:
        cal = calibrate(m["image"])
        if not cal.ok:
            continue
        a = m["axis_affine"]
        e = max(abs(cal.sx - a["sx"]) / abs(a["sx"]),
                abs(cal.sy - a["sy"]) / abs(a["sy"]))
        erros.append(float(e))
    assert len(erros) >= 0.5 * len(test_samples), "aceitou amostras demais de menos"
    frac = float(np.mean(np.asarray(erros) < 0.01))
    record_p2("2.3", "Erro relativo de sx, sy", "< 1% em ≥ 95%",
              f"{frac:.3f} (n={len(erros)})", frac >= 0.95)
    assert frac >= 0.95
```

- [ ] **Passo 2: rodar para confirmar que reprova**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_2_3_erro_das_escalas -v`
Esperado: FAIL com `ImportError: cannot import name 'calibrate'`.

- [ ] **Passo 3: implementar leitura dos rótulos via Tesseract**

Acrescente a `identify/calibrate.py`:

```python
import re

import pytesseract
from PIL import Image

# Faixa recortada ao redor do tick para o OCR, em px. Calibrar neste bloco.
LABEL_W, LABEL_H = 90, 34
_NUM_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?$")
_OCR_CFG = "--psm 7 -c tessedit_char_whitelist=0123456789.,-+eE"


def _ocr_number(crop: np.ndarray) -> float | None:
    """Lê um número de um recorte. Devolve None em qualquer ambiguidade."""
    if crop.size == 0:
        return None
    img = Image.fromarray(crop.astype(np.uint8), mode="L")
    img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
    txt = pytesseract.image_to_string(img, config=_OCR_CFG).strip()
    txt = txt.replace(" ", "")
    if not _NUM_RE.match(txt):
        return None
    try:
        return float(txt.replace(",", "."))
    except ValueError:
        return None


def read_tick_labels(gray: np.ndarray, bbox: tuple[int, int, int, int],
                     ticks: dict[str, list[float]]) -> dict[str, list[tuple[float, float]]]:
    """Para cada tick, tenta ler o rótulo vizinho. Pares (pixel, valor) lidos."""
    x0, y0, x1, y1 = bbox
    h, w = gray.shape
    pares: dict[str, list[tuple[float, float]]] = {"x": [], "y": []}
    for px in ticks["x"]:
        cx = int(round(px))
        crop = gray[min(y1 + 2, h - 1):min(y1 + 2 + LABEL_H, h),
                    max(cx - LABEL_W // 2, 0):min(cx + LABEL_W // 2, w)]
        v = _ocr_number(crop)
        if v is not None:
            pares["x"].append((float(px), v))
    for py in ticks["y"]:
        cy = int(round(py))
        crop = gray[max(cy - LABEL_H // 2, 0):min(cy + LABEL_H // 2, h),
                    max(x0 - 2 - LABEL_W, 0):max(x0 - 2, 1)]
        v = _ocr_number(crop)
        if v is not None:
            pares["y"].append((float(py), v))
    return pares
```

- [ ] **Passo 4: implementar o RANSAC afim de um eixo**

```python
RANSAC_TOL = 0.02      # tolerância relativa do inlier, em fração do span de valores
RANSAC_MIN = 2         # o PLANO: "bastam 2 ticks corretos por eixo"


def fit_axis_affine(pares: list[tuple[float, float]]) -> tuple[float, float, int] | None:
    """RANSAC exaustivo sobre pares (pixel, valor) -> (escala, offset, n_inliers).

    Exaustivo, não amostrado: são poucos ticks (tipicamente ≤ 12), então todos os
    pares cabem em O(n²) e o resultado fica determinístico — sem RNG, conforme as
    restrições globais.
    """
    if len(pares) < RANSAC_MIN:
        return None
    vals = [v for _, v in pares]
    span = max(vals) - min(vals)
    tol = RANSAC_TOL * span if span > 0 else 1e-9
    melhor: tuple[float, float, int] | None = None
    for i in range(len(pares)):
        for j in range(i + 1, len(pares)):
            (p1, v1), (p2, v2) = pares[i], pares[j]
            if abs(p2 - p1) < 1e-9:
                continue
            s = (v2 - v1) / (p2 - p1)
            o = v1 - s * p1
            inl = [(p, v) for p, v in pares if abs(s * p + o - v) <= tol]
            if melhor is None or len(inl) > melhor[2]:
                if len(inl) >= 2:
                    P = np.asarray([p for p, _ in inl], dtype=float)
                    V = np.asarray([v for _, v in inl], dtype=float)
                    A = np.vstack([P, np.ones_like(P)]).T
                    sol, *_ = np.linalg.lstsq(A, V, rcond=None)
                    melhor = (float(sol[0]), float(sol[1]), len(inl))
    return melhor
```

- [ ] **Passo 5: implementar `calibrate` e o teste de consistência interna**

```python
@dataclass(frozen=True)
class Calibration:
    sx: float = float("nan")
    ox: float = float("nan")
    sy: float = float("nan")
    oy: float = float("nan")
    bbox_px: tuple[int, int, int, int] = (0, 0, 0, 0)
    n_pairs_x: int = 0
    n_pairs_y: int = 0
    ok: bool = False
    reason: str = ""


# Consistência interna (PLANO): ticks equiespaçados em valor E em pixel.
SPACING_TOL = 0.05     # desvio relativo máximo do espaçamento


def _equiespacados(pares: list[tuple[float, float]], tol: float) -> bool:
    if len(pares) < 3:
        return True                       # 2 pontos não têm o que violar
    ps = np.asarray(sorted(p for p, _ in pares), dtype=float)
    vs = np.asarray(sorted(v for _, v in pares), dtype=float)
    for a in (ps, vs):
        d = np.diff(a)
        if d.size == 0 or np.mean(np.abs(d)) < 1e-12:
            return False
        if float(np.max(np.abs(d - np.mean(d))) / np.mean(np.abs(d))) > tol:
            return False
    return True


def calibrate(image_rgb: np.ndarray) -> Calibration:
    """Estágio B completo. Nunca levanta: falha vira ok=False + reason."""
    g = (image_rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114],
                                                 dtype=np.float32))
    gray = g.round().astype(np.uint8)
    bbox = detect_plot_bbox(gray)
    if bbox is None:
        return Calibration(reason="bbox_not_found")
    ticks = detect_tick_pixels(gray, bbox)
    pares = read_tick_labels(gray, bbox, ticks)
    if len(pares["x"]) < RANSAC_MIN or len(pares["y"]) < RANSAC_MIN:
        return Calibration(bbox_px=bbox, n_pairs_x=len(pares["x"]),
                           n_pairs_y=len(pares["y"]), reason="ocr_insuficiente")
    if not (_equiespacados(pares["x"], SPACING_TOL)
            and _equiespacados(pares["y"], SPACING_TOL)):
        return Calibration(bbox_px=bbox, n_pairs_x=len(pares["x"]),
                           n_pairs_y=len(pares["y"]), reason="calibration_failed")
    fx = fit_axis_affine(pares["x"])
    fy = fit_axis_affine(pares["y"])
    if fx is None or fy is None:
        return Calibration(bbox_px=bbox, reason="ransac_failed")
    sx, ox, nx = fx
    sy, oy, ny = fy
    if not (np.isfinite(sx) and np.isfinite(sy)) or sx <= 0 or sy >= 0:
        # sy < 0 é estrutural: o eixo y da imagem cresce para baixo.
        return Calibration(bbox_px=bbox, reason="sinal_de_escala_invalido")
    return Calibration(sx=sx, ox=ox, sy=sy, oy=oy, bbox_px=bbox,
                       n_pairs_x=nx, n_pairs_y=ny, ok=True)


def px_to_data(cal: Calibration, x_px: np.ndarray, y_px: np.ndarray):
    """Converte pixels em unidades físicas com a convenção da Parte 1."""
    return cal.sx * np.asarray(x_px, float) + cal.ox, \
           cal.sy * np.asarray(y_px, float) + cal.oy
```

- [ ] **Passo 6: rodar o teste 2.3 e calibrar `LABEL_W`/`LABEL_H`**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_2_3_erro_das_escalas -v`

Armadilha prevista: `dpi ∈ [60, 200]` e `size_px` até 1600×1200, então uma janela
de recorte fixa em pixels corta o rótulo ao meio no dpi alto e engloba o vizinho no
dpi baixo. Se a medição mostrar isso, torne `LABEL_W`/`LABEL_H` proporcionais ao
`dpi` e registre como Ruling. **Meça antes de trocar** — varra
`LABEL_W ∈ {60, 90, 130}` e registre a tabela.

- [ ] **Passo 7: escrever e rodar os testes dos critérios 2.4 e 2.5**

```python
def test_2_4_2_5_rejeicao_por_consistencia(test_samples):
    from identify.calibrate import calibrate

    rejeitadas, erro_se_aceitasse = 0, []
    for m in test_samples:
        cal = calibrate(m["image"])
        a = m["axis_affine"]
        if cal.ok:
            continue
        rejeitadas += 1
        # Recalibra ignorando a consistência para saber se a rejeição foi justa.
        e = _erro_de_escala_sem_guarda(m["image"], a)
        erro_se_aceitasse.append(e)

    taxa = rejeitadas / len(test_samples)
    record_p2("2.4", "Taxa de rejeição (falso alarme)", "< 5%",
              f"{taxa:.3f}", taxa < 0.05)
    assert taxa < 0.05, f"rejeitou {taxa:.1%} das amostras"

    assert rejeitadas >= 5, (
        "menos de 5 rejeições: o critério 2.5 não tem poder estatístico e "
        "passaria por vacuidade — ver PLANO_PARTE2 Bloco 2"
    )
    corretas = float(np.mean([e > 0.05 for e in erro_se_aceitasse]))
    record_p2("2.5", "Rejeições corretas", "≥ 90%",
              f"{corretas:.3f} (n={rejeitadas})", corretas >= 0.90)
    assert corretas >= 0.90
```

Se `rejeitadas < 5` sobre 300 amostras, você tem duas saídas honestas, e **as duas
exigem Ruling escrito**: aumentar `N_EVAL` até haver rejeições suficientes, ou
declarar 2.5 como não asseverável nesta amostra e reportá-lo como *n insuficiente*
no relatório — nunca como aprovado.

- [ ] **Passo 8: implementar o auxiliar `_erro_de_escala_sem_guarda`**

Em `tests/part2/test_part2.py` — mora no teste, não na biblioteca, porque só existe
para julgar a qualidade da rejeição:

```python
def _erro_de_escala_sem_guarda(image, affine_verdadeira) -> float:
    """Erro relativo de escala que teríamos se a guarda de consistência não existisse."""
    from identify.calibrate import (detect_plot_bbox, detect_tick_pixels,
                                    fit_axis_affine, read_tick_labels)
    gray = to_gray(image)
    bbox = detect_plot_bbox(gray)
    if bbox is None:
        return float("inf")
    pares = read_tick_labels(gray, bbox, detect_tick_pixels(gray, bbox))
    fx, fy = fit_axis_affine(pares["x"]), fit_axis_affine(pares["y"])
    if fx is None or fy is None:
        return float("inf")
    return max(abs(fx[0] - affine_verdadeira["sx"]) / abs(affine_verdadeira["sx"]),
               abs(fy[0] - affine_verdadeira["sy"]) / abs(affine_verdadeira["sy"]))
```

- [ ] **Passo 9: rodar os testes do Bloco 2 inteiros**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py -v -k "g1_ or 2_3 or 2_4"`
Esperado: todos passam.

- [ ] **Passo 10: validar contra defeitos injetados**

| Mutante | Substituição | Deve reprovar |
|---|---|---|
| P2-M05 | `_equiespacados` → `return True` (guarda desligada) | 2.5 (rejeições somem → `rejeitadas < 5` dispara) |
| P2-M06 | `SPACING_TOL = 0.05` → `0.001` | 2.4 (falso alarme explode) |
| P2-M07 | `fit_axis_affine`: usar só o **primeiro** par em vez do maior consenso | 2.3 |
| P2-M08 | `_NUM_RE` aceitar qualquer string (`.*`) | 2.3 |
| P2-M09 | Controle: `RANSAC_TOL 0.02 → 0.021` | **nada** |

- [ ] **Passo 11: escrever `HANDOFF_P2_2.md`**

Obrigatório, além do formato padrão: **a taxa de acerto do OCR por estrato**
(`dpi`, `bg_color` escuro vs. claro, `has_grid`). É o número que o Bloco 5 vai
precisar para explicar a degradação do critério 2.6 — sem ele, um ΔMAPE alto fica
sem diagnóstico, que é justamente o que o `PLANO.md §1.2, ponto 3` quer evitar.

---

# Bloco 3b — Estágio A sem rede: extrator clássico

**Critério do PLANO fechado aqui: metade do 2.10.** Decisão F, `PLANO.md §1.8`.

**Por que antes do Bloco 3:** ele tira a GPU do caminho crítico. Ao fim deste bloco a
Parte 2 **pode fechar** — o Bloco 5 integra, o critério 2.6 é medido, o entregável
existe. O Bloco 3 (U-Net) passa a ser melhoria com número próprio, não pré-requisito.
Se o timebox de driver do Bloco 0 estourou, este é o bloco que salva o cronograma.

**Arquivos:**
- Criar: `identify/extract_classical.py`
- Modificar: `tests/part2/test_part2.py`
- Criar: `HANDOFF_P2_3b.md`

**Interfaces:**
- Consome: `load_sample` (Parte 1); `detect_plot_bbox` (Bloco 1).
- Produz, com **a mesma assinatura** de `predict_mask` do Bloco 3, para ser
  intercambiável no Bloco 5:
  - `extract_mask_classical(image_rgb: np.ndarray, bbox: tuple | None = None) -> np.ndarray`
    — `uint8` 0/255, mesma resolução da entrada.

A igualdade de assinatura é o ponto do bloco. O `pipeline.py` do Bloco 5 recebe o
extrator como parâmetro (`extractor=...`), nunca o escolhe por dentro — é isso que
permite medir os dois lado a lado no critério 2.10 sem duplicar código de avaliação.

### Portões do bloco

| # | Portão | Alvo |
|---|---|---|
| G3b.1 | IoU mediana contra `mask.png` verdadeira, conjunto de teste | ≥ 0,70 (piso baixo de propósito: é baseline, não solução) |
| G3b.2 | Rejeição de grade e distratores: nenhuma reta de span completo na máscara | 0 violações, reusando `_spanning_rows` do critério 1.4e |
| G3b.3 | Não importa `torch` | `import identify.extract_classical` com `torch` ausente do ambiente |
| G3b.4 | Latência por imagem | < 200 ms |

O G3b.3 é o que dá valor de contingência ao bloco, e tem de ser testado de verdade —
um `import torch` acidental no topo do arquivo anula o propósito inteiro. Teste com
`subprocess` num interpretador em que `torch` esteja bloqueado, não confie em leitura.

- [ ] **Passo 1: escrever o teste do G3b.3 (o teste de contingência) primeiro**

Ele é o mais fácil de esquecer e o único que não dá para consertar depois sem
reescrever o módulo. Bloqueie `torch` em `sys.modules` e importe.

- [ ] **Passo 2: implementar a segmentação por cor**

Pipeline determinístico, sem parâmetro aprendido:

1. **cor de fundo** = cor modal da imagem inteira;
2. **modos de cor restantes**: quantize em ~32 níveis por canal e conte; cada modo com
   fração ≥ 0,1 % dos pixels é candidato;
3. **componentes conexas** de cada modo, dentro da `plot_bbox_px`;
4. **rejeição de retas** — o passo que faz o método funcionar: grade, *spines* e
   distratores são segmentos retos de span completo, e o critério 1.4e já implementa
   essa detecção (`_spanning_rows`, com a verificação de cobertura de bins que foi
   calibrada na Parte 1). **Reuse a implementação, não reescreva** — ela levou três
   iterações para acertar o denominador;
5. entre os sobreviventes, o de maior extensão horizontal é a curva;
6. saída `uint8` 0/255.

- [ ] **Passo 3: medir o G3b.1 e registrar por estrato**

Registre IoU por `has_grid`, por fundo escuro e por `n_distractors`. A expectativa é
que o método clássico caia justamente onde a U-Net deve ganhar — e essa comparação
por estrato é o material do critério 2.10, mais informativa que as duas medianas.

- [ ] **Passo 4: escrever `HANDOFF_P2_3b.md`**

Registre explicitamente: **este extrator é baseline e contingência, não a solução**.
Se ele empatar com a U-Net no critério 2.10, o achado é que a U-Net não se justifica
neste dataset — e isso é um resultado a reportar, não um problema a esconder.

---

# Bloco 3 — Estágio A: U-Net e treino

**Critérios do PLANO fechados aqui: 2.1, 2.7.** Este é o bloco de parede longa.
Comece o treino e vá fazer o Bloco 4 enquanto ele roda.

**Arquivos:**
- Criar: `identify/extract.py`
- Criar: `train_unet.py`
- Modificar: `tests/part2/test_part2.py`
- Criar: `HANDOFF_P2_3.md`
- Artefato: `models/unet_stageA.pt` (não versionado)

**Interfaces:**
- Consome: `data/train`, `data/val` (Bloco 0).
- Produz, para os Blocos 4 e 5:
  - `class UNet(nn.Module)` — `forward(x: Tensor[N,1,512,512]) -> Tensor[N,1,512,512]` (logits)
  - `letterbox(gray: np.ndarray, size: int = 512) -> tuple[np.ndarray, LetterboxInfo]`
  - `unletterbox(mask512: np.ndarray, info: LetterboxInfo) -> np.ndarray`
  - `dice_bce_loss(logits: Tensor, target: Tensor) -> Tensor`
  - `load_model(path: str | Path, device: str) -> UNet`
  - `predict_mask(model: UNet, image_rgb: np.ndarray) -> np.ndarray` — uint8 0/255
    na resolução **original** da imagem.

### Critérios

| # | Critério (verbatim do PLANO) | Alvo |
|---|---|---|
| 2.1 | IoU da máscara no conjunto de teste | ≥ 0,85 (mediana) |
| 2.7 | Estratificação: IoU por presença de grade / legenda / fundo escuro | nenhum estrato < 0,75 |

- [ ] **Passo 1: escrever o teste do letterbox (geometria antes do modelo)**

O letterbox é onde erros de meio pixel viram erros de escala no critério 2.2. Ele é
testável sem treinar nada:

```python
def test_letterbox_preserva_geometria_no_roundtrip():
    from identify.extract import letterbox, unletterbox

    alvo = np.zeros((180, 640), dtype=np.uint8)
    alvo[40:140, 100:500] = 255            # retângulo com cantos conhecidos
    pequeno, info = letterbox(alvo, size=512)
    assert pequeno.shape == (512, 512)
    volta = unletterbox(pequeno, info)
    assert volta.shape == alvo.shape
    ys, xs = np.nonzero(volta)
    assert abs(int(xs.min()) - 100) <= 2 and abs(int(xs.max()) - 499) <= 2
    assert abs(int(ys.min()) - 40) <= 2 and abs(int(ys.max()) - 139) <= 2
```

- [ ] **Passo 2: rodar para confirmar que reprova**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_letterbox_preserva_geometria_no_roundtrip -v`
Esperado: FAIL com `ModuleNotFoundError: No module named 'identify.extract'`.

- [ ] **Passo 3: implementar letterbox, U-Net e perda**

`identify/extract.py`:

```python
"""Estágio A — segmentação da curva. U-Net compacta (PLANO §PARTE 2).

Preenchimento preserva a razão de aspecto: distorção anisotrópica alteraria a
geometria da curva e envenenaria o critério 2.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LetterboxInfo:
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    src_w: int
    src_h: int
    size: int


def letterbox(gray: np.ndarray, size: int = 512) -> tuple[np.ndarray, LetterboxInfo]:
    h, w = gray.shape
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), dtype=gray.dtype)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, LetterboxInfo(px, py, nw, nh, w, h, size)


def unletterbox(mask512: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    crop = mask512[info.pad_y:info.pad_y + info.new_h,
                   info.pad_x:info.pad_x + info.new_w]
    return cv2.resize(crop, (info.src_w, info.src_h),
                      interpolation=cv2.INTER_NEAREST)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4 níveis, base 16 canais. Saída = logits, mesma resolução da entrada."""

    def __init__(self, base: int = 16, levels: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(levels + 1)]
        self.enc = nn.ModuleList()
        cin = 1
        for c in chs[:-1]:
            self.enc.append(_block(cin, c))
            cin = c
        self.bott = _block(chs[-2], chs[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(levels - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(_block(chs[i] * 2, chs[i]))
        self.head = nn.Conv2d(chs[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bott(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], dim=1))
        return self.head(x)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor,
                  eps: float = 1.0) -> torch.Tensor:
    """BCE pura colapsa: a classe positiva ocupa < 2% dos pixels (PLANO)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1.0 - num / den).mean()


def load_model(path: str | Path, device: str = "cpu") -> UNet:
    model = UNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_mask(model: UNet, image_rgb: np.ndarray,
                 device: str = "cpu", thr: float = 0.5) -> np.ndarray:
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = (image_rgb.astype(np.float32) @ w).round().astype(np.uint8)
    small, info = letterbox(gray)
    x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(device)
    p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    m512 = np.where(p >= thr, 255, 0).astype(np.uint8)
    return unletterbox(m512, info)
```

- [ ] **Passo 4: rodar o teste do letterbox**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_letterbox_preserva_geometria_no_roundtrip -v`
Esperado: PASS.

- [ ] **Passo 5: registrar a contagem real de parâmetros**

O PLANO diz "~1,2 M". Meça, não presuma:

```python
def test_unet_tamanho_declarado():
    from identify.extract import UNet
    n = sum(p.numel() for p in UNet().parameters())
    record_p2("A.0", "Parâmetros da U-Net", "~1,2 M (PLANO)", f"{n/1e6:.2f} M", None)
    assert 0.5e6 <= n <= 2.5e6, f"{n} parâmetros fogem da ordem de grandeza"
```

Se o valor medido divergir de 1,2 M, **não mude o PLANO**: registre um Ruling no
handoff dizendo o valor real e por que a arquitetura foi mantida.

- [ ] **Passo 6: escrever o script de treino**

`train_unet.py`, na raiz — é script, não biblioteca:

```python
"""Treino do Estágio A. Determinístico: seed fixa, sem RNG global."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.generator import load_sample
from identify.extract import UNet, dice_bce_loss, letterbox


class MaskDataset(Dataset):
    def __init__(self, root: str, size: int = 512):
        self.dirs = sorted(Path(root).glob("sample_*"))
        self.size = size

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, i: int):
        m = load_sample(self.dirs[i])
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gray = (m["image"].astype(np.float32) @ w).round().astype(np.uint8)
        x, _ = letterbox(gray, self.size)
        y, _ = letterbox(m["mask"], self.size)
        return (torch.from_numpy(x.astype(np.float32) / 255.0)[None],
                torch.from_numpy((y > 127).astype(np.float32))[None])


def iou(logits, target, thr: float = 0.5) -> float:
    p = (torch.sigmoid(logits) >= thr).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    union = ((p + target) >= 1).float().sum(dim=(1, 2, 3))
    return float((inter / union.clamp(min=1.0)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="models/unet_stageA.pt")
    a = ap.parse_args()

    torch.manual_seed(20260817)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tr = DataLoader(MaskDataset("data/train", a.size), batch_size=a.batch,
                    shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(MaskDataset("data/val", a.size), batch_size=a.batch,
                    num_workers=2)
    model = UNet().to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    melhor = -1.0
    for ep in range(a.epochs):
        t0 = time.perf_counter()
        model.train()
        for x, y in tr:
            x, y = x.to(a.device), y.to(a.device)
            opt.zero_grad()
            loss = dice_bce_loss(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ious = [iou(model(x.to(a.device)), y.to(a.device)) for x, y in va]
        m = float(np.mean(ious))
        print(f"epoca {ep:02d}  IoU_val={m:.4f}  {time.perf_counter()-t0:.0f}s",
              flush=True)
        if m > melhor:
            melhor = m
            torch.save(model.state_dict(), a.out)
    print(f"melhor IoU_val={melhor:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Passo 7: medir uma época antes de comprometer a noite**

Não dispare 25 épocas às cegas. Rode uma:

```bash
.venv/bin/python train_unet.py --epochs 1 --size 512 2>&1 | tail -3
```

Anote o tempo real por época. **Ponto de decisão, com números:**
- **≤ 20 min/época** → siga a 512², 25 épocas, ~8 h.
- **20–90 min/época** (o esperado em CPU) → treine a `--size 256` para iterar, e
  reserve **uma** rodada final a 512². A 256² o custo cai ~4×.
- **> 90 min/época** → só a 256² é viável; registre isso como Ruling e diga no
  handoff que o critério 2.2 foi medido sob resolução reduzida.

Cuidado com RAM: a máquina tem 15 GB e ~11 GB já em uso. Se `num_workers=4` causar
OOM, caia para 2 e registre.

- [ ] **Passo 8: treinar**

```bash
.venv/bin/python train_unet.py --epochs 25 2>&1 | tee logs/train_unet.log
```

Rode em segundo plano e **vá para o Bloco 4**. Não fique esperando.

- [ ] **Passo 9: escrever o teste do critério 2.1**

```python
def test_2_1_iou_mediana(test_samples):
    import torch
    from identify.extract import load_model, predict_mask

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    ious = []
    for m in test_samples:
        pred = predict_mask(model, m["image"], dev) > 127
        alvo = m["mask"] > 127
        inter = float(np.logical_and(pred, alvo).sum())
        union = float(np.logical_or(pred, alvo).sum())
        ious.append(inter / max(union, 1.0))
    med = float(np.median(ious))
    record_p2("2.1", "IoU da máscara", "≥ 0,85 (mediana)", f"{med:.4f}", med >= 0.85)
    assert med >= 0.85
```

- [ ] **Passo 10: escrever o teste do critério 2.7 (estratos)**

```python
def test_2_7_iou_por_estrato(test_samples):
    import torch
    from identify.extract import load_model, predict_mask

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    estratos: dict[str, list[float]] = {}
    for m in test_samples:
        pred = predict_mask(model, m["image"], dev) > 127
        alvo = m["mask"] > 127
        v = float(np.logical_and(pred, alvo).sum()) / max(
            float(np.logical_or(pred, alvo).sum()), 1.0)
        r = m["render"]
        escuro = int(r["bg_color"].lstrip("#")[:2], 16) < 128
        for nome in (f"grade={r['has_grid']}", f"legenda={r['has_legend']}",
                     f"fundo_escuro={escuro}", f"traco={r['line_style']}"):
            estratos.setdefault(nome, []).append(v)
    for nome, vs in sorted(estratos.items()):
        med = float(np.median(vs))
        record_p2(f"2.7[{nome}]", f"IoU — {nome}", "≥ 0,75",
                  f"{med:.4f} (n={len(vs)})", med >= 0.75)
        assert med >= 0.75, f"estrato {nome}: IoU mediano {med:.4f}"
```

O PLANO nomeia grade/legenda/fundo escuro. `line_style` foi acrescentado porque o
`HANDOFF.md §4` mede que o traço pontilhado deixa **43% das colunas sem tinta** —
é o estrato com maior chance de falhar, e omiti-lo seria escolher não enxergar.
Registre esse acréscimo como Ruling (é adição de diagnóstico, não afrouxamento).

- [ ] **Passo 11: rodar os testes do Bloco 3**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py -v -k "2_1 or 2_7 or unet or letterbox"`
Esperado: todos passam. Se 2.7 reprovar num estrato só, esse é o alvo dos dias de
folga — o `PLANO.md §4` reservou os Dias 13–14 exatamente para "correção do estrato
mais fraco".

- [ ] **Passo 12: validar contra defeitos injetados**

| Mutante | Substituição | Deve reprovar |
|---|---|---|
| P2-M10 | `dice_bce_loss` → só `bce` | 2.1 (colapso para "tudo fundo") |
| P2-M11 | `letterbox` com `cv2.resize` direto para (512,512), sem preservar razão | 2.2 no Bloco 5 |
| P2-M12 | `unletterbox` sem recortar o *padding* | 2.1 |
| P2-M13 | Controle: `thr` 0,50 → 0,51 | **nada** |

- [ ] **Passo 13: escrever `HANDOFF_P2_3.md`**

Obrigatório: tempo real por época, dispositivo, resolução final de treino, curva
de IoU por época (cole o log), IoU mediano por estrato, contagem real de
parâmetros e todo Ruling de resolução reduzida.

---

# Bloco 4 — Estágio A: pós-processamento até a polilinha

**Critério do PLANO fechado aqui: 2.2.** Pode ser feito **em paralelo** ao treino
do Bloco 3, porque se desenvolve contra a `mask.png` verdadeira.

**Arquivos:**
- Criar: `identify/polyline.py`
- Modificar: `tests/part2/test_part2.py`
- Criar: `HANDOFF_P2_4.md`

**Interfaces:**
- Consome: `Calibration` e `px_to_data` (Bloco 2); máscara uint8 (Bloco 3 ou
  `mask.png` verdadeira).
- Produz, para o Bloco 5:
  - `mask_to_polyline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]` —
    `(x_px, y_px)`, uma amostra por coluna com tinta, vãos interpolados.
  - `polyline_to_series(x_px, y_px, cal) -> tuple[np.ndarray, np.ndarray]` — `(t, y)`
    em unidades físicas.

### Critério

| # | Critério (verbatim do PLANO) | Alvo |
|---|---|---|
| 2.2 | Erro da polilinha extraída vs. verdade | RMSE ≤ 2 px (p95 ≤ 5 px) |

**Duas medições, um portão.** O 2.2 será medido (a) contra a `mask.png` verdadeira,
que é o **piso** do extrator, e (b) contra a máscara predita, que é o valor de
operação. O portão do PLANO vale sobre (b), no Bloco 5; (a) entra no relatório como
diagnóstico. Isso não altera o critério — acrescenta a decomposição de erro que o
`PLANO.md §1.2, ponto 3` pede.

- [ ] **Passo 1: escrever o teste do 2.2 sobre a máscara verdadeira**

```python
def test_2_2_polilinha_contra_mascara_verdadeira(test_samples):
    from identify.polyline import mask_to_polyline

    rmses = []
    for m in test_samples:
        xp, yp = mask_to_polyline(m["mask"])
        if xp.size == 0:
            rmses.append(np.inf)
            continue
        a = m["axis_affine"]
        t_col = a["sx"] * xp + a["ox"]
        y_col = np.interp(t_col, m["series"]["t"], m["series"]["y"])
        yp_true = (y_col - a["oy"]) / a["sy"]        # de volta a pixels
        dentro = (t_col >= m["series"]["t"][0]) & (t_col <= m["series"]["t"][-1])
        if dentro.sum() < 10:
            rmses.append(np.inf)
            continue
        rmses.append(float(np.sqrt(np.mean((yp[dentro] - yp_true[dentro]) ** 2))))
    r = np.asarray(rmses, dtype=float)
    med, p95 = float(np.median(r)), float(np.percentile(r, 95))
    record_p2("2.2-piso", "Polilinha vs. máscara VERDADEIRA",
              "RMSE ≤ 2 px, p95 ≤ 5 px", f"RMSE={med:.2f} px, p95={p95:.2f} px",
              med <= 2.0 and p95 <= 5.0)
    assert med <= 2.0 and p95 <= 5.0
```

- [ ] **Passo 2: rodar para confirmar que reprova**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_2_2_polilinha_contra_mascara_verdadeira -v`
Esperado: FAIL com `ModuleNotFoundError: No module named 'identify.polyline'`.

- [ ] **Passo 3: implementar o pós-processamento**

`identify/polyline.py`:

```python
"""Máscara -> polilinha -> série física. Determinístico, sem torch.

O `HANDOFF.md §4` mede o dado de projeto que dimensiona este módulo: o extrator
ingênuo "mediana por coluna" erra 0,19 px em linha sólida contra 0,92 px em
pontilhada, e o estilo `:` deixa 43% das colunas SEM TINTA. Por isso a
interpolação de vãos não é enfeite: sem ela, quase metade do domínio some.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

MAX_GAP_FRAC = 0.15    # vão máximo interpolado, como fração da largura da curva


def mask_to_polyline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maior componente conexa -> esqueleto -> mediana por coluna -> polilinha."""
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        return np.empty(0), np.empty(0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.empty(0), np.empty(0)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    skel = skeletonize(lab == k)

    xs, ys = [], []
    for x in range(skel.shape[1]):
        linhas = np.flatnonzero(skel[:, x])
        if linhas.size:
            xs.append(float(x))
            ys.append(float(np.median(linhas)))
    if len(xs) < 2:
        return np.empty(0), np.empty(0)

    x_arr, y_arr = np.asarray(xs), np.asarray(ys)
    x_full = np.arange(int(x_arr[0]), int(x_arr[-1]) + 1, dtype=float)
    y_full = np.interp(x_full, x_arr, y_arr)

    # Vão longo demais não é traço pontilhado: é ausência de dado. Descarta.
    largura = x_arr[-1] - x_arr[0]
    if largura > 0:
        vaos = np.diff(x_arr)
        for i in np.flatnonzero(vaos > MAX_GAP_FRAC * largura):
            corte = (x_full > x_arr[i]) & (x_full < x_arr[i + 1])
            y_full[corte] = np.nan
    ok = ~np.isnan(y_full)
    return x_full[ok], y_full[ok]


def polyline_to_series(x_px: np.ndarray, y_px: np.ndarray, cal) -> tuple[np.ndarray, np.ndarray]:
    """Pixels -> unidades físicas, com a afim estimada pelo Estágio B."""
    from identify.calibrate import px_to_data
    return px_to_data(cal, x_px, y_px)
```

- [ ] **Passo 4: rodar e medir**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_2_2_polilinha_contra_mascara_verdadeira -v`

Se reprovar no p95, estratifique por `line_style` antes de mexer em qualquer
constante — o `HANDOFF.md §4` já prevê que `:` é o pior caso, com número medido.
Ajuste `MAX_GAP_FRAC` por medição varrida, nunca por intuição, e registre a
varredura.

- [ ] **Passo 5: escrever o teste do marcador esparso**

`has_marker` desenha marcadores que engrossam a curva localmente e puxam a mediana
por coluna. É um modo de falha distinto do pontilhado e merece portão próprio:

```python
def test_2_2_estrato_marcador_e_estilo(test_samples):
    from identify.polyline import mask_to_polyline

    por: dict[str, list[float]] = {}
    for m in test_samples:
        xp, yp = mask_to_polyline(m["mask"])
        a = m["axis_affine"]
        if xp.size == 0:
            e = float("inf")
        else:
            t_col = a["sx"] * xp + a["ox"]
            y_col = np.interp(t_col, m["series"]["t"], m["series"]["y"])
            e = float(np.sqrt(np.mean((yp - (y_col - a["oy"]) / a["sy"]) ** 2)))
        r = m["render"]
        por.setdefault(f"traco={r['line_style']}", []).append(e)
        por.setdefault(f"marcador={r['has_marker']}", []).append(e)
    for nome, vs in sorted(por.items()):
        med = float(np.median(vs))
        record_p2(f"2.2[{nome}]", f"RMSE da polilinha — {nome}", "≤ 2 px",
                  f"{med:.2f} px (n={len(vs)})", med <= 2.0)
        assert med <= 2.0, f"estrato {nome}: RMSE mediano {med:.2f} px"
```

- [ ] **Passo 6: rodar os testes do Bloco 4**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py -v -k "2_2"`
Esperado: todos passam.

- [ ] **Passo 7: validar contra defeitos injetados**

| Mutante | Substituição | Deve reprovar |
|---|---|---|
| P2-M14 | `skeletonize(...)` → usar a componente cheia sem esqueletonizar | 2.2 no estrato `line_width` grosso |
| P2-M15 | Remover a interpolação de vãos (`y_full = np.interp` → só colunas com tinta) | 2.2 no estrato `traco=:` |
| P2-M16 | `np.median(linhas)` → `linhas[0]` (topo em vez da mediana) | 2.2 |
| P2-M17 | Controle: `MAX_GAP_FRAC` 0,15 → 0,16 | **nada** |

- [ ] **Passo 8: escrever `HANDOFF_P2_4.md`**

Obrigatório: RMSE e p95 por estrato de `line_style`, `has_marker` e `line_width`,
e a comparação explícita com os 0,19 px / 0,92 px medidos na Parte 1 — se o seu
número for pior que o do extrator ingênuo, o módulo regrediu e isso precisa
aparecer, não ser silenciado.

---

# Bloco 5 — Integração, degradação e relatório

**Critérios do PLANO fechados aqui: 2.6 e 2.8.** É o bloco do veredito.

**Arquivos:**
- Criar: `identify/pipeline.py`
- Modificar: `tests/part2/conftest.py` (gerar `reports/part2_strata.md`)
- Modificar: `tests/part2/test_part2.py`
- Criar: `HANDOFF_P2_5.md` (e atualizar `HANDOFF.md §4` para "Parte 2 concluída")

**Interfaces:**
- Consome: `calibrate` (Bloco 2), `predict_mask` (Bloco 3), `mask_to_polyline` e
  `polyline_to_series` (Bloco 4), `identify.classical.identify` (Parte 1, **inalterado**).
- Produz, para a Parte 3:
  - `identify_from_image(image_rgb: np.ndarray, model: UNet, device: str = "cpu") -> dict`
    com chaves `order: str`, `params: dict`, `ok: bool`, `reason: str`,
    `latency_ms: float`, `n_points: int`.

### Critérios

| # | Critério (verbatim do PLANO) | Alvo |
|---|---|---|
| 2.6 | Degradação end-to-end vs. oráculo da Parte 1 (mesmas amostras, estágio D idêntico) | ΔMAPE ≤ 3 pontos percentuais |
| 2.8 | Tempo de inferência por imagem | < 500 ms |

**A regra que faz 2.6 significar alguma coisa:** o oráculo e o pipeline real têm de
rodar sobre **as mesmas amostras**, com **o mesmo estágio D** e **a mesma métrica**.
A Parte 1 mede `θ` em NMAE normalizado por `T_dom`, não em MAPE
(`tests/conftest.py:441`) — repita essa convenção exatamente, ou o Δ compara coisas
diferentes e não quer dizer nada.

- [ ] **Passo 1: implementar o pipeline**

`identify/pipeline.py`:

```python
"""Cola dos estágios A, B e D. Única porta de entrada para a Parte 3."""
from __future__ import annotations

import time

import numpy as np

from identify.calibrate import calibrate
from identify.classical import identify
from identify.extract import predict_mask
from identify.polyline import mask_to_polyline, polyline_to_series


def identify_from_image(image_rgb: np.ndarray, model, device: str = "cpu") -> dict:
    """Imagem -> parâmetros físicos. Nunca levanta: falha vira ok=False."""
    t0 = time.perf_counter()
    vazio = {"order": "", "params": {}, "ok": False, "reason": "",
             "latency_ms": 0.0, "n_points": 0}

    cal = calibrate(image_rgb)
    if not cal.ok:
        return {**vazio, "reason": cal.reason,
                "latency_ms": (time.perf_counter() - t0) * 1e3}

    mask = predict_mask(model, image_rgb, device)
    x_px, y_px = mask_to_polyline(mask)
    if x_px.size < 10:
        return {**vazio, "reason": "polilinha_curta",
                "latency_ms": (time.perf_counter() - t0) * 1e3}

    t, y = polyline_to_series(x_px, y_px, cal)
    ordem = np.argsort(t)
    fit = identify(t[ordem], y[ordem])
    return {"order": fit.order, "params": fit.params, "ok": bool(fit.success),
            "reason": "" if fit.success else "ajuste_falhou",
            "latency_ms": (time.perf_counter() - t0) * 1e3,
            "n_points": int(x_px.size)}
```

- [ ] **Passo 2: escrever o teste do critério 2.8 (latência)**

```python
def test_2_8_latencia_por_imagem(test_samples):
    import torch
    from identify.extract import load_model
    from identify.pipeline import identify_from_image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)
    identify_from_image(test_samples[0]["image"], model, dev)   # aquecimento
    lat = [identify_from_image(m["image"], model, dev)["latency_ms"]
           for m in test_samples[:100]]
    p95 = float(np.percentile(lat, 95))
    record_p2("2.8", "Latência por imagem", "< 500 ms",
              f"mediana {np.median(lat):.0f} ms, p95 {p95:.0f} ms", p95 < 500.0)
    assert p95 < 500.0
```

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_2_8_latencia_por_imagem -v`

Ponto de atenção honesto: a Parte 1 mediu o estágio D sozinho em **mediana 55 ms,
p95 143 ms**. O orçamento restante para A+B é de ~350 ms no p95. Se estourar em CPU,
registre o número real e o dispositivo — o critério 2.8 do PLANO não diz "em CPU",
e medir em GPU é legítimo desde que **declarado no relatório**.

- [ ] **Passo 3: escrever o teste do critério 2.6**

```python
def test_2_6_degradacao_vs_oraculo(test_samples):
    """ΔMAPE ≤ 3 p.p. Mesmas amostras, mesmo estágio D, mesma métrica da Parte 1."""
    import torch
    from identify.classical import identify as estagio_d
    from identify.extract import load_model
    from identify.pipeline import identify_from_image
    from tests.conftest import meta_t_dom

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("models/unet_stageA.pt", dev)

    err_oraculo: dict[str, list[float]] = {}
    err_real: dict[str, list[float]] = {}
    aceitas = 0
    for m in test_samples:
        alvo = m["params"]
        t_dom = meta_t_dom(m)

        # Oráculo: série VERDADEIRA do meta -> estágio D (idêntico à Parte 1).
        o = estagio_d(m["series"]["t"], m["series"]["y"])
        r = identify_from_image(m["image"], model, dev)
        if not (o.success and r["ok"] and r["order"] == m["order"]
                and o.order == m["order"]):
            continue
        aceitas += 1
        for k in ("K", "tau", "wn", "zeta"):
            if alvo.get(k) is None:
                continue
            for saida, acc in ((o.params, err_oraculo), (r["params"], err_real)):
                if saida.get(k) is None:
                    continue
                acc.setdefault(k, []).append(
                    abs(saida[k] - alvo[k]) / max(abs(alvo[k]), 1e-12) * 100.0)
        # θ em NMAE/T_dom — convenção da Parte 1 (tests/conftest.py:441).
        for saida, acc in ((o.params, err_oraculo), (r["params"], err_real)):
            if saida.get("theta") is not None:
                acc.setdefault("theta", []).append(
                    abs(saida["theta"] - alvo["theta"]) / max(t_dom, 1e-12) * 100.0)

    assert aceitas >= 100, f"apenas {aceitas} amostras comparáveis: sem poder"
    piores = []
    for k in sorted(set(err_oraculo) & set(err_real)):
        d = float(np.median(err_real[k])) - float(np.median(err_oraculo[k]))
        piores.append((k, d))
        record_p2(f"2.6[{k}]", f"ΔMAPE — {k}", "≤ 3 p.p.",
                  f"{d:+.2f} p.p. (oráculo {np.median(err_oraculo[k]):.2f}%, "
                  f"real {np.median(err_real[k]):.2f}%)", d <= 3.0)
    pior = max(d for _, d in piores)
    record_p2("2.6", "Degradação end-to-end (pior parâmetro)", "≤ 3 p.p.",
              f"{pior:+.2f} p.p. (n={aceitas})", pior <= 3.0)
    assert pior <= 3.0, f"pior degradação: {piores}"
```

- [ ] **Passo 4: rodar o teste 2.6**

Run: `.venv/bin/python -m pytest tests/part2/test_part2.py::test_2_6_degradacao_vs_oraculo -v`

Se reprovar, **não mexa no alvo**. Use a decomposição que os blocos anteriores
produziram: o handoff do Bloco 2 tem o erro de escala, o do Bloco 3 o IoU por
estrato, o do Bloco 4 o RMSE da polilinha. O `PLANO.md §1.2, ponto 3` existe
justamente para que essa pergunta tenha resposta — use-a antes de mudar código.

- [ ] **Passo 5: gerar o relatório estratificado**

Acrescente a `tests/part2/conftest.py` um `pytest_sessionfinish` que escreve
`reports/part2_strata.md` a partir de `RESULTS_P2`, no mesmo formato de
`reports/part1_metrics.md` (tabela `critério | alvo | medido | veredito`), com o
mesmo banner de parcialidade quando a sessão rodou com filtro. Reaproveite
`_md_table` e `_fmt` de `tests/conftest.py` **importando**, não copiando.

Obrigatório no cabeçalho do relatório: dispositivo usado, resolução de treino,
número de amostras avaliadas e a lista de amostras descartadas por
`calibration_failed` — a taxa de descarte é parte do resultado, não rodapé.

- [ ] **Passo 6: rodar a suíte inteira das duas partes**

Run: `.venv/bin/python -m pytest -q`
Esperado: **33 da Parte 1 + os da Parte 2, todos verdes**, e os dois relatórios
regenerados. Se `reports/part1_metrics.md` sair com banner de parcialidade, a
guarda do Bloco 0 está mal ajustada — conserte antes de declarar a Parte 2 pronta.

- [ ] **Passo 7: campanha de mutação da Parte 2 inteira**

Rode os 17 mutantes P2-M01…P2-M17 dos blocos anteriores de uma vez, sobre a suíte
completa, e preencha a tabela `mutante | detectado por | veredito`. É o que autoriza
citar os números da Parte 2 na monografia — o mesmo argumento do
`HANDOFF.md §3.5.1`. Nenhum mutante escapado pode ficar sem correção ou sem Ruling
explicando por que ele é inofensivo.

- [ ] **Passo 8: escrever `HANDOFF_P2_5.md` e atualizar `HANDOFF.md`**

No `HANDOFF.md`, mude a linha da tabela de status de `Parte 2 | ... | **não
iniciada**` para o estado real, e acrescente uma seção "O que a Parte 2 entrega
para a Parte 3".

**Atenção — isto mudou com a revisão de 22/08/2026.** Não há mais Estágio C, então
as distribuições de erro medidas aqui **não** alimentam treino de estimador nenhum.
Elas passam a ter dois usos, ambos de medição:

1. **Decomposição de erro para a monografia** (`PLANO.md §1.2`, ponto 3): quanto do
   erro final vem da segmentação, quanto da calibração, quanto do estágio D. Registre
   jitter de extração em px, erro de escala em %, taxa de truncamento e fração de
   colunas sem tinta — são os números dessa decomposição.
2. **O gatilho do critério 3.12** (`PLANO.md §1.3`): meça e registre a **taxa de
   convergência de `identify` sobre as séries extraídas**. Se ela cair abaixo de 99 %,
   ou se o NRMSE p95 passar de 0,02, a decisão de remover o Estágio C tem de ser
   reaberta. Este é o único número que pode ressuscitar o estimador neural, e o
   `HANDOFF_P2_5.md` **tem de reportá-lo explicitamente**, mesmo que passe folgado.

---

## Autorrevisão

**Cobertura da spec.** Os onze critérios do `PLANO.md §PARTE 2` estão alocados
(2.9 e 2.11 → Bloco 2; 2.10 → Blocos 3b e 3):
2.1 e 2.7 → Bloco 3; 2.2 → Bloco 4 (piso) e Bloco 5 (operação); 2.3, 2.4, 2.5 →
Bloco 2; 2.6 e 2.8 → Bloco 5. Os artefatos nomeados na spec — `identify/extract.py`,
`identify/calibrate.py`, `tests/test_part2.py`, `reports/part2_strata.md` — todos
têm bloco dono. Divergência de caminho, registrada aqui: os testes ficam em
`tests/part2/test_part2.py` em vez de `tests/test_part2.py`, para que o
`conftest.py` da Parte 2 não interfira no da Parte 1; `identify/polyline.py` e
`identify/pipeline.py` são acréscimos de decomposição, não de escopo.

**Consistência de tipos.** `Calibration` é produzida no Bloco 2 e consumida nos
Blocos 4 e 5 com os mesmos campos; `LetterboxInfo` é produzida e consumida só
dentro do Bloco 3; `mask_to_polyline` devolve `(x_px, y_px)` em ambos os usos;
`predict_mask` devolve uint8 0/255 na resolução original, que é o que
`mask_to_polyline` espera. A convenção `sy < 0` é asseverada em `calibrate` e usada
com o mesmo sinal em `px_to_data` e nos testes.

**O que este plano deliberadamente não faz.** Não altera o `PLANO.md`, não altera
nenhum alvo numérico, não toca em `dataset/` nem em `identify/classical.py`, e não
executa `git`. A única modificação em código já aprovado da Parte 1 é a guarda do
`pytest_sessionfinish` (Bloco 0, Passo 9), que existe para **proteger** o relatório
da Parte 1, e está registrada como Ruling.
