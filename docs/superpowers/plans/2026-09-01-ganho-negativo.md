# Ganho negativo (K < 0) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer a pipeline identificar sistemas de ganho negativo (K < 0) sem tocar na matemática do Estágio D, espelhando a série descendente antes do ajuste e invertendo o sinal de K depois.

**Architecture:** Três mudanças independentes. (1) `identify/calibrate.py` passa a normalizar glifos de sinal que o tesseract devolve no lugar do minus do matplotlib. (2) `identify/classical.py` ganha um detector de direção do degrau e, dentro de `identify()`, espelha a série quando ela desce — o ajuste roda no código atual, intocado, e só o `K` devolvido troca de sinal. (3) `dataset/generator.py` ganha um estrato opt-in de ganho negativo, no molde do `reta_no_patamar`, para que o corpus base fique byte a byte idêntico.

**Tech Stack:** Python 3.11 (`.venv` do repositório), NumPy, SciPy (`least_squares`), OpenCV, pytesseract, PyTorch (só para o Estágio A nos testes fim a fim), pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-ganho-negativo-design.md`

## Global Constraints

- **Branch dedicada.** Todo o trabalho em `bloco9-ganho-negativo`, criada a partir de `bloco8-caso-real`. Nada direto na branch atual.
- **Python do repositório.** Todo comando é `.venv/bin/python -m ...`. Nunca `python` do sistema.
- **NUNCA rodar a suíte inteira.** Esta máquina é estação de trabalho (16 GB), não CI. Sempre passar o caminho do arquivo: `.venv/bin/python -m pytest tests/part2/test_x.py -q`. Proibido `pytest` sem caminho.
- **NUNCA busca abrangente.** Nada de `grep -r` / `rg` / `find` a partir da raiz do repositório. Escopar por diretório.
- **NUNCA `git commit` sem autorização explícita do dono naquele momento.** Os passos de commit deste plano são propostas: pare e pergunte antes de cada um.
- **NUNCA `Co-Authored-By` nem rodapé de geração** em mensagem de commit. A mensagem termina no conteúdo técnico.
- **Nenhuma função do pipeline levanta exceção em amostra malformada.** Falha vira `ok=False` + `reason`, nunca `raise` (contrato §6).
- **Nunca `np.random` global.** Sempre `np.random.default_rng(seed)`.
- **Convenção da afim:** `x_dados = sx*x_px + ox`, `y_dados = sy*y_px + oy`, com `sy < 0`.

---

### Task 0: Criar a branch de trabalho

**Files:**
- Nenhum arquivo alterado; só estado do git.

**Interfaces:**
- Consumes: nada.
- Produces: a branch `bloco9-ganho-negativo`, base de todas as tarefas seguintes.

- [ ] **Step 1: Confirmar que a árvore está limpa do que importa**

```bash
git status --short
```

Esperado: `identify/`, `dataset/` e `tests/` sem modificações pendentes. Arquivos não rastreados (`specs/`, `rg.py` modificado) podem ficar — não entram nesta branch.

- [ ] **Step 2: Criar e entrar na branch**

```bash
git checkout -b bloco9-ganho-negativo
git branch --show-current
```

Esperado: `bloco9-ganho-negativo`.

---

### Task 1: OCR aceita o sinal negativo que o tesseract devolve

**Files:**
- Modify: `identify/calibrate.py` (a função `_texto_para_numero`, hoje em ~`:286`)
- Test: `tests/part2/test_ocr_sinal.py` (criar)

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces: `identify.calibrate._texto_para_numero(txt: str) -> float | None` com o mesmo contrato de hoje, agora tolerante a glifos de sinal. Nenhum chamador muda de assinatura.

**Contexto para quem implementa.** O matplotlib desenha o menos como U+2212 (MINUS SIGN), não como o hífen ASCII. O tesseract lê o DÍGITO certo e devolve o sinal como em-dash. Medido em `Figure_dn2.png`, texto cru por recorte: `'0.0'`, `'—0.2'`, `'—0.4'`, `'—0.6'`, `'—-0.8'`, `'—1.0'`, `'=1.2'`, `'—1.4'`, `'—16'`. O filtro `_NUM_RE` rejeita tudo que não é `[+-]?\d+(?:[.,]\d+)?`, então 8 de 9 rótulos viravam `None`. O blob e a dilatação já estão certos — o glifo ENTRA no recorte.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/part2/test_ocr_sinal.py`:

```python
"""O sinal negativo que o tesseract devolve não é o hífen ASCII.

O matplotlib desenha U+2212 (MINUS SIGN); o tesseract lê o dígito certo e
devolve o sinal como em-dash. Medido em `Figure_dn2.png`, texto cru: '—0.2',
'—0.4', '—0.6', '—-0.8', '—1.0', '=1.2', '—1.4', '—16'. Ver §40.1.
"""
import pytest

from identify.calibrate import _texto_para_numero


@pytest.mark.parametrize("txt, esperado", [
    ("—0.2", -0.2),    # em-dash, o que o tesseract mais devolve
    ("−0.2", -0.2),    # minus sign, o que o matplotlib desenha
    ("–0.2", -0.2),    # en-dash
    ("—-0.8", -0.8),   # em-dash + hífen: sinal duplicado, um só vale
    ("-0.2", -0.2),         # hífen ASCII, que já funcionava
    ("—1.0", -1.0),
])
def test_le_sinal_negativo(txt, esperado):
    assert _texto_para_numero(txt) == pytest.approx(esperado)


@pytest.mark.parametrize("txt", [
    "=1.2",     # '=' NÃO é sinal: mapear seria inventar leitura
    "abc",
    "",
    "--",
    "1.2.3",
    "—",   # sinal sozinho, sem dígito
])
def test_rejeita_o_que_nao_e_numero(txt):
    assert _texto_para_numero(txt) is None


@pytest.mark.parametrize("txt, esperado", [
    ("0.0", 0.0), ("12", 12.0), ("1,5", 1.5), ("+3", 3.0), ("2e3", 2000.0),
])
def test_nao_regride_o_caminho_positivo(txt, esperado):
    assert _texto_para_numero(txt) == pytest.approx(esperado)
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
.venv/bin/python -m pytest tests/part2/test_ocr_sinal.py -q
```

Esperado: FAIL nos casos de em-dash/minus/en-dash (`assert None == -0.2`), PASS nos de rejeição e nos positivos.

- [ ] **Step 3: Implementar**

Em `identify/calibrate.py`, logo acima de `_texto_para_numero`, acrescentar a tabela e o `re` (o módulo já importa `re`):

```python
# Glifos que o OCR devolve no lugar do sinal negativo. O matplotlib desenha
# U+2212 (MINUS SIGN) e o tesseract quase nunca o devolve como tal: medido em
# `Figure_dn2.png`, 8 de 9 rótulos vieram com EM DASH. Sem esta normalização o
# `_NUM_RE` rejeita o rótulo inteiro e o eixo perde o par — foi assim que uma
# imagem com os NOVE rótulos legíveis saiu com um só. Ver §40.1.
#
# '=' NÃO entra nesta tabela de propósito, embora o tesseract também o devolva
# (o '=1.2' da mesma imagem). Mapear '=' para '-' é inventar leitura: um sinal
# de igual pode vir de qualquer coisa no eixo, e o RANSAC já descarta o par
# perdido de graça. Aqui vale a regra de sempre: perder um par é barato,
# inventar um é caro.
_GLIFOS_DE_SINAL = str.maketrans({
    "−": "-",   # MINUS SIGN (o que o matplotlib desenha)
    "—": "-",   # EM DASH
    "–": "-",   # EN DASH
    "‐": "-",   # HYPHEN
    "‑": "-",   # NON-BREAKING HYPHEN
    "˗": "-",   # MODIFIER LETTER MINUS SIGN
})
_SINAL_REPETIDO = re.compile(r"^-{2,}")
```

E trocar o corpo de `_texto_para_numero` por:

```python
def _texto_para_numero(txt: str) -> float | None:
    """`str` -> número, com o MESMO filtro de `_ocr_number` (contrato §1.7).

    Normaliza os glifos de sinal ANTES do filtro (`_GLIFOS_DE_SINAL`): o
    tesseract devolve em-dash onde o matplotlib desenhou U+2212, e sem isso o
    rótulo negativo inteiro é descartado.
    """
    txt = txt.strip().replace(" ", "").translate(_GLIFOS_DE_SINAL)
    txt = _SINAL_REPETIDO.sub("-", txt)
    if not _NUM_RE.match(txt):
        return None
    try:
        return float(txt.replace(",", "."))
    except ValueError:
        return None
```

- [ ] **Step 4: Rodar e ver passar**

```bash
.venv/bin/python -m pytest tests/part2/test_ocr_sinal.py -q
```

Esperado: todos PASS.

- [ ] **Step 5: Verificar o efeito na imagem real**

```bash
.venv/bin/python -c "
import numpy as np; from PIL import Image
from identify.calibrate import calibrate
for p in ('/home/loizm/Figure_dn2.png', '/home/loizm/Figure_dn.png'):
    c = calibrate(np.asarray(Image.open(p).convert('RGB')))
    print(p.split('/')[-1], 'ok=', c.ok, 'n_pares_y=', c.n_pairs_y)
"
```

Esperado: `dn2 ok= True n_pares_y= 9` (era `ok=False`, 1 par) e `dn ok= True n_pares_y= 7` (era 3).

- [ ] **Step 6: Não regrediu a calibração do corpus**

```bash
.venv/bin/python -m pytest tests/part2/test_part2.py -q -p no:randomly -k "2_3 or 2_9 or 2_4"
```

Esperado: PASS. Os critérios 2.3 (erro de escala), 2.9 (cobertura) e 2.4/2.5 (rejeição) são os que a mudança poderia mover.

- [ ] **Step 7: Commit (PEDIR AUTORIZAÇÃO ANTES)**

```bash
git add identify/calibrate.py tests/part2/test_ocr_sinal.py
git commit -m "Parte 2 / Bloco 9: OCR aceita o sinal negativo que o tesseract devolve

O matplotlib desenha U+2212 e o tesseract devolve em-dash. Medido em
Figure_dn2.png: 8 de 9 rotulos viravam None no _NUM_RE, com os digitos
lidos CERTOS. A normalizacao leva a imagem de 1 par de eixo y para 9.

'=' fica de fora da tabela de proposito: mapea-lo seria inventar leitura,
e o RANSAC descarta o par perdido de graca."
```

---

### Task 2: Detector da direção do degrau

**Files:**
- Modify: `identify/classical.py` (nova função privada, junto dos utilitários de série, depois de `_span` em ~`:191`)
- Test: `tests/part2/test_sinal_do_degrau.py` (criar)

**Interfaces:**
- Consumes: nada.
- Produces: `identify.classical._sinal_do_degrau(y: np.ndarray) -> float`, devolvendo `-1.0` ou `+1.0`. Consumida pela Task 3.

**Contexto para quem implementa.** NÃO reusar `pipeline._nivel_de_repouso`. Ele lê as 5 PRIMEIRAS colunas e supõe a curva parada ali — o próprio `pipeline.py:96-104` documenta essa suposição como frágil sob truncagem à esquerda. `classical.py` é a camada de baixo e não pode importar `pipeline`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/part2/test_sinal_do_degrau.py`:

```python
"""Direção do degrau, lida da série. Base do caminho C (§40.2)."""
import numpy as np
import pytest

from identify.classical import _sinal_do_degrau, model_response


def _serie(K, order="fopdt", n=400):
    t = np.linspace(0.0, 12.0, n)
    p = ({"K": K, "tau": 1.0, "theta": 2.0, "wn": None, "zeta": None}
         if order == "fopdt" else
         {"K": K, "tau": None, "theta": 1.0, "wn": 3.0, "zeta": 0.3})
    return t, model_response(order, p, t)


@pytest.mark.parametrize("order", ["fopdt", "second"])
def test_degrau_positivo_da_mais_um(order):
    _, y = _serie(2.0, order)
    assert _sinal_do_degrau(y) == 1.0


@pytest.mark.parametrize("order", ["fopdt", "second"])
def test_degrau_negativo_da_menos_um(order):
    _, y = _serie(-2.0, order)
    assert _sinal_do_degrau(y) == -1.0


def test_subamortecido_negativo_nao_se_confunde_com_o_sobressinal():
    """zeta baixo faz a curva ULTRAPASSAR o patamar. A direção é do patamar,
    não do pico — por isso o detector usa decis, não o extremo."""
    _, y = _serie(-1.0, "second")
    assert float(np.min(y)) < -1.0        # confirma que há sobressinal
    assert _sinal_do_degrau(y) == -1.0


def test_serie_plana_devolve_mais_um():
    assert _sinal_do_degrau(np.zeros(200)) == 1.0


def test_serie_curta_nao_levanta():
    assert _sinal_do_degrau(np.array([1.0, 2.0])) in (-1.0, 1.0)
    assert _sinal_do_degrau(np.array([])) == 1.0


def test_ruido_nao_inverte_o_sinal():
    rng = np.random.default_rng(7)
    _, y = _serie(-2.0)
    y = y + rng.normal(0.0, 0.05, size=y.size)
    assert _sinal_do_degrau(y) == -1.0
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
.venv/bin/python -m pytest tests/part2/test_sinal_do_degrau.py -q
```

Esperado: FAIL com `ImportError: cannot import name '_sinal_do_degrau'`.

- [ ] **Step 3: Implementar**

Em `identify/classical.py`, depois de `_span` (~`:196`):

```python
def _sinal_do_degrau(y: np.ndarray) -> float:
    """Direção do degrau: `-1.0` se a resposta DESCE, `+1.0` caso contrário.

    Compara a mediana do último decil com a do primeiro. Decis, e não as
    primeiras colunas: `pipeline._nivel_de_repouso` lê as 5 primeiras e supõe a
    curva parada ali — suposição que o próprio `pipeline.py:96-104` documenta
    como frágil quando o Estágio A perde o trecho plano inicial, que é o
    defeito §39.3 e acontece justamente nas imagens de ganho negativo.

    Mediana do decil, e não o extremo: numa 2ª ordem subamortecida a curva
    ULTRAPASSA o patamar, então o extremo e o patamar têm direções que podem
    divergir. O que interessa é para onde a resposta ASSENTA.

    Empate (série plana) devolve `+1.0`: sem excursão não há sinal a recuperar,
    e `+1.0` mantém o comportamento anterior byte a byte.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 4:
        return 1.0
    k = max(1, int(round(0.10 * y.size)))
    inicio = float(np.median(y[:k]))
    fim = float(np.median(y[-k:]))
    if not (np.isfinite(inicio) and np.isfinite(fim)):
        return 1.0
    return -1.0 if fim < inicio else 1.0
```

> **REFUTADO PELA MEDIÇÃO, 2026-09-02.** O estimador acima (mediana do
> primeiro decil contra a do último) usa o PRIMEIRO DECIL COMO PROXY DO NÍVEL
> DE REPOUSO, e esse proxy é inválido exatamente quando o §39.3 come o platô
> inicial: o primeiro decil cai dentro do transitório, e numa subamortecida o
> transitório passa ALÉM do valor final, pelo lado oposto — a leitura inverte.
>
> Medido em `Figure_dn2.png` (ζ=0,2, θ=2 s, série extraída começando em
> t=2,09 s): devolvia `+1` numa resposta que desce, e a `dn2` não fechava.
> O defeito é SIMÉTRICO — com degrau positivo e a mesma cabeça cortada devolvia
> `-1`, espelhando uma série ascendente. Varrendo ζ e τ nos dois sinais, com e
> sem cabeça: **errava 8 de 44 casos**, todas as subamortecidas (ζ ≤ 0,4) sem
> cabeça. Não é defeito de ganho negativo; morde o caminho positivo também.
>
> Ajustar a FRAÇÃO do decil não conserta: na série sobrevivente o decil pousa
> onde a oscilação estiver, e a leitura oscila com a fração (0,05 → −1;
> 0,10 → +1; 0,20 → −1; 0,30 → +1 na MESMA imagem).
>
> **Segunda tentativa, TAMBÉM refutada:** repouso = extremo MAIS DISTANTE do
> valor assentado. Media 0 erros no sintético, mas pressupõe que o último decil
> é o valor FINAL — e não é quando a janela acaba antes de assentar. Numa 2ª
> ordem muito subamortecida o último decil pousa no ringing, o primeiro pico
> fica mais longe dele que o repouso, e o pico é eleito repouso. **Espelhava 2
> das 900 séries do oráculo do corpus**, todas com K > 0: duas amostras boas
> destruídas. Aparecia na Parte 1 como MAPE(K) do estrato limpo w<3 saindo de
> 0,000 % para 0,239 % (`sample_00307`, ζ=0,104; `sample_00889`, ζ=0,121).
>
> **O que foi implementado:** o repouso é o extremo que aparece PRIMEIRO —
> se o máximo vem antes do mínimo, a série desceu. A informação está no TEMPO,
> não no valor: uma resposta ao degrau nunca cruza de volta o nível de onde
> partiu, então o repouso É um dos dois extremos, e o que o distingue do
> sobressinal é a ORDEM. Ler a ordem dispensa saber onde a resposta assenta,
> que é justamente o que a janela curta esconde.
>
> Medido: **0 erros** na varredura sintética (44/44) e **0 espelhos indevidos**
> nas 900 do oráculo — as duas provas que derrubaram as tentativas anteriores,
> cada uma numa ponta diferente da série. Imune a ruído nesta aplicação: 0
> erros em 1600 séries com σ até 0,30 sobre um salto de 2.
>
> **Contrato e limite, medidos e asseverados.** Vale enquanto o repouso ainda
> ESTIVER na série. Cortada a cabeça além dele, o primeiro extremo que sobra é
> um sobressinal e a regra inverte — a direção passa a viver só no envelope
> decadente, que exige ajuste. Registrado em
> `test_o_limite_do_contrato_quando_o_repouso_sai_da_serie` com `xfail` estrito.

- [ ] **Step 4: Rodar e ver passar**

```bash
.venv/bin/python -m pytest tests/part2/test_sinal_do_degrau.py -q
```

Esperado: todos PASS.

- [ ] **Step 5: Commit (PEDIR AUTORIZAÇÃO ANTES)**

```bash
git add identify/classical.py tests/part2/test_sinal_do_degrau.py
git commit -m "Parte 2 / Bloco 9: detector da direcao do degrau

Le a direcao por mediana de decis, nao pelas 5 primeiras colunas: o
Estagio A perde o trecho plano inicial nas imagens de ganho negativo
(defeito A do 39.3), e um estimador ancorado no inicio erraria ali.

Decil e nao extremo porque a 2a ordem subamortecida ultrapassa o
patamar. Serie plana devolve +1 e mantem o caminho atual intocado."
```

---

### Task 3: `identify()` espelha a série descendente

**Files:**
- Modify: `identify/classical.py` (a função `identify`, hoje em ~`:778`)
- Test: `tests/part2/test_ganho_negativo.py` (criar)

**Interfaces:**
- Consumes: `_sinal_do_degrau(y) -> float` (Task 2).
- Produces: `identify.classical.identify(t, y) -> FitResult` com a MESMA assinatura de hoje, agora capaz de devolver `params["K"] < 0`. `identify_both()` NÃO muda — segue ajustando as duas estruturas na série como recebida.

**Contexto para quem implementa.** O modelo é linear em K (`model_response` devolve `(K * STEP_AMPLITUDE) * base`) e a base é livre de sinal, então ajustar `-y` e negar o `K` resultante é EXATAMENTE equivalente a ajustar `y` com K livre de sinal. `sse`, `nrmse` e `aic` são invariantes ao espelho porque dependem só de resíduos ao quadrado — não precisam de correção.

Cuidado com dois detalhes:
- `FitResult` é um `@dataclass` mutável; trocar `r.params` no lugar é seguro, mas prefira montar um dict novo para não mexer no que `identify_both` devolveu.
- Quando `s == +1.0`, o caminho tem de ser byte a byte o de hoje. Não chame `-1.0 * y` nesse ramo: multiplicar por `1.0` é inofensivo em teoria, mas o critério é "sem diferença nenhuma", e um `if` explícito torna isso verificável.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/part2/test_ganho_negativo.py`:

```python
"""Caminho C: espelhar a serie descendente (§40.3).

O ganho negativo era estruturalmente inexprimivel — `K_BOUNDS = (1e-3, 1e4)`
trava K positivo e o ajuste saia com NRMSE 0,90-0,96. Espelhar e negar K
depois e exatamente equivalente a parametrizar `K = s*|K|`, porque o modelo e
linear em K e a base e livre de sinal.
"""
import numpy as np
import pytest

from identify.classical import identify, identify_both, model_response

VERDADES = [
    ("fopdt",  {"K": -2.0, "tau": 1.4, "theta": 2.0, "wn": None, "zeta": None}),
    ("fopdt",  {"K": -0.4, "tau": 0.5, "theta": 0.0, "wn": None, "zeta": None}),
    ("second", {"K": -3.0, "tau": None, "theta": 1.0, "wn": 2.0, "zeta": 0.30}),
    ("second", {"K": -1.0, "tau": None, "theta": 0.5, "wn": 4.0, "zeta": 1.50}),
]


def _serie(order, p, n=512, t_fim=14.0):
    t = np.linspace(0.0, t_fim, n)
    return t, model_response(order, p, t)


@pytest.mark.parametrize("order, p", VERDADES)
def test_recupera_ganho_negativo(order, p):
    t, y = _serie(order, p)
    r = identify(t, y)
    assert r.order == order, f"ordem {r.order!r}, esperada {order!r}"
    for nome in ("K", "tau", "theta", "wn", "zeta"):
        alvo = p[nome]
        if alvo is None or abs(alvo) < 1e-9:
            continue
        obtido = r.params[nome]
        assert obtido is not None, f"{nome} ausente"
        erro = abs(obtido - alvo) / abs(alvo)
        assert erro <= 0.01, f"{nome} = {obtido:.5f}, esperado {alvo} (erro {erro:.2%})"


@pytest.mark.parametrize("order, p", VERDADES)
def test_o_K_devolvido_e_negativo(order, p):
    t, y = _serie(order, p)
    assert identify(t, y).params["K"] < 0.0


def test_theta_zero_sai_proximo_de_zero():
    """theta=0 e excluido do teste relativo acima (denominador nulo); aqui vai
    o absoluto, normalizado pela janela."""
    p = {"K": -0.4, "tau": 0.5, "theta": 0.0, "wn": None, "zeta": None}
    t, y = _serie("fopdt", p)
    r = identify(t, y)
    assert abs(r.params["theta"]) <= 0.01 * float(t[-1] - t[0])


@pytest.mark.parametrize("order, p", [
    ("fopdt",  {"K": 2.0, "tau": 1.4, "theta": 2.0, "wn": None, "zeta": None}),
    ("second", {"K": 3.0, "tau": None, "theta": 1.0, "wn": 2.0, "zeta": 0.30}),
])
def test_caminho_positivo_intocado(order, p):
    """Com K > 0 o espelho nao dispara e o resultado tem de ser IDENTICO ao que
    `identify_both` + a regra de escolha produzem — nao 'parecido'."""
    t, y = _serie(order, p)
    r = identify(t, y)
    r1, r2 = identify_both(t, y)
    referencia = r1 if r.order == "fopdt" else r2
    assert r.params == referencia.params
    assert r.sse == referencia.sse
    assert r.nrmse == referencia.nrmse


def test_metricas_sao_invariantes_ao_espelho():
    """sse/nrmse dependem so de residuos ao quadrado: a versao negativa de uma
    serie tem de dar exatamente as mesmas metricas que a positiva."""
    p_pos = {"K": 2.0, "tau": 1.4, "theta": 2.0, "wn": None, "zeta": None}
    p_neg = {**p_pos, "K": -2.0}
    t, y_pos = _serie("fopdt", p_pos)
    _, y_neg = _serie("fopdt", p_neg)
    a, b = identify(t, y_pos), identify(t, y_neg)
    assert a.nrmse == pytest.approx(b.nrmse, rel=1e-9)
    assert a.params["tau"] == pytest.approx(b.params["tau"], rel=1e-9)
    assert a.params["K"] == pytest.approx(-b.params["K"], rel=1e-9)
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
.venv/bin/python -m pytest tests/part2/test_ganho_negativo.py -q
```

Esperado: FAIL nos testes de ganho negativo (K sai `0.001`, travado no piso de `K_BOUNDS`); PASS nos de caminho positivo.

- [ ] **Step 3: Implementar**

Em `identify/classical.py`, trocar o corpo de `identify` (mantendo a docstring existente e ACRESCENTANDO o parágrafo novo ao fim dela):

```python
def identify(t, y) -> FitResult:
    """Estágio D: ajusta FOPDT e 2ª ordem e escolhe pela verossimilhança
    penalizada com nº de pontos EFETIVO (ver `_n_efetivo`), com uma guarda
    contra polo extra que só ajusta o traço do render
    (`_polo_rapido_e_artefato`).

    Equivale ao AIC quando o resíduo é branco; difere dele exatamente na medida
    em que a polilinha extraída é autocorrelacionada. Os campos `.aic` dos dois
    `FitResult` continuam sendo o AIC clássico e NÃO mudaram — `tests/conftest`
    os reporta na Parte 1.

    **Ganho negativo (§40.3).** `K_BOUNDS` é positivo por construção, e alargá-lo
    poria K=0 dentro da caixa — o modelo degenerado, um mínimo local trivial que
    hoje não existe. Em vez disso, uma resposta que DESCE é espelhada antes do
    ajuste e o `K` devolvido troca de sinal. É exatamente equivalente a
    parametrizar `K = s·|K|`, porque `model_response` é linear em K e a base é
    livre de sinal — e não toca uma linha da matemática do módulo. `sse`,
    `nrmse` e `aic` são invariantes ao espelho (dependem de resíduos ao
    quadrado), então nenhuma métrica precisa de correção.

    Com `s = +1` o caminho é byte a byte o de antes desta mudança, e o teste
    `test_caminho_positivo_intocado` assevera isso contra `identify_both`.
    """
    s = _sinal_do_degrau(y)
    if s < 0.0:
        r = _identify_ascendente(t, -np.asarray(y, dtype=float))
        r.params = {k: (-v if (k == "K" and v is not None) else v)
                    for k, v in r.params.items()}
        return r
    return _identify_ascendente(t, y)


def _identify_ascendente(t, y) -> FitResult:
    """O `identify` de sempre, supondo resposta que SOBE. Ver `identify`."""
    r1, r2 = identify_both(t, y)
    if not np.isfinite(r1.aic) and not np.isfinite(r2.aic):
        return r1
    if not np.isfinite(r2.aic):
        return r1
    if not np.isfinite(r1.aic):
        return r2
    tc, yc = _clean(t, y)
    if tc.size < 3:
        return r1 if r1.aic <= r2.aic else r2
    n_eff = _n_efetivo(tc, yc, r2)
    ganho = n_eff * np.log(max(r1.sse, 1e-300) / max(r2.sse, 1e-300))
    if ganho <= 2.0 * (r2.n_params - r1.n_params):
        return r1
    return r1 if _polo_rapido_e_artefato(tc, yc, r1, r2) else r2
```

O corpo de `_identify_ascendente` é o corpo ATUAL de `identify`, movido sem alteração nenhuma. Não reescreva de memória: recorte e cole.

- [ ] **Step 4: Rodar e ver passar**

```bash
.venv/bin/python -m pytest tests/part2/test_ganho_negativo.py tests/part2/test_sinal_do_degrau.py -q
```

Esperado: todos PASS.

- [ ] **Step 5: Não regrediu o Estágio D**

```bash
.venv/bin/python -m pytest tests/test_part1.py -q -p no:randomly
```

Esperado: 25 passed. Este é o portão que protege a matemática do módulo — a Parte 1 mede `identify` no caminho oráculo.

- [ ] **Step 6: Verificar nas imagens reais**

```bash
.venv/bin/python identificar.py /home/loizm/Figure_dn2.png /home/loizm/Figure_dn.png /home/loizm/Figure_dl3.png
```

Esperado: `dn2` fecha com K ≈ −1,0, ωₙ ≈ 5,0, ζ ≈ 0,20, θ ≈ 2,0. `dn` segue recusada (`ajuste_inconsistente`, NRMSE ≈ 0,173). `dl3` responde com K ≈ −3,0 e θ/ζ errados. Os três resultados são os esperados pela spec — nenhum é surpresa.

- [ ] **Step 7: Commit (PEDIR AUTORIZAÇÃO ANTES)**

```bash
git add identify/classical.py tests/part2/test_ganho_negativo.py
git commit -m "Parte 2 / Bloco 9: identify() espelha a serie descendente (caminho C)

K_BOUNDS e positivo por construcao e alarga-lo poria K=0 dentro da caixa
— o modelo degenerado, um minimo local trivial que hoje nao existe. Em
vez disso a serie que DESCE e espelhada antes do ajuste e o K devolvido
troca de sinal.

Exatamente equivalente a parametrizar K = s*|K|, porque model_response e
linear em K e a base e livre de sinal, mas sem tocar a matematica do
modulo. sse/nrmse/aic sao invariantes ao espelho.

Com s=+1 o caminho e byte a byte o anterior, asseverado contra
identify_both em test_caminho_positivo_intocado."
```

---

### Task 4: Estrato opt-in de ganho negativo no gerador

**Files:**
- Modify: `dataset/generator.py` (`generate_sample` e `render_sample`, os mesmos pontos que já recebem `reta_no_patamar`)
- Test: `tests/part2/test_estrato_ganho_negativo.py` (criar)

**Interfaces:**
- Consumes: nada das tarefas anteriores.
- Produces: `dataset.generator.generate_sample(..., ganho_negativo: bool = False)`. Com `False` (padrão) o resultado é byte a byte o de hoje.

**Contexto para quem implementa.** Leia primeiro como `reta_no_patamar` é passado — é o molde exato. A regra estrutural do §anti-vazamento vale: `sample_style` NÃO pode ver o spec, então o sinal do ganho é aplicado ao SPEC, nunca ao estilo. O corpus base precisa continuar byte a byte idêntico, e é por isso que o estrato é opt-in em vez de sair do sorteio de `sample_spec`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/part2/test_estrato_ganho_negativo.py`:

```python
"""Estrato opt-in de ganho negativo (§40.4). Molde: `reta_no_patamar` (§34.5)."""
import json
import tempfile
from pathlib import Path

import numpy as np

from dataset.generator import generate_sample, load_sample


def _gera(tmp, seed, **kw):
    d = Path(tmp) / f"s{seed}_{int(kw.get('ganho_negativo', False))}"
    generate_sample(str(d), seed=seed, **kw)
    return load_sample(d)


def test_o_padrao_nao_muda_um_byte():
    """Sem o flag, a amostra tem de ser identica a de antes desta mudanca."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"; b = Path(tmp) / "b"
        generate_sample(str(a), seed=4242)
        generate_sample(str(b), seed=4242, ganho_negativo=False)
        assert (a / "image.png").read_bytes() == (b / "image.png").read_bytes()
        assert (a / "mask.png").read_bytes() == (b / "mask.png").read_bytes()
        assert (a / "meta.json").read_text() == (b / "meta.json").read_text()


def test_com_o_flag_o_ganho_e_negativo():
    with tempfile.TemporaryDirectory() as tmp:
        m = _gera(tmp, 4242, ganho_negativo=True)
        assert m["params"]["K"] < 0.0


def test_a_serie_desce():
    with tempfile.TemporaryDirectory() as tmp:
        m = _gera(tmp, 4242, ganho_negativo=True)
        y = np.asarray(m["series"]["y"], dtype=float)
        k = max(1, y.size // 10)
        assert float(np.median(y[-k:])) < float(np.median(y[:k]))


def test_a_magnitude_e_a_mesma_do_positivo():
    """So o SINAL muda: o mesmo seed tem de dar |K| identico, senao o estrato
    nao e comparavel com o base."""
    with tempfile.TemporaryDirectory() as tmp:
        pos = _gera(tmp, 4242)
        neg = _gera(tmp, 4242, ganho_negativo=True)
        assert abs(neg["params"]["K"]) == abs(pos["params"]["K"])
        assert neg["params"]["theta"] == pos["params"]["theta"]
        assert neg["order"] == pos["order"]


def test_determinismo():
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"; b = Path(tmp) / "b"
        generate_sample(str(a), seed=99, ganho_negativo=True)
        generate_sample(str(b), seed=99, ganho_negativo=True)
        assert (a / "image.png").read_bytes() == (b / "image.png").read_bytes()
        assert (a / "meta.json").read_text() == (b / "meta.json").read_text()


def test_a_mascara_continua_valida():
    with tempfile.TemporaryDirectory() as tmp:
        m = _gera(tmp, 4242, ganho_negativo=True)
        mk = np.asarray(m["mask"])
        assert set(np.unique(mk)) <= {0, 255}
        acesos = int((mk > 127).sum())
        assert acesos >= 40, f"mascara degenerada: {acesos} px acesos"
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
.venv/bin/python -m pytest tests/part2/test_estrato_ganho_negativo.py -q
```

Esperado: FAIL com `TypeError: generate_sample() got an unexpected keyword argument 'ganho_negativo'`, exceto `test_o_padrao_nao_muda_um_byte`, que também falha pelo mesmo motivo.

- [ ] **Step 3: Implementar**

Localizar a assinatura de `generate_sample` (a que já tem `reta_no_patamar: bool = False`, ~`dataset/generator.py:622`) e acrescentar o parâmetro no mesmo estilo:

```python
                    ganho_negativo: bool = False,
```

Logo depois de `sample_spec` produzir o `spec` e ANTES de qualquer uso dele (renderização ou série), aplicar:

```python
    if ganho_negativo:
        # Estrato OOD opt-in (§40.4), molde do `reta_no_patamar` (§34.5). So o
        # SINAL de K muda; `sample_spec` continua sorteando K > 0 e |K| fica
        # identico ao do mesmo seed sem o flag, para que o estrato seja
        # comparavel amostra a amostra com o base.
        #
        # Opt-in, e nao um sorteio dentro de `sample_spec`, porque mexer no
        # sorteio moveria TODA amostra do corpus base — e com ela todo numero
        # historico da Parte 1 e da Parte 2. O corpus base fica byte a byte
        # identico e o estrato novo entra ao lado.
        #
        # Aplicado ao SPEC e nunca ao estilo: `sample_style` nao pode ver o
        # spec (regra anti-vazamento de `randomize.py`), e o sinal do ganho e
        # rotulo, nao aparencia.
        spec = replace(spec, K=-spec.K)
```

Verificar que `replace` (de `dataclasses`) já está importado no módulo; `generator.py` já o usa para o `reta_no_patamar`.

Propagar o parâmetro do `generate_sample` externo para o `render_sample`/gerador interno pelo MESMO caminho que `reta_no_patamar` percorre (`dataset/generator.py:351` e `:622` são os dois pontos).

- [ ] **Step 4: Rodar e ver passar**

```bash
.venv/bin/python -m pytest tests/part2/test_estrato_ganho_negativo.py -q
```

Esperado: todos PASS.

- [ ] **Step 5: Confirmar que o corpus base não se moveu**

```bash
.venv/bin/python -m pytest tests/test_part1.py -q -p no:randomly -k "determinism or meta_contract or dataset_independent"
```

Esperado: PASS. São os testes que travam a geração byte a byte.

- [ ] **Step 6: Commit (PEDIR AUTORIZAÇÃO ANTES)**

```bash
git add dataset/generator.py tests/part2/test_estrato_ganho_negativo.py
git commit -m "Parte 2 / Bloco 9: estrato opt-in de ganho negativo no gerador

Molde do reta_no_patamar: opt-in, aplicado ao SPEC e nunca ao estilo. So
o sinal de K muda, entao |K| fica identico ao do mesmo seed sem o flag e
o estrato e comparavel amostra a amostra com o base.

Opt-in e nao sorteio dentro de sample_spec porque mexer no sorteio
moveria toda amostra do corpus base, e com ela todo numero historico da
Parte 1 e da Parte 2."
```

---

### Task 5: Medir o estrato novo e a não-regressão do corpus

**Files:**
- Modify: `tests/part2/test_estrato_ganho_negativo.py` (criado na Task 4 — acrescentar ao fim)
- Nenhum código de produção muda nesta tarefa.

**Interfaces:**
- Consumes: `generate_sample(..., ganho_negativo=True)` (Task 4), `identify()` com espelho (Task 3), OCR com sinal (Task 1).
- Produces: nada consumido depois; produz os números que vão para o HANDOFF na Task 7.

**Contexto para quem implementa.** Esta é a tarefa que responde "quanto isso custou". O projeto mede tudo contra `data/test` antes de aceitar mudança. Gerar 200 amostras leva ~2 s; rodar a pipeline nelas com GPU leva ~1 min.

- [ ] **Step 1: Escrever o teste de portão do estrato**

Acrescentar ao fim de `tests/part2/test_estrato_ganho_negativo.py`:

```python
def test_pipeline_fecha_no_estrato_negativo():
    """Portao do estrato: o caminho ORACULO (serie do meta, sem render) tem de
    recuperar K com sinal em >= 95% de 60 amostras de ganho negativo.

    Oraculo e nao imagem: este teste mede o Estagio D, que e o que a Task 3
    mudou. O Estagio A no ganho negativo e assunto do 39.3 e nao deste portao.
    """
    import identify.classical as CL

    acertos = n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(9000, 9060):
            m = _gera(tmp, seed, ganho_negativo=True)
            t = np.asarray(m["series"]["t"], dtype=float)
            y = np.asarray(m["series"]["y"], dtype=float)
            tc, yc = CL._clean(t, y)
            if tc.size < 3:
                continue
            n += 1
            r = CL.identify(tc, yc)
            k = r.params.get("K")
            if k is None or k >= 0:
                continue
            if abs(k - m["params"]["K"]) / abs(m["params"]["K"]) <= 0.05:
                acertos += 1
    assert n >= 55, f"amostras uteis de menos: {n}"
    taxa = acertos / n
    assert taxa >= 0.95, f"K recuperado com sinal em {taxa:.1%} de {n} (alvo 95%)"
```

- [ ] **Step 2: Rodar**

```bash
.venv/bin/python -m pytest tests/part2/test_estrato_ganho_negativo.py::test_pipeline_fecha_no_estrato_negativo -q
```

Esperado: PASS. Se falhar abaixo de 95 %, NÃO afrouxe o alvo — investigue com a skill `superpowers:systematic-debugging`, porque o caminho oráculo não tem ruído de percepção e deveria fechar quase sempre.

- [ ] **Step 3: Medir a não-regressão no corpus inteiro**

```bash
.venv/bin/python -m pytest tests/part2/ -q -p no:randomly
```

Esperado: todos PASS, incluindo os 79 testes que já existiam. Este comando escopa em `tests/part2/` — não rode `pytest` sem caminho.

- [ ] **Step 4: Conferir os relatórios gerados**

```bash
git diff --stat reports/
git diff reports/part2_strata.md | grep -E "^[-+]\|" | head -30
```

Esperado: nenhum critério muda de veredito (✅ continua ✅). Números podem oscilar na terceira casa por carga de máquina; o que não pode é 2.12-ordem, 2.6 ou 2.9 se moverem de forma sistemática. Se moverem, pare e investigue antes de seguir.

- [ ] **Step 5: Commit (PEDIR AUTORIZAÇÃO ANTES)**

```bash
git add tests/part2/test_estrato_ganho_negativo.py reports/
git commit -m "Parte 2 / Bloco 9: portao do estrato de ganho negativo

Mede o caminho ORACULO em 60 amostras: K recuperado com sinal e a 5% em
>= 95%. Oraculo e nao imagem de proposito — o portao mede o Estagio D,
que foi o que mudou; o Estagio A no ganho negativo e assunto do 39.3."
```

---

### Task 6: Fixtures das três imagens reais

**Files:**
- Create: `tests/fixtures/caso_real_neg_fopdt.png` (de `/home/loizm/Figure_dn.png`)
- Create: `tests/fixtures/caso_real_neg_sub.png` (de `/home/loizm/Figure_dn2.png`)
- Create: `tests/fixtures/caso_real_neg_super.png` (de `/home/loizm/Figure_dl3.png`)
- Create: `tests/part2/test_caso_real_negativo.py`

**Interfaces:**
- Consumes: a pipeline completa depois das Tasks 1–4.
- Produces: nada consumido depois.

**Contexto para quem implementa.** **A verdade de cada imagem tem de ser CONFIRMADA com o dono antes de escrever o teste.** Os valores abaixo vêm do título e da legenda das figuras mais o ajuste medido, e um deles JÁ divergiu: `dn2` foi lido como ζ=0,3 pelo título e o ajuste deu 0,201 — o sobressinal de ~52 % dá ζ ≈ 0,204, então o ajuste está certo e a leitura visual estava errada. Não escreva o teste com número que você inferiu de um gráfico. Pergunte, e use `rg.py` como fonte se ele gerar estas três.

- [ ] **Step 1: Confirmar a verdade com o dono**

Perguntar, e só então prosseguir:

> As três imagens de ganho negativo entram como fixture. Preciso da verdade declarada — de preferência o trecho do `rg.py` que as gera. O que tenho medido é: `dn` FOPDT K=−2, τ=0,5, degrau em t=2 s e θ=1 s (θ da janela = 3,0); `dn2` 2ª ordem K=−1, ωₙ=5, ζ≈0,20, θ=2 s; `dl3` 2ª ordem K=−3, degrau em t=3 s e θ=0,5 s (θ da janela = 3,5). Confere?

- [ ] **Step 2: Copiar as fixtures**

```bash
cp /home/loizm/Figure_dn.png  tests/fixtures/caso_real_neg_fopdt.png
cp /home/loizm/Figure_dn2.png tests/fixtures/caso_real_neg_sub.png
cp /home/loizm/Figure_dl3.png tests/fixtures/caso_real_neg_super.png
ls -la tests/fixtures/
```

- [ ] **Step 3: Escrever o teste**

Criar `tests/part2/test_caso_real_negativo.py`. Substituir os valores de `VERDADE` pelos confirmados no Step 1:

```python
"""Regressao das tres imagens de GANHO NEGATIVO do `rg.py` (§40).

`theta` aqui e sempre o theta DA JANELA — medido do inicio do eixo x, nao do
instante do degrau. As figuras aplicam o degrau em t != 0 e declaram o atraso
do processo separadamente; a pipeline nao ve o sinal de entrada, entao o unico
theta que ela pode reportar e a soma. Ver §40.5.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIX = Path(__file__).resolve().parents[1] / "fixtures"

NEG_SUB = {"path": FIX / "caso_real_neg_sub.png", "order": "second",
           "K": -1.0, "wn": 5.0, "zeta": 0.20, "theta": 2.0}
NEG_FOPDT = {"path": FIX / "caso_real_neg_fopdt.png", "order": "fopdt",
             "K": -2.0, "tau": 0.5, "theta": 3.0}
NEG_SUPER = {"path": FIX / "caso_real_neg_super.png", "order": "second",
             "K": -3.0, "theta": 3.5}

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
    return identify_from_image(np.asarray(Image.open(caso["path"]).convert("RGB")), m, dev)


def _erro(medido, esperado):
    return abs(medido - esperado) / abs(esperado)


def test_sub_fecha_completamente(modelo):
    """A unica das tres que fecha fim a fim depois do caminho C."""
    r = _roda(NEG_SUB, modelo)
    assert r["order"] == "second", f"ordem {r['order']!r}"
    assert r["ok"], f"sem fisico: reason={r['reason']!r} cal={r['calibration']}"
    for nome in ("K", "wn", "zeta", "theta"):
        e = _erro(r["params"][nome], NEG_SUB[nome])
        assert e <= TOL, (f"{nome} = {r['params'][nome]:.4f}, esperado "
                          f"{NEG_SUB[nome]} (erro {e:.1%})")


def test_sub_calibra_o_eixo_y_com_rotulo_negativo():
    """Portao da Task 1 na imagem real: 9 rotulos negativos, nenhum perdido."""
    from identify.calibrate import calibrate
    cal = calibrate(np.asarray(Image.open(NEG_SUB["path"]).convert("RGB")))
    assert cal.ok_y, f"eixo y reprovado: {cal.reason!r}"
    assert cal.n_pairs_y >= 6, f"so {cal.n_pairs_y} pares no eixo y"


def test_fopdt_recupera_os_parametros_apesar_da_recusa(modelo):
    """O ajuste ACERTA os tres parametros; quem recusa e a guarda de NRMSE,
    porque a polilinha mistura a tracejada de ENTRADA com a saida (defeito 4,
    fora do escopo do §40). Este teste trava o que ja funciona, para que o dia
    em que o defeito 4 for fechado seja uma mudanca de `ok`, nao de numero."""
    import identify.classical as CL
    from identify.calibrate import calibrate
    from identify.extract import predict_mask
    from identify.polyline import mask_to_polyline, polyline_to_series

    m, dev = modelo
    img = np.asarray(Image.open(NEG_FOPDT["path"]).convert("RGB"))
    cal = calibrate(img)
    assert cal.ok, f"calibracao falhou: {cal.reason!r}"
    xs, ys = mask_to_polyline(predict_mask(m, img, dev), bbox=cal.bbox_px)
    t, y = polyline_to_series(xs, ys, cal)
    o = np.argsort(t)
    r = CL.identify(*CL._clean(t[o], y[o]))
    assert r.order == "fopdt"
    for nome in ("K", "tau", "theta"):
        e = _erro(r.params[nome], NEG_FOPDT[nome])
        assert e <= TOL, f"{nome} = {r.params[nome]:.4f}, esperado {NEG_FOPDT[nome]}"


@pytest.mark.xfail(strict=True,
                   reason="defeito 4 (§40): a polilinha mistura a tracejada de "
                          "ENTRADA com a saida e o NRMSE sai 0,173 > 0,13")
def test_fopdt_deveria_sair_pela_pipeline(modelo):
    r = _roda(NEG_FOPDT, modelo)
    assert r["ok"], f"reason={r['reason']!r}"


@pytest.mark.xfail(strict=True,
                   reason="§39.3: o Estagio A come o plato e a cauda (62,5% de "
                          "cobertura), entao theta e zeta saem errados")
def test_super_deveria_acertar_theta(modelo):
    r = _roda(NEG_SUPER, modelo)
    assert r["ok"]
    assert _erro(r["params"]["theta"], NEG_SUPER["theta"]) <= TOL


def test_super_pelo_menos_acerta_o_ganho(modelo):
    """O K sai certo mesmo com theta e zeta errados — e o que o caminho C
    entrega nesta imagem, e o que NAO pode regredir."""
    r = _roda(NEG_SUPER, modelo)
    assert r["ok"], f"reason={r['reason']!r}"
    e = _erro(r["params"]["K"], NEG_SUPER["K"])
    assert e <= TOL, f"K = {r['params']['K']:.4f}, esperado {NEG_SUPER['K']}"
```

- [ ] **Step 4: Rodar**

```bash
.venv/bin/python -m pytest tests/part2/test_caso_real_negativo.py -q -p no:randomly -rxX
```

Esperado: 4 passed, 2 xfailed. Se algum `xfail` virar XPASS, o defeito correspondente foi fechado por acidente — pare, confirme, e converta o teste em portão removendo a marca.

- [ ] **Step 5: Commit (PEDIR AUTORIZAÇÃO ANTES)**

```bash
git add tests/fixtures/caso_real_neg_*.png tests/part2/test_caso_real_negativo.py
git commit -m "Parte 2 / Bloco 9: fixtures das tres imagens de ganho negativo

A subamortecida fecha fim a fim. As outras duas entram com xfail estrito
nomeando o defeito que as bloqueia: a FOPDT pela polilinha que mistura a
tracejada de ENTRADA com a saida, a superamortecida pelo 39.3.

O teste da FOPDT trava os parametros que o ajuste JA acerta, para que
fechar o defeito 4 seja mudanca de ok e nao de numero."
```

---

### Task 7: Registrar o ruling no HANDOFF

**Files:**
- Modify: `HANDOFF_P2_7.md` (acrescentar `## 40.` ao fim)

**Interfaces:**
- Consumes: os números medidos nas Tasks 1–6.
- Produces: nada em código.

- [ ] **Step 1: Escrever a seção**

Acrescentar ao fim de `HANDOFF_P2_7.md` uma seção `## 40. Ruling 63 — ganho negativo: o caminho C e os dois defeitos que ele NÃO fecha`, cobrindo, com os números REAIS medidos na execução (não os da spec, que foram medidos em sonda):

1. **§40.1** — o OCR e o glifo de sinal: o texto cru que o tesseract devolve, por que `=` ficou de fora da tabela, e o efeito medido (`dn2` de 1 para 9 pares).
2. **§40.2** — o detector de direção: por que decis e não as 5 primeiras colunas, e por que mediana e não extremo.
3. **§40.3** — o caminho C e as duas alternativas descartadas (A põe K=0 na caixa; B espalha o sinal por três funções). A equivalência é exata porque o modelo é linear em K.
4. **§40.4** — o estrato opt-in e a taxa medida no portão do oráculo.
5. **§40.5** — **o que o caminho C NÃO fecha, e o custo que ele introduz.** Declarar explicitamente que a `dl3` passou de recusa para ERRO CONFIANTE, que a causa é o §39.3 e não o caminho C, e que a guarda de cobertura de máscara NÃO foi adicionada porque o §37.11 já a testou e refutou (Spearman +0,020, p = 0,57).
6. **§40.6** — a semântica de `theta` quando o degrau é aplicado em t ≠ 0: a pipeline não vê o sinal de entrada, então reporta a SOMA, e as fixtures são escritas com o θ da janela.

- [ ] **Step 2: Commit (PEDIR AUTORIZAÇÃO ANTES)**

```bash
git add HANDOFF_P2_7.md docs/superpowers/
git commit -m "Parte 2 / Bloco 9: Ruling 63 — ganho negativo, o caminho C e o que ele nao fecha"
```

- [ ] **Step 3: Oferecer o push (PEDIR AUTORIZAÇÃO ANTES)**

```bash
git push -u origin bloco9-ganho-negativo
```

Push exige autorização explícita e separada da do commit.
