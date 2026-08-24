# HANDOFF_P2_0 — Bloco 0: Infraestrutura, dataset em disco e guarda do relatório

## 1. Estado

| Componente | Estado | Evidência |
|---|---|---|
| `.venv` (Python 3.11) | pronto | `.venv/bin/python --version` → `Python 3.11.15` |
| Dependências Parte 2 (`torch`, `cv2`, `skimage`, `pytesseract`) | instaladas | `tests/part2/test_env.py::test_g0_1_*` PASSED |
| Binário `tesseract` | instalado (pacote de sistema) | `tesseract 4.1.1` |
| Guarda do relatório da Parte 1 (`tests/conftest.py`) | aplicada e validada | `test_g0_4_*` PASSED + sanity check real (ver §3) |
| `data/train` (4.200), `data/val` (900), `data/test` (900) | gerados, disjuntos | `test_g0_3_*` PASSED |
| `requirements.txt` | congelado | `pip freeze` após instalar tudo |
| `reports/part1_metrics.md` | íntegro (394 linhas, sem banner de parcialidade) | ver §3, Armadilha 1 |

## 2. Interface publicada

Nada novo de código de produção neste bloco — só infraestrutura e a guarda em
`tests/conftest.py::pytest_sessionfinish`. Assinatura consumida pelo Bloco 1:

```python
from dataset.generator import load_sample, generate_dataset  # já existiam da Parte 1
```

Variável de decisão: **`TCC_DEVICE = "cpu"`** (não há GPU NVIDIA nesta máquina —
ver §4, Ruling 1). Não há variável de ambiente de fato exportada; a decisão é
lida em runtime por `torch.cuda.is_available()` em qualquer ponto do código
(ver `test_g0_2_dispositivo_declarado`).

## 3. Números medidos

| Portão | Alvo | Medido | Veredito |
|---|---|---|---|
| G0.1 | `import torch, cv2, skimage, pytesseract` sem erro | passou | ✅ |
| G0.1 (binário) | `tesseract` no PATH | `tesseract 4.1.1` | ✅ |
| G0.2 | Decisão de dispositivo registrada | `cpu` (sem hardware NVIDIA — `lspci` só mostra Intel UHD) | ✅ |
| G0.3 | 4.200/900/900, seeds disjuntas | 4200/900/900 confirmado, `seeds[train] ∩ seeds[val] = seeds[train] ∩ seeds[test] = seeds[val] ∩ seeds[test] = ∅` | ✅ |
| G0.4 | `pytest tests/part2` não corrompe `reports/part1_metrics.md` | testado com o teste sintético **e** com uma execução real (`pytest tests/part2 -q`, diff byte-a-byte contra backup) | ✅ |
| — | `pytest -q` (suíte completa, com a guarda aplicada) | **37 passed** (33 da Parte 1 + 4 de `test_env.py`, antes de escrever `test_g0_3`) em 787 s | ✅ |
| — | Geração dos 3 splits | train 4m37s, val 1m01s, test 1m01s (total ≈ 6m40s), 422 MB em disco | dentro do orçado, bem mais rápido que os 30 min do PLANO_PARTE2 §Bloco 0 |

## 4. Rulings

1. **Não há timebox de driver NVIDIA — não há hardware NVIDIA nesta máquina.**
   O `PLANO_PARTE2.md` (Bloco 0, Passo 1) supõe o notebook do `PLANO.md`
   (RTX 4050, Fedora 43 + Secure Boot). O ambiente real de execução é
   **Ubuntu 22.04, GPU apenas Intel UHD integrada** (`lspci` confirma: nenhum
   controlador NVIDIA presente). `nvidia-smi` nem existe no sistema. A decisão
   de dispositivo é `cpu` por ausência de hardware, não por falha de driver —
   o timebox de 3h do Passo 1 foi pulado por não haver o que tentar.
   **Consequência prática:** o Bloco 3 (U-Net) roda inteiramente em CPU nesta
   máquina, sem contingência de GPU disponível. Isso pesa contra o orçamento de
   20–35h/rodada que o próprio `PLANO_PARTE2.md §1.8` já cita como pior caso —
   ver o handoff do Bloco 3 para os números medidos aqui.
2. **`python3.11` e `tesseract-ocr` exigiram instalação via `apt` com `sudo`**,
   não `dnf` como o `PLANO_PARTE2.md` assume (ambiente é Debian/Ubuntu, não
   Fedora). Pacotes: `python3.11`, `python3.11-venv`, `tesseract-ocr`
   (a PPA `deadsnakes` já estava configurada no sistema). Executado pelo
   usuário fora desta sessão (não tenho — nem devo ter — acesso à senha
   `sudo`).
3. **Guarda de `pytest_sessionfinish` estendida com uma chamada extra a
   `record_block("selection", sel)`** depois de anexar a detecção de seleção
   por caminho (`args`). O trecho literal do Passo 9 do `PLANO_PARTE2.md`
   só recalcula a string `sel` local sem gravá-la de volta em
   `RESULTS["blocks"]["selection"]` — sem essa segunda chamada, o banner de
   "relatório parcial" (`tests/conftest.py:379`) nunca dispararia para seleção
   por caminho, que é exatamente o caso que o passo diz querer cobrir. Testado:
   `pytest tests/part2` produz relatório idêntico byte a byte ao anterior
   (guarda de curto-circuito age antes, então o banner nem chega a ser
   relevante neste caso específico — mas ficaria correto para uma seleção por
   caminho que *misturasse* testes da Parte 1 e da Parte 2, cenário que a
   guarda de `if not RESULTS["criteria"]: return` sozinha não cobre).
4. **`add_noise` não foi passado explicitamente na geração dos splits** —
   ficou no default de `generate_dataset` (`add_noise=True`), que é o mesmo
   comportamento usado pela Parte 1 para o conjunto "noisy". Não há instrução
   no `PLANO_PARTE2.md` dizendo o contrário; os testes de percepção (Bloco 1
   em diante) precisam de imagens realistas, então ruído ligado é a escolha
   coerente com o resto do plano (2.6 mede degradação sobre o **mesmo** tipo
   de amostra que a Parte 1 usa como oráculo).

## 5. Armadilhas

1. **O defeito do Passo 7 é real e mordeu na prática, não só em teoria.**
   Antes de aplicar a guarda, rodar o teste sintético
   `test_g0_4_rodar_parte2_nao_apaga_relatorio_da_parte1` (que invoca
   `pytest_sessionfinish` diretamente com `RESULTS["criteria"] = {}`)
   **de fato sobrescreveu** `reports/part1_metrics.md`, reduzindo-o de ~500
   para 16 linhas. Tive que regenerar com `pytest -q` completo (13 min) para
   restaurar. **Não rode o teste antes da guarda em um ambiente onde o
   relatório da Parte 1 importa** — rode primeiro num branch/cópia se for
   repetir isso por algum motivo.
2. **A máquina de execução é ~6× mais lenta que a do `HANDOFF.md` da Parte 1
   em wall-clock.** A suíte completa (`pytest -q`) levou 787–822s aqui contra
   os 128s citados no `HANDOFF.md §9`. O tempo de CPU total (`user`) é
   parecido (~95 min), então não é falta de paralelismo — é clock/IPC mais
   baixo por núcleo (CPU Intel de notebook mais antiga, sem GPU dedicada).
   Isso é o sinal mais forte de que o Bloco 3 (treino da U-Net) vai custar
   mais que os 20–35h já citados como pior caso no `PLANO_PARTE2.md §1.8` —
   tratar como estimativa otimista.
3. **Geração dos splits foi bem mais rápida que o orçado** (6m40s vs. ~30min
   estimados para 6.000 imagens) — nenhuma armadilha aqui, só registrar que o
   Passo 12 do plano superestimou o tempo para esta configuração de hardware
   (8 threads, sem I/O lento).

## 6. O que o próximo bloco precisa saber

1. **`.venv/bin/python` é Python 3.11.15**; todo comando subsequente desta
   Parte 2 deve usar esse interpretador, nunca o `python3` do sistema
   (3.10.12) nem o `python3.10` usado informalmente na Parte 1.
2. **`TCC_DEVICE = "cpu"` é definitivo nesta máquina** — não há hardware
   NVIDIA. Qualquer código do Bloco 3 que pergunte
   `torch.cuda.is_available()` vai receber `False`; não há necessidade de
   revisitar essa decisão.
3. **`data/train`, `data/val`, `data/test` já existem e estão validados**
   (4200/900/900, seeds disjuntas, `add_noise=True`). O Bloco 1 pode consumir
   `data/test` diretamente via a fixture `test_samples` do
   `tests/part2/conftest.py` a ser criada.
4. **A guarda em `tests/conftest.py` está ativa.** Rodar `pytest tests/part2`
   isoladamente é seguro; rodar `pytest -q` (tudo) continua regenerando o
   relatório da Parte 1 normalmente, porque nesse caso `RESULTS["criteria"]`
   não fica vazio.
5. **Disco:** 422 MB usados pelo dataset, 19 GB livres no filesystem raiz
   (bem menos que os 305 GB que o `PLANO_PARTE2.md` presumia). Suficiente para
   o restante da Parte 2, mas sem margem para datasets adicionais grandes sem
   necessidade — evitar gerar splits extras "por via das dúvidas".
6. Próximo passo: **Bloco 1** — `identify/calibrate.py`
   (`detect_plot_bbox`, `detect_tick_pixels`), sem OCR.
