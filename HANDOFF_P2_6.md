# HANDOFF_P2_6 — Bloco 6: triagem das hipóteses do Ruling 10 (capacidade × dados)

## 0. Leia primeiro

**As duas hipóteses pendentes do `HANDOFF_P2_3.md` Ruling 10 foram TRIADAS, não
resolvidas.** O que existe aqui é a etapa barata que o próprio Ruling 10a
prescreve ("treinar 2–3 épocas e comparar a curva de IoU_val inicial antes de
comprometer as ~19h"), executada como um fatorial 2×2 completo. O veredito:

| Hipótese | Veredito da triagem |
|---|---|
| **(a) capacidade do modelo** (`base=24`) | **APROVADA** — melhor célula, efeito consistente nos dois níveis de dados |
| **(b) tamanho do dataset** (8.400) | **REPROVADA** — piora nos dois níveis de capacidade |

**A rodada completa de 25 épocas de `base=24` NÃO foi executada.** É o próximo
passo, e está planejada para outra máquina — ver §7 para o procedimento exato.

Nenhum critério do `PLANO §PARTE 2` mudou de veredito nesta sessão. A tabela
consolidada do `HANDOFF_P2_5.md §6` continua válida.

## 1. Estado

| Componente | Estado | Evidência |
|---|---|---|
| Ambiente reconstruído do zero em máquina nova | pronto | 5/5 portões do Bloco 0 passam |
| `data/train`, `data/val`, `data/test` regenerados | pronto | G0.3 passa; 4.200/900/900, seeds disjuntas |
| `data/train_extra` (4.200, seed-base 4) | gerado, **hipótese reprovada** | seeds disjuntas de 1/2/3 verificadas |
| Linha de base desta máquina (2.1/2.6/2.7/2.10) | medida | `reports/part2_strata_base_rodada5.md` |
| Diagnóstico sub-ajuste × sobre-ajuste | medido, **inédito no projeto** | §3.2 |
| Triagem fatorial 2×2 (4 pilotos, 12 épocas) | completa | §3.3 |
| Rodada completa `base=24`, 25 épocas | **NÃO EXECUTADA** | §7 |
| Critério 2.8 (latência) | **não remedido nesta sessão** | ver Armadilha 3 |

## 2. Interface publicada

Duas mudanças, ambas retrocompatíveis. Nenhuma assinatura do `PLANO_PARTE2.md`
foi quebrada.

```python
# identify/extract.py
def load_model(path, device="cpu") -> UNet: ...
```
Agora infere `base` e `levels` do próprio `state_dict` (`enc.0.0.weight`) em vez
de instanciar `UNet()` fixo. Sem isso ele recusa qualquer checkpoint `base=24`
ou `base=32`. Checkpoints `base=16` antigos carregam exatamente como antes.

```bash
# train_unet.py — três flags novas
--base N                 # canais da primeira camada (16 padrão, 24, 32)
--train-dir DIR          # repetível; soma splits. Padrão: data/train
--batches-per-epoch N    # limita os passos de gradiente por época (0 = split inteiro)
```

`--batches-per-epoch` existe por uma razão metodológica, não de conveniência:
sem ele, comparar 4.200 com 8.400 amostras mudaria **duas** variáveis de uma vez
(diversidade dos dados **e** número de passos de gradiente por época). Fixando os
passos em 525, a única diferença entre as células é a diversidade.

## 3. Números medidos

### 3.1 Linha de base desta máquina — e a reprodutibilidade entre máquinas

Medida com `models/unet_stageA_rodada5_alvo_continuo_ep_final.pt` sobre o
`data/test` regenerado. **Os números reproduzem os documentados:**

| Critério | Documentado (rodada 5) | Medido aqui |
|---|---|---|
| 2.1 IoU mediana (U-Net) | 0,6205 | **0,6205** (idêntico) |
| 2.7 `fundo_escuro=False` | 0,5856 (n=173) | **0,5856 (n=173)** (idêntico) |
| 2.10 extrator clássico | 0,7153 | **0,7153** (idêntico) |
| 2.6 ζ (pior parâmetro) | +3,73 p.p. | +3,65 p.p. |
| 2.6 amostras comparáveis | 168/300 | 173/300 |
| 2.6 clássico, ζ | +4,38 p.p. | +4,36 p.p. |

2.6 completo aqui: K +1,01 · τ +2,44 · θ +0,50 · ωₙ +2,03 · **ζ +3,65** (alvo ≤ 3,00).

**A barra que qualquer rodada nova precisa superar é ζ = +3,65 p.p.** Faltam
0,65 p.p. Este número é ligeiramente melhor que os dois checkpoints documentados
(+3,73 da rodada 5 e +3,64 da rodada 4 ficam de cada lado dele).

**Ruling 11 — o dataset É reprodutível entre máquinas, mas os sha256 NÃO são.**
Os hashes das 5 seeds de referência de `reports/part1_metrics.md §6` não
reproduzem nesta máquina. Investigado: o RNG e a geometria são idênticos (as 5
amostras saem com exatamente as mesmas dimensões, dpi e estilo de traço), e os
IoU batem em quatro casas decimais — logo **os pixels são idênticos e a diferença
é só o encoder PNG** (versão de zlib/libpng comprime os mesmos pixels em bytes
diferentes). A instrução do `HANDOFF_P2_5.md §7.6` está correta na prática; o que
precisa de ressalva é o teste de determinismo por hash, que é válido **dentro** de
uma máquina e não **entre** máquinas.

**Ruling 12 — os critérios de OCR são sensíveis à versão do tesseract.** A única
diferença real na linha de base (173 vs 168 amostras comparáveis, ζ +3,65 vs
+3,73) vem do `tesseract 5.5.3` desta máquina, que calibra 5 amostras a mais.
Os critérios 2.3/2.4/2.5/2.6/2.9 carregam essa sensibilidade e o handoff anterior
não a registrava. Quem comparar números de OCR entre máquinas precisa registrar a
versão do tesseract junto.

### 3.2 Diagnóstico: a rede sub-ajusta (medição inédita, e ela prevê o resultado)

IoU em 900 amostras de treino (mesmo n que val) contra 900 de validação:

| Checkpoint | IoU_train | IoU_val | gap |
|---|---|---|---|
| rodada 4 (limiar 32) | 0,5551 | 0,5511 | **+0,0040** |
| rodada 5 (alvo contínuo) | 0,7502 | 0,7439 | **+0,0062** |

Um gap de meio ponto percentual significa que o modelo **não sobre-ajusta** — ele
nem consegue ajustar os dados que já tem. Isso é diagnóstico de viés (capacidade
insuficiente), não de variância. **A previsão feita ANTES de qualquer treino foi:
a hipótese (b) não vai ajudar, porque não há sobre-ajuste para mais dados
combaterem; a (a) é a única com mecanismo.** A triagem confirmou as duas metades.

Isto vale como metodologia para a monografia: uma medição de 20 minutos previu
corretamente o resultado de 12 épocas de treino. As cinco rodadas anteriores
nunca a fizeram.

### 3.3 Triagem fatorial 2×2

Todas as células: 3 épocas × **525 passos de gradiente**, `torch.manual_seed(20260817)`,
mesmo `data/val`, mesmo escalonador. Para `base=16`/`base=24` com 4.200 amostras,
525 passos = uma época inteira (4.200/8 = 525); para as células de 8.400, 525
passos = **meia** passagem pelo split.

| célula | ep 0 | ep 1 | ep 2 | s/época |
|---|---|---|---|---|
| **A1** base=16, 4.200 (controle) | 0,3908 | 0,4893 | 0,6794 | 2.037–2.233 |
| **A2** base=24, 4.200 (capacidade) | 0,4156 | 0,4787 | **0,7445** | 2.982–3.075 |
| **B** base=16, 8.400 (dados) | 0,3500 | 0,4926 | 0,6405 | 1.768–1.775 |
| **C** base=24, 8.400 (interação) | 0,4379 | 0,5538 | 0,7203 | 2.926–3.460 |

Efeitos, medidos no IoU_val da época 2:

| | 4.200 | 8.400 | **efeito dos dados** |
|---|---|---|---|
| **base=16** | 0,6794 | 0,6405 | **−0,039** |
| **base=24** | 0,7445 | 0,7203 | **−0,024** |
| **efeito da capacidade** | **+0,065** | **+0,080** | |

O que sustenta o veredito, apesar de uma seed só: **cada efeito aparece nas duas
linhas do fatorial, com sinal igual e magnitude parecida.** Capacidade ajuda em
+0,065 e +0,080; dados atrapalham em −0,039 e −0,024. Não são pontos soltos.

**Sobre a interação (célula C).** A penalidade dos dados encolhe de −0,039 para
−0,024 quando a rede cresce — ou seja, mais capacidade **absorve** melhor mais
dados, que é o mecanismo que se esperaria. Mas o efeito é pequeno demais: continua
negativo, e a célula C fica **abaixo** da A2. A interação existe e não resgata a
hipótese (b). Sem essa célula, a conclusão "dados não ajudam" ficaria confundida
com "a rede é pequena demais para aproveitá-los"; com ela, essa confusão está
descartada.

### 3.4 Custo de treino nesta máquina (i7-13620H, 16 threads, CPU-only, 15,6 GB)

Ótimo medido: **`OMP_NUM_THREADS=6`** (só P-cores). 10 threads é 6% mais lento,
16 threads é 17% mais lento — as E-cores atrapalham em vez de ajudar.

| `base` | Parâmetros | s/época (525 passos) | 25 épocas |
|---|---|---|---|
| 16 | 1.942.289 | ~1.800–2.200 | ~13 h |
| 24 | 4.367.641 | ~3.030 | **~21 h** |
| 32 | 7.762.465 | ~5.100 (extrapolado) | ~36 h |

As contagens de parâmetros batem exatamente com a tabela do Ruling 10a.
Geração do dataset: **71 s** para as 6.000 amostras (o `HANDOFF_P2_5.md §7.6`
estimava ~7 min; esta máquina é bem mais rápida nisso e mais lenta no treino).

## 4. Rulings

11. **Dataset reprodutível entre máquinas; sha256 não** — ver §3.1.
12. **Critérios de OCR são sensíveis à versão do tesseract** — ver §3.1.
13. **A rede sub-ajusta; o diagnóstico previu a triagem** — ver §3.2. Consequência
    prática: qualquer hipótese futura deve ser triada primeiro contra o gap
    treino−validação. Se o gap continuar perto de zero, hipóteses de
    regularização/mais dados/augmentation são previsivelmente inúteis, e o eixo
    a atacar é capacidade ou arquitetura.
14. **A trajetória de treino desta máquina é sistematicamente melhor que a da
    máquina original, com o mesmo código e os mesmos dados.** A rodada 5 chegou a
    IoU_val 0,5895 em 25 épocas; o piloto A1, mesma configuração, passou disso em
    **3** (0,6794). Como os pixels são idênticos (§3.1), a causa mais provável é a
    versão do torch (2.13.0+cpu aqui), que muda inicialização de pesos,
    embaralhamento e numérica. **Se isso se confirmar numa rodada completa, parte
    do teto de IoU atribuído à arquitetura nas cinco rodadas pode ter sido da
    stack de software.** NÃO afirmar isso sem uma rodada de 25 épocas — 3 épocas
    não autorizam a conclusão. Fica também sem explicação, e sem impacto em
    critério nenhum, a discrepância do IoU_val do log da rodada 5 (0,5895 lá,
    0,7439 ao recarregar o mesmo checkpoint aqui).
15. **A triagem mede IoU de máscara, não ζ.** Todo o ganho de +0,065 da hipótese
    (a) é em IoU_val. O critério que decide é o 2.6, e a rodada 5 já provou uma
    vez que **ganhar IoU não implica fechar ζ** (melhor IoU das cinco rodadas,
    e mesmo assim ζ piorou 0,09 p.p.). A triagem autoriza gastar as 21 h; não
    promete o resultado.

## 5. Armadilhas

1. **Instalar o torch exige o índice do PyTorch.** `pip install -r requirements.txt`
   falha em `torch==2.13.0+cpu` — o sufixo `+cpu` não existe no PyPI. Instalar o
   resto primeiro e depois:
   `pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu`
2. **O binário do tesseract não vem com o `pytesseract`.** É pacote de sistema
   (`sudo dnf install -y tesseract` no Fedora). Sem ele, `test_env.py` reprova e
   todos os critérios de OCR ficam sem medição.
3. **A latência (2.8) não foi remedida nesta sessão** e o número do
   `HANDOFF_P2_5.md` continua sendo o da máquina antiga. Foi excluído de propósito:
   a medição da linha de base rodou com um treino em paralelo, e latência sob
   contenção é um número falso. Quem retomar precisa medir 2.8 com a máquina ociosa.
4. **Pressão de memória degrada o treino de forma não óbvia.** O piloto C levou
   4h37 de parede para 2h39 somadas nas épocas — ~2 h caem **fora** dos
   cronômetros por época, coincidindo com o swap saturado (7,8 de 8,0 GB) por uso
   simultâneo do desktop. O custo por época subiu só ~15%; o resto do overhead não
   está explicado. Para a rodada completa, **rodar com a máquina livre** e, se a
   RAM for apertada, baixar `num_workers` no `DataLoader` de `train_unet.py`.
5. **`reports/part2_strata.md` é sobrescrito a cada execução de `tests/part2/`.**
   A linha de base desta sessão foi arquivada em
   `reports/part2_strata_base_rodada5.md` antes de qualquer rodada nova.

## 6. Artefatos desta sessão

Checkpoints dos pilotos (3 épocas cada, **não são modelos finais**):
`models/piloto_base16_4200.pt` · `models/piloto_base24_4200.pt` ·
`models/piloto_base16_8400.pt` · `models/piloto_base24_8400.pt`

Logs: `logs/piloto_*.log` (treino) · `logs/medir_base_rodada5.log` (linha de base)
· `logs/regen_dataset.log` (geração).

Relatório da linha de base: `reports/part2_strata_base_rodada5.md`.

## 7. Como executar a rodada completa

**Nesta máquina (a que produziu a triagem), §7.1 e §7.2 JÁ ESTÃO FEITOS** — o
`.venv` está montado, o `tesseract 5.5.3` instalado e os três splits gerados e
validados. Quem retomar aqui pode ir direto ao §7.3. As duas primeiras seções
existem para reconstruir tudo em uma máquina limpa.

### 7.1 Reconstruir o ambiente (só em máquina nova)

```bash
python3.11 -m venv .venv
grep -v '^torch==' requirements.txt > /tmp/req.txt
.venv/bin/pip install -r /tmp/req.txt
.venv/bin/pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
sudo dnf install -y tesseract        # ou o equivalente da distro
```

Com GPU, trocar a linha do torch pela build de CUDA — `train_unet.py` já
seleciona `cuda` sozinho quando `torch.cuda.is_available()`.

### 7.2 Regenerar o dataset (só em máquina nova; ~1 a 7 min)

```bash
.venv/bin/python -m dataset.generator data/train 4200 1
.venv/bin/python -m dataset.generator data/val   900  2
.venv/bin/python -m dataset.generator data/test  900  3
.venv/bin/python -m pytest tests/part2/test_env.py -q      # 5 portões do Bloco 0
```

**Não gerar `data/train_extra`** — a hipótese (b) está reprovada (§3.3).

### 7.3 A rodada completa: `base=24`, 25 épocas

```bash
OMP_NUM_THREADS=6 .venv/bin/python train_unet.py \
    --epochs 25 --base 24 \
    --out models/unet_stageA_rodada6_base24.pt \
    2>&1 | tee logs/train_unet_rodada6_base24.log
```

`--batches-per-epoch` **não** é necessário aqui: 4.200/8 = 525, então a época
inteira já são os mesmos 525 passos da triagem. Custo: ~21 h em CPU nesta
classe de máquina; muito menos com GPU.

**Não sobrescrever `models/unet_stageA.pt` durante o treino** — os testes o usam
como modelo padrão, e a linha de base depende dele estar intacto.

### 7.4 Medir contra a linha de base

```bash
cp models/unet_stageA_rodada6_base24.pt models/unet_stageA.pt
.venv/bin/python -m pytest tests/part2/test_part2.py -q      # com a máquina OCIOSA (2.8)
cp reports/part2_strata.md reports/part2_strata_rodada6_base24.md
```

Comparar contra §3.1. O número que decide é **ζ: precisa cair de +3,65 para ≤ 3,00 p.p.**
Registrar também a versão do tesseract junto (Ruling 12).

### 7.5 Se ζ não fechar

Os três eixos de preparo do alvo (limiar 127/0/32, contínuo) e as duas hipóteses
genéricas do Ruling 10 estarão todos esgotados. O próximo passo deixa de ser
genérico e passa a ser o do **Ruling 10c**, que continua sem execução: olhar
amostra a amostra **quais** casos de ζ erram mais (sistemas muito subamortecidos,
picos estreitos de *overshoot*) em vez de mais uma mudança global de treino.
O Ruling 13 acrescenta um filtro barato a aplicar antes de qualquer hipótese nova:
medir o gap treino−validação primeiro.
