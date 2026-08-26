# HANDOFF_P2_7 — Bloco 7: a rodada completa `base=24` em GPU, e o fechamento do critério 2.6

## 0. Leia primeiro

**O critério 2.6 FECHOU.** A rodada completa que o `HANDOFF_P2_6.md` §7.3 prescreveu
foi executada, em GPU, e o pior parâmetro caiu de **ζ +3,65 p.p. para +1,05 p.p.**
contra um alvo de ≤ 3,00. A hipótese (a) do Ruling 10 — capacidade do modelo —
está **confirmada fora do regime de triagem**: ela fecha o critério que cinco
rodadas anteriores não fecharam.

| | Base (rodada 5) | **Rodada 6 (`base=24`)** | Alvo |
|---|---|---|---|
| **2.6 (pior parâmetro)** | +3,65 p.p. ❌ | **+1,05 p.p. ✅** | ≤ 3,00 |

Falhas da suíte: **7 → 6**.

Duas coisas que este handoff **não** autoriza a dizer:

1. **A Parte 2 não está concluída.** Seis critérios seguem reprovados
   (2.1, 2.2-piso, 2.4, 2.5, 2.7, 2.8) e o §7 mostra que eles se dividem em
   duas famílias independentes, só uma das quais tem a ver com a rede.
2. **Aumentar o dataset continua reprovado** — ver Ruling 24 e §3.5.

## 1. Estado

| Componente | Estado | Evidência |
|---|---|---|
| Driver NVIDIA + build CUDA do torch | instalado e verificado | §2.1 |
| Rodada completa `base=24`, 25 épocas, GPU | **executada** | `logs/train_unet_rodada6_base24.log` |
| Linha de base (rodada 5) remedida em GPU | **medida** | `reports/part2_strata_base_rodada5_gpu.md` |
| Rodada 6 medida contra os critérios | **medida** | `reports/part2_strata_rodada6_base24.md` |
| Critério 2.6 (ζ) | **APROVADO** (+1,05 p.p.) | §3.3 |
| Critério 2.8 (latência) | medido com máquina ociosa, **reprovado** | §3.4 |
| `models/unet_stageA.pt` | **é a rodada 6** (`base=24`) | §5, Armadilha 4 |
| Critérios 2.1, 2.2-piso, 2.4, 2.5, 2.7 | reprovados | §3.4, §7 |

## 2. Interface publicada

Nenhuma assinatura mudou. O que mudou é ambiente, e é retrocompatível.

### 2.1 O `.venv` agora tem CUDA

```bash
.venv/bin/pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

**É a mesma versão dos pilotos (2.13.0), só trocando `+cpu` por `+cu130`.** Isso é
deliberado e importa: o Ruling 14 documentou que este projeto é sensível à versão
do torch, então manter a versão fixa isola o eixo "dispositivo" do eixo "stack de
software". Sem isso, a rodada 6 mudaria duas variáveis de uma vez.

O driver proprietário (`akmod-nvidia` 580.178.04) já está instalado e carregado.
`nvidia-smi` **não** está no PATH — o pacote `xorg-x11-drv-nvidia-cuda` não foi
instalado —, mas isso não bloqueia nada: `libcuda.so` vem do `-cuda-libs`, que
está presente. Ver a Armadilha 6 do `HANDOFF_P2_6.md` Ruling 17 sobre por que
`nvidia-smi` ausente **não** é evidência de ausência de GPU.

`requirements.txt` **não foi alterado** — ele continua fixando `torch==2.13.0+cpu`.
Quem reconstruir o ambiente para GPU precisa trocar essa linha à mão (§6.1).

## 3. Números medidos

### 3.1 A rodada 6

```
treino: 4200 amostras de ['data/train']  base=24  525 passos/epoca
parametros: 4367641
melhor IoU_val=0.7880 -> models/unet_stageA_rodada6_base24.pt
```

Melhor IoU_val da história do projeto, atingido na **época 15**:

| Checkpoint | IoU_val (medido nesta máquina) |
|---|---|
| rodada 4 (limiar 32) | 0,5511 |
| rodada 5 (alvo contínuo) | 0,7439 |
| **rodada 6 (`base=24`)** | **0,7880** |

Trajetória: convergência rápida e platô firme. Depois da época 05 nada saiu da
faixa **0,777–0,788** — 20 épocas dentro de 0,011. O `ReduceLROnPlateau` cortou o
LR **nove vezes**, de 3,00e-04 a 5,86e-07, sem destravar patamar novo.

A queda da época 01 (0,4099 → 0,3991) destoou das quatro células do fatorial, que
subiram todas nessa época. Não teve consequência: a época 02 recuperou para 0,7526,
acima dos 0,7445 do piloto A2, e a trajetória voltou a colar na triagem.

### 3.2 Custo real em GPU

| | s/época | 25 épocas |
|---|---|---|
| Estimativa CPU (Ruling 16) | 3.030 | ~21,0 h |
| **Medido em GPU (RTX 4050 Mobile)** | **258 (média)** | **1,79 h** |

**11,7×.** O Ruling 17 supôs "uma ordem de grandeza" sem medir; confirmado com folga.

Memória: pico alocado **4,27 GB**, reservado **4,74 GB** de 5,64 GB. **Batch 8 e
512² foram preservados**, então a comparabilidade com os pilotos do
`HANDOFF_P2_6.md` §3.3 está intacta. Nenhum OOM em 25 épocas. Rodou com
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` por causa dos 0,73 GB de folga
— é opção de alocador, não toca em numérica.

### 3.3 O fechamento do 2.6

Todos os cinco parâmetros melhoraram, não só o que decide:

| Parâmetro | Base (rodada 5) | **Rodada 6** | Alvo |
|---|---|---|---|
| K | +1,01 p.p. | **+0,25** | ≤ 3,00 ✅ |
| τ | +2,44 p.p. | **+0,54** | ≤ 3,00 ✅ |
| θ | +0,50 p.p. | **+0,27** | ≤ 3,00 ✅ |
| ωₙ | +2,03 p.p. | **+0,82** | ≤ 3,00 ✅ |
| **ζ** | **+3,65 ❌** | **+1,05 ✅** | ≤ 3,00 |

Amostras comparáveis subiram de **173/300 para 188/300** — mais casos passam a
convergir na mesma ordem.

**O controle interno que valida o resultado.** O diff entre os dois relatórios
mostra que **só os critérios que dependem da U-Net mudaram**. Saíram byte a byte
idênticos: G1.1, G1.2x/y, G1.3-2/3/4, G3b.1 (e todos os seus estratos), G3b.2,
G3b.3, 2.3, 2.4, 2.5, 2.9 (e estratos), 2.11, 2.2-piso (e todos os estratos).

O mais forte é o **2.6-clássico: +4,36 p.p., inalterado até a segunda casa**. O
extrator clássico não usa a rede, logo tinha que ficar parado — e ficou. Isso
descarta que a melhora venha de mudança de ambiente, de versão de OCR ou de acaso
na suíte. **A única variável que se moveu foi o checkpoint.**

### 3.4 O que continua reprovado

| Critério | Base | Rodada 6 | Alvo | Depende da U-Net? |
|---|---|---|---|---|
| 2.1 IoU mediana | 0,6205 | 0,6478 | ≥ 0,85 ❌ | **sim** |
| 2.7 `fundo_escuro=False` | 0,5856 | 0,6330 | ≥ 0,75 ❌ | **sim** |
| 2.8 latência | 885 ms | 854 ms | < 500 ms ❌ | parcial |
| 2.2-piso p95 | 6,70 px | 6,70 px | ≤ 5 px ❌ | **não** |
| 2.4 falso alarme | 0,197 | 0,197 | < 0,05 ❌ | **não** |
| 2.5 rejeições corretas | 0,678 | 0,678 | ≥ 0,90 ❌ | **não** |
| 2.9 cobertura | 0,803 | 0,803 | ≥ 0,90 ❌ | **não** |

**Critério 2.8, primeira medição honesta nesta máquina.** A Armadilha 3 do
`HANDOFF_P2_6.md` fica resolvida: medido com a máquina ociosa (load average 0,09
antes do disparo), **mediana 854 ms, p95 2266 ms**. Reprova. Como o extrator
clássico faz o mesmo percurso em **14 ms** (G3b.4), a latência é dominada pelo
OCR e pela calibração, **não** pela rede — nem a GPU a resolveu.

### 3.5 Ruling 13 aplicado à rodada 6: a rede AINDA sub-ajusta

Mesmo protocolo do `HANDOFF_P2_6.md` §3.2 — 900 amostras de treino (mesmo n que
val) contra as 900 de validação:

| Checkpoint | IoU_train | IoU_val | gap |
|---|---|---|---|
| rodada 5 (`base=16`) | 0,7502 | 0,7439 | **+0,0062** |
| **rodada 6 (`base=24`)** | **0,7985** | **0,7880** | **+0,0105** |

A linha da rodada 5 **reproduz o §3.2 em todas as quatro casas decimais**, o que
valida o protocolo e reforça o Ruling 18.

O gap quase dobrou (+0,0062 → +0,0105), que é a direção esperada quando se
aumenta capacidade. Mas **em valor absoluto continua ~1 p.p.**: a rede `base=24`
ainda ajusta os dados de treino quase tão mal quanto os de validação. **Isso é
sub-ajuste, não sobre-ajuste — mesmo diagnóstico do Bloco 6, um nível de
capacidade acima.**

Consequência direta, e é o filtro que o Ruling 13 manda aplicar antes de qualquer
hipótese nova: **mais dados, regularização e *augmentation* continuam
previsivelmente inúteis.** Não há variância para eles combaterem. O eixo que
resta é capacidade ou arquitetura.

## 4. Rulings

18. **Trocar CPU por GPU não move nenhum critério de acurácia.** A linha de base
    da rodada 5 foi remedida inteira em GPU e reproduziu a de CPU em **todas as
    casas decimais**: 2.1 = 0,6205, 2.7 = 0,5856 (n=173), 2.10 = 0,7153,
    2.6 ζ = +3,65 p.p. (n=173), 2.6-clássico ζ = +4,36 p.p. O §7.6.3 do
    `HANDOFF_P2_6.md` temia que a troca de dispositivo invalidasse comparações;
    o temor era legítimo em princípio e **falso na prática** — a inferência é
    determinística o bastante para o pós-processamento absorver a diferença
    numérica. **O único critério que muda é o 2.8 (latência)**, e esse muda por
    construção. Quem comparar números entre CPU e GPU neste projeto pode fazê-lo
    para tudo, **menos** para 2.8.

19. **Custo real em GPU: 1,79 h contra ~21 h estimadas em CPU (11,7×).** Ver §3.2.
    Batch 8 e 512² preservados em 4,74 de 5,64 GB de VRAM. **Isto muda a economia
    do projeto**: experimentos que o Bloco 6 precisou triar com 3 épocas agora
    cabem completos. As quatro células do fatorial, 25 épocas cada, custariam
    ~7 h — menos que a metade do que UMA célula custava em CPU.

20. **A hipótese (a) do Ruling 10 está CONFIRMADA, e fecha o critério 2.6.**
    `base=24` levou ζ de +3,65 para +1,05 p.p. A triagem do Bloco 6 previu a
    direção (+0,065 de IoU na época 2) e a rodada completa entregou muito mais do
    que o mínimo necessário: faltavam 0,65 p.p. e o corte foi de 2,60 p.p.
    **A hipótese (b) — mais dados — segue reprovada, e agora por duas vias
    independentes** (o fatorial do Bloco 6 e o gap do §3.5).

21. **IoU de máscara é um proxy ruim para acurácia end-to-end, e o critério 2.10
    inverte de sinal.**

    | | IoU de máscara | ζ (end-to-end) |
    |---|---|---|
    | extrator clássico | **0,7153** (melhor) | +4,36 p.p. (pior) |
    | U-Net rodada 6 | 0,6478 (pior) | **+1,05 p.p.** (melhor) |

    O extrator clássico **continua ganhando em IoU e perde em ζ por um fator de
    4**. Não é diferença sutil: é inversão completa de ranking entre os dois
    critérios. E internamente vale o mesmo — a rodada 6 ganhou só +0,027 de IoU
    sobre a rodada 5 (0,6205 → 0,6478) e derrubou ζ em 2,60 p.p. **O ganho não
    veio de "máscara mais parecida", veio de máscara melhor onde ζ é lido.**

    Consequência para a monografia: **2.1 (IoU ≥ 0,85) foi calibrado sobre a
    premissa de que IoU prediz qualidade end-to-end, e os dados agora contradizem
    essa premissa.** Isso NÃO autoriza mover a meta — autoriza discutir, com
    número, o que 2.1 de fato mede. O Ruling 10c (olhar amostra a amostra QUAIS
    casos erram) é o caminho para essa discussão.

22. **As 25 épocas foram desnecessárias.** O melhor checkpoint saiu na **época 15**,
    e a **época 05 já entregava 0,7817 — 99,2% do resultado final, em 1/5 do
    tempo.** Nove cortes de LR não destravaram patamar novo. Para experimentos de
    triagem futuros, 8–10 épocas bastam para ranquear configurações de `base`.
    Isso reforça o Ruling 13 pelo outro lado: `base=24` também satura, só que num
    teto mais alto.

23. **As falhas restantes se dividem em DUAS famílias independentes, e o diff
    prova a separação.** Ver §3.4 e §7.2. Trocar o checkpoint mexeu em 2.1, 2.7 e
    2.8; deixou 2.2-piso, 2.4, 2.5 e 2.9 **byte a byte idênticos**. Melhorar a
    U-Net, por melhor que fique, **não pode** fechar esses quatro. Qualquer plano
    que trate "os critérios que faltam" como um bloco único está errado.

24. **A rede sub-ajusta também em `base=24`** — gap +0,0105. Ver §3.5. Mais dados,
    regularização e *augmentation* seguem previsivelmente inúteis.

## 5. Armadilhas

1. **`models/unet_stageA.pt` agora é `base=24`, não `base=16`.** Ele foi
   sobrescrito pela rodada 6 conforme o `HANDOFF_P2_6.md` §7.4. Quem instanciar
   `UNet()` fixo e der `load_state_dict` nele **quebra**. `load_model()` já infere
   `base`/`levels` do próprio `state_dict` (Bloco 6 §2) e carrega sem problema.
   A rodada 5 continua íntegra em
   `models/unet_stageA_rodada5_alvo_continuo_ep_final.pt` — reverter é um `cp`.

2. **A linha A.0 do relatório está errada, e o teste tem um bug latente.**
   `test_unet_tamanho_declarado` faz `sum(p.numel() for p in UNet().parameters())`
   — instancia o **default `base=16`** em vez de ler o checkpoint em uso. Reporta
   1,94 M para um modelo que tem **4,37 M**. Pior: o `assert 0.5e6 <= n <= 2.5e6`
   **reprovaria** se algum dia medisse o modelo real. Não afeta veredito nenhum
   hoje, mas a linha A.0 passou a descrever um modelo que não está em uso.

3. **O n do 2.6 mudou: 173 → 188.** Os subconjuntos comparáveis não são idênticos
   entre base e rodada 6, então **a comparação não é estritamente pareada**. A
   magnitude (2,60 p.p.) é grande demais para composição de amostra explicar
   sozinha, mas o número exato carrega essa imprecisão. Quem quiser o valor
   pareado precisa reprocessar restringindo à interseção dos dois conjuntos.

4. **Sondar a máquina durante a suíte contamina o 2.8.** Durante a primeira
   execução, comandos de acompanhamento levaram o load average de 0,09 a 3,22.
   O 2.8 mede latência; qualquer `pgrep`/`python -c` concorrente falseia o número.
   Rodar a suíte e **não tocar na máquina** até terminar.

5. **`reports/part2_strata.md` continua sendo sobrescrito a cada execução de
   `tests/part2/`** (herdada do Bloco 6). Ambas as execuções desta sessão foram
   arquivadas logo após terminarem.

6. **`requirements.txt` ainda fixa `torch==2.13.0+cpu`.** Não foi alterado. Quem
   reconstruir o ambiente seguindo o `HANDOFF_P2_6.md` §7.1 ao pé da letra
   **volta para CPU** e vai achar que a GPU não funciona. Ver §6.1.

## 6. Artefatos desta sessão

Checkpoint final: `models/unet_stageA_rodada6_base24.pt` (16,7 MB, `base=24`),
copiado sobre `models/unet_stageA.pt`.

Logs: `logs/train_unet_rodada6_base24.log` (25 épocas) ·
`logs/medir_base_rodada5_gpu.log` (linha de base em GPU, 37min18s) ·
`logs/medir_rodada6_base24_gpu.log` (rodada 6, 36min57s).

Relatórios: `reports/part2_strata_base_rodada5_gpu.md` (a linha de base a bater) ·
`reports/part2_strata_rodada6_base24.md` (o resultado).

### 6.1 Ambiente

Idêntico ao de `reports/ambiente_maquina_triagem.md`, **exceto** o torch e a GPU:

- `torch 2.13.0+cu130` (era `2.13.0+cpu`) — **mesma versão**, só o dispositivo muda
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9, **5,64 GB** de
  VRAM (a especificação de 6 GB do Ruling 17 era nominal), bf16 suportado
- driver `akmod-nvidia` 580.178.04; `nvidia-smi` **não** instalado
- `tesseract 5.5.3` / `leptonica-1.87.0` — **a mesma da linha de base**, então o
  Ruling 12 está satisfeito e os critérios de OCR são comparáveis

Para reconstruir em GPU, o §7.1 do Bloco 6 vale com uma linha trocada:

```bash
.venv/bin/pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

## 7. Para onde ir agora

### 7.1 O que NÃO fazer

**Não aumentar o dataset.** Reprovado por duas vias independentes: o fatorial 2×2
do Bloco 6 (piora em ambos os níveis de capacidade) e o gap de +0,0105 do §3.5
(não há sobre-ajuste para mais dados combaterem). O mesmo vale para
*augmentation* e regularização — Ruling 24.

**Não tratar os critérios restantes como um bloco único** — Ruling 23.

### 7.2 As duas famílias

**Família A — depende da U-Net: 2.1 (IoU 0,6478 / 0,85) e 2.7 (0,6330 / 0,75).**
O gap diz que o eixo é capacidade ou arquitetura. O candidato imediato é
`base=32` (7.762.465 parâmetros, na tabela do Ruling 10a e **nunca testado**).
Em CPU custaria ~36 h; **em GPU, pelo fator 11,7× do Ruling 19, ~3 h.** Cabe na
VRAM? `base=24` reservou 4,74 de 5,64 GB — `base=32` provavelmente **não cabe**
com batch 8, e baixar o batch quebraria a comparabilidade (mesma armadilha do
`HANDOFF_P2_6.md` §7.6.2). **Medir o fit antes**, como foi feito nesta sessão.

Antes disso, porém, vale o Ruling 21: **2.1 pode estar medindo a coisa errada.**
A rodada 6 fecha o 2.6 com IoU 0,6478 enquanto o extrator clássico reprova o 2.6
com IoU 0,7153. Gastar horas perseguindo 0,85 de IoU pode ser otimizar um proxy
que já se mostrou desalinhado do objetivo. O Ruling 10c — inspeção amostra a
amostra — é mais barato e responde se 2.1 ainda faz sentido.

**Família B — independente da U-Net: 2.2-piso, 2.4, 2.5, 2.9 (e boa parte do 2.8).**
Nenhum treino vai mexer nesses. São OCR, calibração e extração de polilinha:

- **2.9 (cobertura 0,803 / 0,90)** tem estratificação por dpi já medida:
  dpi 100–149 = 0,878 · dpi 150–200 = 0,770 · **dpi 60–99 = 0,741**. O problema é
  OCR em baixa resolução. Caminho: pré-processamento/upscaling antes do tesseract.
- **2.4 (falso alarme 0,197 / < 0,05) e 2.5 (rejeições corretas 0,678 / 0,90)**
  são a lógica de rejeição por consistência. Ambos reprovam desde sempre e nunca
  tiveram hipótese própria registrada.
- **2.2-piso**: RMSE 1,49 px **passa** (alvo ≤ 2), o que reprova é o **p95 =
  6,70 px** (alvo ≤ 5). É problema de cauda, não de erro típico — e medido contra
  a máscara VERDADEIRA, ou seja, é do extrator de polilinha, não da segmentação.
- **2.8 (854 ms / 500 ms)**: o extrator clássico faz o mesmo percurso em **14 ms**
  (G3b.4). O tempo está no OCR/calibração. Atacar isso provavelmente ataca 2.9
  junto.

### 7.3 Higiene pendente

- Corrigir `test_unet_tamanho_declarado` para ler o checkpoint em uso (Armadilha 2).
- Decidir o que fazer com `requirements.txt` (Armadilha 6).
- Se alguém quiser o ζ pareado sobre a interseção dos dois conjuntos (Armadilha 3).

## 8. Ruling 10c — EXECUTADO

O `HANDOFF_P2_3.md:527` definiu o 10c como *"olhar amostra a amostra quais casos
de ζ erram mais (ex.: sistemas muito subamortecidos, picos estreitos)"*, com o
gatilho *"se qualquer uma delas rodar e ζ não fechar de novo"*. **ζ fechou**, então
o 10c foi executado com propósito mudado: responder à pergunta do Ruling 21 —
o IoU por amostra prediz o erro end-to-end por amostra?

Dado bruto: `reports/part2_10c_por_amostra.json` (300 registros; IoU, erro do
oráculo e do real por parâmetro, motivo de descarte e estratos de render).
Reproduz o teste exatamente: **188 aceitas**, os mesmos do §3.3.

### 8.1 Ruling 25 — o IoU não prediz o erro end-to-end, e a estimativa pontual é anticorrelacionada

Erro de ζ por quartil de IoU (119 amostras aceitas de 2ª ordem):

| Quartil de IoU | n | ζ real | oráculo | degradação |
|---|---|---|---|---|
| Q1 — 0,226–0,471 (pior máscara) | 30 | **1,60%** | 1,16% | +0,44 p.p. |
| Q2 — 0,471–0,633 | 30 | 2,31% | 1,00% | +1,30 |
| Q3 — 0,633–0,738 | 30 | 2,48% | 0,84% | +1,64 |
| Q4 — 0,738–0,926 (melhor máscara) | 30 | **4,22%** | 1,95% | **+2,26** |

Monótono nos quatro quartis, na direção **contrária** à esperada. Spearman
IoU × erro de ζ = **+0,192** (p=0,036).

**Ressalva que impede a leitura causal:** o erro do *oráculo* também sobe no Q4
(1,95% contra 0,84–1,16%). O oráculo não vê a imagem — recebe a série verdadeira.
Logo o Q4 concentra problemas **intrinsecamente mais difíceis de identificar**, e
parte do padrão é confundimento. Coerente com isso, a correlação da degradação
(real − oráculo) **não é significativa**: Spearman +0,105, p=0,255.

**Afirmação defensável:** não há evidência de que subir o IoU reduza o erro
end-to-end; a estimativa pontual aponta para o lado oposto. **NÃO afirmar** que
IoU alto causa erro alto.

Evidência independente e mais direta: as **112 descartadas têm IoU mediano 0,6583,
MAIOR que as 188 aceitas (0,6456)**. A qualidade da máscara não determina sequer
se a amostra é utilizável.

**Consequência prática: `base=32` sai da fila por evidência.** Gastar ~3 h de GPU
para subir IoU otimiza um proxy que duas medições independentes mostram
desalinhado do objetivo.

### 8.2 Ruling 26 — a hipótese original do 10c está errada

Degradação de ζ por faixa de ζ verdadeiro:

| ζ verdadeiro | n | degradação |
|---|---|---|
| [0,0–0,3) — os mais subamortecidos | 9 | **+0,49 p.p.** |
| [0,3–0,5) | 9 | **+3,32 p.p.** |
| [0,5–0,7) | 10 | +0,38 |
| [0,7–0,9) | 11 | −1,83 |
| [0,9–2,0) | 42 | +0,94 |

Os **mais subamortecidos estão entre os melhores** — o oposto do que o Bloco 3
supôs. A única faixa que reprovaria sozinha é ζ ∈ [0,3–0,5). **Com n de 9 a 11 por
faixa isto é subdimensionado**: serve para descartar a hipótese original, não para
eleger a faixa [0,3–0,5) como culpada. Quem quiser conclusão sobre faixas precisa
de um `data/test` maior ou estratificado por ζ.

### 8.3 Ruling 27 — as 112 descartadas: duas causas, nenhuma delas a U-Net

| Motivo | n |
|---|---|
| **Ordem identificada errada** | **51** |
| **OCR / calibração falhou** | **55** (28 `ocr_insuficiente` + 27 `calibration_failed`) |
| escala inválida / bbox | 4 |
| oráculo divergiu | 2 |

**103 dos 112 descartes vêm de duas causas que nenhum treino de segmentação
alcança.**

**Assimetria de ordem — 48 dos 51 são fopdt:**

| Ordem verdadeira | aceitas | ordem divergiu | OCR falhou |
|---|---|---|---|
| **fopdt** (n=149) | **46,3%** | **32,2%** | 20,8% |
| second (n=151) | **78,8%** | 2,0% | 18,5% |

O pipeline classifica sistemas de 1ª ordem como 2ª ordem de forma sistemática
(48 × 3 no sentido inverso). **Não é falta de dado**: as que divergiram têm *mais*
pontos extraídos (mediana 850) que as aceitas (702).

**Falha de OCR/calibração por dpi — é um U, não uma rampa:**

| dpi | falha do pipeline |
|---|---|
| 60–99 | 25,9% |
| 100–149 | **12,2%** |
| 150–200 | 23,0% |

Alta resolução falha quase tanto quanto baixa. **Isso desmonta "upscalar imagens
de baixo dpi" como conserto suficiente** — a faixa alta precisa de outra explicação.

Fundo escuro (definição do 2.7: `int(bg_color[:2],16) < 128`) quase não afeta
aceitação: 63,6% claro × 61,4% escuro — apesar de afetar bastante o IoU (2.7).
Mais um ponto de desacoplamento entre máscara e resultado.

**Teto de aproveitamento:** hoje 188/300 (62,7%); se OCR/calibração nunca
falhasse, **243/300 (81,0%)**.

### 8.4 Armadilha 7 — `bg_color` é hex arbitrário

`render.bg_color` é um hex aleatório por amostra (mais de 200 valores distintos em
300), **não** um rótulo `"white"`/`"black"`. Derivar "fundo escuro" por comparação
de string dá 299/300 e está errado. A definição correta é a do teste 2.7:
`int(bg_color.lstrip("#")[:2], 16) < 128`, que reproduz o n=173/127 esperado.

## 9. A confusão de ordem — investigada

Dado bruto: `reports/part2_10c_ordem_fopdt.json` (149 amostras fopdt de
`data/test`; AIC/SSE/NRMSE dos dois ajustes no oráculo e no pipeline, mais os
parâmetros do ajuste de 2ª ordem FORÇADO). 118 chegaram ao ajuste; 31 pararam
antes em OCR/calibração.

### 9.1 Ruling 28 — o AIC decide pelo `n` inflado da polilinha, não pelo ajuste

| | acerta a ordem em fopdt | n mediano da série |
|---|---|---|
| **Oráculo** (série verdadeira) | **94,1%** (111/118) | 512 |
| **Real** (pipeline) | **59,3%** (70/118) | **806** |

Mesmo estágio D, mesmo AIC, mesma seleção. **A única variável é a série.**

`identify()` escolhe por `AIC = n·log(SSE) + 2k` (k=3 fopdt, k=4 segunda), logo a
2ª ordem vence quando `SSE₁/SSE₂ > exp(2/n)`:

| | valor |
|---|---|
| limiar `exp(2/n)`, n=806 | **1,00234** → basta **0,234%** de ganho de SSE |
| ganho observado nas 48 erradas | **1,010%** (4× o limiar) |
| margem mediana sobre o limiar | +0,0077 |

Nas 70 que acertam, `SSE₁/SSE₂ = 1,0000` e `ΔAIC = +2,00` exatos — só a
penalidade do parâmetro extra, sem ganho nenhum de ajuste.

**Não é ganho real de modelo:** NRMSE mediano **0,00353 (fopdt) × 0,00351
(segunda)**. Empate técnico. A 2ª ordem vence por um fio que o `n` amplifica.

**Causa:** a polilinha tem ~800 pontos que **não são independentes** — pixels
vizinhos com erro de extração correlacionado. O polo extra absorve essa
correlação, ganha ~1% de SSE em cima de artefato, e o AIC trata isso como 800
evidências independentes.

**Trocar AIC por BIC NÃO resolve:** com penalidade `log(n)·k`, **29 das 48
(60,4%) ainda escolheriam 2ª ordem**. O problema não é a constante da penalidade,
é o `n` inflado.

### 9.2 Ruling 29 — 2ª ordem forçada recupera K e τ, e só custa θ

Ajuste de 2ª ordem imposto às 118 amostras fopdt, contra o ajuste FOPDT correto:

| Parâmetro | FOPDT | **2ª ordem forçada** | |
|---|---|---|---|
| **K** | 0,24% | **0,23%** | empate |
| **τ** (via polo lento) | 0,51% | **0,56%** | empate |
| **θ** | 0,25% | **1,01%** | **4× pior** |

**ζ atribuído foi ≥ 1 em 100% dos casos** (mediana 6,35, mín 2,61, máx 10,0 — o
máximo bate no teto do ajuste). A 2ª ordem **nunca** inventou um sistema
subamortecido a partir de um de 1ª ordem: caiu sempre no regime sobreamortecido,
como a teoria prevê (1ª ordem é o limite degenerado com um polo dominante).

τ sai do polo lento por `1/[ωₙ(ζ−√(ζ²−1))]` com **0,56%** de erro, contra 0,51%
do ajuste direto — recuperação praticamente perfeita.

**θ piora porque o polo rápido imita tempo morto.** Numa resposta sobreamortecida
parte do atraso é absorvida pelo segundo polo e a identificabilidade de θ cai. Em
valor absoluto 1,01% de NMAE/T_dom continua folgado dentro do orçamento de 3 p.p.

### 9.3 A decisão em aberto: adotar "sempre 2ª ordem"?

**A favor:** elimina 48 dos 112 descartes (não há ordem para errar); aproveitamento
iria de 188/300 (62,7%) para **~239/300 (79,7%)**, quase o teto de 81,0% que só o
conserto de OCR alcançaria (§8.3). Remove o ramo de seleção de modelo do estágio D.

**Contra:** o critério 2.6 é definido sobre `r["order"] == m["order"]`. Com saída
sempre `"second"`, nenhuma amostra fopdt casa e **o critério deixa de funcionar
como está escrito**. Isso não é mudança de pipeline, é **mudança de definição do
critério** — a comparação passaria a ser numa parametrização comum (K, τ
equivalente, θ). Legítimo, mas precisa ser declarado na monografia, não
introduzido de lado. Custa também θ 4× pior.

**Alternativa não medida:** manter a seleção e corrigir o `n` do AIC para o número
de pontos efetivamente independentes (subamostrar a polilinha ou estimar o
comprimento de correlação do resíduo). Ataca a causa do Ruling 28 sem tocar no
critério. **Não foi medida** — não assumir que funciona.

## 10. Ruling 30 — o caminho 2 resolveu: AIC com `n` efetivo

Implementado em `identify/classical.py`: `_rho1()`, `_n_efetivo()` e a nova
`identify()`. **Os campos `.aic` dos `FitResult` continuam sendo o AIC clássico**
e não mudaram — `tests/conftest.py:235` os reporta na Parte 1. O que mudou é só
a comparação dentro de `identify()`.

A 2ª ordem passa a vencer quando `n_eff·log(SSE₁/SSE₂) > 2`, com
`n_eff = n·(1−ρ)/(1+ρ)` e ρ a autocorrelação de defasagem 1 do resíduo da
estrutura **mais flexível** (2ª ordem — num modelo subespecificado a correlação
mistura ruído de extração com erro de estrutura e superestimaria a correção).

### 10.1 Escolha de ordem, medida nas 241 amostras que chegam ao ajuste

| Critério | fopdt | second | **total** |
|---|---|---|---|
| **AIC com `n` cru (antes)** | 59,3% | **97,6%** | **78,8%** |
| BIC com `n` cru | 75,4% | 94,3% | 85,1% |
| **`n_eff` (adotado)** | **89,8%** | 92,7% | **91,3%** |

Custa 4,9 p.p. em 2ª ordem para ganhar 30,5 p.p. em fopdt — troca 6 acertos por 36.

A varredura mostra que a fórmula principiada fica perto do teto do método:
`n_eff × 0,25` chega a 92,9%, apenas 1,6 p.p. acima. **Não vale introduzir um
fator ajustado à mão por isso** — o `n_eff` puro não tem parâmetro livre.

ρ medido: mediana **0,714** (resíduo de 2ª ordem) e 0,806 (resíduo de 1ª).
`n` mediano 738 contra `n_eff` mediano **112,5** — redução de **6,6×**.

### 10.2 Efeito nos critérios

| | sem `n_eff` | **com `n_eff`** |
|---|---|---|
| **2.6-aceitas** | 188/300 (62,7%) | **215/300 (71,7%)** |
| 2.6[K] | +0,25 p.p. | **+0,19** |
| 2.6[tau] | +0,54 | **+0,39** |
| 2.6[theta] | +0,27 | +0,27 |
| 2.6[wn] | +0,82 | **+0,73** |
| 2.6[zeta] | +1,05 | **+0,93** |
| **2.6 (pior parâmetro)** | +1,05 ✅ | **+0,93 ✅** |
| **2.6-clássico (pior)** | +4,36 ❌ | **+2,42 ✅** |

**Todos os cinco parâmetros melhoraram COM 27 amostras a mais na conta** — o
oposto do risco esperado (amostras novas costumam ser as difíceis).

**O diagnóstico do extrator clássico virou de ❌ para ✅.** A confusão de ordem
estava no estágio D, que é **compartilhado** pelos dois extratores; ela penalizava
os dois. Consequência para o Ruling 21: a vantagem da U-Net em ζ era
+1,05 × +4,36 (fator 4) e agora é **+0,93 × +2,42 (fator 2,6)**. A U-Net segue
ganhando com folga, mas **parte do que o §8.1 creditou a ela era ruído de seleção
de ordem**. O Ruling 25 (IoU não prediz erro end-to-end) NÃO é afetado — ele foi
medido dentro das aceitas, por quartil de IoU.

### 10.3 Sem regressão

- **Parte 1: 33 passed, exit 0.** A correção é **no-op em série verdadeira**:
  resíduo branco → ρ≈0 → `n_eff`≈n. Era o desenho, e se confirmou.
- **Parte 2: as mesmas 6 falhas** (2.1, 2.2-piso, 2.4/2.5, 2.7, 2.8). Nenhum
  critério novo quebrou.
- 2.8 foi de 854 para 891 ms. A correção acrescenta uma avaliação de
  `model_response` por chamada (desprezível) e o G3b.4, não tocado, oscilou
  14,2 → 12,9 ms na mesma rodada. **Provável ruído de medição, não afirmado.**

### 10.4 A opção "sempre 2ª ordem" fica ARQUIVADA

O §9.3 a descrevia como plausível mas cara: exigia redefinir o critério 2.6 fora
de `r["order"] == m["order"]` e declarar isso na monografia. **O caminho 2 entrega
o mesmo alvo sem tocar no critério** — 215/300 contra os ~239/300 que a opção 1
projetava, mas preservando a definição. Se alguém quiser os ~24 pontos restantes,
a opção 1 continua no §9.2/9.3 com os números medidos.

Relatório: `reports/part2_strata_rodada6_neff.md`. Dado bruto do ρ:
`reports/part2_10c_neff.json`.

### 10.5 Pendência: `reports/part1_metrics.md` precisa ser regenerado nesta máquina

Rodar `tests/test_part1.py` reescreve o relatório da Parte 1. A execução do §10.3
foi **parcial** (só `test_part1.py` + `test_leakage.py`), e o próprio relatório
insere um aviso disso — ele foi **restaurado via `git checkout`** e o registro
commitado, gerado na máquina original, está intacto.

O que a execução parcial mostrou, e que fica **sem atribuição**:

| | commitado (máquina original) | execução parcial (aqui, com `n_eff`) |
|---|---|---|
| RULING C, MAPE(K) mediana `w<3` | 1,784% | 1,809% |
| RULING C, MAPE(K) média `w<3` | 127,624% | 127,627% |
| 1.7 geração de 200 amostras | 14,15 s | 2,04 s |
| sha256 das 5 seeds | — | todos diferentes |

**Não dá para separar as causas com o que existe hoje**, porque não há linha de
base da Parte 1 nesta máquina *antes* do `n_eff`. Os hashes e o 1.7 são a máquina
(Rulings 11 e 16 do Bloco 6, já documentados); os deslocamentos do RULING C são
pequenos e podem vir de qualquer uma das duas causas — provavelmente de umas poucas
amostras que trocam de ordem também no oráculo.

**Todos os 33 testes passam nas duas configurações**, então nenhum critério da
Parte 1 mudou de veredito. Para citar números da Parte 1 na monografia,
regenerar com `.venv/bin/python -m pytest -q` **sem filtros** nesta máquina.

## 11. Investigação 2.8 / 2.9 — o estágio B

Dado bruto: `reports/part2_calib_perfil.json` (300 amostras; tempo de cada etapa
do estágio B, nº de chamadas ao tesseract, ticks detectados × pares lidos por
eixo e o `reason` final). Reproduz o 2.9 exatamente: **241/300 = 80,3% ok**.

### 11.1 Ruling 31 — 98,7% do estágio B é partida de processo do tesseract

| Etapa | mediana | % do total |
|---|---|---|
| conversão para cinza | 1,3 ms | 0,2% |
| `detect_plot_bbox` | 1,3 ms | 0,2% |
| `detect_tick_pixels` | 1,6 ms | 0,2% |
| **`read_tick_labels`** | **791,0 ms** | **98,9%** |
| ↳ dos quais tesseract | **789,7 ms** | **98,7%** |
| pós-OCR (RANSAC + consistência) | 0,3 ms | 0,0% |
| **total estágio B** | **799,9 ms** | |

Todo o resto soma **4,5 ms**. Chamadas ao tesseract por imagem: mediana **16**,
p95 **57**, máx **105**, a **51,2 ms** cada.

**Não é OCR, é spawn de processo.** Medido com recortes sintéticos:

| recorte | tempo | pixels |
|---|---|---|
| 1 rótulo (28×60) | 52,6 ms | 1.680 |
| 10 rótulos (28×600) | 55,0 ms | 16.800 |
| 100 rótulos (280×600) | 82,1 ms | 168.000 |

**~52 ms fixos por invocação, ~0,3 ms por rótulo adicional.** O `pytesseract`
grava arquivo temporário e faz spawn a cada `_ocr_number`.

**Conserto e ganho previsto:** empacotar todos os rótulos numa única invocação
leva 790 ms → **~55 ms**, e o 2.8 de 891 ms → **~156 ms**, dentro do alvo de
500 ms. `pytesseract.image_to_data` devolve TSV com caixas, o que permite mapear
cada leitura de volta ao recorte. Ganho secundário: `_ocr_number` é chamado **até
3× por blob** (recorte + 2 fallbacks), então parte das 16 chamadas são
retentativas. **Previsão, não medição** — o conserto não foi implementado.

### 11.2 Ruling 32 — o 2.9 é taxa de LEITURA, não detecção, e o gargalo é o eixo y

**A detecção de ticks funciona:** mediana de 10 ticks no x e 8 no y, e
**nenhuma das 299 amostras teve zero ticks detectados**. A hipótese anterior
("OCR em baixa resolução", §7.2 do handoff) errou o alvo.

**A taxa de leitura é 42,9% no x e 50,0% no y** — mais da metade dos rótulos
detectados não é lida.

Das 28 falhas por `ocr_insuficiente` (precisa ≥2 pares por eixo):

| | falta o eixo | mediana de pares lidos | zero pares |
|---|---|---|---|
| eixo x | 5 | 4,0 | 2/28 |
| **eixo y** | **24** | **1,0** | **11/28** |

**O eixo y concentra 24 das 28 perdas.**

As outras 27 falhas (`calibration_failed`) são caso distinto: leram 5–6 pares, o
RANSAC manteve 5, mas não ficam equiespaçados — **erro de leitura sobrevivendo ao
RANSAC**. Problema de acurácia, não de quantidade. Atacar volume de leitura não
conserta essas 27.

### 11.3 Hipótese aberta: as margens são constantes em pixels

`MARGIN_Y_W = 90` e `MARGIN_X_H = 40` são fixos, enquanto o dpi varia de 60 a 200.

| dpi | taxa de leitura, eixo y |
|---|---|
| 60–80 | 50,0% |
| 80–100 | **66,7%** |
| 100–125 | 50,0% |
| 125–150 | 38,9% |
| 150–175 | 34,5% |
| 175–201 | **33,3%** |

Spearman dpi × taxa de leitura no y = **−0,177 (p=0,0021)**: significativo mas
**fraco**. Coerente: dpi × ticks detectados no y = +0,268 (p<0,0001), mas
dpi × pares **lidos** = +0,035 (p=0,55) — em dpi alto detecta-se mais tick e
lê-se o mesmo número absoluto.

**NÃO afirmar que a faixa fixa é a causa.** A correlação é fraca e o corte de
rótulo não foi medido diretamente. O que resolveria, e é barato: medir a caixa de
cada blob do eixo y contra a largura da faixa, antes de mexer nas constantes.

### 11.4 Ordem recomendada

1. **Empacotar as chamadas ao tesseract** (Ruling 31). Ganho grande, previsão
   sólida, e é a única coisa que faz o 2.8 passar. Não mexe em acurácia.
2. **Medir o corte de rótulo no eixo y** (§11.3) antes de tocar em `MARGIN_Y_W`.
3. **As 27 de `calibration_failed`** são um problema separado — leitura errada
   passando pelo RANSAC. Precisa de hipótese própria; nenhuma foi registrada.

## 12. Ruling 33 — a faixa fixa do eixo y corta rótulos, mas NÃO é a causa do 2.9

Dado bruto: `reports/part2_eixoy_corte.json` (299 amostras; largura da faixa,
caixa de cada blob de texto, blobs encostados na borda, nos dois eixos).

A faixa do eixo y tem **82 px** úteis (`MARGIN_Y_W=90` − `TICK_GAP=8`):

| dpi | maior rótulo | ocupação | blob encostado na borda (Y) |
|---|---|---|---|
| 60–99 | 32 px | 39,0% | 3,5% |
| 100–149 | 42 px | 51,2% | 9,6% |
| 150–200 | 46 px | **56,1%** | **28,3%** |

dpi × ocupação da faixa: Spearman **+0,349 (p<0,0001)**. O corte é real e escala
com o dpi. E encostar tem custo medido:

| | n | taxa de leitura (Y) | calibração ok |
|---|---|---|---|
| sem blob encostado | 257 | **50,0%** | **82,9%** |
| com blob encostado | 42 | **33,3%** | **66,7%** |

**Mas a hipótese não explica o 2.9.** Só 22/299 (7,4%) têm rótulo ocupando a
faixa inteira, a ocupação mediana é 51% e a correlação com a taxa de leitura é
fraca (−0,129, p=0,026).

**O dado decisivo: mesmo SEM corte a taxa de leitura é 50%.** Metade dos rótulos
falha com espaço sobrando. Alargar `MARGIN_Y_W` (ou torná-lo proporcional ao dpi)
é um conserto barato que recupera ~42 amostras da faixa alta, **mas o problema
dominante do 2.9 não é geometria** — é o OCR errar metade das leituras em
condições folgadas. Hipótese do §11.3 fica **parcialmente confirmada e
despromovida a causa secundária**.

## 13. Ruling 34 — a Decisão E (§1.7 do PLANO) NÃO está implementada, e o 2.11 não a protege

O `PLANO.md §1.7` ("Decisão E — OCR opcional, não estrutural") especifica saída em
dois níveis: o **adimensional** (estrutura, ζ, ωₙ·T, θ/T, K/y_faixa) sai sempre,
e o **físico** é acrescentado só quando a calibração fecha. `PLANO.md:135`:
*"`Calibration.ok = False` degrada a saída, não a invalida"*. `PLANO.md:142` lista
**ζ como não dependente de OCR** — "adimensional, vem da forma da curva".

**O código faz o oposto.** `identify/pipeline.py:29`:

```python
cal = calibrate(image_rgb)
if not cal.ok:
    return {**vazio, "reason": cal.reason, ...}   # aborta a amostra inteira
```

`grep -rn "dimensionless\|adimension" identify/*.py` → **zero ocorrências**. O
bloco de saída adimensional não existe.

**E o teste que leva o nome do critério não testa o critério.**
`tests/part2/test_part2.py:263`, `test_2_11_saida_adimensional_sempre_presente`,
docstring *"a saída adimensional existe mesmo com ok=False (§1.7)"*, corpo:

```python
cal = calibrate(m["image"])
assert cal.ok in (True, False)
assert isinstance(cal.reason, str)
```

Verifica que `calibrate()` não levanta. Nada mais. Reporta ✅ 300/300.

Em favor dele: o nome que **registra** no relatório é honesto ("calibrate() nunca
levanta exceção"). O problema é ocupar o slot do 2.11, cuja definição em
`PLANO.md:299` é *"100% das amostras com `dimensionless` preenchido"*. **O 2.11
implementado é mais fraco que o 2.11 do plano, e a diferença é exatamente a
Decisão E.**

### 13.1 Consequência para a prioridade

As **55 amostras** perdidas em OCR/calibração são descartadas por inteiro; por
projeto deveriam entregar ζ, ωₙ·T e θ/T. **ζ é o critério que levou seis rodadas
de treino para fechar** (§3.3, §10.2).

Reenquadramento do 2.9: o alvo de cobertura ≥90% governa a saída **física**. Com
a Decisão E implementada, calibração falhando custaria as unidades físicas, não a
amostra — as 55 voltariam para a conta adimensional e os 18,3% de perda deixariam
de ser perda total.

**Isto NÃO substitui o conserto do 2.9**, que continua sendo um critério do plano.
Muda o custo de ele falhar.

### 13.2 Ordem revista

1. **Implementar a Decisão E** e endurecer o 2.11 para o que o plano pede. Maior
   retorno da fila: destrava 55 amostras para o parâmetro mais difícil do projeto
   e fecha uma divergência plano × código guardada por um teste que não guarda.
2. **Empacotar as chamadas ao tesseract** (Ruling 31) — único conserto que faz o
   2.8 passar; previsão de 891 ms → ~156 ms.
3. **`MARGIN_Y_W` proporcional ao dpi** (Ruling 33) — barato, recupera ~42
   amostras, mas é causa secundária.
4. **A taxa de leitura de 50% em condições folgadas** (Ruling 33) é o problema
   real do 2.9 e **não tem hipótese registrada**. Junto com as 27 de
   `calibration_failed` (Ruling 32), é o que sobra de genuinamente aberto no
   estágio B.

## 14. Ruling 35 — OCR em lote: o critério 2.8 FECHOU

Implementado em `identify/calibrate.py`: `_texto_para_numero()`,
`_ocr_numeros_lote()` e `read_tick_labels()` reescrito. **A precedência dos
recortes é idêntica à anterior** — o que mudou é que os candidatos de todos os
blobs dos dois eixos vão para **uma única invocação** do tesseract, num mosaico
horizontal, e a precedência é resolvida depois.

O mosaico é uma linha só de propósito: preserva o `--psm 7` do `_OCR_CFG`, então
o tesseract continua vendo "uma linha de texto". O mapeamento de volta usa as
caixas do `image_to_data`, atribuindo cada palavra ao recorte cuja faixa
horizontal contém o centro dela.

### 14.1 A folga do mosaico é o parâmetro crítico, e foi varrida

Primeira tentativa com `_LOTE_GAP = 60` **degradou muito**: calibração ok caiu de
241/299 (80,6%) para 192/299 (64,2%) e `ocr_insuficiente` explodiu de 28 para 80.
Causa: folga pequena funde dois rótulos numa palavra só e o `_NUM_RE` rejeita
ambos. Varredura em 99 amostras, contra a implementação anterior como referência:

| config | pares | calibração ok | ms/amostra |
|---|---|---|---|
| **ANTIGO (referência)** | 882 | **77/99 (77,8%)** | 1020,3 |
| lote único gap=200 | 867 | 75/99 (75,8%) | 93,0 |
| **lote único gap=400** | **912** | **77/99 (77,8%)** | **152,5** |
| lote único gap=700 | 868 | 74/99 (74,7%) | 362,1 |
| cascata gap=200/400/700 | ~850 | 72–73/99 | 129–281 |

`gap=400` **empata** com a referência lendo mais pares. A folga tem ótimo
intermediário: 200 funde, 700 espalha e o `--psm 7` perde a linha.

**A variante em cascata perdeu** (72–73/99): fazer 3 lotes menores (primários,
depois fallbacks) dá menos contexto ao `psm 7` do que um mosaico único. Registrado
porque era a hipótese mais "óbvia" e está refutada.

### 14.2 Validação nas 300, contra a implementação anterior

| | ANTIGO | **EM LOTE (gap=400)** |
|---|---|---|
| tempo mediano de `read_tick_labels` | 797,8 ms | **98,3 ms** (8,1×) |
| tempo p95 | 2.939,5 ms | **170,2 ms** (17,3×) |
| pares lidos | 2.599 | **2.598** |
| calibração ok | 241/299 (80,6%) | **239/299 (79,9%)** |

−0,7 p.p. = **2 amostras** em 299, dentro do ruído.

**Ressalva: NÃO é superconjunto.** 24 amostras viraram falha e 22 viraram
sucesso, saldo −2. O conjunto é diferente, não estritamente melhor.

**Limitação da validação:** a acurácia foi aferida pelo **veredito de calibração
de ponta a ponta** (que embute `_equiespacados`, sensível a leitura errada), não
por comparação rótulo a rótulo contra a verdade. Uma tentativa de checar cada
valor lido contra o `axis_affine` do meta foi feita e **descartada** por erro na
convenção do affine — dava 0% de acerto até para a implementação antiga, que
sabidamente calibra 80,6%. Quem quiser acurácia por rótulo precisa acertar essa
convenção primeiro.

### 14.3 Efeito nos critérios — suíte completa, `pytest -q` sem filtros

**59 testes em 9min36s** (a Parte 2 sozinha levava ~37 min). **5 failed, 54
passed** — era 6 failed.

| Critério | ANTES | AGORA | |
|---|---|---|---|
| **2.8 latência** | 891 ms · p95 2.325 ❌ | **172 ms · p95 254 ✅** | **FECHOU** |
| 2.3 erro de escala | 0,950 (n=241) | **0,958** (n=239) | ✅ |
| 2.5 rejeições corretas | 0,678 (n=59) | **0,721** (n=61) | ❌ melhorou |
| 2.9 cobertura | 0,803 | 0,797 | ❌ −0,6 p.p. |
| 2.9[dpi 60–99] | 0,741 | **0,788** | melhorou |
| 2.9[dpi 100–149] | 0,878 | 0,826 | piorou |
| 2.4 falso alarme | 0,197 | 0,203 | ❌ −0,6 p.p. |
| **2.6 (ζ)** | +0,93 (n=215) | **+0,99** (n=214) | ✅ |
| **2.6-clássico** | +2,42 (n=179) | **+1,62** (n=183) | ✅ melhorou |

Os movimentos de ±0,6 p.p. são o mesmo *churn* 24↔22 do §14.2. **Parte 1 e
vazamento: zero falha.**

### 14.4 Pendência §10.5 RESOLVIDA

A suíte rodou **sem filtros**, então `reports/part1_metrics.md` foi regenerado por
completo nesta máquina e o aviso de "relatório parcial" desapareceu. Contra o
registro commitado (máquina original), o RULING C moveu só na 3ª/4ª casa
(127,624% → 127,627%; 0,914% → 0,907%) e os sha256 das 5 seeds diferem — este
último já explicado pelo Ruling 11 do Bloco 6 (codificador PNG, não os pixels).

## 15. Situação dos critérios ao fim desta sessão

| Critério | início da sessão | fim |
|---|---|---|
| **2.6 (ζ)** | +3,65 p.p. ❌ | **+0,99 p.p. ✅** |
| **2.8 latência** | 885 ms ❌ | **172 ms ✅** |
| 2.6-clássico | +4,36 ❌ | **+1,62 ✅** |
| 2.1 IoU | 0,6205 ❌ | 0,6478 ❌ |
| 2.7 IoU estrato | 0,5856 ❌ | 0,6330 ❌ |
| 2.9 cobertura | 0,803 ❌ | 0,797 ❌ |
| 2.4 / 2.5 | 0,197 / 0,678 ❌ | 0,203 / 0,721 ❌ |
| 2.2-piso p95 | 6,70 px ❌ | 6,70 px ❌ |
| **falhas na suíte** | **7** | **5** |

O que resta aberto, com o estado de conhecimento de cada um:

1. **2.9 / 2.4 / 2.5** — a taxa de leitura de 50% em condições folgadas
   (Ruling 33) e as ~30 de `calibration_failed` (Ruling 32). **Sem hipótese.**
2. **2.1 / 2.7** — capacidade da rede, **mas o Ruling 25 questiona a meta** e o
   Ruling 30 mostrou que parte da vantagem atribuída à U-Net era ruído de ordem.
   `base=32` segue desaconselhado.
3. **2.2-piso** — cauda do p95 (o RMSE passa), extrator de polilinha. Nunca
   investigado.
4. **Decisão E (Ruling 34)** — não implementada; destravaria 55 amostras para ζ.

## 16. Ruling 36 — CORREÇÃO do Ruling 33: o OCR é ~92% preciso, não 50%

**O Ruling 33 (§12) está errado na parte da "taxa de leitura de 50%".** O
denominador usado era a saída de `detect_tick_pixels`, que conta ticks MENORES e
que o `read_tick_labels` **explicitamente não consulta** (a docstring dele diz
isso). Denominador certo: os ticks **realmente rotulados**, que o meta traz em
`ticks[eixo] = [[px, valor], ...]`.

Dado bruto: `reports/part2_item1_rotulos.json` (299 amostras).

| | eixo x | eixo y |
|---|---|---|
| rótulos que existem | 1.433 (med 5) | 1.380 (med 5) |
| blobs varridos | 2.911 (med 7) | 2.477 (med 5) |
| lidos | 1.394 (med 4) | 1.204 (med 4) |
| **leituras corretas** | **91,9%** | **92,7%** |
| valor errado | 6,7% (93) | 4,7% (57) |
| espúrias | 1,4% (20) | 2,6% (31) |
| **recall dos rótulos** | **89,4%** | **80,9%** |

**Esta crença não nasceu no Bloco 7:** a docstring de `_equiespacados` já afirma
*"medido: ler menos da metade é comum"*, vinda do Bloco 2. A medição acima a
corrige para a implementação atual. **A parte geométrica do Ruling 33 (corte de
rótulo pela faixa fixa, dpi × ocupação +0,349) continua válida** — o que cai é a
conclusão de que a leitura falha em metade dos casos.

**A varredura larga de blobs NÃO é problema.** Os blobs excedem os rótulos
verdadeiros em até +16 (p90) — legenda, título, anotações, distratores entram —
mas as leituras espúrias ficam em 1,4–2,6%, porque o `_NUM_RE` descarta texto não
numérico. O desenho está correto nisso.

**Convenção do `axis_affine`, para quem for medir acurácia:** `valor = sx*px + ox`
(verificado: 0,011131·290,13 − 3,2296 = 0,0). A tentativa descartada no §14.2
usava `(px − ox)/sx` e por isso dava 0% de acerto.

## 17. Ruling 37 — o item 1 tem hipótese: podar o par ofensor, com teto medido

O gargalo do 2.9 são as **150 leituras com valor errado**. Uma só basta para o
`_equiespacados` reprovar a amostra INTEIRA, em vez de descartar o par ruim.

Simulação da poda (remover o par cuja saída restaura o equiespaçamento, reajustando
o afim sobre o conjunto podado), 299 amostras:

| config | cobertura (2.9) | erro mediano de escala | **2.3 (< 1% em ≥ 95%)** |
|---|---|---|---|
| **ATUAL** | 239/299 (**79,9%**) | 0,055% | **95,8% ✅** |
| poda 1 par, piso 3 | 253/299 (84,6%) | 0,056% | 94,5% ❌ |
| poda ≤2, piso 3 | 255/299 (**85,3%**) | 0,056% | 94,1% ❌ |
| **poda ≤2, piso 4** | 248/299 (**82,9%**) | 0,056% | **95,2% ✅** |
| poda ≤3, piso 3 | 255/299 (85,3%) | 0,056% | 94,1% ❌ |

A simulação **reproduz o 2.3 exatamente** (95,8% contra os 0,958 com n=239 que a
suíte reportou), então a régua é confiável.

**Há trade-off real entre 2.9 e 2.3.** As amostras recuperadas pela poda são
justamente as de leitura duvidosa, e puxam o 2.3 para baixo. **Só `poda ≤2,
piso 4` mantém o 2.3 aprovado**, e por margem estreita (95,2% × 95,0% exigidos).

**O `piso` não é cosmético:** com menos de 3 pares o `_equiespacados` retorna
`True` vacuamente (`if len(pares) < 3: return True`). Podar sem piso infla a
cobertura aceitando calibração não verificada.

### 17.1 O teto, decomposto

| motivo da falha | n |
|---|---|
| `calibration_failed` | **30** |
| `ocr_insuficiente` | 25 |
| `sinal_de_escala_invalido` | 5 |

Recuperar **todas** as 30 de `calibration_failed` daria (239+30)/299 = **90,0%**,
exatamente o alvo do 2.9. **A poda recupera 16 das 30.** Consertar a consistência
é necessário e quase suficiente — mas não fecha sozinho. Os outros 14 exigem
reduzir os misreads na origem, e as 25 de `ocr_insuficiente` são um terceiro
problema (recall, não precisão — o eixo y tem 80,9%).

### 17.2 Não implementado

A poda **não foi aplicada** ao código. É decisão de projeto: troca +3,0 p.p. de
2.9 por −0,6 p.p. de 2.3, com o 2.3 ficando a 0,2 p.p. do limiar. Nenhum dos dois
critérios passa a ser aprovado com a mudança (2.9 iria a 82,9%, alvo 90%), então
o ganho é de posição, não de veredito.

## 18. A rede de IoU 0,65 NÃO é o gargalo — três Rulings

Dado bruto: `reports/part2_rede_metricas.json` (300 amostras; IoU, Dice,
espessura predita e verdadeira, RMSE da polilinha contra a curva analítica para
máscara verdadeira E predita, distância de centerline e o end-to-end com o
código atual). 214 aceitas, consistente com o §14.3.

### 18.1 Ruling 38 — Dice não pode ajudar: é identidade com IoU

```
max |Dice − 2·IoU/(1+IoU)| = 1,11e-16
```

Dice **é** `2·IoU/(1+IoU)`, transformação estritamente monótona. Spearman é
baseado em ranks, logo as duas dão correlação **idêntica** (+0,114, p=0,234) com
o erro end-to-end. Dice mede 0,7862 contra IoU 0,6478 e **parece** melhor, mas é
o mesmo número reparametrizado.

**Registrado porque é a armadilha natural:** trocar de métrica de área não escapa
do problema de área. Qualquer métrica de sobreposição sofre igual numa estrutura
fina.

### 18.2 Ruling 39 — o IoU aqui mede ESPESSURA DE TRAÇO, não acurácia geométrica

| | Spearman vs IoU |
|---|---|
| **espessura verdadeira do traço (w)** | **+0,860** (p=5e-89) |
| **razão de espessura pred/true (k)** | **−0,879** (p=5e-98) |
| deslocamento de centerline | +0,284 (p=6e-07) |

Estratificado por espessura verdadeira:

| w | n | IoU | **centerline** |
|---|---|---|---|
| < 4 px | 98 | **0,468** | **1,00 px** |
| 4–6 px | 60 | 0,609 | **1,00 px** |
| 6–9 px | 54 | 0,710 | **1,00 px** |
| ≥ 9 px | 88 | **0,782** | 1,08 px |

**O erro geométrico é 1,00 px em TODAS as faixas — constante — e o IoU varia
0,31.** Linha fina × grossa: IoU 0,548 × 0,727, centerline 1,00 px nas duas.
Espessura predita 6,30 px contra 5,77 verdadeira (razão 1,076 mediana, p90 1,409).

Um modelo geométrico trivial (deslocamento `d` numa faixa de largura `w`, com
espessura `k·w`) prevê IoU **0,667** contra 0,648 medido, com correlação
previsto × medido de **+0,932**. O IoU está ~93% explicado por espessura e
deslocamento — e o deslocamento não varia.

**Consequência para o critério 2.1:** o alvo IoU ≥ 0,85 é provavelmente
**inalcançável por construção** para traço fino. 168 das 300 amostras têm linha
de ~4,11 px; com 1 px de centerline, o teto aritmético fica na casa de 0,55. **A
meta foi fixada sem considerar a largura da linha.** Isto fundamenta com número o
que o Ruling 25 só suspeitava.

### 18.3 Ruling 40 — a contribuição geométrica da rede é sub-pixel, e nada na máscara prevê o end-to-end

Convenção do próprio critério 2.2 (RMSE em px contra a curva analítica do meta):

| | RMSE mediano | p95 |
|---|---|---|
| **piso** — polilinha da máscara VERDADEIRA | 1,488 px | 6,701 |
| **rede** — polilinha da máscara PREDITA | 1,924 px | 9,422 |
| **adicionado pela rede** | **+0,359 px** | +2,109 |

A rede acrescenta **um terço de pixel** sobre um piso de 1,49 px que pertence ao
extrator de polilinha. Cobertura de colunas 99,9%, 630 pontos nas duas — sem
lacunas. Centerline predita × verdadeira: mediana 1,00 px, RMSE 1,79 px,
p95 3,52 px. A polilinha é float (centroide), então a métrica tem resolução
sub-pixel — o 1,00 px não é artefato de quantização.

Correlação com a **degradação** de ζ (real − oráculo, que isola a contribuição da
visão), n=110:

| métrica | Spearman | p |
|---|---|---|
| IoU / Dice | +0,001 | 0,995 |
| razão de espessura | −0,025 | 0,792 |
| RMSE da polilinha (rede) | +0,045 | 0,642 |
| **erro adicionado pela rede** | **+0,007** | **0,946** |
| centerline RMSE / p95 / max | +0,033 / +0,034 / +0,054 | > 0,72 |

Todas indistinguíveis de zero. **Ressalva de potência: com n=110 o Spearman só
resolve |r| > ~0,19.** Leitura correta: "o efeito, se existe, é pequeno" — NÃO
"é exatamente zero".

### 18.4 Ruling 41 — a falha do critério 2.2 é 100% do extrator, 0% da rede

O 2.2 exige RMSE ≤ 2 px **e** p95 ≤ 5 px. Usando a máscara **VERDADEIRA**:
RMSE 1,488 (passa) e **p95 6,701 (REPROVA)**. O critério já está reprovado antes
de a rede contribuir. Com a máscara predita vai a 9,422 — a rede piora a cauda,
mas não é a causa.

Isto fecha a atribuição do §15: **2.2 nunca foi problema de segmentação.**

### 18.5 Veredito

`base=32` está descartado por três medições independentes: o IoU não prevê o
end-to-end (Ruling 25), o IoU mede espessura e não acurácia (Ruling 39), e a rede
contribui 0,36 px sobre um piso de 1,49 px que não é dela (Ruling 40). **Treinar
mais capacidade otimizaria uma métrica que os dados mostram desacoplada do
objetivo, para reduzir um termo que já é o menor da cadeia.**

O que 2.1 e 2.7 exigem, se forem mantidos como estão, é revisão da META — e o
Ruling 39 dá a base quantitativa para propor isso na monografia: uma métrica
geométrica (centerline, ou RMSE de polilinha) mede o que o pipeline consome;
IoU de máscara mede a largura do traço que o matplotlib sorteou.

## 19. Ruling 42 — ablação: o extrator de polilinha domina a perda de ACURÁCIA, a calibração domina a perda de AMOSTRA

Dado bruto: `reports/part2_ablacao.json`. Quatro cadeias, trocando um elo por sua
versão perfeita de cada vez:

| | cadeia |
|---|---|
| **E0** | série VERDADEIRA → estágio D (oráculo) |
| **E1** | máscara VERDADEIRA → polilinha → afim VERDADEIRO → D |
| **E2** | máscara PREDITA → polilinha → afim VERDADEIRO → D |
| **E3** | máscara PREDITA → polilinha → afim do OCR → D (= pipeline real) |

### 19.1 Acurácia: teste PAREADO, não diferença de medianas

Wilcoxon pareado por amostra, conjunto comum n=207 (delta em p.p.):

| elo | K | tau | wn | **zeta** | theta |
|---|---|---|---|---|---|
| **extrator de polilinha** | **+0,074**✱ | **+0,135**✱ | **+0,194**✱ | **+0,368**✱ | **+0,065**✱ |
| rede (U-Net) | +0,003 | +0,034 | −0,041 | +0,127 (p=0,056) | **+0,050**✱ |
| calibração / OCR | **+0,059**✱ | +0,000 | +0,200 | +0,120 | **+0,111**✱ |

✱ = p < 0,05.

**O extrator é o único elo que degrada os CINCO parâmetros significativamente**, e
lidera no que decide: **+0,368 p.p. em ζ (p=0,002)**. A rede só tem efeito
significativo em θ (+0,050 p.p.). A calibração pesa em K e θ.

**ARMADILHA METODOLÓGICA, registrada porque quase produziu conclusão errada:** a
diferença de MEDIANAS por estágio sugere que a rede *melhora* ζ (−0,129 p.p.),
porque cada estágio tem subconjunto de convergência diferente. O teste PAREADO
refuta: +0,127 p.p., ou seja piora levemente. **Nunca decompor esta cadeia por
mediana de estágio; usar sempre delta pareado por amostra.**

Erro absoluto de ζ ao longo da cadeia (conjunto comum): E0 1,380% → E1 2,019%
→ E2 1,890% → E3 2,239%.

### 19.2 Perda de amostra

| estágio | converge com ordem certa | derruba | recupera | saldo |
|---|---|---|---|---|
| E0 oráculo | 287/300 (95,7%) | — | — | — |
| E1 + extrator | 277 (92,3%) | 18 | 8 | **−10** |
| E2 + rede | 275 (91,7%) | 13 | 11 | **−2** |
| **E3 + calibração** | **219 (73,0%)** | **58** | 2 | **−56** |

**A calibração responde por 56 das ~68 amostras perdidas. A rede custa 2.**

### 19.3 O piso do estágio D

O E0 perde **13/300** e erra **1,380% em ζ com a série VERDADEIRA na mão**. É o
piso irredutível do ajustador atual. O critério 2.6 mede *degradação*
(real − oráculo), então este piso não o afeta — mas limita qualquer critério de
erro ABSOLUTO que a Parte 3 venha a definir.

### 19.4 Prioridade resultante

1. **Extrator de polilinha.** Maior contribuinte de erro em todos os parâmetros
   E a causa única da reprovação do 2.2 (p95 6,701 px na máscara VERDADEIRA,
   Ruling 41). Dois critérios, uma causa, e **nunca investigado**. O RMSE mediano
   do piso (1,488 px) PASSA o alvo de 2 px — quem reprova é a cauda, logo é um
   subconjunto onde a extração quebra, não erro típico. Candidatos pelos estratos
   já medidos do 2.2: traço `:` (1,80 px), com marcador (1,91 × 1,33 sem), traço
   grosso (1,76 × 1,15 fino). **A cauda em si não está medida** — é onde começar.
2. **Calibração.** Maior alvo para perda de amostra. Dois consertos já medidos:
   a **Decisão E** (Ruling 34) recupera as 56 para ζ, que é adimensional e não
   precisa de calibração; a **poda** (Ruling 37) recupera 16 das 30
   `calibration_failed`.
3. **Rede: nada a ganhar.** Terceira medição independente concordando
   (Rulings 25, 39/40, 42).

## 20. Ruling 43 — a cauda do 2.2 é curva ÍNGREME, e é limite da representação, não bug

Dado bruto: `reports/part2_cauda_polilinha.json`.

A cauda é **27/300 (9,0%)** das amostras com RMSE > 5 px. **Sem ela o p95 seria
4,046 px e o 2.2 passaria.**

### 20.1 As três hipóteses do §15/§19.4 estão REFUTADAS

Fração de cada estrato que cai na cauda:

| estrato | na cauda |
|---|---|
| traço `:` (que o §19.4 apontou como pior) | **7,5%** — o MENOR de todos |
| traço `-.` | 11,9% |
| com marcador × sem | 9,3% × 8,9% (sem sinal) |
| linha grossa × fina | 11,4% × 7,1% (fraco) |
| **dpi 150–200 × 60–99** | **14,0% × 2,4%** |

As diferenças de RMSE MEDIANO que o §19.4 citou (traço `:` 1,80 px, marcador
1,91 px) são reais mas **não se traduzem em pertencer à cauda**. Lição: estrato
que move a mediana não é o mesmo que estrato que produz a cauda.

### 20.2 A causa: inclinação da curva em px/px

| | Spearman vs RMSE da polilinha |
|---|---|
| **inclinação máxima (p99, px/px)** | **+0,868** (p=9e-93) |
| tinta por coluna | +0,763 |
| largura da linha (pontos) | +0,287 |
| dpi | +0,178 |

| inclinação máx | n | RMSE | na cauda |
|---|---|---|---|
| 0,49–2,85 | 75 | 0,62 px | **0,0%** |
| 2,85–6,45 | 75 | 1,00 px | **0,0%** |
| 6,45–13,79 | 75 | 1,77 px | 2,7% |
| 13,79–26,55 | 45 | 3,37 px | 8,9% |
| **26,55–174,94** | 30 | **6,67 px** | **70,0%** |

**Nenhuma amostra com inclinação < 6,45 px/px está na cauda; 70% das acima de
26,55 estão.**

**Mecanismo:** `mask_to_polyline` representa a curva como **um y por coluna**.
Onde `|dy/dx|` é grande, a coluna abrange dezenas de px em y e nenhum valor único
a representa — o erro fica em torno de metade da extensão vertical. **É limite da
REPRESENTAÇÃO, não defeito de implementação.**

Evidência corroborante: o perfil do erro na cauda é **uniforme** ao longo da
janela (6,83 / 7,05 / 7,23 / 7,08 / 7,14 px por quinto), inclusive no trecho já
assentado; o viés é 0,364 px num RMSE de 7 (5%), logo não é deslocamento
sistemático; e a fração de colunas com >1 ramo de tinta é 0,464 na cauda contra
0,024 no resto.

**Correção a uma métrica do Ruling 39:** o que foi chamado de "espessura do
traço" é **tinta por coluna**, e ela é explicada mais pela inclinação (+0,727)
que pela largura de linha (+0,471). Para curva suave as duas coincidem; na cauda,
não. O Ruling 39 segue válido no que afirma (IoU acompanha tinta por coluna e não
acurácia), mas a interpretação "largura do traço" é imprecisa.

### 20.3 A inclinação NÃO vem dos parâmetros do sistema

| | Spearman vs inclinação px/px |
|---|---|
| ζ | −0,163 (p=0,046, marginal) |
| ωₙ | −0,025 (n.s.) |
| τ | +0,004 (n.s.) |
| **inclinação em unidades de dado (normalizada)** | **+0,762** |
| **aspecto do quadro (w/h)** | **−0,600** |
| razão das escalas \|sx/sy\| | +0,123 |

ωₙ e τ absolutos não importam porque a `t_window` é escalada à dinâmica. O que
importa é **quanto da janela o transiente ocupa** (grandeza normalizada) e o
**aspecto do quadro** — quadro largo achata a curva em pixels, quadro alto a
verticaliza. **A cauda é estrato de RENDER, não de dinâmica.** Isso também
explica por que o Ruling 26 não achou padrão limpo por faixa de ζ.

### 20.4 Direção de conserto (não implementada)

O gargalo é a representação "um y por coluna". Candidatos, em ordem de custo:

1. **Descartar/despriorizar colunas íngremes no ajuste.** Uma coluna com
   `|dy/dx|` alto carrega pouca informação sobre `y(x)` e muito ruído. Barato,
   não muda a representação.
2. **Emitir mais de um ponto por coluna íngreme**, ou parametrizar por
   comprimento de arco em vez de por x (traçado de esqueleto).
3. **Reamostrar por aspecto** antes de extrair, normalizando `|dy/dx|`.

Nenhuma foi medida. O item 1 é o mais barato e ataca também a degradação de
acurácia do Ruling 42, que é o mesmo componente.

## 21. Ruling 44 — a Decisão E é implementável: recupera 53 das 61, com ζ a 2,93%

Dado bruto: `reports/part2_decisaoE.json`. Testa se ζ é recuperável SEM
calibração, ajustando num quadro normalizado (`t` em [0,1]; `y` com zero estimado
pela mediana dos 8% de colunas iniciais e sinal invertido, pois o pixel cresce
para baixo; escala de `y` arbitrária, pois ζ é invariante a ela). Corresponde ao
nível adimensional do `PLANO §1.7`.

### 21.1 Recuperação

| | n |
|---|---|
| calibração falhou | **61** |
| ajuste adimensional converge | **61 (100%)** |
| com a **ordem certa** | **53** |
| destas, de 2ª ordem (têm ζ) | **31** |

Por motivo: `calibration_failed` 26/30 (86,7%) · `ocr_insuficiente` 21/25 (84,0%)
· `sinal_de_escala_invalido` 5/5 · `bbox_not_found` 1/1.

### 21.2 Acurácia

| caminho | n | MAPE de ζ, mediana | p95 |
|---|---|---|---|
| físico (calibrado) | 110 | **2,40%** | 43,82% |
| adimensional, cal OK | 105 | 3,27% | 62,02% |
| **adimensional, cal FALHOU** | **31** | **2,93%** | **16,33%** |

O ζ recuperado erra 2,93% — equivalente ao caminho físico nas amostras que
funcionam (2,40%). **Não é resposta degradada; é resposta de qualidade
comparável em amostras hoje descartadas.**

E confirma a hierarquia do plano: onde a calibração fecha, o físico é melhor
(2,40% × 3,27%). O adimensional entra só quando o físico não existe.

### 21.3 RESSALVA que impede tratar os dois caminhos como equivalentes

Controle nas 104 amostras onde dá para comparar os dois:

| | valor |
|---|---|
| \|ζ_adim − ζ_fís\| mediana | 0,0205 |
| erro relativo entre os caminhos, mediana | **1,33%** |
| erro relativo, **p95** | **50,5%** |

Concordam no caso típico e **divergem muito na cauda**. O ponto fraco é o **zero
estimado**: erra quando há tempo morto grande ou ruído no trecho inicial. Logo o
nível adimensional precisa ser reportado como **saída própria, com acurácia
própria** — nunca fundido nos números do nível físico. Isso é exatamente a
estrutura de dois níveis do `PLANO §1.7`, e a ressalva aqui é a razão de ser dela.

### 21.4 O que falta para implementar

1. `identify_from_image` não pode mais retornar `{**vazio}` em `cal.ok == False`:
   precisa devolver o bloco `dimensionless` (ordem, ζ, ωₙ·T, θ/T, K/y_faixa) e
   deixar `physical` nulo.
2. Endurecer `test_2_11` para o que o `PLANO.md:299` pede — hoje ele só verifica
   que `calibrate()` não levanta (Ruling 34).
3. Decidir como o critério 2.6 trata o nível adimensional: ζ entra direto;
   ωₙ·T e θ/T exigem valores de referência derivados do meta (a `t_window` está
   lá); K e τ **não** entram sem calibração. Isso é decisão de projeto, não de
   implementação.

Nada disso foi implementado nesta sessão.

## 22. Resumo das duas investigações do §19.4

| alvo | investigado | conserto | medido |
|---|---|---|---|
| **extrator** (acurácia + 2.2) | Ruling 43 | descartar colunas íngremes / reparametrizar | **não** |
| **calibração — Decisão E** (perda de amostra) | Ruling 44 | bloco adimensional | **sim: 53 recuperadas, ζ a 2,93%** |
| **calibração — poda** (2.9) | Ruling 37 | poda ≤2 com piso 4 | **sim: 2.9 79,9% -> 82,9%, 2.3 cai 0,6 p.p.** |

A Decisão E é o único dos três com ganho medido e sem contrapartida em outro
critério. A poda troca 2.9 por 2.3. O extrator é o maior alvo mas o conserto
ainda não tem número.

## 23. Ruling 45 — Decisão E IMPLEMENTADA, a 5 ms de custo

`identify/pipeline.py` reescrito e `test_2_11` endurecido. O contrato do
`PLANO §1.7` (linhas 150–161) passa a valer.

### 23.1 O que mudou

`identify_from_image` não aborta mais em `cal.ok == False`. Agora sempre calcula
máscara e polilinha, e devolve os blocos do plano:

```
{"order", "params",                     # nível FÍSICO (compatibilidade)
 "dimensionless": {zeta, wn_T, tau_T, theta_T, theta_tau, K_yrange},
 "physical": {...} | None,
 "calibration": {ok, reason, n_pairs_x, n_pairs_y},
 "ok", "reason", "latency_ms", "n_points"}
```

Duas decisões de compatibilidade, deliberadas e documentadas no docstring:

1. **`params` e `order` no topo continuam sendo os do nível FÍSICO**, porque
   `tests/part2` e a Parte 3 os consomem. Os campos do plano entram ao lado, sem
   substituir.
2. **`ok` continua significando "há saída física"**, não "há resposta". É o que o
   próprio §1.7 pede em "Consequência nos critérios": 2.3, 2.4 e 2.5 passam a ser
   medidos sobre o subconjunto em que a calibração declarou sucesso. Quem quer o
   nível adimensional lê `dimensionless`, que nunca é nulo.

**Decisão técnica que resolve a ressalva do §21.3:** quando a calibração fecha, o
bloco adimensional é **derivado do ajuste físico** (ζ direto, `wn_T = wn·T`,
`theta_T = θ/T`, `K_yrange = K/faixa_y`), não de um segundo ajuste. Economiza um
ajuste por imagem **e** faz os dois níveis concordarem por construção — a
divergência de 50,5 % no p95 que o §21.3 mediu entre ajustes independentes deixa
de existir. O ajuste no quadro normalizado roda só quando não há calibração.

`_serie_normalizada` usa `_FRAC_REPOUSO = 0.08` para estimar o nível de repouso
(o degrau parte de zero e o tempo morto mantém a curva parada), valor do §21.

### 23.2 `test_2_11` deixou de ser falso-verde

Antes verificava apenas que `calibrate()` não levanta exceção, e passava com a
Decisão E inteira ausente (Ruling 34). Agora executa `identify_from_image` nas
300 amostras e verifica: o bloco existe com as seis chaves; `physical` é `None`
**exatamente** quando `ok` é falso; `calibration.ok` reflete o calibrador. E
registra como número medido quantas amostras SEM calibração ainda entregam
ζ ou τ/T.

**Medido: 300/300 com bloco, 300/300 com valor, 61/61 das sem calibração.**
Toda amostra entrega resposta adimensional, inclusive as 61 que o pipeline
descartava.

### 23.3 Custo e regressão — suíte completa, `pytest -q` sem filtros

| | antes | depois |
|---|---|---|
| **2.11** | "calibrate() nunca levanta" 300/300 | **300/300 bloco, 300/300 valor, 61/61 sem calibração** |
| **2.8 latência** | 172 ms · p95 254 | **177 ms · p95 265** ✅ (alvo 500) |
| 2.1 / 2.2 / 2.3 / 2.4 / 2.5 / 2.6 / 2.7 / 2.9 | — | **byte a byte idênticos** |
| falhas | 5 | **5** |

**Custo da Decisão E: +5 ms na mediana, +11 ms no p95.** A U-Net passou a rodar
também nas 61 amostras que a calibração derrubava (antes o `return` antecipado
vinha antes dela), e mesmo assim o custo é desprezível por causa da derivação
do §23.1.

**Nenhum critério físico se moveu um decimal** — a preservação da semântica de
`ok` é o que garante isso, e era o principal risco da mudança.

### 23.4 O que a Decisão E ainda NÃO faz

O critério **2.6 continua medindo só o nível físico**. Fazer o ζ adimensional
entrar no 2.6 exige decidir (decisão de projeto, não de implementação):

- ζ entra direto — é o mesmo parâmetro;
- ωₙ·T e θ/T precisam de valores de referência derivados do meta (a `t_window`
  está lá);
- K e τ **não** entram sem calibração.

Enquanto isso não for decidido, o ganho da Decisão E aparece no 2.11 e na
robustez do sistema, **não** no número do 2.6.

## 24. Ruling 46 — extrator: NÃO fechado. Sete regras testadas, todas piores; mas o teto NÃO é a representação

Tentativa de fechar o 2.2 atacando a redução coluna→ponto do `mask_to_polyline`.
**Resultado negativo, registrado com os números para ninguém repetir.**

### 24.1 A hipótese do meio pixel está refutada

`mask_to_polyline` devolve `x` como índice INTEIRO da coluna e `y` como mediana
das linhas do esqueleto — que corresponde à curva no CENTRO do pixel. Parecia um
deslocamento de meio pixel amplificado pela inclinação (com `m=27`, 13,5 px, a
ordem exata da cauda), e explicaria RMSE alto com viés quase nulo. Medido:

| dx usado na comparação | mediana | p95 |
|---|---|---|
| **0,00 (atual)** | **1,488** | **6,701** |
| 0,25 | 1,466 | 6,890 |
| 0,50 | 1,583 | 7,386 |
| 1,00 | 2,009 | 8,867 |

`dx=0` já é praticamente ótimo. **Não é artefato de convenção.**

### 24.2 Sete regras de redução, todas piores que a mediana

| variante | mediana | p95 |
|---|---|---|
| **mediana por coluna (atual)** | **1,488** | **6,701** |
| centro do maior run | 1,547 | 7,480 |
| descarta coluna multi-run | 1,495 | 9,631 |
| continuidade (centro do run) | 1,554 | 9,078 |
| continuidade (borda do run) | 1,580 | 9,099 |
| extremo → borda da tinta | 1,537 | 7,337 |
| extrapolação linear clipada à tinta | 1,616 | 8,257 |
| extremo + extrapolação | 1,626 | 8,155 |

**A mediana por coluna é a melhor das oito.** A hipótese do multi-run (§20.2)
descreve o mecanismo corretamente, mas nenhuma regra LOCAL que a explore melhora
o resultado — continuidade e extrapolação pioram bastante o p95.

### 24.3 Mas existe margem: o oráculo limitado à tinta PASSA

Limite superior de qualquer regra coluna→um-y: em cada coluna, o y ótimo
(a verdade), **limitado à faixa de tinta disponível**.

| | mediana | p95 | 2.2 |
|---|---|---|---|
| atual | 1,488 | 6,701 | **REPROVA** |
| **oráculo limitado à tinta** | **1,186** | **3,947** | **PASSA** |

**A informação ESTÁ na tinta e o teto NÃO é a representação.** A faixa de tinta
contém a verdade quase sempre; num trecho monótono ela fica no meio (a mediana
acerta) e numa coluna que contém um extremo da curva ela fica na borda (a mediana
erra metade da faixa). Margem disponível: 6,701 → 3,947.

### 24.4 Direção que os dados indicam (não implementada)

Nenhuma heurística LOCAL captura a margem. O que sobra é formulação **global**:
escolher a sequência `y(x)` que minimiza a segunda diferença sujeita a
`y(x) ∈ [tinta_min(x), tinta_max(x))` — programação dinâmica sobre as colunas.
É mudança de porte no módulo mais central do pipeline e **não foi tentada**.

Contraindicação a atalhos: as sete regras acima mostram que trocar a heurística
sem otimizar globalmente tende a PIORAR, então uma tentativa parcial é
provavelmente pior que o estado atual.

## 25. Ruling 47 — o ζ adimensional entrou no critério 2.6

`identify_from_image` passa a preencher `order` também no caminho adimensional —
o `PLANO §1.7` lista a estrutura como não dependente de calibração, e sem isso o
2.6 não tem como conferir a ordem nas amostras recuperadas. `params` continua
vazio ali, porque é por contrato o bloco FÍSICO. Seguro para os consumidores
antigos: os dois usos de `r["order"]` em `tests/part2` passam por `r["ok"]`
antes, que continua falso.

`test_2_6` ganhou um acumulador do nível adimensional cujo portão exige acerto de
estrutura mas **NÃO** exige `r["ok"]` — é isso que faz as amostras sem calibração
entrarem.

### 25.1 Desenho, e por que não é substituição

**O 2.6 físico fica intacto** — mesma definição, mesmos números, comparável com
as rodadas 3 a 6. O ζ adimensional entra como critério COMPANHEIRO
(`2.6-adim[zeta]`), com o mesmo alvo de ≤ 3 p.p. e com `assert`.

Razão de não fundir os dois num número só: o `PLANO §1.7` existe para manter a
distinção entre os níveis, e o §21.3 mediu 50,5 % de divergência no p95 entre
eles — são grandezas de acurácia diferente. Fundir apagaria isso e quebraria a
comparabilidade histórica do 2.6.

### 25.2 Medido

| | valor |
|---|---|
| `2.6-adim-aceitas` | **141/300, sendo 31 sem calibração** |
| **`2.6-adim[zeta]`** | **+1,53 p.p.** (oráculo 1,24 %, real 2,78 %) ✅ |
| `2.6[zeta]` físico, para comparar | +0,99 p.p. (~110 amostras) |

O ζ passa a ser medido sobre **141 amostras em vez de ~110** — as 31 de 2ª ordem
que a Decisão E recuperou. A degradação é maior (+1,53 × +0,99) porque inclui
exatamente as amostras difíceis sem calibração, e fica a menos da metade do
limite.

Suíte completa: **5 failed / 54 passed**, os mesmos de antes. 2.8 foi de 177 para
183 ms (ruído, segue aprovado); nenhum outro critério se moveu.

## 26. Estado ao fim do Bloco 7

| critério | início da sessão | fim |
|---|---|---|
| **2.6 (ζ físico)** | +3,65 p.p. ❌ | **+0,99 p.p. ✅** |
| **2.6-adim[zeta]** | não existia | **+1,53 p.p. ✅** (141 amostras) |
| **2.8 latência** | 885 ms ❌ | **183 ms ✅** |
| 2.6-clássico | +4,36 ❌ | **+1,62 ✅** |
| 2.11 | falso-verde | **300/300, 61/61 sem calibração ✅** |
| 2.1 / 2.7 | 0,6205 / 0,5856 ❌ | 0,6478 / 0,6330 ❌ |
| 2.2-piso | 6,70 px ❌ | 6,70 px ❌ |
| 2.4 / 2.5 / 2.9 | ❌ | ❌ |
| **falhas na suíte** | **7** | **5** |

Aberto, com o que se sabe de cada um:

1. **2.2** — margem medida (6,70 → 3,95) mas exige formulação global (Ruling 46).
2. **2.9 / 2.4 / 2.5** — poda medida (+3,0 p.p., custa 0,6 p.p. do 2.3, Ruling 37);
   os 14 restantes de `calibration_failed` e as 25 de `ocr_insuficiente` seguem
   sem hipótese.
3. **2.1 / 2.7** — a meta mede espessura de traço, não acurácia (Ruling 39). O
   caminho é revisar a meta, não treinar mais capacidade.

## 27. Ruling 48 — formulação global também perde, e o prior de SUAVIDADE é errado aqui

Tentativa de fechar o 2.2 pela direção que o §24.4 indicava: sequência mais suave
compatível com a faixa de tinta (suavizar e projetar na caixa
`[tinta_min, tinta_max]` por coluna, iterando; colunas sem tinta ficam livres e a
suavização as interpola).

| variante | mediana | p95 |
|---|---|---|
| **mediana por coluna (atual)** | **1,488** | **6,701** |
| global σ=1, 20 iter | 1,540 | 8,380 |
| global σ=2, 20 iter | 1,502 | 8,548 |
| global σ=2, 60 iter | 1,505 | 8,564 |
| global σ=4, 20/60 iter | 1,536 | 8,604 |
| global σ=8, 60 iter | 1,605 | 8,586 |

Todas perdem, e são **quase idênticas entre si** (12,7 % de cauda em todas), o que
indica convergência para a mesma solução independente de σ e de iterações.

**Diagnóstico: suavidade é o prior ERRADO.** Numa resposta ao degrau a curva é
genuinamente íngreme no transiente, então penalizar curvatura briga com a verdade
exatamente onde a cauda mora. Isso explica de uma vez as **14 tentativas** que
perderam (7 locais no §24.2, o meio pixel no §24.1, 6 globais aqui) e por que só
o oráculo ganha: ele não usa prior, usa a verdade.

O prior correto seria **o próprio modelo** — ajustar `y(t)` a observações em
INTERVALO (`[tinta_min, tinta_max]` por coluna) em vez de a pontos, com perda nula
dentro do intervalo e quadrática fora. Isso elimina a polilinha intermediária e é
mudança arquitetural em `classical.py` + `polyline.py`.

## 28. Ruling 49 — e NÃO vale a mudança: um extrator perfeito não recupera acurácia

Antes de propor a mudança arquitetural, mediu-se o TETO do extrator: redução
oráculo (a que passa o 2.2, §24.3) contra a atual, ambas com afim VERDADEIRO,
Wilcoxon pareado. Dado bruto: `reports/part2_extrator_teto.json`, n=266.

| parâmetro | E0 oráculo de série | E1 atual | **E1 teto** | ganho possível | p |
|---|---|---|---|---|---|
| K | 0,121 | 0,261 | 0,268 | −0,008 | 0,21 (n.s.) |
| tau | 0,233 | 0,431 | 0,424 | +0,007 | 0,10 (n.s.) |
| wn | 0,709 | 1,150 | 0,991 | **+0,160** | **0,041** |
| **zeta** | 1,215 | 2,026 | **1,635** | **+0,391** | **0,28 (n.s.)** |
| theta | 0,062 | 0,160 | 0,134 | +0,026 | **1,4e-08** |

**Um extrator perfeito na redução coluna→ponto não recupera acurácia de forma
significativa.** O ganho em ζ tem p=0,28. Só ωₙ (+0,160) e θ (+0,026) melhoram
significativamente, e θ é desprezível em magnitude.

**E o achado que reorienta tudo:** mesmo no TETO, ζ fica em 1,635 contra 1,215 do
oráculo de série. **A maior parte da degradação que o Ruling 42 atribuiu ao
extrator NÃO está na redução coluna→ponto** — está na esqueletização, na
interpolação de vãos e na discretização em pixel. Fechar o 2.2 não a alcança.

### 28.1 Consequência

**2.2 é, como 2.1, mais desalinhamento entre critério e representação do que
defeito consertável.** As duas coisas que o §19.4 tratou como "dois critérios,
uma causa" são de fato o mesmo COMPONENTE mas defeitos DIFERENTES:

- o p95 do 2.2 vem da redução coluna→ponto — corrigível em princípio (o oráculo
  passa), mas nenhuma das 14 heurísticas testadas o consegue, e corrigi-lo **não
  compra acurácia**;
- a degradação de acurácia do Ruling 42 vem de outra parte do extrator, e o teto
  medido aqui mostra que ela sobrevive a uma redução perfeita.

**Recomendação revista:** parar de investir no extrator. O esforço rende mais em
2.9/2.4/2.5, onde a poda (Ruling 37) tem ganho medido de +3,0 p.p., e na revisão
das metas de 2.1/2.2 com a base quantitativa dos Rulings 39, 43, 46, 48 e 49.

## 29. Ruling 50 — a métrica PERPENDICULAR resolve 2.1 e 2.2: o pipeline é sub-pixel

Dado bruto: `reports/part2_perp.json` e `reports/part2_perp_pred.json`.

O 2.2 mede diferença **vertical** entre polilinha e curva. Num trecho de
inclinação `m`, um erro geométrico de meio pixel aparece como `m/2` px de erro
vertical. A distância **perpendicular** é invariante à inclinação.

| | mediana | p95 |
|---|---|---|
| vertical (métrica atual do 2.2) | 1,488 px | **6,701 px** |
| **perpendicular** | **0,614 px** | **1,135 px** |

Por faixa de inclinação máxima (px/px):

| faixa | vertical | perpendicular |
|---|---|---|
| 0,49–2,86 | 0,619 | 0,519 |
| 2,86–6,51 | 1,000 | 0,588 |
| 6,51–13,82 | 1,770 | 0,657 |
| 13,82–26,73 | 3,371 | 0,651 |
| **26,73–177,47** | **6,665** | **0,802** |

Correlação com a inclinação: **vertical +0,869 · perpendicular +0,326**. O
perpendicular varia 1,5× entre a faixa mais suave e a mais íngreme; o vertical,
10,8×.

**O extrator é sub-pixel em geometria. A reprovação do 2.2 é artefato de medir
distância vertical em curva íngreme.** Isto explica retroativamente o Ruling 49:
a redução "perfeita" não comprou acurácia porque **não havia erro geométrico a
recuperar**.

Perpendicular à curva verdadeira, máscara verdadeira × predita:

| | RMSE mediana | p95 entre amostras |
|---|---|---|
| máscara VERDADEIRA (piso) | **0,614** | 1,135 |
| máscara PREDITA (rede) | **0,800** | 1,699 |
| **adicionado pela rede** | **+0,144** | +0,543 |

### 29.1 Metas revisadas propostas

| critério | métrica atual | medido | **métrica proposta** | **alvo** | medido |
|---|---|---|---|---|---|
| **2.1** | IoU de máscara ≥ 0,85 | 0,6478 ❌ | RMSE perpendicular da polilinha PREDITA | ≤ 1,0 px med, ≤ 2,0 px p95 | **0,800 / 1,699 ✅** |
| **2.2** | RMSE vertical ≤ 2 / p95 ≤ 5 | 1,488 / 6,701 ❌ | RMSE perpendicular da polilinha da máscara VERDADEIRA | ≤ 1,0 px med, ≤ 2,0 px p95 | **0,614 / 1,135 ✅** |

**De onde vem o limiar — e por que não é mover a trave.** O limiar sai do
ORÇAMENTO do 2.6 (3 p.p.), não do resultado atual. A ablação (Ruling 42) dá a
ponte: com 0,800 px de erro perpendicular, a contribuição da rede à degradação de
ζ é +0,127 p.p. (nem significativa), ~4 % do orçamento. O limiar de 1,0 px mantém
a rede em ~5 %. E **não é vacuoso**: 25 % de folga sobre o medido, então uma
regressão real derruba o critério.

A legitimidade da troca vem de a MÉTRICA ter mudado para medir a grandeza certa,
com prova independente de que a antiga era errada (Rulings 39, 43, 48, 49 e as
correlações acima). Afrouxar o limiar da métrica ANTIGA seria mover a trave.

### 29.2 Dois resguardos, obrigatórios

1. **Continuar reportando IoU e RMSE vertical como DIAGNÓSTICO, sem alvo.** IoU
   informa fidelidade de espessura; o vertical informa o efeito da declividade.
   Removê-los esconderia; mantê-los sem veredito preserva comparabilidade com as
   rodadas 3 a 6.

2. **A perpendicular tem ponto cego e precisa de par.** Ela **não penaliza erro
   AO LONGO da curva** — polilinha deslocada no tempo, mas sobre a curva, pontua
   zero. Num degrau isso é exatamente o θ. O par já existe: **`2.6[theta]` mede o
   erro longitudinal** (+0,28 p.p. hoje, alvo ≤ 3). Adotar a perpendicular
   SOZINHA, sem esta observação no plano, abriria vão para um extrator com viés
   temporal.

**Não implementado.** Trocar a métrica de 2.1 e 2.2 é alteração de critério do
`PLANO`, decisão do autor da monografia, não do executor.

## 30. Ruling 51 — a troca 2.9 por 2.3 (poda) NÃO se justifica hoje

| config | 2.9 (alvo ≥ 90 %) | 2.3 (alvo ≥ 95 %) |
|---|---|---|
| **atual** | 79,9 % ❌ | **95,8 % ✅** (margem 0,8 p.p.) |
| poda ≤2, piso 4 | 82,9 % ❌ | 95,2 % ✅ (margem **0,2 p.p.**) |
| poda ≤2, piso 3 | 85,3 % ❌ | 94,1 % ❌ |

Contra:

1. **Nunca converte reprovação em aprovação** — o 2.9 exige 90 % e a poda chega a
   82,9 %/85,3 %.
2. **Queima a margem do 2.3 de 0,8 para 0,2 p.p.**, deixando um critério saudável
   à mercê de variação de tesseract/dataset/máquina (Rulings 11 e 12).
3. **Aceita calibração que o teste de consistência rejeitou** — e o
   `_equiespacados` existe exatamente para pegar leitura errada que sobrevive ao
   RANSAC (Ruling 32).
4. **Efeito em 2.6/2.4/2.5 NÃO medido** — a poda foi simulada, nunca implementada.
5. **A Decisão E mudou o cálculo.** Falha de calibração deixou de matar a amostra:
   agora custa as unidades FÍSICAS, não a resposta (Ruling 47: o 2.6-adim já conta
   as 31 recuperadas). A pressão para subir o 2.9 caiu.

A favor: recupera 16 das 30 `calibration_failed` **com unidades físicas**, que a
Decisão E não dá. Se a Parte 3 (PID/IMC) consome físico, cada amostra conta.

**Recomendação: não fazer a troca agora.** Se for feita, `piso 4` é a única
configuração defensável, e exige rodar a suíte inteira antes de adotar.
