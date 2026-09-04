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

## 31. Ruling 52 — 2.4 e 2.9 são a MESMA grandeza; 2.3 e 2.5 têm limiares INCONSISTENTES

Dado bruto: `reports/part2_tres_criterios.json`, `part2_guarda.json`.

### 31.1 Correção ao Ruling 51: a poda FECHA o 2.5

O Ruling 51 concluiu que a poda "nunca converte reprovação em aprovação". **Isso
estava errado** — a análise olhou só 2.9 e 2.3, e o 2.5 não estava no quadro.
Avaliando os QUATRO critérios juntos:

| config | 2.9 cobertura | 2.4 rejeição | **2.5 precisão** | 2.3 |
|---|---|---|---|---|
| **atual** | 79,7 % | 20,3 % | **72,1 % ❌** | 95,8 % ✅ |
| poda ≤2 piso 4 | 82,7 % | 17,3 % | 82,7 % ❌ | 95,2 % ✅ |
| **poda ≤2 piso 3** | 85,0 % | 15,0 % | **95,6 % ✅** | 94,1 % ❌ |

**As 17 rejeições desnecessárias são 100 % `calibration_failed`** — o guardião de
consistência é a fonte inteira dos falsos alarmes. A poda recupera 15 das 17.

### 31.2 2.4 SUBSUME 2.9

`test_2_4` mede `rejeitadas/total`, que é exatamente `1 − cobertura` do 2.9.
São a mesma medição: 2.4 exige rejeição < 5 % (cobertura > 95 %) e 2.9 exige
cobertura ≥ 90 %. **Se 2.4 passa, 2.9 passa por construção.** Ter os dois como
critérios independentes dá peso duplo à mesma grandeza.

Consequência prática: **2.4 é inalcançável hoje.** Exige ≤ 15 rejeições em 300;
as melhores configurações dão 45. Mesmo recuperando TODAS as 30
`calibration_failed`, sobram as 25 de `ocr_insuficiente` → 31 rejeições (10,3 %).
O 2.4 só fecha atacando os dois motivos até quase zero.

### 31.3 2.3 e 2.5 são incompatíveis por construção

2.3 exige erro de escala ≤ 1 % nas ACEITAS; 2.5 chama a rejeição de "justificada"
quando o erro seria > 5 %. As amostras na faixa 1–5 % são **simultaneamente**
"não deviam ter sido rejeitadas" (2.5) e "ruins o bastante para estragar o 2.3".

Prova aritmética de que nenhum subconjunto fecha os dois (16 recuperáveis,
11 com erro ≤ 1 %, 15 com erro ≤ 5 %):

| escolha | 2.3 | 2.5 |
|---|---|---|
| recuperar as 16 | 94,1 % ❌ | 95,6 % ✅ |
| recuperar só as 11 (≤ 1 %) | **96,0 % ✅** | 88,0 % ❌ |

O guardião por resíduo do ajuste foi testado (o resíduo SEPARA: mediana 0,0005
nas boas × 0,0033 nas ruins) e **não resolve**: a varredura alterna direto de
(2.3 ✅, 2.5 ❌) em `res ≤ 0,002` para (2.5 ✅, 2.3 ❌) em `res ≤ 0,005`. Não
existe limiar onde ambos passam.

### 31.4 A margem de rótulos NÃO conserta `ocr_insuficiente`

`MARGIN_Y_W`/`MARGIN_X_H` são fixos em px enquanto o dpi varia 3,3× (Ruling 33).
Sete configurações testadas:

| config | cobertura | `ocr_insuf` | `cal_failed` |
|---|---|---|---|
| atual (90/40) | 79,7 % | 25 | 30 |
| fixo ×2 | 80,3 % | 23 | **31** |
| **proporcional 0,9·dpi** | **81,0 %** | 24 | 28 |
| proporcional 1,2·dpi | 80,3 % | 22 | **32** |

Ganho máximo de **1,3 p.p. (4 amostras)**, e na maioria das configurações o
`calibration_failed` SOBE: alargar a faixa captura número que não é rótulo
(legenda, anotação), que passa o `_NUM_RE` e entra como par falso. **É troca de
motivo, não conserto.** Confirma o Ruling 33: o corte geométrico é causa
secundária.

### 31.5 Veredito: os três NÃO fecham

| critério | alvo | melhor alcançado | fecha? |
|---|---|---|---|
| 2.9 | cobertura ≥ 90 % | 85,0 % (poda piso 3) + ~1 p.p. | **não** |
| 2.4 | rejeição < 5 % | 15,0 % (45 rejeições × 15 permitidas) | **não** |
| 2.5 | precisão ≥ 90 % | **95,6 %** (poda piso 3) | **sim, mas custa o 2.3** |

O que está disponível é uma **troca um-por-um**: 2.5 fecha, 2.3 abre. Não há
ganho líquido em contagem de critérios, e a escolha depende de qual importa mais
para a Parte 3.

### 31.6 Recomendação: revisar os critérios, não o código

O bloqueio não é de implementação. Três propostas, todas decisão do autor:

1. **Unificar 2.4 e 2.9** numa só medição de cobertura, com um limiar. Manter
   duas com limiares diferentes sobre a mesma grandeza é peso duplo.
2. **Alinhar os limiares de 2.3 e 2.5** (hoje 1 % e 5 %). Enquanto divergirem,
   existe uma faixa de amostras que nenhuma configuração resolve.
3. **Reconsiderar o alvo do 2.4** (< 5 % de rejeição). Com precisão de OCR de
   92 % e recall de ~85 % (Ruling 36), exigir 95 % de cobertura de calibração é
   incompatível com o desempenho do OCR disponível — e a Decisão E já reduziu o
   custo de uma calibração que falha (Ruling 47).

## 32. Ruling 53 — a revisão dos critérios foi ESCRITA no plano e IMPLEMENTADA

`PLANO.md` ganhou a **§2.12** (as quatro classes, com a evidência de cada uma) e a
tabela de critérios foi atualizada. `tests/part2/` implementa as Classes A, B e C.

### 32.1 O que foi implementado

`tests/part2/conftest.py`: `erro_perpendicular()` (distância ponto-a-poligonal,
vetorizada em blocos), mais os limiares `PERP_MED_MAX = 1.0`, `PERP_P95_MAX = 2.0`
e `ESCALA_TOL = 0.01`.

| critério | antes | agora |
|---|---|---|
| **2.1** | IoU ≥ 0,85 | RMSE perpendicular ≤ 1,0 / 2,0 px; **IoU vira diagnóstico** |
| **2.2** | vertical ≤ 2 / 5 px | RMSE perpendicular ≤ 1,0 / 2,0 px; **vertical vira diagnóstico** |
| **2.7** | IoU por estrato ≥ 0,75 | perpendicular por estrato ≤ 1,0 px; **IoU vira diagnóstico** |
| **2.4** | rejeição < 5 % | **aposentado** — diagnóstico, unificado no 2.9 |
| **2.5** | erro > 5 % | erro > **`ESCALA_TOL`**, alinhado ao 2.3 |
| **2.3** | `0.01` literal | `ESCALA_TOL` — alinhamento garantido por construção |

**A poda NÃO foi ativada.** A decisão de não fazer a troca 2.5↔2.3 foi mantida;
alinhar o limiar sozinho move o 2.5 de 0,721 para 0,885 **sem** abrir o 2.3.

### 32.2 Armadilha 8 — justificar o limiar com uma estatística e implementar com outra

Na primeira versão, o 2.1 e o 2.7 mediam a **mediana** do erro perpendicular por
amostra, enquanto o limiar de 1,0 px foi derivado do **RMSE** por amostra
(Ruling 50: 0,800 px → +0,127 p.p. em ζ). A mediana é mais leniente e dava
**0,549 px**, deixando o critério mais fácil que a própria justificativa.

Corrigido para RMSE por amostra, coerente com o 2.2 e com a derivação: o 2.1
passa a reportar **0,800 / 1,699 px**, os números exatos do Ruling 50.

**Registrado porque é o modo de falha específico de uma revisão de critério:** não
basta a métrica ser a certa e o limiar ser derivado — a *estatística* medida tem de
ser a mesma da derivação, ou o critério fica mais frouxo do que o argumento que o
sustenta.

### 32.3 Resultado

| critério | antes da revisão | depois |
|---|---|---|
| **2.1** | IoU 0,6478 ❌ | **0,800 / 1,699 px ✅** (IoU 0,6478 diagnóstico) |
| **2.2** | vertical 1,49 / 6,70 ❌ | **0,614 / 1,135 px ✅** (vertical diagnóstico) |
| **2.7** | IoU por estrato ❌ | **0,680 – 0,956 px, todos ✅** |
| 2.3 | 0,958 ✅ | 0,958 ✅ |
| 2.4 | 0,203 ❌ | diagnóstico ❓ |
| **2.5** | 0,721 ❌ | **0,885 ❌** (+16 p.p.; falta 1,5 p.p. ≈ 1 amostra em 61) |
| 2.9 | 0,797 ❌ | 0,797 ❌ (limiar EM ABERTO, Classe D) |

**Falhas na suíte: 5 → 2.** Restam o 2.5 (a 1,5 p.p.) e o 2.9 (limiar não decidido).

**Achado que a métrica nova revela:** o estrato **`traco=:` fica em 0,956 px contra
alvo de 1,0** — 4,4 % de folga, o mais apertado de todos. É coerente com a
docstring do `mask_to_polyline`, que documenta que o estilo `:` deixa 43 % das
colunas sem tinta. Com a métrica de área isso ficava escondido no ruído da
espessura.

### 32.4 O que continua em aberto

1. **Limiar da cobertura (2.9)** — Classe D. Deliberadamente não proposto: exige
   argumento sobre o que o sistema precisa entregar para ser útil, e a Decisão E
   mudou o custo de falhar (perde-se a escala, não a resposta).
2. **2.5 a 1,5 p.p.** — a poda fecharia (100 %) ao custo do 2.3; decisão mantida de
   não fazer. Fechar sem essa troca exige reduzir os misreads de OCR na origem.

## 33. Ruling 54 — OOD de aquisição: a Decisão E é a resposta robusta, e a fronteira é a ROTAÇÃO

**Este NÃO é um caso real.** Não há figura real no repositório (`img.py` é outro
gerador matplotlib). É o eixo mais próximo que se constrói sem uma: degradações de
**aquisição** que o gerador nunca produz — JPEG, reescala, rotação, ruído de
sensor. O espaço do gerador varia dpi, grade, legenda, traço, cor, fundo,
marcador, anotações e SNR, mas nada disso.

Dado bruto: `reports/part2_ood_aquisicao.json`, `part2_ood_adimensional.json`.
120 amostras × 10 degradações; a tabela de ζ restrita às 58 de 2ª ordem.

### 33.1 A calibração desaba; a estrutura e o nível adimensional não

| degradação | calibração ok | ordem ok | adimensional sai |
|---|---|---|---|
| original | 78,3 % | 90,0 % | 53 % |
| JPEG q=90 | 75,0 % | 90,0 % | 53 % |
| JPEG q=60 | 55,0 % | 91,7 % | 48 % |
| JPEG q=30 | 45,0 % | 90,8 % | 46 % |
| reescala 0,5× | 65,8 % | 90,0 % | 50 % |
| reescala 0,33× | 38,3 % | 89,2 % | 51 % |
| **rotação 0,5°** | **3,3 %** | 93,3 % | 48 % |
| **rotação 2°** | **0,0 %** | 88,3 % | 50 % |
| **ruído σ=8** | **5,0 %** | 86,7 % | 45 % |
| JPEG60 + 0,5× + rot 1° | **0,0 %** | 90,0 % | 50 % |

**Meio grau de rotação leva a calibração de 78,3 % a 3,3 %.** Causa:
`detect_plot_bbox` e `detect_tick_pixels` supõem moldura alinhada aos eixos, e
projeção por linha/coluna não sobrevive a inclinação. Ruído σ=8 faz o mesmo com a
limiarização de tinta (5,0 %).

Enquanto isso a **ordem acerta 86,7–93,3 % em TODAS as degradações**, inclusive
rotação de 2° e a combinada.

### 33.2 E o ζ adimensional continua CERTO, não só saindo

Disponibilidade não é acurácia. Restrito às 58 amostras de 2ª ordem:

| degradação | adim sai | **MAPE ζ adim** | p95 |
|---|---|---|---|
| original | 94,8 % | **2,55 %** | 32,6 % |
| JPEG q=30 | 87,9 % | **3,13 %** | 71,6 % |
| reescala 0,33× | 91,4 % | **3,24 %** | 54,4 % |
| **rotação 0,5°** | 93,1 % | **2,89 %** | 65,6 % |
| **ruído σ=8** | 79,3 % | **2,60 %** | 51,0 % |
| JPEG60 + 0,5× + rot 1° | 91,4 % | 5,84 % | 83,2 % |
| **rotação 2°** | 89,7 % | **14,48 %** | 99,9 % |

**A previsão do `PLANO §1.7` se confirma quantitativamente:** *"no conjunto OOD é
onde o OCR tem mais chance de falhar e onde a resposta adimensional é
provavelmente a única robusta"*. Em rotação de 0,5° a calibração entrega 3,3 % das
amostras e o adimensional entrega 93,1 % **com acurácia intacta** (2,89 % contra
2,55 % da linha de base).

**A fronteira é a rotação, e o mecanismo é claro.** 2° quebra o adimensional
(14,48 %, 5,7× pior) porque rotação distorce a **forma** da curva, e ζ vem da razão
de *overshoot*, que é forma. JPEG, ruído e reescala degradam a máscara sem torcer a
geometria; rotação torce. Até ~1° a degradação é tolerável (5,84 % na combinada).

**Ressalva: o p95 é alto em todas as condições** (32,6 % já na linha de base). A
mediana é robusta, a cauda não — consistente com o p95 de 43,82 % que o caminho
físico já apresentava (Ruling 44).

### 33.3 Consequências

1. **A Decisão E deixou de ser conveniência e passou a ser a característica que
   sustenta o sistema fora da distribuição.** É argumento de monografia com número:
   a resposta adimensional é a única que sobrevive a aquisição degradada.
2. **A calibração tem uma fragilidade não documentada até aqui: rotação.** Meio
   grau — imperceptível a olho, comum em qualquer digitalização — a derruba
   quase por completo. Nenhum critério da Parte 2 mede isso, e o gerador não
   produz o estrato. Candidato natural a critério novo ou a estrato do gerador.
3. **O caso REAL continua pendente** e depende de uma figura fornecida (Ogata,
   Nise, print de planilha, gráfico de artigo). A previsão testável, a partir
   destes números: a calibração provavelmente falha por desalinhamento e o ζ sai
   pelo nível adimensional com erro na casa de 3 %, desde que a figura esteja
   dentro de ~1° do alinhamento.

## 34. Ruling 55 — CASO REAL: uma imagem encontrou dois defeitos que 300 sintéticas não

Imagem: `resposta_degrau.png` (842×569), produzida fora do gerador do projeto.
Verdade declarada: `num = [ωn²]`, `den = [1, 2ζωn, ωn²]` com **ζ = 0,5 e ωn = 2,0**;
degrau unitário, janela 10 s, θ = 0. Segunda ordem subamortecida, *overshoot*
observado ~16,4 % (que corresponde exatamente a ζ = 0,5: 16,3 %).

### 34.1 O pipeline como está FALHOU

```
ok=False   order='fopdt'   reason='calibration_failed'
physical      = None
dimensionless = {zeta: None, wn_T: None, tau_T: 0.0632, theta_T: 0.0530, K_yrange: 0.7732}
```

A calibração reprovar era previsto (Ruling 54) e a Decisão E cobriu. **O erro
inaceitável é a ORDEM:** primeira ordem para uma curva com *overshoot* visível.
Com ordem errada, ζ e ωₙ·T saem nulos e a resposta é inútil.

### 34.2 Defeito 1 — a máscara da U-Net

Sobreposição da máscara sobre a imagem mostra três problemas:

- **traça a curva corretamente até t ≈ 6,2** e **perde os últimos 38 % da janela**
  — exatamente onde a curva azul passa a COINCIDIR com a reta vermelha tracejada
  de referência em y = 1,0;
- **captura a reta de referência** no trecho t ≈ 1,2 a 3;
- **captura glifos de texto**: a polilinha vai de y = 21 a 551, fora da moldura
  (39 a 503) — pegou título e rótulo do eixo x.

Trocando apenas a máscara, com o MESMO estágio D:

| máscara | ordem | ζ | ωₙ |
|---|---|---|---|
| **verdade** | second | **0,5000** | **2,000** |
| U-Net (pipeline) | **fopdt ❌** | — | — |
| extrator clássico do projeto | second ✅ | 0,3847 | 2,173 |
| cor azul isolada | second ✅ | 0,4000 | 2,179 |

**A falha de ordem é da máscara, não do estágio D.**

**Contaminação adicional — parágrafo RETRATADO, ver §35.2.** A redação original
dizia que a **legenda** contamina 37 colunas com dois ramos azuis (exemplo em
x=536). Aquelas 37 colunas estavam numa máscara de **limiar de cor** que construí
para diagnosticar, **não na máscara da U-Net**, e atribuí o mecanismo ao objeto
errado. Na máscara da U-Net são **44** colunas multi-bloco, todas em x=[81,389] —
na parte ASCENDENTE da curva —, e a amostra de linha da legenda (25 px de tinta em
y≈453) não contamina nenhuma delas. A ambiguidade multi-bloco é real e o
mecanismo de desambiguação é necessário (§35.1); o **exemplo** estava errado.
Desambiguando por continuidade, o `nrmse` cai de 0,0826 para **0,0316** — este
número foi medido na máscara por cor e continua valendo para ela.

### 34.3 Defeito 2 — o estimador de repouso da Decisão E (meu)

Mesmo com máscara limpa e colunas multi-bloco desambiguadas — **não** "legenda
desambiguada", como esta linha dizia; ver a retratação do §34.2 e o §35.2 —,
ζ = 0,4372 contra 0,5 (**12,6 %**).
Causa: `_FRAC_REPOUSO = 0.08` estima o nível de repouso pela **mediana das 8 %
primeiras colunas**, supondo prefixo plano por tempo morto. Aqui **θ = 0**: a curva
sobe desde t = 0 e em t = 0,8 já vale ~0,28, então a "mediana do repouso" fica em
~0,1. Isso subestima o patamar em **3,8 %**, e 3,8 % no patamar viram **12,6 % em ζ**,
porque ζ vem da razão de *overshoot* (o ajuste implica 21,7 % contra 16,4 % reais).

| estimador do repouso | ζ | erro | ωₙ | erro | nrmse |
|---|---|---|---|---|---|
| mediana dos 8 % (ATUAL) | 0,4372 | 12,6 % | 2,2015 | 10,1 % | 0,0316 |
| mediana dos 3 % | 0,4903 | 1,9 % | 2,0452 | 2,3 % | 0,0048 |
| **primeiras 3 colunas** | **0,5018** | **0,4 %** | **2,0377** | **1,9 %** | **0,0029** |
| percentil 99 (extremo robusto) | 0,4954 | 0,9 % | 2,0245 | 1,2 % | 0,0028 |

**Por que o sintético nunca pegou isso:** o gerador sorteia θ e quase sempre
positivo, então o prefixo plano existe e o viés não aparece.

### 34.4 A resposta que o sistema É capaz de dar

Com moldura respeitada, colunas multi-bloco desambiguadas (**não** "legenda
desambiguada" — §34.2 retratado, §35.2) e repouso estimado corretamente:

> **ζ = 0,5018** (erro **0,4 %**) · **ωₙ = 2,0377** (erro **1,9 %**) ·
> *overshoot* implicado 16,2 % contra 16,4 % observados · `nrmse` **0,0029**

`nrmse` equivalente ao do sintético (~0,0035). **A capacidade existe; falharam dois
elos consertáveis.**

### 34.5 O terceiro problema, NÃO consertável nos mesmos termos

A perda da cauda (t > 6,2) acontece onde a curva coincide com a reta de referência.
O extrator clássico atravessa esse trecho (737 pontos, largura inteira); a U-Net
não. Não há tinta nas colunas perdidas, então nenhuma regra de polilinha recupera:
é limitação da máscara.

**E o gerador não produz o estrato.** `dataset/randomize.py:317-332` sorteia 1 a 3
distratores com `frac ~ U(0.03, 0.97)` — retas de span completo, mas em posição
uniforme e "sem relação com o rótulo". Uma reta de referência **no patamar** (o
caso real corriqueiro: *setpoint* marcado) tem probabilidade baixa de sair por
acaso. Enquanto o estrato não existir, a falha não é mensurável nem treinável.

## 35. Ruling 56 — as correções do caso real: o que fechou, o que a suíte NÃO media, e o que fica aberto

O §34 (Ruling 55) registrou dois defeitos que uma única imagem real expôs e que
300 sintéticas não. Os dois estão fechados. Esta seção registra o que os fechou,
uma correção ao meu próprio diagnóstico do §34.2, um **ponto cego de medição** que
a rodada revelou e que é o achado mais transferível do bloco, um **erro de método**
que vale para o TCC inteiro, e os itens que ficam abertos.

Toda medição abaixo vem acompanhada da **população**: o `n` e se é o corpus
inteiro ou o subconjunto **sem calibração**. Omitir isso foi exatamente o erro que
custou uma rodada de correção (§35.4).

### 35.1 Os dois defeitos e o que os fechou

**Defeito 1 — a ORDEM (`fopdt` onde a curva tem *overshoot* visível).** Fechado
por duas mudanças em `identify/polyline.py`:

1. **recorte da polilinha à moldura do gráfico** (`mask_to_polyline(mask, bbox=)`,
   chamado de `identify/pipeline.py`): a máscara da U-Net capturava glifos de
   título e de rótulo de eixo, e a polilinha ia de y=21 a y=551 contra uma moldura
   de 39 a 503;
2. **desambiguação de colunas com mais de um bloco de tinta**: a coluna passa a
   seguir o bloco mais próximo do ponto anterior, em vez da mediana de todas as
   linhas com tinta, com portão de 3× a espessura mediana e **reset da referência
   após vão largo** (o mesmo fator de 3×).

Efeito medido no corpus sintético (n=300): `2.1` (erro perpendicular da máscara
predita) vai de **0,800/1,699 px** para **0,802/1,529 px** — a mediana é estável e
o p95 MELHORA 0,17 px. `2.2-piso` (máscara verdadeira) fica em 0,614/1,135 →
0,615/1,137 px, ou seja neutro, como esperado: a máscara verdadeira tem só 2,4% de
colunas multi-bloco contra 22,3% da predita, então a desambiguação atua onde o 2.1
mede e quase não atua onde o 2.2 mede. `2.7[traco=:]` — o estrato pontilhado, com
43% das colunas sem tinta, que era o risco do reset — fica em **0,956 px**,
idêntico à linha de base.

**Defeito 2 — ζ e ωₙ.** Precisou de **duas** correções em `identify/pipeline.py`,
e só a primeira estava prevista:

*(a) nível de repouso.* `_FRAC_REPOUSO = 0.08` (mediana das 8% primeiras colunas)
virou `_N_REPOUSO = 5` colunas fixas, em `_nivel_de_repouso()`. A varredura está
em `reports/part2_repouso_varredura.md`; a coluna "REAL" é o caso real (**n=1**) e
as duas seguintes são o subconjunto **sem calibração e de 2ª ordem** das 300
primeiras de `data/test` (**n=33**):

| estimador do repouso | ζ real (n=1) | erro | MAPE ζ sint. (n=33) | ordem ok (n=33) |
|---|---|---|---|---|
| mediana de 8% da largura (anterior) | 0,4851 | 3,0% | 2,92% | 97,0% |
| mediana de 3% da largura | 0,5045 | 0,9% | 3,02% | 100,0% |
| **mediana de 5 colunas (ESCOLHIDO)** | **0,5057** | **1,1%** | **1,34%** | **100,0%** |
| mediana de 3 colunas | 0,5061 | 1,2% | 2,72% | 100,0% |
| percentil 99 | 0,5052 | 1,0% | 3,64% | 100,0% |

O critério de escolha foi o do plano, nas três alíneas: melhora no real, **não
piora** no sintético, e é o único que ganha nas duas populações ao mesmo tempo — a
mediana de 3% ganha 0,2 p.p. no real e perde 0,1 p.p. no sintético.

*(b) SEGUNDA CAUSA RAIZ, descoberta durante a medição e não prevista no plano —
a normalização do tempo.* `_serie_normalizada` normalizava `t` pela **extensão
observada da polilinha** (`x[-1] − x[0]`). No caso real a curva assentada se
sobrepõe à reta de referência tracejada e a máscara não separa as duas, de modo
que **~38% da largura fica sem tinta detectada**: a polilinha cobre só **0,617**
da moldura. Normalizar por uma janela truncada **comprime o tempo e infla ωₙ na
mesma proporção** — erro de ωₙ de **37,7%** (n=1). Normalizar pela largura da
**moldura** leva esse erro a **1,0%** (n=1).

A mudança NÃO é incondicional. No corpus sintético a moldura **não é** a janela de
dados: ela é de **+2,2% a +11,7%** mais larga (mediana +6,9%; n=179, as sem
calibração com polilinha ≥ 10 pontos e moldura válida), por margem do matplotlib
(`x_margin_lo`/`x_margin_hi`, que o próprio `sample_style` sorteia). Usar a
moldura sempre injetaria essa margem como viés sistemático onde a extensão
observada já é a melhor referência. Ficou **condicionada à truncagem detectada**,
por `_COBERTURA_MIN_MOLDURA = 0.75`; a origem de `t` passa a vir do mesmo
referencial que a escala (`bbox_px[0]` quando a moldura é usada), senão o
deslocamento vaza como viés ADITIVO em θ (mediana 3,1%, p95 5,5%, máx 11,4% da
largura da moldura, n=179).

Resultado no critério: **`2.6-adim[zeta]` vai de +1,53 p.p. (real 2,78%) para
+0,89 p.p. (real 2,08%)**, com as amostras comparáveis subindo de 141/300 para
**143/300** (31 → 33 sem calibração). No caso real, os dois testes de
`tests/part2/test_caso_real.py` passam: ordem `second`, ζ e ωₙ dentro da
tolerância.

**Sobre a folga do limiar, com o número certo.** A cobertura das 179 sem
calibração vai de **0,7713** (`sample_00639`, fopdt) a 0,9873, mediana 0,9332, e
**nenhuma** cai abaixo de 0,75 — o ramo da moldura não dispara em nenhuma amostra
do sintético. A folga do limiar até o mínimo observado é de **2,1 p.p.**, não os
~13 p.p. que a primeira rodada documentou (aquilo vinha de olhar só as 75 de 2ª
ordem). Para baixo a folga é confortável: o caso real está em 0,617.

### 35.2 CORREÇÃO ao §34.2 — a legenda não era o mecanismo, e o erro é meu

O §34.2 atribuía a contaminação da máscara à **legenda**, com "37 colunas com dois
ramos azuis". Está errado, e o erro é de diagnóstico meu, não do código.

As 37 colunas estavam numa máscara de **limiar de COR** que construí para
diagnosticar, **não na máscara da U-Net**. Medido na máscara da U-Net: são **44**
colunas multi-bloco, **todas** em x=[81,389] — a parte ASCENDENTE da curva —, e a
amostra de linha da legenda tem **25 px de tinta em y≈453**, sem contaminar
nenhuma coluna daquele intervalo. A U-Net quase não vê a legenda.

Consequências, e a distinção importa:

- **A mudança da Task 3 continua justificada** — a ambiguidade multi-bloco é real
  (44 colunas), e o ganho está medido no `2.1` p95 (1,699 → 1,529 px, n=300). O
  mecanismo é a curva passando perto da reta de referência e perto do eixo, não a
  legenda.
- **O primeiro teste escrito para ela era vacuoso** e foi substituído. Ele
  inspecionava x ≥ 485, região onde há **zero** colunas multi-bloco: passava por
  razão alheia ao que afirmava testar. Foi trocado por dois testes sintéticos e
  determinísticos, com discriminação provada nas duas direções. Números medidos:
  com o mecanismo neutralizado, a coluna sob teste erra **40,0 px** contra a
  tolerância de **2,0 px** do teste; com ele ativo, passa. (O "< 1 px" que esta
  linha trazia antes não tinha fonte; a tolerância real é 2,0 px.)

O parágrafo do §34.2 foi marcado como retratado no lugar, com remissão para cá.

### 35.3 O PONTO CEGO DE MEDIÇÃO — a suíte inteira passava verde por uma regressão de escala de tempo

Este é o achado mais transferível do bloco, e não é sobre este código.

A revisão da mudança (b) do §35.1 mediu que usar a moldura **sempre** degrada ωₙ.
Nenhum teste da suíte viu isso, e a razão é estrutural: **não existia critério
medindo ωₙ no caminho adimensional**. Havia só `2.6-adim[zeta]` — e **ζ é
invariante à escala do tempo**. Ou seja: o caminho que existe justamente para
dispensar a calibração media a única grandeza cega ao defeito que a escala de `t`
produz. Uma regressão de escala passava verde pela suíte inteira, 36 testes.

Foi acrescentado o diagnóstico `2.6-adim[wn_T]` (sem meta, sem `assert` — não há
limiar decidido; o objetivo é tornar o número visível em toda rodada). E, na
rodada seguinte, uma SEGUNDA linha, `2.6-adim[wn_T/sem-calib]` (**n=33**), porque
a primeira versão media **n=143** e **110 dessas passam pelo caminho FÍSICO**, que
não executa `_serie_normalizada`: o sinal ficava diluído ~7×, visível mas fraco
demais para alguém notar numa tabela de ~80 linhas.

**Prova de que funciona**, feita em cópia do repositório diferindo da árvore só em
`_COBERTURA_MIN_MOLDURA = 1.01` (a regressão reintroduzida), e **verificada de
forma independente por quem revisou**, rodando em vez de replicar:

| linha | produção | com a regressão | move |
|---|---|---|---|
| `2.6-adim[wn_T]` (corpus, **n=143**) | +1,04 p.p. (real 1,76%) | +1,82 p.p. (real 2,54%) | **+0,78 p.p.** |
| `2.6-adim[wn_T/sem-calib]` (**n=33**) | +0,63 p.p. (real 1,35%) | +6,43 p.p. (real 7,15%) | **+5,80 p.p.** |

**Duas razões diferentes, e a distinção importa.** O que mede a FORÇA do
diagnóstico é a razão dos **deslocamentos**: `5,80 / 0,78 = `**`7,44`** — a linha
restrita à população certa reage ~7× mais à mesma regressão, que é exatamente o
fator de diluição que o §35.3 existe para eliminar. A razão dos **MAPE reais** na
linha sem-calibração é outra coisa: `7,15 % / 1,35 % = `**`5,30`** — é quanto o
erro em si se multiplica quando a regressão entra. O "fator 5,3" que a review
anterior atribuiu ao deslocamento era esta segunda razão; ficam as duas
nomeadas, para ninguém ter de adivinhar qual é qual. Na linha certa a regressão
aparece como uma multiplicação por 5 do erro; na linha diluída, como
arredondamento.

**A lição, que é do §2.12 e não deste arquivo:** um critério pode existir, passar,
e ainda assim não cobrir nada — quando a grandeza que ele mede é *invariante* ao
modo de falha do caminho que ele deveria vigiar, ou quando a *população* em que
ele mede é majoritariamente de amostras que não passam por aquele caminho. As duas
falhas aconteceram aqui, na mesma linha, e a segunda só foi vista porque alguém
exigiu a prova de sensibilidade.

### 35.4 O erro de MÉTODO — achatar uma distribuição num escalar (§2.12, classe A)

Uma review pediu que `_COBERTURA_MIN_MOLDURA` subisse de 0,75 para o **"ponto de
empate ≈ 0,872"**, onde as duas referências de escala erram igual. A derivação é
correta: com `c` = cobertura e `m` = soma das margens do matplotlib, o erro de
escala é `|c(1+m) − 1|` pela extensão observada e `m` pela moldura, e os dois
empatam em **c\* = (1−m)/(1+m)**.

O que a medição mostrou é que **`c*` não é um número, é uma distribuição**. Como
`m` varia por amostra, `c*` varia de **0,7912 a 0,9563** (n=179), e **0,8716 é a
MEDIANA dela** — era daí que saía o "0,872". Medido com o limiar em 0,872:

- afeta **3 das 179** (coberturas 0,7713, 0,8505, 0,8514 — todas fopdt);
- **nenhum critério com meta se move** (`2.6-adim[zeta]` e as duas linhas de
  `wn_T` ficam idênticas dígito a dígito, nas 300 e nas 900: as três afetadas são
  de 1ª ordem e não entram em métrica de ωₙ/ζ);
- nas grandezas que elas movem (sem calibração, n=184; aceitas 75 para τ/T e 151
  para θ/T): τ/T p95 melhora 10,37% → 6,42%, |Δθ/T| p95 **piora** 0,0164 → 0,0181,
  medianas e `ordem_ok` (85,9%) inalterados.

**Sem ganho líquido medido.** A razão física: o empate modela só a **ESCALA**, e
trocar de referência move junto a **ORIGEM** de `t`. Quando a truncagem é à
direita — o regime do caso real — a origem observada já estava correta, e a troca
importa a margem esquerda como viés aditivo em θ.

E o próprio revisor foi além, retratando a sua ressalva: em **`sample_00828`**,
que tem `c < c*` e para a qual o modelo do empate **prevê melhora**, o τ medido
**PIORA de 4,9% para 19,1%**. O erro de escala não é sequer preditivo amostra a
amostra.

**Decisão: recuar, o limiar fica 0,75**, com o empate, a sua derivação e o seu
ponto cego (a origem) registrados no comentário do código.

**Registro na taxonomia do §2.12 do `PLANO.md`.** Isto é **classe A** — *a
grandeza usada para justificar o limiar não é a grandeza que decide o resultado* —
na forma específica de **achatar uma distribuição num escalar**. É o mesmo erro
que o IoU cometia no 2.1 (medir espessura de traço e chamar de acurácia), com
outra roupa: aqui a estatística de resumo de uma distribuição per-amostra foi
tratada como constante do sistema, e a decisão que ela justificava é tomada
amostra a amostra. **A forma correta de propor um limiar assim é medir a
distribuição da grandeza de decisão e o saldo antes/depois no corpus, nunca
derivar um ponto de equilíbrio médio e adotá-lo.** Vale para o TCC inteiro, não
para esta linha de código.

### 35.5 Estado da suíte ao fim do bloco

```
.venv/bin/python -m pytest tests/part2/  ->  2 failed, 34 passed  (556 s)
```

Rodada em PRIMEIRO PLANO, sem filtros, **depois** de todas as edições — incluindo
as duas do §35.7-2 —, de modo que o número acima é da árvore final e não de uma
intermediária. Instantâneo em `reports/part2_strata_pos_caso_real.md`.

As **duas falhas são PRÉ-EXISTENTES** e não são colateral deste bloco: `2.5` em
**0,885 (n=61)** e `2.9` em **0,797 (n=300)**, as duas de cobertura da calibração
(`identify/calibrate.py`). Conferidas por `diff` contra
`git show HEAD:reports/part2_strata.md`: **idênticas dígito a dígito** à linha de
base commitada. Não foram consertadas aqui — a classe D do §2.12 as mantém em
aberto por decisão própria.

**A referência da comparação é `git show HEAD:reports/part2_strata.md`**, não o
`reports/part2_strata_26adim.md` que o plano deste bloco mandava usar: aquele
arquivo é anterior à revisão de critérios do Ruling 53 (traz `2.1` ainda como IoU
com alvo ≥ 0,85, `2.2-piso` ainda vertical, `2.5` em 0,721), então diffar contra
ele mistura o efeito deste bloco com o da revisão de critérios de dois blocos
atrás. O HEAD é a última linha de base commitada e é a comparação que isola este
bloco.

O que se moveu em `reports/part2_strata.md`, e por quê:

| critério | antes | agora | causa |
|---|---|---|---|
| `2.1` | 0,800 / 1,699 px | 0,802 / 1,529 px | desambiguação multi-bloco (§35.1) |
| `2.2-piso` | 0,614 / 1,135 px | 0,615 / 1,137 px | ruído; a máscara verdadeira quase não tem multi-bloco |
| `2.7[traco=-]` | 0,680 px | 0,663 px | idem §35.1 |
| `2.6-adim-aceitas` | 141/300 (31 s/calib) | 143/300 (33 s/calib) | normalização de `t` recupera 2 amostras |
| `2.6-adim[zeta]` | +1,53 p.p. | **+0,89 p.p.** | nível de repouso + normalização de `t` |
| `2.6-adim[wn_T]` | *não existia* | +1,04 p.p. (n=143) | §35.3 |
| `2.6-adim[wn_T/sem-calib]` | *não existia* | +0,63 p.p. (n=33) | §35.3 |
| `2.6[K,tau,theta]`, `2.6-classico[*]` | — | ±0,2 p.p. | mesma causa: as duas mudanças de polilinha |
| `2.8` (latência por imagem) | 176 / 261 ms | 184 / 270 ms | jitter, alvo < 500 ms |
| `G3b.4` (latência do extrator clássico) | 13,4 / 36,5 ms | 13,3 / 33,9 ms | jitter, alvo < 200 ms |

As duas linhas de latência são do instantâneo da rodada completa
(`reports/part2_strata_pos_caso_real.md`); rodadas escopadas posteriores as
movem alguns ms por jitter de máquina, sem tocar em nenhum alvo.

A Parte 1 foi rodada porque este bloco tocou `dataset/generator.py`,
`dataset/randomize.py` e `tests/test_part1.py` (o estrato novo, abaixo):

```
.venv/bin/python -m pytest tests/test_part1.py tests/test_leakage.py -q
  ->  33 passed (130 s)
```

Nenhuma falha, inclusive nas duas guardas que o campo `has_reference_line` podia
quebrar (`test_meta_contract` e `test_1_4b`).

### 35.6 O estrato novo do gerador, e o que ele NÃO conserta

`dataset/generator.py` ganhou o argumento de render `reta_no_patamar`, que desenha
uma **reta de referência tracejada NO patamar** — o caso real corriqueiro
(*setpoint* marcado) que o sorteio de distratores só produzia por acaso, e que é o
caso difícil, porque o tracejado fragmenta em blocos por coluna. `RenderStyle`
ganhou o campo `has_reference_line` (default `False`, **não sorteado**), presente
em `to_meta()`, o que mantém de pé as duas guardas que o campo poderia quebrar:
`test_meta_contract` (`_RENDER_KEYS` continua **conjunto fechado** comparado com
`==`, só cresceu um membro) e a guarda anti-vazamento de `sample_style`.

**A perda de cauda sob reta coincidente (§34.5) CONTINUA NÃO CONSERTADA.** O
estrato torna o **objeto** gerável, não a conserta: não há tinta nas colunas
perdidas, então nenhuma regra de polilinha as recupera — é limitação da máscara, e
fechá-la exige retreinar a U-Net com o estrato, que é bloco próprio. O que este
bloco fez foi **tornar o dano contornável a jusante**: a normalização de `t` pela
moldura (§35.1b) faz o pipeline dar a resposta certa *apesar* da cauda perdida,
com cobertura de 0,617 — **no caso real, e só nele**.

> **CORREÇÃO (rodada pós-review final, §35.8.3).** Onde este parágrafo dizia que o
> estrato torna a perda de cauda "mensurável", estava **errado por antecipação**:
> media-se o objeto, não o fenômeno. A reta sintética **não** faz a U-Net perder a
> curva assentada — em 30 seeds o ramo da moldura não dispara uma única vez e a
> cobertura até **sobe**. Os números estão na §35.8.3. Nada aqui valida a
> normalização pela moldura; a validação dela continua sendo de `n = 1`.

### 35.7 O que fica ABERTO

1. **`identify/polyline.py` — a coluna logo depois de um vão largo.** Ela pode
   errar **~80 px mesmo com o reset ativo**, por empate de distância entre blocos
   resolvido pela ordem da lista (o `min` estável do Python escolhe o *topmost*).
   Peso maior do que parece: a Task 4 mediu que **"vão largo com objeto
   concorrente por perto" é o regime REAL deste código** — os ~38% de largura sem
   tinta do caso real são exatamente isso —, não hipótese de laboratório. E um
   erro grande numa coluna vira o `anterior` da coluna seguinte: há **risco de
   propagação, NÃO MEDIDO**. Item de acompanhamento com número a levantar.
2. **`tests/part2/test_part2.py` — o acoplamento invertido: CORRIGIDO, e agora
   COM GUARDA PERMANENTE — que cobre a invariante, não o arquivo.**
   O bloco `2.6-adim` inteiro ficava atrás de um `return` antecipado que depende
   da contagem `aceitas` do caminho **FÍSICO**. O acoplamento era invertido:
   quanto pior a calibração, mais o diagnóstico que existe justamente para cobrir
   a falta dela desaparecia — e desaparecia **sem** registrar "n insuficiente",
   ou seja, silencioso no pior momento possível, reabrindo o ponto cego do §35.3.
   O `return` virou `if/else`, o `assert` do 2.6 ficou condicionado a `pior is not
   None`, e nada mais mudou. Discriminação provada nas duas direções em cópia do
   repositório com `N_EVAL = 30` (`aceitas = 24 < 100`): com o `return`, **0**
   linhas `2.6-adim` no relatório; com o `if/else`, **3** —
   `2.6-adim[zeta]` e `2.6-adim[wn_T]` registrando "n insuficiente" em vez de
   sumirem, e `2.6-adim[wn_T/sem-calib]` medindo de verdade (−0,21 p.p., n=3).
   Comportamento na árvore de produção **inalterado** (`aceitas = 214`).
   O mesmo modo de falha existia **um nível abaixo**, e foi corrigido junto: o
   `record_p2("2.6-adim-aceitas", ...)` — a **única** linha do bloco adimensional
   que carrega o `n` e o tamanho do subconjunto sem calibração — estava dentro do
   `if aceitas_adim >= 100:` sem par no `else`, e sumia em silêncio com `n` baixo.
   Foi movido para fora do portão. Provado na mesma cópia: agora aparece como
   `16/30 (3 sem calibração)`, e o `3` bate com o `n=3` da linha `sem-calib`.
   **A guarda.** O padrão reincidiu duas vezes no mesmo arquivo no mesmo dia, as
   duas provadas fora da suíte e as duas desfazíveis sem que nada acusasse — e é
   o mesmo padrão que produziu o ponto cego do §35.3. Por isso ganhou teste
   permanente: `tests/part2/test_instrumentacao.py`, que sustenta UMA invariante —
   **nenhum critério declarado desaparece do relatório por causa de um portão de
   `n`; se o `n` for insuficiente, a linha aparece dizendo isso, com o `n` real.**
   Ele recorta por AST a região de relatório de `test_2_6_degradacao_vs_oraculo`
   (tudo depois do laço de medição) e a executa duas vezes com **acumuladores
   injetados**, `n` alto e `n` baixo, contra um `record_p2` dublê. Não carrega o
   modelo, não lê `data/`, não roda o pipeline: **2 passed em 0,02 s**.
   Discriminação provada em cópia, nos dois modos de falha reais deste bloco — com
   o `return` reintroduzido acusa os quatro ids do bloco adimensional; com o
   `2.6-adim-aceitas` devolvido para dentro do portão acusa aquele um; restaurada
   a cópia, diff vazio contra a árvore e volta a passar.
   **O que a guarda NÃO cobre, e é o que fica aberto:** ela vigia a *região de
   relatório* de UMA função (`test_2_6_degradacao_vs_oraculo`), que é onde as duas
   reincidências aconteceram. Os outros portões numéricos de `test_part2.py`
   (linhas 644, 671, 685, 761) foram conferidos um a um e **todos** registram em
   vez de sumir, mas essa conferência é de hoje e não tem teste que a mantenha;
   um portão novo em OUTRA função nasceria sem guarda. E nada aqui vigia o
   **conteúdo** dos critérios — só a presença deles.
3. **A guarda de round-trip `test_1_4b` nunca exercita `reta_no_patamar=True`.**
   Ela reconstrói o estilo **a partir da seed**, e `reta_no_patamar` é argumento
   de *render*: não sai da seed, então o round-trip sempre reconstruiria `False`.
   Cobrir exige (a) uma fixture com o estrato e (b) o teste ler
   `m["render"]["has_reference_line"]` e aplicá-lo ao estilo reconstruído, no
   mesmo padrão que ele já usa para `snr_db` (`tests/test_leakage.py:293`). O
   split OOD que criaria essa fixture **não foi gerado neste bloco**, então o item
   fica aberto tal como estava. Mitigação existente: `test_estrato_referencia.py`
   confere a chave, e o caminho padrão (todo o resto do corpus) continua coberto.
4. **Limpezas menores já catalogadas e não feitas** (fora do escopo, nenhuma com
   evidência de morder hoje): `_nivel_de_repouso` com série vazia; `np.median` em
   vez de `nanmedian`; `any(bbox_px)` duplicado; `bbox` invertido/degenerado zera
   a máscara e falha seguro, mas **silenciosamente**; `dentro.size == 0` em
   `polyline.py` é código morto.

5. **A folga de `_COBERTURA_MIN_MOLDURA = 0.75` só está examinada de UM lado.**
   Para baixo é confortável (caso real em 0,617). Para cima é de **2,1 p.p.**: o
   mínimo do corpus sintético é 0,7713 (`sample_00639`, fopdt, n=179). Uma amostra
   que caia entre 0,75 e 0,77 passa a usar a moldura **sem nunca ter sido
   examinada** — nenhuma existe hoje, e é o número mais frágil do bloco. O lado de
   cima do limiar mantém o comportamento pré-bloco, então o risco não examinado é
   só o dessa faixa estreita.

6. **Validação externa com `n = 1`.** Duas decisões de projeto — o estimador de
   repouso (`_N_REPOUSO = 5`) e o condicionamento da moldura — estão ancoradas em
   **uma única imagem real**. O corpus sintético diz que nenhuma das duas piora
   nada; ele **não** diz que elas generalizam para *outras* imagens reais, porque
   o modo de falha que as motivou (curva coincidente com reta de referência, θ = 0)
   é justamente o que o gerador não produzia. O `n=1` está escrito em toda linha
   que vem dela, mas a defesa vai querer mais de uma imagem antes de chamar isto
   de validação externa. É o item mais barato de fechar dos seis: são imagens, não
   código.

7. **`2.6-adim[wn_T/sem-calib]` continua sem meta, com `n = 33`.** O §35.3 prova
   que ele detecta uma regressão **grande** (fator 7,44 de deslocamento). Não há
   evidência de que detecte uma pequena, e `n = 33` não sustenta limiar. Decidir
   uma meta para ele — e portanto transformá-lo de diagnóstico em critério — é
   trabalho de bloco próprio.

8. **Calibrar o estrato `reta_no_patamar` até que ele reproduza a TRUNCAGEM.**
   Hoje ele simula o objeto (reta tracejada no patamar) e **não** o fenômeno
   (fusão da máscara → polilinha truncada): em 30 seeds a cobertura mediana é
   0,9403 e nenhuma amostra cruza `_COBERTURA_MIN_MOLDURA` (§35.8.3). Enquanto
   isso valer, o estrato **não** pode ser citado como validação da normalização
   pela moldura. Alavancas, todas em `dataset/generator.py` (hoje `#d62728`,
   1,5 pt, `zorder=1`, **abaixo** da curva): aproximar a cor da reta da cor da
   curva, engrossar o traço, e/ou subir a reta no `zorder` para **acima** da
   curva. **Critério de aceitação, objetivo:** *a cobertura mediana do estrato
   tem de cair abaixo de 0,75* — hoje 0,94. Só depois disso faz sentido gerar
   `data/ood_referencia` e medir `2.6-adim` ali, como o plano previa.
   **Não executado nesta rodada de propósito** — é trabalho novo, e mexer no
   gerador é mexer no rótulo: qualquer alavanca acima tem de ser reconferida
   contra a invariante de que `mask.png` é byte-idêntico entre
   `reta_no_patamar=False` e `True` na mesma seed. Enquanto não fechar, a
   grandeza que o estrato **de fato** degrada é a ordem, e essa passou a ser
   medida (`2.12-ordem`, §35.8.2).

9. **Ponto cego da guarda de planura: o corte profundo que RECOMEÇA plano.**
   `_PLANURA_MAX_FRAC` verifica que as primeiras colunas observadas estão
   paradas, que é a condição de que `_nivel_de_repouso` precisa. Ela **não**
   distingue "parado no repouso" de "parado no patamar assentado": um corte à
   esquerda profundo o bastante para que a série remanescente comece já
   assentada tem planura ~0 e passa. Medido em 82 séries determinísticas: ocorre
   3 vezes, sempre com cobertura ≤ 0,45 (o caso real é 0,617), com erro de ζ de
   0,49 %, 6,43 % e **20,33 %**. O segundo facet óbvio do mesmo invariante —
   exigir que o nível lido seja o extremo de `y` — foi testado e **não fecha** o
   buraco (contra-sinal −0,0000 nos três). Fechar exige distinguir repouso de
   patamar sem proxy geométrico, e isso é trabalho de bloco próprio; o
   §35.9.1 explica por que NÃO se fecha isso com mais um limiar de geometria.
   Nenhuma amostra de nenhuma das duas populações reais está nesse regime hoje.

### 35.8 Rodada pós-review final — três itens

A review final da branch aprovou **sem bloqueantes**. Esta rodada fecha os três
itens que ela motivou. Numeração: 35.8.1 é o achado C3 da review, 35.8.2 é a
resposta à sua Pergunta 3 (o próximo ponto cego) e 35.8.3 é a correção de
honestidade que a Pergunta 2 exigiu.

#### 35.8.1 C3 — a Task 4 se contradizia nos extremos, e agora VERIFICA o lado

> **REVERTIDO EM PARTE — leia a §35.9.1 antes de citar esta subseção.** O
> diagnóstico do defeito e a decisão de RECUSAR continuam válidos. O que caiu foi
> a CONDIÇÃO: `_FALTA_ESQ_MAX_FRAC = 0.15`, o proxy geométrico descrito abaixo,
> foi substituído pelo invariante direto de PLANURA das primeiras colunas. O
> limiar do proxy estava no lugar errado da curva de dano — admitia 68 % de erro
> em ζ logo antes de si. Os números do proxy ficam abaixo como registro do que
> foi feito e desfeito, não como descrição do código atual.

**O defeito.** As duas metades da Task 4 supunham lados opostos e nenhuma
verificava o seu. `_nivel_de_repouso` lê as **5 primeiras colunas observadas** e
supõe a curva parada ali; `_COBERTURA_MIN_MOLDURA` dispara em **qualquer**
truncagem, porque a condição é a cobertura TOTAL (`span_observado /
span_moldura`), que não distingue de que lado falta tinta. Numa truncagem à
**esquerda** — a U-Net perdendo o trecho plano inicial, plausível pelo mesmo modo
de falha do patamar no caso real — a origem passava a vir de `bbox_px[0]`,
correta, mas as 5 primeiras colunas observadas já estavam **na subida**: o defeito
de 12,6 % em ζ do §34.3, reintroduzido pelo outro remédio da mesma task. A
docstring escrevia a suposição ("quando a truncagem é à DIREITA, o regime do caso
real") e o código não a asseverava.

**A correção.** `identify/pipeline.py` ganhou `_FALTA_ESQ_MAX_FRAC = 0.15` e
condiciona pelo **lado**:

| regime | condição verificada | saída |
|---|---|---|
| sem truncagem | cobertura ≥ 0,75 | extensão e origem OBSERVADAS (como antes) |
| truncagem à DIREITA | cobertura < 0,75 **e** falta à esquerda ≤ 0,15 | escala e origem da MOLDURA (o caso real) |
| truncagem à ESQUERDA ou simétrica | cobertura < 0,75 **e** falta à esquerda > 0,15 | **RECUSA** — `(None, None)`, e o bloco `dimensionless` sai vazio |

**Por que 0,15 (medido, população escrita).** O deslocamento à esquerda não é zero
nem sem truncagem: a moldura inclui a margem do matplotlib. Medido nas **300
primeiras de `data/test`** com a cadeia de produção (U-Net + calibração +
polilinha recortada à moldura), **n = 299** com moldura válida e polilinha ≥ 10
pontos:

| grandeza | mín. | mediana | p95 | máx. |
|---|---|---|---|---|
| cobertura (n=299) | 0,8388 | 0,9315 | — | 0,9756 |
| `(x[0]−bbox[0])/largura` (n=299) | 0,0024 | 0,0335 | 0,0553 | **0,1227** |
| `(x[0]−bbox[0])/largura`, sem calibração (n=60) | 0,0024 | 0,0301 | 0,0559 | 0,0569 |

Abaixo de 0,75: **0/299** — o ramo continua sem disparar no sintético, e o
comportamento do corpus é bit a bit o anterior. 0,15 fica **2,2 p.p. acima** do
maior deslocamento de margem observado. No caso real o deslocamento é
`1/746 = 0,0013` (`bbox=(75,39,821,503)`, `x[0]=76`, `x[-1]=536`), três ordens de
grandeza abaixo do limiar: o regime da direita continua valendo lá.

**Por que RECUSAR, e não cair no comportamento anterior — a escolha foi medida.**
27 séries de 2ª ordem determinísticas (ζ ∈ {0,3; 0,5; 0,7} × ωₙ ∈ {1; 2; 4} rad/s
× corte de 30/40/50 % da janela pela esquerda), rasterizadas na moldura do caso
real, com o nível de repouso lido nas 5 primeiras colunas **já na subida**:

| saída candidata | ajustes que convergem | MAPE de ζ mediano | máx. |
|---|---|---|---|
| trocar pela moldura (comportamento atual) | 8/27 | **27,6 %** | 80,0 % |
| manter extensão observada | 6/27 | **14,2 %** | 51,1 % |
| recusar (escolhida) | — | sem número | — |

Qualquer das duas alternativas é **pior que o defeito de 12,6 % que a Task 4
corrigiu** e muito além da tolerância de 5 % do caso real, e as duas falham em
converger na maioria das amostras. Um número errado que ninguém distingue de um
certo é pior que nenhum, e o contrato do `dimensionless` já sabe sair vazio
(`_vazio_adimensional`) — critério 2.11 honrado pela estrutura.

**Teste, com discriminação nas duas direções.**
`tests/part2/test_truncagem_lateral.py`, 6 testes, **0,8 s**, máscaras sintéticas
determinísticas passando pelo `mask_to_polyline` de produção — sem rede, sem
`data/`, porque o defeito é geométrico. Cobre os três regimes da tabela, mais a
geometria do caso real (n=1, sem carregar a imagem) e o piso do limiar contra o
maior deslocamento de margem medido. Contra uma cópia em `/tmp` com a condição de
lado removida (o código anterior à correção): **2 failed, 4 passed** — falham
exatamente `test_truncagem_a_esquerda_recusa_a_serie` e
`test_truncagem_simetrica_recusa_a_serie`. Na árvore corrigida: **6 passed**.

#### 35.8.2 O próximo ponto cego: θ (a ORIGEM de `t`) e o acerto de ORDEM

**O argumento.** O §35.3 fechou a **escala** de `t` acrescentando
`2.6-adim[wn_T]` e `2.6-adim[wn_T/sem-calib]`. Mas `_serie_normalizada` produz
**dois** parâmetros de referência de tempo — escala **e origem** — e a suíte
vigiava só um. Quem pega a origem é **θ**: ζ é invariante às duas, ωₙ·T só à
escala. E essa origem já valeu ~25 % de erro em θ (fix round 1, B1), achada por
**revisor humano, não por teste**. O segundo buraco: **nenhum** `record_p2` da
suíte media acerto de **ordem** — ela entrava só como *filtro* de aceitação, que é
a forma clássica do ponto cego (uma regressão de ordem não piora mediana nenhuma:
ela **encolhe** a amostra, e as que somem são as difíceis, então as medianas que
restam até melhoram).

**O que entrou** em `tests/part2/test_part2.py`, no molde exato do par de ωₙ —
diagnóstico **sem meta de aprovação**, população escrita em toda linha, e nenhum
registro atrás de portão de `n` sem par no `else`:

| id | medido na árvore corrigida (300 primeiras de `data/test`) |
|---|---|
| `2.6-adim[theta_T]` | **+0,13 p.p.** (oráculo 0,03 %, real 0,16 %, n=267) |
| `2.6-adim[theta_T/sem-calib]` | **+0,21 p.p.** (oráculo 0,04 %, real 0,25 %, n=53) |
| `2.12-ordem` | **91,3 %** (274/300, n=300) |
| `2.12-ordem[sem-calib]` | **90,2 %** (55/61, n=61) |

Métrica de θ: **NMAE sobre a janela** (`|Δθ|/T`, ×100), a convenção da Parte 1 —
não MAPE, porque θ pode ser 0 e a razão relativa explode. `theta_T` já é θ/T,
então a diferença absoluta é o erro em pontos percentuais **de T**.

**A guarda de instrumentação acompanhou.** `tests/part2/test_instrumentacao.py`
**quebrou ruidosamente** com `NameError: name 'aceitas_thT_adim' is not defined`
ao recortar a região de relatório — que é exatamente o modo de falha previsto na
docstring dela ("ancora na forma da função"). O dublê `_acumuladores` ganhou os
oito acumuladores novos; nenhuma lógica de guarda mudou. Depois: **2 passed**, e
agora são as **quatro** linhas novas que ela prova não sumirem com `n` baixo.

**Prova de discriminação — e o achado que veio junto.** O teste que a review
propôs (reverter `origem = bbox_px[0]` para `x[0]` e ver a linha de θ mexer)
**não move dígito nenhum**:

| configuração | `2.6-adim[theta_T]` | `.../sem-calib` |
|---|---|---|
| produção | +0,13 (n=267) | +0,21 (n=53) |
| origem revertida para `x[0]` | +0,13 | +0,21 |

**O motivo é estrutural e vale registrar: no corpus sintético o ramo da moldura
NUNCA dispara** (cobertura mínima 0,8388 contra limiar 0,75), então
`origem = bbox_px[0]` é código que o corpus **não executa**. A linha de θ não é
cega — o corpus é que não tem o que mostrar ali. É a mesma constatação da §35.8.3
por outro caminho, e reforça o item 6 do §35.7 (validação externa de `n = 1`).

Com o ramo **vivo** (moldura incondicional, isolando a origem), a linha responde:

| configuração | `wn_T` | `wn_T/sem-calib` | `theta_T` | `theta_T/sem-calib` |
|---|---|---|---|---|
| produção | +1,04 | +0,63 | +0,13 | +0,21 |
| moldura incondicional, origem `bbox[0]` | +1,82 | +6,43 | +0,19 | **+1,76** |
| moldura incondicional, origem `x[0]` (defeito B1) | +1,82 | +6,43 | +0,18 | **+1,23** |
| origem deslocada +3,35 % da moldura | +1,04 | +0,63 | +0,21 | **+3,24** |

Leitura: as linhas de ωₙ **não distinguem** origem certa de origem errada (+1,82 e
+6,43 nas duas), e as de θ distinguem — 0,53 p.p. de diferença na linha sensível.
E numa regressão de origem pura (+3,35 % da largura da moldura, que é a **mediana
medida** do deslocamento no corpus) a linha `theta_T/sem-calib` salta de **+0,21
para +3,24 p.p., fator 15**, com ζ e ωₙ inalterados dígito a dígito. O par
corpus/sem-calib repete a lição do §35.3: a linha do corpus mexe 0,08 p.p.
(diluída por 214 amostras que passam pelo caminho físico), a das sem calibração
mexe 3,03 p.p.

**Controle positivo da linha de ordem** (que as configurações acima não movem, e
com razão — ordem não depende de escala nem de origem): mesmo corpus, trocando a
U-Net pelo extrator clássico, `2.12-ordem` cai de **91,3 % (274/300) para 78,3 %
(235/300)** e `2.12-ordem[sem-calib]` de 90,2 % para 80,3 %. A linha enxerga uma
degradação de ordem de 13 p.p. que **nenhum critério do relatório registrava**.

#### 35.8.3 O estrato simula o OBJETO, não o FENÔMENO — e o que ele degrada de verdade

Este item é de honestidade do registro, e desmente por medição uma frase que
estava a caminho do TCC.

**1. O estrato NÃO valida a normalização pela moldura.** A review mediu 30 seeds
com e sem a reta de referência, rodando U-Net, calibração e polilinha, e olhou a
**cobertura**, que é a grandeza que decide o ramo da Task 4:

| corpus | mín. | mediana | abaixo de 0,75 |
|---|---|---|---|
| sem a reta (n=27 com moldura) | 0,8924 | 0,9281 | **0/27** |
| com a reta do estrato (n=27) | 0,9045 | 0,9403 | **0/27** |
| caso real (n=1) | **0,6166** | — | **1/1** |

O ramo `_COBERTURA_MIN_MOLDURA` **não dispara uma única vez** no estrato novo, e a
cobertura até **sobe** de leve com a reta. A reta sintética não faz a U-Net perder
a curva assentada: no caso real faltam 285 dos 746 px de moldura à direita. O
estrato simula o **objeto** (reta tracejada no patamar) e não o **fenômeno**
(fusão da máscara → polilinha truncada), que é o que a correção da Task 4
endereça. A §35.6 foi corrigida onde dizia o contrário.

**2. A normalização pela moldura tem validação externa de `n = 1`.** A tabela
acima é o número que sustenta a frase: o único caso em que o ramo dispara é a
imagem real. Nada no corpus sintético — nem o estrato novo — o exercita. Ver
§35.7-6, e a §35.8.2 chega à mesma conclusão pela via do diagnóstico.

**3. O estrato degrada de verdade, mas OUTRA grandeza — e ela agora é medida.**
Nas mesmas 30 seeds, a reta faz **uma** amostra trocar a ordem certa pela errada:
**seed 21, `second` → `fopdt`, com a calibração OK e cobertura 0,937**. Numa
sondagem anterior a **seed 2024** fez o mesmo, e ali as colunas multi-bloco
saltaram de **0 para 30**. É estrato **legítimo**, com degradação **reprodutível**
— só que a grandeza que ele degrada é a **ordem**, que até esta rodada não tinha
`record_p2` nenhum medindo e agora tem (`2.12-ordem`, §35.8.2). O caminho de
fechar o estrato como validação da moldura está no §35.7-8, com critério de
aceitação objetivo: **cobertura mediana abaixo de 0,75**, hoje 0,94.

### 35.9 Rodada final — a reversão do limiar, e o fechamento da instrumentação

A re-review aprovou a rodada pós-review sem bloqueantes e validou a substituição
da prova do θ. Motivou quatro itens; o primeiro **reverte uma decisão já
arquivada**, e é o mais importante dos quatro.

#### 35.9.1 REVERSÃO — `_FALTA_ESQ_MAX_FRAC = 0.15` estava no lugar errado da curva de dano

**O que estava escrito, e por que era o argumento errado.** A §35.8.1 condicionou
a troca de referência de tempo por um **proxy geométrico**: só trocar quando
`(x[0] − bbox_px[0]) / largura da moldura ≤ 0,15`. A ressalva registrada na época
dizia que a fragilidade era "a mesma do 0,75 — falta de dados na faixa
intermediária". **Isso foi arquivado como limite epistêmico e estava errado.** A
diferença entre os dois limiares é de natureza, não de grau:

- o **0,75** fica ENTRE duas populações medidas (0,7713 do corpus, 0,617 do caso
  real) e o erro cresce suavemente ao redor;
- o **0,15** tinha 68 % de erro de ζ de um lado e **nenhuma amostra** do outro.

Não era falta de dados na faixa intermediária: era limiar posto no lugar errado.
Medido com corte à direita de 30-40 %, variando o corte à esquerda:

| falta à esquerda | cobertura | decisão do 0,15 | erro de ζ |
|---|---|---|---|
| 0,0402 | 0,64 | aceita | 0,1 % |
| 0,0590 | 0,62 | aceita | 4,3 % |
| 0,0858 | 0,60 | aceita | 19,6 % |
| 0,1139 | 0,57 | aceita | 45,7 % |
| **0,1327** | 0,55 | **aceita** | **68,0 %** |
| 0,1501 | 0,53 | RECUSA | — |

Imediatamente **antes** do limiar, 68 % — **cinco vezes** o defeito de 12,6 % que
a Task 4 corrigiu, e pior que os 27,6 % que motivaram a recusa do outro lado.

**O erro de MÉTODO, que vale mais que o número — e é o SEGUNDO caso dele neste
bloco.** O 0,15 veio do deslocamento máximo de margem do matplotlib (0,1227,
n=299), medido num corpus cuja **cobertura mínima é 0,8388**: nenhuma daquelas
299 amostras entra neste ramo. A constante foi calibrada sobre uma estatística de
uma população que **não é a que ela governa** — protegendo contra um falso
positivo que ali não pode ocorrer, enquanto o falso negativo, que ocorre, chegava
a 68 %. É o mesmo erro que o §35.3 já tinha catalogado na medição: o diagnóstico
de ωₙ nasceu diluído 7× por ser medido no corpus (n=143) em vez do subconjunto
que o caminho de fato serve (n=33). **Duas vezes no mesmo bloco, nas duas pontas
— na constante e no diagnóstico.** É o padrão que a próxima pessoa precisa
reconhecer: *antes de calibrar um limiar, pergunte se a população em que ele foi
medido é a população em que ele vai decidir.*

**O conserto: proxy fora, invariante direto dentro.** `_PLANURA_MAX_FRAC = 0.03`
exige que as `_N_REPOUSO` primeiras colunas observadas sejam **planas** —
dispersão de `y` ali ≤ 3 % da faixa total de `y`. É literalmente a condição que
`_nivel_de_repouso` precisa, é independente da margem, e — ao contrário do
proxy — é **transferível entre as populações**, porque a truncagem à direita não
altera as primeiras colunas. Apertar o proxy para 0,02 (a outra proposta) só
moveria o corte na mesma curva ruim e manteria o proxy.

**Os dois lados do novo limiar, medidos.**

*(a) Teto de dano* — 82 séries de 2ª ordem determinísticas (ζ ∈ {0,3; 0,5; 0,7} ×
ωₙ ∈ {1; 2; 4} rad/s × corte à esquerda de 0 a 28 % em passos de 2 %, corte à
direita fixo em 35 %):

| decisão | n | erro de ζ mediano | p90 | máx. |
|---|---|---|---|---|
| **aceitas** (planura ≤ 0,03) | 21 | **0,49 %** | 6,43 % | 20,33 %¹ |
| recusadas (planura > 0,03) | 61 | **28,77 %** | — | 99,86 % |

¹ o único ponto acima de 13 % entre as aceitas é o ponto cego do §35.7-9.

A varredura fina em ζ=0,5 / ωₙ=2 é monótona e é a curva de dano indexada pela
grandeza que o código usa: planura 0,0064 → 0,08 %; 0,0423 → 4,25 %; 0,0613 →
13,09 %; 0,0860 → 27,34 %; 0,1171 → 45,44 %; 0,1587 → 67,82 %.

*(b) Piso de legitimidade* — a planura não é zero nem sem truncagem nenhuma (com
θ = 0 e ωₙ alto a curva já se move dentro de 5 colunas). Medida nas 300 primeiras
de `data/test` com a cadeia de produção, **n = 299**: mediana 0,0044, p95 0,0316,
p99 0,0485, máximo 0,0722; sem calibração (n=60) máximo 0,0483. Acima de 0,03:
**19/299 = 6,4 %** (e 5/60 = 8,3 % sem calibração) — ou seja, 0,03 fica em ~p94 e
não recusa uma série pela movimentação normal de início de curva. Com 0,02 seriam
12,7 %, e o p90 do dano nem melhora (5,36 % contra 6,43 %). Caso real (n=1):
planura **0,0039**, ~8× de folga, e é a única amostra que de fato usa este ramo.

**Nenhuma decisão mudou nas duas populações reais.** Corpus: cobertura mínima
0,8388 > 0,75, o ramo continua sem disparar, e a rodada completa da suíte
confirmou — `reports/part2_strata.md` não moveu nenhum critério de acurácia. Caso
real: cobertura 0,617 e planura 0,0039, continua no ramo da moldura, e
`test_caso_real.py` passa.

**Ponto cego, medido e registrado (§35.7-9).** Um corte à esquerda tão profundo
que a série remanescente **recomeça plana** — já no patamar assentado — tem
planura ~0 e passa. Ocorre 3 vezes nas 82 séries, todas com cobertura ≤ 0,45 (o
caso real é 0,617), com erro de ζ de 0,49 %, 6,43 % e 20,33 %. Testei um segundo
facet do mesmo invariante — exigir que o nível lido seja o **extremo** de `y`
(desvio de um sinal só) — e **não fecha**: nesses três casos o desvio já é de um
sinal só (contra-sinal −0,0000). Fica aberto, com o número escrito, em vez de
fechado com um proxy novo.

**Discriminação, três direções.** `tests/part2/test_truncagem_lateral.py` (6
testes, 0,8 s, máscaras determinísticas sem rede) falha contra o código original
do C3 (sem guarda nenhuma) — 2 failed, 4 passed — e falha **também contra o proxy
0,15 revertido** — 2 failed, 4 passed —, porque o caso de 68 % que o proxy
aceitava é justamente o que ele agora recusa. Na árvore corrigida, 6 passed. O
teste `test_limiar_da_esquerda_acima_da_margem_do_matplotlib`, cuja premissa não
se sustentava, foi substituído por
`test_limiar_de_planura_esta_entre_as_duas_populacoes_medidas`, que assevera os
dois lados na grandeza que o código de fato usa.

#### 35.9.2 A guarda de instrumentação passou a cobrir o que vigia

Dois buracos da rodada anterior, os dois em `tests/part2/test_instrumentacao.py`:

1. **As linhas novas entraram só no teste de PRESENÇA.**
   `test_linha_de_n_insuficiente_carrega_o_n_real` continuava cobrando o `n` real
   apenas dos ids antigos; que as novas o carregavam foi **conferido à mão** — que
   é exatamente o padrão que esta guarda existe para fechar. Agora as seis linhas
   com portão de `n` estão na lista, e as quatro linhas `sem-calib` são cobradas
   em laço. Discriminação: apagando só o `n=` da linha de θ numa cópia, a guarda
   acusa — `2.6-adim[theta_T] não diz o `n` real com n baixo` (antes: passava).
2. **O piso `len(ids) >= 4` não crescia com os ids.** Com 12 critérios
   declarados, oito poderiam ser apagados sem a guarda reclamar. Virou
   `N_CRITERIOS_LITERAIS = 12`, constante de módulo, atualizada à mão quando um
   critério entra ou sai de propósito. Discriminação: apagando o par do
   `K_yrange` numa cópia, a guarda acusa `declara 10 critérios literais, menos que
   os 12 correntes` (antes: 10 ≥ 4, passava em silêncio).

#### 35.9.3 O terceiro eixo: `2.6-adim[K_yrange]`

Depois da **escala** de `t` (§35.3) e da **origem** de `t` (§35.8.2), o terceiro
eixo do mesmo padrão — e os dois primeiros já morderam. `K_yrange` é a única das
seis grandezas do bloco `dimensionless` sensível à escala de **y**, e nada a
vigiava: `2.6[K]` é do caminho FÍSICO, que só existe quando a calibração fecha.
Entrou no molde dos irmãos (corpus + sem-calibração, diagnóstico sem meta,
população em toda linha, sob a guarda). Verdade: `K_alvo / ptp(série verdadeira)`,
as duas do meta. Métrica: MAPE relativo — `K_yrange` de um degrau vale
~1/(1+overshoot) e não passa por zero, ao contrário de θ, que por isso usa NMAE.

Com ele, o bloco `dimensionless` do PLANO §1.7 tem **quatro** das seis grandezas
sob diagnóstico (`zeta`, `wn_T`, `theta_T`, `K_yrange`); ficam de fora `tau_T` e
`theta_tau`, que são redundantes com as medidas (τ/T e θ/τ derivam das mesmas
escalas já vigiadas).

---

## 36. Ruling 57 — o critério 2.9: quatro hipóteses testadas, três refutadas, e a ordem dos próximos passos

O 2.9 (cobertura da calibração, alvo ≥ 90%) está reprovado em **0,797** há vários
blocos. Este bloco investigou por quê, com três imagens reais de matplotlib como
casos de teste externos. O resultado útil é sobretudo **negativo**: quase tudo o
que parecia promissor não sobrevive à medição, e vale estar escrito para que
ninguém repita as mesmas quatro investigações.

### 36.1 RETRATAÇÃO — "o gargalo é recall do OCR" saiu de um teste circular

A conclusão registrada no fim do bloco anterior — que completar os rótulos não
lidos fazia o gate aprovar os seis eixos das três imagens — **está errada como
evidência**. No teste, os pixels dos rótulos faltantes foram gerados a partir do
PRÓPRIO ajuste afim, ou seja caíam perfeitamente sobre a reta. O gate passava por
construção; o teste não tinha como falhar.

Refeito de forma não circular (completar um valor faltante SÓ quando existe blob
de texto no pixel previsto, tolerância de 6 px): **716 → 715 amostras**. Efeito
nulo. A retratação não anula o resto da análise, mas troca o mecanismo: mais
rótulos ajudam, e não por preencher lacunas — ver §36.2.

Lição de método, a terceira deste bloco do mesmo tipo (ver §35.4 e §35.9.1):
quando o experimento constrói a evidência a partir do modelo que ele deveria
testar, o resultado é tautologia. O sinal de alerta aqui era simples e foi
ignorado: o teste não tinha um caso em que pudesse falhar.

### 36.2 A anatomia real, em três camadas medidas

Corpus `data/test`, n=895 com moldura detectada.

**Camada 1 — segmentação de blobs de texto: NÃO é o problema.**
`_text_blobs` acha o que precisa. Perda total de 56 rótulos no eixo x e 13 no y
sobre 895 amostras; recall de segmentação mediano de 100%. Descartada como
suspeita.

**Camada 2 — leitura (Tesseract + `_NUM_RE`): perda real, mas não é o gargalo.**
Recall no nível do rótulo: **86,9% em x e 80,0% em y**. O padrão é
contraintuitivo e diagnóstico — o recall SOBE com o comprimento da string:

| forma do número | exemplo | lidos/total | recall |
|---|---|---|---|
| decimal, 1 casa | `0.2` | 2133/2704 | **78,9%** |
| inteiro, 1 dígito | `5` | 2725/3261 | 83,6% |
| decimal, 2 casas | `0.25` | 1010/1190 | 84,9% |
| decimal, 3 casas | `0.125` | 75/84 | 89,3% |
| inteiro, 2 dígitos | `10` | 1017/1124 | 90,5% |
| inteiro, 3 dígitos | `100` | 203/217 | **93,5%** |

É comportamento conhecido do Tesseract com `--psm 7`: o motor LSTM é de LINHA de
texto e um numeral isolado e curto quase não lhe dá contexto. Rótulo de eixo é
exatamente o pior caso dele. Note que isto não é ajustável por parâmetro — é a
natureza do modelo escolhido.

**Camada 3 — EMPARELHAMENTO: é aqui que o gate morre.**
Comparando o pixel atribuído a cada rótulo lido com o pixel verdadeiro do tick
(do meta):

| eixo/forma | n | mediana do erro | **desvio padrão** |
|---|---|---|---|
| x / inteiro 1 díg | 1324 | +0,58 px | **26,54 px** |
| x / decimal 1 casa | 903 | +0,46 px | **10,94 px** |
| x / inteiro 2 díg | 879 | +0,57 px | **19,87 px** |
| x / decimal 2 casas | 439 | +0,37 px | 0,73 px |
| x / inteiro 3 díg | 203 | +0,66 px | 0,83 px |
| y / inteiro 1 díg | 1428 | +0,80 px | **25,54 px** |
| y / decimal 2 casas | 571 | +0,81 px | 0,47 px |

Mediana excelente (sub-pixel) e desvio de 10 a 27 px nas formas CURTAS. Isso não
é imprecisão de centroide: é **valor lido corretamente e atribuído ao pixel de
OUTRO rótulo**. Afeta 5,7% dos rótulos de x e 1,4% dos de y com |erro| > 2 px.

Consequência sobre o gate: com 4 ou 5 pares e um deles grosseiramente
deslocado, o RANSAC não tem massa para identificá-lo como outlier, e
`_equiespacados` lê o conjunto resultante como não equiespaçado. É por isso que
mais rótulos ajudam — dão ao RANSAC dados para **rejeitar o par ruim** —, e não
porque preenchem lacunas.

### 36.3 As três hipóteses refutadas

**(a) Melhorar a PRECISÃO da detecção de ticks.** A detecção tem precisão
péssima: 5 ticks verdadeiros, 10 detectados, 7 falsos; e nas imagens reais 68 a
70 detecções, causadas por grade pontilhada coincidente com o spine (o corpus
nunca reproduz isso porque `y_margin_lo` desloca o `ylim`). Implementado um
discriminador que anula estrutura paralela ao eixo: espúrios 7 → 3 no corpus e
70 → 5 nas reais. **`ok`, `ok_x`, `ok_y` e falsos positivos ficaram dígito a
dígito iguais.** Os espúrios já eram inofensivos — os recortes que geram são
descartados pelo `_NUM_RE`. Revertido por ter custo (recall p10 40% → 30,5%) e
benefício zero. A função fica documentada em `_sem_paralela`, sem uso, como
registro.

**(b) Ancorar o rótulo lido na marca de tick mais próxima.** Parecia o conserto
direto do defeito da camada 3. Medido:

| variante | ok | ok_x | ok_y | falsos+ |
|---|---|---|---|---|
| atual | **79,6%** | 87,7% | 88,4% | 55 (7,7%) |
| snap, tol 4 px | 73,6% | 85,0% | 84,3% | 58 (8,8%) |
| snap, tol 8 px | 67,6% | 80,6% | 81,3% | 73 (12,0%) |
| snap, tol 15 px | 66,8% | 80,4% | 80,7% | 73 (12,1%) |
| filtra, tol 8 px | 77,6% | 87,2% | 85,0% | **48 (6,9%)** |
| filtra, tol 15 px | 77,1% | 86,9% | 85,3% | 53 (7,6%) |

Encostar (`snap`) piora nos dois eixos, e monotonamente com a tolerância. Causa:
com 7 marcas espúrias por amostra, o snap desloca rótulos BONS para marcas
falsas. Não dá para usar a detecção de ticks como referência de POSIÇÃO enquanto
a precisão dela for essa — e (a) mostrou que consertar a precisão não paga.

**Achado lateral que vale seguir, mas para OUTRO critério.** Descartar o par sem
marca por perto (`filtra`, tol 8 px) é a ÚNICA variante testada em todo o bloco
que REDUZ falso positivo: 55 → 48, de 7,7% para 6,9% das aceitas. Custa 18
amostras de cobertura (716 → 698), então é ruim para o 2.9. Mas o critério **2.5**
(rejeições corretas, ≥ 90%, hoje **0,885 ❌**) mede precisamente a precisão da
rejeição, e está reprovado por pouco. Vale medir o 2.5 sob `filtra` antes de
descartar a variante — é a primeira pista de melhora daquele critério que
aparece neste bloco, e o trade cobertura-por-precisão que aqui é ruim pode ser
exatamente o desejado lá. Registrado como item próprio porque a tentação é
descartar a variante pelo veredito do 2.9 e perder a pista do 2.5.

**(c) Completar a rede de valores.** Refutado em §36.1.

### 36.4 A hipótese que sobreviveu, e por que é a mais forte

**Prior de origem: o primeiro tick do eixo x é ZERO em 900 de 900 amostras** — e
nas três imagens reais também.

Isso reformula o problema. Hoje a calibração exige QUATRO pares mutuamente
consistentes, e é a exigência de consistência que o emparelhamento errado da
camada 3 destrói. Com a origem conhecida, **um** par lido determina a escala. Um
par ruim deixa de contaminar uma consistência que não é mais exigida.

Prior secundário, aplicável ao mundo real mas NÃO validável no corpus: os
locators do matplotlib com defaults produzem passo `1, 2, 2.5, 5 × 10^k`. Medido
no gerador, só **41,3%** dos eixos x têm passo "bonito", porque ele sorteia
contagens de bins; as três imagens reais têm passo 2, 1 e 0,5 — todos bonitos.
É mais uma divergência gerador-vs-real, da mesma família das do §35: o prior é
forte no alvo e o corpus não permite medi-lo.

### 36.5 PRÓXIMOS PASSOS, em ordem de valor medido

**1. Prior de origem + um rótulo (fazer primeiro).**
Trocar "quatro pares consistentes" por "um par + âncora conhecida" no eixo x.
Ataca a camada 3 pela raiz em vez de tentar consertar o emparelhamento. Precisa
de guarda própria — a candidata natural é redundância entre pares lidos,
aceitando a escala que mais deles corroborarem, em vez de exigir que todos
concordem. Critério de aceitação a fixar ANTES de implementar, e tem de incluir
falso positivo: qualquer coisa que suba cobertura sem medir falso positivo
repete o erro de §36.3(b). Teto esperado bem acima dos 84,8% que a varredura de
24 variantes do `_equiespacados` mediu.

**2. Classificador de dígitos treinado no gerador.**
Ataca a camada 2 (80-87%) e, por consequência, dá massa ao RANSAC para rejeitar
o par ruim da camada 3. O gerador CONHECE o texto exato que renderiza, então os
rótulos de treino são gratuitos e exatos — a mesma jogada que já funcionou no
Estágio A. Substitui o Tesseract no ponto em que ele é estruturalmente ruim
(numeral curto isolado), sem depender de parâmetro nenhum dele. Custo: um treino
a mais. Fazer DEPOIS do item 1, porque o item 1 pode tornar o recall menos
crítico e mudar o alvo.

**3. Prior de passo bonito.**
Restringe as leituras a uma grade candidata pequena e permite rejeitar leitura
inconsistente sem exigir consistência entre pares. **Bloqueado pelo gerador**:
com 41,3% de passo bonito no corpus não há como validar. Exige primeiro alinhar
os locators do gerador ao comportamento default do matplotlib — o que, note, é
um conserto de FIDELIDADE do corpus e não do pipeline, e provavelmente melhora
outras coisas junto.

**4. Medir o critério 2.5 sob a variante `filtra` (barato, e independente dos
itens acima).**
Único caminho medido neste bloco que reduz falso positivo (55 → 48). Ruim para o
2.9, possivelmente bom para o 2.5, que está reprovado em 0,885 e cuja métrica é
justamente a precisão da rejeição. Custo: uma rodada da suíte. Ver §36.3(b).

**5. O que NÃO fazer: continuar ajustando `_equiespacados`.**
Já varridas 24 variantes (limiar 0,05 a 0,15 × unidade por `min(d)` ou por rede
de mínimos quadrados × checar pixel+valor ou só valor). Toda variante que sobe
cobertura sobe falso positivo, ~1,3 falso positivo por ponto percentual, e o
máximo alcançável é **84,8% a 10,6% de falso positivo** — abaixo da meta de 90%.
O gate não está mal calibrado; está mal condicionado por receber poucos pontos, e
isso se resolve acima dele.

### 36.6 Limite que atravessa tudo isto

Validação externa é **n=3**. As três imagens já expuseram defeitos que 900
amostras sintéticas nunca produziram (§35 e §36.3(a)), o que é o argumento a
favor de validar fora do próprio gerador — e simultaneamente o limite: várias
decisões deste bloco se apoiam nessas três. Ampliar o acervo externo continua
sendo o passo de maior alavancagem para o TCC inteiro, e não só para o 2.9.

---

## 37. Ruling 58 — a camada que faltava: os blobs de rótulo fundiam com as MARCAS de tick

Executando o item 1 do §36.5 (prior de origem), a porta de verificação do
prior reprovou — e a investigação do porquê encontrou um defeito uma camada
ABAIXO de tudo que o §36 mediu. O conserto sobe cobertura e desce falso
positivo ao mesmo tempo, o que nenhuma das quatro hipóteses do §36 conseguiu.

### 37.1 Como o defeito apareceu

O critério de aceitação do item 1 foi fixado ANTES de implementar (exigência
do próprio §36.5) com quatro portas, a primeira delas:

> O blob mais externo da faixa de rótulos tem de coincidir com o pixel
> verdadeiro do primeiro tick a <= 3 px em >= 99% das amostras. Se não
> coincidir, a âncora não existe e o item cai ANTES de qualquer implementação.

Medido, n=895: **76,1% no eixo x e 75,5% no y**. Reprovada.

O prior em si é sólido — verificado direto no `meta.ticks`, o primeiro tick
do eixo x vale 0 em **900/900**, e (achado novo, que o §36.4 não tinha visto)
o menor tick do eixo y vale 0 em **896/900 = 99,56%**. O que falhava era
LOCALIZAR o pixel desse tick. Nos piores casos havia 1 ou 2 blobs no MEIO do
eixo e nenhum rótulo detectado; numa amostra o OCR leu literalmente
`"1020304050"` — os cinco rótulos colados num texto só.

### 37.2 A causa

A faixa de rótulos do eixo x começa em `y1 + 1`, colada na moldura. As
MARCAS de tick apontam para fora, então **elas estão dentro da faixa**: são
traços de ~1 px espaçados de ~15,7 px, nas 3 primeiras linhas.

Com `BLOB_DILATE_X = 8` (dilatação de ±8 px, ou seja 16 px de alcance) as
marcas se costuram umas nas outras numa barra horizontal contínua; com
`BLOB_DILATE_Y = 2` essa barra encosta nos rótulos logo abaixo. Resultado:
**um blob só**, cobrindo o eixo inteiro, cujo centro é o meio do eixo.

E isso não é um defeito cosmético, porque `read_tick_labels` usa
**o centro do blob COMO O PIXEL DO TICK**. Blob fundido = posição errada em
TODO par daquela amostra. O mesmo vale no eixo y, onde as marcas ficam na
borda direita da faixa.

Um segundo defeito, independente, saiu da mesma investigação: a faixa do x
ia de `x0` a `x1` exatos, e o rótulo do primeiro tick fica a poucos px de
`x0` quando a margem do eixo é pequena (o gerador sorteia 1% a 6%). Metade
do texto caía fora e o centro do blob escorregava para a direita.

### 37.3 O conserto e o que ele mede

Três mudanças em `identify/calibrate.py`, todas em `read_tick_labels` e
`_text_blobs`:

1. `BLOB_DILATE_X` de **8 para 3** px.
2. Zerar a banda de tinta encostada na moldura antes de dilatar —
   `TICK_GAP` px, a MESMA constante que o `_candidatos` já aparava do
   recorte do OCR. A intenção existia no código; faltava aplicá-la à busca
   por blob. Aparar o recorte conserta o texto que o tesseract vê, não a
   POSIÇÃO do blob.
3. Folga na faixa: **20 px lateral no x** (devolve o rótulo decepado) e
   **2 px vertical no y**. A folga do y tem de ser pequena pelo motivo
   oposto: esticar a faixa do y para baixo de `y1` captura o topo do rótulo
   "0" do eixo x, que vira blob espúrio abaixo do último tick — justo onde a
   leitura do y começa (medido: folga 12 px derruba o y de 99,8% para 97,2%).

Medido em `data/test` (n=900), baseline gerado pelo MESMO script com a
configuração de hoje — comparação cache-com-cache, e o baseline reproduz o
§36.3 dígito a dígito (716 aceitas, 55 falso+):

| | hoje | com o conserto |
|---|---|---|
| âncora x a <= 3 px | 76,09% | **99,89%** |
| âncora y a <= 3 px | 75,53% | **99,89%** |
| `ok` | 79,56% (716) | **91,78% (826)** |
| falso positivo | 55 (7,68% das aceitas) | **20 (2,42%)** |
| `ok_x` | 87,67% | 94,56% |
| falso positivo em x | 57 (7,22%) | **7 (0,82%)** |
| `ok_y` | 88,44% | 94,89% |
| `calibration_failed` | 76 | **8** |
| `sinal_invalido` | 19 | 3 |

Falso positivo = escala aceita com erro relativo >= 1% contra o
`axis_affine` do meta — a mesma TOL_ESC das varreduras do §36.3.

**É o único ajuste medido em todo este bloco que sobe cobertura E desce
falso positivo.** Todas as 24 variantes de `_equiespacados` do §36.5(5)
trocavam uma coisa pela outra, a ~1,3 falso positivo por ponto percentual.

### 37.4 O que isto corrige no §36

- **O §36.2 está incompleto.** A anatomia em três camadas (moldura ->
  OCR -> emparelhamento) não nomeia a camada de BLOBS, que fica entre a
  moldura e o OCR e alimenta as duas seguintes. Os 80-87% de "recall do
  OCR" da camada 2 foram medidos com essa camada quebrada.
- **O §36.4 apontou a hipótese errada.** Ele culpou a exigência de
  consistência entre quatro pares. A consistência estava certa: ela
  rejeitava, corretamente, amostras cujas posições de blob eram lixo. Com a
  entrada consertada, `calibration_failed` cai de 76 para 8 sem tocar em
  uma linha do gate.
- **O §36.3(a) fica explicado.** O discriminador `_sem_paralela` reduziu
  espúrios de tick sem mover `ok` em um dígito. Faz sentido: as marcas de
  tick nunca foram usadas como posição — o blob é que era. Consertar a
  detecção de marcas não podia mesmo mudar nada.
- **O §36.3(b) fica sob suspeita, não refutado.** O experimento de `snap`
  encostava rótulos em marcas de tick usando posições de blob fundidas dos
  dois lados. A conclusão ("não usar ticks como referência de posição")
  continua de pé pela precisão das marcas, mas os números teriam de ser
  refeitos para valer como medida.

### 37.5 Verificação na imagem real

A fixture `caso_real_2ordem.png` passa de `ok=False` (`calibration_failed`
no eixo y) para **`ok=True`**. Os pares lidos no eixo x vão de 4 para 6, e
os dois que apareceram são **`0.0` e `10.0`** — exatamente o primeiro e o
último, os dois que a borda da faixa decepava. O corte lateral se confirma
fora do gerador.

Limite: das três imagens externas do §35/§36, só esta sobreviveu em disco;
as outras duas foram coladas na conversa e não viraram fixture. A porta 4
do critério (nenhum eixo hoje aprovado pode reprovar) foi verificada em
n=1, não em n=3. **Salvar as três como fixtures é dívida aberta**, e o §36.6
já dizia que ampliar o acervo externo é a maior alavanca do TCC.

### 37.6 O que isto faz com os próximos passos do §36.5

- **Item 1 (prior de origem): NÃO implementado, e agora é opcional.** O
  prior foi validado (100% no x, 99,56% no y) e a âncora ficou utilizável
  (99,89% nos dois eixos), mas o conserto de blob sozinho já entrega mais
  do que o item prometia. O prior continua valendo como duas coisas
  distintas, a medir separadas: **validador** (rejeitar ajuste cujo zero
  implícito caia longe do primeiro blob — ataca falso positivo, logo o
  critério 2.5) e **âncora** (ajustar com um par só — ataca cobertura).
- **Item 2 (classificador de dígitos): remedir antes de dimensionar.** O
  recall de 80-87% que o justificava foi medido sob a camada quebrada.
- **Item 3 (passo bonito): sem mudança**, continua bloqueado pelo gerador.
- **Item 4 (2.5 sob `filtra`): reavaliar.** O falso positivo caiu de 55
  para 20 por outro caminho; o trade que a variante `filtra` oferecia pode
  ter deixado de valer a pena.
- **Item 5 (não mexer em `_equiespacados`): confirmado com mais força.** O
  gate nunca foi o problema; ele estava recebendo lixo.

### 37.7 A lição de método

O item 1 foi para a fila em primeiro lugar por ser a hipótese mais forte do
§36. Ele não foi implementado — o que produziu o resultado foi a PORTA de
verificação dele, fixada antes por exigência do próprio §36.5, e a decisão
de investigar por que ela reprovou em vez de descartar o item. Uma porta de
aceitação que reprova é informação sobre o sistema, não sobre a hipótese.

### 37.8 Suíte completa: os dois critérios travados fecharam

`pytest tests/part2 -q` → **1 falha, 43 passam** (eram 4 falhas), 382 s.

| critério | alvo | antes | depois |
|---|---|---|---|
| **2.9** cobertura da calibração | ≥ 90% | 0,797 ❌ | **0,923 ✅** (n=300) |
| **2.5** rejeições corretas | ≥ 90% | 0,885 ❌ | **0,957 ✅** (n=23) |
| 2.3 erro relativo de sx, sy | < 1% em ≥ 95% | ✅ | 0,989 ✅ (n=277) |
| 2.4 taxa de rejeição (diagnóstico) | 1 − cobertura do 2.9 | 0,203 | 0,077 |
| 2.11 bloco `dimensionless` | 100% | ✅ | 300/300 ✅ |
| **2.1** erro perpendicular da máscara | ≤ 1,0 px med / ≤ 2,0 px p95 | 0,807 / 2,077 ❌ | 0,807 / 2,077 ❌ |

O 2.9 estava reprovado **desde o Bloco 5** e resistiu às 24 variantes de
`_equiespacados` do §36.5(5), ao `snap` e ao `_sem_paralela` do §36.3. O 2.5
nunca tinha tido sequer uma pista de melhora antes do §36.3(b) — e fechou
por um caminho diferente do que aquela pista sugeria: não por rejeitar mais,
e sim por deixar de produzir a leitura ruim que precisava ser rejeitada.

O 2.1 é a máscara da U-Net, que este conserto não toca; segue reprovado com
os MESMOS 0,807 / 2,077 px do §35, herdados da promoção do modelo RGB. Fica
como o único critério aberto da Parte 2, e precisa de decisão própria:
investigar a cauda do modelo ou revisar o alvo de 2,0 px.

O estrato mais fraco do 2.9 passa a ser `dpi 60-99` (0,800, n=85), contra
0,974 e 0,970 nos dpi maiores — o que é coerente com a causa consertada,
porque em dpi baixo o rótulo tem menos pixels e a banda de marcas pesa mais
na faixa.

### 37.9 Ruling 59a — colapso do lote de OCR (achado por imagem externa)

`_ocr_numeros_lote` monta todos os recortes num mosaico e chama o tesseract
com `--psm 7`. Quando a análise de LAYOUT decide que aquilo não é uma linha
de texto, ele **não levanta exceção**: devolve zero palavras, e o lote inteiro
vira `None` de uma vez. A amostra fica sem nenhum par nos DOIS eixos.

Não é previsível pelo tamanho. Medido numa imagem externa, variando só o
tamanho do lote sobre os MESMOS recortes: 14 → 11 lidos, 16 → **0**,
18 → **0**, 20 → 17, 21 → **0**.

Custo, medido em `data/test` (amostras com zero pares em ambos os eixos):
**16 antes**, 12 depois do conserto de blobs do §37.3. Ou seja: pré-existente,
e o conserto de blobs não o criou — mudou o conjunto de recortes e caiu num
caso ruim. Para a imagem externa foi regressão do §37.3 (o HEAD lia 7 pares
em x, a versão nova lia 0), e está registrado como tal.

Conserto: teto de `_LOTE_MAX = 12` recortes por mosaico (o dano fica preso a
um bloco) mais releitura individual de qualquer bloco que colapse. O custo do
recuo só existe no caminho raro.

| | antes do §37.3 | com §37.3 | com §37.3 + teto |
|---|---|---|---|
| `ok` | 79,56% | 91,78% | **93,00%** |
| falso positivo | 55 | 20 | 20 |
| `ok_x` | 87,67% | 94,56% | 95,67% |
| `ok_y` | 88,44% | 94,89% | 96,22% |

### 37.10 Ruling 59b — dois fundos na figura (achado por imagem externa)

O fundo era estimado pela mediana do quadro inteiro, o que assume um fundo
só. O `generator.py:200-202` garante isso; o matplotlib real não — um
`ax.set_facecolor()` diferente do `figure.facecolor` é comum em tema escuro.

Quando a área de dados é a MAIOR das duas regiões, a mediana cai no fundo DOS
EIXOS e a moldura inteira da figura passa a contar como tinta. Medido numa
imagem externa de tema escuro (figura 43, eixos 30, mediana 30):
`detect_plot_bbox` devolveu **(0, 0, 799, 460)** — o quadro inteiro — em vez
de (100, 55, 720, 410). Faixas de rótulo vazias, zero pares, calibração
física perdida.

Conserto: `_fundo()` = **moda da borda** da imagem. A borda é fundo da FIGURA
por construção. Com um fundo só, moda da borda e mediana coincidem:
verificado idêntico nas 900 amostras, e o bbox saiu igual em **900/900**.

### 37.11 Ruling 59c — guardas de plausibilidade: uma refutada, duas implementadas

**REFUTADA: descontinuidade da máscara.** Parecia óbvia — na imagem externa
que erra, o maior buraco entre colunas com tinta é 19,3%, contra 5,5% da
segunda pior das outras sete. Medida no corpus (n=837 com físico e verdade
comparável), NÃO sobrevive:

| sinal | Spearman contra o erro da identificação | p |
|---|---|---|
| maior buraco | **+0,020** | 0,57 |
| densidade de colunas | +0,003 | 0,93 |

E o maior buraco do próprio corpus (20,9%) é MAIOR que o da imagem que erra.
Em qualquer limiar a guarda pega no máximo 3 de 79 identificações ruins.

**O motivo é estrutural e é a lição desta subseção**: o corpus não contém o
modo de falha que essa guarda existia para pegar, então ele media só o CUSTO
dela, nunca o benefício. A ordem "guarda primeiro, estrato depois" estava
errada — o estrato do §37.12 é PRÉ-REQUISITO para validar essa guarda, não um
passo posterior. Registrado porque a separação em n=8 era convincente e n=895
a desmente.

**IMPLEMENTADA: resíduo do ajuste (`_NRMSE_MAX = 0.13`).** Escapa do problema
acima por não depender do modo de falha — depende de o ajuste ficar ruim
quando a série é lixo, e disso o corpus tem 79 exemplos.

| | valor |
|---|---|
| Spearman `nrmse` × erro | **+0,386** (p = 4,5e-31) |
| limiar (p98 do corpus) | 0,1286 → arredondado para 0,13 |
| rejeita | 17 de 837 |
| precisão | **88,2%** (15 das 17 de fato ruins) |
| custo | 2 boas perdidas em 837 (0,24%) |
| recall | 19% — recusa o absurdo, não audita o aceitável |

**IMPLEMENTADA: resposta inversa (`_UNDERSHOOT_MAX = 0.10`).** Fase não-mínima
não pertence à família de modelos do Estágio D: nem FOPDT nem 2ª ordem sem
zero representam resposta inversa, então o ajuste devolve um sistema plausível
e estruturalmente errado.

A primeira definição da métrica tinha FALSO POSITIVO: olhava a série inteira
contra o "valor final" e marcava 0,147 numa imagem externa de ζ=0 cuja
identificação estava CERTA (ωₙ a 0,001% do verdadeiro) — porque a série não
assenta e o valor final caía num ponto qualquer da oscilação. Redefinida pela
física (a excursão contrária só conta ANTES de a resposta arrancar), aquela
imagem cai para 0,004 e a de fase não-mínima fica em 0,143.

Custo no corpus: **2 de 900 (0,22%)**. **ATENÇÃO ao que NÃO está medido**: o
gerador não produz fase não-mínima, então o corpus dá só o custo. O benefício
está apoiado em n=1.

**Efeito nas oito imagens externas**: as seis corretas ficam intactas; as duas
que respondiam errado com `ok=true` passam a recusar com motivo nomeado —
`ajuste_inconsistente` e `resposta_inversa`. Suíte: 1 falha, 43 passam (só o
2.1, inalterado).

### 37.12 Ruling 60 — estrato OOD novo: banda de acomodação e anotação com seta

Dois campos de RENDER opt-in, `has_settling_band` e `has_annotation_arrow`,
seguindo exatamente o padrão do `has_reference_line` (§34.5): `sample_style`
nunca os toca, `render_sample` os marca por `replace`, e o caminho padrão foi
verificado **byte a byte idêntico**.

São DOIS campos e não um porque são fenômenos separáveis, e separados dão
ablação. Ablação pareada, n=60 seeds, `janela_assentada=True` em todos os
braços para a geometria ser comparável:

| braço | IoU mediana | IoU p10 | Δ vs base | p (Wilcoxon) |
|---|---|---|---|---|
| base | 0,6121 | 0,3959 | — | — |
| só banda | 0,5352 | 0,1980 | −0,0629 | 2,6e-10 |
| só seta | 0,4969 | 0,3041 | −0,1130 | 1,6e-11 |
| **banda+seta** | **0,4381** | **0,1866** | **−0,1898** | 1,6e-11 |

A IoU cai **31% em mediana** com os dois juntos, e o efeito é aditivo. É o
oposto da primeira tentativa do §35, onde o estrato não movia a métrica e o
retreino teria sido desperdício: este reproduz o fenômeno.

**A seta pesa quase o dobro da banda**, o que confirma o diagnóstico da imagem
externa — o que desvia a máscara é um segmento espesso SAINDO da curva, não a
sombra no patamar. Texto sem seta já existia em `style.annotations` e nunca
degradou nada.

**Limite honesto do que este estrato prova.** O erro de identificação quase
não se move (mediana 0,0079 → 0,0175; o p90 até melhora). A máscara piora
muito e a resposta final quase não piora, porque a polilinha e o ajuste
absorvem o estrago. A imagem externa mostra que existe um regime em que não
absorvem, mas o corpus, mesmo com o estrato, ainda não o alcança. **O estrato
prova que o Estágio A sofre, não que o Estágio D quebra.**

### 37.13 Placar das imagens externas ao fim do bloco

Oito imagens, seis identificadas corretamente, duas recusadas com motivo:

| imagem | verdade | resultado |
|---|---|---|
| caso_real_2ordem | ζ=0,5 ωₙ=2 | 0,502 / 2,021 ✅ |
| Figure_21 | ζ=0,2 ωₙ=5 | 0,200 / 5,034 ✅ |
| Figure_12 | ζ=0 ωₙ=4 | 0,001 / 4,000 ✅ (ζ no piso) |
| Figure_15 | 1ª ordem τ≈2 | τ=1,981 ✅ |
| Figure_11 | 1ª ordem τ≈0,2 | τ=0,195 ✅ |
| Figure_122 | 1ª ordem τ≈0,34 | τ=0,335 ✅ |
| Figure_16 | ζ=0,6 ωₙ≈10 | recusa `ajuste_inconsistente` |
| Figure_22 | fase não-mínima | recusa `resposta_inversa` |

**Dívida aberta e prioritária: nenhuma delas é fixture.** Só a primeira está
versionada; as outras sete vieram por conversa e vivem em `/home/loizm/`. Sete
das oito não protegem nada em regressão. Somando ao §37.5, esta é a tarefa de
maior alavancagem que resta.

Item conhecido, não corrigido: o `ZETA_BOUNDS` de `classical.py:44` tem piso
`1e-3`, então ζ=0 verdadeiro sai como 0,001. Não é medição, é o limite da
caixa de parâmetros.

---

## 38. Ruling 61 — o que treze imagens externas mediram DEPOIS da promoção do base 32

O §37.13 fechou com oito imagens. Cinco outras vieram depois, escolhidas para
sondar os limites declarados em vez de repetir o caso fácil. Duas delas
encontraram buracos que as guardas do §37.11 não cobrem, e um deles tem
conserto barato porque o sinal já existe no código e não está sendo lido.

### 38.1 Placar das treze

| imagem | verdade declarada | resultado | veredito |
|---|---|---|---|
| caso_real_2ordem | ζ=0,5 ωₙ=2 | 0,500 / 2,019 | ✅ |
| Figure_11 | 1ª ordem | τ=0,195 | ✅ |
| Figure_12 | ζ=0 ωₙ=4 | 0,006 / 4,004 | ✅ (ζ no piso) |
| Figure_122 | 1ª ordem | τ=0,336 | ✅ |
| Figure_15 | 1ª ordem τ≈2 | τ=1,983 | ✅ |
| Figure_16 | ζ=0,6 ωₙ≈10 | 0,604 / 10,124 | ✅ |
| Figure_21 | ζ=0,2 ωₙ=5 | 0,200 / 5,034 | ✅ |
| Figure_22 | fase não-mínima | recusa `resposta_inversa` | ✅ |
| Figure_f1 | FOPDT K=2 θ=2 | K=2,0275 θ=1,9690 | ✅ |
| Figure_f2 | 2ª ordem θ=1,5 K=1 | K=1,0007 θ=1,4922 | ✅ |
| **Figure_f3** | **2ª ordem crítica θ=3 K=1,5** | **`fopdt`** | ❌ ordem |
| **Figure_222** | duas curvas sobrepostas | `fopdt` sem sinalizar | ⚠️ |
| **Figure_322** | instável, polo em s=+1,5 | `second` sem sentido | ❌ |

### 38.2 Figure_f3 — não é bug, é identificabilidade

Uma 2ª ordem CRITICAMENTE amortecida (ζ=1) com atraso foi classificada como
FOPDT. O mecanismo, medido:

| | fopdt | second |
|---|---|---|
| nrmse | 0,03308 | 0,03284 |
| SSE | 1,8078 | 1,7815 |
| AIC | −3395,4 | **−3402,0** |
| θ | 3,1782 (5,9 % de erro) | **3,0486 (1,6 %)** |
| ζ | — | 1,44 (verdade 1,0) |

O ajuste de 2ª ordem é melhor em SSE E em AIC, e recupera θ com 1,6 % contra
5,9 %. Mesmo assim o teste de ordem o rejeita:

```
razão SSE₁/SSE₂ = 1,0148  ->  ln = 0,01465
n_eff = 21,0
ganho = 21,0 × 0,01465 = 0,308     contra limiar 2,0
```

A 2ª ordem é só **1,5 % melhor em SSE**. Isso não é artefato de extração — é a
natureza do problema: com ζ ≥ 1 e atraso na janela, as duas estruturas
produzem curvas quase idênticas. Nem uma máscara perfeita criaria a informação
que falta. É o mesmo fenômeno que o PLANO §1.3 já quantificou (100 % de acerto
em ζ < 1,6; 42,4 % em ζ ≥ 2,2) e que o critério 2.12 mede em 91,3 %.

Note que a 2ª ordem também não recupera ζ=1 — ela devolve 1,44, uma família
superamortecida diferente que passa pelos mesmos pontos. Escolher a estrutura
certa não teria dado o parâmetro certo.

**O defeito não é a escolha, é o silêncio.** A pipeline afirma `fopdt` com
`ok=true` e sem nenhum aviso de que a evidência era de 0,308 contra 2,0.

### 38.3 Figure_222 — duas curvas no mesmo quadro

A figura tem DUAS curvas (2ª ordem real e aproximação de 1ª ordem) desenhadas
quase sobrepostas, o que é o assunto do próprio gráfico. A pipeline devolveu
`fopdt`, K=0,9867, τ=1,21 — concordando com a APROXIMAÇÃO, não com o sistema
real.

A resposta é defensável para o dado visível, mas duas coisas ficam sem
sinalização:

- **Múltiplas curvas é escopo congelado** (PLANO §1.4) e a pipeline não avisa
  que viu mais de uma. Ela mescla as componentes e segue.
- O `nrmse` de **0,08727** é o mais alto entre todas as imagens aceitas — 2,7×
  o segundo pior. Alto o bastante para ser sintoma, baixo o bastante para não
  disparar o limiar de 0,13.

### 38.4 Figure_322 — instável, e a saída é sem sentido com `ok=true`

Exponencial divergente (polo em s=+1,5). Saída:

```
K    = 1e+04     <- TETO exato de K_BOUNDS
zeta = 0.001     <- PISO exato de ZETA_BOUNDS
wn   = 0.1071
nrmse = 0.03182  <- excelente
```

Os dois parâmetros estão cravados em **extremos opostos da caixa**. O modelo
não representa divergência, então o otimizador aproxima a exponencial crescente
com o primeiro quarto de período de uma senoide quase não amortecida de ganho
gigantesco. O ajuste ENCAIXA bem no trecho visível — por isso o resíduo é
baixo e a guarda do §37.11 não dispara.

O candidato FOPDT também tem K no teto (1e+04, τ=62,98), então os DOIS
ajustes bateram na parede. Isso não é coincidência: nenhum membro da família
consegue crescer sem limite, e a única forma de chegar perto é ganho infinito.

### 38.5 O sinal grátis que o sistema ignora

**Parâmetro cravado na borda da caixa não é medição — é o otimizador
desistindo.** A informação já existe (`K_BOUNDS`, `WN_BOUNDS`, `ZETA_BOUNDS`,
`TAU_BOUNDS` em `classical.py:41-44`) e custa uma comparação exata, sem limiar
calibrado contra corpus.

Verificado nas treze imagens: só a Figure_322 tem parâmetro na borda **na
estrutura escolhida**. Mas há uma armadilha medida:

- a **Figure_12** (ζ=0 verdadeiro) encosta no piso do ζ **legitimamente** —
  uma guarda ingênua sobre ζ a rejeitaria, e ela está certa;
- na **Figure_222**, o ajuste de 2ª ordem REJEITADO tem ζ=10 (teto), o que
  mostra que borda aparece também em candidatos que não viram saída.

Então o critério não pode ser "algum parâmetro na borda". Precisa ser mais
fino — a hipótese a testar é **K na borda**, que não tem caso legítimo à
vista, possivelmente combinado com ζ na borda.

### 38.6 PRÓXIMOS PASSOS, em ordem

**1. Guarda de parâmetro na borda (fazer primeiro).**
Ataca a Figure_322, que é a pior saída possível: quatro números sem sentido
com `ok=true`. Critério de aceitação a fixar ANTES de implementar, e a
primeira pergunta a responder é o CUSTO: quantas das 900 amostras do corpus
têm algum parâmetro na borda na estrutura escolhida? Se for perto de zero, a
guarda é quase de graça; se for muito, não serve. Medir separado para K, τ,
ωₙ e ζ — a Figure_12 já prova que ζ tem caso legítimo no piso e K, até agora,
não tem nenhum.

**2. Sinalização de ordem ambígua.**
Ataca a Figure_f3. Quando o ganho do teste de ordem fica muito abaixo do
limiar, a evidência para a estrutura mais complexa não existe — e hoje a
pipeline escolhe a mais simples em silêncio. Uma faixa de indecisão marcaria
`ordem_incerta` e entregaria os parâmetros das DUAS estruturas, deixando a
escolha para quem tem o contexto físico. Validar contra o critério 2.12
(acurácia de ordem, 91,3 %): a guarda não pode derrubá-lo.

**3. Detecção de múltiplas curvas.**
Ataca a Figure_222. Escopo congelado no PLANO §1.4, então o objetivo NÃO é
identificar duas curvas — é **recusar** em vez de mesclar em silêncio. Sinal
candidato: componentes conexas com extensão horizontal comparável à da
moldura, em número maior que um, após o filtro de retas de span completo.

**4. Medir a latência com o base 32 (critério 3.11).**
O alvo é < 2 s em CPU e o modelo cresceu 78 %. Não foi remedido depois da
promoção. É barato e pode exigir decisão.

**5. Salvar as treze imagens como fixtures.**
Continua sendo a dívida de maior alavancagem, e cresceu: agora são treze
imagens e uma versionada. Ver §37.13.

### 38.7 Ferramenta nova

`identificar.py` na raiz — casca de linha de comando sobre
`identify_from_image`, sem lógica própria de propósito. Traduz os códigos de
recusa para português corrente, mostra o bloco adimensional com instrução de
uso quando a calibração falha, e aceita `--classico` (dispensa torch e GPU),
`--json` e várias imagens de uma vez.

### 38.8 O que as treze dizem sobre o envelope

Das cinco imagens novas, três estão DENTRO do envelope declarado (f1, f2, 222)
e duas FORA (f3 no limite de identificabilidade, 322 francamente fora). O
placar dentro do envelope é 3/3 quando a estrutura é distinguível; fora dele, a
pipeline erra em silêncio nos dois casos.

Isso reforça o que o §37 já registrava: **o sistema não tem detector para a
maior parte do que está fora da família**. Hoje há guarda para resposta inversa
e para resíduo alto. Não há para instável, para ordem superior, para ganho
negativo, para zero no semiplano esquerdo, nem para múltiplas curvas — e a
Figure_322 mostra que resíduo baixo não é garantia de nada quando o otimizador
tem uma caixa grande para fugir.

---

## 39. Ruling 62 — os três sistemas do `rg.py`: dois defeitos consertados e um que exige RETREINO

Três imagens novas, produzidas por `rg.py` (script do próprio autor, versionado
na raiz), com a verdade declarada na função de transferência — não há
estimativa envolvida em nenhum número deste ruling.

| sistema | verdade | antes | depois |
|---|---|---|---|
| 1 — FOPDT dominado pelo atraso | K=5, τ=1, θ=4 | `second`, ζ=2,44 ❌ | `fopdt`, K=5,01 τ=0,997 θ=3,99 ✅ |
| 2 — 2ª ordem superamortecida | K=2, ωₙ=2, ζ=1,5, θ=2 | ✅ | ✅ (controle negativo) |
| 3 — 2ª ordem subamortecida | K=1, ωₙ=5, ζ=0,15, θ=1 | sem K (`calibration_failed`) ❌ | K=0,994 ωₙ=4,99 ζ=0,150 θ=1,000 ✅ |

As três estão agora em `tests/fixtures/caso_real_rg_*.png`, com
`tests/part2/test_caso_real_rg.py` — paga uma parte da dívida do §38.6 item 5.

**O que as três têm em comum, e por que o corpus não as encontrou.** `rg.py` usa
`plt.ylim(0, ...)`. Isso encosta duas coisas na moldura inferior: a CURVA no
patamar de repouso (2 a 3 px, ~0,9 % do span do eixo y) e o RÓTULO extremo do
eixo y. O gerador sorteia `y_margin_lo ~ U(0.03, 0.15)` e **nunca desce abaixo
de 3 %** — a geometria inteira está fora da distribuição de treino e fora da
distribuição de teste. Uma causa, três defeitos.

### 39.1 Sistema 3 — o eixo y era reprovado com os nove rótulos lidos CERTOS

O OCR leu 1,8 / 1,6 / 1,4 / 1,2 / 0,8 / 0,6 / 0,4 / 0,2 / 0,0 — todos corretos.
Quem reprovou foi `_equiespacados`, por um artefato geométrico do recorte:

```
blob y=[  0, 11) altura=11  centro=51.5   <- COLADO NO TOPO DO STRIP (glifo cortado)
blob y=[ 32, 46) altura=14  centro=85.0
...  (todos os interiores: altura 14)
blob y=[313,324) altura=11  centro=364.5  <- COLADO NA BASE (glifo cortado)
```

`FOLGA_FAIXA_Y = 2` é pequeno de propósito (§37: estender para baixo captura o
rótulo "0" do eixo x). O efeito colateral só aparece quando o rótulo extremo
encosta na moldura: o strip decepa 3 px de cada ponta e o centróide — que É o
pixel do tick — entra 1,5 px. Espaçamentos: `[33.5, 35, 35, 71, 35, 35, 35, 33.5]`.
`_equiespacados` usava `unit = min(d)` = 33,5, o valor enviesado, inflando toda
razão em 4,5 %; a lacuna do rótulo "1.0" (não lido) virava razão 2,1194 e o erro
`|2,1194−2|/2 = 0,0597` estourava `SPACING_TOL = 0,05`. **Reprovado por 1,2 ponto
percentual.** O Sistema 1 passava raspando no mesmo defeito (erro 0,045).

Duas correções, medidas separadamente no corpus (n=900):

| variante | ok | ok_y | FP_y | perdidas |
|---|---|---|---|---|
| base | 837 | 866 | 14 | — |
| `FOLGA_FAIXA_Y` 2→8 | 836 | 866 | 13 | **1** |
| C: unidade robusta em `_equiespacados` | 837 | **867** | 14 | 0 |
| D: `_centros_y_corrigidos` | 837 | 866 | **13** | 0 |
| **C+D (adotada)** | 837 | **867** | **13** | **0** |

Aumentar a folga foi **testado e descartado**: perde uma amostra e não ganha
nenhuma. D é a correção de causa — reconstrói o centro do blob cortado pela
meia-altura mediana dos não cortados, e recupera `sy = −0,005690` contra
`−0,005717`, que é a inclinação dos rótulos INTERIORES, os não enviesados. C é
defesa em profundidade: `min(d)` é o estimador menos robusto possível para uma
unidade de espaçamento.

Incidência do defeito no corpus: 35 amostras em 895 (3,9 %) têm rótulo do y
cortado. O corpus mede o CUSTO (zero) e quase não mede o BENEFÍCIO — pelo mesmo
motivo do §37.11: ele não contém a geometria que produz a falha.

### 39.2 Sistema 1 — a ordem era decidida por DOIS pixels

O ajuste FOPDT acerta tudo (K=5,0203, τ=0,9966, θ=3,9942, NRMSE=0,0020) e mesmo
assim perdia:

```
SSE1/SSE2 = 1,174   n_eff = 131   ganho = 21,0  >  limiar 2,0  ->  SECOND
polos do SECOND: −1,0061 e −21,9623
   τ_dominante = 0,994 s (= o τ verdadeiro)
   τ_rápido    = 0,0455 s = 2,35 amostras = 3,6 px
```

Onde estava esse ganho de 17 %:

```
t=[3,4)s   n=  2   ganho = +0,006544   (+107,5 % do total)
t=[4,5)s   n= 52   ganho = −0,000826   ( −13,6 %)
t=[5,12)s  ...     no líquido NEGATIVO
```

**Dois pontos, de 406.** O polo extra tem constante de tempo de 3,6 px — a
espessura do próprio traço no canto do degrau. A 2ª ordem não achou dinâmica;
ajustou o antialiasing. As guardas do §37.11 não pegam isso por construção:
`_NRMSE_MAX = 0,13` contra NRMSE de 0,0019.

`_polo_rapido_e_artefato` (novo, em `classical.py`): se o escolhido é 2ª ordem
**superamortecida** e um trecho CONTÍGUO de 3 % da série responde por ≥ 100 % da
vantagem de SSE, devolve FOPDT. O limiar 1,0 é estrutural, não calibrado — quer
dizer "fora do trecho o 2ª ordem perde". Só o TAMANHO do trecho saiu de medição.

**Contíguo importa, e foi medido.** A primeira versão usava os pontos de maior
ganho onde quer que estivessem, e custava caro no caminho ORÁCULO, que nem tem
render — era a guarda pegando picos de ruído. Um canto rasterizado é um acidente
LOCAL; ruído de aquisição é disperso por construção:

| estatístico | imagem (n=837) | oráculo 20 dB (n=300) |
|---|---|---|
| sem guarda | 88,89 % | 88,3 % |
| top 1 % DISPERSO > 1 | 92,23 % | **85,7 %** (−2,6 p.p.) |
| trecho CONTÍGUO 3 % > 1 | **92,95 %** | 87,7 % (−0,6 p.p.) |

Varredura do tamanho do trecho (o único número escolhido por medição):

| frac | imagem | custo | oráculo 20 dB |
|---|---|---|---|
| 0,005 | 90,08 % | −1 | — |
| 0,01 | 91,64 % | −3 | 88,0 % |
| 0,02 | 92,71 % | −6 | 87,3 % |
| **0,03** | **92,95 %** | **−7** | **87,7 %** |
| 0,05 | 92,95 % | −9 | 87,7 % |
| 0,08 | 91,64 % | −23 | 86,7 % |

0,03 e 0,05 empatam; 0,03 leva por ter o menor custo do platô e ser a janela
mais curta — trecho menor é afirmação mais forte de localidade.

**A restrição a ζ > 1 é estrutural.** Com polos complexos não existe polo rápido
separado a descartar: a oscilação é a assinatura inteira e nenhum FOPDT a
representa. Rebaixar destruiria o ζ — a grandeza que o nível adimensional da
Decisão E existe para entregar, e exatamente o que o Sistema 3 recupera. Sem a
restrição o corpus rebaixaria 1 amostra a mais (e ela é de 1ª ordem, ou seja,
ganho): a restrição custa quase nada e fecha um modo de falha inteiro. O Sistema
2 é o controle negativo versionado: estatístico 0,639 contra limiar 1,0.

Custo real das rebaixadas: são 2ª ordens genuinamente superamortecidas (ζ de 2 a
3,5) cujo polo rápido é quase invisível. Medido nelas, o τ reportado ficou a
0,1 % / 0,4 % / 0,1 % / 1,5 % do τ dominante verdadeiro e K a menos de 1 %.
**Perde-se o rótulo, não a física.**

### 39.3 O defeito que SOBROU: são DOIS, e os dois exigem RETREINO

Nenhuma correção em código resolve estes. A U-Net dá `prob <= 0,004` em colunas
onde há traço colorido, puro e NÃO ocluído, desenhado:

```
Sistema 1: máscara cobre 406/622 colunas (65,3 %) — falta t=[0, 3,95] s inteiro,
           o platô do tempo morto: 33 % do gráfico
Sistema 3: máscara cobre 511/622 colunas (82,2 %) — falta t=[10,37, 12] s,
           a cauda assentada, e t=[0, 0,46] s, o platô inicial
Sistema 2: 96,9 % — passa
```

E é isso que CAUSA o §39.2: sem o platô, o canto em t=4 s do Sistema 1 fica
apoiado nos dois pontos que decidiam a estrutura.

As duas perdas parecem a mesma coisa e **não são**. A ablação separa:

| ablação (só o trecho perdido move, resto do render idêntico) | Sistema 1 | Sistema 3 |
|---|---|---|
| original | 0/200 | 0/85 |
| ondulação de ±1 px, MESMA altura | **0/200** | **79/85** |
| 12 px acima da moldura, ainda reto | **98/200** | — (é no meio do gráfico) |
| ondulação + 12 px acima | 165/200 | — |

#### Defeito A — curva rente à moldura inferior (Sistema 1)

Curva dose-resposta, deslocando só o platô (span do eixo = 318 px):

| distância à moldura | cobertura no platô |
|---|---|
| 1 px (0,3 %) | 0 / 150 |
| 3 px (0,9 %) — **original** | 0 / 150 |
| 5 px (1,6 %) | 0 / 150 |
| 7 px (2,2 %) | 107 / 150 |
| 9 px (2,8 %) | 106 / 150 |
| 13 px (4,1 %) | 122 / 150 |

**O degrau está em ~2 % do span, logo abaixo do piso de 3 % que o gerador
sorteia em `y_margin_lo ~ U(0.03, 0.15)`.** Não é limiar mal escolhido nem bug
de código: é região que não existe no treino. `plt.ylim(0, ...)` — repouso em
zero, o caso normal de um gráfico de controle real — cai sempre nela.

**Três hipóteses REFUTADAS por ablação**, e vale registrar as três porque as três
pareciam óbvias:

- a banda cinza do `axvspan` — apagá-la não muda nada, 0/150 -> 0/150. O gerador
  nem produz banda VERTICAL, só `axhspan`, então era suspeita natural;
- a COR do traço — recolorir o Sistema 1 para o rosa do Sistema 2 (que funciona)
  mantém 0/150, e recolorir o Sistema 2 para o ciano mantém 125/150. O
  checkpoint é RGB, então a hipótese era plausível;
- a PLANURA do trecho — ondular ±1 px na mesma altura mantém 0/200. É o que
  separa este defeito do B.

#### Defeito B — trecho perfeitamente RETO e horizontal (Sistema 3)

A cauda assentada some, e a causa não é nenhuma das suspeitas:

| ablação | cobertura da cauda |
|---|---|
| original | 0/85 |
| cinza puro da tracejada apagado | 5/85 |
| tracejada removida, curva intacta | 7/85 |
| cauda deslocada 25 px, longe da tracejada | 0/85 |
| **ondulação de ±1 px na mesma cauda** | **79/85** |

A rede vê amarelo PURO e não ocluído em x=640 e responde 0,047; a probabilidade
despenca de 0,954 em x=634 para 0,027 em x=636, que é exatamente onde a
ondulação própria da curva cai abaixo de 1 px. **O que ela suprime é a
RETIDÃO**, não a oclusão.

A explicação é o próprio treino, e é uma consequência não intencional de um
critério existente: o G3b.2 ("Sem reta de span completo na máscara") ensina o
modelo a REJEITAR reta horizontal de span completo, porque distrator é isso. Só
que uma resposta assentada É uma reta horizontal — e quando a ondulação residual
fica sub-pixel os dois objetos passam a ser literalmente os mesmos pixels. O
modelo resolve a ambiguidade suprimindo, e leva a curva junto.

**CORREÇÃO de um erro da primeira versão deste ruling:** eu havia escrito que o
estrato `reta_no_patamar` (§34.5) era opt-in e que o checkpoint promovido não
tinha sido treinado com ele. **Está errado.** O `logs/train_base32.log` registra
o treino com `['data/train', 'data/train_reta', 'data/train_janela',
'data/train_banda_seta', 'data/train_banda', 'data/train_seta',
'data/train_reta_banda', 'data/train_reta_seta', 'data/train_reta_banda_seta']`
— o estrato ESTÁ lá. O defeito B sobrevive a ele, e as ablações acima mostram
por quê: o estrato foi construído para o problema da OCLUSÃO por reta de
referência, e o que quebra não é a oclusão.

#### O que o retreino precisa

1. **Estrato de margem inferior quase nula — NÃO EXISTE.** `y_margin_lo` sorteado
   em 0 a 1 % do span, contra o piso atual de 3 %. Ataca o defeito A.
2. **Estrato de cauda LONGA e assentada — o que existe não basta.** O `train_reta`
   ataca oclusão, não retidão. O que falta é janela larga o bastante para a
   resposta assentar de verdade e ficar reta por muitas colunas, com e sem reta
   de referência em cima. Vale notar que o corpus atual tem 442 de 600 amostras
   com `w < 3` (janela truncada, §RULING C), ou seja, a maioria **nunca assenta**
   — o caso que quebra é justamente o sub-representado. Ataca o defeito B.
3. **Remedir o critério G3b.2 depois.** Ele é o que ensina a suprimir reta, e o
   defeito B é efeito colateral dele. Retreinar sem remedi-lo pode trocar um
   defeito por outro: reta de referência voltando para dentro da máscara.

**A fragilidade é INTERMITENTE, e esse é o argumento central para retreino e não
para guarda.** Os três sistemas têm a curva à MESMA distância da moldura (3, 3 e
2 px) e o Sistema 2 sobrevive enquanto o Sistema 1 colapsa a zero. Comportamento
fora da distribuição é assim: não dá para prever qual imagem cai, então não dá
para consertar com limiar. Precisa de dado.

**Registrado em código, não só aqui.**
`tests/part2/test_caso_real_rg.py::test_estagio_a_cobre_a_janela_inteira` está
marcado `xfail(strict=True)` para os Sistemas 1 e 3, com alvo de 90 % de
cobertura. No dia em que o retreino consertar, o teste vira XPASS e em modo
estrito **reprova a suíte** — obrigando quem consertou a remover a marca e
converter o defeito documentado em portão de regressão. É o mecanismo que impede
esta dívida de sumir do jeito que as oito imagens do §37.13 sumiram. O ponteiro
para o defeito A está também em `dataset/randomize.py`, na linha do sorteio.

### 39.4 Validação: o que mudou no corpus e o que NÃO mudou

Corpus `data/test` (n=900), antes → depois das duas correções:

| métrica | antes | depois |
|---|---|---|
| ordem correta, caminho imagem (n=837) | 88,89 % | **93,0 %** |
| — plantas de 1ª ordem | 82,6 % | **92,1 %** |
| — plantas de 2ª ordem | 95,6 % | 93,9 % |
| erro máximo de parâmetros ≤ 5 % | 60,1 % | **64,5 %** |
| erro máximo de parâmetros ≤ 10 % | 73,6 % | **78,4 %** |
| calibração aceita | 93,0 % | 93,0 % |
| `ok_y` / falso positivo de escala em y | 96,2 % / 14 | **96,3 % / 13** |
| critério 2.12-ordem | 89,0 % | **92,3 %** |
| critério 2.6 (degradação, alvo ≤ 3 p.p.) | +1,67 | **+1,63** ✅ |

Suíte de `tests/test_part1.py` + `tests/part2/`: **73 testes passam**, nenhum
critério mudou de veredito.

**A regressão que existe, declarada.** A acurácia de ordem no caminho ORÁCULO a
20 dB (diagnóstico da Parte 1) caiu de 0,888 para 0,878 — 6 amostras em 600. É
o preço do trecho contíguo num caminho que não tem render, e é o que sobrou
depois de trocar a estatística dispersa (que custava 2,8 p.p.). Fica dentro de
1 sigma do ruído amostral. Em troca, o caminho da imagem — que é para o que o
sistema existe — sobe 4,1 p.p.

**A latência NÃO regrediu, apesar do relatório.** O critério 2.8 saiu de 160 ms
para ~220 ms entre execuções, mas medido em isolamento a guarda custa **35 µs**
por chamada e a pipeline dá 177 ms com ela contra 180 ms sem ela. O número do
relatório oscila com a carga da execução do pytest; o alvo é 500 ms.

## 40. Ruling 63 — ganho negativo: o caminho C funciona, e o detector de direção custou TRÊS formulações

Três imagens novas, de `rg_negativo.py` (versionado na raiz, análogo ao `rg.py`
do §39), com a verdade declarada na função de transferência. Todas com **degrau
negativo**, que era estruturalmente inexprimível: `K_BOUNDS = (1e-3, 1e4)` trava
K positivo, o ajuste saía com NRMSE 0,90–0,96, e as recusas eram efeito
colateral, não detecção. A primeira era rejeitada como `resposta_inversa` —
diagnóstico FALSO: não é fase não-mínima, é degrau negativo.

| imagem | verdade | antes | depois |
|---|---|---|---|
| `neg_sub` | K=−1, ωₙ=5, ζ=0,2, θ=2 | `ajuste_inconsistente` ❌ | K=−0,998 ωₙ=5,002 ζ=0,201 θ=2,005 ✅ |
| `neg_super` | K=−3, ωₙ=4, ζ=1,25, θ=3,5 | `ajuste_inconsistente` ❌ | K=−2,979 ✅; ωₙ/ζ/θ ❌ (§39.3) |
| `neg_fopdt` | K=−2, τ=0,5, θ=3 | `resposta_inversa` ❌ | segue recusada ❌ (defeito 4) |

Fixtures em `tests/fixtures/caso_real_neg_*.png`, com
`tests/part2/test_caso_real_negativo.py`.

### 40.1 O caminho C: espelhar, não alargar a caixa

Alargar `K_BOUNDS` para `(-1e4, 1e4)` foi DESCARTADO por uma razão estrutural,
não de gosto: põe **K=0 dentro da caixa**, e K=0 é o modelo degenerado (resposta
plana com τ e θ livres). Cria um mínimo local trivial que hoje não existe, e
destrói o sinal de borda que o §38.5 quer usar como guarda.

O que entrou: se a resposta DESCE, nega-se `y` antes do Estágio D, ajusta-se com
o código atual intocado, e o `K` devolvido troca de sinal. É exatamente
equivalente a parametrizar `K = s·|K|` — `model_response` é linear em K e a base
é livre de sinal — mas não toca uma linha da matemática do módulo. `sse`, `nrmse`
e `aic` são invariantes ao espelho, então nenhuma métrica precisa de correção, e
com `s = +1` o caminho é byte a byte o anterior.

O OCR também precisou de conserto: o matplotlib desenha o menos como U+2212 e o
tesseract devolve EM DASH. Em `neg_sub`, **8 de 9 rótulos viravam `None`** no
`_NUM_RE` com os dígitos lidos CERTOS — a imagem saía com 1 par de eixo y em vez
de 9. `'='` ficou de fora da tabela de normalização de propósito: mapeá-lo seria
inventar leitura, e o RANSAC descarta o par perdido de graça.

### 40.2 O detector de direção: duas formulações refutadas por medição

Este é o resultado que importa levar adiante, porque as duas primeiras
formulações eram plausíveis e as duas estavam erradas — **cada uma quebrando
numa ponta diferente da série**.

**(1) Mediana do primeiro decil contra a do último.** Usa o primeiro decil como
proxy do nível de REPOUSO. O proxy morre quando o Estágio A come o platô inicial
(§39.3): o decil cai dentro do transitório, e numa subamortecida o transitório
passa ALÉM do valor final, pelo lado oposto. Errava **8 de 44** casos sintéticos
— todas as subamortecidas (ζ ≤ 0,4) sem cabeça — e era a causa de `neg_sub` não
fechar. O defeito era SIMÉTRICO: com degrau positivo e a mesma cabeça cortada
devolvia −1, espelhando uma subida. Não era problema de ganho negativo; mordia o
caminho positivo também.

Ajustar a FRAÇÃO do decil não conserta, e é importante que fique registrado: na
série sobrevivente o decil pousa onde a oscilação estiver, e a leitura oscila com
a fração — 0,05 → −1; 0,10 → +1; 0,20 → −1; 0,30 → +1 **na mesma imagem**.
Sintonizar isso seria calibrar um limiar contra um exemplo, que é o erro que o
comentário do `_UNDERSHOOT_MAX` já registra como cometido e pago neste projeto.

**(2) Extremo MAIS DISTANTE do valor assentado.** Corrige (1) e media 0 erros no
sintético. Mas pressupõe que o último decil é o valor FINAL, e não é quando a
janela acaba antes de assentar: numa 2ª ordem muito subamortecida o último decil
pousa no ringing, o primeiro pico fica mais longe dele que o repouso, e **o pico
é eleito repouso**. Espelhava **2 das 900** séries do oráculo do corpus — todas
com K > 0, logo duas amostras boas destruídas. Aparecia na Parte 1 como MAPE(K)
do estrato limpo `w<3` saindo de 0,000 % para 0,239 % (`sample_00307`, ζ=0,104;
`sample_00889`, ζ=0,121).

**(3) O que ficou: o repouso é o extremo que aparece PRIMEIRO.** Se o máximo vem
antes do mínimo, a série desceu. A informação está no TEMPO, não no valor — uma
resposta ao degrau nunca cruza de volta o nível de onde partiu (o sobressinal de
1ª/2ª ordem nunca ultrapassa 100 % do salto, com igualdade só no limite ζ→0),
então o repouso **é** um dos dois extremos, e o que o distingue do sobressinal é
a ORDEM. Ler a ordem dispensa saber onde a resposta assenta, que é justamente o
que uma janela curta esconde.

Medido: **0 erros em 44** no sintético (ζ e τ varridos, dois sinais, com e sem
cabeça) e **0 espelhos indevidos nas 900** do oráculo — as duas provas que
derrubaram (1) e (2). Imune a ruído nesta aplicação: 0 erros em 1600 séries com
σ até 0,30 sobre um salto de 2.

**Contrato e limite, medidos e asseverados.** Vale enquanto o repouso ainda
ESTIVER na série. Cortada a cabeça além dele, o primeiro extremo que sobra é um
sobressinal e a regra inverte — a direção passa a viver só no envelope decadente,
que exige ajuste, e isso é o que `identify` faz, não o que cabe num estimador de
pré-ajuste. Em ζ=0,2 e θ=2 s: corte em t₀ ≤ 2,2 s lê certo; t₀ ∈ [2,3; 2,7] lê
errado. Registrado em `test_o_limite_do_contrato_quando_o_repouso_sai_da_serie`
com `xfail` estrito.

### 40.3 Não-regressão: o caminho positivo não se moveu

| prova | resultado |
|---|---|
| `tests/test_part1.py` (portão do Estágio D, oráculo) | 25/25; `part1_metrics.md` sem UMA mudança de acurácia |
| critério 2.6 (degradação, n=300) | +1,63 p.p. — idêntico ao anterior |
| critério 2.12-ordem | 92,3 % (277/300) — idêntico |
| critérios 2.3 / 2.4 / 2.5 / 2.9 (n=900) | idênticos |
| oráculo do corpus (900 séries, todas K>0) | 0 espelhos indevidos |
| corpus extraído (900 imagens pelo Estágio A) | 1 espelho, em amostra já rejeitada por `resposta_inversa` antes e depois |

O critério 1.7 (throughput de geração) oscila entre execuções (3,49 → 3,66 →
8,83 s para 200 amostras) porque mede carga de máquina, não algoritmo. Não é
regressão e não deve ser lido como uma. O mesmo vale para 2.8 e G3b.4.

**O custo que existe, declarado: o extrator CLÁSSICO.** `2.6-classico-aceitas`
caiu de 195 para 193 em 300. A causa está medida: sob o extrator clássico
(Bloco 3b, sem rede), **32 das 295** séries do corpus (10,8 %) são lidas como
descendentes e portanto espelhadas — todas com K > 0, logo todas indevidamente.
A U-Net erra 1 em 900; o extrator clássico erra 32 em 295, porque produz máscara
muito mais suja e série suja confunde qualquer leitura de direção.

O que salva é o modo de falhar: **as 32 são recusadas com
`ajuste_inconsistente`, nenhuma produz saída física**. O espelho errado não vira
resposta confiante e errada — a série espelhada não sustenta modelo nenhum e a
guarda de NRMSE pega. O custo é 2 amostras que passaram de comparáveis a
recusadas, num caminho que é DIAGNÓSTICO (marcado ❓) e não o de produção.

Nenhum critério com alvo mudou de veredito: 48 ✅ e 36 ❓, antes e depois. O
caminho da U-Net — 2.6, 2.12, 2.6-adim, 2.9 — não se moveu em nenhum dígito.

### 40.4 O defeito 4 tem um número agora: a polilinha começa no objeto ERRADO

`neg_fopdt` continua recusada, e a causa é mais funda do que "a polilinha pula um
pouco". `rg_negativo.py` desenha o degrau de ENTRADA como tracejada branca no
mesmo quadro, e a série extraída **começa em y = −1,995** — o patamar da entrada.
O repouso da RESPOSTA (y ≈ 0) só aparece 38 amostras depois.

O dano é anterior ao Estágio D e contamina tudo o que vem depois: o mínimo
precede o máximo, `_sinal_do_degrau` lê subida, o espelho não dispara, e K trava
no piso (0,001, NRMSE 0,90). **Nenhum estimador de direção sobrevive a isso, e
nenhum deveria** — o dado está errado antes de chegar nele.

Isso corrige uma leitura anterior desta mesma sessão, feita com a formulação (2)
antes de ela ser descartada, de que a física de `neg_fopdt` saía certa por baixo
(K=−1,997, τ=0,4998, θ=2,998). Aquilo valia para (2), não para o que ficou. A
evidência de que o caminho C funciona é `neg_sub`, que fecha fim a fim a 0,55 %,
e o K de `neg_super` a 0,7 % — `neg_fopdt` nunca foi evidência de outra coisa
além do defeito 4.

Fechar exige distinguir dois objetos de curva no mesmo quadro: envelope novo,
spec própria. `test_neg_fopdt_a_polilinha_comeca_no_degrau_de_entrada` mede
`y[0] < −1,5` e a ordem dos extremos, então quem separar os objetos verá esse
teste falhar e os três `xfail` de recuperação virarem XPASS.

### 40.5 O que o Bloco 9 NÃO entregou

- **Treino.** O estrato de ganho negativo entrou no GERADOR
  (`generate_sample(..., ganho_negativo=True)`, §40.6) e há portão medindo o
  Estágio D nele, mas nenhum corpus de treino foi gerado e a U-Net **não viu
  ganho negativo**. O que o Estágio A faz nessas imagens continua apoiado em
  três exemplos reais.
- **`neg_super` (§39.3).** Segue com ωₙ, ζ e θ errados. Exige retreino do
  Estágio A, e o §37.11 já REFUTOU a guarda de cobertura de máscara como remédio
  (Spearman buraco × erro = +0,020, p = 0,57): o maior buraco do corpus é MAIOR
  que o desta imagem. Não repetir esse experimento.
- **`neg_fopdt` (defeito 4).** Ver §40.4.

### 40.6 O estrato de ganho negativo no gerador, e o portão que ele permitiu

`generate_sample(..., ganho_negativo=True)` — opt-in, no molde do
`reta_no_patamar` (§34.5), propagado por `_generate_one` e `generate_dataset`.
Só o SINAL de K muda: `sample_system` continua sorteando K > 0 e `|K|` fica
idêntico ao do mesmo seed sem o flag, o que torna o estrato comparável AMOSTRA A
AMOSTRA com o base. Opt-in e não sorteio, porque mexer em `sample_system` moveria
toda amostra do corpus base e com ela todo número histórico. Verificado hash a
hash: o corpus base é byte a byte o mesmo. O sinal é aplicado ao SPEC e nunca ao
estilo — `sample_style` não pode ver o spec (anti-vazamento), e um traço que
mudasse de cor por causa do sinal ensinaria a rede a ler o sinal do RENDER.

**O alvo de 95 % que o plano tinha escrito estava medindo a coisa errada**, e
vale registrar por quê. Ele exigia K recuperado a 5 % em ≥ 95 % do estrato. O
estrato mede 88,3 % — mas o caminho POSITIVO, histórico e sem espelho nenhum,
mede **90,0 % nos mesmos seeds**. O alvo estava capturando a dificuldade do
estrato de janela truncada (RULING C: MAPE(K) de 127 % a 20 dB), não o caminho C.

Os três portões que ficaram no lugar dele são mais fortes:

1. **Equivalência exata, série limpa.** O mesmo seed com ganho negativo devolve
   os mesmos parâmetros do positivo, com K de sinal trocado. Comparação por
   tolerância relativa (1e-9) e não bit a bit, porque a equivalência é
   MATEMÁTICA e não numérica: ajustar `-y` e ajustar `y` acumula somas em ordem
   diferente dentro do `least_squares`, e o desvio medido é ~1,5e-14.
2. **Recuperação total na série limpa:** 60/60. Alvo abaixo de 100 % aqui
   esconderia regressão.
3. **Paridade sob ruído**, e não valor absoluto: o negativo acompanha o positivo
   dentro de 5 p.p. A diferença de uma amostra em 60 é ESPERADA — o ruído é
   somado DEPOIS da inversão do sinal, então `-y_negativo = y_limpo - ruído`
   enquanto `y_positivo = y_limpo + ruído`, e nada obriga realizações diferentes
   a falharem nas mesmas amostras.

### 40.7 Ruling: o Estágio A PRECISA de retreino para ganho negativo, e a causa é um prior de POSIÇÃO

Experimento pareado, n=150 pares. Mesmo seed com e sem `ganho_negativo`, então
spec, estilo, janela e |K| são idênticos e a ÚNICA diferença é o sinal —
qualquer diferença de máscara é atribuível a ele por construção.

| métrica | positivo | negativo | Δ |
|---|---|---|---|
| IoU da máscara | 0,6217 (med 0,6471) | 0,5859 (med 0,6005) | −0,036 |
| cobertura de colunas | 0,9544 (med 0,9789) | 0,8663 (med 0,9062) | −0,088 |
| **cobertura do PLATÔ** | **0,9317 (med 0,9697)** | **0,5523 (med 0,5826)** | **−0,379** |

Por par: 84 de 150 pioram mais de 0,10 na cobertura do platô, contra 23 de 150
no IoU. O corpo da curva quase não sofre; **o platô de repouso desaba**.

**A causa não é "curva rente à moldura" (§39.3 defeito A).** `y_margin_lo` e
`y_margin_hi` saem os dois de `uniform(0.03, 0.15)` em `randomize.py:322-323`, e
o mesmo seed dá o mesmo estilo — o platô fica igualmente distante da SUA borda
nos dois sinais. O que muda é QUAL borda: com K > 0 o repouso fica no rodapé,
com K < 0 fica no topo.

**Também não é planura em si (§39.3 defeito B).** O platô do caminho positivo é
igualmente plano e sobrevive em 93 % das colunas. Planura dentro da distribuição
não quebra o Estágio A; planura em posição NUNCA VISTA quebra.

A conclusão é que a U-Net aprendeu um prior de POSIÇÃO VERTICAL: todo o corpus
tem K > 0, então ela nunca viu platô de repouso na metade de cima do quadro, e o
critério G3b.2 a treinou para suprimir reta horizontal de span completo. Uma reta
horizontal no alto do quadro é exatamente o que ela aprendeu a apagar.

**Isto responde a pergunta "precisa retreinar?" e separa as duas metades:**

- **Estágio D: NÃO.** O caminho C é exatamente equivalente ao caso positivo
  (§40.6, portão de equivalência a 1e-9). Não há nada a aprender.
- **Estágio A: SIM**, e por um motivo que só dado de treino conserta — prior
  aprendido de posição não se remedia com guarda nem com pós-processamento.

O estrato do §40.6 é o insumo: `generate_dataset(..., ganho_negativo=True)`
produz o dado de treino que falta. Nenhum corpus de treino foi gerado ainda.

**Confundimento RESOLVIDO por ablação fatorial 2x2 (n=80 seeds por célula).**
O eixo y foi invertido monkeypatchando `_axis_limits`, que alimenta tanto a
figura da imagem quanto a da máscara — a verdade fica consistente. Isso cruza
SINAL com ORIENTAÇÃO e move o platô independentemente dos dois:

| sinal | eixo | platô em | cob. do platô | cob. de colunas | IoU |
|---|---|---|---|---|---|
| K>0 | normal | **rodapé** | **0,9443** | 0,9556 | 0,6245 |
| K<0 | invertido | **rodapé** | **0,9398** | 0,9538 | 0,6220 |
| K<0 | normal | **topo** | **0,5274** | 0,8627 | 0,5920 |
| K>0 | invertido | **topo** | **0,5238** | 0,8572 | 0,5882 |

Os resultados agrupam por POSIÇÃO e ignoram o SINAL. Ganho positivo com eixo
invertido perde o platô exatamente como ganho negativo (0,5238 contra 0,5274,
dentro do ruído); ganho negativo com eixo invertido o preserva exatamente como o
positivo normal (0,9398 contra 0,9443).

**O sinal do ganho é IRRELEVANTE. O que a U-Net não sabe fazer é segmentar platô
de repouso na metade de cima do quadro, qualquer que seja o sinal.** O ganho
negativo não é o defeito — é só o que expôs o defeito.

Isso amplia o escopo do problema para além do Bloco 9: qualquer figura de eixo y
invertido cai nele, mesmo com K > 0, e essa é convenção corrente em parte da
engenharia. Não há nenhuma amostra assim no corpus.

E dá uma validação INDEPENDENTE para o retreino: treinar com ganho negativo
(platô no topo) e validar em positivo de eixo invertido (platô no topo, condição
NUNCA treinada). Se o retreino aprendeu posição em vez de decorar o estrato, as
duas melhoram juntas.

**E isto refina o §39.3.** Parte do que foi atribuído a "trecho reto" nas imagens
`neg_sub` e `neg_super` é, na verdade, este prior de posição — as duas são de
ganho negativo, e o platô que elas perdem está no topo. O retreino do §39.3 e o
retreino do ganho negativo são provavelmente O MESMO retreino.

### 40.8 O que a VALIDAÇÃO do retreino exige, medido antes de treinar

- **Receita do checkpoint promovido** (`logs/train_base32.log`): 9450 amostras de
  9 diretórios, `base=32`, 25 épocas, 1575 passos/época, ~703 s/época —
  **~4,9 h** nesta máquina. Melhor `IoU_val = 0,7551`.
- **`train_unet.py` não precisa mudar.** `--train-dir` e `--val-dir` são
  repetíveis e fazem glob de `sample_*`; o estrato novo entra como diretório.
- **LACUNA CRÍTICA: `data/val` (n=900) é 100 % K > 0, logo 100 % platô no
  rodapé.** `IoU_val` não mediria NADA do que o retreino pretende consertar, e o
  early-stopping selecionaria o checkpoint pelo critério errado. Um estrato de
  validação com platô no topo é pré-requisito, não opcional.
- **Bases de seed em uso:** 1 (`train`), 2 (`val`), 3 (`test`), 4
  (`train_extra`), 77–84 e 90001–90003 (estratos). Livres para o estrato novo:
  90004 em diante. `generate_dataset` deriva `seed·1_000_003 + i`, então bases
  distintas garantem conjuntos disjuntos — sem vazamento entre treino, validação
  e teste.
- **Baseline a bater, já medido** com o checkpoint atual sobre platô no topo:
  cobertura do platô **0,527**, cobertura de colunas **0,863**, IoU **0,592**.
  O alvo é chegar perto do que ele já faz no rodapé: 0,944 / 0,956 / 0,625.

### 40.9 O `IoU_val` é CEGO ao defeito do platô — e é ele que seleciona o checkpoint

Medido com o checkpoint promovido (`base=32`, `in_ch=3`, 7 768 947 parâmetros):

| conjunto de validação | n | IoU_val |
|---|---|---|
| `data/val` (todo K > 0, platô no rodapé) | 900 | 0,7814 |
| `data/val_kneg` (todo K < 0, platô no topo) | 300 | **0,7304** |
| `data/val` + `data/val_kneg` | 1200 | 0,7694 |
| `data/val` + `data/val_reta` | 1200 | 0,7572 |
| `data/val` + os três estratos de val | 1800 | 0,7429 |

**O estrato que a rede não sabe segmentar custa 5 pontos de IoU, enquanto a
cobertura do platô nele desaba 42 pontos (0,944 → 0,527, §40.7).** A razão é
geométrica: o platô é uma linha FINA, poucos pixels contra o corpo da curva.
IoU é dominado pelo corpo, então perder o platô inteiro quase não aparece nele.

Isso é um problema de PROCESSO, não de modelo. `train_unet.py:166` seleciona o
melhor checkpoint por `m > melhor` sobre o `IoU_val` — a métrica quase cega ao
defeito que o retreino existe para consertar. Somar `val_kneg` ao `--val-dir`
melhora pouco (0,7694 contra 0,7814): a diluição continua.

Consequência prática: a época que aprender o platô pode não ser a selecionada, e
a que for selecionada pode não ter aprendido. **Retreinar sem resolver isto é
gastar ~6 h numa loteria.**

Duas saídas, e a segunda é a recomendada por ser menos invasiva:

1. Plumbar cobertura do platô até a validação do `train_unet.py` e selecionar por
   ela (ou por combinação). Exige `MaskDataset` devolver `theta` e a afim, que
   hoje ela não devolve.
2. **Salvar um checkpoint por época** e escolher depois, pela métrica de platô
   que já existe (§40.7). Não toca métrica nenhuma, preserva o `IoU_val`
   histórico para comparabilidade, e custa ~31 MB × 25 = ~775 MB temporários.

**Baseline reconciliado, com uma ressalva.** O `logs/train_base32.log` reporta
melhor `IoU_val = 0,7551`, e `data/val` + `data/val_reta` dá 0,7572 no
checkpoint promovido — é quase certo que foi esse o conjunto de validação da
rodada. Os 0,0021 de diferença NÃO foram reconciliados; quem for comparar
números de época a época precisa fechar isso primeiro.

### 40.10 O que o *smoke test* do retreino encontrou (e por que ele existe)

Antes de comprometer ~6 h, o comando exato do retreino foi rodado com
`--epochs 2 --batches-per-epoch 3`. Encontrou **duas** coisas, e a segunda não
está escrita em lugar nenhum do projeto.

**1. `--batch 6`, não 8.** `logs/train_base32.log` não registra o batch, mas o
codifica: `passos = len(ds) // batch`, e 9450 amostras com **1575 passos** só
fecham com 6. O §HANDOFF_P2_7:312 tinha PREVISTO que `base=32` não caberia com
batch 8 e avisado que baixar o batch quebraria a comparabilidade com os pilotos
do `HANDOFF_P2_6.md` §3.3. O batch foi baixado e o registro disso se perdeu —
sobrou só a pegada na contagem de passos.

Isso tem consequência além do OOM: **os pilotos de §3.3 e o checkpoint promovido
NÃO são comparáveis em batch**, e qualquer conclusão que cruze os dois herda
essa diferença. Fica registrado aqui porque a próxima pessoa a ler o §3.3 não
tem como descobrir sozinha.

**2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** Registrado no §3.2
como necessário por causa dos 0,73 GB de folga, e ausente do script até o smoke
test. É opção de alocador — não toca em numérica, só em fragmentação.

Medido nesta sessão, com a GPU livre e nada mais rodando nela:

| configuração | resultado |
|---|---|
| batch 8, sem `expandable_segments` | OOM no backward, 5,42 de 5,64 GB |
| batch 8, com `expandable_segments` | OOM no backward, 5,55 de 5,64 GB (faltaram 16 MiB) |
| **batch 6, com `expandable_segments`** | **passa**, 2 épocas, checkpoints escritos |

O alocador sozinho não salva o batch 8. As duas coisas são necessárias.

**Custo revisado:** 10.950 amostras / 6 = **1825 passos/época** (contra 1575 da
rodada promovida), ~818 s/época, **~5,7 h**.

## 41. Ruling 64 — o retreino: o prior de posição caiu, e com ele o "defeito 4"

Rodada de 6h05, 25 épocas, `base=32`, `in_ch=3`, **batch 6**, 512²,
`expandable_segments:True`. 10.950 amostras dos 9 diretórios da rodada
promovida mais `data/train_kneg` (1500, platô no topo). Validação em
`data/val` + `data/val_reta` + `data/val_kneg`. Log em `logs/train_kneg.log`,
checkpoint por época em `models/epocas_kneg/` (não versionado).

**Promovido: época 13** (`IoU_val = 0,7618`, o que o próprio treino escolheu).

### 41.1 O defeito foi consertado, e a prova é a coluna NÃO TREINADA

| | topo treinado (K<0) | **topo NUNCA treinado** (K>0, eixo invertido) | rodapé (controle) |
|---|---|---|---|
| checkpoint anterior | 0,5433 | 0,5384 | 0,9299 |
| **época 13** | **0,9246** | **0,9174** | 0,9322 |

A coluna do meio é ganho POSITIVO com eixo y invertido — platô no topo, condição
que não existe em nenhuma amostra de treino. Ela subiu junto com a treinada, e
as duas andam paradas uma na outra em todas as 25 épocas. **A rede aprendeu
POSIÇÃO, não decorou o estrato.** O rodapé não pagou nada.

### 41.2 O "defeito 4" era o MESMO defeito, e a atribuição do §40.4 estava errada

`caso_real_neg_fopdt.png` era recusada, e o §40.4 concluiu que a causa era "dois
objetos de curva no mesmo quadro — envelope novo, spec própria", com a polilinha
começando sobre a tracejada de ENTRADA (y[0] = −1,995). Depois do retreino ela
**fecha**: K 0,15 %, τ 0,01 %, θ 0,02 %, e a série começa em y[0] = +0,005, sobre
a resposta.

O mecanismo descrito no §40.4 estava certo; a CAUSA estava errada. Não era a
existência de dois objetos: era o mesmo prior de posição do §40.7. A rede não
via o platô de repouso da resposta, que fica no topo, e a única linha visível
naquela altura era a da entrada. **Um retreino fechou os dois defeitos porque
eram um só.**

Lição de método: "dois objetos no mesmo quadro" era uma explicação plausível e
consistente com o sintoma, e passou porque ninguém pediu a ela que previsse mais
nada. O prior de posição foi encontrado por ablação fatorial, não por inspeção.

### 41.3 Efeito no corpus: melhor em quase tudo

| critério | antes | depois |
|---|---|---|
| **2.6** (degradação, pior parâmetro) | +1,63 p.p. | **+1,08 p.p.** |
| 2.6[ζ] / 2.6[ωₙ] | +1,63 / +1,00 | **+1,08 / +0,85** |
| 2.6-aceitas | 254/300 | **260/300** |
| **2.12** acerto de ordem | 92,3 % | **94,0 %** |
| 2.1 erro perpendicular p95 | 1,703 px | **1,489 px** |
| 2.11 amostras com valor | 292/300 | **295/300** |
| 2.9 cobertura da calibração | 0,933 | 0,933 |
| 2.1 mediana / 2.1-iou | 0,799 px / 0,6482 | 0,804 px / 0,6473 |

`IoU_val` medido nos mesmos conjuntos, para comparação honesta:

| | `data/val` (900) | `data/val`+`val_reta` (1200) | os três (1500) |
|---|---|---|---|
| anterior | 0,7822 | 0,7572 | 0,7519 |
| época 13 | 0,7821 | 0,7559 | **0,7618** |
| época 11 | 0,7742 | 0,7480 | 0,7541 |

Isso fecha a ressalva de comparabilidade: acrescentar `val_kneg` ABAIXA o IoU do
checkpoint anterior (0,7572 → 0,7519), então o 0,7618 da época 13 no mesmo
conjunto é ganho real de +0,0099, não artefato de composição.

**E inverte a recomendação que a seleção por platô sozinha produzia.** A época 11
tem o melhor platô (0,9272) mas custa **−0,0092** no conjunto histórico; a época
13 preserva os números históricos (0,7821 contra 0,7822 no `data/val`) e ainda
ganha no conjunto novo. Escolher pela métrica de platô sozinha era o mesmo erro
de método do §40.9 com o sinal trocado: **nenhuma das duas métricas decide
sozinha.**

### 41.4 O §40.9 estava certo como risco e exagerado como diagnóstico

O `IoU_val` cego escolheu a época 13 — **e escolheu certo**, porque ele carrega
exatamente a informação que a métrica de platô ignora. A época que o platô
preferia era pior no conjunto histórico. O risco do §40.9 é real, mas nesta
rodada a métrica acusada de cega foi a que protegeu o que o platô não vê.

Fica registrado também o que o §40.9 NÃO tinha notado: o `IoU_val` alimenta
**dois** consumidores, a seleção do checkpoint e o `ReduceLROnPlateau`. Nesta
rodada o `IoU_val` melhorou em 11 das 25 épocas e o scheduler cortou o LR **10
vezes**, porque os ganhos desta fase vêm em passos de 0,001 a 0,007 e o
`--lr-threshold` é 0,01 ABSOLUTO. O LR chegou a 2,34e-06 na época 17 e as
últimas 7 épocas não moveram nada. São dois problemas somados: limiar mal
calibrado para a escala dos ganhos, e métrica que não vê o fenômeno.

**O platô saturou na época 5** (0,9128, contra 0,9246 da 13). O conserto
aconteceu nas primeiras 5 épocas; as 13 últimas não moveram nem `IoU_val` nem
platô. Uma rodada de ~12 épocas teria bastado — e agora isso é medição.

### 41.5 O custo, e o que sobrou

**A cauda assentada com reta de referência coincidente piorou.** Em
`caso_real_2ordem.png` a máscara perde 51 das 747 colunas, todas nos dois
últimos decis. Não é limiar: probabilidade mediana **0,0004** nas colunas
perdidas, contra 0,1811 do checkpoint anterior — supressão confiante.
Sistêmico mas pequeno: `data/val_reta` isolado cai de 0,682 para 0,677.

Isso NÃO afeta o resultado físico: ωₙ sai 2,0203 contra 2,0220 do anterior, os
dois a ~1 % da verdade. Registrado em
`test_caso_real_cobre_a_cauda_assentada` com `xfail` estrito.

**Uma falsa regressão, e o erro de método que ela expôs.** O
`test_caso_real_recupera_zeta_e_wn` acusou 17 % de erro em ωₙ. Não havia erro
em ωₙ. O teste lia ωₙ como `wn_T / 10`, onde `wn_T` é normalizado pelo SPAN DA
SÉRIE e o 10 é a janela do eixo — a suposição de que a máscara cobre a janela
inteira. Com o span caindo de 9,81 s para 8,21 s, a conta erra 18 %.

Duas coisas estavam desatualizadas no teste: a premissa (`cal.ok` era False
quando ele foi escrito, hoje é True, então o bloco FÍSICO está disponível) e a
confusão entre COBERTURA e ACURÁCIA. Reescrito para assertar o físico — mais
forte, não mais fraco — com ζ seguindo no adimensional, que é invariante a
escala e a truncagem. A cobertura ganhou teste próprio.

**O que o retreino NÃO consertou:** os dois `xfail` do §39.3 em
`test_caso_real_rg.py` seguem valendo (sistema 1, 65,3 % de cobertura; sistema
3, 82,2 %). Coerente com o prior de posição: aquelas imagens são de ganho
positivo com `plt.ylim(0, ...)`, então o platô delas fica no RODAPÉ, que a rede
já dominava. E em `caso_real_neg_super.png`, ωₙ e ζ seguem errados (26 % e
30 %). **A causa afirmada aqui — "precisa da cauda, que a rede ainda perde" —
foi REFUTADA pelo §43:** a cauda tem rms 0,0073, está perfeita. É oclusão pela
legenda na faixa de acomodação. θ e K passaram a sair certos e viraram portão.

Suíte: `tests/part2/` **131 passam, 9 xfail, zero falhas**.

## 42. Ruling 65 — `K` é `K_planta × U`, e isso não é conserto de software

Reportado como classificação errada em `Figure_dn.png`: a pipeline devolveu
`K = −1,997` onde a função de transferência do `rg_negativo.py` diz `K = 1`.

**Não houve erro.** A planta é `2/(s+2)`, cujo ganho DC é `2/2 = 1`, e o degrau
aplicado tem amplitude **−2** (`amplitude_degrau=-2` no `rg_negativo.py`). A
curva desenhada é o produto: sai de 0 e assenta em −2,00, o que se lê no próprio
eixo y da figura. `STEP_AMPLITUDE = 1.0` é convenção do projeto, então o `K`
reportado é a excursão por unidade de entrada, `K_planta × U = 1 × (−2) = −2`.

Verificado reconstruindo a curva a partir dos parâmetros reportados contra a
verdade analítica: **erro máximo 0,0039, RMSE 0,0026**. `tau = 0,5` (erro
0,01 %), `theta = 2,999` contra 3,0 (erro 0,02 %; são os 2 s do instante do
degrau mais 1 s de atraso da planta). Estrutura FOPDT, correta.

### 42.1 Por que não é corrigível

Não é limitação de implementação, é **identificabilidade**. Da curva de saída
sozinha, `K_planta` e `U` não são separáveis — só o produto é observável:

| `K_planta` | `U` | curva observada |
|---|---|---|
| 1,997 | −1 | idêntica ponto a ponto |
| 0,999 | −2 | idêntica ponto a ponto |
| 0,499 | −4 | idêntica ponto a ponto |

Nenhum algoritmo distingue entre elas sem conhecer `U`. O leitor humano acerta
porque a figura **desenha a entrada** (a tracejada branca em −2) e ele divide;
a pipeline não lê a entrada.

### 42.2 O que fica registrado, e o que fica aberto

Registrado na especificação, nos dois lugares onde alguém tropeça:
`identify/classical.py` na definição de `STEP_AMPLITUDE`, e `ARQUITETURA.md` §4
("O que `K` significa, e o que ele NÃO significa").

**Aberto, e agora plausível:** ler a amplitude do degrau da imagem quando a
entrada está plotada. Com `U` lido, `K_planta = K_reportado / U` sai de graça.
Exige detectar e CLASSIFICAR um segundo objeto de curva como "entrada" — o que
era impossível antes do §41, e passou a ser plausível porque a máscara agora
separa a resposta da tracejada (é exatamente o que fechou o antigo "defeito 4").
Envelope próprio, spec própria.

**Aberto, barato:** o `identificar.py` imprime `K -1.997` sem dizer que é ganho
por degrau unitário. Um rótulo explícito na saída evitaria esta leitura sem
tocar em nada do cálculo. NÃO implementado — decisão do dono.

## 43. Ruling 66 — era a LEGENDA, e eu errei a atribuição duas vezes antes disso

`caso_real_neg_super.png` (`Figure_dl3.png`) devolvia wn=2,95 (erro 26 %) e
zeta=0,87 (erro 30 %). O dono moveu a legenda de `lower left` para
`upper right`, gerou a mesma figura de novo, e a pipeline passou a devolver
**wn=3,88 (2,97 %) e zeta=1,22 (2,23 %)**, com o NRMSE do ajuste caindo de
0,01207 para 0,00192.

### 43.1 A causa, medida por faixa

| faixa de t | original | legenda movida |
|---|---|---|
| platô inicial (0–3,3) | 0,0077 | 0,0077 |
| arranque (3,3–3,7) | 0,0090 | 0,0090 |
| transitório rápido (3,7–4,5) | 0,0201 | 0,0205 |
| **acomodação (4,5–6,0)** | **0,1674** | **0,0090** |
| cauda assentada (6–15) | 0,0073 | 0,0068 |

(rms do erro de extração contra a verdade analítica.)

**Uma faixa mudou, 19×. Todas as outras são idênticas.** A caixa da legenda em
`lower left` ocupa t 0,3..5,6 e y −2,4..−3,05, e a curva atravessa essa faixa
exatamente na acomodação. A polilinha segue a borda da caixa e cria um patamar
falso em −2,93 onde a resposta verdadeira ainda está em −2,52 — antecipando a
acomodação em ~0,7 s. O ajuste compensa com polo dominante mais lento e menos
amortecimento.

**O Estágio D está inocente, e isso foi medido:** o oráculo na MESMA grade de
593 pontos recupera wn=4,0000 e zeta=1,2500 com **NRMSE zero**. A amostragem é
suficiente; os 87 pontos da acomodação é que estão tortos.

### 43.2 Duas atribuições erradas, e o que as produziu

Esta imagem teve a causa errada atribuída duas vezes, as duas escritas no
repositório antes de serem refutadas:

1. **"Perde a cauda assentada"** (§41.5, e antes disso no `xfail`). Herdado do
   §39.3, que era sobre OUTRA imagem. A cauda tem rms 0,0073 — está perfeita, e
   não há buraco nenhum na cobertura.
2. **"Atração pela tracejada de entrada em −3"**. A polilinha era puxada para
   −2,93, mais negativa que a verdade, e "mais perto de −3" foi tratado como
   evidência da tracejada. Mas a caixa da legenda ocupava a MESMA vizinhança
   (y −2,4..−3,05). **A evidência disponível era compatível com as duas
   hipóteses, e uma foi escolhida sem o teste que as separa.**

O que resolveu foi um experimento de uma variável: mover a legenda. Nenhuma das
duas hipóteses anteriores sobrevive a ele.

Lição de método, e é a mesma do §41.2: hipótese consistente com o sintoma não é
hipótese confirmada. A pergunta que faltou nas duas vezes foi "qual observação
distinguiria isto da alternativa?".

### 43.3 O que muda no prognóstico

A `neg_super` **não precisa de retreino nem de estrato de cauda** — era o que o
`xfail` anterior afirmava. O ajuste recupera tudo quando a curva não está
ocluída. O que falta é o Estágio A atravessar a legenda.

**E o corpus já media isso, fraco demais para alguém agir.** O critério 2.7
estratificado por legenda está no relatório desde antes:

    2.7-iou[legenda=False]   0,6758  (n=155)
    2.7-iou[legenda=True]    0,6147  (n=145)

6,1 pontos de IoU, em quase metade do corpus. O número existia e nunca foi
ligado a nada, porque IoU DILUI: a legenda estraga um trecho pequeno da curva, e
o IoU é dominado pelo corpo. A imagem real mostrou o mesmo dano em unidade de
parâmetro físico — 26 % em wn. É o padrão do §40.9 outra vez, com outra métrica.

**Candidato para o próximo bloco, agora com número:** um estrato de legenda
SOBREPOSTA AO TRECHO DE ACOMODAÇÃO, em vez de legenda em posição sorteada. O
gerador já tem `has_legend`; o que falta é posicioná-la onde faz estrago.

### 43.4 O par controlado ficou versionado

`rg_negativo.py` gera as duas variantes, e `caso_real_neg_super_legenda_movida.png`
entrou como fixture ao lado da original. `test_neg_super_sem_oclusao_recupera_tudo`
é portão (K 0,20 %, wn 2,97 %, zeta 2,23 %, theta 0,06 %); o `xfail` da original
continua falhando, agora com a razão certa.

Enquanto os dois coexistirem, nenhuma explicação alternativa sobrevive: uma
variável muda, o resultado muda com ela.
