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
