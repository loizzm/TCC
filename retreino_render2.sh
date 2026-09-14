#!/usr/bin/env bash
# Retreino do Estagio A com o estrato TERCEIRA FAMILIA DE RENDER (§73).
#
# O GAP, medido com o mesmo instrumento dos dois lados (fracao das colunas em
# que a mascara acende a +-5 px da curva VERDADEIRA):
#     rg_aleatorio (avaliacao)    p50 = 0,713    79 % abaixo de 80 %
#     dataset/generator (treino)  p50 = 0,925    29 % abaixo de 80 %
#     Mann-Whitney p = 1,05e-12
# A rede treina numa familia de desenho e e' avaliada em outra, e perde um
# quinto da curva so por isso.
#
# COMO O DIAGNOSTICO CHEGOU AQUI (lote_k_menor1, 75 % ESTRITO):
#   - o Estagio D acerta 100/100 com a serie analitica -> nao e' o ajustador;
#   - a calibracao erra 0,24 % em X e a MELHOR afim possivel PIORA (77 % -> 74 %)
#     -> nao e' o Estagio B;
#   - a forma da serie extraida e' 30x pior nas reprovadas (p = 1,1e-04)
#     -> e' o Estagio A;
#   - atribuindo coluna a coluna: culpa da MASCARA 10,9 %, da polilinha 3,5 %;
#     a mascara domina em 17 das 23 figuras.
#
# SEIS DECOMPOSICOES NEGATIVAS, para ninguem refazer: |K| nao prediz a falha
# (p = 0,68); espessura (p = 0,30), dpi (p = 0,17) e contraste (que ate vai na
# direcao contraria) nao separam; a perda e' UNIFORME na figura (33 % repouso,
# 33 % transitorio, 40 % cauda) e sem efeito de moldura; desligar linha de
# entrada, preenchimentos E o tema de uma vez nao recupera nada; e a geometria
# da avaliacao cai 100 % dentro de [p1, p99] do treino.
#
# O QUE O ESTRATO E, E COMO ELE FOI CALIBRADO. Uma TERCEIRA familia — nem a do
# corpus, nem a do `rg_aleatorio`, cujos dois temas ficaram proibidos por teste
# (treinar no render de avaliacao transformaria os lotes de controle em
# memorizacao). A primeira versao so mexia em temas e FALHOU na verificacao:
# ficou a 0,903 de cobertura contra 0,944 do treino, p = 0,23, indistinguivel.
# O que resolveu foi medir APARENCIA em vez de desempenho:
#     medida        familia 1   v1 (so tinta)   v2 (painel)   rg que FALHAM
#     tinta            0,021        0,024          0,374          0,380
#     entropia         0,288        0,443          1,265          1,239
#     distancia p25    3,32         3,07           2,25           2,23
# `_new_figure` pintava a figura E o painel dos eixos com a MESMA cor, entao o
# corpus nunca produziu o contraste painel/moldura que o `seaborn-darkgrid` da
# avaliacao produz — sao ~40 % dos pixels, e era a lacuna inteira.
#
# A COMPOSICAO MUDA UM ESTRATO SO. `train_legenda` existe e fica de fora de
# proposito: o retreino anterior misturou frentes e o ganho ficou inatribuivel.
#
# CUSTO. 17.850 + 1.800 = 19.650 amostras. Medido: 1.194 s/epoca com 16.050.
# Escalando, ~1.460 s. 18 epocas = 7,3 h.  WORKERS=2 sob pressao de memoria.
set -euo pipefail
cd "$(dirname "$0")"

WORKERS="${WORKERS:-4}"
EPOCAS="${EPOCAS:-18}"

mkdir -p models/epocas_render2 logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python train_unet.py \
  --base 32 --in-ch 3 --epochs "$EPOCAS" --batch 6 --size 512 \
  --workers "$WORKERS" \
  --lr-threshold 0.002 \
  --train-dir data/train \
  --train-dir data/train_reta \
  --train-dir data/train_janela \
  --train-dir data/train_banda_seta \
  --train-dir data/train_banda \
  --train-dir data/train_seta \
  --train-dir data/train_reta_banda \
  --train-dir data/train_reta_seta \
  --train-dir data/train_reta_banda_seta \
  --train-dir data/train_kneg \
  --train-dir data/train_multi \
  --train-dir data/train_plato \
  --train-dir data/train_plato_kneg \
  --train-dir data/train_nmp \
  --train-dir data/train_nmp_kneg \
  --train-dir data/train_ruido \
  --train-dir data/train_ruido_kneg \
  --train-dir data/train_render2 \
  --train-dir data/train_render2_kneg \
  --val-dir data/val \
  --val-dir data/val_reta \
  --val-dir data/val_kneg \
  --val-dir data/val_multi \
  --val-dir data/val_plato \
  --val-dir data/val_nmp \
  --val-dir data/val_ruido \
  --val-dir data/val_render2 \
  --save-epoch-dir models/epocas_render2 \
  --out models/unet_stageA_render2.pt \
  2>&1 | tee logs/train_render2.log

echo
echo "NAO promova pelo IoU_val — ele mistura OITO populacoes e ja"
echo "anticorrelacionou com a metrica real neste projeto. A selecao e um passo"
echo "separado, e ela NAO usa os lotes de controle:"
echo
echo "  .venv/bin/python mede_render2.py --modelo models/epocas_render2/<ep>.pt"
echo
echo "  ALVO  : lote_selecao (familia rg, semente nova, so para escolher epoca)"
echo "          sobe de p50 = 0,716"
echo "  APRENDEU? val_render2 sobe de p50 = 0,810 — se nao subir, o treino nao"
echo "          pegou o estrato e nao adianta olhar o alvo"
echo "  GUARDA: data/val NAO cai de p50 = 0,939"
echo
echo "  Compare com reports/baseline_pre_render2.txt, do mesmo script."
echo
echo "DEPOIS de escolher, e so entao, medir ponta a ponta nos lotes de CONTROLE"
echo "(k_maior1, k_menor1, ruido) e rodar tests/part2. Os controles ficam fora"
echo "da selecao para o numero final continuar sendo generalizacao."
echo
echo "E REMEDIR AS CONSTANTES A JUSANTE. _UNDERSHOOT_MAX, _K_SIGMA,"
echo "_PERSISTENCIA_MIN e SALTO_MAX_ESPESSURA foram calibrados contra a mascara"
echo "PROMOVIDA e nao transferem entre checkpoints — cada um diz isso no proprio"
echo "comentario."
