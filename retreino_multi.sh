#!/usr/bin/env bash
# Retreino do Estagio A com o estrato MULTI-DEGRAU e a cabeca de contagem.
#
# NAO PROMOVE NADA: escreve em models/unet_stageA_multi.pt e guarda um
# checkpoint por epoca em models/epocas_multi/. A promocao e decisao separada,
# tomada DEPOIS de `seleciona_checkpoint_contagem.py` — pela mesma razao do
# retreino de ganho negativo: o IoU_val e cego ao defeito-alvo, e aqui ele e
# duplamente cego, porque nem sequer mede a contagem.
#
# ANTES DE RODAR, rode a sondagem barata:
#     .venv/bin/python sonda_cabeca_contagem.py
# Ela congela o checkpoint promovido e treina SO a cabeca sobre features
# cacheadas, em minutos. Se o acerto ja subir bem acima da taxa-base, as
# features atuais servem e este treino conjunto de ~8 h e dispensavel.
#
# Receita: a mesma do checkpoint promovido (base=32, in_ch=3, 25 epocas, os 10
# diretorios do `retreino_kneg.sh`) MAIS `data/train_multi`, com a cabeca
# ligada. `data/val_multi` entra no --val-dir para que o acerto de contagem
# apareca por epoca no log.
#
# DOIS DETALHES HERDADOS DO RETREINO ANTERIOR, e que sao a diferenca entre
# rodar e morrer com CUDA OOM no primeiro backward:
#
#   --batch 6, e NAO 8. Medido: batch 8 estoura os 5,64 GB desta GPU mesmo
#   livre. A cabeca de contagem acrescenta 184.513 parametros (0,6 % do
#   modelo) e uma ativacao (B, 512, 32) — desprezivel perto do decoder —, mas
#   NAO suba o batch por causa disso.
#
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Opcao de alocador: nao
#   toca em numerica, so em fragmentacao. Sozinha nao basta.
#
# --lambda-contagem 0.2 e um PONTO DE PARTIDA, nao um valor medido. As duas
# perdas tem escalas diferentes. Varra 0.05 / 0.1 / 0.2 / 0.5 e reporte o par
# (IoU_val, acerto_contagem) de cada — subir a contagem as custas do IoU nao e
# aprender a tarefa nova, e trocar uma pela outra.
#
# --lr-threshold 0.002, e NAO o default de 0.01. Esta e a UNICA divergencia
# deliberada da receita do `retreino_kneg.sh`, e ela foi medida, nao chutada.
#
# O `ReduceLROnPlateau` usa `threshold_mode="abs"`: com 0.01 ele exige UM PONTO
# INTEIRO de IoU para contar como melhora, e `patience=1` corta o LR pela
# metade depois de 2 epocas sem isso. Esse criterio foi calibrado para um
# treino que ia de ~0,50 a ~0,78 de IoU. Aqui o modelo parte de um checkpoint
# promovido e satura na casa de 0,75-0,76 na PRIMEIRA metade do treino, e o
# criterio passa a cortar o LR num regime em que a segmentacao ja convergiu
# — enquanto a CONTAGEM, que e o alvo deste retreino, ainda esta aprendendo.
#
# Medido na primeira tentativa (log em logs/train_multi_thr01_abortado.log,
# checkpoints em models/epocas_multi_thr01_abortado/):
#     ep03 IoU=0,7549  <- o scheduler trava o "melhor" aqui, barra = 0,7649
#     ep04 IoU=0,7447  ep05 IoU=0,7516  -> LR 3,0e-04 => 1,5e-04
#     ep06 IoU=0,7600  ep07 IoU=0,7644  -> LR 1,5e-04 => 7,5e-05
# O IoU subia MONOTONICAMENTE nas tres ultimas e mesmo assim virou "plato";
# na epoca 07 faltaram 0,0005 para a barra. Projetado, o LR cairia a ~9e-06 na
# epoca 13 e as 11 epocas finais (~3,1 h de GPU) nao produziriam nada.
#
# 0,002 e ~5x o ganho tipico por epoca observado nessa faixa (0,004 a 0,005),
# entao ainda distingue melhora real de ruido, sem tratar progresso como plato.
# CONSEQUENCIA A REGISTRAR: isto quebra a comparabilidade direta com o
# checkpoint promovido. E receita nova, e o relatorio tem de dizer isso.
#
# Custo estimado: 12.450 amostras / 6 = 2075 passos/epoca, ~1000 s/epoca, ~7 h.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p models/epocas_multi logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python train_unet.py \
  --base 32 --in-ch 3 --epochs 25 --batch 6 --size 512 \
  --contagem --lambda-contagem 0.2 \
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
  --val-dir data/val \
  --val-dir data/val_reta \
  --val-dir data/val_kneg \
  --val-dir data/val_multi \
  --save-epoch-dir models/epocas_multi \
  --out models/unet_stageA_multi.pt \
  2>&1 | tee logs/train_multi.log

echo
echo "Treino terminado. AGORA escolha o checkpoint pelas DUAS metricas:"
echo "  .venv/bin/python seleciona_checkpoint_contagem.py models/epocas_multi \\"
echo "      --referencia models/unet_stageA.pt"
