#!/usr/bin/env bash
# Retreino do Estagio A com o estrato de plato no topo (HANDOFF_P2_7 secao 40.7).
#
# NAO promove nada: escreve em models/unet_stageA_kneg.pt e guarda um
# checkpoint por epoca em models/epocas_kneg/. A promocao e decisao separada,
# tomada DEPOIS de `seleciona_checkpoint.py` — porque o IoU_val e quase cego ao
# defeito do plato (secao 40.9) e nao pode ser quem escolhe.
#
# Receita: a mesma do checkpoint promovido (base=32, in_ch=3, 25 epocas, os 9
# diretorios de treino de `logs/train_base32.log`) MAIS `data/train_kneg`.
# O `data/val_kneg` entra no --val-dir para o IoU_val ao menos VER o estrato,
# ainda que o registre mal (secao 40.9).
#
# DOIS DETALHES QUE O SMOKE TEST ENCONTROU, e que nao estavam escritos em lugar
# nenhum. Sem os dois, o treino morre com CUDA OOM no primeiro backward:
#
#   --batch 6, e NAO 8. O log da rodada promovida nao registra o batch, mas o
#   codifica: `passos = len(ds) // batch`, e 9450 amostras com 1575 passos so
#   fecham com 6. O HANDOFF_P2_7:312 previu que base=32 nao caberia com batch 8
#   e avisou que baixar o batch quebraria a comparabilidade com os pilotos de
#   HANDOFF_P2_6 §3.3 — o batch foi baixado e o registro disso se perdeu.
#   Medido nesta sessao: batch 8 estoura 5,64 GB de VRAM mesmo com a GPU livre.
#
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, registrado no
#   HANDOFF_P2_7:97 como necessario por causa dos 0,73 GB de folga. E opcao de
#   alocador: nao toca em numerica, so em fragmentacao. Sozinha NAO basta —
#   medido, com batch 8 ainda estoura, por 16 MiB.
#
# Custo estimado: 10.950 amostras / 6 = 1825 passos/epoca, ~818 s/epoca, ~5,7 h.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p models/epocas_kneg logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python train_unet.py \
  --base 32 --in-ch 3 --epochs 25 --batch 6 --size 512 \
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
  --val-dir data/val \
  --val-dir data/val_reta \
  --val-dir data/val_kneg \
  --save-epoch-dir models/epocas_kneg \
  --out models/unet_stageA_kneg.pt \
  2>&1 | tee logs/train_kneg.log

echo
echo "Treino terminado. AGORA escolha o checkpoint pela metrica certa:"
echo "  .venv/bin/python seleciona_checkpoint.py models/epocas_kneg --tambem models/unet_stageA.pt"
