#!/usr/bin/env bash
# Retreino do Estagio A com o estrato RUIDO ALTO.
#
# O GAP. `dataset/randomize.py` sorteia `snr_db` em U(20, 60) e nunca desce de
# 20 — medido nos quatro corpora, minimo 20,0 a 20,3 dB, mediana 40. Metade das
# amostras de treino e praticamente limpa. A rede aprendeu a segmentar curva
# NITIDA, e com ruido a curva vira uma FAIXA: ela precisa decidir onde dentro
# da faixa esta "a curva", decisao que nunca treinou.
#
# O DANO, medido em 100 figuras de avaliacao com ruido (acerto CONJUNTIVO no
# nivel PRATICO, ja com a guarda adaptativa do §69 ligada):
#     30 dB  90 %     25 dB  90 %     20 dB  70 %     15 dB  60 %     10 dB  55 %
# O joelho cai na borda da distribuicao de treino. Nao e coincidencia.
#
# O ALVO MEDIVEL: em `helpers/mede_ruido.py`, a faixa 5-20 dB subir e a faixa 20-60 dB
# NAO cair. Subir um as custas do outro e trocar de vies, nao aprender — mesmo
# criterio dos retreinos anteriores.
#
# O QUE ESTE RETREINO NAO RESOLVE. O ruido do estrato e GAUSSIANO BRANCO. Ruido
# de processo real costuma ser correlacionado — deriva e oscilacao lenta, o que
# se ve numa tela de trending como ondulacao alem do tremido rapido. O estrato
# cobre a parte facil, e quanto isso limita a transferencia nao foi medido.
#
# E A AVALIACAO PRECISA SER REFEITA COM SNRs QUE O TREINO NAO VIU. O lote atual
# (`reports/amostras_aleatorias/lote_ruido`) usa 30/25/20/15/10 dB e o treino
# cobre 5-20: ha sobreposicao. Depois de promover, gerar um lote com niveis
# intermediarios e semente nova:
#     .venv/bin/python gera_lote_ruidoso.py --seed <nova> --out .../lote_ruido_v2
# senao a medicao vira treino disfarcado.
#
# CUSTO. 12.450 + 1.800 (plato) + 1.800 (nmp) + 1.800 (ruido) = 17.850 amostras.
# Medido no retreino anterior: 1.194 s/epoca com 16.050. Escalando: ~1.330 s.
# 18 epocas = 6,6 h.  WORKERS=2 se a maquina estiver sob pressao de memoria.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-4}"
EPOCAS="${EPOCAS:-18}"

mkdir -p models/epocas_ruido logs

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
  --val-dir data/val \
  --val-dir data/val_reta \
  --val-dir data/val_kneg \
  --val-dir data/val_multi \
  --val-dir data/val_plato \
  --val-dir data/val_nmp \
  --val-dir data/val_ruido \
  --save-epoch-dir models/epocas_ruido \
  --out models/unet_stageA_ruido.pt \
  2>&1 | tee logs/train_ruido.log

echo
echo "Treino terminado. NAO promova pelo IoU_val — ele mistura SETE populacoes"
echo "e ja anticorrelacionou com a metrica real neste projeto. A selecao e um"
echo "passo separado:"
echo
echo "  .venv/bin/python helpers/seleciona_checkpoint_nmp.py models/epocas_ruido \\"
echo "      --referencia models/unet_stageA.pt"
echo "  .venv/bin/python helpers/mede_ruido.py --modelo models/epocas_ruido/<ep>.pt"
echo
echo "  objetivo: faixa 5-20 dB subir em helpers/mede_ruido.py"
echo "  guardas : faixa 20-60 dB nao cair, plato em val_nmp e val nao cairem,"
echo "            IoU em val+val_multi nao cair"
echo
echo "  Compare com reports/baseline_pre_ruido.txt, do mesmo script."
echo
echo "DEPOIS de promover, gerar lote de avaliacao com SNRs INEDITOS (ver o"
echo "comentario no topo deste arquivo) e so entao citar numero novo."
