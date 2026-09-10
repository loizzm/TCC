#!/usr/bin/env bash
# Retreino do Estagio A com os estratos PLATO NO MEIO e FORA DA FAMILIA.
#
# O QUE ESTE RETREINO ATACA, e o que ele NAO ataca.
#
# A rede aprendeu, por acidente do corpus, que curva de dados e "aproximacao
# monotona a um patamar" — nenhuma outra forma existia para ela ver. Medido com
# `models/unet_stageA.pt` (base 32, promovido), em `mede_plato_repouso.py` e
# `mede_fora_da_familia.py`:
#
#   corpus         cobertura do plato   recall da mascara   veredito
#   data/val               0,884              0,784         92 % ok
#   data/val_plato         0,861                 —              —
#   data/val_nmp           0,286              0,729         55 % recusa
#
# O `plato_no_meio` cobre um GAP REAL de enquadramento — o repouso nunca cai
# longe da borda no corpus (p05-p95 de 0,038 a 0,137 em `data/train`, 0 % na
# faixa do meio) —, mas ele NAO e a causa do defeito: 0,861 contra 0,884 e
# empate. Entra porque e barato e porque cobrir o gap e certo, nao porque cura.
#
# Quem cura e o `fase_nao_minima`. Ele e o UNICO estrato que sai da familia de
# modelos de proposito: dentro dela a resposta e continua em t = theta e vale
# zero ali, entao NENHUM estrato in-family produz a descontinuidade que falta.
#
# O ALVO MEDIVEL: `data/val_nmp` subir (cobertura do plato de 0,286, taxa de
# recusa por `resposta_inversa` de 55 %) e `data/val` NAO cair (0,884 / 92 %).
# Subir um as custas do outro e trocar de vies, nao aprender — mesmo criterio
# do `seleciona_checkpoint.py`.
#
# O QUE ESTE RETREINO NAO RESOLVE: o limiar `_UNDERSHOOT_MAX`. Ele foi
# remedido contra conjunto positivo real (200 amostras) e FICA em 0,08; descer
# para 0,02 troca 28 recusas certas por 28 erradas. O teto e a mascara, nao o
# limiar — por isso o limiar so deve ser remedido DEPOIS deste treino.
#
# A CABECA DE CONTAGEM NAO VE O ESTRATO NOVO. `train_unet.py` zera o peso da
# perda de contagem nas amostras `fora_da_familia`: elas tem `n_degraus = 1` e
# uma FORMA de dois eventos (mergulha, depois sobe), que e exatamente o padrao
# que a cabeca aprendeu a chamar de "mais de um degrau". A perda de segmentacao
# continua cheia; so a de contagem se cala.
#
# CUSTO. Medido no retreino multi: 922 s/epoca com 12.450 treino + 1.900
# validacao. Escalando para 16.050 + 2.500: ~1.190 s/epoca. 25 epocas = 8,3 h,
# 18 epocas = 6,0 h. Naquele retreino o melhor apto foi a epoca 08, o IoU
# platoou na 13 e o LR caiu a 9,37e-06 na 17 — as epocas 14-24 nao produziram
# nada que fosse selecionado.
#
# MEMORIA. Esta maquina tem 16 GB e nao e' CI. Medido com a receita cheia, o
# pico de RSS da arvore de treino e' 3,6 GB com os 4+2 workers padrao e 2,6 GB
# com WORKERS=2. Se a sessao estiver com swap alto, o treino e' o maior
# processo e portanto o alvo do OOM killer — e morrer na hora 5 custa a hora 5
# inteira. O numero de workers NAO muda resultado nenhum: a ordem das amostras
# vem do sampler no processo principal, com `torch.manual_seed` fixo.
#   WORKERS=2 ./retreino_nmp.sh
set -euo pipefail
cd "$(dirname "$0")"

WORKERS="${WORKERS:-4}"
EPOCAS="${EPOCAS:-25}"

mkdir -p models/epocas_nmp logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python train_unet.py \
  --base 32 --in-ch 3 --epochs "$EPOCAS" --batch 6 --size 512 \
  --workers "$WORKERS" \
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
  --train-dir data/train_plato \
  --train-dir data/train_plato_kneg \
  --train-dir data/train_nmp \
  --train-dir data/train_nmp_kneg \
  --val-dir data/val \
  --val-dir data/val_reta \
  --val-dir data/val_kneg \
  --val-dir data/val_multi \
  --val-dir data/val_plato \
  --val-dir data/val_nmp \
  --save-epoch-dir models/epocas_nmp \
  --out models/unet_stageA_nmp.pt \
  2>&1 | tee logs/train_nmp.log

echo
echo "Treino terminado. NAO promova pelo IoU_val — ele mistura seis populacoes"
echo "e ja anticorrelacionou com a metrica real neste projeto (Spearman -0,401"
echo "depois da epoca 08 do retreino multi): train_unet.py teria escolhido o"
echo "checkpoint PIOR. A selecao e um passo separado, com objetivo e guardas:"
echo
echo "  .venv/bin/python seleciona_checkpoint_nmp.py models/epocas_nmp \\"
echo "      --referencia models/unet_stageA.pt --auc-minima 0.6838"
echo
echo "  objetivo: cobertura do plato em data/val_nmp (baseline 0,286)"
echo "  guardas : plato em data/val (0,884), IoU em val+val_multi,"
echo "            e AUC de contagem real >= 0,6838 — a barra e ABSOLUTA porque"
echo "            o modelo promovido nao tem cabeca de contagem; 0,6838 e o"
echo "            melhor apto do retreino multi (epoca_08)."
echo
echo "  Compare contra reports/baseline_pre_nmp.txt, tirado com o mesmo script."
echo
echo "E DEPOIS de promover — nunca antes — remedir o limiar da guarda:"
echo "  .venv/bin/python remede_undershoot.py"
echo "  (o 0,08 atual foi varrido contra conjunto positivo real e FICA ate que"
echo "   a mascara mude; ver o bloco de _UNDERSHOOT_MAX em identify/pipeline.py)"
