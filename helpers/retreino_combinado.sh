#!/usr/bin/env bash
# Retreino COMBINADO: terceira familia de render (§73) + par de oclusao por
# legenda (§74), com peso maior.
#
# ESTE ARQUIVO E' UM EXPERIMENTO COM PREDICAO REGISTRADA. Leia antes de rodar.
#
# O QUE JA SE SABE, medido:
#   1. O estrato `familia_alt` FUNCIONA no que se propos. O retreino anterior
#      (`helpers/retreino_render2.sh`) levou o ESTRITO de 77 % para 81 % nos tres lotes
#      de controle e o |K|<1 de 75 % para 82 % — a meta que nenhum caminho de
#      codigo tinha alcancado. Pareado: 26 melhoram, 14 pioram, p = 0,08.
#   2. E ele QUEBRA cinco portoes de imagem real:
#        test_caso_real_negativo::test_neg_super_recupera_a_dinamica[wn]
#        test_caso_real_negativo::test_neg_super_recupera_a_dinamica[zeta]
#        test_caso_real_rg::test_sistema1_nao_inventa_polo_irresoluvel
#        test_caso_real_rg::test_sistema1_recupera_K_tau_theta
#        test_caso_real_rg::test_estagio_a_cobre_a_janela_inteira[caso2-sistema3]
#      Por isso a epoca 17 foi promovida e REVERTIDA.
#   3. Os dois primeiros sao OCLUSAO, isolada por par controlado: com a legenda
#      sobre a curva, wn/zeta a 19,7 %/18,1 %; com ela movida, 2,1 %/1,5 %. O
#      retreino de RUIDO produziu o MESMO defeito (32,1 %/32,2 %) com corpus
#      diferente — duas corridas independentes, mesmo mecanismo.
#
# A HIPOTESE: existe tensao entre COBERTURA e SELETIVIDADE. Um Estagio A mais
# agressivo ganha cobertura em render dificil e passa a fechar buracos
# horizontais — inclusive os que a caixa da legenda cria. O par de oclusao
# ensina que a caixa NAO muda o alvo (a mascara e' identica nas duas metades),
# e deveria impedir a troca.
#
# ======================================================================
# A PRIMEIRA CORRIDA RODOU, E A PREDICAO FOI FALSIFICADA. O que vem abaixo
# e' a SEGUNDA, com o corpus corrigido. Historico, medido:
#
#   A corrida morreu na epoca 12 de 18 (a sessao caiu e levou o processo).
#   Dos 13 checkpoints, UM passa o portao de oclusao: a epoca 10.
#   CONTROLE: a corrida `render2`, mesmo objetivo SEM o estrato pareado,
#   passa 3 de 18. Fisher p = 0,62 — o par nao mudou nada.
#   A epoca 10 entrega +1,7 pp de ESTRITO nos tres lotes (230 -> 235 de 300,
#   18 sobem / 13 descem, p = 0,47) e fica no piso exato da guarda
#   (`val_nmp[:60]` = 90,0 %, folga zero). Nao foi promovida.
#
# POR QUE FALHOU — medido, e e' um defeito do GERADOR, nao do treino.
# A epoca 10 nao e' anomalia: ordenando os 14 checkpoints pela DEFORMACAO DA
# SERIE na janela da transicao do par real, a ordem e' limpa e os dois unicos
# abaixo de 9 px sao os dois unicos que passam (promovido 6,8 px; epoca 10
# 8,0 px; epoca 11 9,97 px e ja quebra). O portao e' um corte binario numa
# variavel continua, e a epoca 11 perdeu por 2 pixels.
# O `theta` e' o pivo: >= 3,57 s da `fopdt`, <= 3,48 s da `second` com zeta
# 0,75-0,96, e a faixa inteira entre os checkpoints e' de 0,18 s, 1,8 % da
# janela.
#
# E o corpus pareado nunca apresentou esse caso. Medido nos 200 pares de
# `val_parleg_*`: a deformacao na mesma janela deu p50 = 0,0 px e p90 de 1,0 a
# 2,0 px nos SETE checkpoints medidos — chao da escala, duas ordens de grandeza
# abaixo da figura real. A causa e' a ancora: `theta + U(0,6; 3,0)*t_dom` poe a
# caixa no patamar assentado, nao na transicao. O joelho caiu DENTRO da caixa
# em 6 de 193 pares (3,1 %), a 298 px (p50) dela nos outros, e em 12 % das
# amostras a ancora era grampeada na borda do quadro para qualquer sorteio.
# Sem dano no corpus nao ha o que aprender NEM o que medir — e e' por isso que
# os quatro instrumentos sinteticos nao ordenavam.
#
# O QUE MUDOU NO GERADOR (`dataset/generator.py`, `_OCLUSAO_CHEGADA`):
#   1. A referencia passou a ser a CHEGADA ao patamar, nao `theta + k*t_dom`.
#      `t_dom` nao normaliza entre um transitorio de 5 % da janela (a figura
#      real: 33 px de 577) e um de 60 % (o corpus).
#   2. O deslocamento passou a ser em MEIAS-LARGURAS DA CAIXA, medidas depois
#      do layout. Era erro de UNIDADE: a caixa e' objeto de layout, `t_dom` e'
#      escala da dinamica, e na figura real a caixa tem NOVE `t_dom` de largura.
#   3. A caixa ficou mais larga (`_OCLUSAO_N_TEXTOS` 1-3 -> 2-4 textos).
#
# MEDIDO DEPOIS DA CORRECAO, em 60 pares, contra o corpus antigo e a real:
#                                  antigo   corrigido   figura real
#   curva tapada pela caixa (p50)   0,051     0,080        0,1505
#   deformacao na chegada (p50)      3,5       4,1          6,8  (promovido)
#                                    3,0       4,8          8,0  (epoca 10)
# O corpus continua mais fraco que a figura real, mas agora na mesma ordem de
# grandeza — e, pela primeira vez, ordena os dois checkpoints na MESMA DIRECAO
# que o par real (epoca 10 acima do promovido), onde o antigo INVERTIA. Com
# n=2 isso e indicio, nao instrumento validado.
#
# A PREDICAO DESTA CORRIDA, registrada ANTES de rodar:
#     o ganho do render2 se mantem (ESTRITO total ~81 %, |K|<1 ~82 %)
#     E a taxa de checkpoints que passam o portao de oclusao sobe acima dos
#     3/18 do controle `render2` e dos 1/13 da corrida anterior.
# A segunda metade e' o teste do corpus corrigido, e e' falsificavel: se a taxa
# ficar de novo em torno de 1 em 15, a conclusao e' que a oclusao sintetica nao
# transfere nem depois de reproduzir a geometria certa, e o unico caminho que
# resta e' conseguir FIGURAS REAIS com legenda sobre a curva.
#
# O QUE ESTE EXPERIMENTO AINDA NAO RESPONDE. O portao real continua com n=1.
# A taxa de sobrevivencia ao longo das epocas e' o que da poder aqui, nao o
# resultado de um checkpoint.
#
# PESO. O render2 sozinho era 1.800 de 19.650 (9 %) e as epocas tardias o
# REABSORVERAM — o ganho no alvo caiu de +0,029 na epoca 03 para +0,009 na 17,
# monotonicamente. Com o par (3.000) o corpus novo vai a 4.800 de 22.650, ou
# 21 %. Se o padrao se repetir mesmo assim, o problema nao e' peso.
#
# CUSTO. 22.650 amostras. Medido: 1.457 s/epoca com 19.650. Escalando, ~1.680 s.
# 18 epocas = 8,4 h.  WORKERS=2 sob pressao de memoria.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-4}"
EPOCAS="${EPOCAS:-18}"

mkdir -p models/epocas_combinado2 logs

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
  --train-dir data/train_parleg2_canto \
  --train-dir data/train_parleg2_ocl \
  --val-dir data/val \
  --val-dir data/val_reta \
  --val-dir data/val_kneg \
  --val-dir data/val_multi \
  --val-dir data/val_plato \
  --val-dir data/val_nmp \
  --val-dir data/val_ruido \
  --val-dir data/val_render2 \
  --val-dir data/val_parleg2_ocl \
  --save-epoch-dir models/epocas_combinado2 \
  --out models/unet_stageA_combinado2.pt \
  2>&1 | tee logs/train_combinado2.log

echo
echo "================== COMO AVALIAR, NESTA ORDEM =================="
echo
echo "NAO promova pelo IoU_val. Ele mistura NOVE populacoes, ja anticorrelacionou"
echo "com a metrica real neste projeto, e na corrida anterior a epoca de melhor"
echo "IoU_val (16) nao foi a melhor ponta a ponta (17)."
echo
echo "1. PORTOES DE IMAGEM REAL — eliminatorio, e vem PRIMEIRO."
echo "   Foi o que reprovou a corrida anterior DEPOIS de ela passar em todas as"
echo "   metricas agregadas. Varra os 18 checkpoints:"
echo "     .venv/bin/python helpers/varre_legenda_epocas.py models/epocas_combinado2 \\"
echo "         --referencia models/unet_stageA_pre_render2_backup.pt"
echo "   Quem 'quebra' esta fora, por melhor que seja no resto."
echo
echo "2. ALVO, so nos sobreviventes:"
echo "     .venv/bin/python helpers/mede_render2.py --modelo models/epocas_combinado2/<ep>.pt"
echo "   lote_selecao sobe de p50 = 0,716 | val_render2 sobe de 0,810"
echo "   GUARDA: data/val nao cai de 0,939"
echo
echo "3. PORTAO DA GUARDA, no finalista:"
echo "     .venv/bin/python helpers/mede_fora_da_familia.py --modelo <ep>"
echo "   recusa em val_nmp[:60] >= 90 % — a epoca 03 e a 13 da corrida anterior"
echo "   reprovaram aqui (89 % e 88,3 %), e nenhum valor de _PERSISTENCIA_MIN"
echo "   recupera, porque o termo so pode REDUZIR disparo."
echo
echo "4. So entao os tres lotes de CONTROLE, e a suite inteira de tests/part2."
echo
echo "5. SE PROMOVER: remedir _UNDERSHOOT_MAX, _K_SIGMA, _PERSISTENCIA_MIN e"
echo "   SALTO_MAX_ESPESSURA. Todos foram calibrados contra a mascara promovida"
echo "   e cada um diz no proprio comentario que nao transfere. Medido na"
echo "   corrida anterior: com a mascara da epoca 17 o plato do _PERSISTENCIA_MIN"
echo "   estreitou de 0,05 para 0,03 de folga."
