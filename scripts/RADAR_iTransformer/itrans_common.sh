#!/usr/bin/env bash
# Shared settings of RADAR + plain iTransformer (thuml/iTransformer), sourced by the dataset scripts here, which
# define: root_path_name data_path_name model_id_name data_name seq_len cycle e3_period RADAR_BS and
# set_backbone_args <pred_len> (GTR_ARGS = ITR_BASE + the official per-dataset / per-horizon hyper-parameters).
#
# Backbone training (S0 = B0) follows the official iTransformer run.py defaults: Adam lr 1e-4, batch 32,
# 10 epochs, patience 3, lradj type1 (lr halves every epoch), dropout 0.1, n_heads 8, factor 1, gelu, timeF time
# features fed as x_mark_enc (None on Solar, as the official code), use_norm = 1; no OneCycleLR (GTR-only quirk).
# RADAR: same v6 plug-in and test conditions as the GTR runs; S2 shortened to at most PRE_EPOCHS (default 10)
# epochs with early stopping after PRE_PATIENCE (default 3) epochs without a val improvement; S2 / S3 batch
# RADAR_BS (S0 keeps the official batch).
# Results: ./results_itrans/ (separate from the GTR results), logs ./logs_radar_itrans, figures ./figures_radar_itrans.
ITR_BASE=(--features M --seq_len "$seq_len" --cycle "$cycle" --train_epochs 10 --patience 3 --dropout 0.1 --itr 1
          --batch_size 32 --learning_rate 0.0001 --lradj type1 --n_heads 8 --factor 1 --activation gelu
          --embed timeF --use_revin 1)
RESULTS_DIR=${RESULTS_DIR:-./results_itrans/}
LOG_DIR=${LOG_DIR:-./logs_radar_itrans}
RADAR_ARGS="${RADAR_ARGS:-} --pre_epochs ${PRE_EPOCHS:-10} --pre_patience ${PRE_PATIENCE:-3} --radar_batch_size ${RADAR_BS:-128} --fig_dir ./figures_radar_itrans/"
set_backbone_args 96
source "$(dirname "${BASH_SOURCE[0]}")/../RADAR/common.sh"
