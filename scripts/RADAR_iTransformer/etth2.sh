#!/usr/bin/env bash
# RADAR + plain iTransformer on ETTh2. Backbone hyper-parameters: official thuml/iTransformer
# scripts/multivariate_forecasting/ETT/iTransformer_ETTh2.sh (+ run.py defaults, see itrans_common.sh).
# Usage: bash scripts/RADAR_iTransformer/etth2.sh   (PRED_LENS="96" for one horizon; knobs: scripts/RADAR/common.sh)
model_name=iTransformer
root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
seq_len=96
cycle=24
e3_period=24
RADAR_BS=${RADAR_BS:-128}
set_backbone_args() { GTR_ARGS=("${ITR_BASE[@]}" --enc_in 7 --e_layers 2 --d_model 128 --d_ff 128); }
source "$(dirname "$0")/itrans_common.sh"
run_all
