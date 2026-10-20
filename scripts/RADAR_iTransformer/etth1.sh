#!/usr/bin/env bash
# RADAR + plain iTransformer on ETTh1. Backbone hyper-parameters: official thuml/iTransformer
# scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh (+ run.py defaults, see itrans_common.sh).
# Usage: bash scripts/RADAR_iTransformer/etth1.sh   (PRED_LENS="96" for one horizon; knobs: scripts/RADAR/common.sh)
model_name=iTransformer
root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
seq_len=96
cycle=24
e3_period=24
RADAR_BS=${RADAR_BS:-128}
set_backbone_args() {  # d_model = d_ff = 256 for H 96 / 192, 512 for 336 / 720
  local d=256; case $1 in 336|720) d=512 ;; esac
  GTR_ARGS=("${ITR_BASE[@]}" --enc_in 7 --e_layers 2 --d_model $d --d_ff $d)
}
source "$(dirname "$0")/itrans_common.sh"
run_all
