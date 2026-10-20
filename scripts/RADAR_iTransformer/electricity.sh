#!/usr/bin/env bash
# RADAR + plain iTransformer on Electricity. Backbone hyper-parameters: official thuml/iTransformer
# scripts/multivariate_forecasting/ECL/iTransformer.sh (+ run.py defaults, see itrans_common.sh).
# Usage: bash scripts/RADAR_iTransformer/electricity.sh   (PRED_LENS="96" for one horizon; knobs: scripts/RADAR/common.sh)
model_name=iTransformer
root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
seq_len=96
cycle=168
e3_period=24
RADAR_BS=${RADAR_BS:-32}
set_backbone_args() { GTR_ARGS=("${ITR_BASE[@]}" --enc_in 321 --e_layers 3 --d_model 512 --d_ff 512 --batch_size 16 --learning_rate 0.0005); }
source "$(dirname "$0")/itrans_common.sh"
run_all
