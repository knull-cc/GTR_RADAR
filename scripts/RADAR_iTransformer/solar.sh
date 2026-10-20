#!/usr/bin/env bash
# RADAR + plain iTransformer on Solar. Backbone hyper-parameters: official thuml/iTransformer
# scripts/multivariate_forecasting/SolarEnergy/iTransformer.sh (+ run.py defaults, see itrans_common.sh).
# Usage: bash scripts/RADAR_iTransformer/solar.sh   (PRED_LENS="96" for one horizon; knobs: scripts/RADAR/common.sh)
model_name=iTransformer
root_path_name=./dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar
seq_len=96
cycle=144
e3_period=144
RADAR_BS=${RADAR_BS:-64}
set_backbone_args() { GTR_ARGS=("${ITR_BASE[@]}" --enc_in 137 --e_layers 2 --d_model 512 --d_ff 512 --learning_rate 0.0005); }
source "$(dirname "$0")/itrans_common.sh"
run_all
