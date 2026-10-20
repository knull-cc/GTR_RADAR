#!/usr/bin/env bash
# RADAR + plain iTransformer on weather. Backbone hyper-parameters: official thuml/iTransformer
# scripts/multivariate_forecasting/Weather/iTransformer.sh (+ run.py defaults, see itrans_common.sh).
# Usage: bash scripts/RADAR_iTransformer/weather.sh   (PRED_LENS="96" for one horizon; knobs: scripts/RADAR/common.sh)
model_name=iTransformer
root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
seq_len=96
cycle=144
e3_period=144
RADAR_BS=${RADAR_BS:-128}
set_backbone_args() { GTR_ARGS=("${ITR_BASE[@]}" --enc_in 21 --e_layers 3 --d_model 512 --d_ff 512); }
source "$(dirname "$0")/itrans_common.sh"
run_all
