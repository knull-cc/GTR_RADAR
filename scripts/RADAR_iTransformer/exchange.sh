#!/usr/bin/env bash
# RADAR + plain iTransformer on Exchange. Backbone hyper-parameters: official thuml/iTransformer
# scripts/multivariate_forecasting/Exchange/iTransformer.sh (+ run.py defaults, see itrans_common.sh).
# Usage: bash scripts/RADAR_iTransformer/exchange.sh   (PRED_LENS="96" for one horizon; knobs: scripts/RADAR/common.sh)
model_name=iTransformer
if [ -f ./dataset/exchange_rate/exchange_rate.csv ]; then root_path_name=./dataset/exchange_rate/; else root_path_name=./dataset/; fi
data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom
seq_len=96
cycle=7
e3_period=0   # no seasonal period -> E3 off
RADAR_BS=${RADAR_BS:-128}
set_backbone_args() {  # the official script trains H = 336 for a single epoch (--train_epochs 1); kept as is
  GTR_ARGS=("${ITR_BASE[@]}" --enc_in 8 --e_layers 2 --d_model 128 --d_ff 128)
  if [ "$1" = 336 ]; then GTR_ARGS+=(--train_epochs 1); fi
}
source "$(dirname "$0")/itrans_common.sh"
run_all
