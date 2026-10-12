#!/usr/bin/env bash
# RADAR pipeline for weather. Backbone hyper-parameters copied verbatim from scripts/GTR/weather.sh.
# Usage: bash scripts/RADAR/weather.sh   (QUICK=1 for 1 seed / pred_len 96; see common.sh for knobs)
model_name=GTR

root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

seq_len=96
cycle=144
GTR_ARGS=(--features M --seq_len $seq_len --enc_in 21 --cycle $cycle --train_epochs 30 --patience 5 --dropout 0.5 --itr 1 --batch_size 64 --learning_rate 0.001 )

source "$(dirname "$0")/common.sh"
run_all
