#!/usr/bin/env bash
# RADAR pipeline for traffic. Backbone hyper-parameters copied verbatim from scripts/GTR/traffic.sh.
# Usage: bash scripts/RADAR/traffic.sh   (QUICK=1 for 1 seed / pred_len 96; see common.sh for knobs)
model_name=GTR

root_path_name=./dataset/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

seq_len=96
cycle=168
e3_period=24   # RADAR E3: daily period of the data (steps); E3 is off when e3_period >= seq_len
GTR_ARGS=(--features M --seq_len $seq_len --enc_in 862 --cycle $cycle --individual 1 --train_epochs 30 --patience 5 --itr 1 --batch_size 16 --learning_rate 0.003 )

source "$(dirname "$0")/common.sh"
run_all
