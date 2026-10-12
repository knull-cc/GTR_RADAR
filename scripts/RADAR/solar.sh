#!/usr/bin/env bash
# RADAR pipeline for Solar. Backbone hyper-parameters copied verbatim from scripts/GTR/solar.sh.
# Usage: bash scripts/RADAR/solar.sh   (QUICK=1 for 1 seed / pred_len 96; see common.sh for knobs)
model_name=GTR

root_path_name=./dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar

seq_len=96
cycle=144
GTR_ARGS=(--features M --seq_len $seq_len --enc_in 137 --cycle $cycle --train_epochs 30 --patience 5 --use_revin 0 --itr 1 --batch_size 64 --learning_rate 0.003 )

source "$(dirname "$0")/common.sh"
run_all
