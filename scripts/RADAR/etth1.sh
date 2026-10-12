#!/usr/bin/env bash
# RADAR pipeline for ETTh1. Backbone hyper-parameters copied verbatim from scripts/GTR/etth1.sh.
# Usage: bash scripts/RADAR/etth1.sh   (QUICK=1 for 1 seed / pred_len 96; see common.sh for knobs)
model_name=GTR

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

seq_len=96
cycle=24
GTR_ARGS=(--features M --seq_len $seq_len --enc_in 7 --cycle $cycle --train_epochs 30 --patience 5 --dropout 0.5 --itr 1 --batch_size 256 --learning_rate 0.001 )

source "$(dirname "$0")/common.sh"
run_all
