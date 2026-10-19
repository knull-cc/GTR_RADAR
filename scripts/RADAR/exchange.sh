#!/usr/bin/env bash
# RADAR pipeline for Exchange (daily exchange rates of 8 countries, 7,588 days; Autoformer benchmark split).
# The original GTR repository ships no Exchange script: the backbone uses the GTR defaults of the other datasets
# (dropout 0.5, lr 1e-3, 30 epochs, patience 5) with batch 32 and a weekly cycle (7 days) for GTR's cycle index.
# RADAR E3 is off (e3_period=0): exchange rates have no seasonal period; E2 / E4 / E1 remain.
# Data: dataset/exchange_rate/exchange_rate.csv (or dataset/exchange_rate.csv).
# Usage: bash scripts/RADAR/exchange.sh   (QUICK=1 for 1 seed / pred_len 96; see common.sh for knobs)
model_name=GTR

if [ -f ./dataset/exchange_rate/exchange_rate.csv ]; then root_path_name=./dataset/exchange_rate/; else root_path_name=./dataset/; fi
data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom

seq_len=96
cycle=7
e3_period=0   # no seasonal period -> E3 off
GTR_ARGS=(--features M --seq_len $seq_len --enc_in 8 --cycle $cycle --train_epochs 30 --patience 5 --dropout 0.5 --itr 1 --batch_size 32 --learning_rate 0.001 )

source "$(dirname "$0")/common.sh"
run_all
