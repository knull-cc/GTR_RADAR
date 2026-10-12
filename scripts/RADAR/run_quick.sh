#!/usr/bin/env bash
# Quick pipeline for one dataset: 1 seed (2024), pred_len 96, including the run.py reproduction check.
#   bash scripts/RADAR/run_quick.sh ETTh1      (ETTh1 | ETTh2 | ETTm1 | ETTm2 | weather | electricity | traffic | solar)
# S0 reproduction check -> S1 (GTR+Aug) -> S2 -> S3 plug-in [-> S3 joint if ABLATION=1] -> eval -> summarize -> plot
set -eo pipefail
ds=$(echo "${1:-ETTh1}" | tr '[:upper:]' '[:lower:]')
script="$(dirname "$0")/${ds}.sh"
if [ ! -f "$script" ]; then echo "unknown dataset '$1' (no $script)"; exit 1; fi
QUICK=1 bash "$script"
