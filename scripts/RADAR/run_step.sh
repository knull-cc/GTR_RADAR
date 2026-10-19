#!/usr/bin/env bash
# RADAR v4 diagnostics (validation split, seed 2024, pred_len 96 unless PRED_LENS / SEEDS are set).
#   bash scripts/RADAR/run_step.sh revin   ETTh1 [more datasets]   # step 1: RevIN 2×2 on B0
#   bash scripts/RADAR/run_step.sh experts ETTh1                   # step 2: offline expert repair errors
#   bash scripts/RADAR/run_step.sh det     ETTh1                   # step 3: S2 + detector AUROC (v4 features)
#   DET_FEATS=v3 bash scripts/RADAR/run_step.sh det ETTh1          #         the same with the v3 features
# Results: results_radar_v5/diag/<setting>__<step>.md / .json
set -eo pipefail
step=${1:?usage: run_step.sh revin|experts|det DATASET...}
shift
dir="$(dirname "$0")"
list="${*:-etth1}"
for ds in $list; do
  script="$dir/$(echo "$ds" | tr '[:upper:]' '[:lower:]').sh"
  if [ ! -f "$script" ]; then echo "unknown dataset '$ds' (no $script)"; exit 1; fi
  MODE=$step QUICK=${QUICK:-1} REPRO=0 bash "$script"
done
