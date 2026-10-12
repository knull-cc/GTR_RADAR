#!/usr/bin/env bash
# Clean-input headroom diagnostic (validation split only), seed 2024, pred_len 96.
#   bash scripts/RADAR/run_diag.sh ETTh1                 # needs/creates S0 (base.pth); uses plugin.pth if present
#   DIAG_PLUGIN=1 bash scripts/RADAR/run_diag.sh weather # also trains S2 + plug-in so the RADAR repair is audited
set -eo pipefail
ds=$(echo "${1:-ETTh1}" | tr '[:upper:]' '[:lower:]')
script="$(dirname "$0")/${ds}.sh"
if [ ! -f "$script" ]; then echo "unknown dataset '$1' (no $script)"; exit 1; fi
MODE=diag QUICK=1 REPRO=0 bash "$script"
