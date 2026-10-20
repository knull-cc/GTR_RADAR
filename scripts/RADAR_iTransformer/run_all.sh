#!/usr/bin/env bash
# RADAR + plain iTransformer, H 96/192/336/720, seed 2024, the 8 datasets (no Traffic).
#   bash scripts/RADAR_iTransformer/run_all.sh                 # sequential
#   bash scripts/RADAR_iTransformer/run_all.sh etth1 ettm1     # a subset
set -eo pipefail
dir="$(dirname "$0")"
if [ $# -gt 0 ]; then list="$*"; else list="etth1 etth2 ettm1 ettm2 weather exchange solar electricity"; fi
for ds in $list; do
  bash "$dir/$(echo "$ds" | tr '[:upper:]' '[:lower:]').sh"
done
python -u run_radar.py summarize --results_dir "${RESULTS_DIR:-./results_itrans/}"
