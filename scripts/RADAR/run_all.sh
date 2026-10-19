#!/usr/bin/env bash
# Full grid (pred_len 96/192/336/720 x seed 2024) of the RADAR v4 main experiment.
#   bash scripts/RADAR/run_all.sh                 # the 7 main datasets
#   bash scripts/RADAR/run_all.sh etth1 ettm1     # a subset
set -eo pipefail
dir="$(dirname "$0")"
if [ $# -gt 0 ]; then list="$*"; else list="etth1 etth2 ettm1 ettm2 weather electricity traffic"; fi
for ds in $list; do
  bash "$dir/$(echo "$ds" | tr '[:upper:]' '[:lower:]').sh"
done
python -u run_radar.py summarize --results_dir "${RESULTS_DIR:-./results_radar_v4/}"
