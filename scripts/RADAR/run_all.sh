#!/usr/bin/env bash
# Full grid (pred_len 96/192/336/720 x seed 2024) of the RADAR main experiment (v6).
#   bash scripts/RADAR/run_all.sh                 # 9 datasets: TameR's 8 (ICML 2026) + electricity (SEER)
#   bash scripts/RADAR/run_all.sh etth1 ettm1     # a subset
set -eo pipefail
dir="$(dirname "$0")"
if [ $# -gt 0 ]; then list="$*"; else list="etth1 etth2 ettm1 ettm2 weather exchange solar electricity traffic"; fi
for ds in $list; do
  bash "$dir/$(echo "$ds" | tr '[:upper:]' '[:lower:]').sh"
done
python -u run_radar.py summarize --results_dir "${RESULTS_DIR:-./results_radar_v6/}"
