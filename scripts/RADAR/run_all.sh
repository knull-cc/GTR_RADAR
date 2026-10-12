#!/usr/bin/env bash
# Full grid (pred_len 96/192/336/720 x seeds 2024/2025/2026) in priority order.
#   bash scripts/RADAR/run_all.sh                 # all datasets
#   bash scripts/RADAR/run_all.sh etth1 ettm1     # a subset
set -eo pipefail
dir="$(dirname "$0")"
if [ $# -gt 0 ]; then list="$*"; else list="etth1 ettm1 weather etth2 ettm2 solar electricity traffic"; fi
for ds in $list; do
  bash "$dir/$(echo "$ds" | tr '[:upper:]' '[:lower:]').sh"
done
python -u run_radar.py summarize --results_dir "${RESULTS_DIR:-./results_radar/}"
