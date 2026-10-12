#!/usr/bin/env bash
# Shared RADAR pipeline, sourced by scripts/RADAR/<dataset>.sh.
# The dataset script defines: model_name root_path_name data_path_name model_id_name data_name seq_len cycle GTR_ARGS
#
# Environment knobs:
#   QUICK=1          pred_len 96, seed 2024, plus the original run.py reproduction check (REPRO=1)
#   PRED_LENS="..."  SEEDS="..."   override the grid (default: 96 192 336 720 x 2024 2025 2026)
#   REPRO=0/1        run run_orig.py (= original run.py) + repro_check before continuing
#   REPRO_STRICT=1   stop if |S0 - run.py| / run.py >= 2%  (default 1)
#   BACKBONE=GTR     GTR | GTRiTransformer | GTRPatchTST | GTRDLinear
#   EXTRA_ARGS="..." extra run.py-compatible args for every call (e.g. "--num_workers 4")
#   RADAR_ARGS="..." extra RADAR-only args (e.g. "--hind_level channel", "--conditions clean,T1_raw,T2_k4,T3_flat")
#   ABLATION=1       also train/evaluate Ours (joint training, backbone fine-tuned) as an ablation
#   SKIP_EXISTING=1  reuse checkpoints that already exist (default 1; set 0 to retrain)
#   PY=python  LOG_DIR=./logs_radar  RESULTS_DIR=./results_radar/  CKPT_DIR=./checkpoints_radar/
set -eo pipefail

QUICK=${QUICK:-0}
BACKBONE=${BACKBONE:-$model_name}
if [ "$QUICK" = "1" ]; then
  PRED_LENS=${PRED_LENS:-"96"}; SEEDS=${SEEDS:-"2024"}; REPRO=${REPRO:-1}
else
  PRED_LENS=${PRED_LENS:-"96 192 336 720"}; SEEDS=${SEEDS:-"2024 2025 2026"}; REPRO=${REPRO:-0}
fi
REPRO_STRICT=${REPRO_STRICT:-1}
ABLATION=${ABLATION:-0}
PY=${PY:-python}
LOG_DIR=${LOG_DIR:-./logs_radar}
RESULTS_DIR=${RESULTS_DIR:-./results_radar/}
CKPT_DIR=${CKPT_DIR:-./checkpoints_radar/}
SKIP_EXISTING=${SKIP_EXISTING:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
RADAR_ARGS=${RADAR_ARGS:-}
mkdir -p "$LOG_DIR" "$RESULTS_DIR/raw"

check_data() {
  if [ ! -f "${root_path_name}${data_path_name}" ]; then
    echo "ERROR: dataset file ${root_path_name}${data_path_name} not found." >&2
    echo "       Put the original csv/txt under ${root_path_name} (no synthetic data is generated)." >&2
    exit 1
  fi
}

run_one() {
  local pred_len=$1 seed=$2
  local mid=${model_id_name}_${seq_len}_${pred_len}
  local setting=${mid}_${BACKBONE}_${data_name}_ftM_sl${seq_len}_pl${pred_len}_cycle${cycle}_seed${seed}
  local data_args=(--root_path "$root_path_name" --data_path "$data_path_name" --model_id "$mid" --data "$data_name"
                   --pred_len "$pred_len" --random_seed "$seed")
  # shellcheck disable=SC2206
  local common=("${data_args[@]}" "${GTR_ARGS[@]}" --backbone "$BACKBONE" --radar_ckpt_dir "$CKPT_DIR"
                --results_dir "$RESULTS_DIR" --skip_existing "$SKIP_EXISTING" $EXTRA_ARGS $RADAR_ARGS)
  local log=$LOG_DIR/$setting
  echo "================ $setting ================"

  if [ "$REPRO" = "1" ]; then
    # shellcheck disable=SC2086
    RADAR_REPRO_OUT="$RESULTS_DIR/raw/${setting}__orig.json" \
      $PY -u run_orig.py --is_training 1 --model "$BACKBONE" "${data_args[@]}" "${GTR_ARGS[@]}" $EXTRA_ARGS \
      2>&1 | tee "${log}__orig.log"
  fi
  $PY -u run_radar.py train_base "${common[@]}" 2>&1 | tee "${log}__S0.log"
  if [ "$REPRO" = "1" ]; then
    if ! $PY -u run_radar.py repro_check "${common[@]}" 2>&1 | tee "${log}__repro.log"; then
      if [ "$REPRO_STRICT" = "1" ]; then echo "reproduction check failed -> stopping (REPRO_STRICT=0 to continue)"; exit 2; fi
    fi
  fi
  $PY -u run_radar.py train_aug      "${common[@]}" 2>&1 | tee "${log}__S1.log"
  $PY -u run_radar.py pretrain_radar "${common[@]}" 2>&1 | tee "${log}__S2.log"
  $PY -u run_radar.py train_radar    "${common[@]}" --freeze_backbone 2>&1 | tee "${log}__S3plugin.log"
  local methods="B0,B1,B2,Ours-plugin"
  if [ "$ABLATION" = "1" ]; then  # joint training (backbone also fine-tuned) = ablation only
    $PY -u run_radar.py train_radar  "${common[@]}" 2>&1 | tee "${log}__S3.log"
    methods="$methods,Ours"
  fi
  $PY -u run_radar.py eval           "${common[@]}" --methods "$methods" 2>&1 | tee "${log}__eval.log"
}

run_diag_one() {
  local pred_len=$1 seed=$2
  local mid=${model_id_name}_${seq_len}_${pred_len}
  local setting=${mid}_${BACKBONE}_${data_name}_ftM_sl${seq_len}_pl${pred_len}_cycle${cycle}_seed${seed}
  local data_args=(--root_path "$root_path_name" --data_path "$data_path_name" --model_id "$mid" --data "$data_name"
                   --pred_len "$pred_len" --random_seed "$seed")
  # shellcheck disable=SC2206
  local common=("${data_args[@]}" "${GTR_ARGS[@]}" --backbone "$BACKBONE" --radar_ckpt_dir "$CKPT_DIR"
                --results_dir "$RESULTS_DIR" --skip_existing 1 $EXTRA_ARGS $RADAR_ARGS)
  local log=$LOG_DIR/$setting
  echo "================ [diag] $setting ================"
  $PY -u run_radar.py train_base "${common[@]}" 2>&1 | tee "${log}__S0.log"
  if [ "${DIAG_PLUGIN:-0}" = "1" ]; then
    $PY -u run_radar.py pretrain_radar "${common[@]}" 2>&1 | tee "${log}__S2.log"
    $PY -u run_radar.py train_radar "${common[@]}" --freeze_backbone 2>&1 | tee "${log}__S3plugin.log"
  fi
  $PY -u diag_clean.py "${common[@]}" 2>&1 | tee "${log}__diag.log"
}

run_all() {
  check_data
  if [ "${MODE:-full}" = "diag" ]; then
    for pred_len in $PRED_LENS; do for seed in $SEEDS; do run_diag_one "$pred_len" "$seed"; done; done
    return
  fi
  for pred_len in $PRED_LENS; do
    for seed in $SEEDS; do
      run_one "$pred_len" "$seed"
    done
  done
  # shellcheck disable=SC2086
  $PY -u run_radar.py summarize --results_dir "$RESULTS_DIR" $RADAR_ARGS 2>&1 | tee "$LOG_DIR/summary_${model_id_name}.log"
  local p0 s0
  p0=$(echo $PRED_LENS | awk '{print $1}'); s0=$(echo $SEEDS | awk '{print $1}')
  # shellcheck disable=SC2086
  $PY -u run_radar.py plot --root_path "$root_path_name" --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${p0}" --data "$data_name" --pred_len "$p0" --random_seed "$s0" \
    "${GTR_ARGS[@]}" --backbone "$BACKBONE" --radar_ckpt_dir "$CKPT_DIR" --results_dir "$RESULTS_DIR" \
    $EXTRA_ARGS $RADAR_ARGS 2>&1 | tee "$LOG_DIR/plot_${model_id_name}.log"
}
