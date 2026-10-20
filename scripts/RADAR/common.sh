#!/usr/bin/env bash
# Shared RADAR pipeline (default architecture v6), sourced by scripts/RADAR/<dataset>.sh.
# The dataset script defines: model_name root_path_name data_path_name model_id_name data_name seq_len cycle
#                             e3_period GTR_ARGS
#
# Environment knobs:
#   MODE=full        S0 -> [S1 GTR+Aug] -> [B3 denoise front] -> S2 -> S3 plug-in -> eval (default)
#   MODE=revin       step 1: S0 (reused if base.pth exists) + RevIN 2×2 diagnostic on B0   (diag_v4.py revin)
#   MODE=experts     step 2: offline expert repair errors (+ E1 if pre_<arch>.pth exists)   (diag_v4.py experts)
#   MODE=det         step 3: S0 + S2 + detector AUROC per condition                         (diag_v4.py det)
#   MODE=timing      inference time of the trained plug-in per module vs B0                (diag_v4.py timing)
#                    DET_FEATS=v3 evaluates the v3-feature detector (separate checkpoint pre_<arch>_detv3.pth)
#   BASELINES="B0"   baselines to train / evaluate besides Ours-plugin (default B0 only). Add B1 (GTR+Aug, S1),
#                    B2 (Hampel) and/or B3 (denoise front), e.g. BASELINES="B0,B1,B2,B3"; S1 / B3 are trained only
#                    when listed
#   MODE=eval        evaluation only (checkpoints must exist); METHODS="$BASELINES,Ours-plugin" by default.
#                    e.g. ARCH=v5 METHODS=Ours-plugin MODE=eval adds the v5 plug-in as Ours-plugin-v5
#   MODE=diag        v3 clean-headroom audit (diag_clean.py)
#   QUICK=1          pred_len 96, seed 2024, plus the original run.py reproduction check (REPRO=1)
#   PRED_LENS="..."  SEEDS="..."   override the grid (default: 96 192 336 720 x 2024)
#   REPRO=0/1        run run_orig.py (= original run.py) + repro_check before continuing
#   REPRO_STRICT=1   stop if |S0 - run.py| / run.py >= 2%  (default 1)
#   BACKBONE=GTR     GTR | GTRiTransformer | GTRPatchTST | GTRDLinear
#   EXTRA_ARGS="..." extra run.py-compatible args for every call (e.g. "--num_workers 4")
#   RADAR_ARGS="..." extra RADAR-only args (e.g. "--test_grid main", "--router h", "--experts e1,e2",
#                    "--pre_patience 5" = early stopping of S2 after 5 epochs without val improvement; default off)
#   ABLATION=1       also train/evaluate Ours (joint training, backbone fine-tuned) as an ablation
#   ARGMAX=1         also evaluate the plug-in with hard (argmax) routing (Ours-argmax)
#   ORACLE=1         also evaluate the plug-in with a perfect gate (Ours-oracle, upper bound of the gate)
#   CONDITIONS="..." test conditions (default clean,tamer_re_sap,tamer_re_smp = TameR recent single anomalous point
#                    (+N(0,(3σ)²) on the last point) and recent single missing point (last point = raw 0, = T1_raw);
#                    "tamer" = clean + all 7 TameR scenarios; combine e.g. "tamer,T2_k4"; "grid" = the 53-condition grid)
#   SKIP_EXISTING=1  reuse checkpoints that already exist (default 1; set 0 to retrain)
#   PY=python  LOG_DIR=./logs_radar_v6  RESULTS_DIR=./results_radar_v6/  CKPT_DIR=./checkpoints_radar/
#   ARCH=v5|v4       run an earlier architecture instead of v6 (its RADAR results are recorded as Ours-plugin-v5 ...)
set -eo pipefail

QUICK=${QUICK:-0}
MODE=${MODE:-full}
BACKBONE=${BACKBONE:-$model_name}
if [ "$QUICK" = "1" ]; then
  PRED_LENS=${PRED_LENS:-"96"}; SEEDS=${SEEDS:-"2024"}; REPRO=${REPRO:-1}
else
  PRED_LENS=${PRED_LENS:-"96 192 336 720"}; SEEDS=${SEEDS:-"2024"}; REPRO=${REPRO:-0}
fi
REPRO_STRICT=${REPRO_STRICT:-1}
ABLATION=${ABLATION:-0}
ARGMAX=${ARGMAX:-0}
ORACLE=${ORACLE:-0}
BASELINES=${BASELINES:-B0}
PY=${PY:-python}
LOG_DIR=${LOG_DIR:-./logs_radar_v6}
RESULTS_DIR=${RESULTS_DIR:-./results_radar_v6/}
CKPT_DIR=${CKPT_DIR:-./checkpoints_radar/}
SKIP_EXISTING=${SKIP_EXISTING:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
RADAR_ARGS=${RADAR_ARGS:-}
CONDITIONS=${CONDITIONS:-clean,tamer_re_sap,tamer_re_smp}
if [ -n "${DET_FEATS:-}" ]; then RADAR_ARGS="$RADAR_ARGS --det_feats $DET_FEATS"; fi
if [ -n "${ARCH:-}" ]; then RADAR_ARGS="$RADAR_ARGS --arch $ARCH"; fi
E3_ARGS=()
if [ -n "${e3_period:-}" ]; then E3_ARGS=(--e3_period "$e3_period"); fi
mkdir -p "$LOG_DIR" "$RESULTS_DIR/raw"

check_data() {
  if [ ! -f "${root_path_name}${data_path_name}" ]; then
    echo "ERROR: dataset file ${root_path_name}${data_path_name} not found." >&2
    echo "       Put the original csv/txt under ${root_path_name} (no synthetic data is generated)." >&2
    exit 1
  fi
}

# sets: mid setting data_args common log
# A dataset script may define set_backbone_args <pred_len> to set GTR_ARGS per horizon (official per-H
# hyper-parameters, e.g. scripts/RADAR_iTransformer/).
_setup() {
  local pred_len=$1 seed=$2 skip=$3
  if declare -F set_backbone_args >/dev/null; then set_backbone_args "$pred_len"; fi
  mid=${model_id_name}_${seq_len}_${pred_len}
  setting=${mid}_${BACKBONE}_${data_name}_ftM_sl${seq_len}_pl${pred_len}_cycle${cycle}_seed${seed}
  data_args=(--root_path "$root_path_name" --data_path "$data_path_name" --model_id "$mid" --data "$data_name"
             --pred_len "$pred_len" --random_seed "$seed")
  # shellcheck disable=SC2206
  common=("${data_args[@]}" "${GTR_ARGS[@]}" "${E3_ARGS[@]}" --backbone "$BACKBONE" --radar_ckpt_dir "$CKPT_DIR"
          --results_dir "$RESULTS_DIR" --skip_existing "$skip" --conditions "$CONDITIONS" $EXTRA_ARGS $RADAR_ARGS)
  log=$LOG_DIR/$setting
}

run_one() {
  _setup "$1" "$2" "$SKIP_EXISTING"
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
  if [[ ",$BASELINES," == *",B1,"* ]]; then
    $PY -u run_radar.py train_aug    "${common[@]}" 2>&1 | tee "${log}__S1.log"
  fi
  if [[ ",$BASELINES," == *",B3,"* ]]; then
    $PY -u run_radar.py train_denoise "${common[@]}" 2>&1 | tee "${log}__B3.log"
  fi
  $PY -u run_radar.py pretrain_radar "${common[@]}" 2>&1 | tee "${log}__S2.log"
  $PY -u run_radar.py train_radar    "${common[@]}" --freeze_backbone 2>&1 | tee "${log}__S3plugin.log"
  local methods="$BASELINES,Ours-plugin"
  if [ "$ARGMAX" = "1" ]; then methods="$methods,Ours-argmax"; fi
  if [ "$ORACLE" = "1" ]; then methods="$methods,Ours-oracle"; fi
  if [ "$ABLATION" = "1" ]; then  # joint training (backbone also fine-tuned) = ablation only
    $PY -u run_radar.py train_radar  "${common[@]}" 2>&1 | tee "${log}__S3.log"
    methods="$methods,Ours"
  fi
  $PY -u run_radar.py eval           "${common[@]}" --methods "$methods" 2>&1 | tee "${log}__eval.log"
}

run_step_one() {  # MODE = revin | experts | det | timing
  _setup "$1" "$2" 1
  echo "================ [$MODE] $setting ================"
  $PY -u run_radar.py train_base "${common[@]}" 2>&1 | tee "${log}__S0.log"
  if [ "$MODE" = "det" ]; then
    $PY -u run_radar.py pretrain_radar "${common[@]}" 2>&1 | tee "${log}__S2${DET_FEATS:+_det$DET_FEATS}.log"
  fi
  $PY -u diag_v4.py "$MODE" "${common[@]}" 2>&1 | tee "${log}__diag_${MODE}${DET_FEATS:+_det$DET_FEATS}.log"
}

run_eval_one() {
  _setup "$1" "$2" 1
  echo "================ [eval] $setting ================"
  $PY -u run_radar.py eval "${common[@]}" --methods "${METHODS:-$BASELINES,Ours-plugin}" 2>&1 | tee "${log}__eval${ARCH:+_$ARCH}.log"
}

run_diag_one() {
  _setup "$1" "$2" 1
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
  case "$MODE" in
    revin|experts|det|timing)
      for pred_len in $PRED_LENS; do for seed in $SEEDS; do run_step_one "$pred_len" "$seed"; done; done
      return ;;
    diag)
      for pred_len in $PRED_LENS; do for seed in $SEEDS; do run_diag_one "$pred_len" "$seed"; done; done
      return ;;
    eval)
      for pred_len in $PRED_LENS; do for seed in $SEEDS; do run_eval_one "$pred_len" "$seed"; done; done
      $PY -u run_radar.py summarize --results_dir "$RESULTS_DIR" 2>&1 | tee "$LOG_DIR/summary_${model_id_name}.log"
      return ;;
    full) ;;
    *) echo "unknown MODE=$MODE (full | revin | experts | det | timing | eval | diag)"; exit 1 ;;
  esac
  for pred_len in $PRED_LENS; do
    for seed in $SEEDS; do
      run_one "$pred_len" "$seed"
    done
  done
  # shellcheck disable=SC2086
  $PY -u run_radar.py summarize --results_dir "$RESULTS_DIR" 2>&1 | tee "$LOG_DIR/summary_${model_id_name}.log"
  local p0 s0
  p0=$(echo $PRED_LENS | awk '{print $1}'); s0=$(echo $SEEDS | awk '{print $1}')
  if declare -F set_backbone_args >/dev/null; then set_backbone_args "$p0"; fi
  # shellcheck disable=SC2086
  $PY -u run_radar.py plot --root_path "$root_path_name" --data_path "$data_path_name" \
    --model_id "${model_id_name}_${seq_len}_${p0}" --data "$data_name" --pred_len "$p0" --random_seed "$s0" \
    "${GTR_ARGS[@]}" "${E3_ARGS[@]}" --backbone "$BACKBONE" --radar_ckpt_dir "$CKPT_DIR" \
    --results_dir "$RESULTS_DIR" $EXTRA_ARGS $RADAR_ARGS 2>&1 | tee "$LOG_DIR/plot_${model_id_name}.log"
}
