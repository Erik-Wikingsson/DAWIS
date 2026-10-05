#!/bin/bash
#SBATCH -J sevir_gibbs
#SBATCH -t 1-00:00:00
#SBATCH --gpus=1 -C "fat"
#SBATCH --output ./slurm_logs/%A_%x.out
#
# Block-Gibbs smoother on SEVIR: repeatedly re-samples the trajectory in
# overlapping blocks (GIBBS_SWEEPS passes) using the FMW window model.
#
# Sources _common.sh (conda + shared settings) via $REPO_ROOT; do not `set -u` before it.
if [[ -n "${REPO_ROOT:-}" ]]; then
    _sevir_common="$REPO_ROOT/assimilation/scripts/SEVIR/_common.sh"
else
    _sevir_common="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
fi
if [[ ! -f "$_sevir_common" ]]; then
    echo "ERROR: no assimilation/scripts/SEVIR/_common.sh at: $_sevir_common" >&2
    echo "  \$REPO_ROOT is unset. Submit with scripts/submit.sh, which sets it." >&2
    exit 1
fi
source "$_sevir_common"
unset _sevir_common

INIT_STATES="${INIT_STATES:-4}"
if (( INIT_STATES % 2 != 0 )); then
    echo "ERROR: GIBBS needs an even --init_states (sda_w = init_states/2), got ${INIT_STATES}." >&2
    exit 1
fi
FMW_CKPT="${FMW_CKPT:-$(fmw_ckpt_for "$INIT_STATES")}"
require_ckpt "$FMW_CKPT" "FMW_CKPT (the Gibbs sampler's prior)" \
  "forecasting/training_scripts/SEVIR/train_all_fmw.sh"
check_window_fits "$INIT_STATES"

# Warm start (empty = start from scratch):
#   TRAJECTORY_PATH     explicit .nc/.npy/.pt file (wins if both are set)
#   INITIAL_TRAJECTORY  method name, looked up in $PAPER_RUNS_ROOT as
#                       <method>_<experiment>_run_<data_index>_*.nc
#   TRAJECTORY_VAR      variable to read (x_smooth or x_assim)
traj_args=()
if [[ -n "${TRAJECTORY_PATH:-}" ]]; then
    if [[ ! -f "$TRAJECTORY_PATH" ]]; then
        echo "ERROR: TRAJECTORY_PATH not found at: ${TRAJECTORY_PATH}" >&2
        exit 1
    fi
    traj_args=(--trajectory_path "${TRAJECTORY_PATH}")
elif [[ -n "${INITIAL_TRAJECTORY:-}" ]]; then
    traj_args=(--initial_trajectory "${INITIAL_TRAJECTORY}")
fi
if [[ ${#traj_args[@]} -gt 0 ]]; then
    traj_args+=(--trajectory_var "${TRAJECTORY_VAR:-x_smooth}")
fi

# Random block geometry (BLOCK_MODE=random only); passed only when set.
block_args=()
if [[ -n "${RANDOM_BLOCK_COUNT:-}" ]]; then
    block_args+=(--random_block_count "${RANDOM_BLOCK_COUNT}")
fi
if [[ "${RANDOM_BLOCK_REPLACE:-false}" == "true" ]]; then
    block_args+=(--random_block_replace)
fi

# Per-sweep logging:
#   none     final trajectory only
#   metrics  one wandb run per sweep
#   full     also saves each sweep's trajectory as x_gibbs_<k>
case "${SWEEP_LOGGING:-none}" in
    none)    sweep_args=() ;;
    metrics) sweep_args=(--sweep_metrics) ;;
    full)    sweep_args=(--save_sweeps) ;;
    *)
        echo "ERROR: SWEEP_LOGGING must be none, metrics or full (got" \
             "'${SWEEP_LOGGING}')." >&2
        exit 1 ;;
esac

# No propagator; a stray FORWARD_MODEL is rejected.
fwd_args=($(forward_model_args "none"))

exp_name="${EXP_NAME:-GIBBS_SEVIR_${EXPERIMENT}_init${INIT_STATES}_run_${DATA_INDEX}}"

print_sevir_config
echo "  method       GIBBS sweeps=${GIBBS_SWEEPS:-10} block_mode=${BLOCK_MODE:-sliding}"
echo "  checkpoint   ${FMW_CKPT}"
echo "  warm start   ${TRAJECTORY_PATH:-${INITIAL_TRAJECTORY:-none}}" \
     "var=${TRAJECTORY_VAR:-x_smooth} sweep_logging=${SWEEP_LOGGING:-none}"
date

# --eps is the stage-2 sampler eps (SAMPLER_EPS kept as a fallback).
python3 -m assimilation.smoothing \
    --exp_name "${exp_name}" \
    --method GIBBS \
    "${fwd_args[@]}" \
    --model_path "${FMW_CKPT}" \
    --init_states "${INIT_STATES}" \
    --window_batch_size "${WINDOW_BATCH_SIZE:-50}" \
    $(fmw_arch_args "${INIT_STATES}") \
    --schedule linear_scalar \
    --sampler stochastic \
    --eps "${EPS:-${SAMPLER_EPS:-0.03}}" \
    --noise_embedding positional \
    --guide_method "${GUIDE_METHOD:-MMPS}" \
    --guidance_strength "${GUIDANCE_STRENGTH:-1.0}" \
    --noise "${NOISE:-SDEdit}" \
    --corrections "${CORRECTIONS:-0}" \
    --tau "${TAU:-0.5}" \
    --euler_steps "${EULER_STEPS:-100}" \
    --invert_eps "${INVERT_EPS:-0.03}" \
    --gibbs_sweeps "${GIBBS_SWEEPS:-10}" \
    --block_mode "${BLOCK_MODE:-sliding}" \
    --block_stride "${BLOCK_STRIDE:-3}" \
    --block_direction "${BLOCK_DIRECTION:-alternate}" \
    --gibbs_endpoint_tmin "${GIBBS_ENDPOINT_TMIN:-0.7}" \
    --gibbs_midpoint_tmin "${GIBBS_MIDPOINT_TMIN:-0.7}" \
    --gibbs_num_endpoints "${GIBBS_NUM_ENDPOINTS:-1}" \
    "${block_args[@]}" \
    "${sweep_args[@]}" \
    "${traj_args[@]}" \
    $(sevir_common_args)

date
fix_results_permissions
