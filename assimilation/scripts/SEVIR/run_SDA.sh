#!/bin/bash
#SBATCH -J sevir_sda
#SBATCH -t 1-00:00:00
#SBATCH --gpus=1 -C "fat"
#SBATCH --output ./slurm_logs/%A_%x.out
#
# SDA smoother on SEVIR (assimilation/smoothing.py) over the FMW window model;
# no forward model. Half-window sda_w = INIT_STATES / 2, so INIT_STATES is even.
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
    echo "ERROR: SDA needs an even --init_states (sda_w = init_states/2), got ${INIT_STATES}." >&2
    exit 1
fi
FMW_CKPT="${FMW_CKPT:-$(fmw_ckpt_for "$INIT_STATES")}"
require_ckpt "$FMW_CKPT" "FMW_CKPT (SDA's generative prior)" \
  "forecasting/training_scripts/SEVIR/train_all_fmw.sh"
check_window_fits "$INIT_STATES"

# No propagator; a stray FORWARD_MODEL is rejected.
fwd_args=($(forward_model_args "none"))

exp_name="${EXP_NAME:-SDA_SEVIR_${EXPERIMENT}_init${INIT_STATES}_run_${DATA_INDEX}}"

# Windows per UNet call; the UNet sees WINDOW_BATCH_SIZE * BATCH_SIZE windows.
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-1}"

print_sevir_config
echo "  method       SDA (smoother) init_states=${INIT_STATES} sda_w=$((INIT_STATES / 2))"
echo "  batching     window_batch_size=${WINDOW_BATCH_SIZE} x batch_size=${BATCH_SIZE}" \
     "-> $((WINDOW_BATCH_SIZE * BATCH_SIZE)) windows/forward" \
     "(batch_size is ensemble members per call, as in run_DAWIS.sh)"
echo "  checkpoint   ${FMW_CKPT}"
date

python3 -m assimilation.smoothing \
    --exp_name "${exp_name}" \
    --method SDA \
    "${fwd_args[@]}" \
    --model_path "${FMW_CKPT}" \
    --init_states "${INIT_STATES}" \
    --window_batch_size "${WINDOW_BATCH_SIZE}" \
    $(fmw_arch_args "${INIT_STATES}") \
    --schedule linear_scalar \
    --sampler stochastic \
    --sampler_eps "${SAMPLER_EPS:-1}" \
    --noise_embedding positional \
    --guide_method "${GUIDE_METHOD:-MMPS}" \
    --guidance_strength "${GUIDANCE_STRENGTH:-1.0}" \
    --noise "${NOISE:-None}" \
    --corrections "${CORRECTIONS:-1}" \
    --tau "${TAU:-0.5}" \
    --euler_steps "${EULER_STEPS:-100}" \
    --guide_first "${GUIDE_FIRST:-init}" \
    --x0_sigma "${X0_SIGMA:-3}" \
    $(sevir_common_args)

date
fix_results_permissions
