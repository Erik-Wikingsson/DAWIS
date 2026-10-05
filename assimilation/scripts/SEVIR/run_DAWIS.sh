#!/bin/bash
#SBATCH -J sevir_dawis
#SBATCH -t 10:00:00
#SBATCH --gpus=1 -C "fat"
#SBATCH --output ./slurm_logs/%A_%x.out
#
# DAWIS-family worker on SEVIR (all `--method DAWIS` over an FMW window model):
#   DAWIS          FORWARD_MODEL=dawis, NOISE=invert
#   GuidedForecast FORWARD_MODEL=none, NOISE=None, per-slot TMIN pattern
#   ForcingDAS     FORWARD_MODEL=none, NOISE=None, PYRAMID=true, N_FIXED
#   SDA_filter     FORWARD_MODEL=none, NOISE=SDEdit, CORRECTIONS/TAU
# Knobs are environment variables; see run_all_DAWIS.sh for a sweep.
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

INIT_STATES="${INIT_STATES:-2}"
FMW_CKPT="${FMW_CKPT:-$(fmw_ckpt_for "$INIT_STATES")}"
require_ckpt "$FMW_CKPT" "FMW_CKPT (the DAWIS window model)" \
  "forecasting/training_scripts/SEVIR/train_all_fmw.sh"
check_window_fits "$INIT_STATES"

# Prior for the analysis:
#   dawis    none; the window model generates t+1 itself (the method)
#   none     persistence (ablation)
#   unet     deterministic U-Net
#   fmw      FMW propagator (FWD_INIT_STATES picks which)
#   flowdas  FlowDAS interpolant (needs INIT_STATES 6)
#   self     this run's window model (GUIDE_FORECAST guides it)
FORWARD_MODEL="${FORWARD_MODEL:-dawis}"
fwd_args=($(forward_model_args "dawis none unet fmw flowdas self"))

# Inversion depth, length init_states + 1:
#   TMIN                   explicit per-slot list (comma- or space-separated)
#   TMIN_START/TMIN_END    linear ramp; ignored when TMIN is set
TMIN="${TMIN:-}"
TMIN_START="${TMIN_START:-}"
TMIN_END="${TMIN_END:-}"
if [[ -z "$TMIN" && -z "$TMIN_START" ]]; then
    # Default: deepest inversion on the oldest slot.
    TMIN_START=0.4
    TMIN_END=0.1
fi
tmin_args=()
if [[ -n "$TMIN" ]]; then
    # shellcheck disable=SC2206  # deliberate word splitting into an array
    tmin_args=(--tmin ${TMIN//,/ })
else
    tmin_args=(--tmin_start "${TMIN_START}" --tmin_end "${TMIN_END:-$TMIN_START}")
fi

# Per-slot floor on the conditioning states (length init_states); empty = off.
TMIN_INIT="${TMIN_INIT:-}"
tmin_init_args=()
[[ -n "$TMIN_INIT" ]] && tmin_init_args=(--tmin_init ${TMIN_INIT//,/ })

# store_true flags: "off" is the absence of the flag.
guide_all_steps_args=()
[[ "${GUIDE_ALL_STEPS:-false}" == "true" ]] && guide_all_steps_args=(--guide_all_steps)

save_window_args=()
[[ "${SAVE_WINDOW_STATES:-false}" == "true" ]] && save_window_args=(--save_window_states)

# ForcingDAS fixed-lag pyramid; needs N_TIMES > INIT_STATES.
pyramid_args=()
if [[ "${PYRAMID:-false}" == "true" ]]; then
    if (( N_TIMES <= INIT_STATES )); then
        echo "ERROR: --pyramid needs N_TIMES > INIT_STATES; got ${N_TIMES} and ${INIT_STATES}." >&2
        echo "  An event is 25 frames, so START_TIME ${START_TIME} leaves room for" \
             "N_TIMES up to $((25 - START_TIME))." >&2
        exit 1
    fi
    pyramid_args=(--pyramid --n_fixed "${N_FIXED:-0}")
fi

# Forward model is part of the name so the comparison runs don't collide.
exp_name="${EXP_NAME:-DAWIS-${FORWARD_MODEL}_SEVIR_${EXPERIMENT}_init${INIT_STATES}_run_${DATA_INDEX}}"

print_sevir_config
echo "  method       DAWIS forward_model=${FORWARD_MODEL} noise=${NOISE:-invert}"
echo "  window       init_states=${INIT_STATES} tmin=[${TMIN:-${TMIN_START}..${TMIN_END}}]"
echo "  checkpoint   ${FMW_CKPT}"
date

python3 -m assimilation.assimilate \
    --exp_name "${exp_name}" \
    --method DAWIS \
    "${fwd_args[@]}" \
    --model_path "${FMW_CKPT}" \
    --init_states "${INIT_STATES}" \
    --window "${INIT_STATES}" \
    "${tmin_args[@]}" \
    "${tmin_init_args[@]}" \
    $(fmw_arch_args "${INIT_STATES}") \
    --schedule linear_scalar \
    --sampler stochastic \
    --sampler_eps 0.03 \
    --noise_embedding positional \
    --guide_method "${GUIDE_METHOD:-MMPS}" \
    --guidance_strength "${GUIDANCE_STRENGTH:-1.0}" \
    --noise "${NOISE:-invert}" \
    --euler_steps "${EULER_STEPS:-100}" \
    --invert_steps "${INVERT_STEPS:-100}" \
    --invert_eps "${INVERT_EPS:-0.03}" \
    --eps "${EPS:-0.03}" \
    --corrections "${CORRECTIONS:-0}" \
    --tau "${TAU:-0.0001}" \
    --guide_first "${GUIDE_FIRST:-none}" \
    --x0_sigma "${X0_SIGMA:-1}" \
    "${guide_all_steps_args[@]}" \
    "${pyramid_args[@]}" \
    "${save_window_args[@]}" \
    $(sevir_online_args) \
    $(sevir_common_args)

date
fix_results_permissions
