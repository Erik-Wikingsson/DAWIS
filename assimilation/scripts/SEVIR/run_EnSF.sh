#!/bin/bash
#SBATCH -J sevir_ensf
#SBATCH -t 10:00:00
#SBATCH --gpus=1 -C "thin"
#SBATCH --output ./slurm_logs/%A_%x.out
#
# EnSF on SEVIR (training-free score; needs a learned propagator).
#
# GUIDANCE_STRENGTH damps the likelihood score; the explicit Euler reverse SDE
# diverges at 1.0, hence the default 0.01.
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

# Propagator:
#   flowdas  FlowDAS interpolant (default); stochastic, 6 conditioning states
#   fmw      FMW window model; FWD_INIT_STATES picks which (2, 4 or 6)
#   unet     deterministic U-Net; refused with the GT start (no spread)
FORWARD_MODEL="${FORWARD_MODEL:-flowdas}"
fwd_args=($(forward_model_args "flowdas unet fmw"))

# Spread the ensemble is rescaled to after every analysis. Unset = spread of the
# first forecast. ENSF_SPREAD is in run units, ENSF_SPREAD_COUNTS in VIL counts.
if [[ -z "${ENSF_SPREAD:-}" && -n "${ENSF_SPREAD_COUNTS:-}" ]]; then
    ENSF_SPREAD="$(_from_counts "$ENSF_SPREAD_COUNTS")"
fi
ensf_spread_args=()
[[ -n "${ENSF_SPREAD:-}" ]] && ensf_spread_args=(--ensf_spread "$ENSF_SPREAD")

exp_name="${EXP_NAME:-EnSF_SEVIR_${EXPERIMENT}_run_${DATA_INDEX}}"

print_sevir_config
echo "  method       EnSF euler_steps=${EULER_STEPS:-1000} eps_alpha=${EPS_ALPHA:-0.05} ensf_spread=${ENSF_SPREAD:-first forecast}"
echo "  propagator   ${FORWARD_MODEL}"
date

python3 -m assimilation.assimilate \
    --exp_name "${exp_name}" \
    --method EnSF \
    "${fwd_args[@]}" \
    --init_states "${INIT_STATES:-1}" \
    --euler_steps "${EULER_STEPS:-1000}" \
    --eps_alpha "${EPS_ALPHA:-0.05}" \
    --guidance_strength "${GUIDANCE_STRENGTH:-0.01}" \
    "${ensf_spread_args[@]}" \
    $(sevir_online_args) \
    $(sevir_common_args)

date
fix_results_permissions
