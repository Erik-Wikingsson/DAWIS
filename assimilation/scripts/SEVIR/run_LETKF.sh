#!/bin/bash
#SBATCH -J sevir_letkf
#SBATCH -t 10:00:00
#SBATCH --gpus=1 -C "thin"
#SBATCH --output ./slurm_logs/%A_%x.out
#
# LETKF on SEVIR. Localization uses the non-periodic grid geometry
# (3 km pixels, 384 km domain); HCOVLOCAL_SCALE is in metres.
# LETKF assumes Gaussian errors, which bounded, skewed VIL is not: a baseline.
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

exp_name="${EXP_NAME:-LETKF_SEVIR_${EXPERIMENT}_run_${DATA_INDEX}}"

print_sevir_config
echo "  method       LETKF"
echo "  propagator   ${FORWARD_MODEL}"
date

python3 -m assimilation.assimilate \
    --exp_name "${exp_name}" \
    --method LETKF \
    "${fwd_args[@]}" \
    --init_states "${INIT_STATES:-1}" \
    --hcovlocal_scale "${HCOVLOCAL_SCALE}" \
    --covinflate1 "${COVINFLATE1}" \
    --covinflate2 "${COVINFLATE2}" \
    $(sevir_online_args) \
    $(sevir_common_args)

date
fix_results_permissions
