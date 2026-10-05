#!/bin/bash
#SBATCH -J sevir_daisi
#SBATCH -t 10:00:00
#SBATCH --gpus=1 -C "thin"
#SBATCH --output ./slurm_logs/%A_%x.out
#
# DAISI on SEVIR, using an unconditional prior (PRIOR_CKPT, 128x128, VARIANT=lr_vil).
#   FORWARD_MODEL=flowdas   full method with the FlowDAS propagator (default)
#   FORWARD_MODEL=unet|fmw  full method with a trained UNET_CKPT / FMW_CKPT
#   FORWARD_MODEL=none      prior + observations only (samples from pure noise)
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
# The DAISI prior was trained on vil/255; must be set before sourcing _common.sh.
ASSIM_NORM="${ASSIM_NORM:-01}"

source "$_sevir_common"
unset _sevir_common

require_ckpt "$PRIOR_CKPT" "PRIOR_CKPT (the unconditional prior)" \
  "unconditional_generation/scripts/SEVIR/train_prior.sh"

# Must match the prior.
HIDDEN_DIM="${HIDDEN_DIM:-32}"

# See forward_model_args in _common.sh. --hidden_dim below is the prior's only.
FORWARD_MODEL="${FORWARD_MODEL:-flowdas}"
fwd_args=($(forward_model_args "none flowdas unet fmw"))

exp_name="${EXP_NAME:-DAISI_SEVIR_${EXPERIMENT}_run_${DATA_INDEX}}"

print_sevir_config
echo "  method       DAISI forward_model=${FORWARD_MODEL} prior=${PRIOR_CKPT}"
date

python3 -m assimilation.assimilate \
    --exp_name "${exp_name}" \
    --method DAISI \
    "${fwd_args[@]}" \
    --model_path "${PRIOR_CKPT}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --attn_resolutions "${ATTN_RESOLUTIONS}" \
    --guide_method "${GUIDE_METHOD:-MMPS}" \
    --guidance_strength "${GUIDANCE_STRENGTH:-1.0}" \
    --noise "${NOISE:-invert}" \
    --tmin "${TMIN:-0.3}" \
    --euler_steps "${EULER_STEPS:-100}" \
    --invert_steps "${INVERT_STEPS:-100}" \
    --invert_eps "${INVERT_EPS:-0.03}" \
    --eps "${EPS:-0.03}" \
    $(sevir_online_args) \
    $(sevir_common_args)

date
fix_results_permissions
