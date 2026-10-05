#!/bin/bash
#SBATCH -J sevir_flowdas
#SBATCH -t 12:00:00
#SBATCH --gpus=1 -C "fat"
#SBATCH --output ./slurm_logs/%A_%x.out
#
# FlowDAS on SEVIR (Jia et al., https://github.com/umjiayx/FlowDAS).
#
# Guidance is applied inside the interpolant's sampling loop, so it runs with
# --forward_model none. The architecture is FLOWDAS_PRESETS["SEVIR"] in
# forecasting/flowdas_forecaster.py. The checkpoint was trained on `train`,
# which overlaps `val` in date range: report `test` only.
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

# The checkpoint conditions on exactly 6 past states.
WINDOW=6
START_TIME="${START_TIME:-6}"
check_window_fits "$WINDOW"

require_ckpt "$FLOWDAS_CKPT" "FLOWDAS_CKPT (FlowDAS's SEVIR checkpoint)" \
    "the FlowDAS repository (https://github.com/umjiayx/FlowDAS)"

# FlowDAS starts from the conditioning window (assimilate.py asserts init_std 0).
INIT_STD=0

exp_name="${EXP_NAME:-FlowDAS_SEVIR_${EXPERIMENT}_run_${DATA_INDEX}}"

print_sevir_config
echo "  method       FlowDAS window=${WINDOW} euler_steps=${EULER_STEPS:-500}"
echo "               guidance=${GUIDANCE_STRENGTH:-1} mc_times=${MC_TIMES:-25}"
echo "  checkpoint   ${FLOWDAS_CKPT}"
date

python3 -m assimilation.assimilate \
    --exp_name "${exp_name}" \
    --method FlowDAS \
    --forward_model none \
    --model_path "${FLOWDAS_CKPT}" \
    --window "${WINDOW}" \
    --euler_steps "${EULER_STEPS:-500}" \
    --guidance_strength "${GUIDANCE_STRENGTH:-1}" \
    --mc_times "${MC_TIMES:-25}" \
    --flowdas_tmin "${FLOWDAS_TMIN:-0.001}" \
    --flowdas_tmax "${FLOWDAS_TMAX:-0.999}" \
    $(sevir_online_args) \
    $(sevir_common_args)

date
fix_results_permissions
