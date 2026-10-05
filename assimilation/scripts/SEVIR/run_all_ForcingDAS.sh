#!/bin/bash
# ForcingDAS-Pyr on SEVIR: the DAWIS window model with the fixed-lag pyramid
# schedule (--pyramid; --n_fixed oldest slots frozen at data). Sweeps guidance.
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_ForcingDAS.sh   # print, don't submit

source "$(dirname "${BASH_SOURCE[0]}")/_launcher.sh"

NODE_TYPE="${NODE_TYPE:-fat}"

# Members per GPU call (memory only; 20 members at init_states 6 don't fit at once).
BATCH_SIZE="${BATCH_SIZE:-5}"

# Normalization (see _common.sh): one network does both stages, so both are `flowdas`.
ASSIM_NORM="${ASSIM_NORM:-flowdas}"
FORWARD_NORM="${FORWARD_NORM:-flowdas}"
OUTPUT_NORM="${OUTPUT_NORM:-physical}"

# The DAISI paper's observation network (defined in _common.sh).
experiments=("sevir")

# Trajectories: 0 = tuning, 1-10 = evaluation.
data_indices=(1 2 3 4 5 6 7 8 9 10)   # the held-out comparison
# data_indices=(0 1 2 3 4 5 6 7 8 9 10) # everything
# data_indices=(0)                        # tune

# --sweep true: the grid below; --sweep false: the tuned point.
sweep="${SWEEP}"   # --sweep false -> the tuned point (see ../_sweep_arg.sh)

# Media (wandb videos) off for sweeps.
LOG_MEDIA="${LOG_MEDIA:-false}"

# Start from the clean GT window (run_DAWIS.sh defaults to GT_edit).
INIT_STATE="${INIT_STATE:-GT}"

init_states_values=(6)
n_fixed_values=(1) # 1 Fixed is most true to the original ForcingDAS paper.
guidance_strengths=(3)
euler_steps_values=(20)

if [[ "$sweep" == "false" ]]; then
    n_fixed_values=(0)
    guidance_strengths=(3)
fi

require_nonempty init_states_values n_fixed_values guidance_strengths \
                 euler_steps_values experiments data_indices

apply_data_indices
# After the --sweep false block, which reassigns init_states_values.
apply_init_states
# Wall-clock limit (TIME_LIMIT overrides); sweeps reach slower settings.
if [[ "$sweep" == "false" ]]; then
    time_limit="${TIME_LIMIT:-01:00:00}"
else
    time_limit="${TIME_LIMIT:-06:00:00}"
fi

for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
for init_states in "${init_states_values[@]}"; do
for n_fixed in "${n_fixed_values[@]}"; do
for guidance in "${guidance_strengths[@]}"; do
for euler_steps in "${euler_steps_values[@]}"; do

    if (( n_fixed >= init_states )); then
        echo "  skipping n_fixed ${n_fixed}: not fewer than init_states ${init_states}"
        continue
    fi

    name="ForcingDAS_SEVIR"
    [[ "$sweep" == "true" ]] && name="${name}_nf${n_fixed}_g${guidance}"
    # `_init<N>` suffix only when the window differs from the default.
    name="${name}$(init_states_tag "$init_states")"
    name="${name}_run_${data_index}"

    sevir_submit "$name" run_DAWIS.sh -t "$time_limit" -C "$NODE_TYPE" -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" BATCH_SIZE="$BATCH_SIZE" \
        LOG_MEDIA="$LOG_MEDIA" EXPERIMENT="$experiment" DATA_INDEX="$data_index" \
        FORWARD_MODEL=none NOISE=None INIT_STATES="$init_states" \
        INIT_STATE="$INIT_STATE" \
        PYRAMID=true N_FIXED="$n_fixed" \
        GUIDE_ALL_STEPS=true GUIDANCE_STRENGTH="$guidance" \
        EULER_STEPS="$euler_steps"

done; done; done; done; done; done

sevir_summary ForcingDAS
