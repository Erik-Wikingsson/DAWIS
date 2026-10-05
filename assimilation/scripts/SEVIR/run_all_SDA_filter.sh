#!/bin/bash
# SDA's corrector run as a filter on SEVIR (the smoother is run_all_SDA.sh):
# --noise SDEdit at one fixed level (tmin_start == tmin_end), denoised with
# `corrections` Langevin steps of strength `tau`. Sweeps (corrections, tau) jointly.
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_SDA_filter.sh   # print, don't submit

source "$(dirname "${BASH_SOURCE[0]}")/_launcher.sh"

NODE_TYPE="${NODE_TYPE:-fat}"

# Members per GPU call (memory only). Does not bound an INIT_STATE=GT_edit
# seeding pass, which runs at the full n_ens.
BATCH_SIZE="${BATCH_SIZE:-5}"

# Normalization (see _common.sh): one network does both stages, so both are `flowdas`.
ASSIM_NORM="${ASSIM_NORM:-flowdas}"
FORWARD_NORM="${FORWARD_NORM:-flowdas}"
OUTPUT_NORM="${OUTPUT_NORM:-physical}"

# The DAISI paper's observation network (defined in _common.sh).
experiments=("sevir")

# Trajectories: 0 = tuning, 1-10 = evaluation.
# data_indices=(1 2 3 4 5 6 7 8 9 10)   # the held-out comparison
# data_indices=(0 1 2 3 4 5 6 7 8 9 10) # everything
data_indices=(0)                        # tune

# --sweep true: the grid below; --sweep false: the tuned point.
sweep="${SWEEP}"   # --sweep false -> the tuned point (see ../_sweep_arg.sh)

# Media (wandb videos) off for sweeps.
LOG_MEDIA="${LOG_MEDIA:-false}"

# Start from the clean GT window (run_DAWIS.sh defaults to GT_edit).
INIT_STATE="${INIT_STATE:-GT}"

init_states_values=(6)
tmin_values=(0.0)
corrections_values=(1)
tau_values=(0.3)
guidance_strengths=(10)
x0_sigma_values=(0.1)

if [[ "$sweep" == "false" ]]; then
    tmin_values=(0.0)
    corrections_values=(1)
    tau_values=(0.3)
    x0_sigma_values=(0.1)
    guidance_strengths=(10)
fi

if [[ -n "${X0_SIGMA_VALUES:-}" ]]; then
    read -r -a x0_sigma_values <<< "$X0_SIGMA_VALUES"
    echo "X0_SIGMA_VALUES override: x0_sigma_values=(${x0_sigma_values[*]})"
fi

require_nonempty init_states_values tmin_values corrections_values tau_values \
                 guidance_strengths x0_sigma_values experiments data_indices

apply_data_indices
# After the --sweep false block, which reassigns init_states_values.
apply_init_states
# Wall-clock limit (TIME_LIMIT overrides); sweeps reach slower settings.
if [[ "$sweep" == "false" ]]; then
    time_limit="${TIME_LIMIT:-06:00:00}"
else
    time_limit="${TIME_LIMIT:-12:00:00}"
fi

for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
for init_states in "${init_states_values[@]}"; do
for tmin in "${tmin_values[@]}"; do
for corrections in "${corrections_values[@]}"; do
for tau in "${tau_values[@]}"; do
for guidance in "${guidance_strengths[@]}"; do
for x0_sigma in "${x0_sigma_values[@]}"; do

    if [[ "$x0_sigma" == "none" ]]; then
        guide_first=none x0_sigma_arg=1   # unused under guide_first none
    else
        guide_first=all x0_sigma_arg="$x0_sigma"
    fi

    name="SDA_filter_SEVIR"
    [[ "$sweep" == "true" ]] && \
        name="${name}_t${tmin//./p}_g${guidance}_c${corrections}_tau${tau//./p}"
    # x0s tag whenever the anchor is on.
    [[ "$guide_first" == "all" ]] && name="${name}_x0s${x0_sigma//./p}"
    # `_init<N>` suffix only when the window differs from the default.
    name="${name}$(init_states_tag "$init_states")"
    name="${name}_run_${data_index}"

    # One fixed noise level across the window.
    sevir_submit "$name" run_DAWIS.sh -t "$time_limit" -C "$NODE_TYPE" -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" BATCH_SIZE="$BATCH_SIZE" \
        LOG_MEDIA="$LOG_MEDIA" EXPERIMENT="$experiment" DATA_INDEX="$data_index" \
        FORWARD_MODEL=none NOISE=SDEdit EPS=1 \
        INIT_STATES="$init_states" INIT_STATE="$INIT_STATE" \
        TMIN_START="$tmin" TMIN_END="$tmin" \
        CORRECTIONS="$corrections" TAU="$tau" \
        GUIDE_ALL_STEPS=true GUIDANCE_STRENGTH="$guidance" \
        GUIDE_FIRST="$guide_first" X0_SIGMA="$x0_sigma_arg" \
        EULER_STEPS=100

done; done; done; done; done; done; done; done

sevir_summary SDA_filter
