#!/bin/bash
# SDA smoother on SEVIR. Knobs: guidance_strength, the Langevin corrector
# (corrections, tau; swept as a cross product), euler steps, and the window
# init_states (even; 2, 4 or 6 have checkpoints).
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_SDA.sh   # print, don't submit

source "$(dirname "${BASH_SOURCE[0]}")/_launcher.sh"

NODE_TYPE="${NODE_TYPE:-fat}"

# The UNet sees WINDOW_BATCH_SIZE * BATCH_SIZE windows per forward:
#   BATCH_SIZE         ensemble members per call (as in run_all_DAWIS.sh)
#   WINDOW_BATCH_SIZE  overlapping windows per call (SDA only); keep at 1
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-1}"
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

init_states_values=(6)
guidance_strengths=(3)
tau_values=(0.5)
corrections_values=(1)
euler_steps_values=(100)

# x0_sigma: obs error std of the fully observed GT slot that `--guide_first init`
# prepends at start_time-1, in run units (flowdas: 1 = 25.5 VIL counts; field
# std ~1.58). Smaller anchors harder but stiffens guidance. Must be > 0.
x0_sigma_values=(0.001)

if [[ "$sweep" == "false" ]]; then
    init_states_values=(6)
    guidance_strengths=(3)
    tau_values=(0.5)
    corrections_values=(1)
    x0_sigma_values=(0.001)   # the paper runs' value; change once tuned
fi

# X0_SIGMA_VALUES="0.3 0.1" overrides x0_sigma_values, in either mode.
if [[ -n "${X0_SIGMA_VALUES:-}" ]]; then
    read -r -a x0_sigma_values <<< "$X0_SIGMA_VALUES"
    echo "X0_SIGMA_VALUES override: x0_sigma_values=(${x0_sigma_values[*]})"
fi

require_nonempty init_states_values guidance_strengths tau_values \
                 corrections_values euler_steps_values x0_sigma_values \
                 experiments data_indices

apply_data_indices
# After the --sweep false block, which reassigns init_states_values.
apply_init_states
# Wall-clock limit (TIME_LIMIT overrides); sweeps reach slower settings.
if [[ "$sweep" == "false" ]]; then
    time_limit="${TIME_LIMIT:-04:30:00}"
else
    time_limit="${TIME_LIMIT:-1-00:00:00}"
fi

for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
for init_states in "${init_states_values[@]}"; do
for guidance in "${guidance_strengths[@]}"; do
for tau in "${tau_values[@]}"; do
for corrections in "${corrections_values[@]}"; do
for euler_steps in "${euler_steps_values[@]}"; do
for x0_sigma in "${x0_sigma_values[@]}"; do

    name="SDA_SEVIR"
    [[ "$sweep" == "true" ]] && \
        name="${name}_init${init_states}_g${guidance}_tau${tau//./p}_c${corrections}_x0s${x0_sigma//./p}"
    # Tagged only when x0_sigma differs from 3.
    [[ "$sweep" == "false" && "$x0_sigma" != "3" ]] && \
        name="${name}_x0s${x0_sigma//./p}"
    # `_init<N>` suffix (sweep false only; a sweep already names init_states).
    [[ "$sweep" == "false" ]] && name="${name}$(init_states_tag "$init_states")"
    name="${name}_run_${data_index}"

    sevir_submit "$name" run_SDA.sh -t "$time_limit" -C "$NODE_TYPE" -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" WINDOW_BATCH_SIZE="$WINDOW_BATCH_SIZE" \
        BATCH_SIZE="$BATCH_SIZE" \
        LOG_MEDIA="$LOG_MEDIA" EXPERIMENT="$experiment" DATA_INDEX="$data_index" \
        INIT_STATES="$init_states" GUIDANCE_STRENGTH="$guidance" \
        TAU="$tau" CORRECTIONS="$corrections" EULER_STEPS="$euler_steps" \
        X0_SIGMA="$x0_sigma"

done; done; done; done; done; done; done; done

sevir_summary SDA
