#!/bin/bash
# DAWIS on SEVIR, swept over the inversion schedule and window size.
#   tmin_start/tmin_end  re-noising depth per window slot
#   init_states          window width (2, 4 or 6; hidden_dim = 32*(init+1))
#   guide_all_steps      guide every window state (smoother) or only the analysis
#   eps / invert_eps     interpolant noise in stage 2 (guided) / stage 1 (inversion)
# FORWARD_MODEL: dawis (DAWIS-Joint, no propagator), flowdas (needs init_states 6),
# unet, fmw, or none (ablation).
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_DAWIS.sh   # print, don't submit

source "$(dirname "${BASH_SOURCE[0]}")/_launcher.sh"

NODE_TYPE="${NODE_TYPE:-fat}"

# Members per GPU call (memory only; MMPS keeps the graph per member, so 20
# members at init_states 6 don't fit at once).
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

# SAVE_WINDOW_STATES=true saves the whole window (x_window) for lag analysis.
SAVE_WINDOW_STATES="${SAVE_WINDOW_STATES:-false}"

# Uncompressed x_window size in MB, for the log line.
window_size_mb () {
    awk -v t="${N_TIMES:-19}" -v w="$1" -v e="${N_ENS:-20}" -v x=128 \
        'BEGIN{printf "%.0f", t * (w + 1) * e * x * x * 4 / 1048576}'
}

init_states_values=(6)
tmin_starts=(0.4) # 
tmin_ends=(0.1) # 
guide_all_steps_values=(true)
guidance_strengths=(5.0 7.0 10.0)
# tmin_init: head over the leading init_states-1 slots, then ic; "none" disables.
tmin_init_heads=(1.0)
tmin_init_ics=(0.9)
forward_models=(flowdas dawis)

# Interpolant noise: eps = stage 2 (guided pass), invert_eps = stage 1
# (unguided inversion; used because NOISE=invert).
eps_values=(1.0)
invert_eps_values=(2.5)

# Seeding at t=0: GT = clean ground-truth window (required by the `sevir`
# preset); GT_edit = an SDEdit pass over the initial condition.
init_state_values=(GT) # locked: the `sevir` experiment refuses anything but GT

if [[ "$sweep" == "false" ]]; then
    # Tuned point: the paper's SEVIR rows.
    #   flowdas  DAWIS Filter / Lagged Smoother (archived as DAWISnumerical)
    #   dawis    DAWIS-Joint Filter / Lagged Smoother (archived as DAWISdawis)
    forward_models=(flowdas dawis)
    init_states_values=(6)
    tmin_starts=(0.4)
    tmin_ends=(0.1)
    guide_all_steps_values=(true)
    guidance_strengths=(2.5)
    tmin_init_heads=(1.0)
    tmin_init_ics=(0.9)
    init_state_values=(GT)
    # eps/invert_eps were tuned with forward_model=dawis only.
    eps_values=(1.0)
    invert_eps_values=(2.5)
fi

require_nonempty init_states_values tmin_starts tmin_ends \
                 guide_all_steps_values guidance_strengths forward_models \
                 init_state_values experiments data_indices \
                 eps_values invert_eps_values

# Tag the seeding / noise levels in the name only when swept.
if (( ${#init_state_values[@]} > 1 )); then
    tag_init_state=true
else
    tag_init_state=false
fi

if (( ${#eps_values[@]} > 1 )); then
    tag_eps=true
else
    tag_eps=false
fi
if (( ${#invert_eps_values[@]} > 1 )); then
    tag_invert_eps=true
else
    tag_invert_eps=false
fi

# `head` repeated n-1 times, then `ic` (length init_states).
make_tmin_init () {
    awk -v h="$1" -v c="$2" -v n="$3" 'BEGIN{
        for (i = 0; i < n; i++)
            printf "%s%.6g", (i ? "," : ""), (i == n - 1 ? c : h)
    }'
}

apply_data_indices
# After the --sweep false block, which reassigns init_states_values.
apply_init_states
# Wall-clock limit (TIME_LIMIT overrides); sweeps reach slower settings.
if [[ "$sweep" == "false" ]]; then
    time_limit="${TIME_LIMIT:-03:00:00}"
else
    time_limit="${TIME_LIMIT:-10:00:00}"
fi

for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
for forward_model in "${forward_models[@]}"; do
for init_states in "${init_states_values[@]}"; do
for tmin_start in "${tmin_starts[@]}"; do
for tmin_end in "${tmin_ends[@]}"; do
for guide_all in "${guide_all_steps_values[@]}"; do
for guidance in "${guidance_strengths[@]}"; do
for eps in "${eps_values[@]}"; do
for invert_eps in "${invert_eps_values[@]}"; do
for ti_head in "${tmin_init_heads[@]}"; do
for ti_ic in "${tmin_init_ics[@]}"; do
for init_state in "${init_state_values[@]}"; do

    # tmin_init is enabled or disabled as a pair.
    if [[ "$ti_head" == "none" || "$ti_ic" == "none" ]]; then
        if [[ "$ti_head" != "$ti_ic" ]]; then
            echo "  skipping tmin_init head=${ti_head} ic=${ti_ic}: 'none' must be in both"
            continue
        fi
        tmin_init=""
    else
        tmin_init="$(make_tmin_init "$ti_head" "$ti_ic" "$init_states")"
    fi

    # No SEVIR checkpoint beyond 6.
    if (( init_states > 6 )); then
        echo "  skipping init_states ${init_states}: no SEVIR checkpoint"
        continue
    fi

    # FlowDAS conditions on exactly 6 frames; skip shallower windows.
    if [[ "$forward_model" == "flowdas" ]] && (( init_states < 6 )); then
        echo "  skipping forward_model flowdas at init_states ${init_states}:" \
             "FlowDAS conditions on 6 frames"
        continue
    fi

    # Forward model is part of the name.
    name="DAWIS-${forward_model}_SEVIR"
    if [[ "$sweep" == "true" ]]; then
        name="${name}_init${init_states}_te${tmin_end//./p}_ga${guide_all}"
    fi
    [[ "$tag_eps" == "true" ]] && name="${name}_eps${eps//./p}"
    [[ "$tag_invert_eps" == "true" ]] && name="${name}_ieps${invert_eps//./p}"
    [[ "$tag_init_state" == "true" ]] && name="${name}_${init_state}"
    # `_init<N>` suffix (sweep false only; a sweep already names init_states).
    [[ "$sweep" == "false" ]] && name="${name}$(init_states_tag "$init_states")"
    name="${name}_run_${data_index}"

    sevir_submit "$name" run_DAWIS.sh -t "$time_limit" -C "$NODE_TYPE" -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" \
        BATCH_SIZE="$BATCH_SIZE" \
        LOG_MEDIA="$LOG_MEDIA" EXPERIMENT="$experiment" DATA_INDEX="$data_index" \
        FORWARD_MODEL="$forward_model" INIT_STATES="$init_states" \
        TMIN_START="$tmin_start" TMIN_END="$tmin_end" \
        TMIN_INIT="$tmin_init" GUIDE_ALL_STEPS="$guide_all" \
        INIT_STATE="$init_state" \
        SAVE_WINDOW_STATES="$SAVE_WINDOW_STATES" \
        GUIDANCE_STRENGTH="$guidance" NOISE=invert \
        EPS="$eps" INVERT_EPS="$invert_eps"

    if [[ "$SAVE_WINDOW_STATES" == "true" ]]; then
        echo "        saving $((init_states + 1)) lags x ${N_ENS:-20} members per time" \
             "(~$(window_size_mb "$init_states") MB of x_window, uncompressed)"
    fi

done; done; done; done; done; done; done; done; done; done; done; done; done

sevir_summary DAWIS

if [[ "$SAVE_WINDOW_STATES" == "true" ]]; then
    echo "Every window slot is stored as x_window in the result files."
fi
