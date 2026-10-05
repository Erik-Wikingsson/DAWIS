#!/bin/bash
# Guided-forecast (JointAR) ablation on SEVIR: the DAWIS window model with
# FORWARD_MODEL=none, NOISE=None and a per-slot tmin pattern per context mode:
#   past    conditioning slots held at data, only the target regenerated
#   future  oldest slot held, the rest regenerated (tmin_init pins the conditioning)
#   uncond  whole window regenerated from the observations
# Guidance strength is strongly mode-dependent, hence a wide log grid.
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_GuidedForecast.sh   # print, don't submit

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

context_modes=(future)
init_states_values=(6)
guidance_strengths=(1)
euler_steps_values=(100)

if [[ "$sweep" == "false" ]]; then
    # Tuned point: both paper rows (JointAR past and future); guidance is set
    # per mode by tuned_guidance.
    context_modes=(past future)
    init_states_values=(6)
    euler_steps_values=(100)
    guidance_strengths=(tuned)   # replaced per mode; see tuned_guidance below
fi

# Tuned guidance per context mode.
tuned_guidance () {
    case "$1" in
        past)   echo 3 ;;
        future) echo 1 ;;
        *) echo "No tuned guidance for context_mode=$1" >&2; exit 1 ;;
    esac
}

require_nonempty context_modes init_states_values guidance_strengths \
                 euler_steps_values experiments data_indices

# Per-slot tmin pattern (length init_states + 1) -> "<tmin>|<tmin_init>";
# empty tmin_init = none.
context_tmin () {
    local mode="$1" n="$2"
    awk -v mode="$mode" -v n="$n" 'BEGIN{
        if (mode == "past") {
            for (i = 0; i < n; i++) printf "%s1.0", (i ? "," : "");
            printf ",0.0|";
        } else if (mode == "future") {
            printf "1.0";
            for (i = 1; i < n; i++) printf ",0.0";
            printf ",0.0|";
            for (i = 0; i < n; i++) printf "%s1.0", (i ? "," : "");
        } else {
            for (i = 0; i <= n; i++) printf "%s0.0", (i ? "," : "");
            printf "|";
        }
    }'
}

apply_data_indices
# After the --sweep false block, which reassigns init_states_values.
apply_init_states
# Wall-clock limit (TIME_LIMIT overrides); sweeps reach slower settings.
if [[ "$sweep" == "false" ]]; then
    time_limit="${TIME_LIMIT:-02:30:00}"
else
    time_limit="${TIME_LIMIT:-10:00:00}"
fi

for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
for context_mode in "${context_modes[@]}"; do
for init_states in "${init_states_values[@]}"; do
for guidance in "${guidance_strengths[@]}"; do
for euler_steps in "${euler_steps_values[@]}"; do

    IFS='|' read -r tmin tmin_init <<< "$(context_tmin "$context_mode" "$init_states")"

    if [[ "$sweep" == "false" ]]; then
        guidance="$(tuned_guidance "$context_mode")"
    fi

    # `future` is a smoother: guide every window state. `past` guides the analysis only.
    guide_all=false
    [[ "$context_mode" == "future" ]] && guide_all=true

    name="JointAR-${context_mode}_SEVIR"
    [[ "$sweep" == "true" ]] && name="${name}_g${guidance}"
    # `_init<N>` suffix only when the window differs from the default.
    name="${name}$(init_states_tag "$init_states")"
    name="${name}_run_${data_index}"

    sevir_submit "$name" run_DAWIS.sh -t "$time_limit" -C "$NODE_TYPE" -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" BATCH_SIZE="$BATCH_SIZE" \
        LOG_MEDIA="$LOG_MEDIA" EXPERIMENT="$experiment" DATA_INDEX="$data_index" \
        FORWARD_MODEL=none NOISE=None EPS=1.0 INVERT_EPS=0 \
        INIT_STATES="$init_states" TMIN="$tmin" TMIN_INIT="$tmin_init" \
        INIT_STATE="$INIT_STATE" \
        GUIDE_ALL_STEPS="$guide_all" GUIDANCE_STRENGTH="$guidance" \
        EULER_STEPS="$euler_steps"

done; done; done; done; done; done

sevir_summary GuidedForecast
