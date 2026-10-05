#!/bin/bash
# Block-Gibbs smoother on SEVIR. Refines an existing trajectory: set
# TRAJECTORY_PATH to a result .nc, or `initial_trajectory` to an archived method
# name (looked up in $PAPER_RUNS_ROOT); TRAJECTORY_VAR picks x_smooth or x_assim.
# Knobs: block geometry, per-variant re-noising depths, sweep count, and the
# sampler settings shared with DAWIS.
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_Gibbs.sh   # print, don't submit

source "$(dirname "${BASH_SOURCE[0]}")/_launcher.sh"

NODE_TYPE="${NODE_TYPE:-fat}"

# Members per GPU call (memory only; MMPS keeps the graph per member).
BATCH_SIZE="${BATCH_SIZE:-10}"

# Normalization (see _common.sh): one network does both stages, so both are `flowdas`.
ASSIM_NORM="${ASSIM_NORM:-flowdas}"
FORWARD_NORM="${FORWARD_NORM:-flowdas}"
OUTPUT_NORM="${OUTPUT_NORM:-physical}"

# The DAISI paper's observation network (defined in _common.sh).
experiments=("sevir")

# Trajectories: 0 = tuning, 1-10 = evaluation.
data_indices=(1 2 3 4 5 6 7 8 9 10)   # the held-out comparison
#data_indices=(0 1 2 3 4 5 6 7 8 9 10) # everything
#data_indices=(1)                        # tune

# --sweep true: the grid below; --sweep false: the tuned point.
sweep="${SWEEP}"   # --sweep false -> the tuned point (see ../_sweep_arg.sh)

# Media (wandb videos) off for sweeps.
LOG_MEDIA="${LOG_MEDIA:-false}"

# Warm start: archived method name (ignored when TRAJECTORY_PATH is set).
initial_trajectory="${INITIAL_TRAJECTORY:-DAWISdawis}"
trajectory_var_values=("${TRAJECTORY_VAR:-x_smooth}")   # or x_assim
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"

# Per-sweep logging: none | metrics (one wandb run per sweep) | full (also saves
# each sweep's trajectory, ~25 MB per sweep).
SWEEP_LOGGING="${SWEEP_LOGGING:-full}"

WANDB_MODE="${WANDB_MODE:-online}"

time_limit="${TIME_LIMIT:-0-10:00:00}"

# Sampler settings inherited from DAWIS.
guide_method="MMPS"
guidance_strengths=(1.0)
corrections_values=(0)
tau_values=(0.5)
# eps (stage 2, the guided forward pass, --eps) is set per variant below.
invert_eps_values=(1)      # stage 1, the unguided inversion
sampler_steps_values=(100)    # --euler_steps
noise_values=("SDEdit")       # ("SDEdit" "invert")
window_batch_sizes=(50)

# A SEVIR event is 25 frames; start_time 6 leaves 19.
n_times_values=("${N_TIMES:-19}")
n_ens_values=("${N_ENS:-20}")

# Block geometry and sweeps
init_states_values=(6)
block_modes=(sliding)
block_stride_values=(3)
block_directions=("alternate")   # ("forward" "backward" "both")
# gibbs_num_endpoints (per variant, below): conditioning steps frozen on each
# side of a block. Needs 2k < init_states + 1 and block_stride <= init_states + 1 - 2k.
random_block_counts=(32)         # only read when block_mode=random
gibbs_sweeps_values=(4)

# Variants, each fixing the (endpoint, midpoint) t_min pair:
#   Tied   (X,   X)
#   Gibbs  (1.0, 0.0)  plain block Gibbs; runs once
#   DAWIS  (1.0, X)
#   Soft   (0.9, X)    0.9 = $soft_endpoint_tmin
# Runs are named Gibbs-<variant> when more than one is selected.
gibbs_variants=("DAWIS" "Gibbs")   # ("Tied" "Gibbs" "DAWIS" "Soft")
soft_endpoint_tmin=0.9

# Per-variant lists (space-separated, swept; multi-valued lists are tagged in the name):
#   variant_tmin           re-noising depth X (not used by Gibbs)
#   variant_eps            --eps, the stage-2 guided pass
#   variant_num_endpoints  --gibbs_num_endpoints
declare -A variant_tmin=(
    [DAWIS]="0.9"
    [Soft]="0.9"
)
declare -A variant_eps=(
    [Gibbs]="0.03"
    [DAWIS]="0.03"
    [Soft]="0.03"
)
declare -A variant_num_endpoints=(
    [Gibbs]="2"
    [DAWIS]="1"
    [Soft]="2"
)

if [[ "$sweep" == "false" ]]; then
    block_modes=(sliding)
    block_stride_values=(3)
fi

require_nonempty init_states_values block_modes block_stride_values \
                 block_directions random_block_counts \
                 gibbs_sweeps_values gibbs_variants \
                 guidance_strengths corrections_values tau_values \
                 invert_eps_values sampler_steps_values \
                 noise_values window_batch_sizes n_times_values n_ens_values \
                 trajectory_var_values experiments data_indices

# Every selected variant needs entries in the tables above.
for v in "${gibbs_variants[@]}"; do
    for table in variant_eps variant_num_endpoints; do
        declare -n _t="$table"
        if [[ -z "${_t[$v]:-}" ]]; then
            echo "ERROR: no ${table} entry for gibbs_variant '${v}'" >&2
            exit 1
        fi
        unset -n _t
    done
    if [[ "$v" != "Gibbs" && -z "${variant_tmin[$v]:-}" ]]; then
        echo "ERROR: no variant_tmin entry for gibbs_variant '${v}'" >&2
        exit 1
    fi
done

if [[ -n "$TRAJECTORY_PATH" && ! -f "$TRAJECTORY_PATH" ]]; then
    echo "ERROR: TRAJECTORY_PATH not found: $TRAJECTORY_PATH" >&2
    exit 1
fi

if [[ -z "$TRAJECTORY_PATH" && -z "$initial_trajectory" ]]; then
    echo "NOTE: neither TRAJECTORY_PATH nor initial_trajectory is set, so every"
    echo "  job starts from scratch rather than refining a DAWIS/SDA"
    echo "  trajectory. Set TRAJECTORY_PATH to a .nc in \$RESULTS_DIR, or"
    echo "  initial_trajectory to an archived method name, for the intended"
    echo "  experiment."
fi

# Tag the variant in the run name.
tag_variant=true

# Tag invert_eps only when it is swept.
if (( ${#invert_eps_values[@]} > 1 )); then
    tag_invert_eps=true
else
    tag_invert_eps=false
fi

apply_data_indices
# After the --sweep false block, which reassigns init_states_values.
apply_init_states
for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
for init_states in "${init_states_values[@]}"; do
for block_mode in "${block_modes[@]}"; do
for block_direction in "${block_directions[@]}"; do
for random_block_count in "${random_block_counts[@]}"; do
    [[ "$block_mode" != "random" && "$random_block_count" != "${random_block_counts[0]}" ]] && continue
for gibbs_variant in "${gibbs_variants[@]}"; do
for sweeps in "${gibbs_sweeps_values[@]}"; do
for guidance_strength in "${guidance_strengths[@]}"; do
for corrections in "${corrections_values[@]}"; do
for tau in "${tau_values[@]}"; do
for invert_eps in "${invert_eps_values[@]}"; do
for sampler_steps in "${sampler_steps_values[@]}"; do
for noise in "${noise_values[@]}"; do
for window_batch_size in "${window_batch_sizes[@]}"; do
for n_times in "${n_times_values[@]}"; do
for n_ens in "${n_ens_values[@]}"; do
for trajectory_var in "${trajectory_var_values[@]}"; do

    # `random` samples block positions, so it runs once at stride 1.
    if [[ "$block_mode" == "random" ]]; then
        block_strides=(1)
    else
        block_strides=("${block_stride_values[@]}")
    fi

    # Block count only for `random`; empty leaves the flag off.
    if [[ "$block_mode" == "random" ]]; then
        random_count="$random_block_count"
    else
        random_count=""
    fi

    for block_stride in "${block_strides[@]}"; do

    # The variant decides the midpoints to sweep; empty endpoint_tmin = follow
    # the midpoint (Tied).
    case "$gibbs_variant" in
        Tied)  endpoint_tmin="";                    read -ra midpoint_tmin_values <<< "${variant_tmin[Tied]}";;
        Gibbs) endpoint_tmin=1.0;                   midpoint_tmin_values=(0.0);;
        DAWIS) endpoint_tmin=1.0;                   read -ra midpoint_tmin_values <<< "${variant_tmin[DAWIS]}";;
        Soft)  endpoint_tmin="$soft_endpoint_tmin"; read -ra midpoint_tmin_values <<< "${variant_tmin[Soft]}";;
        *) echo "ERROR: unknown gibbs_variant '${gibbs_variant}'" >&2; exit 1;;
    esac
    read -ra eps_values <<< "${variant_eps[$gibbs_variant]}"
    read -ra gibbs_num_endpoints_values <<< "${variant_num_endpoints[$gibbs_variant]}"

    tag_eps=false;       (( ${#eps_values[@]} > 1 )) && tag_eps=true
    tag_endpoints=false; (( ${#gibbs_num_endpoints_values[@]} > 1 )) && tag_endpoints=true
    tag_tmin=false;      (( ${#midpoint_tmin_values[@]} > 1 )) && tag_tmin=true

    for gibbs_num_endpoints in "${gibbs_num_endpoints_values[@]}"; do
    for eps in "${eps_values[@]}"; do
    for midpoint_tmin in "${midpoint_tmin_values[@]}"; do

    endpoint="${endpoint_tmin:-$midpoint_tmin}"

    name="GIBBS_SEVIR"
    [[ "$tag_variant" == "true" ]] && name="Gibbs-${gibbs_variant}_SEVIR"
    if [[ "$sweep" == "true" ]]; then
        name="${name}_${block_mode}_s${block_stride}"
        [[ "$block_mode" == "random" ]] && name="${name}_n${random_block_count}"
        name="${name}_t${midpoint_tmin//./p}"
    elif [[ "$tag_tmin" == "true" ]]; then
        name="${name}_t${midpoint_tmin//./p}"
    fi
    [[ "$tag_endpoints" == "true" ]] && name="${name}_k${gibbs_num_endpoints}"
    [[ "$tag_eps" == "true" ]] && name="${name}_eps${eps//./p}"
    [[ "$tag_invert_eps" == "true" ]] && name="${name}_ieps${invert_eps//./p}"
    name="${name}_run_${data_index}"

    sevir_submit "$name" run_Gibbs.sh -t "$time_limit" -C "$NODE_TYPE" \
        -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" \
        BATCH_SIZE="$BATCH_SIZE" \
        LOG_MEDIA="$LOG_MEDIA" EXPERIMENT="$experiment" DATA_INDEX="$data_index" \
        INIT_STATES="$init_states" BLOCK_MODE="$block_mode" \
        BLOCK_STRIDE="$block_stride" BLOCK_DIRECTION="$block_direction" \
        RANDOM_BLOCK_COUNT="$random_count" \
        GIBBS_SWEEPS="$sweeps" \
        GIBBS_ENDPOINT_TMIN="$endpoint" \
        GIBBS_MIDPOINT_TMIN="$midpoint_tmin" \
        GIBBS_NUM_ENDPOINTS="$gibbs_num_endpoints" \
        GUIDE_METHOD="$guide_method" GUIDANCE_STRENGTH="$guidance_strength" \
        CORRECTIONS="$corrections" TAU="$tau" NOISE="$noise" \
        EPS="$eps" INVERT_EPS="$invert_eps" \
        EULER_STEPS="$sampler_steps" \
        WINDOW_BATCH_SIZE="$window_batch_size" \
        N_TIMES="$n_times" N_ENS="$n_ens" \
        TRAJECTORY_PATH="$TRAJECTORY_PATH" \
        INITIAL_TRAJECTORY="$initial_trajectory" \
        TRAJECTORY_VAR="$trajectory_var" \
        SWEEP_LOGGING="$SWEEP_LOGGING" WANDB_MODE="$WANDB_MODE"

    done; done; done

done; done; done; done; done; done; done; done; done; done
done; done; done; done; done; done; done; done; done

sevir_summary Gibbs
