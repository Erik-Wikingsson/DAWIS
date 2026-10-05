#!/bin/bash
# Block-Gibbs smoother on SQG, initialised from a finished paper run.
# Usage: DRY_RUN=1 bash assimilation/scripts/SQG/run_all_Gibbs.bash
# Env knobs: DRY_RUN, DATA_INDICES, INIT_STATES_VALUES, ASSIM_INTERVAL,
# INITIAL_TRAJECTORY, TRAJECTORY_PATH, PAPER_RUNS_ROOT, BATCH_SIZE, TIME_LIMIT.

set -euo pipefail

# Sets cwd, $REPO_ROOT and $PYTHONPATH.
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"

# Job-mail options from $MAIL_TYPE / $MAIL_USER (must follow repo_root.sh).
source "$REPO_ROOT/scripts/sbatch_opts.sh"

# DRY_RUN=1 prints sbatch commands instead of submitting; --sweep true|false;
# DATA_INDICES selects trajectories; INIT_STATES_VALUES selects window sizes.
source "$REPO_ROOT/assimilation/scripts/_dry_run.sh"
source "$REPO_ROOT/assimilation/scripts/_sweep_arg.sh" "$@"
source "$REPO_ROOT/assimilation/scripts/_data_indices.sh"
source "$REPO_ROOT/assimilation/scripts/_init_states.sh"

# Window sizes with checkpoints; keep in sync with the `case $init_states` below.
_init_states_supported="6"
_init_states_override_prefix="ASSIM_FMW_PATH_SQG_ETA01_CHANNEL"

# sweep=true tags run names with block geometry and t_min. Pinned to false
# here, so --sweep is ignored.
sweep=false

# Exports $RESULTS_DIR for the submitted jobs.
source "$REPO_ROOT/scripts/results_dir.sh"

time_limit="${TIME_LIMIT:-1-00:00:00}"

# Ensemble members per GPU call (--batch_size); empty = whole ensemble at once.
# Only affects speed and memory, not results.
batch_size="${BATCH_SIZE:-}"

experiments=("noisy" "sparse" "multimodal" "saturating" )
methods=("GIBBS")

# Sampler settings shared with DAWIS; eps (stage 2) is set per variant below.
guidance_strength_default=(1.0)
guidance_strength_sat=(150)
corrections_values=(0)
tau_values=(0.5)
invert_eps_values=(0.03)   # stage 1, the unguided inversion
sampler_steps_values=(100) # --sampler_steps -> --euler_steps
noise_values=("SDEdit")    # ("SDEdit" "invert")
window_batch_sizes=(50)

# Block geometry and sweeps
block_modes=("sliding")
block_stride_values=(3)
block_directions=("alternate") # ("forward" "backward")
random_block_counts=(32)       # only read when block_mode=random
gibbs_sweeps_values=(2)
trajectory_var_values=("${TRAJECTORY_VAR:-x_smooth}") # or x_assim

# Variants, each an (endpoint, midpoint) t_min pair; runs are named Gibbs-<variant>:
#   Tied   (X,   X)
#   Gibbs  (1.0, 0.0)  plain block Gibbs
#   DAWIS  (1.0, X)
#   Soft   (0.9, X)    0.9 = $soft_endpoint_tmin
gibbs_variants=("Gibbs" "DAWIS" "Soft")   # ("Tied" "Gibbs" "DAWIS" "Soft")
soft_endpoint_tmin=0.9

# X per experiment (re-noising depth); every experiment needs an entry.
declare -A experiment_midpoint_tmin=(
    [noisy]=0.9
    [sparse]=0.7
    [multimodal]=0.7
    [saturating]=0.9
)

# Per-variant space-separated lists (multi-valued lists are tagged in run names):
#   variant_tmin           overrides X for all experiments (ignored by Gibbs)
#   variant_eps            --eps, the stage-2 guided forward pass
#   variant_num_endpoints  --gibbs_num_endpoints k, states frozen on each side
#                          of a block; needs 2*k < init_states+1 and
#                          block_stride <= init_states+1-2k.
declare -A variant_tmin=(
)
declare -A variant_eps=(
    [Tied]="0.03"
    [Gibbs]="0.03"
    [DAWIS]="0.03"
    [Soft]="0.03"
)
declare -A variant_num_endpoints=(
    [Tied]="1"
    [Gibbs]="1"
    [DAWIS]="1"
    [Soft]="1"
)

init_states_values=(6)      # = 2*sda_w
# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
# Override with ASSIM_INTERVAL=k; values != 1 are tagged into the run name.
# For this smoother, skipped steps get an empty observation mask.
assim_interval="${ASSIM_INTERVAL:-1}"
ai_tag=""
[[ "$assim_interval" != "1" ]] && ai_tag="_ai${assim_interval}"

n_times_values=(100)
n_ens_values=(20)

wandb_mode="${WANDB_MODE:-online}"

# Per-sweep logging: "none", "metrics" (one wandb run per sweep) or "full"
# (that plus the sweep trajectories as x_gibbs_k in the result file).
sweep_logging="${SWEEP_LOGGING:-full}"

# Method whose paper run (<METHOD>_<Experiment>_run_<i>_*.nc) initialises the sweeps.
initial_trajectory="${INITIAL_TRAJECTORY:-DAWISnumerical}"

# A specific .nc to start from instead (overrides initial_trajectory for all jobs).
trajectory_path="${TRAJECTORY_PATH:-}"
if [[ -n "$trajectory_path" && ! -f "$trajectory_path" ]]; then
    echo "ERROR: TRAJECTORY_PATH not found: $trajectory_path" >&2
    exit 1
fi

# Directory of the paper runs.
paper_runs_root="${PAPER_RUNS_ROOT:-}"
if [[ -z "$trajectory_path" && ! -d "$paper_runs_root" ]]; then
    echo "ERROR: paper-run directory not found: $paper_runs_root" >&2
    echo "  Set PAPER_RUNS_ROOT in your environment or the repo .env." >&2
    exit 1
fi

# Model checkpoints
source "$REPO_ROOT/assimilation/scripts/SQG/_models.sh"
for _w in 1 2 3 4 5 6; do printf -v "DAWIS_$_w" '%s' "$(sqg_fmw_ckpt $_w 2>/dev/null)"; done

# Test trajectory indices (0 = tuning trajectory, 1-10 = evaluation).
data_indices=(0 1 2 3 4 5 6 7 8 9 10)
# Env overrides.
apply_data_indices
apply_init_states

# Fail here rather than in a queued job if a table above is missing an entry.
for experiment in "${experiments[@]}"; do
    if [[ -z "${experiment_midpoint_tmin[$experiment]:-}" ]]; then
        echo "ERROR: no midpoint t_min for experiment '${experiment}'" >&2
        echo "  Add it to experiment_midpoint_tmin in ${BASH_SOURCE[0]}." >&2
        exit 1
    fi
done
for v in "${gibbs_variants[@]}"; do
    case "$v" in Tied|Gibbs|DAWIS|Soft) ;;
        *) echo "ERROR: unknown gibbs_variant '${v}'" >&2; exit 1;;
    esac
    for table in variant_eps variant_num_endpoints; do
        declare -n _t="$table"
        if [[ -z "${_t[$v]:-}" ]]; then
            echo "ERROR: no ${table} entry for gibbs_variant '${v}'" >&2
            exit 1
        fi
        unset -n _t
    done
    # 2*k < block length (= init_states + 1), checked by the parser too.
    for k in ${variant_num_endpoints[$v]}; do
        for init_states in "${init_states_values[@]}"; do
            if (( 2 * k >= init_states + 1 )); then
                echo "ERROR: gibbs_variant '${v}': gibbs_num_endpoints=${k} needs" \
                     "2*k < init_states+1 = $((init_states + 1))" >&2
                exit 1
            fi
        done
    done
done

# Only multi-valued dimensions are tagged into the run name.
tag_invert_eps=false; (( ${#invert_eps_values[@]} > 1 )) && tag_invert_eps=true

for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
    if [[ "$experiment" == "saturating" ]]; then
        guidance_strengths=("${guidance_strength_sat[@]}")
    else
        guidance_strengths=("${guidance_strength_default[@]}")
    fi

    # Check the initial run exists (capitalised or lowercase experiment name).
    if [[ -z "$trajectory_path" ]]; then
        experiment_cap="${experiment^}"
        initial_trajectory_matches=(
            "${paper_runs_root}/${initial_trajectory}_${experiment_cap}_run_${data_index}_"*.nc
        )
        if [[ ! -e "${initial_trajectory_matches[0]}" ]]; then
            initial_trajectory_matches=(
                "${paper_runs_root}/${initial_trajectory}_${experiment}_run_${data_index}_"*.nc
            )
        fi
        if [[ ! -e "${initial_trajectory_matches[0]}" ]]; then
            echo "ERROR: no run '${initial_trajectory}_${experiment_cap}_run_${data_index}_*.nc' in ${paper_runs_root}" >&2
            echo "  Available methods for ${experiment_cap}/d${data_index}:" >&2
            ls "${paper_runs_root}" 2>/dev/null \
                | sed -n -E "s/^(.+)_${experiment_cap}_run_${data_index}_.*\\.nc$/    \\1/p" \
                | sort -u >&2
            exit 1
        fi
    fi

for method in "${methods[@]}"; do
for init_states in "${init_states_values[@]}"; do
    case $init_states in
        1) model_path="$DAWIS_1"; hidden_dim=64;  channel_mult_emb=4; channel_mult_noise=2;;
        2) model_path="$DAWIS_2"; hidden_dim=96;  channel_mult_emb=4; channel_mult_noise=2;;
        3) model_path="$DAWIS_3"; hidden_dim=128; channel_mult_emb=4; channel_mult_noise=2;;
        4) model_path="$DAWIS_4"; hidden_dim=160; channel_mult_emb=4; channel_mult_noise=2;;
        5) model_path="$DAWIS_5"; hidden_dim=192; channel_mult_emb=4; channel_mult_noise=2;;
        6) model_path="$DAWIS_6"; hidden_dim=224; channel_mult_emb=4; channel_mult_noise=2;;
        *) echo "Unknown init_states=${init_states}" >&2; exit 1;;
    esac
for sampler_steps in "${sampler_steps_values[@]}"; do
for invert_eps in "${invert_eps_values[@]}"; do
for guidance_strength in "${guidance_strengths[@]}"; do
for corrections in "${corrections_values[@]}"; do
for tau in "${tau_values[@]}"; do
for gibbs_sweeps in "${gibbs_sweeps_values[@]}"; do
for block_mode in "${block_modes[@]}"; do
    # block_stride only applies to the sliding schedule.
    if [[ "$block_mode" == "random" ]]; then
        block_strides=(1)
    else
        block_strides=("${block_stride_values[@]}")
    fi
for block_direction in "${block_directions[@]}"; do
for block_stride in "${block_strides[@]}"; do
for random_block_count in "${random_block_counts[@]}"; do
    [[ "$block_mode" != "random" && "$random_block_count" != "${random_block_counts[0]}" ]] && continue
for n_times in "${n_times_values[@]}"; do
for n_ens in "${n_ens_values[@]}"; do
for window_batch_size in "${window_batch_sizes[@]}"; do
for noise in "${noise_values[@]}"; do
for trajectory_var in "${trajectory_var_values[@]}"; do
for gibbs_variant in "${gibbs_variants[@]}"; do

    # Empty endpoint_tmin means endpoint = midpoint (Tied).
    read -ra midpoint_tmin_values <<< "${variant_tmin[$gibbs_variant]:-${experiment_midpoint_tmin[$experiment]}}"
    case "$gibbs_variant" in
        Tied)  endpoint_tmin="";;
        Gibbs) endpoint_tmin=1.0; midpoint_tmin_values=(0.0);;
        DAWIS) endpoint_tmin=1.0;;
        Soft)  endpoint_tmin="$soft_endpoint_tmin";;
    esac
    read -ra eps_values <<< "${variant_eps[$gibbs_variant]}"
    read -ra gibbs_num_endpoints_values <<< "${variant_num_endpoints[$gibbs_variant]}"

    tag_eps=false;       (( ${#eps_values[@]} > 1 )) && tag_eps=true
    tag_endpoints=false; (( ${#gibbs_num_endpoints_values[@]} > 1 )) && tag_endpoints=true
    tag_tmin=false;      (( ${#midpoint_tmin_values[@]} > 1 )) && tag_tmin=true

for gibbs_num_endpoints in "${gibbs_num_endpoints_values[@]}"; do
for sampler_eps in "${eps_values[@]}"; do
for gibbs_midpoint_tmin in "${midpoint_tmin_values[@]}"; do

    gibbs_endpoint_tmin="${endpoint_tmin:-$gibbs_midpoint_tmin}"

    # Empty at the default window, `_init<N>` otherwise.
    exp_name="Gibbs-${gibbs_variant}_${experiment}$(init_states_tag "$init_states")${ai_tag}"
    if [[ "$sweep" == "true" ]]; then
        exp_name="${exp_name}_${block_mode}_s${block_stride}"
        [[ "$block_mode" == "random" ]] && exp_name="${exp_name}_n${random_block_count}"
        exp_name="${exp_name}_t${gibbs_midpoint_tmin//./p}"
    elif [[ "$tag_tmin" == "true" ]]; then
        exp_name="${exp_name}_t${gibbs_midpoint_tmin//./p}"
    fi
    [[ "$tag_endpoints" == "true" ]] && exp_name="${exp_name}_k${gibbs_num_endpoints}"
    [[ "$tag_eps" == "true" ]] && exp_name="${exp_name}_eps${sampler_eps//./p}"
    [[ "$tag_invert_eps" == "true" ]] && exp_name="${exp_name}_ieps${invert_eps//./p}"
    exp_name="${exp_name}_run_${data_index}"

    extra_args=()
    [[ -n "$batch_size" ]] && extra_args+=(--batch_size "${batch_size}")
    [[ -n "$trajectory_path" ]] && extra_args+=(--trajectory_path "${trajectory_path}")

    echo "Submitting ${exp_name}"

    sbatch "${SBATCH_MAIL_ARGS[@]}" -t "${time_limit}" \
        --job-name="${exp_name}" \
        assimilation/scripts/SQG/run_experiment_Gibbs.bash \
        --exp_name "${exp_name}" \
        --method "${method}" \
        --experiment "${experiment}" \
        --data_index "${data_index}" \
        --assim_interval "${assim_interval}" \
        --model_path "${model_path}" \
        --hidden_dim "${hidden_dim}" \
        --channel_mult_emb "${channel_mult_emb}" \
        --channel_mult_noise "${channel_mult_noise}" \
        --init_states "${init_states}" \
        --n_times "${n_times}" \
        --n_ens "${n_ens}" \
        --window_batch_size "${window_batch_size}" \
        --sampler_steps "${sampler_steps}" \
        --eps "${sampler_eps}" \
        --invert_eps "${invert_eps}" \
        --guide_method MMPS \
        --guidance_strength "${guidance_strength}" \
        --noise "${noise}" \
        --corrections "${corrections}" \
        --tau "${tau}" \
        --gibbs_sweeps "${gibbs_sweeps}" \
        --block_mode "${block_mode}" \
        --block_direction "${block_direction}" \
        --block_stride "${block_stride}" \
        --random_block_count "${random_block_count}" \
        --gibbs_endpoint_tmin "${gibbs_endpoint_tmin}" \
        --gibbs_midpoint_tmin "${gibbs_midpoint_tmin}" \
        --gibbs_num_endpoints "${gibbs_num_endpoints}" \
        --fm_loss eta01_channel \
        --resample_filter "1,3,3,1" \
        --channel_mult "2,2,2" \
        --attn_resolutions 32 \
        --schedule linear_scalar \
        --noise_embedding positional \
        --wandb_mode "${wandb_mode}" \
        --sweep_logging "${sweep_logging}" \
        --trajectory_var "${trajectory_var}" \
        --initial_trajectory "${initial_trajectory}" \
        "${extra_args[@]}"

done; done; done
done; done; done; done; done; done; done; done; done; done
done; done; done; done; done; done; done; done; done; done
