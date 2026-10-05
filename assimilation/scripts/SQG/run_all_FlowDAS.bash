#!/bin/bash
# FlowDAS baseline (Jia et al.) across all SQG experiments. Observations are used
# inside the interpolant's Euler-Maruyama loop, so one pass yields forecast and
# analysis. FlowDAS conditions on a 6-state window, so runs start at t=6.
#   DRY_RUN=1 bash assimilation/scripts/SQG/run_all_FlowDAS.bash

set -euo pipefail

# Sets cwd, $REPO_ROOT and $PYTHONPATH.
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"

# Job-mail options from $MAIL_TYPE / $MAIL_USER (must follow repo_root.sh).
source "$REPO_ROOT/scripts/sbatch_opts.sh"

# DRY_RUN=1 prints sbatch commands instead of submitting; --sweep true|false; DATA_INDICES selects trajectories.
source "$REPO_ROOT/assimilation/scripts/_dry_run.sh"
source "$REPO_ROOT/assimilation/scripts/_sweep_arg.sh" "$@"
source "$REPO_ROOT/assimilation/scripts/_data_indices.sh"

sweep="${SWEEP}"
mkdir -p slurm_logs

DRY_RUN="${DRY_RUN:-0}"

# Guided Euler-Maruyama sampling is much slower than DAISI.
time_limit="12:00:00"

gpu_resources=(--gpus=1 -C thin)

# Experiments
experiments=("noisy" "sparse" "saturating" "multimodal")

# Test trajectory indices (0 = tuning trajectory, 1-10 = evaluation).
data_indices=(0 10)


# Hyperparameters
# Reference EM_sample_steps / MC_times. Raising MC_times is cheap.
euler_steps=(500)
mc_times_values=(25)

# guidance_strength is set per experiment in the loop below.

# Fixed by the SQG checkpoint, which conditions on 6 past states.
window=6
start_time=6

# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
# Override with ASSIM_INTERVAL=k; values != 1 are tagged into the run name.
assim_interval="${ASSIM_INTERVAL:-1}"
ai_tag=""
[[ "$assim_interval" != "1" ]] && ai_tag="_ai${assim_interval}"

n_ens_values=(20)
n_times_values=(100)
seed=0
init_state="GT"
device="cuda"
wandb_mode="online"

# Fail early if any array is empty (no jobs would be submitted).
for arr in experiments data_indices euler_steps \
           mc_times_values n_ens_values n_times_values; do
    declare -n _ref="$arr"
    if [[ ${#_ref[@]} -eq 0 ]]; then
        echo "ERROR: array '${arr}' is empty -- no jobs would be submitted." >&2
        exit 1
    fi
    unset -n _ref
done

counter=0
# DATA_INDICES=... overrides the list above.
apply_data_indices
for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do

    # Standard guidance strength per experiment.
    case $experiment in
        "multimodal") guidance_strengths=(0.5);;
        "saturating") guidance_strengths=(5);;
        "noisy")      guidance_strengths=(1);;
        "sparse")     guidance_strengths=(3);;
        *)
            echo "Unknown experiment=${experiment}"
            exit 1
            ;;
    esac

    for euler_step in "${euler_steps[@]}"; do
    for guidance in "${guidance_strengths[@]}"; do
    for mc_times in "${mc_times_values[@]}"; do
    for n_ens in "${n_ens_values[@]}"; do
    for n_times in "${n_times_values[@]}"; do

        exp_name="FlowDAS_${experiment}${ai_tag}_run_${data_index}"

        counter=$((counter + 1))
        echo "Submitting job ${exp_name}"


        sbatch "${SBATCH_MAIL_ARGS[@]}" -t "$time_limit" "${gpu_resources[@]}" --job-name="$exp_name" \
            assimilation/scripts/SQG/run_experiment_FlowDAS.bash \
            --exp_name "${exp_name}" \
            --experiment "${experiment}" \
            --data_index "${data_index}" \
            --assim_interval "${assim_interval}" \
            --euler_steps "${euler_step}" \
            --guidance_strength "${guidance}" \
            --mc_times "${mc_times}" \
            --window "${window}" \
            --start_time "${start_time}" \
            --device "${device}" \
            --n_ens "${n_ens}" \
            --n_times "${n_times}" \
            --seed "${seed}" \
            --init_state "${init_state}" \
            --fixed_obs \
            --wandb_mode "${wandb_mode}"

    done; done; done; done; done
done
done

echo "Total FlowDAS jobs: ${counter} (dry_run=${DRY_RUN})"
