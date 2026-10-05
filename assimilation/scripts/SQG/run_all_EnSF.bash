#!/bin/bash
# EnSF filter baseline (Bao, Zhang & Zhang) across all SQG experiments, with the
# same trajectories and ensemble settings as the DAISI / LETKF launchers.
#   DRY_RUN=1 bash assimilation/scripts/SQG/run_all_EnSF.bash

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

time_limit="03:00:00"

gpu_resources=(--gpus=1 -C thin)

# Experiments
experiments=("noisy" "sparse" "saturating" "multimodal")

# Test trajectory indices (0 = tuning trajectory, 1-10 = evaluation).
data_indices=(0 10)


# Hyperparameters
# EnSF is controlled only by the diffusion parameter and the number of Euler steps.
eps_alpha_values=(0.05)
euler_steps=(1000)

# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
# Override with ASSIM_INTERVAL=k; values != 1 are tagged into the run name.
assim_interval="${ASSIM_INTERVAL:-1}"
ai_tag=""
[[ "$assim_interval" != "1" ]] && ai_tag="_ai${assim_interval}"

n_ens_values=(20)
n_times_values=(100)
seed=0
init_state="GT_edit"
forward_model="numerical"
device="cuda"
wandb_mode="online"

# Fail early if any array is empty (no jobs would be submitted).
for arr in experiments data_indices eps_alpha_values euler_steps n_ens_values n_times_values; do
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
    tlimit="$time_limit"
    resources=("${gpu_resources[@]}")

    for eps_alpha in "${eps_alpha_values[@]}"; do
    for euler_step in "${euler_steps[@]}"; do
    for n_ens in "${n_ens_values[@]}"; do
    for n_times in "${n_times_values[@]}"; do

        exp_name="EnSF_${experiment}${ai_tag}_run_${data_index}"

        counter=$((counter + 1))
        echo "Submitting job ${exp_name}"


        sbatch "${SBATCH_MAIL_ARGS[@]}" -t "$tlimit" "${resources[@]}" --job-name="$exp_name" \
            assimilation/scripts/SQG/run_experiment_EnSF.bash \
            --exp_name "${exp_name}" \
            --experiment "${experiment}" \
            --data_index "${data_index}" \
            --assim_interval "${assim_interval}" \
            --eps_alpha "${eps_alpha}" \
            --euler_steps "${euler_step}" \
            --forward_model "${forward_model}" \
            --device "${device}" \
            --n_ens "${n_ens}" \
            --n_times "${n_times}" \
            --seed "${seed}" \
            --init_state "${init_state}" \
            --fixed_obs \
            --wandb_mode "${wandb_mode}"

    done; done; done; done
done
done

echo "Total EnSF jobs: ${counter} (dry_run=${DRY_RUN})"
