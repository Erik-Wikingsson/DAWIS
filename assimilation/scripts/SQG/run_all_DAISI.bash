#!/bin/bash
# DAISI across the SQG experiments at the tuned guidance settings, with the same
# trajectories and ensemble settings as run_all_LETKF.bash.
#   DRY_RUN=1 bash assimilation/scripts/SQG/run_all_DAISI.bash [--sweep true|false]

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
experiments=("noisy")

# Test trajectory indices (0 = tuning trajectory, 1-10 = evaluation).
data_indices=(0 10)

# Hyperparameters
# guide_method, tmin and guidance_strength are set per experiment in the loop.
noise_values=("invert")
eps_values=(0.03)
euler_steps=(100)
invert_steps=(100)

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

# --sweep false reproduces the paper runs: all four experiments at the tuned values.
if [[ "$sweep" == "false" ]]; then
    experiments=("noisy" "sparse" "saturating" "multimodal")
fi

# Fail early if any array is empty (no jobs would be submitted).
for arr in experiments data_indices noise_values eps_values \
           euler_steps invert_steps n_ens_values n_times_values; do
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
    # Tuned settings from the DAISI paper (arXiv:2512.00252), Table 8. Suffixed
    # variants (-5, ...) inherit the row of their base experiment.
    case $experiment in
        noisy|noisy-5)
            guide_methods=("DPS_scale");  tmins=(0.4); guidance_strengths=(10);;
        noisy-12h)
            guide_methods=("MMPS"); tmins=(0.3); guidance_strengths=(1);;
        sparse*)
            guide_methods=("MMPS"); tmins=(0.3); guidance_strengths=(1);;
        multimodal*)
            guide_methods=("MMPS"); tmins=(0.3); guidance_strengths=(1);;
        saturating*|A3)
            guide_methods=("MMPS"); tmins=(0.3); guidance_strengths=(20);;
        *)
            echo "Unknown experiment=${experiment}: add a row to the case block" \
                 "above with its tuned guide_method / tmin / guidance_strength." >&2
            exit 1;;
    esac

    tlimit="$time_limit"
    resources=("${gpu_resources[@]}")

    for guide_method in "${guide_methods[@]}"; do
    for noise in "${noise_values[@]}"; do
    for eps in "${eps_values[@]}"; do
    for tmin in "${tmins[@]}"; do
        # tmin only has a meaning when the prior is renoised by inversion.
        if [[ "$noise" == "None" && $(echo "$tmin > 0" | bc) -eq 1 ]]; then
            echo "Skipping combination: noise=None with tmin=$tmin"
            continue
        fi
    for euler_step in "${euler_steps[@]}"; do
    for invert_step in "${invert_steps[@]}"; do
    for guidance in "${guidance_strengths[@]}"; do
    for n_ens in "${n_ens_values[@]}"; do
    for n_times in "${n_times_values[@]}"; do

        exp_name="DAISI_${experiment}${ai_tag}_run_${data_index}"

        counter=$((counter + 1))
        echo "Submitting job ${exp_name}"


        sbatch "${SBATCH_MAIL_ARGS[@]}" -t "$tlimit" "${resources[@]}" --job-name="$exp_name" \
            assimilation/scripts/SQG/run_experiment_DAISI.bash \
            --exp_name "${exp_name}" \
            --experiment "${experiment}" \
            --data_index "${data_index}" \
            --assim_interval "${assim_interval}" \
            --guide_method "${guide_method}" \
            --guidance_strength "${guidance}" \
            --noise "${noise}" \
            --eps "${eps}" \
            --invert_eps "${eps}" \
            --euler_steps "${euler_step}" \
            --invert_steps "${invert_step}" \
            --tmin "${tmin}" \
            --forward_model "${forward_model}" \
            --device "${device}" \
            --n_ens "${n_ens}" \
            --n_times "${n_times}" \
            --seed "${seed}" \
            --init_state "${init_state}" \
            --fixed_obs \
            --wandb_mode "${wandb_mode}"

    done; done; done; done; done; done; done; done; done
done
done

echo "Total DAISI jobs: ${counter} (dry_run=${DRY_RUN})"
