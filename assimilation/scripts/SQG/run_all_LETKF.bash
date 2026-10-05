#!/bin/bash
# LETKF filter baseline (Hunt, Kostelich & Szunyogh 2007) across the SQG
# experiments, forward-model backends and localization/inflation grid.
#   DRY_RUN=1 bash assimilation/scripts/SQG/run_all_LETKF.bash

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

time_limit="06:00:00"

cpu_resources=(-p berzelius-cpu -n1 -c16)
gpu_resources=(--gpus=1 -C thin)

# Experiments
experiments=("noisy" "sparse" "saturating" "multimodal")

# Backends: name | forward_model | device
backends=(
  "cpu|numerical|cpu"           # NumPy reference
#   "torchcpu|numerical_gpu|cpu"  # torch on CPU
#   "gpu|numerical_gpu|cuda"      # torch on GPU
)

# Hyperparameters
# Empty = use the tuned per-experiment default from EXPERIMENTS.
hcovlocal_scales=("")             
covinflate1_values=("")           

# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
# Override with ASSIM_INTERVAL=k; values != 1 are tagged into the run name.
assim_interval="${ASSIM_INTERVAL:-1}"
ai_tag=""
[[ "$assim_interval" != "1" ]] && ai_tag="_ai${assim_interval}"

n_ens_values=(20)
n_times_values=(100)
seed=0
wandb_mode="online"

# Initial ensemble: truth + N(0, init_std) per member (not a zero-spread ensemble).
# GT_edit needs a generative prior, which LETKF does not have. The paper runs used GT.
init_state="GT"

# Test trajectory indices (0 = tuning trajectory, 1-10 = evaluation).
data_indices=(0 10)

# Fail early if any array is empty (no jobs would be submitted).
for arr in experiments backends hcovlocal_scales covinflate1_values n_ens_values n_times_values data_indices; do
    declare -n _ref="$arr"
    if [[ ${#_ref[@]} -eq 0 ]]; then
        echo "ERROR: array '${arr}' is empty -- no jobs would be submitted. Use (\"\") for 'experiment default'." >&2
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

        for backend in "${backends[@]}"; do
            IFS='|' read -r tag forward_model device <<< "$backend"

            if [[ "$device" == "cpu" ]]; then
                resources=("${cpu_resources[@]}")
            else
                resources=("${gpu_resources[@]}")
            fi

            for n_ens in "${n_ens_values[@]}"; do
                for n_times in "${n_times_values[@]}"; do
                    for hcovlocal_scale in "${hcovlocal_scales[@]}"; do
                        for covinflate1 in "${covinflate1_values[@]}"; do
                            # Sweep the tuning grid on the reference backend only.
                            if [[ "$tag" != "cpu" && ( -n "$hcovlocal_scale" || -n "$covinflate1" ) ]]; then
                                continue
                            fi
                            # One knob at a time.
                            if [[ -n "$hcovlocal_scale" && -n "$covinflate1" ]]; then
                                continue
                            fi

                            exp_name="LETKF_${experiment}${ai_tag}_run_${data_index}"
                    
                            # Only passed when overridden by the sweep.
                            tuning_args=()
                            [[ -n "$hcovlocal_scale" ]] && tuning_args+=(--hcovlocal_scale "${hcovlocal_scale}")
                            [[ -n "$covinflate1" ]] && tuning_args+=(--covinflate1 "${covinflate1}")

                            counter=$((counter + 1))
                            echo "Submitting job ${exp_name}"


                            sbatch "${SBATCH_MAIL_ARGS[@]}" -t "$tlimit" "${resources[@]}" --job-name="$exp_name" \
                                assimilation/scripts/SQG/run_experiment_LETKF.bash \
                                --exp_name "${exp_name}" \
                                --experiment "${experiment}" \
                                --data_index "${data_index}" \
                                --assim_interval "${assim_interval}" \
                                --forward_model "${forward_model}" \
                                --device "${device}" \
                                --n_ens "${n_ens}" \
                                --n_times "${n_times}" \
                                --seed "${seed}" \
                                --init_state "${init_state}" \
                                --fixed_obs \
                                --wandb_mode "${wandb_mode}" \
                                "${tuning_args[@]}"
                        done
                    done
                done
            done
        done
    done
done

echo "Total LETKF jobs: ${counter} (dry_run=${DRY_RUN})"
