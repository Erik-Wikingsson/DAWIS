#!/bin/bash
# Worker for the FlowDAS baseline (submitted by run_all_FlowDAS.bash, which also sets resources).
# FlowDAS (Jia et al.) forecasts and assimilates in one guided sampling pass, so
# --forward_model none; an unobserved step is the same pass without guidance.

# Job mail is set at submission time ($MAIL_TYPE / $MAIL_USER).
#SBATCH --output ./slurm_logs/%A_%x.out

# Activate the conda env. Uses $REPO_ROOT (exported by the launcher), since sbatch runs a copy of this script.
_conda_env_sh="${REPO_ROOT:-$(dirname "${BASH_SOURCE[0]}")/../../..}/scripts/conda_env.sh"
if [[ ! -f "$_conda_env_sh" ]]; then
    echo "ERROR: no scripts/conda_env.sh at: $_conda_env_sh" >&2
    echo "  \$REPO_ROOT is unset. Submit through a run_all_*.bash launcher," >&2
    echo "  or with scripts/submit.sh, both of which set it." >&2
    exit 1
fi
source "$_conda_env_sh"
unset _conda_env_sh

set -euo pipefail

# Defaults (overridable from the launcher)
exp_name="FlowDAS_default"
experiment="sparse"
# Test trajectory index (0 = tuning trajectory, 1-10 = evaluation).
data_index=0
# The reference's EM_sample_steps, grad_scale, MC_times and sampling interval.
euler_steps=200
guidance_strength="0.1"
mc_times=1
flowdas_tmin="0.001"
flowdas_tmax="0.999"
# Fixed by the checkpoint (conditions on 6 past states).
window=6
start_time=6
# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
assim_interval=1
n_ens=20
n_times=100
seed=0
init_state="GT"
device="cuda"
fixed_obs=true
wandb_mode="online"

# Follow $REPO_ROOT if exported, else resolve this checkout.
if [[ -n "${REPO_ROOT:-}" ]]; then
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT"
else
    source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"
fi

# Shared results dir (checked before the run); defines fix_results_permissions.
source "$REPO_ROOT/scripts/results_dir.sh"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_name) exp_name="$2"; shift ;;
        --experiment) experiment="$2"; shift ;;
        --data_index) data_index="$2"; shift ;;
        --euler_steps) euler_steps="$2"; shift ;;
        --guidance_strength) guidance_strength="$2"; shift ;;
        --mc_times) mc_times="$2"; shift ;;
        --flowdas_tmin) flowdas_tmin="$2"; shift ;;
        --flowdas_tmax) flowdas_tmax="$2"; shift ;;
        --window) window="$2"; shift ;;
        --start_time) start_time="$2"; shift ;;
        --assim_interval) assim_interval="$2"; shift ;;
        --n_ens) n_ens="$2"; shift ;;
        --n_times) n_times="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --init_state) init_state="$2"; shift ;;
        --device) device="$2"; shift ;;
        --fixed_obs) fixed_obs=true ;;
        --no-fixed_obs) fixed_obs=false ;;
        --wandb_mode) wandb_mode="$2"; shift ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ "$wandb_mode" == "disabled" ]]; then
    export WANDB_MODE=disabled
    export WANDB_DISABLED=true
elif [[ "$wandb_mode" == "offline" ]]; then
    export WANDB_MODE=offline
    wandb offline
else
    export WANDB_MODE=online
    wandb online
fi

export PYTHONPATH=$(pwd)
[[ "$device" == "cpu" ]] && export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

fixed_obs_args=()
if [[ "$fixed_obs" == true ]]; then
    fixed_obs_args=(--fixed_obs)
fi

echo "Running experiment: ${exp_name} (method=FlowDAS, experiment=${experiment}, data_index=${data_index}, device=${device})"
[[ "$device" != "cpu" ]] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

python3 -m assimilation.assimilate \
    --exp_name "${exp_name}" \
    --experiment "${experiment}" \
    --data_index "${data_index}" \
    --method FlowDAS \
    --forward_model none \
    --assim_interval "${assim_interval}" \
    --window "${window}" \
    --start_time "${start_time}" \
    --euler_steps "${euler_steps}" \
    --guidance_strength "${guidance_strength}" \
    --mc_times "${mc_times}" \
    --flowdas_tmin "${flowdas_tmin}" \
    --flowdas_tmax "${flowdas_tmax}" \
    --init_state "${init_state}" \
    --n_ens "${n_ens}" \
    --n_times "${n_times}" \
    --seed "${seed}" \
    --device "${device}" \
    "${fixed_obs_args[@]}"

date
fix_results_permissions
