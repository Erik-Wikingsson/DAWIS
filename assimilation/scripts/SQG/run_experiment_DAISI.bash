#!/bin/bash
# Worker for the DAISI filter (submitted by run_all_DAISI.bash).
# Resources are passed on the sbatch command line by the launcher.

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
exp_name="DAISI_default"
experiment="sparse"
# Test trajectory index (0 = tuning trajectory, 1-10 = evaluation).
data_index=0
guide_method="MMPS"
guidance="1"
noise="invert"
eps="0.03"
invert_eps="0.03"
euler_steps=100
invert_steps=100
tmin="0.3"
# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
assim_interval=1
n_ens=20
n_times=100
seed=0
init_state="GT"
forward_model="numerical"
forward_precision="single"
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
        --guide_method) guide_method="$2"; shift ;;
        --guidance_strength) guidance="$2"; shift ;;
        --noise) noise="$2"; shift ;;
        --eps) eps="$2"; shift ;;
        --invert_eps) invert_eps="$2"; shift ;;
        --euler_steps) euler_steps="$2"; shift ;;
        --invert_steps) invert_steps="$2"; shift ;;
        --tmin) tmin="$2"; shift ;;
        --assim_interval) assim_interval="$2"; shift ;;
        --n_ens) n_ens="$2"; shift ;;
        --n_times) n_times="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --init_state) init_state="$2"; shift ;;
        --forward_model) forward_model="$2"; shift ;;
        --forward_precision) forward_precision="$2"; shift ;;
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

echo "Running experiment: ${exp_name} (method=DAISI, experiment=${experiment}, data_index=${data_index}, guide_method=${guide_method}, device=${device})"
[[ "$device" != "cpu" ]] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

python3 -m assimilation.assimilate \
    --exp_name "${exp_name}" \
    --experiment "${experiment}" \
    --data_index "${data_index}" \
    --method DAISI \
    --guide_method "${guide_method}" \
    --guidance_strength "${guidance}" \
    --noise "${noise}" \
    --eps "${eps}" \
    --invert_eps "${invert_eps}" \
    --euler_steps "${euler_steps}" \
    --invert_steps "${invert_steps}" \
    --tmin "${tmin}" \
    --init_state "${init_state}" \
    --forward_model "${forward_model}" \
    --forward_precision "${forward_precision}" \
    --assim_interval "${assim_interval}" \
    --n_ens "${n_ens}" \
    --n_times "${n_times}" \
    --seed "${seed}" \
    --device "${device}" \
    "${fixed_obs_args[@]}"

date
fix_results_permissions
