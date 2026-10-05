#!/bin/bash
# Worker for Gibbs block resampling (assimilation/smoothing.py); submitted by run_all_Gibbs.bash.
#SBATCH -t 10:00:00
#SBATCH --gpus=1
#SBATCH -C "thin"
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

# Follow $REPO_ROOT if exported, else resolve this checkout.
if [[ -n "${REPO_ROOT:-}" ]]; then
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT"
else
    source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"
fi

# Shared results dir (checked before the run); defines fix_results_permissions.
source "$REPO_ROOT/scripts/results_dir.sh"

# Defaults

exp_name="SDA_default"

method="SDA"
experiment="saturating"
# Test trajectory index (0 = tuning trajectory, 1-10 = evaluation).
data_index=0

model_path=""
hidden_dim=224
channel_mult_emb=4
channel_mult_noise=2

init_states=4

# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
# Unobserved steps get an empty obs mask, dropping their term from the cost.
assim_interval=1
n_times=100
n_ens=20
window_batch_size=5
# Ensemble members per GPU call; empty = whole ensemble at once.
batch_size=""

sampler_steps=100
# --eps and --sampler_eps both set this (the guided forward pass eps).
sampler_eps=0.03
invert_eps=0.03
trajectory_var='x_smooth'

guide_method="MMPS"
guidance_strength=1.0
noise="none"

# Guidance towards the initial condition: none, or init/all (extra fully observed
# slot with error std x0_sigma; init_std is the noise on that x0 observation).
guide_first="none"
x0_sigma=3.0
init_std=1000

corrections=1
tau=0.5

gibbs_sweeps=1
block_mode="random"
block_direction="forward"
block_stride=1
random_block_count=50
# Conditioning steps frozen on each side of a block. 1 is the Markovian case.
gibbs_num_endpoints=1

trajectory_path="None"

# Paper-run method name to start from (e.g. DAWISnumerical), used when trajectory_path is None.
initial_trajectory="None"

# Per-sweep logging: none (final only), metrics (one wandb run per sweep), full (also x_gibbs_k).
sweep_logging="none"

fm_loss="eta01_channel"
resample_filter="1,3,3,1"
channel_mult="2,2,2"
attn_resolutions="32"
schedule="linear_scalar"
noise_embedding="positional"

wandb_mode="online"

# Parse arguments

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_name) exp_name="$2"; shift ;;
        --method) method="$2"; shift ;;
        --experiment) experiment="$2"; shift ;;
        --data_index) data_index="$2"; shift ;;

        --model_path) model_path="$2"; shift ;;
        --hidden_dim) hidden_dim="$2"; shift ;;
        --channel_mult_emb) channel_mult_emb="$2"; shift ;;
        --channel_mult_noise) channel_mult_noise="$2"; shift ;;

        --init_states) init_states="$2"; shift ;;

        --assim_interval) assim_interval="$2"; shift ;;
        --n_times) n_times="$2"; shift ;;
        --n_ens) n_ens="$2"; shift ;;
        --window_batch_size) window_batch_size="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;

        --sampler_steps) sampler_steps="$2"; shift ;;
        --sampler_eps|--eps) sampler_eps="$2"; shift ;;
        --invert_eps) invert_eps="$2"; shift ;;

        --guide_method) guide_method="$2"; shift ;;
        --guide_first) guide_first="$2"; shift ;;
        --x0_sigma) x0_sigma="$2"; shift ;;
        --init_std) init_std="$2"; shift ;;
        --guidance_strength) guidance_strength="$2"; shift ;;
        --noise) noise="$2"; shift ;;

        --corrections) corrections="$2"; shift ;;
        --tau) tau="$2"; shift ;;

        --gibbs_sweeps) gibbs_sweeps="$2"; shift ;;
        --block_mode) block_mode="$2"; shift ;;
        --block_direction) block_direction="$2"; shift ;;
        --block_stride) block_stride="$2"; shift ;;
        --random_block_count) random_block_count="$2"; shift ;;
        --gibbs_endpoint_tmin) gibbs_endpoint_tmin="$2"; shift ;;
        --gibbs_midpoint_tmin) gibbs_midpoint_tmin="$2"; shift ;;
        --gibbs_num_endpoints) gibbs_num_endpoints="$2"; shift ;;

        --sweep_logging) sweep_logging="$2"; shift ;;

        --trajectory_path) trajectory_path="$2"; shift ;;
        --initial_trajectory) initial_trajectory="$2"; shift ;;
        --trajectory_var) trajectory_var="$2"; shift ;;

        --fm_loss) fm_loss="$2"; shift ;;
        --resample_filter) resample_filter="$2"; shift ;;
        --channel_mult) channel_mult="$2"; shift ;;
        --attn_resolutions) attn_resolutions="$2"; shift ;;
        --schedule) schedule="$2"; shift ;;
        --noise_embedding) noise_embedding="$2"; shift ;;

        --wandb_mode) wandb_mode="$2"; shift ;;

        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

# WandB

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

case "${sweep_logging}" in
    none) sweep_flag="" ;;
    metrics) sweep_flag="--sweep_metrics" ;;
    full) sweep_flag="--save_sweeps" ;;
    *) echo "Unknown --sweep_logging '${sweep_logging}' (none|metrics|full)"; exit 1 ;;
esac

batch_args=()
[[ -n "${batch_size}" ]] && batch_args=(--batch_size "${batch_size}")

echo "Running ${exp_name}"

python3 -m assimilation.smoothing \
    --exp_name "${exp_name}" \
    --data_path "${SQG_ROOT:?set SQG_ROOT in the repo .env}/test/64_3h/" \
    --assim_interval "${assim_interval}" \
    --n_times "${n_times}" \
    --nx 64 \
    --n_ens "${n_ens}" \
    --window_batch_size "${window_batch_size}" \
    --experiment "${experiment}" \
    --data_index "${data_index}" \
    --device cuda \
    --model_path "${model_path}" \
    --euler_steps "${sampler_steps}" \
    --guide_method "${guide_method}" \
    --guide_first "${guide_first}" \
    --x0_sigma "${x0_sigma}" \
    --init_std "${init_std}" \
    --guidance_strength "${guidance_strength}" \
    --noise "${noise}" \
    --method "${method}" \
    --forward_model none \
    --fm_loss "${fm_loss}" \
    --resample_filter "${resample_filter}" \
    --channel_mult "${channel_mult}" \
    --attn_resolutions "${attn_resolutions}" \
    --schedule "${schedule}" \
    --invert_eps "${invert_eps}" \
    --hidden_dim "${hidden_dim}" \
    --noise_embedding "${noise_embedding}" \
    --channel_mult_emb "${channel_mult_emb}" \
    --channel_mult_noise "${channel_mult_noise}" \
    --alpha_beta_mult 1 \
    --init_states "${init_states}" \
    --corrections "${corrections}" \
    --tau "${tau}" \
    --gibbs_sweeps "${gibbs_sweeps}" \
    --block_mode "${block_mode}" \
    --block_direction "${block_direction}" \
    --block_stride "${block_stride}" \
    --trajectory_path "${trajectory_path}" \
    --initial_trajectory "${initial_trajectory}" \
    --trajectory_var "${trajectory_var}" \
    --random_block_count "${random_block_count}" \
    --gibbs_endpoint_tmin "${gibbs_endpoint_tmin}" \
    --gibbs_midpoint_tmin "${gibbs_midpoint_tmin}" \
    --gibbs_num_endpoints "${gibbs_num_endpoints}" \
    --eps "${sampler_eps}" \
    "${batch_args[@]}" \
    --random_block_replace \
    ${sweep_flag} \
    --no_media_wandb

fix_results_permissions