#!/bin/bash
#SBATCH --gpus=4
#SBATCH -C "fat"
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

# Follow $REPO_ROOT if exported, else resolve this checkout.
if [[ -n "${REPO_ROOT:-}" ]]; then
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT"
else
    source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"
fi

export WANDB_MODE=online
wandb online

# Parse parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --wandb_run_name) wandb_run_name="$2"; shift ;;
        --init_states) init_states="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --hidden_dim) hidden_dim="$2"; shift ;;
        --channel_mult_emb) channel_mult_emb="$2"; shift ;;
        --channel_mult_noise) channel_mult_noise="$2"; shift ;;
        --alpha_beta_mult) alpha_beta_mult="$2"; shift ;;

        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

PYTHONPATH=$(pwd) python3 forecasting/trainer.py \
    --model FMW \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    --wandb_project DAWIS \
    --wandb_run_name "${wandb_run_name}" \
    --fm_loss eta01_channel \
    --n_workers 8 \
    --val_steps_to_log 1 \
    --val_interval 5 \
    --ar_steps_eval 1 \
    --init_states "${init_states}" \
    --step_length 3 \
    --epochs 50 \
    --batch_size "${batch_size}" \
    --resample_filter 1,3,3,1 \
    --channel_mult 2,2,2 \
    --attn_resolutions 32 \
    --schedule linear_scalar \
    --sampler stochastic \
    --sampler_eps 0.03 \
    --hidden_dim "${hidden_dim}" \
    --noise_embedding positional \
    --channel_mult_emb "${channel_mult_emb}" \
    --channel_mult_noise "${channel_mult_noise}" \
    --alpha_beta_mult "${alpha_beta_mult}" \
