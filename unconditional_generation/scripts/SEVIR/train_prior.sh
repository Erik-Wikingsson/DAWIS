#!/bin/bash
#SBATCH -J sevir_prior
#SBATCH -t 2-00:00:00
#SBATCH --gpus=1 -C "thin"
#SBATCH --output ./slurm_logs/%A_%x.out
#
# Unconditional generative prior for SEVIR on 128x128 VIL frames (shapes come
# from data/SEVIR/config.yaml). Env: CHANNELS, NAME, BATCH_SIZE, EPOCHS, N_WORKERS.

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
wandb online

# Follow $REPO_ROOT if exported, else resolve this checkout.
if [[ -n "${REPO_ROOT:-}" ]]; then
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT"
else
    source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"
fi
export PYTHONPATH=$(pwd)

channels="${CHANNELS:-64}"
name="${NAME:-sevir_prior_${channels}}"

python3 unconditional_generation/trainer.py \
    --dataset SEVIR \
    --variant lr_vil \
    --wandb_run_name "${name}" \
    --wandb_project ScoreDA_SEVIR \
    --target_fn b_loss \
    --channels "${channels}" \
    --batch_size "${BATCH_SIZE:-64}" \
    --num_epochs "${EPOCHS:-50}" \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --n_workers "${N_WORKERS:-16}"
