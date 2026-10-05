#!/bin/bash
#SBATCH -J b_loss_64
#SBATCH -t 3-00:00:00
#SBATCH --gpus=1 -C "thin"
# Job mail is set at submission time ($MAIL_TYPE / $MAIL_USER).

# Unconditional generative prior for SQG (base variant, 64 channels).

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
wandb online

# Follow $REPO_ROOT if exported, else resolve this checkout.
if [[ -n "${REPO_ROOT:-}" ]]; then
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT"
else
    source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"
fi
channels=64

name="b_loss_${channels}"
PYTHONPATH=$(pwd) python3 unconditional_generation/trainer.py \
    --wandb_run_name ${name} \
    --target_fn b_loss \
    --dataset SQG --variant base \
    --batch_size 100 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --num_epochs 50 \
    --channels ${channels} \
