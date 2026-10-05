#!/bin/bash
# Submits one FMW training job (train_one_fmw.sh) per number of initial states.

set -euo pipefail

# Sets cwd, $REPO_ROOT and $PYTHONPATH.
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"

# Job-mail options from $MAIL_TYPE / $MAIL_USER (must follow repo_root.sh).
source "$REPO_ROOT/scripts/sbatch_opts.sh"
cd "$REPO_ROOT/forecasting"

time_limit="20:00:00"

init_states_values=(1 2 3 4 5 6) 

for init_states in "${init_states_values[@]}"; do
    wandb_run_name="eta_channel_init_${init_states}"
    hidden_dim=$((32 * (init_states + 1))) # Scale width with the number of input states.
    channel_mult_emb=4
    channel_mult_noise=2
    alpha_beta_mult=1

    case $init_states in
        1) batch_size=64;;
        2) batch_size=32 ;;
        3) batch_size=32 ;;
        4) batch_size=16 ;;
        5) batch_size=16 ;;
        6) batch_size=16 ;;
    esac
    echo "Starting job with name ${wandb_run_name}"
    sbatch "${SBATCH_MAIL_ARGS[@]}" --time="$time_limit" --job-name="$wandb_run_name" training_scripts/SQG/train_one_fmw.sh \
        --wandb_run_name "$wandb_run_name" \
        --init_states "$init_states" \
        --batch_size "$batch_size" \
        --hidden_dim "$hidden_dim" \
        --channel_mult_emb "$channel_mult_emb" \
        --channel_mult_noise "$channel_mult_noise" \
        --alpha_beta_mult "$alpha_beta_mult"
done