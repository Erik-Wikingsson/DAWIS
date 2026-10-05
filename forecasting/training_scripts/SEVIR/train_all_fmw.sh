#!/bin/bash

set -euo pipefail

# Sweep FMW over init_states on SEVIR (same hyperparameters as the SQG sweep).
# Usage: bash forecasting/training_scripts/SEVIR/train_all_fmw.sh
# Env: TIME_LIMIT, VARIANT, NORM, LIMIT_VAL_BATCHES, N_WORKERS.

# Sets cwd, $REPO_ROOT and $PYTHONPATH.
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"

# Job-mail options from $MAIL_TYPE / $MAIL_USER (must follow repo_root.sh).
source "$REPO_ROOT/scripts/sbatch_opts.sh"
cd "$REPO_ROOT/forecasting"

time_limit="${TIME_LIMIT:-3-00:00:00}"

# lr_vil (1 channel, 128x128) or lr_vil_64 (64x64). The batch sizes below are
# sized for lr_vil; lr_vil_64 is ~4x cheaper and allows larger batches.
variant="${VARIANT:-lr_vil}"

# Model-space normalization of VIL:
#   01        vil/255 (data much narrower than the N(0,1) noise endpoint; FMW can stall)
#   flowdas   (vil/255 - 0.5)/0.1, the FlowDAS/DAISI transform
#   standard  (vil - 33.44)/47.54, zero-mean/unit-std
# Non-01 modes are appended to the run name.
norm="${NORM:-flowdas}" # default to FlowDAS
case "$norm" in
    ""|01|flowdas|standard) ;;
    *)
        echo "ERROR: NORM must be one of 01, flowdas, standard (got '$norm')" >&2
        exit 1
        ;;
esac

# Per-rank val batches per validation run (1.0 = full split).
limit_val_batches="${LIMIT_VAL_BATCHES:-10}"

# DataLoader workers per rank (0 = load in the main process).
n_workers="${N_WORKERS:-16}"

norm_args=()
norm_tag=""
if [[ -n "$norm" ]]; then
    norm_args=(--sevir_norm "$norm")
    if [[ "$norm" != "01" ]]; then
        norm_tag="_${norm}"
    fi
fi

# Absolute log dir, so logs land here regardless of the submitting cwd.
log_dir="$REPO_ROOT/forecasting/training_scripts/SEVIR/slurm_logs"
mkdir -p "$log_dir"

init_states_values=(2 4 6)

for init_states in "${init_states_values[@]}"; do
    # Train FMW model for the given number of initial states.
    wandb_run_name="SEVIR_FLOWDAS_SPLIT_${variant}_eta_channel_init_${init_states}${norm_tag}"
    hidden_dim=$((32 * (init_states + 1))) # Scale hidden dimension with number of states, as a simple heuristic to keep model capacity proportional.
    channel_mult_emb=4
    channel_mult_noise=2
    # Must equal 2/D for eta01_channel (D = variables per step): SEVIR D=1 -> 2.
    alpha_beta_mult=2

    # Per-rank batch size that fits an 80 GB A100 at lr_vil, with headroom for DDP.
    case $init_states in
        1) batch_size=32 ;;
        2) batch_size=16 ;;
        3) batch_size=16 ;;
        4) batch_size=12 ;;
        5) batch_size=8 ;;
        6) batch_size=8 ;;
    esac
    echo "Starting job with name ${wandb_run_name}"
    sbatch "${SBATCH_MAIL_ARGS[@]}" --time="$time_limit" --job-name="$wandb_run_name" \
        --output "$log_dir/%A_%x.out" \
        training_scripts/SEVIR/train_one_fmw.sh \
        --wandb_run_name "$wandb_run_name" \
        --variant "$variant" \
        --init_states "$init_states" \
        --batch_size "$batch_size" \
        --hidden_dim "$hidden_dim" \
        --channel_mult_emb "$channel_mult_emb" \
        --channel_mult_noise "$channel_mult_noise" \
        --alpha_beta_mult "$alpha_beta_mult" \
        --limit_val_batches "$limit_val_batches" \
        --n_workers "$n_workers" \
        "${norm_args[@]}"
done
