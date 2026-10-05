#!/bin/bash
# FlowDAS on SEVIR at the published settings (no sweep; only data_indices changes):
#     euler_steps 500   guidance_strength 0.1   window 6
#     init_std 0        n_ens 20
# guidance_strength 0.1 is the reference implementation's `grad_scale`. Report
# `test` only (the checkpoint's train split overlaps val).
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_FlowDAS.sh   # print, don't submit

source "$(dirname "${BASH_SOURCE[0]}")/_launcher.sh"

NODE_TYPE="${NODE_TYPE:-fat}"

# Wall-clock limit (TIME_LIMIT overrides).
time_limit="${TIME_LIMIT:-02:00:00}"

# Normalization (see _common.sh): the forecaster bridges from ASSIM_NORM into
# the `flowdas` checkpoint, so the run assimilates in `01`.
ASSIM_NORM="${ASSIM_NORM:-01}"
FORWARD_NORM="${FORWARD_NORM:-flowdas}"
OUTPUT_NORM="${OUTPUT_NORM:-physical}"

# The DAISI paper's observation network (defined in _common.sh).
experiments=("sevir")

# Trajectories: 0 = tuning, 1-10 = evaluation.
# data_indices=(0)                      # the tuning trajectory alone
# data_indices=(0 1 2 3 4 5 6 7 8 9 10) # everything
data_indices=(1 2 3 4 5 6 7 8 9 10)     # the held-out comparison

# LOG_MEDIA=true logs wandb videos.
LOG_MEDIA="${LOG_MEDIA:-false}"

n_ens_values=(20)

require_nonempty n_ens_values experiments data_indices

apply_data_indices
for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
for n_ens in "${n_ens_values[@]}"; do

    name="FlowDAS_SEVIR_run_${data_index}"

    sevir_submit "$name" run_FlowDAS.sh -t "$time_limit" -C "$NODE_TYPE" -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" \
        EXPERIMENT="$experiment" DATA_INDEX="$data_index" N_ENS="$n_ens" \
        LOG_MEDIA="$LOG_MEDIA" \
        EULER_STEPS="${EULER_STEPS:-500}" \
        GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-0.1}" \
        MC_TIMES="${MC_TIMES:-25}" \
        INIT_STATE="${INIT_STATE:-GT}"

done; done; done

sevir_summary FlowDAS
