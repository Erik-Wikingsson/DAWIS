#!/bin/bash
# DAISI on SEVIR at the published settings (no sweep; only data_indices changes):
#     guide_method MMPS   guidance_strength 1   tmin 0.3
#     euler_steps 100     invert_steps 100      noise invert
#     eps 0.03            invert_eps 0.03       init_std 0
#     forward_model FlowDAS                     n_ens 20
# Uses the 128x128 prior (daisi_sevir_128.pth). Override a setting through the
# environment, e.g. GUIDANCE_STRENGTH=3.
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_DAISI.sh   # print, don't submit

source "$(dirname "${BASH_SOURCE[0]}")/_launcher.sh"

NODE_TYPE="${NODE_TYPE:-thin}"

# Wall-clock limit (TIME_LIMIT overrides).
time_limit="${TIME_LIMIT:-01:30:00}"

# Normalization (see _common.sh): the prior was trained on vil/255 (`01`), the
# FlowDAS propagator on `flowdas`. ASSIM_NORM=flowdas here silently degrades results.
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

    name="DAISI_SEVIR_run_${data_index}"

    sevir_submit "$name" run_DAISI.sh -t "$time_limit" -C "$NODE_TYPE" -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" \
        EXPERIMENT="$experiment" DATA_INDEX="$data_index" N_ENS="$n_ens" \
        LOG_MEDIA="$LOG_MEDIA" \
        FORWARD_MODEL="${FORWARD_MODEL:-flowdas}" \
        GUIDE_METHOD="${GUIDE_METHOD:-MMPS}" \
        GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-1}" \
        TMIN="${TMIN:-0.3}" \
        EULER_STEPS="${EULER_STEPS:-100}" \
        INVERT_STEPS="${INVERT_STEPS:-100}" \
        NOISE="${NOISE:-invert}" \
        EPS="${EPS:-0.03}" INVERT_EPS="${INVERT_EPS:-0.03}" \
        INIT_STD="${INIT_STD:-0}" \
        INIT_STATE="${INIT_STATE:-GT}"

done; done; done

sevir_summary DAISI
