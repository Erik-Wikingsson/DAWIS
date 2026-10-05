#!/bin/bash
# EnSF on SEVIR at the published settings (no sweep; only data_indices changes):
#     euler_steps 1000   eps_alpha 0.05   n_ens 20   forward_model FlowDAS
#     init_std 0, init_state GT           guidance_strength 0.01
#     ensf_spread = first-forecast spread (ENSF_SPREAD_COUNTS=12.75 = published value)
# GUIDANCE_STRENGTH bounds the stability of the explicit Euler reverse SDE;
# raising it much above 0.01 gives NaNs.
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_EnSF.sh   # print, don't submit

source "$(dirname "${BASH_SOURCE[0]}")/_launcher.sh"

NODE_TYPE="${NODE_TYPE:-thin}"

# Wall-clock limit (TIME_LIMIT overrides).
time_limit="${TIME_LIMIT:-01:00:00}"

# Normalization (see _common.sh). ASSIM_NORM is free (vil/255 here);
# FORWARD_NORM must match the propagator.
ASSIM_NORM="${ASSIM_NORM:-01}"
FORWARD_NORM="${FORWARD_NORM:-flowdas}"
OUTPUT_NORM="${OUTPUT_NORM:-physical}"

# The DAISI paper's observation network (defined in _common.sh).
experiments=("sevir")

# Trajectories: 0 = tuning, 1-10 = evaluation.
# data_indices=(0)                      # the tuning trajectory alone
# data_indices=(0 1 2 3 4 5 6 7 8 9 10) # everything
data_indices=(1 2 3 4 5 6 7 8 9 10)     # the held-out comparison

# One job per trajectory: media on.
LOG_MEDIA="${LOG_MEDIA:-true}"

n_ens_values=(20)

require_nonempty n_ens_values experiments data_indices

apply_data_indices
for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
for n_ens in "${n_ens_values[@]}"; do

    name="EnSF_SEVIR_run_${data_index}"

    sevir_submit "$name" run_EnSF.sh -t "$time_limit" -C "$NODE_TYPE" -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" \
        EXPERIMENT="$experiment" DATA_INDEX="$data_index" N_ENS="$n_ens" \
        LOG_MEDIA="$LOG_MEDIA" \
        EULER_STEPS="${EULER_STEPS:-1000}" \
        EPS_ALPHA="${EPS_ALPHA:-0.05}" \
        GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-0.01}" \
        ENSF_SPREAD_COUNTS="${ENSF_SPREAD_COUNTS:-}"

done; done; done

sevir_summary EnSF
