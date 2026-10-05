#!/bin/bash
# LETKF baseline on SEVIR at the published settings (no sweep; only data_indices changes):
#     hcovlocal_scale 9000 m (3 grid cells)   covinflate1 0   covinflate2 -1
# set in _common.sh. To re-tune, override from the environment, e.g.
#   for hc in 4500 9000 18000 36000; do HCOVLOCAL_SCALE=$hc bash .../run_all_LETKF.sh; done
#
# PROPAGATOR=flowdas (default) is stochastic and supplies the spread, hence no
# inflation. PROPAGATOR=unet adds no spread and is refused with the GT start.
# Keep n_ens at 20: small ensembles diverge under this localization.
#
#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_LETKF.sh   # print, don't submit

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

# LOG_MEDIA=true logs wandb videos.
LOG_MEDIA="${LOG_MEDIA:-false}"

n_ens_values=(20)

# flowdas | unet | fmw; validated by the worker.
PROPAGATOR="${PROPAGATOR:-flowdas}"
fwd_envs=(FORWARD_MODEL="$PROPAGATOR")

require_nonempty n_ens_values experiments data_indices

apply_data_indices
for data_index in "${data_indices[@]}"; do
for experiment in "${experiments[@]}"; do
for n_ens in "${n_ens_values[@]}"; do

    name="LETKF_SEVIR_run_${data_index}"

    sevir_submit "$name" run_LETKF.sh -t "$time_limit" -C "$NODE_TYPE" -- \
        ASSIM_NORM="$ASSIM_NORM" FORWARD_NORM="$FORWARD_NORM" \
        OUTPUT_NORM="$OUTPUT_NORM" \
        EXPERIMENT="$experiment" DATA_INDEX="$data_index" N_ENS="$n_ens" \
        LOG_MEDIA="$LOG_MEDIA" "${fwd_envs[@]}"

done; done; done

sevir_summary LETKF
