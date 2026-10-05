#!/bin/bash
# Shared settings for the SEVIR assimilation scripts. Sourced, not executed.
#
# SEVIR-LR: 1 channel (VIL, 0-255), 128x128, 25 frames per event at 10 min.
# SEVIR has no numerical forward model: ensemble methods need a learned
# propagator (FORWARD_MODEL, below).

# Conda activation must happen before `set -u`. $REPO_ROOT is set by the
# launcher (sbatch copies this script, so a relative path is only a fallback).
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

# cwd, $REPO_ROOT and $PYTHONPATH (see scripts/repo_root.sh).
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"

# $RESULTS_ROOT and fix_results_permissions (see scripts/results_dir.sh).
source "$REPO_ROOT/scripts/results_dir.sh"

DATASET="SEVIR"
VARIANT="${VARIANT:-lr_vil}"
SPLIT="${SPLIT:-test}"

# Normalization (see README.md, "Three normalizations"):
#   ASSIM_NORM    units the run works in (state, obs, --obs_sigma, --init_std);
#                 must match the assimilation network's training normalization.
#   FORWARD_NORM  units the forward-propagator checkpoint was trained in.
#   OUTPUT_NORM   units results are reported in: physical (VIL counts), assim,
#                 or a norm-mode name (01 = vil/255, as in the FlowDAS/DAISI papers).
# All SEVIR checkpoints here use flowdas: z = (vil - 127.5) / 25.5.
# SEVIR_NORM is read by data/SEVIR/config.yaml and must be set before import.
ASSIM_NORM="${ASSIM_NORM:-${SEVIR_NORM:-flowdas}}"
FORWARD_NORM="${FORWARD_NORM:-flowdas}"
SEVIR_NORM="$ASSIM_NORM"
export SEVIR_NORM

OUTPUT_NORM="${OUTPUT_NORM:-physical}"

for _n in "$ASSIM_NORM" "$FORWARD_NORM"; do
    case "$_n" in
        01|flowdas|standard) ;;
        *)
            echo "ERROR: unknown normalization '${_n}'. Known: 01, flowdas, standard." >&2
            echo "  Set ASSIM_NORM (the units the run works in) and FORWARD_NORM" >&2
            echo "  (the units the propagator checkpoint was trained in)." >&2
            exit 1 ;;
    esac
done
case "$OUTPUT_NORM" in
    physical|assim|01|flowdas|standard) ;;
    *)
        echo "ERROR: unknown OUTPUT_NORM '${OUTPUT_NORM}'." >&2
        echo "  Known: physical (VIL counts), assim (the run's own units)," >&2
        echo "  or one of 01, flowdas, standard." >&2
        exit 1 ;;
esac
unset _n

# The run works in normalized units (`assim_space: normalized`); results in
# assim units multiply by the norm std to get VIL counts.
case "$ASSIM_NORM" in
    01)       SEVIR_NORM_STD=255.0 ;;
    flowdas)  SEVIR_NORM_STD=25.5 ;;
    standard) SEVIR_NORM_STD=47.54 ;;
esac

# Magnitude in VIL counts -> run units (scale only, no offset).
_from_counts () { awk -v c="$1" -v s="$SEVIR_NORM_STD" 'BEGIN{printf "%.6g", c / s}'; }

# Field std (vil/255: 0.158), used to state obs noise as a fraction of it.
SEVIR_FIELD_STD_COUNTS=40.3
SEVIR_FIELD_STD="$(_from_counts "$SEVIR_FIELD_STD_COUNTS")"

# SEVIR experiment (the DAISI paper's observation network; see README.md).
#   hcovlocal is in metres: 3 grid cells * 3 km = 9000 m.
#   obs_sigma is stated in VIL counts (0.001 * 255) and converted.
#   covinflate2 -1 = relaxation to prior spread; covinflate1 0 = no inflation.
# The `sevir` preset in assimilation/experiments.py refuses contradicting
# OBS_*/INIT_* values. HCOVLOCAL_SCALE/COVINFLATE* are tuning and may be overridden.
EXPERIMENT="${EXPERIMENT:-sevir}"
case "$EXPERIMENT" in
    sevir)
        _obs_fn=linear _obs_prob=0.1
        _obs_sigma="$(_from_counts 0.255)" _init_std=0
        _hcovlocal=9000.0 _covinflate1=0 _covinflate2=-1 ;;
    *)
        echo "ERROR: unknown EXPERIMENT '${EXPERIMENT}'. The only one is 'sevir'." >&2
        echo "  Add a row to the case block in assimilation/scripts/SEVIR/_common.sh" >&2
        echo "  if you genuinely need a second observation network -- but a number" >&2
        echo "  from it is not comparable with any published SEVIR result." >&2
        exit 1 ;;
esac

# Observation network. OBS_SIGMA/INIT_STD are in run units;
# OBS_SIGMA_COUNTS/INIT_STD_COUNTS are in VIL counts and converted here.
OBS_FN="${OBS_FN:-$_obs_fn}"
OBS_PROB="${OBS_PROB:-$_obs_prob}"
if [[ -n "${OBS_SIGMA_COUNTS:-}" ]]; then
    OBS_SIGMA="${OBS_SIGMA:-$(_from_counts "$OBS_SIGMA_COUNTS")}"
fi
OBS_SIGMA="${OBS_SIGMA:-$_obs_sigma}"
# Initial ensemble: must be 0 / GT under the `sevir` preset.
if [[ -n "${INIT_STD_COUNTS:-}" ]]; then
    INIT_STD="${INIT_STD:-$(_from_counts "$INIT_STD_COUNTS")}"
fi
INIT_STD="${INIT_STD:-$_init_std}"
INIT_STATE="${INIT_STATE:-GT}"
# Localization (metres) and inflation for the ensemble filters.
HCOVLOCAL_SCALE="${HCOVLOCAL_SCALE:-$_hcovlocal}"
COVINFLATE1="${COVINFLATE1:-$_covinflate1}"
COVINFLATE2="${COVINFLATE2:-$_covinflate2}"
unset _obs_fn _obs_prob _obs_sigma _init_std _hcovlocal _covinflate1 _covinflate2

# Assimilate every ASSIM_INTERVAL steps (1 = every step, -1 = never); the
# ensemble free-runs in between. See assimilation/obs_schedule.py.
ASSIM_INTERVAL="${ASSIM_INTERVAL:-1}"

N_ENS="${N_ENS:-20}"
# Members per GPU call; only changes peak memory. Valid for the score-based
# methods with independent members (LETKF forces batch_size = n_ens).
BATCH_SIZE="${BATCH_SIZE:-$N_ENS}"
# START_TIME + N_TIMES <= 25 frames; START_TIME >= the conditioning window.
N_TIMES="${N_TIMES:-19}"
START_TIME="${START_TIME:-6}"
SEED="${SEED:-0}"

# --data_index indexes a seeded 11-event sample of the split
# (`assim_sample` in data/SEVIR/config.yaml).
#   index 0      tuning trajectory (do not pool with the evaluation set)
#   indices 1-10 evaluation
TUNE_INDEX="${TUNE_INDEX:-0}"
TEST_INDICES="${TEST_INDICES:-1 2 3 4 5 6 7 8 9 10}"
DATA_INDEX="${DATA_INDEX:-$TUNE_INDEX}"

# Pretrained models live under $MODELS_ROOT (see .env.example). The Python-side
# table is assimilation/checkpoints.py.
models_path () {
    [[ -n "${MODELS_ROOT:-}" ]] && echo "${MODELS_ROOT}/$1"
}
SEVIR_MODELS="${SEVIR_MODELS:-$(models_path DAWIS/models_SEVIR)}"
CKPT_FILE="${CKPT_FILE:-last.ckpt}"

# --init_states -> FMW window checkpoint (hidden_dim 32*(init_states+1)). The
# directory is the one in the released download (Erik-Wikingsson/dawis-checkpoints);
# do not rename. Mirrors FMW_MODELS["SEVIR"] in assimilation/checkpoints.py.
fmw_ckpt_for () {
    local init="$1" dir override
    override="ASSIM_FMW_PATH_SEVIR_ETA01_CHANNEL_INIT_${init}"
    if [[ -n "${!override:-}" ]]; then
        echo "${!override}"
        return 0
    fi
    case "$init" in
        6) dir="SEVIR_FLOWDAS_SPLIT_lr_vil_eta_channel_init_6_flowdas-FMW-224-09_12_05-1441" ;;
        *)
            echo "ERROR: no SEVIR FMW checkpoint for init_states=${init} (have 6)." >&2
            echo "  Train one with forecasting/training_scripts/SEVIR/train_all_fmw.sh" >&2
            echo "  and pass it as FMW_CKPT or ${override}." >&2
            return 1 ;;
    esac
    [[ -n "$SEVIR_MODELS" ]] && echo "${SEVIR_MODELS}/${dir}/${CKPT_FILE}"
}

# Deterministic U-Net forecaster (forecasting/trainer.py --model UNET), usable as a propagator.
UNET_CKPT="${UNET_CKPT:-}"  # not released; train one to use FORWARD_MODEL=unet

FMW_CKPT="${FMW_CKPT:-}"

# Unconditional DAISI prior (128x128).
PRIOR_CKPT="${PRIOR_CKPT:-${SEVIR_PRIOR_PATH:-$(models_path SQG/models/daisi/daisi_sevir_128.pth)}}"

# Prior UNet attention level, by resolution (level 1 = img_resolution / 2).
case "$VARIANT" in
    lr_vil)    _attn=64 ;;
    lr_vil_64) _attn=32 ;;
    *)         _attn=64 ;;
esac
ATTN_RESOLUTIONS="${ATTN_RESOLUTIONS:-$_attn}"
unset _attn

# FlowDAS's pretrained SEVIR checkpoint (their `latest.pt`, downloaded separately;
# see the Pretrained models section of the top-level README).
FLOWDAS_CKPT="${FLOWDAS_CKPT:-${FLOWDAS_SEVIR_MODEL_PATH:-$(models_path SQG/models/flowdas/flowdas_sevir.pt)}}"

WANDB_PROJECT="${WANDB_PROJECT:-ScoreDA_SEVIR}"

# LOG_MEDIA=true uploads videos and field plots to wandb (off for sweeps).
LOG_MEDIA="${LOG_MEDIA:-false}"
case "$LOG_MEDIA" in
    true)  _media_arg="" ;;
    false) _media_arg="--no_media_wandb" ;;
    *)
        echo "ERROR: LOG_MEDIA must be true or false (got '${LOG_MEDIA}')." >&2
        exit 1 ;;
esac

# ONLINE_METRICS streams scalar metrics to wandb during the run. Filters only
# (assimilate.py); the smoothers log at the end.
ONLINE_METRICS="${ONLINE_METRICS:-true}"
case "$ONLINE_METRICS" in
    true|false) ;;
    *)
        echo "ERROR: ONLINE_METRICS must be true or false (got '${ONLINE_METRICS}')." >&2
        exit 1 ;;
esac
sevir_online_args () {
    [[ "$ONLINE_METRICS" == "true" ]] && echo "--online_metrics"
    return 0
}

# wandb is configured through the environment (assimilate.py has no flag).
WANDB_MODE="${WANDB_MODE:-online}"
case "$WANDB_MODE" in
    disabled)
        export WANDB_MODE=disabled WANDB_DISABLED=true ;;
    offline)
        export WANDB_MODE=offline ;;
    online)
        export WANDB_MODE=online ;;
    *)
        echo "ERROR: WANDB_MODE must be online, offline or disabled (got '${WANDB_MODE}')." >&2
        exit 1 ;;
esac

# FMW architecture flags must match training (FMW is built from args, not the
# checkpoint). Defaults match train_all_fmw.sh; hidden_dim = 32*(init_states+1).
FM_LOSS="${FM_LOSS:-eta01_channel}"
ALPHA_BETA_MULT="${ALPHA_BETA_MULT:-2}"

# Used unquoted.
fmw_arch_args () {
    local init="$1"
    echo "--fm_loss ${FM_LOSS}" \
         "--hidden_dim ${HIDDEN_DIM:-$((32 * (init + 1)))}" \
         "--alpha_beta_mult ${ALPHA_BETA_MULT}"
}

require_ckpt () {
    local path="$1" what="$2" how="$3"
    if [[ -z "$path" ]]; then
        echo "ERROR: $what is not set. Set MODELS_ROOT in the repo .env for the" >&2
        echo "  pretrained models, or train one with $how and pass its path." >&2
        exit 1
    fi
    if [[ ! -f "$path" ]]; then
        echo "ERROR: $what not found at: $path" >&2
        exit 1
    fi
}

# Forward model propagating the ensemble between analyses:
#   none     no forecast
#   unet     deterministic U-Net ($UNET_CKPT); adds no spread
#   fmw      FMW window model ($FMW_CKPT or fmw_ckpt_for $FWD_INIT_STATES)
#   flowdas  FlowDAS's pretrained interpolant (6 conditioning frames)
#   self     DAWIS as its own forward model (needed for GUIDE_FORECAST)
#   dawis    DAWIS generates t+1 inside stage 1 (run_DAWIS.sh only)
# Architecture flags are read from the checkpoint. $1 lists the allowed values.
forward_model_args () {
    local allowed="$1" fm="${FORWARD_MODEL:-none}" ckpt
    case " $allowed " in
        *" $fm "*) ;;
        *)
            echo "ERROR: FORWARD_MODEL '${fm}' is not available for this method." >&2
            echo "  Available here: ${allowed}" >&2
            exit 1 ;;
    esac
    case "$fm" in
        none|dawis)
            echo "--forward_model ${fm}" ;;
        self)
            # No path = the method is its own forward model.
            echo "--forward_model fmw" ;;
        flowdas)
            require_ckpt "$FLOWDAS_CKPT" "FLOWDAS_CKPT (the flowdas propagator)" \
              "downloading Jia et al.'s released SEVIR checkpoint"
            echo "--forward_model flowdas" ;;
        unet)
            ckpt="${LEARNED_CKPT:-$UNET_CKPT}"
            require_ckpt "$ckpt" "UNET_CKPT (the unet propagator)" \
              "forecasting/trainer.py --model UNET"
            echo "--forward_model unet --forward_model_path ${ckpt}" ;;
        fmw)
            # Independent of the method's own --init_states.
            ckpt="${LEARNED_CKPT:-${FMW_CKPT:-$(fmw_ckpt_for "${FWD_INIT_STATES:-${INIT_STATES:-2}}")}}"
            require_ckpt "$ckpt" "FMW_CKPT (the fmw propagator)" \
              "forecasting/training_scripts/SEVIR/train_all_fmw.sh"
            echo "--forward_model fmw --forward_model_path ${ckpt}" ;;
        *)
            echo "ERROR: unknown FORWARD_MODEL '${fm}'." >&2
            exit 1 ;;
    esac
}

# Arguments shared by every SEVIR run. Used unquoted.
sevir_common_args () {
    echo "--dataset ${DATASET}" \
         "--variant ${VARIANT}" \
         "--split ${SPLIT}" \
         "--assim_norm ${ASSIM_NORM}" \
         "--forward_norm ${FORWARD_NORM}" \
         "--output_norm ${OUTPUT_NORM}" \
         "--obs_fn ${OBS_FN}" \
         "--obs_prob ${OBS_PROB}" \
         "--obs_sigma ${OBS_SIGMA}" \
         "--fixed_obs" \
         "--assim_interval ${ASSIM_INTERVAL}" \
         "--experiment ${EXPERIMENT}" \
         "--init_std ${INIT_STD}" \
         "--init_state ${INIT_STATE}" \
         "--n_ens ${N_ENS}" \
         "--batch_size ${BATCH_SIZE}" \
         "--n_times ${N_TIMES}" \
         "--start_time ${START_TIME}" \
         "--data_index ${DATA_INDEX}" \
         "--seed ${SEED}" \
         "${_media_arg}" \
         "--wandb_project ${WANDB_PROJECT}"
}

# An event is 25 frames; checked before allocating the GPU.
check_window_fits () {
    local window="${1:-0}" max=25
    if (( START_TIME < window )); then
        echo "ERROR: --start_time ${START_TIME} is less than the conditioning" \
             "window ${window}; there are no earlier states to condition on." >&2
        exit 1
    fi
    if (( START_TIME + N_TIMES > max )); then
        echo "ERROR: start_time ${START_TIME} + n_times ${N_TIMES} =" \
             "$((START_TIME + N_TIMES)) exceeds the ${max} frames in a SEVIR" \
             "event. Lower N_TIMES to $((max - START_TIME)) or START_TIME to" \
             "$((max - N_TIMES))." >&2
        exit 1
    fi
}

# Printed by every worker so the log records the run's units.
print_sevir_config () {
    echo "SEVIR run configuration"
    echo "  dataset      ${DATASET}/${VARIANT} split=${SPLIT}"
    echo "  norm         assim=${ASSIM_NORM} forward=${FORWARD_NORM} output=${OUTPUT_NORM}"
    echo "               the run computes in ${ASSIM_NORM}: the state, obs_sigma"
    echo "               (${OBS_SIGMA}) and init_std (${INIT_STD}) are in those units"
    echo "               -- multiply by ${SEVIR_NORM_STD} to read VIL counts."
    if [[ "$ASSIM_NORM" != "$FORWARD_NORM" ]]; then
    echo "               the propagator checkpoint is ${FORWARD_NORM}; the"
    echo "               forecaster bridges into it and back (it prints the affine)."
    fi
    case "$OUTPUT_NORM" in
        physical) echo "               results (.nc + metrics) are in VIL counts, 0-255." ;;
        assim)    echo "               results (.nc + metrics) are in ${ASSIM_NORM} units." ;;
        *)        echo "               results (.nc + metrics) are in ${OUTPUT_NORM} units." ;;
    esac
    echo "  experiment   ${EXPERIMENT}: obs_fn=${OBS_FN} obs_prob=${OBS_PROB}"
    echo "               obs_sigma=${OBS_SIGMA} init_std=${INIT_STD} init_state=${INIT_STATE}"
    echo "               (field std in these units is ~${SEVIR_FIELD_STD},"
    echo "                i.e. ~${SEVIR_FIELD_STD_COUNTS} VIL counts)"
    echo "  localization hcovlocal=${HCOVLOCAL_SCALE} m covinflate1=${COVINFLATE1}" \
         "covinflate2=${COVINFLATE2}"
    echo "  trajectory   data_index=${DATA_INDEX} of the seeded 11-event sample"
    echo "  run          n_ens=${N_ENS} batch_size=${BATCH_SIZE} n_times=${N_TIMES} start_time=${START_TIME} seed=${SEED}"
    echo "  obs cadence  assim_interval=${ASSIM_INTERVAL}" \
         "(1 = observations at every step)"
    echo "  wandb        ${WANDB_MODE} project=${WANDB_PROJECT} media=${LOG_MEDIA}"
}
