#!/bin/bash
# Launches DAWIS SQG assimilation jobs (forward models dawis and numerical).
# Usage: DRY_RUN=1 bash assimilation/scripts/SQG/run_all_DAWIS.bash [--sweep true|false]
# Env knobs: DRY_RUN, DATA_INDICES, INIT_STATES_VALUES, FORWARD_MODELS, ASSIM_INTERVAL.

set -euo pipefail

# Sets cwd, $REPO_ROOT and $PYTHONPATH.
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"

# Job-mail options from $MAIL_TYPE / $MAIL_USER (must follow repo_root.sh).
source "$REPO_ROOT/scripts/sbatch_opts.sh"

# DRY_RUN=1 prints sbatch commands instead of submitting; --sweep true|false;
# DATA_INDICES selects trajectories; INIT_STATES_VALUES selects window sizes.
source "$REPO_ROOT/assimilation/scripts/_dry_run.sh"
source "$REPO_ROOT/assimilation/scripts/_sweep_arg.sh" "$@"
source "$REPO_ROOT/assimilation/scripts/_data_indices.sh"
source "$REPO_ROOT/assimilation/scripts/_init_states.sh"

# Window sizes with checkpoints; keep in sync with the `case $init_states` below.
_init_states_supported="6"
_init_states_override_prefix="ASSIM_FMW_PATH_SQG_ETA01_CHANNEL"

sweep="${SWEEP}"

time_limit="10:00:00"

experiments=("noisy" "sparse" "multimodal")

# guide_all_steps, guidance_strength and tmin are set per experiment below.
guide_all_steps_values=(true false)

# tmin_init (length init_states) is built from two knobs:
#   tmin_init_heads -> the leading (init_states - 1) slots
#   tmin_init_ics   -> the last (initial-condition) slot
# Set both to ("none") to disable tmin_init.
tmin_init_heads=(0.0)
tmin_init_ics=(0.9)

# "dawis" = learned window model (DAWIS-Joint), "numerical" = SQG solver.
forward_models=("dawis" "numerical")

guide_methods=("MMPS")
noise_values=("invert")
euler_steps=(100)
eps_values=(0.03)
invert_eps_values=(1.0 2.0)
sampler_eps_values=(1.0 2.0) # Only default, not used.
invert_steps=(100)

methods=("DAWIS")
fm_losses=("eta01_channel")

# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
# Override with ASSIM_INTERVAL=k; values != 1 are tagged into the run name.
assim_interval="${ASSIM_INTERVAL:-1}"
ai_tag=""
[[ "$assim_interval" != "1" ]] && ai_tag="_ai${assim_interval}"

n_ens_values=(20)
n_times_values=(100)
resample_filters=("1,3,3,1")
channel_mults=("2,2,2")
attn_resolutions_values=("32")
schedules=("linear_scalar")
samplers=("stochastic")

noise_embeddings=("positional")

init_states_values=(6) # window size W (W+1 states)
init_state_values=("GT_edit")

if [[ "$sweep" == "false" ]]; then
    # Paper configuration (DAWIS and DAWIS-Joint rows on SQG); per-experiment
    # settings are in the case block below.
    experiments=("noisy" "sparse" "multimodal" "saturating")
    forward_models=("dawis" "numerical")

    init_states_values=(6)
    init_state_values=("GT_edit")
    tmin_init_heads=(0.0)
    tmin_init_ics=(0.9)
    guide_methods=("MMPS")
    noise_values=("invert")
    n_ens_values=(20)
    n_times_values=(100)
    euler_steps=(100)
    invert_steps=(100)
    x0_sigma_values=(1)
    # The paper runs used 0.03 for all three noise levels.
    eps_values=(0.03)
    invert_eps_values=(0.03)
    sampler_eps_values=(0.03)
fi

guide_all_steps_values=(false true)
guidance_strengths=(1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)

guide_first_values=("none")

x0_sigma_values=(1)

# true saves every window state as x_window(t, lag, ens, z, y, x) (lag 0 =
# filter, lag W = smoothed).
save_window_states=false
nx=64  # only used for the disk estimate below; matches the --nx default

wandb_mode="online"

source "$REPO_ROOT/assimilation/scripts/SQG/_models.sh"
for _w in 1 2 3 4 5 6; do printf -v "DAWIS_$_w" '%s' "$(sqg_fmw_ckpt $_w 2>/dev/null)"; done

# `head` repeated n-1 times, then `ic`.
make_tmin_init() {
    local head=$1 ic=$2 n=$3
    awk -v h="$head" -v c="$ic" -v n="$n" 'BEGIN{
        for (i = 0; i < n; i++)
            printf "%s%.6g", (i ? " " : ""), (i == n - 1 ? c : h)
    }'
}

# Uncompressed size of x_window in MB (on disk it is zlib-compressed).
window_size_mb() {
    local n_times=$1 init_states=$2 n_ens=$3 nx=$4
    awk -v t="$n_times" -v w="$init_states" -v e="$n_ens" -v x="$nx" \
        'BEGIN{printf "%.0f", t * (w + 1) * e * 2 * x * x * 4 / 1048576}'
}

# 0.9 -> 0p9, so exp_name stays filename-friendly (it becomes results/<name>.nc).
tag() { echo "${1//./p}"; }

# Test trajectories (0 = tuning trajectory, 1-10 = evaluation).
data_indices=(0)


n_jobs=0
window_mb_total=0
# Env overrides; applied after the `--sweep false` block.
apply_data_indices
apply_init_states
for data_index in "${data_indices[@]}"; do
    for experiment in "${experiments[@]}"; do

    # Per-experiment tuned settings (paper hyperparameter table). tmin is set
    # in both modes; guide_all_steps / guidance_strength only for --sweep false.
    # Observation operator and localization/inflation come from the
    # --experiment preset in assimilation/experiments.py.
    case $experiment in
        "noisy")
            tmin_starts=(0.4) tmin_ends=(0.3)
            if [[ "$sweep" == "false" ]]; then
                guide_all_steps_values=(false)
                guidance_strengths=(1.0)
            fi ;;
        "sparse")
            tmin_starts=(0.4) tmin_ends=(0.1)
            if [[ "$sweep" == "false" ]]; then
                guide_all_steps_values=(true)
                guidance_strengths=(1.0)
            fi ;;
        "multimodal")
            tmin_starts=(0.4) tmin_ends=(0.2)
            if [[ "$sweep" == "false" ]]; then
                guide_all_steps_values=(false)
                guidance_strengths=(1.0)
            fi ;;
        "saturating")
            tmin_starts=(0.4) tmin_ends=(0.4)
            if [[ "$sweep" == "false" ]]; then
                guide_all_steps_values=(true)
                guidance_strengths=(20)
            fi ;;
        *)
            echo "Unknown experiment=${experiment}"
            exit 1
            ;;
    esac

    # FORWARD_MODELS=... overrides forward_models. `dawis` requires
    # assim_interval 1, so ASSIM_INTERVAL sweeps use FORWARD_MODELS=numerical.
    if [[ -n "${FORWARD_MODELS+set}" ]]; then
        read -r -a forward_models <<< "${FORWARD_MODELS}"
        if [[ ${#forward_models[@]} -eq 0 ]]; then
            echo "ERROR: FORWARD_MODELS is set but empty -- no jobs would be submitted." >&2
            exit 1
        fi
        echo "FORWARD_MODELS override: forward_models=(${forward_models[*]})"
    fi

    for forward_model in "${forward_models[@]}"; do
    for method in "${methods[@]}"; do
    for fm_loss in "${fm_losses[@]}"; do
    for n_ens in "${n_ens_values[@]}"; do
    for n_times in "${n_times_values[@]}"; do
    for eps in "${eps_values[@]}"; do
    for invert_eps in "${invert_eps_values[@]}"; do
    for euler_step in "${euler_steps[@]}"; do
    for guide_method in "${guide_methods[@]}"; do
    for noise in "${noise_values[@]}"; do
    for invert_step in "${invert_steps[@]}"; do
    for guidance in "${guidance_strengths[@]}"; do
    for resample_filter in "${resample_filters[@]}"; do
    for channel_mult in "${channel_mults[@]}"; do
    for attn_resolutions in "${attn_resolutions_values[@]}"; do
    for schedule in "${schedules[@]}"; do
    for sampler in "${samplers[@]}"; do
    for sampler_eps in "${sampler_eps_values[@]}"; do
    for noise_embedding in "${noise_embeddings[@]}"; do
    for init_states in "${init_states_values[@]}"; do
    for guide_all_steps in "${guide_all_steps_values[@]}"; do
    for guide_first in "${guide_first_values[@]}"; do
    for init_state in "${init_state_values[@]}"; do
    for tmin_start in "${tmin_starts[@]}"; do
    for tmin_end in "${tmin_ends[@]}"; do
    for x0_sigma in "${x0_sigma_values[@]}"; do
    for tmin_init_head in "${tmin_init_heads[@]}"; do
    for tmin_init_ic in "${tmin_init_ics[@]}"; do
    # Checkpoint and architecture per window size.
    case $init_states in
        1) model_path="$DAWIS_1" hidden_dim=64 channel_mult_emb=4 channel_mult_noise=2;;
        2) model_path="$DAWIS_2" hidden_dim=96 channel_mult_emb=4 channel_mult_noise=2;;
        3) model_path="$DAWIS_3" hidden_dim=128 channel_mult_emb=4 channel_mult_noise=2;;
        4) model_path="$DAWIS_4" hidden_dim=160 channel_mult_emb=4 channel_mult_noise=2;;
        5) model_path="$DAWIS_5" hidden_dim=192 channel_mult_emb=4 channel_mult_noise=2;;
        6) model_path="$DAWIS_6" hidden_dim=224 channel_mult_emb=4 channel_mult_noise=2;;
        *)
            echo "No DAWIS architecture for init_states=${init_states} in this launcher (it knows 1-6)" >&2
            exit 1
            ;;
    esac

    # tmin_init is enabled/disabled as a pair; "none" in either knob disables it.
    if [[ "$tmin_init_head" == "none" || "$tmin_init_ic" == "none" ]]; then
        if [[ "$tmin_init_head" != "$tmin_init_ic" ]]; then
            echo "Skipping tmin_init head=${tmin_init_head} ic=${tmin_init_ic}: 'none' must be set in both"
            continue
        fi
        tmin_init=""
        tmin_init_tag="off"
    else
        tmin_init="$(make_tmin_init "$tmin_init_head" "$tmin_init_ic" "$init_states")"
        tmin_init_tag="h$(tag "$tmin_init_head")_ic$(tag "$tmin_init_ic")"
    fi

    # Empty at the default window, `_init<N>` otherwise.
    is_tag="$(init_states_tag "$init_states")"

    exp_name="DAWIS-${forward_model}_${experiment}${is_tag}${ai_tag}_run_${data_index}"

    echo "Submitting job ${exp_name} | tmin_init=[${tmin_init:-none}]"

    save_window_states_args=()
    if [[ "${save_window_states}" == "true" ]]; then
        save_window_states_args=(--save_window_states)
        window_mb=$(window_size_mb "$n_times" "$init_states" "$n_ens" "$nx")
        window_mb_total=$((window_mb_total + window_mb))
        echo "  saving $((init_states + 1)) lags x ${n_ens} members =" \
             "$(( (init_states + 1) * n_ens )) states per time, over" \
             "$((n_times - init_states)) fully-lagged times" \
             "(~${window_mb} MB of x_window, uncompressed)"
    else
        save_window_states_args=(--no-save_window_states)
    fi

    guide_all_steps_args=()
    if [[ "${guide_all_steps}" == "true" ]]; then
        guide_all_steps_args=(--guide_all_steps)
    else
        guide_all_steps_args=(--no-guide_all_steps)
    fi

    tmin_init_args=()
    if [[ -n "$tmin_init" ]]; then
        tmin_init_args=(--tmin_init "${tmin_init}")
    fi

    sbatch "${SBATCH_MAIL_ARGS[@]}" -t "$time_limit" --job-name="$exp_name" assimilation/scripts/SQG/run_experiment_DAWIS.bash \
        --exp_name "${exp_name}" \
        --experiment "${experiment}" \
        --data_index "${data_index}" \
        --assim_interval "${assim_interval}" \
        --euler_steps "${euler_step}" \
        --guide_method "${guide_method}" \
        --eps "${eps}" \
        --noise "${noise}" \
        --tmin_start "${tmin_start}" \
        --tmin_end "${tmin_end}" \
        "${tmin_init_args[@]}" \
        --guidance_strength "${guidance}" \
        --invert_steps "${invert_step}" \
        --invert_eps "${invert_eps}" \
        --n_ens "${n_ens}" \
        --n_times "${n_times}" \
        --forward_model "${forward_model}" \
        --method "${method}" \
        --fm_loss "${fm_loss}" \
        --model_path "${model_path}" \
        --resample_filter "${resample_filter}" \
        --channel_mult "${channel_mult}" \
        --attn_resolutions "${attn_resolutions}" \
        --schedule "${schedule}" \
        --sampler "${sampler}" \
        --sampler_eps "${sampler_eps}" \
        --hidden_dim "${hidden_dim}" \
        --noise_embedding "${noise_embedding}" \
        --channel_mult_emb "${channel_mult_emb}" \
        --channel_mult_noise "${channel_mult_noise}" \
        --init_states "${init_states}" \
        --init_state "${init_state}" \
        --guide_first "${guide_first}" \
        --wandb_mode "${wandb_mode}" \
        --x0_sigma "${x0_sigma}" \
        "${save_window_states_args[@]}" \
        "${guide_all_steps_args[@]}" \

    n_jobs=$((n_jobs + 1))

    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
    done
done

echo "Total jobs: ${n_jobs}"
if [[ "$save_window_states" == "true" ]]; then
    echo "x_window totals ~$(awk -v m="$window_mb_total" \
        'BEGIN{printf "%.1f", m / 1024}') GB uncompressed across all jobs"
fi
