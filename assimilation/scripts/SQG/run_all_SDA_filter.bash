#!/bin/bash
# Launches the SDA-Filter baseline on SQG (DAWIS model, SDEdit filtering).
# Usage: DRY_RUN=1 bash assimilation/scripts/SQG/run_all_SDA_filter.bash [--sweep true|false]
# Env knobs: DRY_RUN, DATA_INDICES, INIT_STATES_VALUES, ASSIM_INTERVAL.

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

time_limit="12:00:00"

# Sweep mode runs one experiment; --sweep false restores all four below.
experiments=("saturating")

guide_all_steps_values=(true)
tmin_starts=(0.0)
tmin_ends=(0.0)

forward_models=("numerical") # or none

guide_methods=("MMPS")
noise_values=("SDEdit")
euler_steps=(100)
eps_values=(1)
invert_eps_values=(0.03)
invert_steps=(100)

# Per-experiment guidance strength and tau; the observation operator and
# localization/inflation come from the --experiment preset.
guidance_strength_default=(2)   # sparse, noisy, multimodal
guidance_strength_sat=(10)      # saturating

corrections_values=(1)
tau_default=(0.3)               # sparse, noisy, saturating
tau_multimodal=(0.5)            # multimodal

x0_sigma_values=(1)

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
sampler_eps_values=(0.03)
noise_embeddings=("positional")

init_states_values=(6) # window size W (W+1 states)
# GT_guide: the ensemble is seeded by guiding the initial window (paper runs).
init_state_values=("GT_guide")

guide_first_values=("all")

wandb_mode="online"

source "$REPO_ROOT/assimilation/scripts/SQG/_models.sh"
for _w in 1 2 3 4 5 6; do printf -v "DAWIS_$_w" '%s' "$(sqg_fmw_ckpt $_w 2>/dev/null)"; done

# Paper configuration: all four experiments; everything else is already at
# the paper values.
if [[ "$sweep" == "false" ]]; then
    experiments=("noisy" "sparse" "saturating" "multimodal")
fi

# Test trajectory indices (0 = tuning trajectory, 1-10 = evaluation).
data_indices=(0 1 2 3 4 5 6 7 8 9 10)
# Env overrides; applied after the `--sweep false` block.
apply_data_indices
apply_init_states
for data_index in "${data_indices[@]}"; do
    for experiment in "${experiments[@]}"; do
    case "$experiment" in
        saturating|saturating-5)
            guidance_strengths=("${guidance_strength_sat[@]}")
            taus=("${tau_default[@]}")
            ;;
        multimodal)
            guidance_strengths=("${guidance_strength_default[@]}")
            taus=("${tau_multimodal[@]}")
            ;;
        *)
            guidance_strengths=("${guidance_strength_default[@]}")
            taus=("${tau_default[@]}")
            ;;
    esac

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
    for corrections in "${corrections_values[@]}"; do
    for tau in "${taus[@]}"; do
    # Checkpoint and architecture per window size.
    case $init_states in
        1) model_path="$DAWIS_1" hidden_dim=64 channel_mult_emb=4 channel_mult_noise=2;;
        2) model_path="$DAWIS_2" hidden_dim=96 channel_mult_emb=4 channel_mult_noise=2;;
        3) model_path="$DAWIS_3" hidden_dim=128 channel_mult_emb=4 channel_mult_noise=2;;
        4) model_path="$DAWIS_4" hidden_dim=160 channel_mult_emb=4 channel_mult_noise=2;;
        5) model_path="$DAWIS_5" hidden_dim=192 channel_mult_emb=4 channel_mult_noise=2;;
        6) model_path="$DAWIS_6" hidden_dim=224 channel_mult_emb=4 channel_mult_noise=2;;
        *)
            echo "No FMW architecture for init_states=${init_states} in this launcher (it knows 1-6)" >&2
            exit 1
            ;;
    esac

    # Empty at the default window, `_init<N>` otherwise.
    is_tag="$(init_states_tag "$init_states")"

    exp_name="SDAfilter_${experiment}${is_tag}${ai_tag}_run_${data_index}"

    echo "Submitting job ${exp_name}"

    guide_all_steps_args=()
    if [[ "${guide_all_steps}" == "true" ]]; then
        guide_all_steps_args=(--guide_all_steps)
    else
        guide_all_steps_args=(--no-guide_all_steps)
    fi

    sbatch "${SBATCH_MAIL_ARGS[@]}" -t "$time_limit" -C "thin" --job-name="$exp_name" assimilation/scripts/SQG/run_experiment_DAWIS.bash \
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
        --corrections "${corrections}" \
        --tau "${tau}" \
        "${guide_all_steps_args[@]}" \

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


