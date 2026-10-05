#!/bin/bash
# Launches the ForcingDAS-Pyr baseline on SQG (DAWIS model in pyramid mode).
# Usage: DRY_RUN=1 bash assimilation/scripts/SQG/run_all_ForcingDAS.bash
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

time_limit="06:00:00"

experiments=("noisy" "sparse" "saturating" "multimodal") # 4

guide_all_steps_values=(true)
tmin_starts=(None)
tmin_ends=(None)
tmin_inits=("")
forward_models=("none")
guide_methods=("MMPS")
noise_values=("None")

euler_steps=(20)
eps_values=(0.03)
invert_eps_values=(0.03)
invert_steps=(100)

save_window_states=false

# Per-experiment guidance strength; the observation operator and
# localization/inflation come from the --experiment preset.
declare -A guidance_strength_per_exp=(
    [sparse]=15 [noisy]=2 [saturating]=150 [multimodal]=4
)

n_fixed_values=(1)

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
# The paper ForcingDAS-Pyr runs used GT (not GT_edit): conditioning starts from
# the unperturbed window and the ensemble gains spread from sampling.
init_state_values=("GT")

guide_first_values=("none")

pyramid_values=(true)

x0_sigma_values=(1)

wandb_mode="online"

source "$REPO_ROOT/assimilation/scripts/SQG/_models.sh"
for _w in 1 2 3 4 5 6; do printf -v "AWIS_$_w" '%s' "$(sqg_fmw_ckpt $_w 2>/dev/null)"; done

# Test trajectory indices (0 = tuning trajectory, 1-10 = evaluation).
data_indices=(0 1 2 3 4 5 6 7 8 9 10)
# Env overrides.
apply_data_indices
apply_init_states
for data_index in "${data_indices[@]}"; do
    for experiment in "${experiments[@]}"; do
    # Keyed on the base experiment name (strips the -5 suffix).
    guidance_strengths=("${guidance_strength_per_exp[${experiment%%-*}]}")

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
    for tmin_init in "${tmin_inits[@]}"; do
    for pyramid in "${pyramid_values[@]}"; do
    for n_fixed in "${n_fixed_values[@]}"; do
    # Checkpoint and architecture per window size.
    case $init_states in
        1) model_path="$AWIS_1" hidden_dim=64 channel_mult_emb=4 channel_mult_noise=2;;
        2) model_path="$AWIS_2" hidden_dim=96 channel_mult_emb=4 channel_mult_noise=2;;
        3) model_path="$AWIS_3" hidden_dim=128 channel_mult_emb=4 channel_mult_noise=2;;
        4) model_path="$AWIS_4" hidden_dim=160 channel_mult_emb=4 channel_mult_noise=2;;
        5) model_path="$AWIS_5" hidden_dim=192 channel_mult_emb=4 channel_mult_noise=2;;
        6) model_path="$AWIS_6" hidden_dim=224 channel_mult_emb=4 channel_mult_noise=2;;
        *)
            echo "No FMW architecture for init_states=${init_states} in this launcher (it knows 1-6)" >&2
            exit 1
            ;;
    esac

    if [[ "${pyramid}" == "true" ]]; then
        exp_name="ForcingDAS-Pyr-X0_${experiment}"
    else
        exp_name="DAWIS-${forward_model}_${experiment}"
    fi
    # Window tag (empty at the default window) and trajectory suffix.
    exp_name="${exp_name}$(init_states_tag "$init_states")${ai_tag}_run_${data_index}"

    echo "Submitting job ${exp_name}"

    guide_all_steps_args=()
    if [[ "${guide_all_steps}" == "true" ]]; then
        guide_all_steps_args=(--guide_all_steps)
    else
        guide_all_steps_args=(--no-guide_all_steps)
    fi

    # The pyramid builds its own noise ladder and needs no forward model.
    pyramid_args=()
    run_forward_model="${forward_model}"
    if [[ "${pyramid}" == "true" ]]; then
        pyramid_args=(--pyramid --n_fixed "${n_fixed}")
        run_forward_model="none"
    else
        pyramid_args=(--no-pyramid)
    fi

    if [[ "${save_window_states}" == "true" ]]; then
        save_window_states_args=(--save_window_states)
    else
        save_window_states_args=(--no-save_window_states)
    fi

    sbatch "${SBATCH_MAIL_ARGS[@]}" -t "$time_limit" --job-name="$exp_name" \
        assimilation/scripts/SQG/run_experiment_DAWIS.bash \
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
        --tmin_init "${tmin_init}" \
        --guidance_strength "${guidance}" \
        --invert_steps "${invert_step}" \
        --invert_eps "${invert_eps}" \
        --n_ens "${n_ens}" \
        --n_times "${n_times}" \
        --forward_model "${run_forward_model}" \
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
        "${guide_all_steps_args[@]}" \
        "${pyramid_args[@]}" \
        "${save_window_states_args[@]}" \

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
done


