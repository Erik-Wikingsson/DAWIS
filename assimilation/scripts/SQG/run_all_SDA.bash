#!/bin/bash
# Score-based data assimilation (SDA) smoother on the SQG experiments, using the
# DAWIS checkpoints as prior.
#   DRY_RUN=1 bash assimilation/scripts/SQG/run_all_SDA.bash
# Env knobs: DATA_INDICES, INIT_STATES_VALUES, ASSIM_INTERVAL.

set -euo pipefail

# Sets cwd, $REPO_ROOT and $PYTHONPATH.
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"

# Job-mail options from $MAIL_TYPE / $MAIL_USER (must follow repo_root.sh).
source "$REPO_ROOT/scripts/sbatch_opts.sh"

# DRY_RUN=1 prints sbatch commands instead of submitting; --sweep true|false;
# DATA_INDICES / INIT_STATES_VALUES override trajectories / windows.
source "$REPO_ROOT/assimilation/scripts/_dry_run.sh"
source "$REPO_ROOT/assimilation/scripts/_sweep_arg.sh" "$@"
source "$REPO_ROOT/assimilation/scripts/_data_indices.sh"
source "$REPO_ROOT/assimilation/scripts/_init_states.sh"

# Windows with SQG checkpoints; keep in sync with the case $init_states table below.
_init_states_supported="6"
_init_states_override_prefix="ASSIM_FMW_PATH_SQG_ETA01_CHANNEL"

sweep="${SWEEP}"

time_limit="1-00:00:00"

# Experiments
experiments=("sparse" "noisy" "saturating" "multimodal")
methods=("SDA")          # or GIBBS

# Tuned per-experiment hyperparameters; observation settings come from the
# --experiment preset. --sampler_steps is forwarded to python as --euler_steps.
declare -A guidance_strength_per_exp=(
    [sparse]=5 [noisy]=2 [saturating]=30 [multimodal]=3
)
declare -A tau_per_exp=(
    [sparse]=0.5 [noisy]=0.5 [saturating]=0.5 [multimodal]=1
)
declare -A corrections_per_exp=(
    [sparse]=1 [noisy]=1 [saturating]=2 [multimodal]=3
)
declare -A sampler_steps_per_exp=(
    [sparse]=100 [noisy]=100 [saturating]=100 [multimodal]=200
)

trajectory_var="none"
trajectory_path="None"

noise_values=("None")
sampler_eps_values=(1)

init_states_values=(6)      # = 2*sda_w
# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
# Override with ASSIM_INTERVAL=k; values != 1 are tagged into the run name.
# For this smoother, skipped steps get an empty observation mask.
assim_interval="${ASSIM_INTERVAL:-1}"
ai_tag=""
[[ "$assim_interval" != "1" ]] && ai_tag="_ai${assim_interval}"

n_times_values=(100)
n_ens_values=(20)
window_batch_sizes=(3)

guide_first="init"
x0_sigma_values=(3)
init_std=1000

wandb_mode="online"

# Model checkpoints
source "$REPO_ROOT/assimilation/scripts/SQG/_models.sh"
for _w in 1 2 3 4 5 6; do printf -v "DAWIS_$_w" '%s' "$(sqg_fmw_ckpt $_w 2>/dev/null)"; done

# Test trajectory indices (0 = tuning trajectory, 1-10 = evaluation).
data_indices=(0 1 2 3 4 5 6 7 8 9 10)
# DATA_INDICES=... overrides the list above.
apply_data_indices
# INIT_STATES_VALUES=... overrides init_states_values.
apply_init_states
for data_index in "${data_indices[@]}"; do
    for experiment in "${experiments[@]}"; do
    # Keyed on the base name, so -5 variants use their base settings.
    exp_key="${experiment%%-*}"
    guidance_strengths=("${guidance_strength_per_exp[$exp_key]}")
    tau_values=("${tau_per_exp[$exp_key]}")
    corrections_values=("${corrections_per_exp[$exp_key]}")
    sampler_steps_values=("${sampler_steps_per_exp[$exp_key]}")

    for method in "${methods[@]}"; do
    for init_states in "${init_states_values[@]}"; do
    for sampler_steps in "${sampler_steps_values[@]}"; do
    for sampler_eps in "${sampler_eps_values[@]}"; do
    for guidance_strength in "${guidance_strengths[@]}"; do
    for corrections in "${corrections_values[@]}"; do
    for tau in "${tau_values[@]}"; do
    for n_times in "${n_times_values[@]}"; do
    for n_ens in "${n_ens_values[@]}"; do
    for window_batch_size in "${window_batch_sizes[@]}"; do
    for noise in "${noise_values[@]}"; do
    for x0_sigma in "${x0_sigma_values[@]}"; do

        case $init_states in
            1) model_path="$DAWIS_1"; hidden_dim=64;  channel_mult_emb=4; channel_mult_noise=2;;
            2) model_path="$DAWIS_2"; hidden_dim=96;  channel_mult_emb=4; channel_mult_noise=2;;
            3) model_path="$DAWIS_3"; hidden_dim=128; channel_mult_emb=4; channel_mult_noise=2;;
            4) model_path="$DAWIS_4"; hidden_dim=160; channel_mult_emb=4; channel_mult_noise=2;;
            5) model_path="$DAWIS_5"; hidden_dim=192; channel_mult_emb=4; channel_mult_noise=2;;
            6) model_path="$DAWIS_6"; hidden_dim=224; channel_mult_emb=4; channel_mult_noise=2;;
            *)
                echo "Unknown init_states=${init_states}"
                exit 1
                ;;
        esac


        # Empty at the default window, _init<N> otherwise.
        is_tag="$(init_states_tag "$init_states")"

        exp_name="${method}_${experiment}${is_tag}${ai_tag}_run_${data_index}"

        echo "Submitting ${exp_name}"

        sbatch "${SBATCH_MAIL_ARGS[@]}" -t "${time_limit}" \
            --job-name="${exp_name}" \
            assimilation/scripts/SQG/run_experiment_SDA.bash \
            --exp_name "${exp_name}" \
            --method "${method}" \
            --experiment "${experiment}" \
            --data_index "${data_index}" \
            --assim_interval "${assim_interval}" \
            --model_path "${model_path}" \
            --hidden_dim "${hidden_dim}" \
            --channel_mult_emb "${channel_mult_emb}" \
            --channel_mult_noise "${channel_mult_noise}" \
            --init_states "${init_states}" \
            --n_times "${n_times}" \
            --n_ens "${n_ens}" \
            --window_batch_size "${window_batch_size}" \
            --sampler_steps "${sampler_steps}" \
            --sampler_eps "${sampler_eps}" \
            --guide_first "${guide_first}" \
            --x0_sigma "${x0_sigma}" \
            --init_std "${init_std}" \
            --guide_method MMPS \
            --guidance_strength "${guidance_strength}" \
            --noise "${noise}" \
            --corrections "${corrections}" \
            --tau "${tau}" \
            --fm_loss eta01_channel \
            --resample_filter "1,3,3,1" \
            --channel_mult "2,2,2" \
            --attn_resolutions 32 \
            --schedule linear_scalar \
            --noise_embedding positional \
            --wandb_mode "${wandb_mode}" \
            --trajectory_var "${trajectory_var}" \
            --trajectory_path "${trajectory_path}"

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
