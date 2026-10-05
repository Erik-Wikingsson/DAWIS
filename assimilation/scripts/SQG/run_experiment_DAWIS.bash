#!/bin/bash
# Worker for DAWIS (submitted by run_all_DAWIS.bash).
#SBATCH -t 10:00:00
#SBATCH --gpus=1
#SBATCH -C "fat"
#SBATCH --output ./slurm_logs/%A_%x.out

# Activate the conda env. Uses $REPO_ROOT (exported by the launcher), since sbatch runs a copy of this script.
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

# Defaults (overridable from the launcher)
exp_name="DAWIS_default"
experiment="sparse"
# Test trajectory index (0 = tuning trajectory, 1-10 = evaluation).
data_index=0
euler_step=100
guide_method="MMPS"
eps=0.03
noise="invert"
tmin="0.3 0.3"
tmin_given=false  # --tmin explicitly passed? (matters in pyramid mode)
tmax=""        # empty/None -> --tmax not passed (DAWIS defaults to 1.0 everywhere)
tmin_init=""   # empty/None -> --tmin_init not passed (feature disabled)
guidance=1.0
invert_step=100
invert_eps=0.03
# Assimilate every k steps (1 = every step, -1 = never). The paper runs used 1.
# With --forward_model dawis an unobserved step is an unguided sampling pass.
assim_interval=1
n_ens=20
n_times=100
forward_model="fmw"
method="DAWIS"
fm_loss="eta01_label"
resample_filter="1,3,3,1"
channel_mult="2,2,2"
attn_resolutions="32"
schedule="linear_scalar"
sampler="stochastic"
sampler_eps=0.03
hidden_dim=32
noise_embedding="positional"
channel_mult_emb=4
channel_mult_noise=2
init_states=1
init_state="GT_edit"
fixed_obs=true
guide_all_steps=false
guide_first="none"
save_window_states=false
pyramid=false
n_fixed=0     # --pyramid: leading window states pinned at clean data
wandb_mode="online"
tmin_start=None
tmin_end=None
x0_sigma=1000
corrections=0
tau=1
 
# Follow $REPO_ROOT if exported, else resolve this checkout.
if [[ -n "${REPO_ROOT:-}" ]]; then
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT"
else
    source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"
fi

# Shared results dir (checked before the run); defines fix_results_permissions.
source "$REPO_ROOT/scripts/results_dir.sh"

DAWIS_pos_1_32="saved_models/AWIS_pos-FMW-32-06_02_16-8811/last.ckpt"
model_path="$DAWIS_pos_1_32"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_name) exp_name="$2"; shift ;;
        --experiment) experiment="$2"; shift ;;
        --data_index) data_index="$2"; shift ;;
        --euler_steps) euler_step="$2"; shift ;;
        --guide_method) guide_method="$2"; shift ;;
        --eps) eps="$2"; shift ;;
        --noise) noise="$2"; shift ;;
        --tmin) tmin="$2"; tmin_given=true; shift ;;
        --tmax) tmax="$2"; shift ;;
        --tmin_init) tmin_init="$2"; shift ;;
        --tmin_start) tmin_start="$2"; shift ;;
        --tmin_end) tmin_end="$2"; shift ;;
        --guidance_strength) guidance="$2"; shift ;;
        --invert_steps) invert_step="$2"; shift ;;
        --invert_eps) invert_eps="$2"; shift ;;
        --assim_interval) assim_interval="$2"; shift ;;
        --n_ens) n_ens="$2"; shift ;;
        --n_times) n_times="$2"; shift ;;
        --forward_model) forward_model="$2"; shift ;;
        --method) method="$2"; shift ;;
        --fm_loss) fm_loss="$2"; shift ;;
        --model_path) model_path="$2"; shift ;;
        --resample_filter) resample_filter="$2"; shift ;;
        --channel_mult) channel_mult="$2"; shift ;;
        --attn_resolutions) attn_resolutions="$2"; shift ;;
        --schedule) schedule="$2"; shift ;;
        --sampler) sampler="$2"; shift ;;
        --sampler_eps) sampler_eps="$2"; shift ;;
        --hidden_dim) hidden_dim="$2"; shift ;;
        --noise_embedding) noise_embedding="$2"; shift ;;
        --channel_mult_emb) channel_mult_emb="$2"; shift ;;
        --channel_mult_noise) channel_mult_noise="$2"; shift ;;
        --init_states) init_states="$2"; shift ;;
        --init_state) init_state="$2"; shift ;;
        --fixed_obs) fixed_obs=true ;;
        --no-fixed_obs) fixed_obs=false ;;
        --guide_first) guide_first="$2"; shift ;;
        --guide_all_steps) guide_all_steps=true ;;
        --no-guide_all_steps) guide_all_steps=false ;;
        --pyramid) pyramid=true ;;
        --n_fixed) n_fixed="$2"; shift ;;
        --no-pyramid) pyramid=false ;;
        --save_window_states) save_window_states=true ;;
        --no-save_window_states) save_window_states=false ;;
        --wandb_mode) wandb_mode="$2"; shift ;;
        --x0_sigma) x0_sigma="$2"; shift ;;
        --corrections) corrections="$2"; shift ;;
        --tau) tau="$2"; shift ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ "$wandb_mode" == "disabled" ]]; then
    export WANDB_MODE=disabled
    export WANDB_DISABLED=true
elif [[ "$wandb_mode" == "offline" ]]; then
    export WANDB_MODE=offline
    wandb offline
else
    export WANDB_MODE=online
    wandb online
fi

# Pyramid mode derives tmin from the window size; pass --tmin only if given.
tmin_pass_args=()
if [[ "$pyramid" != true || "$tmin_given" == true ]]; then
    read -r -a tmin_args <<< "$tmin"
    tmin_pass_args=(--tmin "${tmin_args[@]}")
fi

# Per-state end depth of the guided forward phase (default 1.0; --pyramid overrides).
tmax_args=()
if [[ -n "$tmax" && "$tmax" != "None" ]]; then
    read -r -a tmax_vals <<< "$tmax"
    tmax_args=(--tmax "${tmax_vals[@]}")
fi

tmin_init_args=()
if [[ -n "$tmin_init" && "$tmin_init" != "None" ]]; then
    read -r -a tmin_init_vals <<< "$tmin_init"
    tmin_init_args=(--tmin_init "${tmin_init_vals[@]}")
fi

fixed_obs_args=()
if [[ "$fixed_obs" == true ]]; then
    fixed_obs_args=(--fixed_obs)
fi

guide_all_steps_args=()
if [[ "$guide_all_steps" == true ]]; then
    guide_all_steps_args=(--guide_all_steps)
fi

# Also save every window state as x_window(t, lag, ens, z, y, x) in the result file.
save_window_states_args=()
if [[ "$save_window_states" == true ]]; then
    save_window_states_args=(--save_window_states)
fi

pyramid_args=()
if [[ "$pyramid" == true ]]; then
    pyramid_args=(--pyramid --n_fixed "$n_fixed")
fi

tmin_range_args=()
if [[ "$pyramid" != true && -n "$tmin_start" && -n "$tmin_end" && "$tmin_start" != "None" && "$tmin_end" != "None" ]]; then
    tmin_range_args=(--tmin_start "$tmin_start" --tmin_end "$tmin_end")
fi

echo "Running experiment: ${exp_name} (method=${method}, model=${forward_model}, experiment=${experiment}), wandb_mode=${wandb_mode}, hidden_dim=${hidden_dim}"

python3 -m assimilation.assimilate \
    --exp_name "${exp_name}" \
    --experiment "${experiment}" \
    --data_index "${data_index}" \
    --euler_steps "${euler_step}" \
    --guide_method "${guide_method}" \
    --eps "${eps}" \
    --noise "${noise}" \
    --guidance_strength "${guidance}" \
    --invert_steps "${invert_step}" \
    --invert_eps "${invert_eps}" \
    --assim_interval "${assim_interval}" \
    --n_ens "${n_ens}" \
    --n_times "${n_times}" \
    --forward_model "${forward_model}" \
    "${tmin_pass_args[@]}" \
    "${tmax_args[@]}" \
    "${tmin_init_args[@]}" \
    "${tmin_range_args[@]}" \
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
    --window "${init_states}" \
    "${fixed_obs_args[@]}" \
    --alpha_beta_mult 1 \
    --guide_first "${guide_first}" \
    "${guide_all_steps_args[@]}" \
    "${save_window_states_args[@]}" \
    "${pyramid_args[@]}" \
    --init_state "${init_state}" \
    --x0_sigma "${x0_sigma}" \
    --corrections "${corrections}" \
    --tau "${tau}" \
    --online_metrics \
    --no_media_wandb

fix_results_permissions