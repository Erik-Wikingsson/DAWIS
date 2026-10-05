from data.paths import lookup
from data.paths import checkpoint
from data.registry import add_dataset_args
from argparse import ArgumentParser
import torch

def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def build_parser():
    """Build the argument parser (also used by `arch_defaults`)."""
    parser = ArgumentParser(
        description='Data Assimilation Experiment Configuration')

    # Logging arguments
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=lookup("WANDB_ENTITY"),
        help="Wandb entity (default: $WANDB_ENTITY, else your default entity)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ScoreDA_SQG",
        help="Wandb run project",
    )
    parser.add_argument('--exp_name', type=str, default='NEW_MODEL',
                        help='Experiment name prefix')
    parser.add_argument('--online_metrics', action='store_true',
                        help='Log scalar metrics live during the assimilation loop (into a wandb run '
                             'created up front) instead of only after the run. Media (spectra/video) is '
                             'still uploaded at the end. Default off: all logging happens after the run, '
                             'and assimilate() runs wandb-free.')
    parser.add_argument('--no_media_wandb', action='store_true',
                        help='Skip uploading the heavy animation video to wandb. Per-step spectrum '
                             'images and scalar metrics are still logged. Useful for large ablation '
                             'studies.')
    parser.add_argument('--device', type=torch.device, default=None,
                        help='Device to use (e.g. "cpu", "cuda", "mps"). If None, will use GPU if available.')

    # Method selection
    parser.add_argument("--method", type=str, default='DAISI',
                        help='Data assimilation method to use (e.g. "EnSF", "DAISI", "LETKF", '
                             '"FlowDAS")')

    # Method specific parameters
    # Backbone / UNET options
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Dimensionality of all hidden representations (default: 32)",
    )
    parser.add_argument(
        "--resample_filter",
        type=list_of_ints,
        default="1,3,3,1",
        help="Resample filter for backbone unet (default: 1,1 or 1,3,3,1)",
    )
    parser.add_argument(
        "--channel_mult",
        type=list_of_ints,
        default="2,2,2",
        help="Channel multiplier for backbone unet (depth and width of UNET) (default: 2,2,2,2)",
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="residual",
        help="Type of encoder to use in backbone unet (standard/residual/skip)"
        "(default: 'standard')",
    )

    parser.add_argument(
        "--attn_resolutions",
        type=list_of_ints,
        default="32",
        help="Resolutions to apply attention to in edm model (default: '1')",
    )
    parser.add_argument(
        "--sdl_res_dec",
        type=list_of_ints,
        default="1",
        help="Resolutions to apply sdl noise adapters to in unet model decoder (default: '1')",
    )
    parser.add_argument(
        "--sdl_res_enc",
        type=list_of_ints,
        default="1",
        help="Resolutions to apply sdl noise adapters to in unet model encoder (default: '1')",
    )

    parser.add_argument(
        "--noise_embedding",
        type=str,
        default="fourier",
        help="Type of encoder to use in edm model (positional/fourier/linear)"
        "(default: 'fourier')",
    )
    parser.add_argument(
        "--channel_mult_emb",
        type=float,
        default=4,
        help="Channel multiplier for noise embedding MLP (default: 4)",
    )
    parser.add_argument(
        "--channel_mult_noise",
        type=float,
        default=2,
        help="Channel multiplier for noise level MLP (default: 1)",
    )

    # No default: each method resolves its own checkpoint (assimilation/checkpoints.py).
    parser.add_argument('--model_path', type=str, default=None,
                        help='Trained checkpoint to load. Omit to use the '
                             'per-dataset default from '
                             'assimilation/checkpoints.py (overridable by the '
                             'env vars listed there and in .env).')
    parser.add_argument('--n_ens', type=int, default=20,
                        help='Number of ensemble members')
    parser.add_argument('--euler_steps', type=int, default=100,
                        help='Number of Euler steps for diffusion')
    parser.add_argument('--guide_method', type=str, default='MMPS',
                        help='Guidance method to use')
    parser.add_argument('--noise', type=str, default="invert",
                        help='Noise type (e.g. "invert", or None)')
    parser.add_argument('--eps', type=float, default=0.03,
                        help='Noise during sampler')
    parser.add_argument('--invert_eps', type=float, default=0.03,
                        help='Noise during inversion')
    parser.add_argument('--tmin', nargs="+", type=float, default=None,
                        help='Min time for diffusion')
    parser.add_argument('--tmax', nargs="+", type=float, default=None,
                        help='End time for diffusion')
    parser.add_argument('--guidance_strength', type=float, default=1.0,
                        help='Guidance strength')
    parser.add_argument('--corrections', type=int, default=0,
                        help='Number of corrections in each step')
    parser.add_argument('--tau', type=float, default=0.0001,
                        help='Stepsize for correction')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for processing')
    parser.add_argument('--invert_steps', type=int, default=250,
                        help='Number of steps for inversion')

    # DAWIS settings
    parser.add_argument(
        "--init_states",
        type=int,
        default=1,
        help="Number of initial states "
        "(default: 1)",
    )
    parser.add_argument('--fm_loss', type=str, default='eta01_channel',
                        choices=['eta0', 'eta1', 'eta01_label',
                                 'eta01_model', 'eta01_channel'],
                        help='FMW loss type. Selects the default checkpoint for DAWIS; only '
                             'eta01_channel is released, pass --model_path for others')
    parser.add_argument('--guide_all_steps', action='store_true',
                        help='DAWIS: apply observation guidance on all window steps (smoother), '
                             'not only the last step (filter)')
    parser.add_argument('--pyramid', action='store_true',
                        help='DAWIS: build the ForcingDAS-Pyr rolling --tmin/--tmax '
                             'schedule for fixed-lag smoothing')
    parser.add_argument('--n_fixed', type=int, default=0,
                        help='DAWIS Pyramid: how many leading window states are '
                             'pinned at clean data (tmin = tmax = 1)')
    parser.add_argument('--guide_forecast', action='store_true',
                        help='DAWIS: apply observation guidance during the FMW prediction step '
                             '(only used with --forward_model fmw).')
    parser.add_argument('--guide_stage1', action='store_true',
                        help='DAWIS: apply observation guidance during the FMW stage 1 '
                             '(noise-to-noise) step. Off by default; this was '
                             'unconditional before the flag existed, so runs logged '
                             'earlier than it are not comparable without it.')
    parser.add_argument('--guide_first', type=str, default='none', choices=['none', 'init', 'all'],
                            help='DAWIS: apply guidance on forecast for initial state in window')
    parser.add_argument('--save_window_states', action='store_true',
                        help='DAWIS: additionally save every state of every assimilation '
                             'window to the result file, as x_window(t, lag, ens, z, y, x) '
                             'with lag 0 = filter (== x_assim) and lag init_states = fully '
                             'smoothed (== x_smooth). Gives init_states+1 estimates per '
                             'physical time for post-hoc lag/time-lagged-ensemble analysis. '
                             'Off by default; '
                             'costs n_times*(init_states+1)*n_ens*2*nx^2 floats of disk.')
    parser.add_argument('--debug_dawis', action='store_true',
                        help='DAWIS: log per-window-state diagnostics (posterior-vs-truth RMSE '
                             'for every window step, and post-guidance obs misfit) each '
                             'assimilation step. Off by default; runs the unbatched path.')
    parser.add_argument(
        "--sampler",
        type=str,
        default="stochastic",
        help="The sampler to use when generating trajectories with a flow-based model",
    )
    parser.add_argument(
        "--sampler_eps",
        type=float,
        default=1e-3,
        help="Epsilon parameter for samplers during inference (default: 1e-3)",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="linear_scalar",
        help="The alpha, beta schedule to use (default: linear_scalar)"
    )
    parser.add_argument(
        "--corr_noise",
        type=float,
        default=0.,
        help="The correlation structure for the noise at diffusion time 0 (default: 0.)"
    )
    parser.add_argument(
        "--alpha_beta_spatial",
        action="store_true",
        help="If the alpha, beta values should be concatented as spatial input channels (default: False)"
    )
    parser.add_argument(
        "--alpha_beta_mult",
        type=int,
        default=None,
        help="Channel multiplier for alpha, beta channels (default: None -> will scale linearly with init_states)"
    )
    parser.add_argument(
        "--time_dropout",
        type=float,
        default=None,
        help="Dropout probability for the time dimension (default: None)"
    )
    parser.add_argument(
        "--fm_uncond",
        action="store_true",
        help="If we also want to train a unconditional FM model at the same time (default: False)"
    )
    parser.add_argument('--x0_sigma', type=float, default=3.0,
                        help='Guidance error standard deviation for the initial state (used in DAWIS)')
    parser.add_argument('--tmin_start', type=float, default=None,
                        help='Min time for diffusion at the start of the trajectory')
    parser.add_argument('--tmin_end', type=float, default=None,
                        help='Min time for diffusion at the end of the trajectory')
    parser.add_argument('--tmin_init', nargs="+", type=float, default=None,
                        help='DAWIS: per-slot tmin for the conditioning window while the '
                             'initial condition is still inside it. Length init_states '
                             '(conditioning slots only; the target always keeps tmin[-1]). '
                             'At step ntime the first ntime entries have expired and the '
                             'rest is left-aligned, so the last entry tracks the slot '
                             'holding the initial condition as it slides out of the window; '
                             'slots to its right keep the standard tmin. Standard tmin '
                             'everywhere from ntime >= init_states. Composes with either '
                             '--tmin or --tmin_start/--tmin_end. Default None = disabled.')

    # LETKF settings
    parser.add_argument('--hcovlocal_scale', type=float, default=1500.0*1000,
                        help='Horizontal covariance localization length scale in meters')
    parser.add_argument('--covinflate1', type=float, default=0.5,
                        help='Covariance inflation factor 1')
    parser.add_argument('--covinflate2', type=float, default=-1.0,
                        help='Covariance inflation factor 2')

    parser.add_argument('--upload_results', action='store_true', default=True,
                        help='After a smoothing run, log metrics/plots to wandb via '
                             'utils.upload_results(..., smoother=True). On by default, '
                             'matching the filter path in assimilate.py.')
    parser.add_argument('--no_upload_results', dest='upload_results',
                        action='store_false',
                        help='Skip the wandb upload after a smoothing run.')
    parser.add_argument('--forward_precision', type=str, default='single',
                        choices=['single', 'double'],
                        help='Float precision for --forward_model numerical_gpu. '
                             'Default "single" matches sqg.SQG so the two numerical '
                             'backends differ only in implementation.')

    # EnSF settings
    parser.add_argument('--eps_alpha', type=float, default=0.05,
                        help='Diffusion parameter for EnSF')
    parser.add_argument('--ensf_spread', type=float, default=None,
                        help='EnSF rescales its ensemble to this spread after '
                             'every analysis, in the run\'s units. Defaults to '
                             '--init_std, which is what it always was; with '
                             'init_std 0 (SEVIR) it defaults to the spread of '
                             'the first forecast, before the first analysis.')

    # FlowDAS settings
    # --euler_steps is FlowDAS's EM_sample_steps, --guidance_strength its grad_scale.
    parser.add_argument('--mc_times', type=int, default=1,
                        help="FlowDAS: number of endpoint (x1) estimates averaged in the "
                             "guidance term (the reference's MC_times). Cheap -- the two "
                             "drift evaluations are shared between them.")
    parser.add_argument('--flowdas_tmin', type=float, default=0.001,
                        help='FlowDAS: start of the interpolant sampling interval '
                             "(the reference's t_min_sampling).")
    parser.add_argument('--flowdas_tmax', type=float, default=0.999,
                        help='FlowDAS: end of the interpolant sampling interval '
                             "(the reference's t_max_sampling).")

    # SDA settings
    parser.add_argument('--window_batch_size', type=int, default=5,
                        help='SDA: overlapping trajectory windows per GPU call. '
                             'MULTIPLIES --batch_size (ensemble members per '
                             'call), so the tensor reaching the network is '
                             'window_batch_size * batch_size windows (default: 5)')

    # Gibbs block smoother settings
    parser.add_argument(
        "--gibbs_sweeps",
        type=int,
        default=1,
        help="Number of outer Gibbs sweeps for block smoothing.",
    )
    parser.add_argument(
        "--block_mode",
        type=str,
        default="sliding",
        choices=["sliding", "non_overlapping", "random", "custom"],
        help="How to traverse blocks during each Gibbs sweep.",
    )
    parser.add_argument(
        "--block_direction",
        type=str,
        default="forward",
        choices=["forward", "backward", "both", "alternate"],
        help="Sweep direction for the selected blocks.",
    )
    parser.add_argument(
        "--block_stride",
        type=int,
        default=None,
        help="Optional stride between block centers.",
    )
    parser.add_argument(
        "--random_block_count",
        type=int,
        default=None,
        help="Number of random blocks to sample per sweep when block_mode=random.",
    )
    parser.add_argument(
        "--random_block_replace",
        action="store_true",
        help="Sample random blocks with replacement.",
    )
    parser.add_argument(
        "--gibbs_endpoint_tmin",
        type=float,
        default=1.0,
        help="tmin used for the endpoints of each Gibbs block during the noising step.",
    )
    parser.add_argument(
        "--gibbs_midpoint_tmin",
        type=float,
        default=0.0,
        help="tmin used for the midpoints of each Gibbs block during the noising step.",
    )
    parser.add_argument(
        "--gibbs_num_endpoints",
        type=int,
        default=1,
        help="Number of conditioning steps held at gibbs_endpoint_tmin on each "
             "side of a Gibbs block. 1 is the Markovian case; use k>1 when the "
             "system needs k past/future states to be conditionally "
             "independent of the rest of the trajectory. Requires "
             "2*k < block_length (= init_states + 1).",
    )
    parser.add_argument(
        "--save_sweeps",
        action="store_true",
        help="Gibbs: after every sweep, save that sweep's trajectory into the "
             "result file as x_gibbs_1, x_gibbs_2, ... (same layout as "
             "x_smooth) and log its metrics to its own wandb run "
             "smoother_{exp_name}_gibbs_{k}. Adds one x_smooth-sized variable "
             "per sweep to the .nc.",
    )
    parser.add_argument(
        "--sweep_metrics",
        action="store_true",
        help="Gibbs: log the per-sweep wandb runs without writing the (large) "
             "per-sweep x_gibbs_k trajectories to the result file.",
    )

    # General System settings
    # Data is selected by --dataset/--variant/--split; --data_path overrides with a file.
    add_dataset_args(parser)
    parser.add_argument('--split', type=str, default=None,
                        help='Dataset split to assimilate (default: test)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Explicit path to a trajectory file, overriding '
                             '--variant/--split')
    parser.add_argument('--standardization_path', type=str, default=None,
                        help='Deprecated: statistics come from the data source.')
    parser.add_argument('--n_times', type=int, default=10,
                        help='Number of assimilation times to run')
    parser.add_argument('--trajectory_path', type=str, default=None,
                        help='Optional path to an initial trajectory for smoothing (.npy, .pt, or .pth)')
    parser.add_argument('--trajectory_var', type=str, default='auto',
                        help='Trajectory variable to load from .nc files (auto, x_smooth, x_assim)')
    parser.add_argument('--initial_trajectory', type=str, default=None,
                        help='Method name of an archived paper run to start the '
                             'smoother from, e.g. DAWISnumerical. The run is looked '
                             'up in the paper-run directory (PAPER_RUNS_ROOT) as '
                             '<method>_<experiment>_run_<data_index>_*.nc, using '
                             '--experiment and --data_index. Ignored when '
                             '--trajectory_path is given.')
    parser.add_argument('--assim_interval', type=int, default=1,
                        help='Observations arrive every k-th step (1 = every '
                             'step, the default; -1 = never). A step without '
                             'observations is observed NOWHERE rather than '
                             'observed as zero: methods with a separate forward '
                             'model skip the analysis, methods whose analysis is '
                             'also their forecast run it against an empty '
                             'observation mask, and the smoothers drop that '
                             "step's term from their cost. See "
                             'assimilation/obs_schedule.py.')
    parser.add_argument('--obs_prob', type=float, default=0.05,
                        help='Percentage of grid points to observe')
    parser.add_argument('--fixed_obs', action='store_true',
                        help='Use fixed observation locations')
    parser.add_argument('--init_std', type=float, default=1000.0,
                        help='Initial ensemble perturbation standard deviation')
    parser.add_argument('--obs_sigma', type=float, default=3.0,
                        help='Observation error standard deviation')
    parser.add_argument('--obs_fn', type=str, default='linear',
                        help='Observation function (e.g. "linear", "arctan")')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment setting from named pre-configured experiment')
    parser.add_argument('--init_state', type=str, default='GT', choices=['GT', 'clim', 'None', 'GT_guide', 'GT_edit'],
                        help='Initial state for the ensemble ("GT", "clim" or "None")')
    parser.add_argument('--forward_model_path', type=str, default=None,
                        help='Checkpoint for --forward_model unet / fmw / learned. '
                             'Its architecture is read from the checkpoint itself, '
                             'so no --hidden_dim / --channel_mult / ... needs to '
                             'accompany it.')
    parser.add_argument('--forward_model_init_states', type=int, default=None,
                        help='How many states the propagator checkpoint conditions '
                             'on. Only a FALLBACK now: a Lightning checkpoint '
                             'records this itself, and the recorded value wins. It '
                             'is still needed for a bare state_dict, which records '
                             'nothing. Note that this is independent of '
                             '--init_states -- a 6-state DAWIS window model can be '
                             'driven by a 1-state U-Net.')
    parser.add_argument('--learned_model', type=str, default=None,
                        choices=['fmw', 'unet'],
                        help='Which architecture --forward_model_path holds. Only a '
                             'FALLBACK: the checkpoint records its own model, and '
                             'that wins (a disagreement is reported and ignored). '
                             'Use --forward_model unet / fmw instead.')
    parser.add_argument('--forward_model', type=str, default='numerical',
                        choices=['numerical', 'numerical_gpu', 'flowdas', 'none',
                                 'unet', 'fmw', 'dawis', 'learned'],
                        help='What propagates the ensemble between analyses. '
                             '"numerical"/"numerical_gpu" is the SQG solver; '
                             '"flowdas" is FlowDAS\'s pretrained interpolant (6 '
                             'conditioning frames); "unet" is a deterministic U-Net '
                             'checkpoint and "fmw" a flow-matching window model, '
                             'both from --forward_model_path and usable by any '
                             'method; "none" is no forecast at all. "dawis" means no '
                             'separate forecast: the DAWIS window model generates '
                             't+1 from pure noise during its own stage 1 (requires '
                             '--method DAWIS). "fmw" without --forward_model_path '
                             'reuses the assimilation model itself, which only DAWIS '
                             'and SDA can do. "learned" is the old spelling of '
                             'unet/fmw and still works.')
    parser.add_argument('--ignore_ckpt_arch', action='store_true',
                        help='Do not read the architecture out of the checkpoints; '
                             'use the --hidden_dim / --channel_mult / ... flags as '
                             'given. For a checkpoint whose recorded architecture is '
                             'wrong. Expect load_state_dict to fail if it is not.')
    # TODO: merge initial_states and window.
    parser.add_argument('--window', type=int, default=1,
                        help='Depth of the conditioning buffer. Derived from the '
                             'forward model (1 for numerical/U-Net, 6 for FlowDAS, '
                             'the checkpoint\'s init_states for a window model) '
                             'unless the method pins it; pass it only to override.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed numpy/torch RNGs. The initial ensemble perturbation is '
                             'otherwise unseeded, so set this to compare two runs that '
                             'should differ only in one setting (e.g. --forward_model '
                             'numerical vs numerical_gpu). Default None keeps the '
                             'previous, unseeded behaviour.')
    parser.add_argument('--data_index', type=int, default=0,
                        help='Index of the data sample to use for assimilation')
    parser.add_argument('--start_time', type=int, default=6,  # leaves room for the FlowDAS conditioning window
                        help='Start time for assimilation')
    parser.add_argument('--lres', type=int, default=None,
                        help='Resolution for avg obs')
    parser.add_argument('--radar', type=bool, default=False,
                        help='Use radar obs')
    parser.add_argument('--radar_width', type=int, default=4,
                        help='Width of radar observations')
    parser.add_argument('--radar_dx', type=int, default=4,
                        help='Spacing between radar observations')
    # SQG System settings
    # TODO: infer from the data.
    parser.add_argument('--hrly', type=int, default=3, choices=[3, 12],
                        help='Time interval: 3hrly or 12hrly')
    # Filled from dataset metadata (assimilation/case.py); passing it asserts the grid.
    parser.add_argument('--nx', type=int, default=None,
                        help='Grid resolution. Defaults to the dataset grid; '
                             'give it only to assert that resolution.')

    # Evaluation settings
    parser.add_argument('--plot_every', type=int, default=20,
                        help='Plotting frequency (e.g. every N steps)')

    return parser


def parse_args(args=None):
    """
    Parse command line arguments.

    Returns:
        args: Parsed arguments.
    """
    return build_parser().parse_args(args)


def arch_defaults(keys=None):
    """`{flag: default}` for the architecture flags.

    Used by `forecasting.ckpt_args.merge_arch` to tell user-set values from
    defaults. Parses an empty argv (not `get_default`) so `list_of_ints`
    defaults come back converted.
    """
    from forecasting.ckpt_args import ARCH_KEYS

    defaults = build_parser().parse_args([])
    return {key: getattr(defaults, key, None) for key in (keys or ARCH_KEYS)}

