import os
import uuid
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from netCDF4 import Dataset as NetCDFDataset

# First-party
from utils import upload_results, compute_step_metrics, wandb_config

from assimilation.case import (
    assim_scale,
    initial_conditioning,
    load_assim_case,
    normalization_profile,
)
from assimilation.checkpoints import resolve_prior
from assimilation.registry import build_observer as _build_observer_registry
from assimilation.obs_schedule import (
    blank_observation, is_assimilation_step)
from assimilation.registry import (
    LEARNED_FORWARD_MODELS,
    build_forward_model,
    forward_model_frames,
    require_numerical_model,
)
from forecasting.ckpt_args import warn

from assimilation.methods.ensf import EnSF
from assimilation.methods.letkf import LETKF
from assimilation.methods.daisi import DAISI
from assimilation.methods.dawis import DAWIS
from assimilation.methods.flowdas import FlowDAS
from forecasting.sqg_model import SQGModel, SQGModelGPU

from assimilation.observers.observer import GridObserver
from assimilation.experiments import apply_experiment
from assimilation.parser import parse_args
from data.paths import RESULTS_FILE_MODE, result_file, share_path


def _sync_dawis_window(args):
    """For DAWIS, set `window` to `init_states` (the backbone requires them equal)."""
    if args.method == 'DAWIS' and args.window != args.init_states:
        print(
            f"DAWIS: forcing window {args.window} -> init_states {args.init_states}")
        args.window = args.init_states
    return args


#: Forward models each method refuses, with the reason (None = refuses all).
#: Methods not listed accept every forward model.
_FORWARD_MODEL_LIMITS = {
    'SDA': (None, "SDA denoises the whole window and takes no forecast"),
    'GIBBS': (None, "Gibbs denoises the whole window and takes no forecast"),
}


def check_forward_model_supported(args):
    """Raise early (before loading checkpoints) for unsupported method/forward-model pairs."""
    name = getattr(args, 'forward_model', 'none')
    method = str(args.method).upper()

    # `fmw` without a checkpoint means the method is its own forward model.
    if name == 'fmw' and not getattr(args, 'forward_model_path', None):
        if args.method != 'DAWIS':
            raise ValueError(
                f"--forward_model fmw without --forward_model_path means the "
                f"assimilation model is its own forward model, which only DAWIS "
                f"can do (it is the only method defining `forecast()`); "
                f"--method {args.method} cannot.\n"
                f"  Give --forward_model_path <FMW checkpoint> to propagate with a "
                f"window model, or use --forward_model unet / flowdas / none.")

    if name == 'none' or method not in _FORWARD_MODEL_LIMITS:
        return
    refused, why = _FORWARD_MODEL_LIMITS[method]
    if refused is None or name in refused:
        accepted = (sorted({'unet', 'fmw', 'flowdas'} - refused)
                    if refused else [])
        raise ValueError(
            f"--method {args.method} cannot use --forward_model {name}: {why}.\n"
            + (f"  Use --forward_model {' or '.join(accepted)}.\n"
               if accepted else "  Use --forward_model none.\n"))


def _resolve_window(args, forward_model, window, start_time):
    """Size the conditioning buffer (`window`) for the propagator.

    The buffer is deepened to the propagator's conditioning depth (1 for the
    U-Net, 6 for FlowDAS), and `start_time` raised to match. For DAWIS the buffer
    is its own posterior window (`--init_states`), so a propagator needing more
    frames is an error. Returns (window, start_time).
    """
    required = forward_model_frames(forward_model)

    if args.method == 'DAWIS' and window != required:
        if window < required:
            raise ValueError(
                f"--forward_model {args.forward_model} conditions on {required} "
                f"frames but --init_states is {window}.\n"
                f"  DAWIS replaces the conditioning buffer with its own "
                f"{window}-state posterior window every cycle, so a deeper buffer "
                f"would not survive the first step.\n"
                f"  Use --init_states {required} (with a matching window-model "
                f"checkpoint), or a propagator that conditions on {window} frames "
                f"or fewer.")
        print(f"window {window} (--init_states), of which the "
              f"{args.forward_model} propagator uses the newest {required}",
              flush=True)
        return window, start_time

    if window != required:
        if window != 1:  # 1 is the parser default
            warn(f"--window {window} replaced by {required}, which is what the "
                 f"{args.forward_model} checkpoint conditions on.")
        else:
            print(f"window {required} (from the {args.forward_model} propagator)",
                  flush=True)
        window = required
    # The buffer is filled from the states before `start_time`.
    return window, max(start_time, window)


def _zero_init_std_for_flowdas(args):
    """Set `init_std` to 0 for FlowDAS, which conditions on the clean GT window.

    The perturbation is unused by FlowDAS anyway; this keeps the recorded config
    accurate. Must run after the experiment preset is applied.
    """
    if args.method == 'FlowDAS' and args.init_std != 0:
        print(
            f"FlowDAS: forcing init_std {args.init_std} -> 0 "
            "(conditioning is the clean GT window)")
        args.init_std = 0.0
    return args


def pyramid_schedule(T, n_fixed=0):
    """Steady-state rolling ("pyramid") (tmin, tmax) schedule for T = init_states + 1 states.

    The ForcingDAS-Pyr schedule (Jia et al., 2026) applied to the DAWIS window prior.

    Each cycle advances every state one rung of a noise ladder, from pure noise
    (newest) to clean (oldest, emitted as the smoothed estimate). The first
    `n_fixed` states are pinned clean, trading smoothing lag for clean context
    (n_fixed = init_states is a filter):

        tmin[j] = 1                    for j <  n_fixed
        tmin[j] = (T-1-j)/(T-n_fixed)  for j >= n_fixed
        tmax[j] = tmin[j-1]            (tmax[0] = 1)

    Returns:
        (tmin, tmax), plain float lists of length T.
    """
    if T < 2:
        raise ValueError(f"pyramid_schedule needs T >= 2; got {T}.")
    k = int(n_fixed)
    if not 0 <= k <= T - 1:
        raise ValueError(
            f"n_fixed must be in [0, init_states] = [0, {T-1}]; got {k}. "
            "n_fixed == init_states leaves a single ladder state, i.e. a filter.")
    tmin = [1.0] * k + [(T - 1 - j) / (T - k) for j in range(k, T)]
    tmax = [1.0] + tmin[:-1]
    return tmin, tmax


def pyramid_spinup_schedule(tmin, tmax, n_pinned):
    """Spin-up (tmin, tmax): the steady schedule with the first `n_pinned` slots held clean.

    Early slots still hold clean initialization states. The caller uses
    n_pinned = max(n_fixed, init_states - ntime); once it reaches n_fixed this
    returns the steady schedule unchanged.

    Returns:
        (tmin_eff, tmax_eff), plain float lists of length len(tmin).
    """
    tmin_eff = [1.0 if j < n_pinned else float(t) for j, t in enumerate(tmin)]
    tmax_eff = [1.0 if j < n_pinned else float(t) for j, t in enumerate(tmax)]
    return tmin_eff, tmax_eff


def pyramid_spindown_schedule(tmin, done):
    """Spin-down (tmin, tmax) for pass `done` at the end of a run.

    The window no longer rolls; each pass moves state j from tmin[j-done] to
    tmin[j-done-1] (negative indices = clean), finishing one more state.

    Returns:
        (tmin_eff, tmax_eff), plain float lists of length len(tmin).
    """
    T = len(tmin)

    def rung(i):
        return 1.0 if i < 0 else float(tmin[i])

    tmin_eff = [rung(j - done) for j in range(T)]
    tmax_eff = [rung(j - done - 1) for j in range(T)]
    return tmin_eff, tmax_eff


def build_observer(args, device, md=None):
    """Build (observer, obs_fn) for a run via assimilation.registry."""
    if md is None:
        from data.registry import data_source_from_args
        md = data_source_from_args(args).metadata
    return _build_observer_registry(args, md, device)


def set_seed(args):
    """Seed numpy/torch if --seed was given; otherwise leave RNGs untouched.

    By default only the observer is seeded, so repeated runs differ.
    """
    seed = getattr(args, 'seed', None)
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if args.device is None:
        args.device = device
    else:
        device = args.device
    return device


def _legacy_md_from_args(args):
    """Dataset metadata from `args`, or a warned synthetic stand-in if no dataset is set (tests)."""
    import warnings
    from types import SimpleNamespace

    from data.registry import data_source_from_args

    if getattr(args, "dataset", None):
        return data_source_from_args(args).metadata

    nx = getattr(args, "nx", 64) or 64
    warnings.warn(
        f"args names no dataset; falling back to a {nx}x{nx} 2-channel "
        f"stand-in with unit_factor=1.0. A result file written from this "
        f"metadata carries a placeholder nc.scalefact and must not be scored "
        f"with utils.upload_results.",
        RuntimeWarning,
        stacklevel=2,
    )
    return SimpleNamespace(
        grid=(nx, nx), num_channels=2, unit_factor=1.0,
        describe=lambda: {},
        # No normalization: output is written unchanged.
        output_affine=lambda _norm: (1.0, 0.0),
    )


class _OutputVar:
    """A netCDF variable that applies the `--output_norm` affine on write (reads are unconverted)."""

    __slots__ = ("_var", "_a", "_b")

    def __init__(self, var, a, b):
        self._var = var
        self._a = float(a)
        self._b = float(b)

    def __setitem__(self, key, value):
        self._var[key] = np.asarray(value) * self._a + self._b

    def __getitem__(self, key):
        return self._var[key]

    def __getattr__(self, name):
        # Avoid infinite recursion if slots are unset (e.g. during copy/unpickle).
        if name in _OutputVar.__slots__:
            raise AttributeError(name)
        return getattr(self._var, name)


def _wrap_output(var, md, output_norm, *, raw_units=False):
    """Wrap `var` to convert assimilation units to `output_norm` on write.

    `raw_units=True` is for `ground_truth`, stored in raw model units and read as
    `scalefact * nc['ground_truth']`.
    """
    a, b = md.output_affine(output_norm)
    if raw_units:
        u = float(md.unit_factor)
        a, b = a, b / u
    if a == 1.0 and b == 0.0:
        return var
    return _OutputVar(var, a, b)


def create_nc_file(args, md=None):
    """Create the result netCDF. `md` defaults to metadata resolved from `args`."""
    if md is None:
        md = _legacy_md_from_args(args)
    # Group-readable results file (see data/paths.py).
    nc_filename = result_file(args.exp_name, create_dir=True)
    nc = NetCDFDataset(str(nc_filename), mode='w', format='NETCDF4_CLASSIC')
    share_path(nc_filename, RESULTS_FILE_MODE)

    ny, nx = md.grid
    nc.createDimension('x', nx)
    nc.createDimension('y', ny)
    # Channel dimension is named 'z' for compatibility with readers.
    nc.createDimension('z', md.num_channels)
    nc.createDimension('t', None)
    nc.createDimension('ens', args.n_ens)
    ground_truth = nc.createVariable(
        'ground_truth', np.float32, ('t', 'z', 'y', 'x'), zlib=True)
    x_assim = nc.createVariable(
        'x_assim', np.float32, ('t', 'ens', 'z', 'y', 'x'), zlib=True)
    x_smooth = nc.createVariable(
        'x_smooth', np.float32, ('t', 'ens', 'z', 'y', 'x'), zlib=True)

    nc.scalefact = md.unit_factor
    # Record dataset metadata so result files are self-describing.
    for key, value in md.describe().items():
        setattr(nc, key, value if not isinstance(value, (list, tuple))
                else ' '.join(map(str, value)))

    # Units of the stored values; readers use this to denormalize. Missing
    # means assimilation units.
    output_norm = getattr(args, 'output_norm', None)
    nc.output_norm = str(output_norm) if output_norm else 'assim'
    a, b = md.output_affine(output_norm)
    nc.output_affine = f"out = {a:.10g} * assim + {b:.10g}"
    print(f"Results in '{nc.output_norm}' units: {nc.output_affine}", flush=True)

    return (nc,
            _wrap_output(ground_truth, md, output_norm, raw_units=True),
            _wrap_output(x_assim, md, output_norm),
            _wrap_output(x_smooth, md, output_norm))


def add_window_variables(nc, args, md=None):
    """
    Add the optional all-window-states variables (``--save_window_states``).

    Layout is physical time x lag:

        x_window[t, lag]  = estimate of physical time t produced at
                            assimilation step t + lag

    so that ``lag = 0`` is the filter (identical to ``x_assim[t]``) and
    ``lag = init_states`` is the fully smoothed state (identical to
    ``x_smooth[t]``, for the t where x_smooth is actually fully smoothed).
    ``window_valid[t, lag]`` marks which entries were written: the last
    ``init_states`` times cannot reach every lag, since lag ``L`` at time ``t``
    is only produced if step ``t + L`` is inside the run.

    Returns:
        (x_window, window_valid), or (None, None) if the flag is off.
    """
    if not getattr(args, 'save_window_states', False):
        return None, None

    if md is None:
        md = _legacy_md_from_args(args)
    ny, nx = md.grid
    n_ch = md.num_channels
    n_lags = args.init_states + 1
    nc.createDimension('lag', n_lags)
    # One chunk per (t, lag) slice, matching how the loop writes.
    x_window = nc.createVariable(
        'x_window', np.float32, ('t', 'lag', 'ens', 'z', 'y', 'x'), zlib=True,
        chunksizes=(1, 1, args.n_ens, n_ch, ny, nx))
    window_valid = nc.createVariable(
        'window_valid', np.int8, ('t', 'lag'), zlib=True, fill_value=0)

    # Metadata the analysis notebook needs to interpret the lag axis.
    nc.init_states = args.init_states
    nc.start_time = args.start_time
    nc.n_times = args.n_times
    nc.n_ens = args.n_ens
    nc.method = args.method
    nc.obs_fn = args.obs_fn
    nc.obs_prob = args.obs_prob
    nc.obs_sigma = args.obs_sigma
    nc.exp_name = args.exp_name
    # window_valid is a mask and is not unit-converted.
    return (_wrap_output(x_window, md, getattr(args, 'output_norm', None)),
            window_valid)


def assimilate(args, guidance_strength=None, eps=None, debug=False, run=None):
    """
    Data assimilation function.

    Args:
        args: arguments for the experiment
        guidance: optional guidance function
        eps: optional epsilon function
        run: optional wandb run. If provided, scalar metrics are logged live
            during the assimilation loop. If None (default), no logging is
            performed and the function runs wandb-free.

    Returns:
        exp_name: name of the experiment (for logging purposes)
    """
    set_seed(args)
    device = resolve_device(args)

    if args.tmin_start is not None and args.tmin_end is not None:
        args.tmin = torch.linspace(args.tmin_start, args.tmin_end,
                                   args.init_states+1, device=device)
        print(f"Overriding tmin with linspace: {args.tmin}")

    _sync_dawis_window(args)
    _zero_init_std_for_flowdas(args)

    if getattr(args, 'save_window_states', False) and args.method != 'DAWIS':
        raise ValueError(
            "--save_window_states needs --method DAWIS: only DAWIS returns the "
            f"posterior over the whole window, but method is '{args.method}'.")

    pyramid = bool(getattr(args, 'pyramid', False))
    if pyramid:
        # The pyramid is a DAWIS (tmin, tmax) schedule plus spin-up/spin-down
        # handling in the loop below.
        if args.method != 'DAWIS':
            raise ValueError(
                "--pyramid builds a DAWIS tmin/tmax schedule; got "
                f"--method {args.method}.")
        # Every cycle must run the analysis to advance the ladder; with a separate
        # propagator, unobserved steps would skip it.
        if args.assim_interval != 1 and args.forward_model not in ('none', 'dawis'):
            raise ValueError(
                "--pyramid with --assim_interval "
                f"{args.assim_interval} needs a method that runs its analysis on "
                "every step, because every cycle advances the window one rung of "
                "the ladder. With --forward_model "
                f"'{args.forward_model}' an unobserved step skips the analysis "
                "entirely and the ladder stalls. Use --forward_model none/dawis, "
                "or --assim_interval 1.")
        if getattr(args, 'save_window_states', False):
            print(
                "--save_window_states is not meaningful with --pyramid: only slot 0 "
                "of the posterior window is clean (it is x_smooth), the other lags "
                "are partially denoised latents at beta = tmax[j], so the "
                "x_window(t, lag) lag semantics do not hold.")
        if getattr(args, 'guide_first', 'none') != 'none':
            raise ValueError(
                "--guide_first is not supported with --pyramid: it prepends an "
                "observation for time -1 (which misaligns the window's obs history) "
                "and, for 'all', re-observes conditioning[0], which under a rolling "
                "schedule is a noisy latent rather than an analysis. Use "
                f"--guide_first none; got '{args.guide_first}'.")
        if getattr(args, 'tmin_init', None) is not None:
            raise ValueError(
                "--tmin_init is not supported with --pyramid: it rewrites tmin for "
                "the leading states during the first init_states steps, but the "
                "carried latents actually sit at the ladder's tmin[j], so the "
                "forward phase would start from a beta that does not match the "
                "noise already in the state. The ladder is fixed "
                f"instead; got tmin_init={args.tmin_init}.")
        if not args.guide_all_steps:
            raise ValueError(
                "--pyramid needs --guide_all_steps: the whole point is that every "
                "state in the window is refined against its own observations over "
                "several cycles, and the spin-up relies on the windowed obs layout "
                "(the current observation lands on an interior slot, not the last "
                "one). Without it only the newest slot would ever be guided.")
        if getattr(args, 'corrections', 0):
            raise ValueError(
                "--corrections is not supported with --pyramid: a state pinned at "
                "clean data (which --n_fixed creates, and which the drain creates "
                "as each state finishes) has alpha == 0, so the Langevin score "
                "-1/alpha used by the correction step is infinite. Use "
                f"--corrections 0; got {args.corrections}.")
        if args.tmax is not None or args.tmin is not None: 
            print("--pyramid overrides the given --tmin/--tmax with the pyramid "
                  "ladder.")
        args.tmin, args.tmax = pyramid_schedule(
            args.init_states + 1, args.n_fixed)
        print('tmin,tmax for pyramid schedule:', args.tmin, args.tmax)

        # The newest state enters as pure noise and the window is carried as
        # noisy latents: no forecast, nothing to invert.
        if args.forward_model not in (None, 'None', 'none'):
            warn(f"--pyramid: forcing --forward_model {args.forward_model} -> none "
                 "(the newest window state is generated from pure noise in-place)")
            args.forward_model = 'none'
        if args.noise not in (None, 'None'):
            warn(f"--pyramid: forcing --noise {args.noise} -> None (the window is "
                 "carried as noisy latents, so there is nothing to invert)")
            args.noise = None

        if args.start_time < args.init_states:
            print(f"--pyramid: raising --start_time {args.start_time} -> "
                  f"{args.init_states} (the ladder is seeded from a full "
                  "conditioning window)")
            args.start_time = args.init_states

        if args.n_times <= args.init_states:
            raise ValueError(
                f"--pyramid needs --n_times > init_states = {args.init_states}: "
                "the window has to roll clear of the initialization states, and the "
                "spin-down starts from the first smoothed state the loop emits; got "
                f"{args.n_times}.")
        print(f"--pyramid: n_fixed={args.n_fixed}, smoothing lag="
              f"{args.init_states - args.n_fixed}, spin-down passes="
              f"{args.init_states - args.n_fixed}; the ladder fills up over the "
              f"first {args.init_states - args.n_fixed} rolling cycle(s)")
        print(f"--pyramid: tmin={['%.3f' % t for t in args.tmin]} "
              f"tmax={['%.3f' % t for t in args.tmax]}")

    print(f"Using device: {device}", flush=True)

    # Data source, trajectory and numerical-model parameters.
    src, md, case = load_assim_case(args)
    dataset = case.states
    if normalization_profile(args):
        print("Using DAWIS, loading data statistics used during training.")
        src.require_per_channel_stats()

    ny, nx = md.grid

    start_time = args.start_time
    window = args.window  # 1 if regular forward model, 6 if flowDAS

    # Initialize observer
    observer, obs_fn = build_observer(args, device, md)

    # Result file
    nc, ground_truth, x_assim, x_smooth = create_nc_file(args, md)
    x_window, window_valid = add_window_variables(nc, args, md)
    # Assimilation -> output units, for metrics and console output.
    out_a, out_b = md.output_affine(getattr(args, 'output_norm', None))

    scale, init_sigma = assim_scale(md, args)

    if args.guide_method == 'MMPS' and args.obs_fn == 'arctan':
        def GUIDANCE(t): return args.guidance_strength * \
            (1-t) if guidance_strength is None else guidance_strength(t)
    else:
        def GUIDANCE(t):
            return args.guidance_strength * torch.ones_like(t) if guidance_strength is None else guidance_strength(t)

    def EPS(t): return args.eps * (1 - t) if eps is None else eps(t)

    # Only DAWIS returns the posterior window.
    return_traj = False

    obs_history = []
    obs_mask_history = []
    obs_fn_history = []
    obs_sigma_history = []

    def x0_obs_fn(x): return x
    x0_observer = GridObserver(
        obs_fun=x0_obs_fn,
        obs_prob=1.0,
        obs_mask=None,
        obs_sigma=init_sigma,
        stationary_obs=True,
        random_seed=42,
        nx=nx,
        ny=ny,
        device=device,
        radar=False,
        num_channels=md.num_channels,
        unit_factor=md.unit_factor,
    )

    x0_0bs_list = []
    for _ in range(args.n_ens):
        x0_obs_k, x0_mask, x0_sigma = x0_observer.observe(
            dataset[start_time-1], t=0)
        x0_0bs_list.append(x0_obs_k)
    x0_obs = torch.cat(x0_0bs_list, dim=0)

    xprev_observer = GridObserver(
        obs_fun=x0_obs_fn,
        obs_prob=1.0,
        obs_mask=None,
        obs_sigma=0,
        stationary_obs=True,
        random_seed=42,
        nx=nx,
        ny=ny,
        device=device,
        radar=False,
        num_channels=md.num_channels,
        unit_factor=md.unit_factor,
        )

    print('Shape of x0 observation:', x0_obs.shape, flush=True)

    if args.method == 'DAWIS' and (args.guide_first == 'init' or args.guide_first == 'all'):
        obs_history.append(x0_obs.to(device))
        obs_mask_history.append(x0_mask)
        obs_fn_history.append(x0_obs_fn)
        # Use init_sigma, falling back to --x0_sigma when it is 0 (clean GT start).
        obs_sigma_history.append(
            x0_sigma if float(x0_sigma) > 0 else args.x0_sigma)

    # Initialize model
    if args.method == 'LETKF':
        args.batch_size = args.n_ens
        require_numerical_model(args, md, case, 'LETKF')
        assimilation_model = LETKF(
            args=args, nobs=observer.nobs, model_params=case.model_params,
            geometry=case.grid_geometry, unit_factor=md.unit_factor,
            num_channels=md.num_channels)
    elif args.method == 'DAISI':
        # Unconditional prior, chosen per dataset/variant.
        args.model_path = resolve_prior(md, args)
        print(f"DAISI prior: {args.model_path}", flush=True)

        assimilation_model = DAISI(
            args=args,
            obs_sigma=args.obs_sigma,
            device=device,
            members=args.batch_size if args.batch_size is not None else args.n_ens,
            beta=lambda t: t,
            alpha=lambda t: 1 - t,
            beta_dot=lambda t: 1,
            alpha_dot=lambda t: -1,
            eps=EPS,
            guide_method=args.guide_method,
            guidance_strength=GUIDANCE,
            steps=args.euler_steps,
            noise=args.noise,
            scale=scale,
            metadata=md,
        )

    elif args.method == 'DAWIS':
        args.obs_sigma = args.obs_sigma
        args.guide_all_steps = getattr(args, 'guide_all_steps', False)
        args.sampler_steps = args.euler_steps
        args.n_ens = args.n_ens
        assimilation_model = DAWIS(args, metadata=md)
        return_traj = True

    elif args.method == 'EnSF':
        ISarctan = True if args.obs_fn == 'arctan' else False
        assimilation_model = EnSF(
            nx=nx,
            ensemble_size=args.n_ens,
            eps_alpha=args.eps_alpha,
            device=device,
            obs_sigma=args.obs_sigma,
            steps=args.euler_steps,
            scalefact=md.unit_factor,
            ny=md.ny,
            num_channels=md.num_channels,
            # Spread EnSF resets to after each analysis: --ensf_spread, else
            # init_std (0 -> None, the first forecast's spread).
            init_std_x_state=(args.ensf_spread if args.ensf_spread is not None
                              else (args.init_std if args.init_std > 0
                                    else None)),
            obs_fn=args.obs_fn,
            guidance_strength=args.guidance_strength,
            ISarctan=ISarctan
        )
    elif args.method == 'FlowDAS':
        # Forecast and analysis in one guided pass: --euler_steps is
        # EM_sample_steps, --guidance_strength is grad_scale.
        assimilation_model = FlowDAS(
            args=args,
            obs_sigma=args.obs_sigma,
            device=device,
            members=args.batch_size if args.batch_size is not None else args.n_ens,
            metadata=md,
            steps=args.euler_steps,
            # Constant grad_scale at every EM step.
            guidance_strength=args.guidance_strength,
            mc_times=args.mc_times,
            t_min=args.flowdas_tmin,
            t_max=args.flowdas_tmax,
        )
    else:
        raise NotImplementedError(
            f"Data assimilation method '{args.method}' not implemented.")

    if args.method == 'FlowDAS' and args.forward_model != 'none':
        raise ValueError(
            "--method FlowDAS needs --forward_model none: FlowDAS produces the "
            "forecast and the analysis in a single guided sampling pass. "
            f"Got --forward_model {args.forward_model}. (--forward_model flowdas "
            "is the *unguided* forecaster, for use with another method.)")

    check_forward_model_supported(args)

    if args.forward_model in LEARNED_FORWARD_MODELS or args.forward_model == 'flowdas':
        # Dataset-keyed via the registry.
        forward_model = build_forward_model(args, md, case, None)
        if forward_model is None:
            # `fmw` without a path: the assimilation model is the forward model.
            forward_model = assimilation_model
        else:
            window, start_time = _resolve_window(
                args, forward_model, window, start_time)
        conditioning = initial_conditioning(
            case, start_time, window, args.n_ens, md.unit_factor)
    elif args.forward_model in ('numerical', 'numerical_gpu'):
        require_numerical_model(args, md, case, args.forward_model)
        model_cls = SQGModel if args.forward_model == 'numerical' else SQGModelGPU
        forward_model = model_cls(args, np.expand_dims(
            dataset[start_time], axis=0).repeat(args.n_ens, axis=0),
            model_params=case.model_params)
        conditioning = initial_conditioning(
            case, start_time, window, args.n_ens, md.unit_factor)
    elif args.forward_model == 'dawis':
        # DAWIS generates the target step from pure noise inside assimilate().
        assert args.method == 'DAWIS', "--forward_model dawis requires --method DAWIS"
        # Unobserved steps (--assim_interval > 1) run unguided with an empty mask.
        forward_model = None
        start_time = max(args.start_time, window)
        conditioning = initial_conditioning(
            case, start_time, window, args.n_ens, md.unit_factor)
    else:
        # 'none': the method produces the next state itself. Unobserved steps
        # run with an empty observation mask.
        forward_model = None
        conditioning = initial_conditioning(
            case, start_time, window, args.n_ens, md.unit_factor)

    # Check that the conditioning window and the run fit in the trajectory.
    n_states = len(dataset)
    if start_time < window:
        raise ValueError(
            f"--start_time {start_time} is less than the conditioning window "
            f"{window}: there are no earlier states to condition on.")
    if start_time + args.n_times > n_states:
        raise ValueError(
            f"--start_time {start_time} + --n_times {args.n_times} = "
            f"{start_time + args.n_times} exceeds the {n_states} states in this "
            f"trajectory.\n"
            + (f"  --start_time was raised from {args.start_time} to fit the "
               f"{window}-frame conditioning window the "
               f"'{args.forward_model}' propagator needs.\n"
               if start_time > args.start_time else "")
            + f"  Lower --n_times to {n_states - start_time}, or use a "
              f"propagator that conditions on fewer frames.")

    # Record the architectures (from the checkpoints) and derived window.
    from forecasting.ckpt_args import describe_arch

    nc.window = window
    nc.forward_model = str(args.forward_model)
    for attr, obj in (("model_arch", assimilation_model),
                      ("propagator_arch", forward_model)):
        arch = getattr(obj, "arch", None)
        if arch:
            setattr(nc, attr, describe_arch(arch))

    # Initialization (from at most one, possibly perturbed, initial condition)
    if args.init_state == 'None':
        x_init = None
    elif 'GT' in args.init_state:
        print(
            f"Initializing ensemble from GT with method '{args.init_state}'")
        print(f"nx: {nx}, ny: {ny}")
        # One step before dataset[start_time]
        print("Dataset shape:", dataset[start_time-1].shape)
        x_init = torch.tensor(x0_obs, dtype=torch.float32, device=device)
        x_init = x_init.view(x0_obs.shape[0], md.num_channels, ny, nx)

    # LETKF and EnSF need a perturbed initial ensemble (zero spread gives NaNs).
    if args.method in ('LETKF', 'EnSF') and 'GT' in args.init_state:
        conditioning[-1] = x_init.detach().cpu().numpy()

    # DAISI
    if args.method == 'DAISI':
        # GT_guide: guide towards the (noisy) GT as a full-field observation.
        if args.init_state == 'GT_guide':
            print(f"Assimilating initial condition with likelihood guidance.")
            # First-step settings: full, linear observation of the initial state.
            assimilation_model.obs_sigma = init_sigma
            assimilation_model.guidance_strength = lambda t: 1
            assimilation_model.set_guide_method('MMPS')

            members = assimilation_model.members
            # One member per initial state
            assimilation_model.members = 1

            x_init = torch.as_tensor(x_init, device=device)
            for i, forecast in tqdm(enumerate(x_init), total=x_init.shape[0], desc="Assimilating initial condition"):
                assimilation_kwargs = {
                    "x_forecast": None,
                    "obs": forecast.flatten(),
                    "obs_mask": torch.ones((1, md.num_channels, ny, nx), device=device),
                    "obs_fn": lambda x: x,  # Linear obs function
                    "ntime": 0,
                }
                x_init[i] = assimilation_model.assimilate(
                    **assimilation_kwargs)
            x_init = x_init.cpu().numpy()

            # Restore the regular parameters
            assimilation_model.obs_sigma = args.obs_sigma
            assimilation_model.guidance_strength = GUIDANCE
            assimilation_model.members = members
            assimilation_model.set_guide_method(args.guide_method)

        # GT_edit (SDEdit) is used in the paper.
        elif args.init_state == 'GT_edit':
            print(f"Assimilating initial condition with SDEdit.")
            assimilation_kwargs = {
                "obs": None,
                "obs_mask": torch.ones((1, md.num_channels, ny, nx), device=device),
                "obs_fn": lambda x: x,  # Linear obs function
                "ntime": 0,
            }

            assimilation_model.set_guide_method(None)
            s = x_init.std(axis=0).mean() / scale
            print(f"Ensemble std in model units: {s}")
            # SDEdit runs at its own tmin; restore the model's (not args.tmin) after.
            tmin_saved = assimilation_model.tmin
            assimilation_model.tmin = 1/(1+s)

            if args.batch_size is not None:
                xens_list = []
                for i in range(0, args.n_ens, args.batch_size):
                    batch_end = min(i + args.batch_size, args.n_ens)

                    if x_init is not None:
                        x_init_batch = torch.tensor(
                            x_init[i:batch_end], dtype=torch.float32, device=device)
                    else:
                        x_init_batch = None
                    assimilation_model.noise = assimilation_model.beta(
                        assimilation_model.tmin) * x_init_batch
                    xens_batch = assimilation_model.assimilate(
                        x_init_batch, **assimilation_kwargs)
                    xens_list.append(xens_batch)

                x_init = torch.cat(xens_list, dim=0)

            else:
                assimilation_model.noise = assimilation_model.beta(
                    assimilation_model.tmin) * torch.tensor(x_init, device=device)
                x_init = assimilation_model.assimilate(
                    x_init, **assimilation_kwargs)

            conditioning[0] = x_init.cpu().numpy()
            assimilation_model.set_guide_method(args.guide_method)
            assimilation_model.tmin = tmin_saved
            assimilation_model.noise = args.noise

    # DAWIS
    if args.method == "DAWIS":
        # The initialization passes below set tmin by hand and need a clean window
        # out; tmin_init and tmax are restored afterwards.
        assimilation_model.tmin_init = None
        assimilation_model.tmax = [1.0] * (args.init_states + 1)

        # Random conditioning window for the initialization passes, built here so
        # batched_assimilation can slice it by batch_size. Layout
        # (window, n_ens, D, X, Y); scaled so DAWIS's `init_states / scale` undoes it.
        seed_window = torch.randn(
            (args.init_states, args.n_ens, *md.state_shape),
            device=device) * assimilation_model.scale

        if args.init_state == 'GT':
            print(f"Initializing ensemble with GT.")
            # Keep the GT conditioning window as is.
        elif args.init_state == 'GT_guide':
            print(f"Assimilating initial condition with likelihood guidance.")

            assimilation_kwargs = {
                "obs": x0_obs,
                "obs_mask": x0_mask,
                "obs_fn": x0_obs_fn,  # Linear obs function
                "ntime": 0,
                "init_states": seed_window,
            }
            assimilation_model.set_guidance_strength(
                lambda t: 2 * torch.ones_like(t))

            for i in range(len(assimilation_model.tmin)):
                assimilation_model.tmin[i] = 0
            assimilation_model.noise = None
            assimilation_model.obs_sigma = x0_sigma

            xens = batched_assimilation(
                assimilation_model, None,
                assimilation_kwargs, args.batch_size, device, return_traj=True)
            xens, posterior = xens
            cond_traj = torch.cat([posterior, xens.unsqueeze(1)], dim=1)
            conditioning = cond_traj.permute(
                # We only want 6 states as conditioning
                1, 0, 2, 3, 4).detach().cpu().numpy()[1:]

        # GT_edit (SDEdit) is used in the paper experiments.
        elif args.init_state == 'GT_edit':
            print(f"Assimilating initial condition with SDEdit.")
            # The target slot starts from an SDEdit blend at beta=tmin[-1].
            assimilation_model.generate_target = False
            assimilation_kwargs = {
                "obs": None,
                "obs_mask": torch.zeros((1, md.num_channels, ny, nx), device=device),
                "obs_fn": lambda x: x,  # Linear obs function
                "ntime": 0,
                "init_states": seed_window,
            }

            s = init_sigma / scale.mean()
            for i in range(len(assimilation_model.tmin)):
                assimilation_model.tmin[i] = 0
            print(f"Ensemble std in model units: {s}, tmin[-1] = {1/(1+s)}")
            assimilation_model.tmin[-1] = 1/(1+s)
            zt = (assimilation_model.tmin[-1]).to(device) * \
                torch.tensor(x_init, device=device)
            assimilation_model.noise = zt / assimilation_model.scale

            xens = batched_assimilation(
                assimilation_model, zt,
                assimilation_kwargs, args.batch_size, device, return_traj=True)
            xens, posterior = xens
            cond_traj = torch.cat([posterior, xens.unsqueeze(1)], dim=1)
            conditioning = cond_traj.permute(
                # We only want 6 states as conditioning
                1, 0, 2, 3, 4).detach().cpu().numpy()[1:]

        elif args.init_state == 'None':
            print(f"Assimilating initial condition with from noise with guidance.")
            conditioning = None
            raise NotImplementedError(
                "Init_state None is not yet supported for DAWIS!")
        else:
            raise ValueError(
                f"DAWIS currently only supports GT, GT_guide, GT_edit, and noise initial conditions, but got '{args.init_state}'")
        # Restore the regular inversion parameters
        assimilation_model.set_guidance_strength(GUIDANCE)
        assimilation_model.set_guide_method(args.guide_method)
        assimilation_model.tmin = DAWIS._validate_tmin(
            args.tmin, args.init_states + 1)
        assimilation_model.tmin_init = DAWIS._validate_tmin_init(
            getattr(args, 'tmin_init', None), args.init_states)
        assimilation_model.tmax = DAWIS._validate_tmax(
            args.tmax, assimilation_model.tmin, args.init_states + 1)
        assimilation_model.noise = args.noise
        assimilation_model.obs_sigma = args.obs_sigma
        assimilation_model.generate_target = (args.forward_model == 'dawis')

    # Pyramid bookkeeping (see `pyramid_spinup_schedule`).
    pyr_tmin = list(args.tmin) if pyramid else None
    pyr_tmax = list(args.tmax) if pyramid else None
    last_smoothed = None
    n_spindown = (args.init_states - args.n_fixed) if pyramid else 0

    # Data Assimilation loop
    torch.set_default_dtype(torch.float32)
    diverged_at = None
    for ntime in tqdm(range(args.n_times)):
        gt = dataset[ntime+start_time]  # shape: [2, lat, lon]

        # False on unobserved steps (always True with --assim_interval 1).
        assimilate_now = is_assimilation_step(ntime, args.assim_interval)

        with torch.no_grad():
            # Get observations
            obs, obs_mask, obs_sigma = observer.observe(gt, t=ntime)
            # Copy: `observe` mutates `observer.indxob` in place.
            indxob = np.array(observer.indxob, copy=True)
            if not assimilate_now:
                # Blank before any consumer (forecast, analysis, obs_history) reads it.
                obs, obs_mask, indxob = blank_observation(obs, obs_mask)
            if conditioning is None:
                x_forecast = None
                raise NotImplementedError(
                    "Conditioning is None. This is not intended.")
            elif args.forward_model in ("numerical", "numerical_gpu"):
                x_forecast = forward_model.forward(conditioning[-1])
            elif (args.forward_model == "fmw"
                    and not getattr(args, "forward_model_path", None)):
                # The assimilation model is its own (guided) forward model.
                x_forecast = forward_model.forecast(
                    conditioning, obs=obs.to(device),
                    obs_mask=obs_mask, obs_fn=obs_fn).cpu().numpy()
            elif (args.forward_model in LEARNED_FORWARD_MODELS
                    or args.forward_model == "flowdas"):
                # Standalone propagator; uses the newest frames it needs.
                x_forecast = forward_model.forward(conditioning)
            elif args.forward_model == "dawis":
                x_forecast = None
            else:
                x_forecast = None

        # Stop on a non-finite forecast (diverged ensemble), keeping completed steps.
        if x_forecast is not None and not np.isfinite(np.asarray(x_forecast)).all():
            bad = np.flatnonzero(
                ~np.isfinite(np.asarray(x_forecast)).reshape(len(x_forecast), -1).all(axis=1))
            print(
                f"Forward model '{args.forward_model}' returned non-finite values "
                f"at time {ntime} for member(s) {bad.tolist()}; the ensemble has "
                f"diverged. Stopping after {ntime} completed steps.", flush=True)
            if run is not None:
                run.summary["diverged_at_time"] = ntime
                run.summary["diverged_members"] = bad.tolist()
            diverged_at = ntime
            break

        # Pyramid spin-up: pin slots still holding initialization states.
        if pyramid:
            n_pinned = max(args.n_fixed, args.init_states - ntime)
            assimilation_model.tmin, assimilation_model.tmax = \
                pyramid_spinup_schedule(pyr_tmin, pyr_tmax, n_pinned)

        # Assimilation
        window_refreshed = False  # True once `conditioning` holds this step's posterior
        if not assimilate_now and x_forecast is not None:
            # Unobserved step with a separate forecast: skip the analysis.
            xens = torch.tensor(
                x_forecast, dtype=torch.float32, device=device)
        else:
            assimilation_kwargs = {
                "obs": obs.to(device),
                "obs_mask": obs_mask,
                "indxob": indxob,
                "obs_fn": obs_fn,
                "ntime": ntime,
                "init_states": conditioning,
                "avg": True if args.obs_fn == 'avg' else False
            }

            # Observation history for DAWIS guide_all_steps (right-aligned to the window).
            if args.method == 'DAWIS' and args.guide_all_steps and len(obs_history) > 0:
                assimilation_kwargs["obs_history"] = obs_history[-window:]
                assimilation_kwargs["obs_mask_history"] = obs_mask_history[-window:]
                assimilation_kwargs["obs_fn_history"] = obs_fn_history[-window:]
                assimilation_kwargs["obs_sigma_history"] = obs_sigma_history[-window:]

            xens = batched_assimilation(
                assimilation_model, x_forecast,
                assimilation_kwargs, args.batch_size, device, return_traj=return_traj)
            if return_traj:
                xens, posterior = xens
                conditioning = posterior.permute(
                    1, 0, 2, 3, 4).detach().cpu().numpy()
                window_refreshed = True

        # Save assimilated ensemble and GT
        x_assim[ntime] = xens.cpu().numpy()
        if args.method == 'DAWIS':
            w = assimilation_model.init_states
            if ntime >= w:
                # The oldest window state is the smoothed estimate.
                x_smooth[ntime-w] = conditioning[0]
                # Leftmost slot of the pyramid spin-down window.
                last_smoothed = conditioning[0]
        if x_window is not None:
            # Slot j holds time ntime - w + j at lag w - j (slot w = analysis).
            w = assimilation_model.init_states
            xens_np = xens.detach().cpu().numpy()
            # If the window was not re-sampled, only the analysis is new.
            slots = range(w + 1) if window_refreshed else (w,)
            for j in slots:
                t_slot = ntime - w + j
                if t_slot < 0:
                    continue
                x_window[t_slot, w - j] = xens_np if j == w else conditioning[j]
                window_valid[t_slot, w - j] = 1
        ground_truth[ntime] = gt
        # Console metrics in output units; gt is in raw units, so scale it.
        # `absmax` should stay near max|gt| (early divergence warning).
        _a = xens.cpu().numpy() * out_a + out_b
        _g = np.asarray(gt) * md.unit_factor * out_a + out_b
        print(
            f"RMSE at time {ntime}: {np.sqrt(((_a.mean(axis=0) - _g) ** 2).mean()):.4f} "
            f"| spread {_a.std(axis=0).mean():.4f} | absmax {np.abs(_a).max():.2f} "
            f"(gt {np.abs(_g).max():.2f})", flush=True)
        nc.sync()

        # Live metrics, matching those computed from the saved file.
        if run is not None:
            pred_m = xens.detach().cpu().to(torch.float32)  # (ens, C, y, x)
            gt_m = torch.tensor(
                np.array(gt), dtype=torch.float32) * md.unit_factor
            pred_m = pred_m * out_a + out_b
            gt_m = gt_m * out_a + out_b
            step_metrics, *_ = compute_step_metrics(pred_m, gt_m)
            run.log(step_metrics, step=ntime)

        # Update observation history (keep last `window` entries)
        obs_history.append(obs.to(device))
        obs_mask_history.append(obs_mask)
        obs_fn_history.append(obs_fn)
        obs_sigma_history.append(obs_sigma)
        if len(obs_history) > window:
            obs_history.pop(0)
            obs_mask_history.pop(0)
            obs_fn_history.pop(0)
            obs_sigma_history.pop(0)

        if window > 1:
            conditioning = np.concatenate([conditioning[1:], xens.cpu().numpy()[
                None, ...]], axis=0).astype(np.float32, copy=False)
        else:
            conditioning = xens.unsqueeze(
                0).cpu().numpy().astype(np.float32, copy=False)

        if args.guide_first=="all" and args.method == 'DAWIS' and ntime >= window-1:
            print("Modifying obs_history for guide_first=all")

            xprev_obs_list = []
            for k in range(args.n_ens):
                xprev_obs_k, _, _ = xprev_observer.observe(
                    conditioning[0, k]  / md.unit_factor, t=0)
                xprev_obs_list.append(xprev_obs_k)
            xprev_obs = torch.cat(xprev_obs_list, dim=0)
            
            obs_history[0] = xprev_obs.to(device)
            obs_mask_history[0] = x0_mask
            obs_fn_history[0] = x0_obs_fn
            obs_sigma_history[0] = args.x0_sigma
            print(f"Modified obs_history[0] to xprev_obs with shape {xprev_obs.shape} and std {obs_sigma_history[0]}")

    if args.method == 'DAWIS' and diverged_at is not None:
        print(f"Skipping the DAWIS spin-down: the run stopped at time "
              f"{diverged_at}, so there is no in-flight window to finish.",
              flush=True)
    elif args.method == 'DAWIS':
        w = assimilation_model.init_states
        if pyramid:
            # Spin-down: finish the in-flight states one pass at a time. The window
            # is shifted one step left (last emitted state on the left, time
            # n_times-1 on the right); slot j holds time n_times - w - 1 + j.
            if last_smoothed is None:
                raise ValueError(
                    "--pyramid spin-down needs at least one emitted smoothed state, "
                    f"i.e. --n_times > init_states = {args.init_states}; got "
                    f"{args.n_times}.")
            win = np.concatenate(
                [last_smoothed[None], conditioning], axis=0).astype(
                    np.float32, copy=False)
            # Slots 1..n_fixed are already clean but not yet written.
            for j in range(1, args.n_fixed + 1):
                x_smooth[args.n_times - w - 1 + j] = win[j]
            print(f"--pyramid: spin-down over {n_spindown} pass(es) for the "
                  f"states still in flight (times "
                  f"{args.n_times-w+args.n_fixed}..{args.n_times-1})", flush=True)
            for q in range(n_spindown):
                assimilation_model.tmin, assimilation_model.tmax = \
                    pyramid_spindown_schedule(pyr_tmin, q + 1)
                # Rightmost slot: the real state, in normalized units.
                assimilation_model.noise = (
                    torch.as_tensor(win[-1], dtype=torch.float32, device=device)
                    / assimilation_model.scale)
                drain_kwargs = {
                    # Last observation on the rightmost slot; leftmost slot unobserved.
                    "obs": obs_history[-1],
                    "obs_mask": obs_mask_history[-1],
                    "obs_fn": obs_fn_history[-1],
                    "obs_history": list(obs_history)[-window:-1],
                    "obs_mask_history": list(obs_mask_history)[-window:-1],
                    "obs_fn_history": list(obs_fn_history)[-window:-1],
                    "obs_sigma_history": list(obs_sigma_history)[-window:-1],
                    "ntime": args.n_times + q,
                    "init_states": win[:-1],
                    "avg": True if args.obs_fn == 'avg' else False,
                }
                xens, posterior = batched_assimilation(
                    assimilation_model, None, drain_kwargs, args.batch_size,
                    device, return_traj=True)
                win = np.concatenate([
                    posterior.permute(1, 0, 2, 3, 4).detach().cpu().numpy(),
                    xens.detach().cpu().numpy()[None]], axis=0).astype(
                        np.float32, copy=False)
                # Pass q takes slot n_fixed + q + 1 to beta = 1.
                j = args.n_fixed + q + 1
                x_smooth[args.n_times - w - 1 + j] = win[j]
        else:
            x_smooth[args.n_times-w:args.n_times] = conditioning
    nc.close()
    if debug:
        return args.exp_name, assimilation_model
    else:
        return args.exp_name


def batched_assimilation(assimilation_model, x_forecast, assimilation_kwargs, batch_size, device, return_traj=False):
    """
    Run data assimilation on an ensemble forecast, optionally in batches of
    ``batch_size`` members to reduce memory usage.

    Args:
        assimilation_model: Assimilation model object implementing an
            ``assimilate`` method.
        x_forecast (torch.Tensor or np.ndarray or None): Forecast ensemble
            of shape ``(n_ens, ...)`` containing prior ensemble states.
            If ``None``, the assimilation model is called without forecast
            input.
        assimilation_kwargs (dict): Additional keyword arguments passed
            directly to ``assimilation_model.assimilate``.
        batch_size (int or None): Number of ensemble members to process
            per batch. If ``None``, all ensemble members are processed
            simultaneously.
        device (torch.device): Device on which assimilation is performed.

    Returns:
        torch.Tensor: Assimilated ensemble states of shape ``(n_ens, ...)``
        (and the posterior window if ``return_traj``).
    """
    if batch_size is not None:
        n_ens = x_forecast.shape[0] if x_forecast is not None else assimilation_model.members
        init_states_full = assimilation_kwargs.get("init_states", None)
        # `assimilation_model.noise` (a per-member target latent) is sliced along
        # with `init_states`, and only when `init_states` is given (otherwise DAWIS
        # builds a full-ensemble window). Restored after the loop.
        noise_full = getattr(assimilation_model, "noise", None)
        slice_noise = (init_states_full is not None
                       and isinstance(noise_full, torch.Tensor)
                       and noise_full.shape[0] == n_ens)
        xens_list = []
        posterior_list = [] if return_traj else None
        try:
            for i in range(0, n_ens, batch_size):
                batch_end = min(i + batch_size, n_ens)

                if x_forecast is not None:
                    x_forecast_batch = torch.tensor(
                        x_forecast[i:batch_end], dtype=torch.float32, device=device)
                else:
                    x_forecast_batch = None

                # init_states layout is (window, n_ens, D, X, Y).
                batch_kwargs = assimilation_kwargs
                if init_states_full is not None:
                    batch_kwargs = dict(assimilation_kwargs)
                    batch_kwargs["init_states"] = init_states_full[:, i:batch_end]

                # Slice per-member obs_history entries, shape (n_ens, n_obs), e.g.
                # the --guide_first slot; shared (1, n_obs) entries pass through.
                obs_history_full = assimilation_kwargs.get("obs_history", None)
                if obs_history_full is not None and n_ens > 1:
                    sliced = [o[i:batch_end]
                              if (hasattr(o, "shape") and o.ndim > 0
                                  and o.shape[0] == n_ens) else o
                              for o in obs_history_full]
                    if any(x is not y for x, y in zip(sliced, obs_history_full)):
                        if batch_kwargs is assimilation_kwargs:
                            batch_kwargs = dict(assimilation_kwargs)
                        batch_kwargs["obs_history"] = sliced

                if slice_noise:
                    assimilation_model.noise = noise_full[i:batch_end]

                xens_batch = assimilation_model.assimilate(
                    x_forecast_batch, **batch_kwargs, return_traj=return_traj)
                if return_traj:
                    xens_batch, posterior = xens_batch
                xens_list.append(xens_batch.cpu())
                if return_traj:
                    posterior_list.append(posterior.cpu())
                torch.cuda.empty_cache()
        finally:
            if slice_noise:
                assimilation_model.noise = noise_full

        xens = torch.cat(xens_list, dim=0)
        if return_traj:
            return xens, torch.cat(posterior_list, dim=0)

    else:
        xens = assimilation_model.assimilate(
            x_forecast, **assimilation_kwargs, return_traj=return_traj)
        if return_traj:
            xens, posterior = xens
            return xens, posterior
    return xens


if __name__ == "__main__":
    args = parse_args()

    # Named experiment preset (SEVIR always gets the `sevir` preset).
    apply_experiment(args)

    # Add timestamp and random ID to experiment name
    timestamp = datetime.now().strftime("%m_%d_%H")  # month_day_hour format
    random_id = str(uuid.uuid4())[:4]  # Use first 4 characters of UUID
    args.exp_name += f'_{timestamp}_{random_id}'

    # Data Assimilation
    if args.online_metrics:
        # Stream scalar metrics live; upload media to the same run at the end.
        import wandb
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.exp_name,
            config=wandb_config(args),
        )
        exp_name = assimilate(args, run=run)
        # Media only (scalars already streamed live); finishes the run.
        upload_results(exp_name, args, run=run, log_metrics=False)
    else:
        exp_name = assimilate(args)
        # Compute metrics and upload results after the run
        upload_results(exp_name, args)

    if args.method == 'DAWIS':
        upload_results(exp_name, args, smoother=True)
