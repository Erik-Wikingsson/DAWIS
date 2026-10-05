"""
General smoothing entrypoint.

Handles data loading, observation construction and output writing; the
smoothing algorithm itself is a method class in `assimilation/methods/`.
"""
import os
from datetime import datetime
import uuid
from types import SimpleNamespace

import numpy as np
import torch
from assimilation.methods.sda import SDA
from assimilation.methods.gibbs_smoother import BlockGibbsSmoother

from assimilation.assimilate import (
    apply_experiment, parse_args, build_observer, check_forward_model_supported,
    resolve_device, create_nc_file, set_seed, _wrap_output)
from assimilation.obs_schedule import blank_observation, is_assimilation_step
from assimilation.case import load_assim_case
from assimilation.observers.observer import GridObserver
from data.registry import data_source_from_args
from utils import (upload_results, compute_step_metrics, wandb_config,
                   stored_norm)

SMOOTHING_METHODS = {
    "SDA": SDA,
    "GIBBS": BlockGibbsSmoother,
}

# Methods that support per-step obs operator / sigma histories, and hence the
# extra x0 guidance slot (see `prepend_x0_slot`).
X0_GUIDE_METHODS = {"SDA"}

def load_dataset(args):
    """Resolve the trajectory this run assimilates.

    Uses the same `load_assim_case` as assimilate.py, so smoother and filter
    runs are comparable. Returns `(states, metadata, case)`; the case carries
    the numerical-model parameters and grid geometry.
    """
    src, md, case = load_assim_case(args)
    return case.states, md, case

def prepare_observations(dataset, observer, start_time, n_times, avg=False,
                         assim_interval=1):
    """Observations for every step, plus the observed indices used at each step.

    `observer.indxob` changes per step for non-stationary networks, so the
    per-step indices are returned in `indxob_history`.

    With `assim_interval` k, only every k-th step is observed; the other steps
    get an empty observation (all-False mask, zero-length vector) from
    `blank_observation`, which each method ignores.
    """
    obs_history = []
    obs_mask_history = []
    indxob_history = []
    for t_idx in range(n_times):
        gt = dataset[start_time + t_idx]
        if avg:
            obs, obs_mask, _ = observer.observe(gt)
        else:
            obs, obs_mask, _ = observer.observe(gt, t=t_idx)
        indxob = np.array(observer.indxob, copy=True)
        if not is_assimilation_step(t_idx, assim_interval):
            obs, obs_mask, indxob = blank_observation(obs, obs_mask)
        obs_history.append(obs)
        obs_mask_history.append(obs_mask)
        indxob_history.append(indxob)
    return obs_history, obs_mask_history, indxob_history


def resolve_trajectory_path(args):
    """The initial-trajectory file for this run, or None.

    An explicit --trajectory_path wins. Otherwise --initial_trajectory names a
    method (e.g. DAWISnumerical), looked up in the paper-run archive by
    (method, --experiment, --data_index). The resolved path is written back
    onto `args` so the result file records it.
    """
    explicit = getattr(args, "trajectory_path", None)
    if explicit is not None and explicit != "None":
        return explicit

    method = getattr(args, "initial_trajectory", None)
    if method is None or method == "None":
        return None

    from data.paths import find_paper_run

    path = str(find_paper_run(method, args.experiment, args.data_index))
    print(f"Loader: resolved --initial_trajectory {method} to {path}")
    args.trajectory_path = path
    return path


def load_trajectory_init(args, device):
    """The warm-start trajectory in assimilation units, or None.

    `.nc` results are converted back from their recorded `--output_norm`;
    `.npy` and `.pt` are assumed to be in assimilation units already. The
    caller divides by `smoother.scale` to reach network units.
    """
    trajectory_path = resolve_trajectory_path(args)
    if trajectory_path is None:
        return None

    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    if trajectory_path.endswith(".npy"):
        trajectory = np.load(trajectory_path)
        return torch.as_tensor(trajectory, device=device, dtype=torch.float32)

    if trajectory_path.endswith(".nc"):
        from netCDF4 import Dataset as NetCDFDataset

        trajectory_var = getattr(args, "trajectory_var", "auto")
        with NetCDFDataset(trajectory_path, mode="r") as nc:
            if trajectory_var == "auto":
                for candidate in ("x_smooth", "x_assim"):
                    if candidate in nc.variables:
                        trajectory = nc.variables[candidate][:]
                        print('Selected trajectory variable', candidate, 'from .nc file')
                        break
                else:
                    raise ValueError(
                        f"No trajectory variable found in {trajectory_path}. Expected one of x_smooth, x_assim.")
            else:
                if trajectory_var not in nc.variables:
                    raise ValueError(
                        f"Trajectory variable '{trajectory_var}' not found in {trajectory_path}.")
                trajectory = nc.variables[trajectory_var][:]
            stored = stored_norm(nc)

        trajectory = torch.as_tensor(trajectory, device=device, dtype=torch.float32).permute(1, 0, 2, 3, 4)
        print('Loader: Loaded trajectory', trajectory_path, 'from .nc file with shape', trajectory.shape)
        if trajectory.shape[0] != args.n_ens:
                print(f"Trajectory shape mismatch. Expected ensemble size: {args.n_ens}, got: {trajectory.shape[0]}")
                if trajectory.shape[0] > args.n_ens:
                    print(f"Truncating trajectory to first {args.n_ens} ensemble members.")
                    trajectory = trajectory[:args.n_ens]
                else:
                    print(f"Expanding trajectory to match ensemble size: {args.n_ens}")
                    trajectory = trajectory[0:1].expand(args.n_ens, -1, -1, -1, -1)
        # Invert the file's output affine to get back to assimilation units.
        _md = data_source_from_args(args).metadata
        a, b = _md.output_affine(stored)
        if (a, b) != (1.0, 0.0):
            trajectory = (trajectory - b) / a
            print("Loader: converted warm start from "
                  f"'{stored or 'assim'}' units to assimilation units")
        return trajectory

    if trajectory_path.endswith((".pt", ".pth")):
        trajectory = torch.load(trajectory_path, map_location=device)
        return torch.as_tensor(trajectory, device=device, dtype=torch.float32)

    raise ValueError(
        f"Unsupported trajectory_path format '{trajectory_path}'. Use .npy, .pt, or .pth.")


def build_x0_observation(args, dataset, device):
    """Full-field, noisy observations of the state one step before the window.

    Mirrors `assimilate.py`: noise std `init_std * scalefact` applied to
    `dataset[start_time - 1]`, one draw per ensemble member, so each member is
    guided towards its own perturbed initial condition.

    Returns:
        (obs, obs_mask, indxob): obs in scalefact units, shape
        (n_ens, 2*nx*ny) -- one row per member; an all-True mask of shape
        (1, 2, ny, nx); the observed flat indices.
    """
    if args.start_time < 1:
        raise ValueError(
            f"--guide_first needs a state before the window (--start_time >= 1), got {args.start_time}")

    md = data_source_from_args(args).metadata
    init_sigma = args.init_std * md.unit_factor
    x0_observer = GridObserver(
        obs_fun=lambda x: x,
        obs_prob=1.0,
        obs_mask=None,
        obs_sigma=init_sigma,
        stationary_obs=True,
        random_seed=42,
        nx=md.nx,
        ny=md.ny,
        device=device,
        radar=False,
        num_channels=md.num_channels,
        unit_factor=md.unit_factor,
    )
    obs_list = []
    for _ in range(args.n_ens):
        obs_k, obs_mask, _ = x0_observer.observe(dataset[args.start_time - 1], t=0)
        obs_list.append(obs_k)
    obs = torch.cat(obs_list, dim=0)
    print(f"x0 guidance: {args.n_ens} per-member full-field observations of "
          f"dataset[{args.start_time - 1}] with noise std {init_sigma:.4g} "
          f"(init_std={args.init_std}), guidance sigma {args.x0_sigma:.4g}, "
          f"obs shape {tuple(obs.shape)}", flush=True)
    return obs.to(device), obs_mask, np.array(x0_observer.indxob, copy=True)


def prepend_x0_slot(args, dataset, device, obs_history, obs_mask_history,
                    indxob_history, obs_fn):
    """Add a leading trajectory slot holding the guided initial condition.

    Prepends a slot for `start_time - 1`, observed full-field with sigma
    `--x0_sigma`, and returns per-step operator/sigma histories so the real
    observations keep their own operator and `--obs_sigma`. `run_smoothing`
    drops the extra slot from the result.

    Returns:
        (obs_history, obs_mask_history, indxob_history, obs_fn_history,
         obs_sigma_history, n_times) with n_times = args.n_times + 1.
    """
    x0_obs, x0_mask, x0_indxob = build_x0_observation(args, dataset, device)

    def x0_obs_fn(x): return x

    obs_history = [x0_obs] + list(obs_history)
    obs_mask_history = [x0_mask] + list(obs_mask_history)
    indxob_history = [x0_indxob] + list(indxob_history)
    obs_fn_history = [x0_obs_fn] + [obs_fn] * args.n_times
    obs_sigma_history = [args.x0_sigma] + [args.obs_sigma] * args.n_times

    return (obs_history, obs_mask_history, indxob_history, obs_fn_history,
            obs_sigma_history, args.n_times + 1)


def log_sweep_run(args, traj, gt, sweep):
    """Log one Gibbs sweep's trajectory to its own wandb run.

    Runs are named `smoother_{exp_name}_gibbs_{k}` and grouped under
    `smoother_{exp_name}`, with the same per-step metrics as `upload_results`
    (no animation). Per-sweep means go into the run summary.

    traj: (t, ens, z, y, x) in `--output_norm` units
    gt: (t, z, y, x) in the same units
    """
    import wandb
    from plotting.plotting import save_spectrum

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f'smoother_{args.exp_name}_gibbs_{sweep}',
        group=f'smoother_{args.exp_name}',
        config=wandb_config(args, gibbs_sweep=sweep),
        reinit=True,
    )
    save_dir = run.dir

    totals = {}
    n_steps = traj.shape[0]
    for t in range(n_steps):
        metrics, rad_truth, rad_forecast, rad_prior = compute_step_metrics(
            traj[t], gt[t])
        run.log(metrics, step=t)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)

        if t % args.plot_every == 0:
            fname = save_spectrum(rad_truth, rad_forecast, rad_prior,
                                  time=t, dir=save_dir)
            run.log({"spectrum": wandb.Image(fname)}, step=t)
            try:
                if os.path.exists(fname):
                    os.remove(fname)
            except Exception:
                pass

    means = {key: total / n_steps for key, total in totals.items()}
    run.summary.update({f'mean {key}': value for key, value in means.items()})
    run.finish()
    return means


def build_sweep_logger(args, dataset, smoother, n_times, nc=None, drop_first=False,
                       md=None):
    """Per-sweep checkpointing and metric logging for the Gibbs smoother.

    Returns a callback `(sweep_idx, trajectory)` called after every sweep.
    With `nc` given (--save_sweeps), sweep k is written to the result file as
    `x_gibbs_{k}` (1-based, same layout and units as x_smooth) and synced
    immediately. Metrics are computed in `--output_norm` units so they match
    `upload_results`.
    """
    if md is None:
        md = data_source_from_args(args).metadata
    log_wandb = getattr(args, 'upload_results', True)

    out_a, out_b = md.output_affine(getattr(args, 'output_norm', None))

    n_physical = n_times - 1 if drop_first else n_times
    # Truth in output units, matching how upload_results reads ground_truth.
    gt = torch.tensor(
        np.asarray(dataset[args.start_time:args.start_time + n_physical],
                   dtype=np.float32)) * md.unit_factor
    gt = gt * out_a + out_b

    def callback(sweep_idx, trajectory):
        sweep = sweep_idx + 1  # 1-based, to match the x_gibbs_1... naming
        traj = (trajectory * smoother.scale).detach().cpu().to(torch.float32)
        if drop_first:
            traj = traj[:, 1:]
        traj = traj.permute(1, 0, 2, 3, 4)  # (t, ens, z, y, x), as in the .nc

        if nc is not None:
            # Wrapped so it is stored in output units like x_smooth.
            var = _wrap_output(
                nc.createVariable(
                    f'x_gibbs_{sweep}', np.float32,
                    ('t', 'ens', 'z', 'y', 'x'), zlib=True),
                md, getattr(args, 'output_norm', None))
            for ti in range(traj.shape[0]):
                var[ti] = traj[ti].numpy()
            nc.sync()

        # Convert for metrics only after the write (the wrapper converts itself).
        traj = traj * out_a + out_b

        note = '' if nc is None else f' | saved as x_gibbs_{sweep}'
        if log_wandb:
            means = log_sweep_run(args, traj, gt, sweep)
            summary = ' | '.join(f'{k}={v:.4g}' for k, v in means.items())
            print(f'[sweep {sweep}] {summary}{note}', flush=True)
        else:
            print(f'[sweep {sweep}]{note}', flush=True)

    return callback


def build_smooth_args(args, obs_history, obs_mask_history, obs_fn, n_times, window_radius, steps, avg, trajectory_init,
                      obs_fn_history=None, obs_sigma_history=None):
    smooth_args = SimpleNamespace(
        obs_history=obs_history,
        obs_mask_history=obs_mask_history,
        obs_fn=obs_fn,
        n_times=n_times,
        window_radius=window_radius,
        steps=steps,
        avg=avg,
        trajectory_init=trajectory_init,
        trajectory_path=getattr(args, "trajectory_path", None),
        trajectory_var=getattr(args, "trajectory_var", "auto"),
        return_traj=True,
        # None unless --guide_first added an x0 slot.
        obs_fn_history=obs_fn_history,
        obs_sigma_history=obs_sigma_history,
    )

    if args.method == "GIBBS":
        smooth_args.num_sweeps = getattr(args, "gibbs_sweeps", None)
        smooth_args.block_mode = getattr(args, "block_mode", None)
        smooth_args.block_stride = getattr(args, "block_stride", None)
        smooth_args.block_direction = getattr(args, "block_direction", None)
        smooth_args.gibbs_midpoint_tmin = getattr(args, "gibbs_midpoint_tmin", None)
        smooth_args.gibbs_endpoint_tmin = getattr(args, "gibbs_endpoint_tmin", None)
        smooth_args.gibbs_num_endpoints = getattr(args, "gibbs_num_endpoints", None)

    return smooth_args


def run_smoothing(args):
    set_seed(args)
    device = resolve_device(args)


    # SDA/GIBBS use a symmetric window of size 2w+1, with init_states = 2w.
    if args.method in {'SDA', 'GIBBS'}:
        assert args.init_states%2 == 0, "init_states+1 must be odd for SDA"
        args.sda_w = args.init_states// 2
    else:
        args.sda_w = 0

    # An averaged observation covers the whole field, so unobserved steps
    # cannot be masked out of the window operator.
    if args.obs_fn == 'avg' and args.assim_interval != 1:
        raise NotImplementedError(
            f"--obs_fn avg with --assim_interval {args.assim_interval} is not "
            "supported by the smoothers. An averaged observation covers the "
            "whole field, so there is no mask to empty for the steps in "
            "between -- the window operator would return a full-length H(x) "
            "against a zero-length observation. Use a masked observation "
            "operator (linear/arctan/square_scaled), or --assim_interval 1.")

    dataset, md, case = load_dataset(args)
    observer, obs_fn = build_observer(args, device)
    obs_history, obs_mask_history, indxob_history = prepare_observations(
        dataset, observer, args.start_time, args.n_times,
        avg=args.obs_fn == 'avg', assim_interval=args.assim_interval)
    trajectory_init = load_trajectory_init(args, device)

    # Initial-condition guidance: prepend an x0 slot, trimmed from the result.
    guide_first = getattr(args, "guide_first", "none")
    n_times = args.n_times
    obs_fn_history = obs_sigma_history = None
    guide_x0 = guide_first in ("init", "all")
    if guide_x0:
        if args.method not in X0_GUIDE_METHODS:
            raise NotImplementedError(
                f"--guide_first {guide_first} is only supported for "
                f"{sorted(X0_GUIDE_METHODS)} in smoothing.py, got '{args.method}'")
        if args.obs_fn == 'avg':
            raise NotImplementedError(
                "--guide_first is not supported with --obs_fn avg (averaged "
                "observations have no per-step mask to place the x0 observation in)")
        if args.x0_sigma <= 0:
            raise ValueError(f"--guide_first needs --x0_sigma > 0, got {args.x0_sigma}")
        if guide_first == "all":
            # A smoother does not cycle, so 'all' is equivalent to 'init'.
            print("--guide_first all has no separate meaning for a smoother; "
                  "treating it as 'init' (anchor the prepended x0 slot).")
        (obs_history, obs_mask_history, indxob_history, obs_fn_history,
         obs_sigma_history, n_times) = prepend_x0_slot(
            args, dataset, device, obs_history, obs_mask_history,
            indxob_history, obs_fn)
        if n_times < 2 * args.sda_w + 1:
            raise ValueError(
                f"trajectory length {n_times} is shorter than the window "
                f"{2 * args.sda_w + 1}; increase --n_times")
        if trajectory_init is not None:
            # Seed the extra x0 slot of a warm start with its first state.
            time_dim = 0 if trajectory_init.dim() == 4 else 1
            if trajectory_init.shape[time_dim] == args.n_times:
                first = trajectory_init.narrow(time_dim, 0, 1)
                trajectory_init = torch.cat([first, trajectory_init], dim=time_dim)
                print(f"Warm start extended to {trajectory_init.shape[time_dim]} "
                      "steps for the prepended x0 slot")

    smoother_cls = SMOOTHING_METHODS[args.method]

    # Reject unsupported method/forward-model pairs before loading checkpoints.
    check_forward_model_supported(args)

    smoother = smoother_cls(args, metadata=md)

    # Per-sweep saving writes into the result file, so create it up front.
    nc = ground_truth = x_assim = x_smooth = None
    sweep_callback = None
    if args.method == "GIBBS" and (
            getattr(args, 'save_sweeps', False) or getattr(args, 'sweep_metrics', False)):
        if getattr(args, 'save_sweeps', False):
            nc, ground_truth, x_assim, x_smooth = create_nc_file(args, md)
        sweep_callback = build_sweep_logger(
            args, dataset, smoother, n_times, nc=nc, drop_first=guide_x0, md=md)

    if trajectory_init is not None:
        # Assimilation units -> network units (inverse of the output scaling).
        trajectory_init = trajectory_init / torch.as_tensor(
            smoother.scale, dtype=trajectory_init.dtype,
            device=trajectory_init.device)

    smooth_args = build_smooth_args(
        args=args,
        obs_history=obs_history,
        obs_mask_history=obs_mask_history,
        obs_fn=obs_fn,
        n_times=n_times,
        window_radius=args.sda_w,
        steps=args.euler_steps,
        avg=args.obs_fn == 'avg',
        trajectory_init=trajectory_init,
        obs_fn_history=obs_fn_history,
        obs_sigma_history=obs_sigma_history,
    )
    if sweep_callback is not None:
        smooth_args.sweep_callback = sweep_callback
    try:
        result = smoother.smooth(smooth_args)
    except BaseException:
        # Keep the sweeps written so far readable.
        if nc is not None:
            nc.close()
        raise

    smoothed_np = (result * smoother.scale).detach().cpu().numpy()

    if guide_x0:
        # Drop the prepended x0 slot.
        smoothed_np = smoothed_np[:, 1:]

    if nc is None:
        nc, ground_truth, x_assim, x_smooth = create_nc_file(args, md)
    T = smoothed_np.shape[1]

    ground_truth[:, :, :, :] = dataset[args.start_time:args.start_time + T]
    # netCDF variables are (time, ens, ...); smoothed_np is (ens, time, ...).
    for ti in range(T):
        x_smooth[ti] = smoothed_np[:, ti]
    nc.sync()
    nc.close()

    print(f"Saved smoothed trajectory to {args.exp_name}")
    return args.exp_name


if __name__ == "__main__":
    args = parse_args()

    # Named experiment preset (see assimilation/experiments.py).
    apply_experiment(args)

    # Add timestamp and random ID to experiment name
    timestamp = datetime.now().strftime("%m_%d_%H")  # month_day_hour format
    random_id = str(uuid.uuid4())[:4]  # Use first 4 characters of UUID
    args.exp_name += f'_{timestamp}_{random_id}'

    exp_name = run_smoothing(args)
    # Pass --no_upload_results for a wandb-free run.
    if getattr(args, 'upload_results', True):
        upload_results(exp_name, args, smoother=True)
