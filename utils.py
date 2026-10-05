import os
import wandb
import torch
import numpy as np
from netCDF4 import Dataset
from plotting.plotting import save_spectrum, save_video
from metrics.metrics import rmse, crps_ens, spread_skill_ratio, spread_squared, log_spectral_distance
from assimilation.registry import build_observer, observer_kind
from data.registry import data_source_from_args
from data.paths import result_file


def wandb_config(args, **extra):
    """Return `vars(args)` plus `extra`, recording `sevir_norm` also as `assim_norm`."""
    config = dict(vars(args))
    if config.get("sevir_norm") is not None:
        config.setdefault("assim_norm", config["sevir_norm"])
    config.update(extra)
    return config


def stored_norm(nc):
    """Return the normalization a result file is stored in, from `nc.output_norm`.

    Returns None (assimilation units) if the attribute is missing or 'assim'.
    """
    if "output_norm" not in nc.ncattrs():
        return None
    value = str(nc.getncattr("output_norm"))
    return None if value == "assim" else value


def denormalize_for_plot(md, stored, *fields):
    """Convert stored fields to the physical units used by `md.imshow_kwargs()`.

    `stored` is the file's normalization (see `stored_norm`). For display only;
    metrics stay in the file's own units.
    """
    a, b = md.stored_to_physical_affine(stored)
    if (a, b) == (1.0, 0.0):
        return fields if len(fields) != 1 else fields[0]
    out = [np.asarray(f, dtype=np.float32) * a + b for f in fields]
    return out if len(out) != 1 else out[0]


def plot_noise_scale(md):
    """Scale factor taking `--obs_sigma` from assimilation to physical units."""
    a, _ = md.stored_to_physical_affine(None)
    return a


def compute_step_metrics(pred, gt):
    """Compute the per-step scalar metrics.

    Args:
        pred: (M, D, X, Y) ensemble, in model units.
        gt: (D, X, Y) ground truth, in model units.

    Returns:
        metrics: dict of scalar metrics.
        rad_truth, rad_forecast, rad_prior: radially averaged spectra.
    """
    spectral_loss, rad_truth, rad_forecast, rad_prior = log_spectral_distance(
        pred, gt, ens_dim=0)
    metrics = {
        "RMSE": rmse(torch.mean(pred, dim=0), gt),
        "Avg RMSE": rmse(pred, gt.unsqueeze(0)).mean(dim=0),
        "CRPS": crps_ens(pred, gt, ens_dim=0),
        "Spread Skill Ratio": spread_skill_ratio(pred, gt, ens_dim=0),
        "Spread": spread_squared(pred, ens_dim=0),
        "Spectral Distance": spectral_loss,
    }
    return metrics, rad_truth, rad_forecast, rad_prior


def upload_results(exp_name, args, smoother=False, run=None, log_metrics=True):
    """Compute metrics and plots from a saved .nc result file and log them.

    Args:
        exp_name: name of the experiment file.
        args: arguments used for the experiment (not read from the file).
    """
    wandb_name = exp_name
    if smoother:
        wandb_name = f'smoother_{exp_name}'
    if run is None:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=wandb_name,
            config=wandb_config(args)
        )

    save_dir = run.dir  # Unique per run

    nc_filename = result_file(exp_name)
    nc = Dataset(str(nc_filename), 'r')
    scalefact = nc.scalefact
    forecast = nc['x_assim']  # shape: [time, ens, C, lat, lon]
    if smoother:
        forecast = nc['x_smooth']  # shape: [time, ens, C, lat, lon]
    truth = scalefact * nc['ground_truth']  # shape: [time, C, lat, lon]

    # NOTE: 'avg' uses obs_prob=1.0; radar overwrites args.obs_prob (used as
    # save_video's `obs_p` label).
    md = data_source_from_args(args).metadata
    if observer_kind(args) == 'avg':
        observer, obs_fn = build_observer(args, md, args.device, obs_prob=1.0)
    else:
        if getattr(args, 'radar', False):
            args.obs_prob = args.radar_width / md.nx
        observer, obs_fn = build_observer(args, md, 'cpu')

    for t, (pred, gt) in enumerate(zip(forecast, truth)):
        is_plot_step = t % args.plot_every == 0
        if not (log_metrics or is_plot_step):
            continue

        pred = torch.tensor(np.array(pred), dtype=torch.float32)
        gt = torch.tensor(np.array(gt), dtype=torch.float32)

        metrics, rad_truth, rad_forecast, rad_prior = compute_step_metrics(
            pred, gt)

        if log_metrics:
            run.log(metrics, step=t)

        if is_plot_step:
            fname = save_spectrum(rad_truth, rad_forecast,
                                  rad_prior, time=t, dir=save_dir)
            run.log({
                "spectrum": wandb.Image(fname)
            }, step=t)
            # Remove the temporary file after logging
            try:
                if os.path.exists(fname):
                    os.remove(fname)
            except Exception:
                pass

    if not getattr(args, 'no_media_wandb', False):
        # Video in physical units, except --radar, whose observer works in
        # assimilation units: there only the output affine is undone.
        stored = stored_norm(nc)
        if getattr(args, 'radar', False):
            a, b = md.output_affine(stored)
            if (a, b) == (1.0, 0.0):
                vid_forecast, vid_truth = forecast, truth
            else:
                vid_forecast = (np.asarray(forecast, dtype=np.float32) - b) / a
                vid_truth = (np.asarray(truth, dtype=np.float32) - b) / a
            vid_sigma = args.obs_sigma
        else:
            vid_forecast, vid_truth = denormalize_for_plot(
                md, stored, forecast, truth)
            vid_sigma = args.obs_sigma * plot_noise_scale(md)
        fname = save_video(vid_forecast, vid_truth, vid_sigma,
                           args.obs_prob, obs_fn=obs_fn, dir=save_dir, level=0, lres=args.lres, radar=args.radar, observer=observer,
                           unit_factor=float(scalefact),
                           imshow_kwargs=md.imshow_kwargs(0),
                           spread_kwargs=md.imshow_kwargs(0, spread=True),
                           unit_label=md.variable(0).units)
        run.log({
            "animation": wandb.Video(fname, fps=10, format="mp4")
        })
        # Remove the temporary file after logging
        try:
            if os.path.exists(fname):
                os.remove(fname)
        except Exception:
            pass
    run.finish()


def init_wandb_metrics(wandb_logger, val_steps):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in val_steps:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")
