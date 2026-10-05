import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.animation as animation
import os



def save_spectrum(rad_truth, rad_forecast, rad_prior, time, dir):
    # Plot mean spectra across batch
    plt.figure(figsize=(4, 3))
    x = np.arange(1, rad_truth.shape[1]-1)
    plt.loglog(x, rad_truth.mean(0).numpy()[1:-1], label="Truth")
    plt.loglog(x, rad_forecast.mean(0).numpy()[1:-1], label="Assimilation")
    plt.loglog(x, rad_prior.mean(0).numpy()[1:-1], label="Ens. mean")
    plt.xlabel("Wavenumber")
    plt.ylabel("Power")
    plt.legend()
    plt.title("Spectral Power")
    plt.tight_layout()

    fname = f'{dir}/spectrum_{time}.png'
    plt.savefig(fname, dpi=300)
    plt.close()
    return fname


def save_spectrum_series(series, path, *, members=None, title="Spectral Power",
                         figsize=(4, 4), dpi=None):
    """Log-log radial power spectra for a set of labelled series.

    Args:
        series: {label: (B, R) radial profile}, plotted as the batch mean.
        path: output image path.
        members: optional (M, R), drawn as faint grey traces behind the means.
        title, figsize, dpi: cosmetic.

    Returns:
        The path written.
    """
    import numpy as _np

    fig = plt.figure(figsize=figsize)
    radii = next(iter(series.values())).shape[1]
    x = _np.arange(1, radii - 1)
    if members is not None:
        for m in range(members.shape[0]):
            plt.loglog(x, members[m].numpy()[1:-1], color="gray", alpha=0.1)
    for label, profile in series.items():
        plt.loglog(x, profile.mean(0).numpy()[1:-1], label=label)
    plt.xlabel("Wavenumber")
    plt.ylabel("Power")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if dpi is not None:
        plt.savefig(str(path), dpi=dpi)
    else:
        plt.savefig(str(path))
    plt.close(fig)
    return str(path)


def save_video(forecast, truth, oberrstdev, obs_p, obs_fn, dir, level=0, cmap='jet', lres=None, radar=False, observer=None, unit_factor=1.0, var_vrange=None, imshow_kwargs=None, spread_kwargs=None, unit_label=None):
    """Save an mp4 of truth, obs, ensemble mean/std and up to four members.

    `unit_factor` and `var_vrange` come from DatasetMetadata. `imshow_kwargs` /
    `spread_kwargs` (from `DatasetMetadata.imshow_kwargs()`) override `cmap` and
    `var_vrange`. The obs and std panels always autoscale.
    """
    if observer is not None and radar:
        obs = torch.zeros_like(torch.tensor(truth))
        for ntime in range(truth.shape[0]):
            truth_t = truth[ntime] / unit_factor
            obs_t, obs_mask_t, _ = observer.observe(
                torch.tensor(truth_t), t=ntime)
            obs[ntime, obs_mask_t[0]] = obs_t

        truth = truth[:, level, :, :]  # shape: [time, lat, lon]
        obs = obs[:, level, :, :]  # shape: [time, lat, lon]
        obs = obs.numpy()
    else:
        truth = truth[:, level, :, :]  # shape: [time, lat, lon]
        if lres is None:
            lres = truth.shape[-1]
        obs = obs_fn(torch.tensor(truth)).reshape(
            truth.shape[0], lres, lres).numpy()
        # shape: [time, lat, lon]
        obs += np.random.randn(*obs.shape) * oberrstdev

    forecast = forecast[:, :, level, :, :]  # shape: [time, ens, lat, lon]
    rmse = ((torch.tensor(forecast[:]) - torch.tensor(truth[:]
                                                      ).unsqueeze(1)).pow(2).mean(dim=(2, 3)).sqrt())
    rmse_mean = ((torch.tensor(forecast[:]).mean(
        dim=1) - torch.tensor(truth[:])).pow(2).mean(dim=(1, 2)).sqrt())

    ens_mean = forecast.mean(axis=1)  # shape: [time, lat, lon]
    ens_std = forecast.std(axis=1)  # shape: [time, lat, lon]
    members = forecast[:, :4]  # shape: [time, ens, lat, lon]

    if imshow_kwargs is None:
        if var_vrange is not None and var_vrange[0] is not None:
            vmin, vmax = var_vrange
        else:
            vmin = -20 if cmap == 'jet' else np.min(truth)
            vmax = 20 if cmap == 'jet' else np.max(truth)
        imshow_kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}
    state_cmap = imshow_kwargs.get('cmap', cmap)
    obs_kwargs = {'cmap': state_cmap,
                  'vmin': np.min(obs), 'vmax': np.max(obs)}
    if spread_kwargs is None:
        spread_kwargs = {'cmap': state_cmap}
    spread_kwargs = dict(spread_kwargs)
    spread_kwargs.setdefault('vmin', 0)
    spread_kwargs.setdefault('vmax', ens_std.max())

    fig, axs = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)

    # Bottom row: up to four members.
    n_shown = min(4, members.shape[1])

    # Initial images
    img = []
    titles = ['Truth', 'Obs', 'Mean', 'Std'] + \
        [f'Member {i + 1}' for i in range(n_shown)]

    img.append(axs[0, 0].imshow(truth[0], **imshow_kwargs))
    img.append(axs[0, 1].imshow(obs[0], **obs_kwargs))
    img.append(axs[0, 2].imshow(ens_mean[0], **imshow_kwargs))
    img.append(axs[0, 3].imshow(ens_std[0], **spread_kwargs))

    for i in range(n_shown):
        img.append(axs[1, i].imshow(members[0, i], **imshow_kwargs))
    # Blank out the panels there are no members for.
    for i in range(n_shown, 4):
        axs[1, i].axis('off')

    # Titles and formatting
    for i, ax in enumerate(axs.flatten()[:len(titles)]):
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')

    def set_cbar(im):
        ax = im.axes
        cax = inset_axes(ax,
                         width="100%",
                         height="5%",
                         loc='upper center',
                         bbox_to_anchor=(0, 0.15, 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        return cbar

    sutitle = f't = {0}'
    fig.suptitle(sutitle, fontsize=14)  # , y=1.1)

    def update(frame):
        img[0].set_array(truth[frame])
        img[1].set_array(obs[frame])
        img[2].set_array(ens_mean[frame])
        img[3].set_array(ens_std[frame])
        for i in range(4):
            img[4 + i].set_array(members[frame, i])

        # Update the title with the current time
        sutitle = f't={frame}'
        fig.suptitle(sutitle, fontsize=14, y=1.1)

        # RMSE in the plotted units, which may differ from the logged metrics.
        u = f' {unit_label}' if unit_label else ''
        titles = ['Truth',
                  f'Obs, $\sigma$={oberrstdev}, p={np.round(obs_p*100, 1)}%',
                  f'Mean ({rmse_mean[frame]:.3f}{u})',
                  'Std',
                  ]
        for i in range(4):
            titles.append(f'Member {i+1} ({rmse[frame, i]:.3f}{u})')

        for i, ax in enumerate(axs.flatten()):
            ax.set_title(titles[i], fontsize=14)
            ax.axis('off')

        return img

    fname = f'{dir}/animation.mp4'
    ani = animation.FuncAnimation(
        fig, update, frames=truth.shape[0], interval=10, blit=True)
    ani.save(fname, writer='ffmpeg', fps=10, dpi=300)
    plt.close(fig)
    return fname


def plot_ensemble_prediction(
    traj_rescaled,
    target_rescaled,
    ens_mean,
    ens_std,
    var_name,
    title,
    step,
    var_vrange=None,
    dir=None,
    cmap='jet',
    imshow_kwargs=None,
    spread_kwargs=None,
):
    """Plot ground truth, ensemble mean, spread and members.

    `imshow_kwargs` is the variable's colour scale from
    `DatasetMetadata.imshow_kwargs()` (a cmap plus a `norm` or vmin/vmax);
    `spread_kwargs` is the same for the ensemble std. Without them, `cmap` with
    `var_vrange` is used, or the target's min/max.
    """

    # [ens, var, lat, lon] or [var, lat, lon]
    n_ens = traj_rescaled.shape[0]
    if imshow_kwargs is None:
        if var_vrange is not None:
            vmin, vmax = var_vrange
        else:
            vmin = target_rescaled.min()
            vmax = target_rescaled.max()
        imshow_kwargs = {'cmap': cmap, 'vmin': vmin, 'vmax': vmax}
    if spread_kwargs is None:
        spread_kwargs = {'cmap': imshow_kwargs.get('cmap', cmap)}
    spread_kwargs = dict(spread_kwargs)
    spread_kwargs.setdefault('vmin', ens_std.min())
    spread_kwargs.setdefault('vmax', ens_std.max())

    def _panel(ax, field, panel_title, kwargs):
        ax.set_title(panel_title)
        im = ax.imshow(field, **kwargs)
        # Colorbar on every panel so all panels keep the same size.
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Label every boundary of a discrete norm.
        boundaries = getattr(kwargs.get('norm'), 'boundaries', None)
        if boundaries is not None:
            cbar.set_ticks(list(boundaries))
            cbar.ax.tick_params(labelsize=7)
        ax.axis('off')
        return im

    # Determine number of member rows
    n_member_rows = 1 if n_ens <= 3 else 2
    nrows = 1 + n_member_rows
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # Title for the entire figure
    fig.suptitle(title)

    # First row: Ground truth, ens_mean, ens_std
    _panel(axes[0, 0], target_rescaled, 'Ground Truth', imshow_kwargs)
    _panel(axes[0, 1], ens_mean, 'Ensemble Mean', imshow_kwargs)
    _panel(axes[0, 2], ens_std, 'Ensemble Std', spread_kwargs)

    # Second row: up to 3 ensemble members
    for i in range(min(3, n_ens)):
        _panel(axes[1, i], traj_rescaled[i], f'Ensemble Member {i+1}',
               imshow_kwargs)

    # Third row: next 3 ensemble members (if available)
    if n_member_rows == 2:
        for i in range(3, min(6, n_ens)):
            _panel(axes[2, i-3], traj_rescaled[i], f'Ensemble Member {i+1}',
                   imshow_kwargs)
        # Hide unused axes if less than 6 members
        for j in range(n_ens-3, 3):
            axes[2, j].axis('off')

    plt.tight_layout()
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
        fname = f'{dir}/ensemble_{var_name}_step{step}.png'
        plt.savefig(fname, dpi=300)
    return fig

