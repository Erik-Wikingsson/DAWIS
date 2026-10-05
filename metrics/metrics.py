# Third-party
import torch
import numpy as np


def get_metric(metric_name):
    """
    Get a defined metric with given name

    metric_name: str, name of the metric

    Returns:
    metric: function implementing the metric
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, mean_vars):
    """
    Masks and (optionally) reduces entry-wise metric values

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., D, X, Y), prediction
    target: (..., D, X, Y), target
    mask: (X, Y), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension should be reduced (mean over X, Y)
    mean_vars: boolean, if variable dimension should be reduced (mean over D)

    Returns:
    metric_val: One of (...,), (..., D), (..., X, Y), (..., D, X, Y),
    depending on reduction arguments.
    """
    # Only keep grid nodes in mask
    if mask is not None:
        metric_entry_vals = metric_entry_vals[..., mask]  # (..., D, N)

        if mean_vars:
            metric_entry_vals = torch.mean(
                metric_entry_vals, dim=-2
            )

        if average_grid:
            metric_entry_vals = torch.mean(
                metric_entry_vals, dim=-1
            )
    else:
        if mean_vars:
            metric_entry_vals = torch.mean(
                metric_entry_vals, dim=-3
            )

        if average_grid:
            metric_entry_vals = torch.mean(
                metric_entry_vals, dim=(-2, -1)
            )

    return metric_entry_vals


def rmse(pred, target, mask=None, average_grid=True, mean_vars=True, **kwargs):
    """
    Root Mean Squared Error

    (...,) is any number of batch dimensions, potentially different but broadcastable

    pred: (..., D, X, Y), prediction
    target: (..., D, X, Y), target
    mask: (X, Y), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension should be reduced (mean over X, Y)
    mean_vars: boolean, if variable dimension should be reduced (mean over D)

    Returns:
    metric_val: One of (...,), (..., D), (..., X, Y), (..., D, X, Y),
    depending on reduction arguments.
    """

    return torch.sqrt(mse(pred, target, mask, average_grid, mean_vars))


def mse(pred, target, mask=None, average_grid=True, mean_vars=True):
    """
    Mean Squared Error

    (...,) is any number of batch dimensions, potentially different but broadcastable

    pred: (..., D, X, Y), prediction
    target: (..., D, X, Y), target
    mask: (X, Y), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension should be reduced (mean over X, Y)
    mean_vars: boolean, if variable dimension should be reduced (mean over D)

    Returns:
    metric_val: One of (...,), (..., D), (..., X, Y), (..., D, X, Y),
    depending on reduction arguments.
    """
    entry_mse = torch.nn.functional.mse_loss(
        pred, target, reduction="none"
    )

    return mask_and_reduce_metric(
        entry_mse,
        mask=mask,
        average_grid=average_grid,
        mean_vars=mean_vars,
    )


def mae(pred, target, mask=None, average_grid=True, mean_vars=True):
    """
    Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different but broadcastable

    pred: (..., D, X, Y), prediction
    target: (..., D, X, Y), target
    mask: (X, Y), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension should be reduced (mean over X, Y)
    mean_vars: boolean, if variable dimension should be reduced (mean over D)

    Returns:
    metric_val: One of (...,), (..., D), (..., X, Y), (..., D, X, Y),
    depending on reduction arguments.
    """
    entry_mae = torch.nn.functional.l1_loss(
        pred, target, reduction="none"
    )

    return mask_and_reduce_metric(
        entry_mae,
        mask=mask,
        average_grid=average_grid,
        mean_vars=mean_vars,
    )


def crps_ens(
    pred,
    target,
    mask=None,
    average_grid=True,
    mean_vars=True,
    ens_dim=1,
):
    """
    (Negative) Continuous Ranked Probability Score (CRPS)
    Unbiased estimator from samples. See e.g. Weatherbench 2.

    (..., M, ...,) is any number of batch dimensions, including ensemble dimension M

    pred: (..., M, D, X, Y), prediction
    target: (..., D, X, Y), target
    mask: (X, Y), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension should be reduced (mean over X, Y)
    mean_vars: boolean, if variable dimension should be reduced (mean over D)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    if pred.shape == target.shape:
        pred = pred.unsqueeze(ens_dim)

    num_ens = pred.shape[ens_dim]  # Number of ensemble members
    if num_ens == 1:
        # With one sample CRPS reduces to MAE
        return mae(
            pred.squeeze(ens_dim),
            target,
            mask=mask,
            average_grid=average_grid,
            mean_vars=mean_vars,
        )

    if num_ens == 2:
        mean_mae = torch.mean(
            torch.abs(pred - target.unsqueeze(ens_dim)), dim=ens_dim
        )

        # Use simpler estimator
        pair_diffs_term = -0.5 * torch.abs(
            pred.select(ens_dim, 0) - pred.select(ens_dim, 1)
        )

        crps_estimator = mean_mae + pair_diffs_term
    elif num_ens < 10:
        # This is the rank-based implementation with O(M*log(M)) compute and
        # O(M) memory. See Zamo and Naveau and WB2 for explanation.
        # For smaller ensemble we can compute all of this directly in memory.
        mean_mae = torch.mean(
            torch.abs(pred - target.unsqueeze(ens_dim)), dim=ens_dim
        )

        # Ranks start at 1, two argsorts will compute entry ranks
        ranks = pred.argsort(dim=ens_dim).argsort(ens_dim) + 1

        pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
            (num_ens + 1 - 2 * ranks) * pred,
            dim=ens_dim,
        )

        crps_estimator = mean_mae + pair_diffs_term
    else:
        # For large ensembles we batch this over the variable dimension
        crps_res = []
        # Pred is of shape (..., M, D, X, Y)
        for var_i in range(pred.shape[-3]):
            pred_var = pred[..., var_i, :, :]
            target_var = target[..., var_i, :, :]

            mean_mae = torch.mean(
                torch.abs(pred_var - target_var.unsqueeze(ens_dim)), dim=ens_dim
            )

            # Ranks start at 1, two argsorts will compute entry ranks
            ranks = pred_var.argsort(dim=ens_dim).argsort(ens_dim) + 1

            pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
                (num_ens + 1 - 2 * ranks) * pred_var,
                dim=ens_dim,
            )
            crps_res.append(mean_mae + pair_diffs_term)

        crps_estimator = torch.stack(crps_res, dim=-3)  # (..., D, X, Y)

    return mask_and_reduce_metric(crps_estimator, mask, average_grid, mean_vars)


def afcrps_ens(
    pred,
    target,
    mask=None,
    average_grid=True,
    mean_vars=True,
    ens_dim=1,
    alpha=0.95,
    beta=1.0,
):
    """
    (Negative) Almost Fair Continuous Ranked Probability Score (CRPS)

    (..., M, ...,) is any number of batch dimensions, including ensemble dimension M

    pred: (..., M, D, X, Y), prediction
    target: (..., D, X, Y), target
    mask: (X, Y), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension should be reduced (mean over X, Y)
    mean_vars: boolean, if variable dimension should be reduced (mean over D)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    if pred.shape == target.shape:
        pred = pred.unsqueeze(ens_dim)

    num_ens = pred.shape[ens_dim]  # Number of ensemble members
    eps = (1-alpha) / num_ens
    if num_ens == 1:
        crps_estimator = torch.mean(
            torch.abs(pred - target.unsqueeze(ens_dim))**beta, dim=ens_dim
        )

    elif num_ens == 2 or beta != 1.0:
        mean_mae = torch.mean(
            torch.abs(pred - target.unsqueeze(ens_dim))**beta, dim=ens_dim
        )

        # Use simpler estimator
        pair_diffs_term = -0.5 * torch.abs(
            pred.select(ens_dim, 0) - pred.select(ens_dim, 1)
        )**beta
        crps_estimator = mean_mae + (1-eps) * pair_diffs_term
    elif num_ens < 10:
        # This is the rank-based implementation with O(M*log(M)) compute and
        # O(M) memory. See Zamo and Naveau and WB2 for explanation.
        # For smaller ensemble we can compute all of this directly in memory.
        mean_mae = torch.mean(
            torch.abs(pred - target.unsqueeze(ens_dim)), dim=ens_dim
        )

        # Ranks start at 1, two argsorts will compute entry ranks
        ranks = pred.argsort(dim=ens_dim).argsort(ens_dim) + 1

        pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
            (num_ens + 1 - 2 * ranks) * pred,
            dim=ens_dim,
        )

        crps_estimator = mean_mae + (1-eps) * pair_diffs_term
    else:
        # For large ensembles we batch this over the variable dimension
        crps_res = []
        # Pred is of shape (..., M, D, X, Y)
        for var_i in range(pred.shape[-3]):
            pred_var = pred[..., var_i, :, :]
            target_var = target[..., var_i, :, :]

            mean_mae = torch.mean(
                torch.abs(pred_var - target_var.unsqueeze(ens_dim)), dim=ens_dim
            )

            # Ranks start at 1, two argsorts will compute entry ranks
            ranks = pred_var.argsort(dim=ens_dim).argsort(ens_dim) + 1

            pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
                (num_ens + 1 - 2 * ranks) * pred_var,
                dim=ens_dim,
            )
            crps_res.append(mean_mae + (1-eps) * pair_diffs_term)

        crps_estimator = torch.stack(crps_res, dim=-3)  # (..., D, X, Y)

    return mask_and_reduce_metric(crps_estimator, mask, average_grid, mean_vars)


def nll(pred, target, pred_std=None, mask=None, average_grid=True, mean_vars=True):
    """
    Negative Log Likelihood loss, for isotropic Gaussian likelihood

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    if pred_std is None:
        pred_std = torch.ones_like(pred)
    # Broadcast pred_std if shaped (d_state,), done internally in Normal class
    dist = torch.distributions.Normal(pred, pred_std)  # (..., N, d_state)
    entry_nll = -dist.log_prob(target)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_nll, mask=mask, average_grid=average_grid, mean_vars=mean_vars
    )


def spread_squared(
    pred,
    mask=None,
    average_grid=True,
    mean_vars=True,
    ens_dim=1,
):
    """
    (Squared) spread of ensemble.
    Similarly to RMSE, we want to take sqrt after spatial and sample averaging,
    so we need to average the squared spread.

    (..., M, ...,) is any number of batch dimensions, including ensemble dimension M

    pred: (..., M, D, X, Y), prediction
    target: (..., D, X, Y), target
    mask: (X, Y), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension should be reduced (mean over X, Y)
    mean_vars: boolean, if variable dimension should be reduced (mean over D)
    ens_dim: batch dimension where ensemble members are laid out, to reduce over

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    entry_var = torch.var(pred, dim=ens_dim)
    return mask_and_reduce_metric(entry_var, mask, average_grid, mean_vars)


def spread_skill_ratio(
    pred,
    target,
    mask=None,
    average_grid=True,
    mean_vars=True,
    ens_dim=1
):
    """
    Spread skill ratio of ensemble.

    (..., M, ...,) is any number of batch dimensions, including ensemble dimension M

    pred: (..., M, D, X, Y), prediction
    target: (..., D, X, Y), target
    mask: (X, Y), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension should be reduced (mean over X, Y)
    mean_vars: boolean, if variable dimension should be reduced (mean over D)
    ens_dim: batch dimension where ensemble members are laid out, to reduce over

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """

    spread_squared_tensor = spread_squared(pred, mask, ens_dim=ens_dim)
    ens_mean = torch.mean(pred, dim=ens_dim)
    ens_mse_tensor = mse(ens_mean, target, mask)

    spread = torch.sqrt(spread_squared_tensor)
    skill = torch.sqrt(ens_mse_tensor)

    # Include finite sample correction
    spsk_ratios = np.sqrt(
        (pred.shape[ens_dim] + 1) / pred.shape[ens_dim]
    ) * (
        spread / skill
    )

    return spsk_ratios


def power_spectrum(x):
    """
    Compute 2D power spectrum for batched input.
    x shape: (B, C, H, W)
    returns: (B, H, W) averaged over channels
    """
    fft2 = torch.fft.fft2(x, norm="ortho")      # (B, C, H, W)
    fftshift = torch.fft.fftshift(fft2, dim=(-2, -1))
    psd2D = torch.abs(fftshift) ** 2
    return psd2D.mean(1)   # average over channels -> (B, H, W)


def radial_average(psd2D):
    """
    Radially average 2D power spectrum.
    psd2D shape: (..., H, W) - arbitrary leading dimensions with 2D data at the end
    returns: (..., R) radial profiles, where R = max(H,W)//2
    """
    # Get shape information
    psd2D = psd2D.cpu()  # Work on CPU for bincount
    original_shape = psd2D.shape
    H, W = original_shape[-2], original_shape[-1]

    # Reshape to (-1, H, W) to handle all leading dimensions as one batch dimension
    batch_size = np.prod(original_shape[:-2]) if original_shape[:-2] else 1
    reshaped_psd = psd2D.reshape(batch_size, H, W)

    # Calculate radial coordinates
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).to(psd2D.device)
    r = r.to(torch.int64)

    R = r.max().item() + 1

    # Process each item in the batch
    radial_profiles = []
    for b in range(batch_size):
        tbin = torch.bincount(
            r.flatten(), weights=reshaped_psd[b].flatten(), minlength=R)
        nr = torch.bincount(r.flatten(), minlength=R)
        radial_profiles.append(tbin / torch.clamp(nr, min=1))

    # Stack and reshape back to original leading dimensions
    stacked_profiles = torch.stack(radial_profiles)  # (batch_size, R)

    # Reshape back to match original leading dimensions
    if original_shape[:-2]:
        return stacked_profiles.reshape(*original_shape[:-2], R)
    else:
        return stacked_profiles  # Just (R) if input was (H, W)


def log_spectral_distance(
        pred,
        target,
        mask=None,
        average_grid=True,
        mean_vars=True,
        ens_dim=1):
    """
    Compute log spectral distance between forecast and truth.
    forecast shape: (B, M, H, W)
    truth shape: (B, H, W)
    returns: (B,) log spectral distance
    """
    ps_truth = power_spectrum(target.unsqueeze(ens_dim).cpu())
    ps_forecast = power_spectrum(pred.cpu())
    ps_prior = power_spectrum(torch.mean(
        pred, dim=ens_dim, keepdim=True).cpu())

    # Radially averaged
    rad_target = radial_average(ps_truth)  # (B, R)
    rad_pred = radial_average(ps_forecast)
    rad_mean = radial_average(ps_prior)

    lsd = torch.sqrt(torch.mean((torch.log(rad_pred[:, 1:-1] + 1e-8) -
                                 torch.log(rad_target[:, 1:-1] + 1e-8))**2))
    return lsd, rad_target, rad_pred, rad_mean


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    "crps_ens": crps_ens,
    "spread_squared": spread_squared,
}
