"""Which steps carry observations, and what a step without them looks like.

`--assim_interval k` means observations arrive every k-th step. On a step
without them:

- with a separate forward model (`x_forecast` exists), the analysis is skipped;
- otherwise (`--forward_model none`/`dawis`, FlowDAS, the smoothers) the
  analysis runs against an empty observation mask, i.e. an unguided draw.

An empty mask is used rather than zeroed observations, which would still act
as observations. `smoothing.py` rejects `--obs_fn avg` with an interval != 1.
"""

from __future__ import annotations

import numpy as np
import torch


def is_assimilation_step(ntime: int, assim_interval: int) -> bool:
    """Does step `ntime` carry observations?

    1 (the default) assimilates every step; -1 never assimilates.
    """
    if assim_interval == -1:
        return False
    return ntime % assim_interval == 0


def assimilation_steps(n_times: int, assim_interval: int) -> np.ndarray:
    """The steps that carry observations, as a boolean array of length n_times."""
    return np.array(
        [is_assimilation_step(t, assim_interval) for t in range(n_times)],
        dtype=bool)


def blank_observation(obs, obs_mask):
    """The same step, observed nowhere: `(obs, obs_mask, indxob)`.

    Shapes and dtypes follow the observer's; `obs` keeps its leading dims and
    has an empty observation dim, matching `x[:, mask]` for an all-False mask.
    """
    blank_mask = torch.zeros_like(obs_mask)
    blank_obs = obs[..., :0].clone() if torch.is_tensor(obs) else obs[..., :0].copy()
    blank_indxob = np.empty(0, dtype=np.int64)
    return blank_obs, blank_mask, blank_indxob


def has_observations(obs) -> bool:
    """Is there anything to condition on?

    Guidance checks this first: with no observations the likelihood gradient
    is zero, and the MMPS/DPS solves would return NaN on a zero-size system.
    """
    if obs is None:
        return False
    if torch.is_tensor(obs):
        return obs.numel() > 0
    return np.asarray(obs).size > 0
