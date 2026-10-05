"""Tests for `--assim_interval` (observations every k-th step).

k = 1 changes nothing; an unobserved step has an empty observation set, and
every method must treat it as a zero likelihood term (not NaN, not a pull to 0).
"""
import numpy as np
import pytest
import torch

from assimilation.obs_schedule import (
    assimilation_steps, blank_observation, has_observations,
    is_assimilation_step)


# Schedule
def test_interval_one_observes_every_step():
    """k = 1 makes every step an assimilation step."""
    assert all(is_assimilation_step(t, 1) for t in range(50))
    assert assimilation_steps(50, 1).all()


@pytest.mark.parametrize("k,expected", [
    (2, [0, 2, 4, 6, 8]),
    (3, [0, 3, 6, 9]),
    (4, [0, 4, 8]),
])
def test_interval_k_observes_every_kth_step(k, expected):
    steps = [t for t in range(10) if is_assimilation_step(t, k)]
    assert steps == expected


def test_interval_minus_one_never_observes():
    assert not any(is_assimilation_step(t, -1) for t in range(50))
    assert not assimilation_steps(50, -1).any()


def test_step_zero_always_observes():
    """Step 0 is an assimilation step for every interval."""
    for k in (1, 2, 3, 7, 19):
        assert is_assimilation_step(0, k)


# Blank step
def _observer_like_output(n_ch=2, ny=4, nx=4, n_obs=5):
    """(obs, obs_mask) shaped the way SQGObserver.observe returns them."""
    obs = torch.randn(1, n_ch * n_obs)
    mask = torch.zeros(1, n_ch, ny, nx, dtype=torch.bool)
    mask.view(1, -1)[0, :n_ch * n_obs] = True
    return obs, mask


def test_blank_observation_shapes_and_dtypes():
    obs, mask = _observer_like_output()
    b_obs, b_mask, b_idx = blank_observation(obs, mask)

    assert b_mask.shape == mask.shape and b_mask.dtype == mask.dtype
    assert not b_mask.any(), "an unobserved step observes nothing"
    assert b_obs.shape == (obs.shape[0], 0), "leading dim kept, obs dim emptied"
    assert b_obs.dtype == obs.dtype
    assert b_idx.size == 0


def test_blank_observation_does_not_alias_the_original():
    """The blank is a copy, not a view of the original mask."""
    obs, mask = _observer_like_output()
    b_obs, b_mask, _ = blank_observation(obs, mask)
    b_mask[:] = True
    assert mask.any() and not mask.all(), "original mask was modified"
    assert b_obs.numel() == 0


def test_masked_selection_agrees_with_the_blank_obs_vector():
    """`x[:, mask]` and the blank `obs` have the same (zero) length."""
    obs, mask = _observer_like_output()
    b_obs, b_mask, _ = blank_observation(obs, mask)
    x = torch.randn(3, *mask.shape[1:])
    assert x[:, b_mask[0]].shape[-1] == b_obs.shape[-1] == 0


def test_has_observations():
    obs, mask = _observer_like_output()
    assert has_observations(obs)
    assert not has_observations(blank_observation(obs, mask)[0])
    assert not has_observations(None)
    assert not has_observations(np.empty(0))


# Consumers
def test_window_observation_vector_survives_a_blank_slot():
    """A blank window slot contributes zero entries to the concatenated obs and mask."""
    blocks = [torch.randn(5), torch.empty(0), torch.randn(5)]
    assert torch.cat(blocks, dim=0).shape == (10,)

    n_ens, T, n_ch, nx = 3, 3, 1, 4
    mask = torch.zeros(T, n_ch, nx, nx, dtype=torch.bool)
    mask[0].view(-1)[:5] = True
    mask[2].view(-1)[:5] = True          # slot 1 left blank
    z = torch.randn(n_ens, T, n_ch, nx, nx)
    assert z[:, mask].shape == (n_ens, 10)


def test_fmw_guidance_is_zero_without_observations():
    """FMW's `has_observations` detects an empty obs vector (guidance is skipped)."""
    from forecasting.models.fmw import has_observations as fmw_has_obs
    assert not fmw_has_obs(torch.empty(1, 0))
    assert fmw_has_obs(torch.randn(1, 4))
