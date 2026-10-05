"""Tests that a NaN prior in `DAWIS.assimilate` keeps the `return_traj` return arity.

The NaN check runs before the FMW network is used, so the instance is built with `__new__`.
"""
import pytest
import torch

from assimilation.methods.dawis import DAWIS

WINDOW, N_ENS, D, NX = 6, 3, 2, 8
SCALE = 8.14


def _model():
    m = DAWIS.__new__(DAWIS)
    m.scale = SCALE
    m.device = torch.device("cpu")
    m.init_states = WINDOW
    m.tmin = [0.0] * (WINDOW + 1)
    m.tmin_init = None
    m.tmax = [1.0] * (WINDOW + 1)
    m.generate_target = False
    return m


def _inputs(nan_member=1):
    init_states = torch.randn(WINDOW, N_ENS, D, NX, NX) * SCALE
    x_forecast = torch.randn(N_ENS, D, NX, NX) * SCALE
    x_forecast[nan_member, 0, 0, 0] = float("nan")
    return init_states, x_forecast


def _call(return_traj):
    init_states, x_forecast = _inputs()
    out = _model().assimilate(
        x_forecast, init_states, obs=None, obs_mask=None, obs_fn=None,
        return_traj=return_traj)
    return out, init_states, x_forecast


def test_nan_prior_with_return_traj_returns_a_pair():
    """return_traj=True returns (analysis, posterior) with the prior passed through."""
    out, init_states, x_forecast = _call(return_traj=True)
    assert isinstance(out, tuple) and len(out) == 2
    analysis, posterior = out
    # equal_nan: the prior is handed back untouched, NaNs included.
    torch.testing.assert_close(analysis.cpu(), x_forecast, equal_nan=True)
    # Same layout as the normal return: (n_ens, window, D, X, Y), scalefact units.
    assert posterior.shape == (N_ENS, WINDOW, D, NX, NX)
    assert torch.equal(posterior.cpu(), init_states.permute(1, 0, 2, 3, 4))
    assert not torch.isnan(posterior).any()


def test_nan_prior_without_return_traj_returns_the_prior():
    """return_traj=False returns the prior tensor unchanged."""
    out, _, x_forecast = _call(return_traj=False)
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out.cpu(), x_forecast, equal_nan=True)


@pytest.mark.parametrize("return_traj", [True, False])
def test_return_arity_does_not_depend_on_the_prior_being_finite(return_traj):
    """The return arity depends only on return_traj."""
    out, _, _ = _call(return_traj)
    assert isinstance(out, tuple) is return_traj
