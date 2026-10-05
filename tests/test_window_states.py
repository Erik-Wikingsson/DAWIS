"""Tests for `--save_window_states`: x_window[t, lag] is the estimate of time t made at step t + lag.

Uses a fake loop whose states equal their physical time; lag 0 must match
`x_assim` and lag `init_states` must match `x_smooth`.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from netCDF4 import Dataset

from assimilation.assimilate import add_window_variables, create_nc_file


def _args(exp_name, save_window_states, n_times=15, init_states=6, n_ens=3, nx=8):
    return SimpleNamespace(
        nx=nx, n_ens=n_ens, init_states=init_states, window=init_states,
        n_times=n_times, start_time=6, method='DAWIS', obs_fn='linear',
        obs_prob=0.05, obs_sigma=3.0, exp_name=exp_name,
        save_window_states=save_window_states,
    )


def _state(t, n_ens, nx):
    """A state whose value is its own physical time, so writes can be traced."""
    return np.full((n_ens, 2, nx, nx), t, dtype=np.float32)


def _run_fake_loop(args, refresh_every=1):
    """Mimic the assimilation loop's saving and return the result file path.

    refresh_every stands in for --assim_interval.
    """
    w, n_times = args.init_states, args.n_times
    n_ens, nx = args.n_ens, args.nx

    nc, ground_truth, x_assim, x_smooth = create_nc_file(args)
    x_window, window_valid = add_window_variables(nc, args)

    conditioning = np.stack([_state(t, n_ens, nx) for t in range(-w, 0)])
    for ntime in range(n_times):
        xens = torch.tensor(_state(ntime, n_ens, nx))
        window_refreshed = ntime % refresh_every == 0
        if window_refreshed:
            # The posterior keeps each slot's identity: slot j is time ntime-w+j.
            conditioning = np.stack(
                [_state(t, n_ens, nx) for t in range(ntime - w, ntime)])

        x_assim[ntime] = xens.cpu().numpy()
        if ntime >= w:
            x_smooth[ntime - w] = conditioning[0]
        if x_window is not None:
            xens_np = xens.detach().cpu().numpy()
            slots = range(w + 1) if window_refreshed else (w,)
            for j in slots:
                t_slot = ntime - w + j
                if t_slot < 0:
                    continue
                x_window[t_slot, w - j] = xens_np if j == w else conditioning[j]
                window_valid[t_slot, w - j] = 1
        ground_truth[ntime] = np.full((2, nx, nx), ntime, dtype=np.float32)
        conditioning = np.concatenate(
            [conditioning[1:], xens.cpu().numpy()[None]], axis=0)

    x_smooth[n_times - w:n_times] = conditioning
    path = nc.filepath()
    nc.close()
    return path


@pytest.fixture(autouse=True)
def _results_in_tmp(tmp_path, monkeypatch):
    """Point RESULTS_ROOT at tmp_path so result files stay out of the shared directory."""
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))


@pytest.fixture
def written(request):
    """Write one fake run per test and clean the result file up afterwards."""
    created = []

    def _write(save_window_states=True, refresh_every=1, **kwargs):
        args = _args(f'__test_window_{request.node.name}__',
                     save_window_states, **kwargs)
        path = _run_fake_loop(args, refresh_every)
        created.append(path)
        return args, path

    yield _write
    for path in created:
        if os.path.exists(path):
            os.remove(path)


# Flag off

def test_flag_off_writes_the_same_variables_as_before(written):
    _, path = written(save_window_states=False)
    with Dataset(path) as nc:
        assert set(nc.variables) == {'ground_truth', 'x_assim', 'x_smooth'}
        assert 'lag' not in nc.dimensions


def test_flag_off_returns_no_variables():
    args = _args('__unused__', save_window_states=False)
    assert add_window_variables(None, args) == (None, None)


# Flag on

def test_shape_and_lag_axis(written):
    args, path = written()
    with Dataset(path) as nc:
        assert nc.dimensions['lag'].size == args.init_states + 1
        assert nc['x_window'].shape == (
            args.n_times, args.init_states + 1, args.n_ens, 2, args.nx, args.nx)
        assert int(nc.init_states) == args.init_states


def test_every_stored_state_lands_on_its_own_physical_time(written):
    """x_window[t, lag] must hold the estimate *of time t*, whatever the lag."""
    args, path = written()
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        x_window, valid = nc['x_window'][:], nc['window_valid'][:].astype(bool)
    for t in range(args.n_times):
        for lag in range(args.init_states + 1):
            if valid[t, lag]:
                assert np.allclose(x_window[t, lag], t), (t, lag)


def test_validity_mask_is_exactly_the_reachable_lags(written):
    """Lag L at time t exists iff step t + L happened, so only the tail is short."""
    args, path = written()
    w, n_times = args.init_states, args.n_times
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        valid = nc['window_valid'][:].astype(bool)
    expected = np.array([[t + lag <= n_times - 1 for lag in range(w + 1)]
                         for t in range(n_times)])
    assert (valid == expected).all()
    # Every time has the filter; only the last w times miss the full smoother.
    assert valid[:, 0].all()
    assert valid[:, w].sum() == n_times - w


def test_lag0_is_the_filter_and_lag_w_is_the_smoother(written):
    args, path = written()
    w, n_times = args.init_states, args.n_times
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        x_window = nc['x_window'][:]
        assert np.allclose(x_window[:, 0], nc['x_assim'][:])
        # The last w entries of x_smooth are not fully smoothed.
        assert np.allclose(x_window[:n_times - w, w],
                           nc['x_smooth'][:n_times - w])


def test_skipped_assimilation_only_stores_the_filter(written):
    """With no assimilation the posterior window is stale, so only lag 0 is new."""
    args, path = written(refresh_every=2)
    w, n_times = args.init_states, args.n_times
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        x_window, valid = nc['x_window'][:], nc['window_valid'][:].astype(bool)
    # Lag 0 comes from every step, so it is still complete...
    assert valid[:, 0].all()
    # ...and the stored values are still the estimate of their own time.
    for t in range(n_times):
        for lag in range(w + 1):
            if valid[t, lag]:
                assert np.allclose(x_window[t, lag], t), (t, lag)
    # Lags > 0 are only written on the steps that assimilated.
    assert valid[:, 1:].sum() < (n_times - w) * w
