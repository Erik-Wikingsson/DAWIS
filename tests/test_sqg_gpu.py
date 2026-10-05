"""Agreement tests between the NumPy SQG model (`sqg.py`) and its PyTorch port (`sqg_gpu.py`).

SQG is chaotic, so only short-horizon comparisons use tight tolerances.
"""
import numpy as np
import pytest
import torch
from netCDF4 import Dataset

import assimilation.experiments as experiments
from data.SQG.sqgturb import SQG
from data.SQG.sqgturb.sqg_gpu import SQGTorch

N = 64
SEED = 0


def _model_params():
    """Model parameters from the nature-run climatology file, as the DA code does."""
    from data.paths import MissingRootError
    try:
        files = experiments.nc_files_64
    except (MissingRootError, FileNotFoundError) as e:
        pytest.skip(f"SQG data not present: {e}")
    if not files:
        pytest.skip("SQG test split is empty")
    with Dataset(files[0] + '.nc', 'r') as nc:
        params = dict(
            nsq=float(nc.nsq), f=float(nc.f), dt=float(nc.dt), U=float(nc.U),
            H=float(nc.H), r=float(nc.r), tdiab=float(nc.tdiab),
            symmetric=bool(nc.symmetric), diff_order=int(nc.diff_order),
            diff_efold=float(nc.diff_efold),
        )
        obtimes = nc.variables['t'][:]
        assim_timesteps = int(np.round((obtimes[1] - obtimes[0]) / params['dt']))
    return params, assim_timesteps


def _initial_pv(scale=1.0e-5):
    rng = np.random.default_rng(SEED)
    return rng.normal(0.0, scale, size=(2, N, N))


def _first_trajectory():
    return experiments.nc_files_64[0]


@pytest.fixture(scope='module')
def params():
    return _model_params()


def _build_pair(params, pv):
    p, _ = params
    ref = SQG(pv.copy(), precision='double', **p)
    tst = SQGTorch(pv.copy(), precision='double', device='cpu', **p)
    return ref, tst


def test_invert_agreement(params):
    """Spectral inversion PV -> streamfunction."""
    pv = _initial_pv()
    ref, tst = _build_pair(params, pv)
    psi_ref = ref.invert(ref.pvspec)
    psi_tst = tst.invert(tst.pvspec).squeeze(0).numpy()
    assert np.abs(psi_ref - psi_tst).max() < 1e-12 * np.abs(psi_ref).max()


def test_xyderiv_agreement(params):
    """Dealiased spectral derivatives (2/3-rule padding).

    Float32-level tolerance: `sqg.py` keeps its derivative operators in complex64.
    """
    pv = _initial_pv()
    ref, tst = _build_pair(params, pv)
    xr, yr = ref.xyderiv(ref.pvspec)
    xt, yt = tst.xyderiv(tst.pvspec)
    xt, yt = xt.squeeze(0).numpy(), yt.squeeze(0).numpy()
    assert np.abs(xr - xt).max() < 1e-6 * np.abs(xr).max()
    assert np.abs(yr - yt).max() < 1e-6 * np.abs(yr).max()


def test_gettend_agreement(params):
    """Full PV tendency: Jacobian + thermal relaxation + Ekman damping."""
    pv = _initial_pv()
    ref, tst = _build_pair(params, pv)
    t_ref = ref.gettend(ref.pvspec)
    t_tst = tst.gettend(tst.pvspec).squeeze(0).numpy()
    assert np.abs(t_ref - t_tst).max() < 1e-12 * np.abs(t_ref).max()


def test_single_step_agreement(params):
    """One RK4 step + integrating-factor hyperdiffusion, float64."""
    p, _ = params
    pv = _initial_pv()
    ref, tst = _build_pair(params, pv)
    ref.timesteps = tst.timesteps = 1
    pv_ref = ref.advance(pv.copy())
    pv_tst = tst.advance(pv.copy()).squeeze(0).numpy()
    err = np.abs(pv_ref - pv_tst).max()
    assert err < 1e-10, f"single-step mismatch {err:.3e}"


def test_assim_interval_agreement(params):
    """One full assimilation interval (many RK4 steps)."""
    p, assim_timesteps = params
    pv = _initial_pv()
    ref, tst = _build_pair(params, pv)
    ref.timesteps = tst.timesteps = assim_timesteps
    pv_ref = ref.advance(pv.copy())
    pv_tst = tst.advance(pv.copy()).squeeze(0).numpy()
    rel = np.abs(pv_ref - pv_tst).max() / np.abs(pv_ref).max()
    assert rel < 1e-10, f"relative mismatch over one interval {rel:.3e}"


def test_free_run_agreement(params):
    """Ten assimilation intervals, with a loose tolerance for chaotic round-off growth."""
    p, assim_timesteps = params
    pv = _initial_pv()
    ref, tst = _build_pair(params, pv)
    ref.timesteps = tst.timesteps = assim_timesteps
    x_ref, x_tst = pv.copy(), pv.copy()
    for _ in range(10):
        x_ref = ref.advance(x_ref)
        x_tst = tst.advance(torch.as_tensor(x_tst)).squeeze(0).numpy()
    rel = np.abs(x_ref - x_tst).max() / np.abs(x_ref).max()
    assert rel < 1e-6, f"relative mismatch over 10 intervals {rel:.3e}"


def test_batched_matches_serial(params):
    """Batched ensemble integration equals looping members one at a time."""
    p, assim_timesteps = params
    rng = np.random.default_rng(SEED)
    ens = rng.normal(0.0, 1.0e-5, size=(5, 2, N, N))
    tst = SQGTorch(ens, precision='double', device='cpu', **p)
    tst.timesteps = assim_timesteps
    batched = tst.advance(ens).numpy()
    for i in range(ens.shape[0]):
        single = tst.advance(ens[i]).numpy()
        assert np.abs(batched[i] - single).max() < 1e-12 * np.abs(single).max()


def test_single_precision_does_not_overflow(params):
    """Single precision (the `SQGModelGPU` default) stays finite over ten intervals.

    `mu` must be clipped with the model dtype's epsilon, or `Hovermu` overflows float32.
    """
    p, assim_timesteps = params
    pv = np.load(_first_trajectory() + '.npy')[0].astype(np.float64)
    tst = SQGTorch(pv.copy(), precision='single', device='cpu', **p)
    tst.timesteps = assim_timesteps
    assert torch.isfinite(tst.Hovermu).all()
    assert tst.Hovermu.max() < 1e12, (
        f"Hovermu {tst.Hovermu.max():.3e} too large -- mu clipped with the wrong eps")
    x = torch.as_tensor(pv.copy())
    for i in range(10):
        x = tst.advance(x)
        assert torch.isfinite(x).all(), f"non-finite state after interval {i}"
    assert x.abs().max() < 1e5


def test_single_precision_tracks_numpy(params):
    """Single precision agrees with the NumPy model, which also defaults to single."""
    p, assim_timesteps = params
    pv = np.load(_first_trajectory() + '.npy')[0].astype(np.float64)
    ref = SQG(pv.copy(), precision='single', **p)
    tst = SQGTorch(pv.copy(), precision='single', device='cpu', **p)
    ref.timesteps = tst.timesteps = assim_timesteps
    x_ref = ref.advance(pv.copy())
    x_tst = tst.advance(pv.copy()).squeeze(0).numpy()
    rel = np.abs(x_ref - x_tst).max() / np.abs(x_ref).max()
    assert rel < 1e-4, f"single-precision relative mismatch {rel:.3e}"


def test_advance_is_differentiable(params):
    """Gradients flow through advance()."""
    p, _ = params
    pv = torch.tensor(_initial_pv(), dtype=torch.float64, requires_grad=True)
    tst = SQGTorch(pv.detach(), precision='double', device='cpu', **p)
    tst.timesteps = 4
    out = tst.advance(pv)
    loss = (out ** 2).sum()
    grad, = torch.autograd.grad(loss, pv)
    assert torch.isfinite(grad).all()
    assert grad.abs().max() > 0
