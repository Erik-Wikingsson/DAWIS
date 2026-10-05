import os
import torch
import numpy as np
from netCDF4 import Dataset
from joblib import Parallel, delayed

from data.SQG.sqgturb import SQG
from data.SQG.sqgturb.sqg_gpu import SQGTorch


class _ParamView:
    """netCDF-style attribute/`variables` view over a model_params dict."""

    def __init__(self, params):
        self._p = params
        self.variables = params

    def __getattr__(self, name):
        try:
            return self._p[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _model_params_view(args, model_params):
    """Return model parameters from `model_params`, else from `--data_path`'s .nc."""
    if model_params is not None:
        return _ParamView(model_params)
    data_path = getattr(args, 'data_path', None)
    if not data_path:
        raise ValueError(
            "SQGModel needs model_params (from AssimCase.model_params), or a "
            "--data_path pointing at a trajectory whose sibling .nc holds the "
            "physical parameters."
        )
    if str(data_path).endswith('.npy'):
        data_path = str(data_path)[:-4]
    return Dataset(str(data_path) + '.nc', 'r')


class SQGModel:
    """Numerical SQG model used as an ensemble forecaster.

    Args:
        args: experiment arguments.
        init_states: `(n_ens, 2, ny, nx)` initial ensemble, scaled by scalefact.
        model_params: physical parameters (from `AssimCase.model_params`).
    """

    def __init__(self, args, init_states, model_params=None):
        self.args = args

        # Constants
        threads = int(os.getenv('OMP_NUM_THREADS', '1'))
        diff_efold = None  # use diffusion from climo file

        nc_data = _model_params_view(args, model_params)

        # initialize qg model instances for each ensemble member.
        x = nc_data.variables['x'][:]
        y = nc_data.variables['y'][:]
        x, y = np.meshgrid(x, y)
        dt = nc_data.dt
        if diff_efold == None:
            diff_efold = nc_data.diff_efold

        # parameter used to scale PV to temperature units.
        self.scalefact = nc_data.f*nc_data.theta0/nc_data.g
        pvens = init_states / self.scalefact  # scale to model units

        models = []
        for ens_member in range(args.n_ens):
            models.append(
                SQG(pvens[ens_member],
                    nsq=nc_data.nsq, f=nc_data.f, dt=dt, U=nc_data.U, H=nc_data.H,
                    r=nc_data.r, tdiab=nc_data.tdiab, symmetric=nc_data.symmetric,
                    diff_order=nc_data.diff_order, diff_efold=diff_efold, threads=threads))

        # determine assimilation timesteps
        # NOTE: must match the nature run time steps.
        obtimes = nc_data.variables['t'][:]
        assim_interval = obtimes[1]-obtimes[0]
        assim_timesteps = int(np.round(assim_interval/models[0].dt))

        # initialize model clock
        for ens_member in range(args.n_ens):
            models[ens_member].t = obtimes[0]
            models[ens_member].timesteps = assim_timesteps

        self.models = models

    def forward(self, xens):
        """Forecast one step ahead, X_t -> X_t+1.

        Args:
            xens: `(n_ens, D_state, X, Y)`, scaled by scalefact.

        Returns:
            `(n_ens, D_state, X, Y)`, scaled by scalefact.
        """
        pvens = xens / self.scalefact
        # run forecast ensemble to next analysis time
        pvens_updated = Parallel(n_jobs=-1)(delayed(advance_task)(
            self.models[ens_member], pvens[ens_member]) for ens_member in range(self.args.n_ens))
        for ens_member, updated_pven in enumerate(pvens_updated):
            pvens[ens_member] = updated_pven

        return pvens*self.scalefact


def advance_task(model, ens_member):
    return model.advance(ens_member)


class SQGModelGPU:
    """Differentiable/batched drop-in replacement for `SQGModel`.

    Backed by `SQGTorch`: the ensemble integrates as one batch and gradients can
    flow through `advance`.

    Args:
        args: experiment arguments.
        init_states: `(n_ens, 2, ny, nx)` scaled by scalefact; only the shape
            is used.
        device: torch device. Defaults to `args.device`, else cuda if available.
        precision: 'single' (default, matches `sqg.SQG`) or 'double'.
        model_params: physical parameters (from `AssimCase.model_params`).
    """

    def __init__(self, args, init_states, device=None, precision=None,
                 model_params=None):
        if precision is None:
            precision = getattr(args, 'forward_precision', 'single')
        self.args = args

        nc_data = _model_params_view(args, model_params)
        diff_efold = nc_data.diff_efold
        dt = nc_data.dt
        self.scalefact = nc_data.f * nc_data.theta0 / nc_data.g

        if device is None:
            device = getattr(args, 'device', None)
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.precision = precision
        self.dtype = torch.float64 if precision == 'double' else torch.float32

        pvens = np.asarray(init_states) / self.scalefact
        self.model = SQGTorch(
            pvens,
            nsq=nc_data.nsq, f=nc_data.f, dt=dt, U=nc_data.U, H=nc_data.H,
            r=nc_data.r, tdiab=nc_data.tdiab, symmetric=nc_data.symmetric,
            diff_order=nc_data.diff_order, diff_efold=diff_efold,
            precision=precision, device=self.device)

        obtimes = nc_data.variables['t'][:]
        assim_interval = obtimes[1] - obtimes[0]
        self.model.timesteps = int(np.round(assim_interval / float(self.model.dt)))
        self.model.t = obtimes[0]

    def forward(self, xens):
        """One assimilation interval forward. Mirrors `SQGModel.forward`."""
        is_numpy = not isinstance(xens, torch.Tensor)
        x = torch.as_tensor(np.asarray(xens) if is_numpy else xens,
                            dtype=self.dtype, device=self.device)
        pvens = x / self.scalefact
        with torch.no_grad():
            out = self.model.advance(pvens) * self.scalefact
        return out.cpu().numpy() if is_numpy else out
