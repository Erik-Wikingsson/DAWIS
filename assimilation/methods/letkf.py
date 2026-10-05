import torch
import numpy as np
from netCDF4 import Dataset

from ..enkf_utils import cartdist, enkf_update, gaspcohn, bulk_ensrf
from assimilation.methods.method import AssimilationMethod


class _ParamView:
    """Attribute/`variables` view over a model_params dict.

    Lets the existing `self.nc_climo.nsq` / `.variables['x']` style access in
    this file keep working without reopening a netCDF file.
    """

    def __init__(self, params):
        self._p = params
        self.variables = params

    def __getattr__(self, name):
        try:
            return self._p[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class LETKF(AssimilationMethod):
    def __init__(self, args, nobs, model_params=None, geometry=None,
                 unit_factor=None, num_channels=None):
        super().__init__(args)
        self.nobs = nobs

        # horizontal covariance localization length scale in meters.
        self.hcovlocal_scale = args.hcovlocal_scale
        self.covinflate1 = args.covinflate1
        self.covinflate2 = args.covinflate2

        self.use_letkf = True  # False use LETKF
        self.global_enkf = False  # False # global EnSRF solve

        # LETKF needs only grid geometry (for localization) and a unit factor,
        # so it can run with any forward model.
        if geometry is None and model_params is not None:
            geometry = {
                "x": model_params.get("x"),
                "y": model_params.get("y"),
                "Lx": float(model_params["L"]) if "L" in model_params else None,
                "Ly": float(model_params["L"]) if "L" in model_params else None,
                "rossby_radius_m": None,
                # This fallback is only reached with SQG model parameters,
                # which describe a doubly periodic domain.
                "periodic": (True, True),
            }
        if geometry is None:
            raise ValueError(
                "LETKF needs AssimCase.grid_geometry (x/y coordinates in metres "
                "and the domain extent) for covariance localization."
            )
        self.geometry = geometry
        self.model_params = model_params
        self.nc_climo = _ParamView(model_params) if model_params else None
        self.num_channels = num_channels if num_channels is not None else 2

        # np.float64 (not a Python float) so the analysis runs in float64
        # under NEP 50 promotion.
        if unit_factor is None:
            unit_factor = model_params["scalefact"] if model_params else 1.0
        self.scalefact = np.float64(unit_factor)

        x = geometry['x']
        y = geometry['y']
        self.Lx = geometry.get('Lx')
        self.Ly = geometry.get('Ly', self.Lx)
        # Per-axis periodicity for distance calculations; a non-periodic axis
        # must not wrap distances around the domain (SEVIR).
        self.periodic = tuple(geometry.get('periodic', (False, False)))
        self.x, self.y = np.meshgrid(x, y)
        self.nx = len(x)
        self.ny = len(y)

        # Vertical localization scale; None for single-variable datasets.
        rossby = self.geometry.get('rossby_radius_m')
        if rossby is None and self.nc_climo is not None:
            # Keep the float32 chain (no float()) for bitwise-stable localization.
            rossby = (np.sqrt(self.nc_climo.nsq)
                      * self.nc_climo.H / self.nc_climo.f)
        if rossby is None:
            self.vcovlocal_fact = np.array(1.0)
        else:
            self.vcovlocal_fact = gaspcohn(np.array(rossby/self.hcovlocal_scale))

        self.oberrvar = args.obs_sigma**2*np.ones(nobs, float)
        self.covlocal = np.empty((self.ny, self.nx), float)

        if not self.use_letkf:
            self.obcovlocal = np.empty((nobs, nobs), float)
        else:
            self.obcovlocal = None
        if self.global_enkf:  # model-space localization matrix
            n = 0
            self.covlocal_modelspace = np.empty(
                (self.nx*self.ny, self.nx*self.ny), float)
            x1 = x.reshape(self.nx*self.ny)
            y1 = y.reshape(self.nx*self.ny)
            for n in range(self.nx*self.ny):
                dist = cartdist(x1[n], y1[n], x1, y1,
                                self.Lx, self.Ly, self.periodic)
                self.covlocal_modelspace[n, :] = gaspcohn(
                    dist/self.hcovlocal_scale)

        self.covlocal = np.empty((self.ny, self.nx), float)
        self.covlocal_tmp = np.empty((self.nobs, self.nx*self.ny), float)

    def assimilate(self, x_forecast, obs, indxob, obs_fn, ntime, **kwargs):
        """
        Perform LETKF assimilation step.
        Args:
            x_forecast: Forecast state, shape (n_ens, state_dim).
            obs: Observations, shape (obs_dim,).
            indxob: Indices of observed locations.
            obs_fn: Observation operator function.
            ntime: Current time step.
            **kwargs: Additional keyword arguments that can be passed to specific implementations.

        Returns:
            x_analysis: Analyzed state after assimilation, shape (n_ens, state_dim). NOTE: Should be scaled by scalefact.
        """
        pvens = x_forecast.cpu().numpy() / self.scalefact
        pvob = obs.reshape(self.num_channels, self.nobs).cpu().numpy()

        xob = self.x.ravel()[indxob]
        yob = self.y.ravel()[indxob]

        if not self.args.fixed_obs or ntime == 0:
            for nob in range(self.nobs):
                dist = cartdist(xob[nob], yob[nob], self.x,
                                self.y, self.Lx, self.Ly, self.periodic)
                self.covlocal = gaspcohn(dist/self.hcovlocal_scale)
                self.covlocal_tmp[nob] = self.covlocal.ravel()
                dist = cartdist(xob[nob], yob[nob], xob, yob,
                                self.Lx, self.Ly, self.periodic)
                if not self.use_letkf:
                    self.obcovlocal[nob] = gaspcohn(dist/self.hcovlocal_scale)

        # first-guess spread (need later to compute inflation factor)
        fsprd = ((pvens - pvens.mean(axis=0)) **
                 2).sum(axis=0)/(self.args.n_ens-1)

        # compute forward operator.
        # hxens is ensemble in observation space.
        hxens = np.empty((self.args.n_ens, self.num_channels, self.nobs), float)
        for nanal in range(self.args.n_ens):
            if kwargs["avg"]:
                hxens[nanal] = obs_fn(torch.tensor(self.scalefact*pvens[nanal])).cpu().numpy()  # surface pv obs
            else:
                for k in range(self.num_channels):
                    hxens[nanal, k, ...] = obs_fn(torch.tensor(
                        self.scalefact*pvens[nanal, k, ...].ravel()[indxob])).cpu().numpy()  # surface pv obs

        hxensmean_b = hxens.mean(axis=0)
        obsprd = ((hxens-hxensmean_b)**2).sum(axis=0)/(self.args.n_ens-1)

        # innov stats for background
        obfits = pvob - hxensmean_b
        obfits_b = (obfits**2).mean()
        obbias_b = obfits.mean()
        obsprd_b = obsprd.mean()
        pvensmean_b = pvens.mean(axis=0).copy()

        # EnKF update
        # create 1d state vector.
        xens = pvens.reshape(self.args.n_ens, self.num_channels, self.nx*self.ny)

        # hxens,pvob are in PV units, xens is not
        if self.global_enkf and not self.use_letkf:
            xens = bulk_ensrf(xens, indxob, pvob, self.oberrvar,
                              self.covlocal_modelspace, self.vcovlocal_fact, self.scalefact)
        else:
            xens = enkf_update(xens, hxens, pvob, self.oberrvar, self.covlocal_tmp,
                               self.vcovlocal_fact, obcovlocal=self.obcovlocal)

        # back to 3d state vector
        pvens = xens.reshape(
            (self.args.n_ens, self.num_channels, self.ny, self.nx))

        # forward operator on posterior ensemble.
        for nanal in range(self.args.n_ens):
            if kwargs["avg"]:
                hxens[nanal] = obs_fn(torch.tensor(self.scalefact*pvens[nanal])).cpu().numpy()  # surface pv obs
            else:
                for k in range(self.num_channels):
                    hxens[nanal, k, ...] = obs_fn(torch.tensor(
                        self.scalefact*pvens[nanal, k, ...].ravel()[indxob])).cpu().numpy()  # surface pv obs

        # ob space diagnostics
        hxensmean_a = hxens.mean(axis=0)
        obsprd_a = (((hxens-hxensmean_a)**2).sum(axis=0) /
                    (self.args.n_ens-1)).mean()
        # expected value is HPaHT (obsprd_a).
        obinc_a = ((hxensmean_a-hxensmean_b)*(pvob-hxensmean_a)).mean()
        # expected value is HPbHT (obsprd_b).
        obinc_b = ((hxensmean_a-hxensmean_b)*(pvob-hxensmean_b)).mean()
        # expected value R (oberrvar).
        omaomb = ((pvob-hxensmean_a)*(pvob-hxensmean_b)).mean()

        # posterior multiplicative inflation.
        pvensmean_a = pvens.mean(axis=0)
        pvprime = pvens-pvensmean_a
        asprd = (pvprime**2).sum(axis=0)/(self.args.n_ens-1)
        asprd_over_fsprd = asprd.mean()/fsprd.mean()
        if self.covinflate2 < 0:
            # relaxation to prior stdev (Whitaker & Hamill 2012)
            asprd = np.sqrt(asprd)
            fsprd = np.sqrt(fsprd)
            inflation_factor = 1.+self.covinflate1*(fsprd-asprd)/asprd
        else:
            # Hodyss et al 2016 inflation (covinflate1=covinflate2=1 works well in perfect
            # model, linear gaussian scenario)
            # inflation = asprd + (asprd/fsprd)**2((fsprd/args.n_ens)+2*inc**2/(args.n_ens-1))
            inc = pvensmean_a - pvensmean_b
            inflation_factor = self.covinflate1*asprd + \
                (asprd/fsprd)**2*((fsprd/self.args.n_ens) +
                                  self.covinflate2*(2.*inc**2/(self.args.n_ens-1)))
            inflation_factor = np.sqrt(inflation_factor/asprd)
        pvprime = pvprime*inflation_factor
        pvens = pvprime + pvensmean_a

        return torch.tensor(pvens)*self.scalefact
