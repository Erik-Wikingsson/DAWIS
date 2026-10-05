"""FlowDAS: a flow-based (stochastic-interpolant) data assimilation filter.

Reference implementation: https://github.com/umjiayx/FlowDAS
(`experiments/weather_forecasting/main_research_sevir_gen_sample4GL_621_opensource_0710.py`).

The measurement enters inside the Euler-Maruyama loop, so forecast and
analysis are one sampling pass; run it with `--forward_model none`.

One assimilation step:

  * `cond` is the lookback window of `window` analyses, stacked along channels,
    and `base = z0` the most recent of them;
  * integrate the interpolant SDE from `t_min` to `t_max` with drift `bF` and
    diffusion `sigma(t) = 1 - t`;
  * at every step estimate the endpoint `x1` from `zt` by a second-order
    stochastic Runge-Kutta step (`_taylor_est2rd_x1`), take `MC_times` such
    estimates, and push the measurement residual back onto `zt` as a DPS-style
    gradient scaled by `guidance_strength` (the reference's `grad_scale`).

The ensemble generalization of the reference is marked ENSEMBLE below.
"""

import torch

from assimilation.methods.method import AssimilationMethod
from assimilation.obs_schedule import has_observations


class FlowDAS(AssimilationMethod):
    """FlowDAS filter.

    Args:
        args: parsed arguments.
        obs_sigma: observation noise std. Reporting only; the guidance term
            does not divide by sigma.
        device: torch device.
        members: ensemble size.
        metadata: DatasetMetadata, for the state shape.
        steps: Euler-Maruyama steps per assimilation step (`EM_sample_steps`).
        guidance_strength: the reference's `grad_scale`.
        mc_times: number of endpoint estimates averaged in the guidance term
            (`MC_times`); cheap, since drift evaluations are shared.
        t_min, t_max: sampling time range (`t_min_sampling`, `t_max_sampling`).
    """

    def __init__(self, args, obs_sigma, device, members, metadata,
                 steps=200, guidance_strength=0.1, mc_times=1,
                 t_min=0.001, t_max=0.999, forecaster=None):
        super().__init__(args)
        if metadata is None:
            raise ValueError(
                "FlowDAS needs the dataset metadata. Pass metadata=<DatasetMetadata>."
            )
        self.md = metadata
        self.device = device
        self.members = members
        self.obs_sigma = obs_sigma
        self.steps = int(steps)
        self.guidance_strength = float(guidance_strength)
        self.mc_times = int(mc_times)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        if forecaster is None:
            # Same checkpoint and normalization as --forward_model flowdas;
            # the checkpoint is chosen per dataset.
            from forecasting.flowdas_forecaster import FlowDASModel
            forecaster = FlowDASModel(
                device, dataset=self.md.name,
                path=getattr(args, "model_path", None), md=self.md,
                # --forward_norm sets the drift's units (normally == --assim_norm).
                forward_norm=getattr(args, "forward_norm", None))
        self.forecaster = forecaster
        self.drift = forecaster.drift
        self.interp = forecaster.interp
        self.cond_frames = forecaster.cond_frames

        # Affine map between assimilation units and the drift's training units
        # (identity when the data is already in model space, e.g. SEVIR).
        self.to_norm = forecaster.to_norm
        self.from_norm = forecaster.from_norm
        # The checkpoint's cond_channels fixes the window length.
        window = getattr(args, "window", None)
        if window is not None and window != self.cond_frames:
            raise ValueError(
                f"FlowDAS's checkpoint conditions on {self.cond_frames} past "
                f"states, but --window is {window}. Run with "
                f"--window {self.cond_frames}."
            )

    # Endpoint estimate and guidance
    def _taylor_est2rd_x1(self, xt, t, bF, cond):
        """Second-order stochastic Runge-Kutta estimates of x1 given zt.

        Returns `mc_times` tensors; `bF` and `bF2` are shared, only the
        injected noise is redrawn.
        """
        # ENSEMBLE: widen t to (B,1,1,1) so it broadcasts per member.
        tw = self.interp.wide(t)
        root_t = tw.sqrt()
        # std of the analytic Milstein noise term, integrated over [t, 1]
        noise_std = 2.0 / 3.0 - root_t + (1.0 / 3.0) * root_t ** 3

        hat_x1 = xt + bF * (1 - tw) + torch.randn_like(xt) * noise_std
        t1 = torch.ones(xt.shape[0], device=xt.device, dtype=xt.dtype)
        bF2 = self.drift(hat_x1, t1, cond=cond)

        drift_avg = (bF + bF2) / 2
        return [xt + drift_avg * (1 - tw) + torch.randn_like(xt) * noise_std
                for _ in range(self.mc_times)]

    def _grad_and_value(self, x_prev, x1_hats, obs, observation_fn):
        """DPS-style gradient of the measurement residual w.r.t. `zt`.

        The residual is formed in normalized units (`obs` is mapped in
        `assimilate`); since the norm is 1-homogeneous, the choice of units
        changes the effective guidance strength. As in the reference, the
        residual is averaged over MC estimates before taking the norm.
        """
        # Unobserved step (--assim_interval > 1): zero guidance.
        if not has_observations(obs):
            return (torch.zeros_like(x_prev),
                    torch.zeros((), device=x_prev.device, dtype=x_prev.dtype))

        difference = 0
        for x1_hat in x1_hats:
            difference = difference + (obs - observation_fn(x1_hat))
        difference = difference / len(x1_hats)

        # One global norm over the ensemble, as in the reference. A per-member
        # norm would rescale gradients and require re-tuning guidance_strength.
        norm = torch.linalg.norm(difference)

        grad = torch.autograd.grad(outputs=norm, inputs=x_prev,
                                   allow_unused=True)[0]
        if grad is None:
            print("FlowDAS: no guidance gradient (observation operator "
                  "detached from the state?)")
            grad = torch.zeros_like(x_prev)
        return grad, norm.detach()

    # Conditioning
    def _build_cond(self, init_states):
        """(window, n_ens, C, H, W) physical -> (cond, z0), normalized."""
        x = self.to_norm(torch.as_tensor(init_states, dtype=torch.float32,
                                         device=self.device))
        if x.shape[0] != self.cond_frames:
            raise ValueError(
                f"FlowDAS needs a conditioning window of {self.cond_frames} "
                f"states (the checkpoint's cond_channels), got {x.shape[0]}. "
                f"Run with --window {self.cond_frames}."
            )
        # (window, n_ens, C, H, W) -> (n_ens, window, C, H, W)
        x = x.permute(1, 0, 2, 3, 4)
        return self.forecaster.build_cond_z0_z1(x)

    def assimilate(self, x_forecast, obs, obs_mask, obs_fn, ntime,
                   init_states=None, **kwargs):
        """One FlowDAS step: guided interpolant sampling from the window.

        `x_forecast` is ignored: FlowDAS produces forecast and analysis in
        the same pass.
        """
        if init_states is None:
            raise ValueError(
                "FlowDAS assimilates from the conditioning window; "
                "`init_states` was not passed."
            )

        # Same observation-operator convention as DAISI.
        if kwargs.get("avg", False):
            observation_fn = obs_fn
        else:
            mask = obs_mask[0].bool()
            def observation_fn(x): return obs_fn(x[:, mask])

        cond, z0 = self._build_cond(init_states)
        # Map obs into normalized units with the state's affine. This assumes
        # to_norm(H(x)) == H(to_norm(x)), exact only for identity-like H; it is
        # approximate for nonlinear operators (e.g. saturating, arctan).
        obs = self.to_norm(
            torch.as_tensor(obs, dtype=torch.float32, device=self.device))

        ts = torch.linspace(self.t_min, self.t_max, self.steps,
                            device=self.device)
        dt = ts[1] - ts[0]

        xt = z0.clone()
        mu = xt
        for tscalar in ts:
            t = tscalar.repeat(xt.shape[0])

            with torch.enable_grad():
                xt = xt.detach().requires_grad_(True)
                bF = self.drift(xt, t, cond=cond)
                x1_hats = self._taylor_est2rd_x1(xt, t, bF, cond)
                norm_grad, _ = self._grad_and_value(
                    xt, x1_hats, obs, observation_fn)

            with torch.no_grad():
                # sigma_coef = 1: f = bF and g = sigma(t) = 1 - t.
                f = bF.detach()
                g = self.interp.sigma(t)
                mu = xt.detach() + f * dt
                xt = (mu + g * torch.randn_like(mu) * dt.sqrt()
                      - self.guidance_strength * norm_grad)

        # The reference returns the mean of the final step, not the noised draw.
        return self.from_norm(mu).detach()
