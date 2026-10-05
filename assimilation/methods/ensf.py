import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import sys
import os
from assimilation.methods.method import AssimilationMethod
from assimilation.observers.obsop import OBS_FNS


class EnSF(AssimilationMethod):
    def __init__(self, nx, ensemble_size, eps_alpha, device, obs_sigma, steps,
                 scalefact, init_std_x_state, obs_fn, ISarctan=False,
                 ny=None, num_channels=2, guidance_strength=1.0):
        self.nx = nx
        self.ny = nx if ny is None else ny
        self.num_channels = num_channels
        self.n_dim = self.ny * self.nx * self.num_channels
        self.ensemble_size = ensemble_size
        self.eps_alpha = eps_alpha
        self.device = device
        self.obs_sigma = obs_sigma
        self.obs_fn = obs_fn
        self.ISarctan = ISarctan
        self.euler_steps = steps
        self.scalefact = scalefact
        # Spread the ensemble is rescaled to after each analysis; None = take
        # it from the first forecast.
        self.init_std_x_state = init_std_x_state
        # Damping on the likelihood score. The explicit Euler integration in
        # reverse_SDE is unstable for small normalized obs_sigma without it.
        # The paper runs used 0.01.
        self.guidance_strength = guidance_strength

    def cond_alpha(self, t):
        # alpha(0) = 1, alpha(1) = eps_alpha ~ 0
        return 1 - (1-self.eps_alpha)*t

    def cond_sigma_sq(self, t):
        # sigma^2(t) = t
        return t

    # Drift of the forward SDE.
    def f(self, t):
        # f=d_(log_alpha)/dt
        alpha_t = self.cond_alpha(t)
        f_t = -(1-self.eps_alpha) / alpha_t
        return f_t

    def g_sq(self, t):
        # g = d(sigma_t^2)/dt -2f sigma_t^2
        d_sigma_sq_dt = 1
        g2 = d_sigma_sq_dt - 2*self.f(t)*self.cond_sigma_sq(t)
        return g2

    def g(self, t):
        return np.sqrt(self.g_sq(t))

    def reverse_SDE(self, obs, x0, time_steps, obs_sigma, sparse_idx, save_path=False):
        """Sample from t=1 (standard Gaussian) to t=0 with guided Euler-Maruyama."""
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device
        dt = 1.0/time_steps

        xt = torch.randn(ensemble_size, n_dim, device=device)
        t = 1.0

        if save_path:
            path_all = [xt]
            t_vec = [t]
        # Early stopping once the running ensemble mean stops moving.
        mart_point = int(0.20 * time_steps)
        temp_state = torch.zeros(mart_point, n_dim)
        for i in range(time_steps):
            temp_mean_old = temp_state.mean(dim=0)
            alpha_t = self.cond_alpha(t)
            sigma2_t = self.cond_sigma_sq(t)
            diffuse = self.g(t)

            xt += - dt*(self.f(t)*xt + diffuse**2 * ((xt - alpha_t*x0)/sigma2_t) - diffuse**2 * self.score_likelihood(xt, t, obs, obs_sigma, sparse_idx)) \
                + np.sqrt(dt)*diffuse*torch.randn_like(xt)

            if save_path:
                path_all.append(xt)
                t_vec.append(t)

            temp_state[i % mart_point, :] = xt.mean(dim=0)
            if (abs(temp_state.mean(dim=0) - temp_mean_old) < 0.005).all() and i > mart_point:
                break
            else:
                pass
            if i > 500:
                break
            t = t - dt

        if save_path:
            return path_all, t_vec
        else:
            return xt

    # Likelihood damping: tau(0) = 1, tau(1) = 0.
    def g_tau(self, t):
        return 1-t

    def score_likelihood(self, xt, t, obs, obs_sigma, sparse_idx):
        # obs: (d)
        # xt: (ensemble, d)
        ensemble_size = self.ensemble_size
        n_dim = self.n_dim
        device = self.device
        score_x = torch.zeros(ensemble_size, n_dim, device=device)

        inner_derivative = None
        if self.obs_fn == 'arctan':
            inner_derivative = self.scalefact * 1./(1. + (self.scalefact * xt[:, sparse_idx])**2)
        elif self.obs_fn == 'square_scaled':
            inner_derivative = (self.scalefact / 7.) * 2. * (self.scalefact / 7. * xt[:, sparse_idx])
        elif self.obs_fn == 'linear':
            inner_derivative = self.scalefact * torch.ones_like(xt[:, sparse_idx])
        else:
            raise NotImplementedError("Obs function not implemented in score_likelihood.")

        score_x[:, sparse_idx] = (-(OBS_FNS[self.obs_fn](self.scalefact*xt[:, sparse_idx]) - obs)/(obs_sigma ** 2) * inner_derivative).type_as(score_x)

        score_x[:, ~sparse_idx] = 0
        tau = self.g_tau(t)

        return tau*score_x*self.guidance_strength

    def assimilate(self, x_forecast, obs, obs_mask, obs_fn, ntime, **kwargs):
        """
        Perform a data assimilation step.
        Args:
            x_forecast: Forecast state, shape (n_ens, state_dim). NOTE: Should be scaled by scalefact.
            obs: Observations, shape (obs_dim,).
            obs_mask: Mask for observations.
            obs_fn: Observation operator name (unused; self.obs_fn is used).
            ntime: Current time step.

        Returns:
            x_analysis: Analyzed state after assimilation, shape (n_ens, state_dim). NOTE: Should be scaled by scalefact.
        """
        torch.set_default_dtype(torch.float32)
        obs_mask = obs_mask[0]
        if not isinstance(x_forecast, torch.Tensor):
            x_forecast = torch.tensor(
                x_forecast, device=self.device, dtype=torch.float32)

        if torch.isnan(x_forecast).any():
            print(f"Prior contains NaNs at step {ntime}, skipping sampling")
            return x_forecast

        x_state = torch.as_tensor(
            x_forecast, device=self.device, dtype=torch.float32) / self.scalefact
        mean_x_state_obs = x_state.mean(dim=0)

        # Normalize the ensemble per pixel.
        std_x_state = x_state.std(dim=0)
        if self.init_std_x_state is None:
            # Same measure as the rescaling below.
            self.init_std_x_state = std_x_state.mean().item()
            print(f"EnSF: spread target {self.init_std_x_state:.4g} from the "
                  f"forecast at step {ntime}", flush=True)
            if self.init_std_x_state == 0:
                raise ValueError(
                    "EnSF: the first forecast has zero spread, so there is no "
                    "spread to rescale to. Use a stochastic propagator or pass "
                    "--ensf_spread.")
        mean_x_state = x_state.mean(dim=0)
        x_state = (x_state - mean_x_state) / std_x_state

        temp_obs_sigma = torch.zeros(x_state.size(dim=1), device=self.device)

        if self.obs_fn == 'arctan':
            obs = (torch.atan((torch.tan(obs) / self.scalefact - mean_x_state[obs_mask]) * self.scalefact / std_x_state[obs_mask])).type_as(obs)
            temp_obs_sigma = torch.max(torch.abs((torch.tan(torch.atan(mean_x_state_obs)-0.01) - mean_x_state_obs) / mean_x_state_obs) / std_x_state, torch.tensor(0.0001, device=self.device))
        elif self.obs_fn == 'square_scaled':
            mean_y = OBS_FNS[self.obs_fn](self.scalefact*mean_x_state)
            std_y = 2 * OBS_FNS[self.obs_fn](self.scalefact*torch.ones_like(mean_x_state)) * std_x_state * torch.abs(mean_x_state)

            obs = ((obs - mean_y[obs_mask]) / std_y[obs_mask] ** 2).type_as(obs)
            temp_obs_sigma = 0.001*(self.obs_sigma / std_y).type_as(temp_obs_sigma)

        elif self.obs_fn == 'linear':
            obs = (((obs - mean_x_state[obs_mask] * self.scalefact) / std_x_state[obs_mask])).type_as(obs)
            temp_obs_sigma = (self.obs_sigma / std_x_state).type_as(temp_obs_sigma)
        else:
            raise NotImplementedError("Obs function not implemented in assimilate.")

        torch.cuda.empty_cache()

        # Posterior sample
        x_state = self.reverse_SDE(obs=obs, x0=x_state.flatten(1, -1), time_steps=self.euler_steps,
                                   obs_sigma=temp_obs_sigma[obs_mask], sparse_idx=obs_mask.flatten())

        x_state = x_state.reshape(
            self.ensemble_size, self.num_channels, self.ny, self.nx)
        x_state = x_state * std_x_state + mean_x_state

        # Rescale the spread to init_std_x_state.
        x_state = (x_state - x_state.mean(dim=0)) * \
            (self.init_std_x_state/x_state.std(dim=0).mean()) + x_state.mean(dim=0)
        return x_state*self.scalefact
