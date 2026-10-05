import torch
from functools import partial
from assimilation.methods.method import AssimilationMethod
from assimilation.obs_schedule import has_observations
from unconditional_generation.backbone import build_backbone

try:
    from .linalg.solve import cg, gmres
except ImportError:
    from linalg.solve import cg, gmres

from tqdm import tqdm
import numpy as np


class DAISI(AssimilationMethod):
    def __init__(self, args, obs_sigma, device, members, beta, alpha, beta_dot, alpha_dot, eps,
                 guide_method, guidance_strength, steps, noise=None, debug=False, scale=1.0,
                 metadata=None):
        super().__init__(args)
        if metadata is None:
            raise ValueError(
                "DAISI needs the dataset metadata to size its backbone. Pass "
                "metadata=<DatasetMetadata>; it comes from the data source."
            )
        self.md = metadata
        # Same builder as unconditional_generation/trainer.py; shape from metadata.
        self.model = build_backbone(
            self.md, args, model_channels=getattr(args, "hidden_dim", None) or 32)

        self.model.to(device)
        self.model.eval()
        self.model.load_state_dict(torch.load(
            args.model_path, map_location=device, weights_only=True))

        self.obs_sigma = obs_sigma

        if hasattr(args, 'solver'):
            self.step = {
                'euler': self.euler_step,
                'heun': self.heun_step
            }[args.solver]
        else:
            self.step = self.euler_step

        self.nx = self.md.nx
        # (C, H, W) of one state.
        self.state_shape = self.md.state_shape

        self.device = device
        self.members = members
        self.steps = steps

        self.set_guide_method(guide_method)

        self.guidance_strength = guidance_strength
        # args.tmin is a list (one per time step in other methods); DAISI uses the first.
        self.tmin = args.tmin[0]
        self.corrections = args.corrections
        self.tau = args.tau

        self.scale = scale

        self.beta = beta
        self.beta_dot = beta_dot
        # Rescale since the data are scaled.
        self.alpha = lambda t: alpha(t) * self.scale
        self.alpha_dot = lambda t: alpha_dot(t) * self.scale
        self.gamma = lambda t: self.alpha(
            t) * self.beta_dot(t) - self.beta(t) * self.alpha_dot(t)
        self.eps = lambda t: eps(t) * self.scale**2

        self.debug = debug

        self.invert_eps = lambda t: args.invert_eps*(1-t)

        self.invert_steps = args.invert_steps

        self.noise = noise
        if noise == 'None' or noise is None:
            self.tmin = 0

    def set_guide_method(self, guide_method):
        self.guide_method = guide_method
        if guide_method == 'DPS':
            self.guide = partial(self.DPS)
        elif guide_method == 'DPS_scale':
            self.guide = partial(self.DPS, scale=True)
        elif guide_method == 'MMPS':
            self.guide = partial(self.MMPS)
        elif guide_method is None:
            self.guide = lambda obs, zt, t, eta_1: torch.zeros_like(zt)
        else:
            raise ValueError(f"{guide_method} is invalid guide_method")

    # Guidance methods
    def DPS(self, obs, zt, t, eta_1, scale=False):
        """
        obs: Observations
        zt: Current diffusion state
        t: Current time step
        eta_1: Function to give estimate of final state at t=1
        scale: If true, use the DPS scaling in original paper
        """

        with torch.enable_grad():
            zt = zt.detach().requires_grad_()
            z1 = eta_1(zt, t)

            error = obs - self.observation_fn(z1)
            scalar = 1 / (2 * self.obs_sigma**2)
            dim = tuple(range(1, len(error.shape)))
            if scale:
                # No scalar here
                logp = - torch.linalg.vector_norm(error, dim=dim)
            else:
                logp = -(scalar * error ** 2).sum(axis=dim)

        s_obs = torch.autograd.grad(
            outputs=logp,
            inputs=zt,
            grad_outputs=torch.ones_like(logp),
        )[0]

        return s_obs

    def MMPS(self, obs, zt, t, eta_1, solver="gmres", iterations=1):
        """
        obs: Observations
        zt: Current diffusion state
        t: Current time step
        eta_1: Function to give estimate of final state at t=1
        solver: Solver to use for solving the linear system
        iterations: Number of iterations for the solver
        """
        if solver == "cg":
            solve = partial(cg, iterations=iterations)
        elif solver == "gmres":
            solve = partial(gmres, iterations=iterations)

        with torch.enable_grad():
            zt = zt.detach().requires_grad_()
            z1 = eta_1(zt, t)
            y_hat = self.observation_fn(z1)

        def A(v):
            return torch.func.jvp(self.observation_fn, (z1.detach(),), (v,))[1]

        def At(v):
            return torch.autograd.grad(y_hat, z1, v, retain_graph=True)[0]

        # fmt: off
        def cov_x(v):
            # Divided by beta, as in the original MMPS code.
            return self.alpha(t)**2/self.beta(t) * torch.autograd.grad(z1, zt, v, retain_graph=True)[0] 
        # fmt: on

        def cov_y(v):
            return self.obs_sigma**2 * v + A(cov_x(At(v)))
        grad = obs - y_hat
        grad = solve(A=cov_y, b=grad)
        grad = torch.autograd.grad(y_hat, zt, grad)[0]

        return grad

    def get_drift_and_score(self, zt, t, obs):
        beta_t = self.beta(t)
        alpha_t = self.alpha(t)
        gamma_t = self.gamma(t)
        beta_dot_t = self.beta_dot(t)
        alpha_dot_t = self.alpha_dot(t)
        b_obs = torch.zeros_like(zt)
        s_obs = torch.zeros_like(zt)

        t_tensor = torch.ones((self.members,), device=self.device) * t

        # Scale model input and output when scale != 1.
        b = self.scale * self.model(zt / self.scale, t_tensor)
        def eta_1(x, t): return self.scale * (self.alpha(t) * self.model(x / self.scale, torch.ones(
            (self.members,), device=self.device) * t) - self.alpha_dot(t) * (x / self.scale)) / self.gamma(t)

        if t < 1.0:
            s = (beta_t * b - beta_dot_t * zt) / \
                (alpha_t * gamma_t)  # s = (t * b - zt) / (1 - t)
            if t > 0.0 and not has_observations(obs):
                # Unobserved step (assim_interval > 1): no guidance.
                s_obs, b_obs = torch.zeros_like(zt), torch.zeros_like(zt)
            elif t > 0.0:
                s_obs = self.guide(obs, zt, t, eta_1)
                b_obs = s_obs * alpha_t * gamma_t
                if self.guide_method == "MMPS":
                    b_obs /= beta_t
            else:
                s_obs, b_obs = torch.zeros_like(zt), torch.zeros_like(zt)
        else:
            s = torch.zeros_like(zt)

        return b, s, b_obs, s_obs

    def assimilate(self, x_forecast, obs, obs_mask, obs_fn, ntime, **kwargs):
        """
        Perform a data assimilation step.
        Args:
            x_forecast: Forecast state, shape (n_ens, state_dim). NOTE: Should be scaled by scalefact.
            obs: Observations, shape (obs_dim,).
            obs_mask: Mask for observations.
            obs_fn: Observation operator function.
            ntime: Current time step.
            **kwargs: Additional keyword arguments that can be passed to specific implementations.

        Returns:
            x_analysis: Analyzed state after assimilation, shape (n_ens, state_dim). NOTE: Should be scaled by scalefact.
        """

        """
        b: static diffusion prior
        b_dynamic: dynamic EnSF prior
        b_obs: prior observation likelihood
        b_dynamic_obs: EnSF observation likelihood
        """
        if x_forecast is None:  # No prior, pure noise forecast
            noise = None
            tmin = 0
        else:
            noise = self.noise
            tmin = self.tmin

            if not isinstance(x_forecast, torch.Tensor):
                x_forecast = torch.tensor(
                    x_forecast, device=self.device, dtype=torch.float32)

            if torch.isnan(x_forecast).any():
                print("Prior contains NaNs, skipping sampling")
                return x_forecast

        self.obs_mask = obs_mask
        if "avg" in kwargs.keys() and kwargs["avg"]:
            self.observation_fn = obs_fn
        else:
            self.observation_fn = lambda x: obs_fn(x[:, obs_mask[0].bool()])

        tmax = 1
        ts = torch.linspace(tmin, tmax, self.steps+1, device=self.device)[:-1]
        dt = (tmax - tmin) / self.steps

        # Initial state
        if noise == 'invert':
            zt = self.invert(prior=x_forecast,
                             eps=self.invert_eps, steps=self.invert_steps)
        elif noise == 'repeat':
            zt = torch.randn(
                (self.members, *self.state_shape), device=self.device)
            # Same noise for every member (for testing guidance).
            zt = zt[0].unsqueeze(0).repeat(self.members, 1, 1, 1)
            zt = zt * self.scale
        elif noise == 'None' or noise is None:
            zt = torch.randn(
                (self.members, *self.state_shape), device=self.device)
            zt = zt * self.scale
        elif isinstance(noise, torch.Tensor):
            if noise.shape != (self.members, *self.state_shape):
                raise ValueError(
                    f"Noise tensor must have shape "
                    f"({self.members}, {', '.join(map(str, self.state_shape))}), "
                    f"but got {tuple(noise.shape)}")
            zt = noise
        elif noise == 'SDEdit':
            zt = self.beta(self.tmin) * x_forecast + \
                self.alpha(self.tmin) * torch.randn_like(x_forecast)
        else:
            raise ValueError(
                f"Noise must be 'invert', 'repeat', None, 'None', or a torch.Tensor, but got {noise}")

        self.model.eval()
        zs = [zt.clone().cpu()] if self.debug else None
        vector_fields = []
        if self.debug:
            print("Assimilating...")
            enum = tqdm(ts)
        else:
            enum = ts

        for t in enum:
            eps_t = self.eps(t)

            # CORRECTOR
            for _ in range(self.corrections):
                b, s, b_obs, s_obs = self.get_drift_and_score(zt, t, obs)
                s_obs = self.guidance_strength(t) * s_obs
                b_obs = self.guidance_strength(t) * b_obs
                b = b + b_obs
                s = s + s_obs
                with torch.no_grad():
                    # Step-size heuristic; includes dt so it is robust to the number of steps.
                    delta_t = dt * self.tau / s.square().mean(dim=(1, 2, 3), keepdim=True)

                    dz = s * delta_t
                    dW = torch.randn_like(zt) * torch.sqrt(2 * delta_t)

                    zt = zt + dz + dW

            # PREDICTOR
            if self.debug:
                b, s, b_obs, s_obs = self.get_drift_and_score(zt, t, obs)
                vector_fields.append(torch.stack([
                    b.detach().cpu(),
                    s.detach().cpu(),
                    b_obs.detach().cpu(),
                    s_obs.detach().cpu()
                ]))

            dz = self.step(zt, t, obs, dt)
            dW = torch.randn_like(zt) * torch.sqrt(2*dt * eps_t)

            zt = zt + dz + dW

            if self.debug:
                zs.append(zt.clone().cpu())

        if self.debug:
            return zt, torch.stack(zs, dim=1), torch.stack(vector_fields)
        else:
            return zt  # Shape: (members, channels, height, width)

    def euler_step(self, zt, t, obs, dt):
        eps_t = self.eps(t)
        b, s, b_obs, s_obs = self.get_drift_and_score(zt, t, obs)

        s_obs = self.guidance_strength(t) * s_obs
        b_obs = self.guidance_strength(t) * b_obs
        b, s = b + b_obs, s + s_obs

        with torch.no_grad():
            dz = (b + s * eps_t) * dt

        return dz

    def heun_step(self, zt, t, obs, dt):

        dz1 = self.euler_step(zt, t, obs, dt)
        zt_temp = zt + dz1
        dz2 = self.euler_step(zt_temp, t + dt, obs, dt)

        dz = 0.5 * (dz1 + dz2)

        return dz

    # Inversion
    def invert(self, prior, eps=lambda t: torch.zeros_like(t), steps=100, debug=False):
        with torch.no_grad():
            tmin = 1
            tmax = self.tmin
            zt = prior

            ts = torch.linspace(tmin, tmax, steps+1, device=self.device)[:-1]
            dt = (tmax - tmin) / steps

            self.model.eval()

            if self.debug:
                print("Inverting prior...")
                enum = tqdm(ts)
            else:
                enum = ts

        zs = [zt.clone().cpu()] if debug else None
        for t in enum:
            beta_t = self.beta(t)
            alpha_t = self.alpha(t)
            gamma_t = self.gamma(t)
            beta_dot_t = self.beta_dot(t)
            alpha_dot_t = self.alpha_dot(t)
            eps_t = eps(t)

            t_tensor = torch.ones((self.members,), device=self.device) * t

            b = self.scale * self.model(zt / self.scale, t_tensor)

            with torch.no_grad():
                if t != 1:
                    s = (beta_t * b - beta_dot_t * zt) / \
                        (alpha_t * gamma_t)  # s = (t * b - zt) / (1 - t)
                else:
                    s = 0

                dz = b - eps_t * s
                dW = torch.randn_like(
                    zt) * torch.sqrt(2*np.abs(dt) * eps_t)

                zt = zt + dz * dt + dW
            if debug:
                zs.append(zt.clone().cpu())
        if debug:
            return zt, torch.stack(zs, dim=1)
        return zt

    # Forward sampling from tmin to 1
    def sample(self, prior, eps=lambda t: torch.zeros_like(t), steps=100, debug=False):
        with torch.no_grad():
            tmin = self.tmin
            tmax = 1
            zt = prior

            ts = torch.linspace(tmin, tmax, steps+1, device=self.device)[:-1]
            dt = (tmax - tmin) / steps

            self.model.eval()

            if self.debug:
                print("Inverting prior...")
                enum = tqdm(ts)
            else:
                enum = ts

        zs = [zt.clone().cpu()] if debug else None
        for t in enum:
            beta_t = self.beta(t)
            alpha_t = self.alpha(t)
            gamma_t = self.gamma(t)
            beta_dot_t = self.beta_dot(t)
            alpha_dot_t = self.alpha_dot(t)
            eps_t = eps(t)

            t_tensor = torch.ones((self.members,), device=self.device) * t

            b = self.scale * self.model(zt / self.scale, t_tensor)

            with torch.no_grad():
                if t != 1:
                    s = (beta_t * b - beta_dot_t * zt) / \
                        (alpha_t * gamma_t)  # s = (t * b - zt) / (1 - t)
                else:
                    s = 0

                dz = b - eps_t * s
                dW = torch.randn_like(
                    zt) * torch.sqrt(2*np.abs(dt) * eps_t)

                zt = zt + dz * dt + dW
            if debug:
                zs.append(zt.clone().cpu())
        if debug:
            return zt, torch.stack(zs, dim=1)
        return zt
