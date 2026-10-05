"""Flow-matching window model (FMW) built on multitask stochastic interpolants.

A window of T states z1 (B, T, D, X, Y) is interpolated with Gaussian noise z0
as z_t = alpha(t) z0 + beta(t) z1, where every window slot has its own time
t (B, T). The backbone predicts the denoisers eta0 = E[z0 | z_t] and
eta1 = E[z1 | z_t], selected by ``--fm_loss``:

- ``eta0`` / ``eta1``: regress one of them; the other follows from
  z_t = alpha eta0 + beta eta1.
- ``eta01_label``: one network, class label 0/1 selects eta0 or eta1.
- ``eta01_channel``: one network outputs eta0 and eta1 as stacked channels.
- ``eta01_model``: two separate networks.
"""
# Third-party
import torch
import wandb
import matplotlib.pyplot as plt

from functools import partial

try:
    # Try relative import first (when imported as part of a package)
    from .linalg.solve import cg, gmres
except ImportError:
    # Fall back to absolute import (when run directly)
    from linalg.solve import cg, gmres

from plotting import plotting
from metrics import metrics
from forecasting.models.ar_prob_model import ARProbModel
from forecasting.models.fm_uncond import FMUncond
from networks.diffusion_networks import SongUNetW, SongUNet
from assimilation.obs_schedule import has_observations


def eta0_loss(z0, z1): return z0
def eta1_loss(z0, z1): return z1
def b_loss(z0, z1): return z1 - z0


TARGET_FNS = {
    'eta0': eta0_loss,
    'eta1': eta1_loss,
    'b': b_loss,
    'eta01_label': lambda z0, z1: (z0, z1),
    'eta01_model': lambda z0, z1: (z0, z1),
    'eta01_channel': lambda z0, z1: (z0, z1),
}


class AlphaBetaSchedule(torch.nn.Module):
    """Base class for interpolant schedules.

    All methods take t (B, T) and return (B, T) for scalar schedules or
    (B, T, T) for matrix schedules.
    """

    def __init__(self):
        super().__init__()

    def alpha(self, t):
        """Noise coefficient alpha(t)."""
        raise NotImplementedError("alpha not implemented!")

    def alpha_dot(self, t):
        """Time derivative of alpha(t)."""
        raise NotImplementedError("alpha_dot not implmented!")

    def beta(self, t):
        """Data coefficient beta(t)."""
        raise NotImplementedError("beta not implmented!")

    def beta_dot(self, t):
        """Time derivative of beta(t)."""
        raise NotImplementedError("beta_dot not implmented!")

    def gamma(self, t):
        """alpha * beta_dot - beta * alpha_dot."""
        raise NotImplementedError("gamma not implmented!")

    def alpha_beta(self, alpha, beta):
        """Combine alpha and beta into a (B, -1, 2) conditioning tensor."""
        raise NotImplementedError("alpha_beta not implmented!")


class LinearAlphaBetaSchedule(AlphaBetaSchedule):
    def __init__(self, tstart=0., tend=1.0, scale=1.0):
        """
        Args:
            tstart, tend: float or (B, T), effective interpolant time range.
            scale: float, multiplier on alpha.
        """
        super().__init__()

        self.tstart = tstart
        self.tend = tend
        self.scale = scale

    def alpha(self, t):
        return ((1.0-self.tstart) - t * (self.tend - self.tstart)) * self.scale

    def beta(self, t):
        return self.tstart + t * (self.tend - self.tstart)

    def alpha_dot(self, t):
        return -torch.ones_like(t) * (self.tend - self.tstart) * self.scale

    def beta_dot(self, t):
        return torch.ones_like(t) * (self.tend - self.tstart)

    def gamma(self, t):
        return self.alpha(t) * self.beta_dot(t) - self.beta(t) * self.alpha_dot(t)


class LinearForecastAlphaBetaSchedule(AlphaBetaSchedule):
    """Linear schedule on the last slot only; earlier slots are held at data. Returns (B, T)."""

    def __init__(self):
        super().__init__()

    def alpha(self, t):
        alpha_t = torch.zeros_like(t)
        alpha_t[:, -1] = 1-t[:, -1]
        return alpha_t

    def alpha_dot(self, t):
        alpha_dot_t = torch.zeros_like(t)
        alpha_dot_t[:, -1] = -1
        return alpha_dot_t

    def beta(self, t):
        beta_t = torch.ones_like(t)
        beta_t[:, -1] = t[:, -1]
        return beta_t

    def beta_dot(self, t):
        beta_dot_t = torch.zeros_like(t)
        beta_dot_t[:, -1] = 1
        return beta_dot_t

    def gamma(self, t):
        return torch.ones_like(t)


class MatrixLinearAlphaBetaSchedule(AlphaBetaSchedule):
    """Linear schedule as diagonal (B, T, T) matrices."""

    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return (1-t)[..., None] * torch.eye(t.shape[1], device=t.device)

    def alpha_dot(self, t):
        return -torch.ones_like(t)[..., None] * torch.eye(t.shape[1], device=t.device)

    def beta(self, t):
        return t[..., None] * torch.eye(t.shape[1], device=t.device)

    def beta_dot(self, t):
        return torch.ones_like(t)[..., None] * torch.eye(t.shape[1], device=t.device)

    def gamma(self, t):
        return torch.ones_like(t)[..., None] * torch.eye(t.shape[1], device=t.device)


class MatrixLinearForecastAlphaBetaSchedule(AlphaBetaSchedule):
    """Forecast schedule (last slot only) as diagonal (B, T, T) matrices."""

    def __init__(self):
        super().__init__()

    def alpha(self, t):
        alpha_t = torch.zeros_like(t)
        alpha_t[:, -1] = 1-t[:, -1]
        return alpha_t[..., None] * torch.eye(t.shape[1], device=t.device)

    def alpha_dot(self, t):
        alpha_dot_t = torch.zeros_like(t)
        alpha_dot_t[:, -1] = -1
        return alpha_dot_t[..., None] * torch.eye(t.shape[1], device=t.device)

    def beta(self, t):
        beta_t = torch.ones_like(t)
        beta_t[:, -1] = t[:, -1]
        return beta_t[..., None] * torch.eye(t.shape[1], device=t.device)

    def beta_dot(self, t):
        beta_dot_t = torch.zeros_like(t)
        beta_dot_t[:, -1] = 1
        return beta_dot_t[..., None] * torch.eye(t.shape[1], device=t.device)

    def gamma(self, t):
        return torch.ones_like(t)[..., None] * torch.eye(t.shape[1], device=t.device)


class MatrixCorrAlphaBetaSchedule(AlphaBetaSchedule):
    """Diagonal (B, T, T) linear schedule."""

    def __init__(self):
        super().__init__()

    def alpha(self, t):
        return (1-t)[..., None] * torch.eye(t.shape[1], device=t.device)

    def alpha_dot(self, t):
        return -torch.ones_like(t)[..., None] * torch.eye(t.shape[1], device=t.device)

    def beta(self, t):
        return t[..., None] * torch.eye(t.shape[1], device=t.device)

    def beta_dot(self, t):
        return torch.ones_like(t)[..., None] * torch.eye(t.shape[1], device=t.device)

    def gamma(self, t):
        return torch.ones_like(t)[..., None] * torch.eye(t.shape[1], device=t.device)


class MatrixCorrForecastAlphaBetaSchedule(AlphaBetaSchedule):
    """Diagonal (B, T, T) forecast schedule (last slot only)."""

    def __init__(self):
        super().__init__()

    def alpha(self, t):
        alpha_t = torch.zeros_like(t)
        alpha_t[:, -1] = 1-t[:, -1]
        return alpha_t[..., None] * torch.eye(t.shape[1], device=t.device)

    def alpha_dot(self, t):
        alpha_dot_t = torch.zeros_like(t)
        alpha_dot_t[:, -1] = -1
        return alpha_dot_t[..., None] * torch.eye(t.shape[1], device=t.device)

    def beta(self, t):
        beta_t = torch.ones_like(t)
        beta_t[:, -1] = t[:, -1]
        return beta_t[..., None] * torch.eye(t.shape[1], device=t.device)

    def beta_dot(self, t):
        beta_dot_t = torch.zeros_like(t)
        beta_dot_t[:, -1] = 1
        return beta_dot_t[..., None] * torch.eye(t.shape[1], device=t.device)

    def gamma(self, t):
        return torch.ones_like(t)[..., None] * torch.eye(t.shape[1], device=t.device)


class FMW(ARProbModel):
    """Flow-matching probabilistic window model."""

    def __init__(self, args):
        super().__init__(args)

        # Variables per physical time step, used for the alpha/beta conditioning.
        self.num_vars = self.ch.vars
        channels = self.ch.cond + self.ch.state

        # NOTE: The schedule name must contain "matrix" for the model to accept a matrix schedule.
        T = args.init_states + 1  # The physical time is the same for all variables

        if args.alpha_beta_mult is not None:
            alpha_beta_mult = args.alpha_beta_mult
        else:
            if "scalar" in self.args.schedule:
                # One (alpha, beta) pair per time step, see `eta`.
                alpha_beta_mult = 2 * T
            elif "matrix" in self.args.schedule:
                alpha_beta_mult = T*T * 2  # We have alpha and beta for each time step
            else:
                raise ValueError(
                    f"The schedule name {args.schedule} needs to indicate if it is a scalar or matrix schedule!")

        # Define alpha/beta functions
        if self.args.schedule == "linear_scalar":
            self.schedule = LinearAlphaBetaSchedule()
            self.forecast_schedule = LinearForecastAlphaBetaSchedule()
        elif self.args.schedule == "linear_matrix":
            self.schedule = MatrixLinearAlphaBetaSchedule()
            self.forecast_schedule = MatrixLinearAlphaBetaSchedule()
        else:
            raise NotImplementedError(
                f"The schedule {self.args.schedule} is not implemented!")

        # If we only want to train a forecast model.
        if self.args.fm_forecast:
            self.schedule = self.forecast_schedule

        # Langevin correction steps in the sampler (only defined by the assimilation parser).
        self.corrections = int(getattr(args, "corrections", 0) or 0)
        self.tau = float(getattr(args, "tau", 1e-4) or 1e-4)

        in_channels = channels + alpha_beta_mult if self.args.alpha_beta_spatial else channels

        # Wrap-around padding only for periodic domains.
        circular_padding = all(self.md.periodic)

        if args.fm_loss in ["eta01_model"]:
            self.backbone_model0 = SongUNetW(
                img_resolution=torch.as_tensor(args.nx),
                in_channels=in_channels,
                out_channels=channels,
                label_dim=0,
                embedding_type=args.noise_embedding,
                model_channels=args.hidden_dim,
                resample_filter=args.resample_filter,
                channel_mult=args.channel_mult,
                encoder_type=args.encoder_type,
                attn_resolutions=args.attn_resolutions,
                channel_mult_emb=args.channel_mult_emb,
                channel_mult_noise=args.channel_mult_noise,
                alpha_beta_mult=alpha_beta_mult,
                circular_padding=circular_padding,
            )
            self.backbone_model1 = SongUNetW(
                img_resolution=torch.as_tensor(args.nx),
                in_channels=in_channels,
                out_channels=channels,
                label_dim=0,
                embedding_type=args.noise_embedding,
                model_channels=args.hidden_dim,
                resample_filter=args.resample_filter,
                channel_mult=args.channel_mult,
                encoder_type=args.encoder_type,
                attn_resolutions=args.attn_resolutions,
                channel_mult_emb=args.channel_mult_emb,
                channel_mult_noise=args.channel_mult_noise,
                alpha_beta_mult=alpha_beta_mult,
                circular_padding=circular_padding,
            )
        elif args.fm_loss == 'eta01_channel':
            self.backbone_model = SongUNetW(
                img_resolution=torch.as_tensor(args.nx),
                in_channels=in_channels,
                out_channels=channels*2,  # We output both eta0 and eta1 as different channels
                label_dim=0,
                embedding_type=args.noise_embedding,
                model_channels=args.hidden_dim,
                resample_filter=args.resample_filter,
                channel_mult=args.channel_mult,
                encoder_type=args.encoder_type,
                attn_resolutions=args.attn_resolutions,
                channel_mult_emb=args.channel_mult_emb,
                channel_mult_noise=args.channel_mult_noise,
                alpha_beta_mult=alpha_beta_mult,
                circular_padding=circular_padding,
                eta01_channel=True
            )

        else:
            self.backbone_model = SongUNetW(
                img_resolution=torch.as_tensor(args.nx),
                in_channels=in_channels,
                out_channels=channels,
                label_dim=1 if args.fm_loss == "eta01_label" else 0,
                embedding_type=args.noise_embedding,
                model_channels=args.hidden_dim,
                resample_filter=args.resample_filter,
                channel_mult=args.channel_mult,
                encoder_type=args.encoder_type,
                attn_resolutions=args.attn_resolutions,
                channel_mult_emb=args.channel_mult_emb,
                channel_mult_noise=args.channel_mult_noise,
                alpha_beta_mult=alpha_beta_mult,
                circular_padding=circular_padding,
            )

        if args.fm_uncond:
            self.fm_uncond = FMUncond(args)

        assert args.fm_loss in TARGET_FNS, f"Loss function {args.fm_loss} not recognized. Must be one of {list(TARGET_FNS.keys())}"

        self.set_guide_method(self.args.guide_method)
        self.guidance_strength = lambda t: args.guidance_strength * torch.ones_like(t)

    def t2state_dim(self, t):
        """Repeat per-time-step t (B, T, ...) over variables to (B, T*D, ...)."""
        return t.repeat_interleave(self.num_vars, dim=1)

    def time2state_dim(self, x):
        """Flatten (B, T, D, X, Y) to (B, T*D, X, Y); 4D input is returned as is."""
        if len(x.shape) == 4:
            return x
        elif len(x.shape) == 5:
            B, T, D, H, W = x.shape
            return x.reshape(B, T*D, H, W)
        else:
            raise ValueError(f"Unexpected shape {x.shape}")

    def time_mix(self, A, z):
        """Apply schedule coefficients along the time dimension.

        Args:
            A: (B, T) scalar weights or (B, T, T) mixing matrix.
            z: (B, T, D, X, Y)

        Returns:
            (B, T, D, X, Y)
        """

        if A.ndim == 2:
            # (B,T) -> (B,T,1,1,1) for broadcasting
            return A[..., None, None, None] * z

        elif A.ndim == 3:
            return torch.einsum("bij,bjdxy->bidxy", A, z)

        else:
            raise ValueError(
                f"Unsupported shape {A.shape}. "
                "Expected (B,T) or (B,T,T)."
            )

    def loss_fn(self, pred, z0, z1):
        """MSE between the backbone prediction and the ``--fm_loss`` target.

        Args:
            pred: (B, T, D, X, Y), or a pair (eta0, eta1) for the eta01_* losses.
            z0: (B, T, D, X, Y) noise.
            z1: (B, T, D, X, Y) data.

        Returns:
            loss: (B, T, D, X, Y)
        """
        loss_fn = TARGET_FNS[self.args.fm_loss]
        target = loss_fn(z0, z1)
        if self.args.fm_loss in ["eta01_label", "eta01_model", "eta01_channel"]:
            return metrics.mse(self.time2state_dim(pred[0]), self.time2state_dim(target[0])) + metrics.mse(self.time2state_dim(pred[1]), self.time2state_dim(target[1]))
        else:
            return metrics.mse(self.time2state_dim(pred), self.time2state_dim(target))

    def eta(self, model, zt, alpha, beta, class_labels=None):
        """Evaluate a backbone on zt (B, T, D, X, Y) conditioned on alpha/beta.

        Returns (B, T, D, X, Y), or a pair (eta0, eta1) for eta01_channel.
        """
        B, T, D, H, W = zt.shape
        alpha = alpha.reshape(B, -1)
        beta = beta.reshape(B, -1)
        # (B, T, 2) or (B, T*T, 2)
        alpha_beta = torch.stack([alpha, beta], dim=-1)

        # (B, T, D, X, Y) -> (B, T*D, X, Y)
        latents = zt.reshape(B, T*D, H, W)

        if self.args.alpha_beta_spatial:
            # (B, T*2, X, Y) or (B, T*T*2, X, Y)
            alpha_beta_spatial = torch.cat([alpha, beta], dim=1)[..., None, None].repeat(
                1, 1, latents.shape[2], latents.shape[3])
            latents = torch.cat([latents, alpha_beta_spatial], dim=1)
        output = model(latents, alpha_beta, class_labels)

        if self.args.fm_loss == 'eta01_channel':
            eta01 = output.reshape(B, T, D*2, H, W)  # (B, T, D*2, X, Y)
            eta0 = eta01[:, :, :D]
            eta1 = eta01[:, :, D:]

            return eta0, eta1

        return output.reshape(B, T, D, H, W)  # (B, T, D, X, Y)

    def predict_step_train(self, init_states, true_state):
        """Training step on the window [init_states, true_state].

        Args:
            init_states: (B, T, D, X, Y)
            true_state: (B, D, X, Y)

        Returns:
            next_state: (B, D, X, Y), one-step estimate of X_{t+1}.
            loss: (B, T, D, X, Y)
        """

        # (B, T+1, D, X, Y)
        target = torch.cat([init_states, true_state.unsqueeze(1)], dim=1)

        z0 = torch.randn_like(target)
        if self.args.corr_noise is not None:
            for i in range(1, z0.shape[1]):
                # AR(1) noise over time, keeping each z0 marginally N(0,1)
                z0[:, i] = self.args.corr_noise*z0[:, i-1] + \
                    torch.sqrt(
                        torch.tensor(1-self.args.corr_noise ** 2, device=z0.device))*z0[:, i]

        z1 = target

        # t ~ U(0, 1), shared by all variables of a time step.
        t = torch.rand([target.shape[0], target.shape[1]],
                       device=target.device)

        if self.args.time_dropout is not None:
            time_mask = torch.rand_like(
                t, device=target.device) <= self.args.time_dropout
            # NOTE: t=-1 signals dropped out time.
            t = torch.where(time_mask, -1, t)
        else:
            time_mask = torch.zeros_like(t, device=target.device).bool()

        alpha_t = self.schedule.alpha(t)
        beta_t = self.schedule.beta(t)
        zt = self.time_mix(alpha_t, z0) + self.time_mix(beta_t, z1)

        # Set z0, zt, z1 to zero when we mask it out
        z0 = torch.where(time_mask[..., None, None, None], z0*0., z0)
        zt = torch.where(time_mask[..., None, None, None], zt*0., zt)
        z1 = torch.where(time_mask[..., None, None, None], z1*0., z1)

        eta0, eta1, _ = self.get_etas(zt, alpha_t, beta_t)
        pred = (eta0, eta1)

        loss = self.loss_fn(
            pred,
            z0,
            z1,
        )  # (B)

        # One deterministic step to t=1 gives the predicted E[z1 | zt]
        dt = torch.where(time_mask, 0, 1-t)

        eps = 0.
        z1 = self.euler_step(zt=zt, t=t, dt=dt, eps=eps,
                             schedule=self.schedule)

        if self.args.fm_uncond:
            # We send in the time steps as a batch
            _, uncond_loss = self.fm_uncond.predict_step_train(
                z0.flatten(0, 1), t.flatten(0, 1), z1.flatten(0, 1)
            )
            # Mean over the time dimension
            loss = loss + torch.mean(uncond_loss.reshape(loss.shape[0], -1),
                                     dim=1)

        return z1[:, -1], loss

    def predict_step(self, init_states, obs=None):
        """Sample the next state X_{t+1} given init_states.

        Args:
            init_states: (B, T, D, X, Y)
            obs: optional observations; if given, sampling is guided towards
                them (requires self.observation_fn and self.obs_sigma).

        Returns:
            next_state: (B, D, X, Y)
        """
        latent_target = torch.randn_like(
            init_states[:, 0].unsqueeze(1))  # (B, 1, D, X, Y)

        # (B, T, D, X, Y)
        latents = torch.cat([init_states, latent_target], dim=1)

        # Run through sampler
        if self.args.sampler == "stochastic":
            z1 = self.eta_stochastic_sampler(
                latents=latents, num_steps=self.args.sampler_steps, eps=self.args.sampler_eps,
                schedule=self.forecast_schedule, obs=obs
            )
        else:
            raise NotImplementedError(
                f"Sampler {self.args.sampler} not implemented!")

        return z1[:, -1]

    # Samplers
    def eta_stochastic_sampler(
        self, latents, num_steps=100, eps=1.0, tmin=0.0, tmax=1.0, schedule=AlphaBetaSchedule(),
        snapshot_step=None, obs=None
    ):
        if snapshot_step is not None:
            snapshots = []
        # Time step discretization.
        ts = torch.linspace(tmin, tmax, num_steps+1, device=self.device)[:-1]
        dt = (tmax - tmin) / num_steps

        # Main sampling loop.
        zt = latents  # Initialize with noise
        for i, t in enumerate(ts):
            if snapshot_step is not None:
                if i % snapshot_step == 0:
                    snapshots.append(zt.detach().cpu())
            t = t.unsqueeze(-1).expand(zt.shape[0], zt.shape[1])  # (B, T)
            # Correction step
            if self.corrections > 0:
                for _ in range(self.corrections):
                    zt = self.correction_step(zt, t, dt, eps, schedule, obs)
            # Euler step
            zt = self.euler_step(zt, t, dt, eps, schedule, obs)

        if snapshot_step is not None:
            snapshots.append(zt.detach().cpu())
            return zt.detach(), snapshots
        return zt.detach()  # (B, T, D, X, Y)

    # NOTE: Legacy, unused.
    def euler_step_scalar(self, zt, t, dt, eps=1.0, schedule=AlphaBetaSchedule()):
        alpha_t = schedule.alpha(t)
        beta_t = schedule.beta(t)
        gamma_t = schedule.gamma(t)
        alpha_dot_t = schedule.alpha_dot(t)
        beta_dot_t = schedule.beta_dot(t)
        eps_t = eps * alpha_t

        eta0, eta1 = self.get_etas(zt, alpha_t, beta_t)
        dW = torch.randn_like(zt) * torch.sqrt(2*dt * eps_t)

        deta0 = dt * (alpha_dot_t-eps * alpha_dot_t.abs()) * eta0
        deta1 = dt * beta_dot_t * eta1
        zt = zt + deta0 + deta1 + dW

        if eta0.isnan().any():
            raise ValueError("NaN detected in eta0, stopping sampling.")
        elif eta1.isnan().any():
            raise ValueError("NaN detected in eta1, stopping sampling.")
        elif dW.isnan().any():
            raise ValueError("NaN detected in dW, stopping sampling.")
        elif zt.isnan().any():
            raise ValueError("NaN detected in zt, stopping sampling.")

        return zt

    def euler_step_old(self, zt, t, dt, eps=0., schedule=AlphaBetaSchedule(), obs=None):
        """Legacy Euler-Maruyama step; superseded by `euler_step`."""
        alpha_t = schedule.alpha(t)
        beta_t = schedule.beta(t)
        alpha_dot_t = schedule.alpha_dot(t)
        beta_dot_t = schedule.beta_dot(t)
        dt = torch.tensor(dt, device=zt.device)[..., None, None, None]
        eps_t = eps * alpha_t * alpha_dot_t.abs()  # zero where alpha is not moving

        if eps != 0.:
            eps = torch.tensor(eps, device=zt.device)  # * (1-t)

            xi = torch.randn_like(zt)
            if len(alpha_t.shape) == 2:
                L = torch.sqrt(eps_t)
            elif len(alpha_t.shape) == 3:
                # Stochastic sampling is only supported for scalar alpha.
                raise ValueError(
                    "Currently only support stochastic sampling with scalar alpha")
                T = alpha_dot_t.shape[1]
                I = torch.eye(T, device=alpha_dot_t.device)[None, :, :]
                eps = eps * I
                L = torch.linalg.cholesky(
                    2*dt*alpha_t@eps)  # Needs B,T,T matrix
            dW = torch.sqrt(2 * dt) * self.time_mix(L, xi)
        else:
            dW = torch.tensor(0., device=zt.device)

        guide = self.args.guide_method is not None and obs is not None

        # Detach zt to prevent gradients from flowing into the backbone during sampling
        zt = zt.detach()
        eta0, eta1, zt = self.get_etas(zt, alpha_t, beta_t, grad_eta1=guide)

        # Apply Guidance
        if guide:
            b_obs, s_obs = self.get_cond_drift_and_score(
                zt, t, obs, z1=eta1, schedule=schedule)

            b_obs = b_obs * self.guidance_strength(t)[..., None, None, None]
            s_obs = s_obs * self.guidance_strength(t)[..., None, None, None]

            d_obs = (b_obs + s_obs*eps_t[:, :, None, None, None]) * dt
            zt = zt + d_obs

        # eps_t / alpha_t = eps
        deta0 = dt * self.time_mix(alpha_dot_t - eps * alpha_dot_t.abs(), eta0)
        deta1 = dt * self.time_mix(beta_dot_t, eta1)

        zt = zt + deta0 + deta1 + dW

        if eta0.isnan().any():
            raise ValueError("NaN detected in eta0, stopping sampling.")
        if eta1.isnan().any():
            raise ValueError("NaN detected in eta1, stopping sampling.")
        if dW.isnan().any():
            raise ValueError("NaN detected in dW, stopping sampling.")
        if zt.isnan().any():
            raise ValueError("NaN detected in zt, stopping sampling.")

        return zt

    def euler_step(self, zt, t, dt, eps=0., schedule=AlphaBetaSchedule(), obs=None):
        """One Euler-Maruyama step of the (optionally guided) sampler.

        Args:
            zt: (B, T, D, X, Y)
            t: (B, T)
            dt: float or (B, T)
            eps: diffusion strength.
        """
        alpha_t = schedule.alpha(t)
        alpha_dot_t = schedule.alpha_dot(t)
        dt = torch.tensor(dt, device=zt.device)[..., None, None, None]
        eps_t = eps * alpha_t * alpha_dot_t.abs()  # zero where alpha is not moving

        if eps != 0.:
            xi = torch.randn_like(zt)
            L = torch.sqrt(eps_t)
            dW = torch.sqrt(2 * dt) * self.time_mix(L, xi)
        else:
            dW = torch.tensor(0., device=zt.device)

        drift, _ = self.get_drift_and_score(zt, t, eps, schedule, obs)
        zt = zt.detach() + dt * drift.detach() + dW

        if zt.isnan().any():
            raise ValueError("NaN detected in zt, stopping sampling.")

        return zt

    def corrector_live_mask(self, schedule, t):
        """Mask (B, T, 1, 1, 1) of window slots the Langevin corrector may update.

        Slots pinned at clean data (alpha(t) == 0) have an infinite score and a
        point-mass marginal, so the corrector must leave them untouched.
        """
        alpha_t = schedule.alpha(t)
        if alpha_t.ndim == 2:
            # (B, T) scalar schedules.
            live = alpha_t != 0
        elif alpha_t.ndim == 3:
            # (B, T, T) matrix schedules: slot i is masked if any weight feeding it is zero.
            live = (alpha_t != 0).all(dim=-1)
        else:
            raise ValueError(
                f"Unsupported alpha shape {tuple(alpha_t.shape)}; expected "
                "(B, T) or (B, T, T).")
        return live[..., None, None, None]

    def correction_step(self, zt, t, dt, eps=0., schedule=AlphaBetaSchedule(), obs=None):
        """One Langevin corrector step with step size tau / mean(score^2).

        Args:
            zt: (B, T, D, X, Y)
            t: (B, T)
            dt: float or (B, T)
            eps: diffusion strength.
        """
        _, score = self.get_drift_and_score(zt, t, eps, schedule, obs)

        live = self.corrector_live_mask(schedule, t)
        zero = torch.zeros_like(zt)
        # Zero the infinite score on pinned slots and keep their delta_t finite.
        score = torch.where(live, score, zero)
        mean_sq = score.square().mean(dim=(2, 3, 4), keepdim=True)
        delta_t = self.tau / torch.where(live, mean_sq, torch.ones_like(mean_sq))

        dz = score * delta_t
        dW = torch.randn_like(zt) * torch.sqrt(2 * delta_t)
        zt = zt.detach() + torch.where(live, dz.detach(), zero) \
            + torch.where(live, dW, zero)

        if zt.isnan().any():
            raise ValueError("NaN detected in zt, stopping sampling.")

        return zt


    def get_drift_and_score(self, zt, t, eps=0., schedule=AlphaBetaSchedule(), obs=None):
        """Drift and score of the (optionally guided) sampling SDE.

        Args:
            zt: (B, T, D, X, Y)
            t: (B, T)
            eps: diffusion strength.
            schedule: AlphaBetaSchedule.
            obs: optional observations for guidance.

        Returns:
            drift, score: (B, T, D, X, Y)
        """
        alpha_t = schedule.alpha(t)
        beta_t = schedule.beta(t)
        alpha_dot_t = schedule.alpha_dot(t)
        beta_dot_t = schedule.beta_dot(t)
        eps_t = eps * alpha_t * alpha_dot_t.abs()

        guide = self.args.guide_method is not None and obs is not None
        zt = zt.detach()
        eta0, eta1, zt = self.get_etas(zt, alpha_t, beta_t, grad_eta1=guide)

        # Apply Guidance
        d_obs, s_obs = 0., 0.
        if guide:
            b_obs, s_obs = self.get_cond_drift_and_score(
                zt, t, obs, z1=eta1, schedule=schedule)

            b_obs = b_obs * self.guidance_strength(t)[..., None, None, None]
            s_obs = s_obs * self.guidance_strength(t)[..., None, None, None]

            d_obs = (b_obs + s_obs*eps_t[:, :, None, None, None])

        deta0 = self.time_mix(alpha_dot_t - eps * alpha_dot_t.abs(), eta0)
        deta1 = self.time_mix(beta_dot_t, eta1)

        drift = deta0 + deta1 + d_obs
        score = self.time_mix(-1/alpha_t, eta0) + s_obs

        return drift, score

    def get_etas(self, zt, alpha_t, beta_t, grad_eta1=False):
        """Return (eta0, eta1, zt), each (B, T, D, X, Y).

        If grad_eta1, zt is returned with requires_grad so guidance can
        differentiate eta1 with respect to it.
        """
        const = torch.tensor(1e-16, device=zt.device, dtype=zt.dtype)
        beta_is_zero = torch.isclose(beta_t, torch.zeros_like(beta_t))
        alpha_is_zero = torch.isclose(alpha_t, torch.zeros_like(alpha_t))

        if grad_eta1 == True and self.args.fm_loss != 'eta01_label' and self.args.fm_loss != 'eta01_channel':
            raise NotImplementedError(
                "Grad eta1 is only implemented for eta01_label")

        if self.args.fm_loss == 'eta0':
            if len(alpha_t.shape) > 2 or len(beta_t.shape) > 2:
                raise NotImplementedError(
                    "Matrices for alpha/beta is not supported for eta0")
            eta0 = self.eta(self.backbone_model, zt, alpha_t, beta_t)
            # zt = alpha eta0 + beta eta1
            beta_t = torch.where(beta_is_zero, beta_t + const, beta_t)
            eta1 = (zt - alpha_t * eta0) / (beta_t)
            eta1 = torch.where(
                beta_is_zero, torch.zeros_like(eta1), eta1)

        elif self.args.fm_loss == 'eta1':
            if len(alpha_t.shape) > 2 or len(beta_t.shape) > 2:
                raise NotImplementedError(
                    "Matrices for alpha/beta is not supported for eta1")
            eta1 = self.eta(self.backbone_model, zt, alpha_t, beta_t)
            # zt = alpha eta0 + beta eta1
            alpha_t = torch.where(alpha_is_zero, alpha_t + const, alpha_t)
            eta0 = (zt - beta_t * eta1) / (alpha_t)
            eta0 = torch.where(
                alpha_is_zero, torch.zeros_like(eta0), eta0)

        elif self.args.fm_loss == 'eta01_label':
            if grad_eta1:
                with torch.no_grad():
                    eta0 = self.eta(self.backbone_model, zt, alpha_t, beta_t, class_labels=torch.zeros(
                        [zt.shape[0], 1], device=zt.device))
                with torch.enable_grad():
                    zt = zt.detach().requires_grad_(True)
                    eta1 = self.eta(self.backbone_model, zt, alpha_t, beta_t, class_labels=torch.ones(
                        [zt.shape[0], 1], device=zt.device))
            else:
                eta0 = self.eta(self.backbone_model, zt, alpha_t, beta_t, class_labels=torch.zeros(
                    [zt.shape[0], 1], device=zt.device))
                eta1 = self.eta(self.backbone_model, zt, alpha_t, beta_t, class_labels=torch.ones(
                    [zt.shape[0], 1], device=zt.device))

        elif self.args.fm_loss == 'eta01_channel':
            if grad_eta1:
                with torch.enable_grad():
                    zt = zt.detach().requires_grad_(True)
                    eta0, eta1 = self.eta(
                        self.backbone_model, zt, alpha_t, beta_t)
            else:
                eta0, eta1 = self.eta(self.backbone_model, zt, alpha_t, beta_t)

        elif self.args.fm_loss == 'eta01_model':
            eta0 = self.eta(self.backbone_model0, zt, alpha_t, beta_t)
            eta1 = self.eta(self.backbone_model1, zt, alpha_t, beta_t)

        elif self.args.fm_loss == 'eta01_channel':
            eta0, eta1 = self.eta(self.backbone_model, zt, alpha_t, beta_t)

        else:
            raise NotImplementedError(
                f"Sampler for loss {self.args.fm_loss} not implemented.")

        return eta0, eta1, zt  # (B, T, D, X, Y)

    def get_cond_drift_and_score(self, zt, t, obs, z1, schedule):
        """Observation-guidance drift and score, both (B, T, D, X, Y)."""
        alpha_t = schedule.alpha(t)
        beta_t = schedule.beta(t)
        gamma_t = schedule.alpha(t) * schedule.beta_dot(t) - \
            schedule.beta(t) * schedule.alpha_dot(t)

        # No observations in the window: the likelihood gradient is exactly zero.
        if not has_observations(obs):
            zeros = torch.zeros_like(zt)
            return zeros, zeros.clone()

        s_obs = self.guide(obs, zt, t, z1, schedule=schedule)
        b_obs = s_obs * alpha_t[..., None, None, None] * \
            gamma_t[..., None, None, None]
        if self.args.guide_method == "MMPS":
            b_obs = b_obs / beta_t.clamp_min(1e-6)[..., None, None, None]

        mask = ((t > 0.0) & (t < 1.0))[..., None, None, None].expand_as(b_obs)

        b_obs[~mask] = 0
        s_obs[~mask] = 0

        return b_obs, s_obs

    def set_guide_method(self, guide_method):
        self.guide_method = guide_method
        if guide_method == 'DPS':
            self.guide = partial(self.DPS)
        elif guide_method == 'DPS_scale':
            self.guide = partial(self.DPS, scale=True)
        elif guide_method == 'MMPS':
            self.guide = partial(self.MMPS)
        elif guide_method is None or guide_method == 'None':
            self.guide = lambda obs, zt, t, z1, **kwargs: torch.zeros_like(zt)
        else:
            raise ValueError(f"{guide_method} is invalid guide_method")

    # Guidance methods
    def DPS(self, obs, zt, t, z1, scale=False, **kwargs):
        """Diffusion posterior sampling guidance, grad_zt log p(obs | z1(zt)).

        Args:
            obs: observations.
            zt: current state (requires grad).
            t: current time, (B, T).
            z1: estimate of the clean state, a function of zt.
            scale: if True, use the unsquared residual norm of the DPS paper.
        """

        with torch.enable_grad():
            error = obs - self.observation_fn(z1)
            scalar = 1 / (2 * self.obs_sigma**2)
            dim = tuple(range(1, len(error.shape)))
            if scale:
                # No scalar here
                logp = - torch.linalg.vector_norm(error, dim=dim)
            else:
                logp = -(scalar * error ** 2).sum(dim=dim)

        s_obs = torch.autograd.grad(
            outputs=logp,
            inputs=zt,
            grad_outputs=torch.ones_like(logp),
        )[0]

        return s_obs.detach()

    def MMPS(self, obs, zt, t, z1, schedule, solver="gmres", iterations=1, **kwargs):
        """Moment-matching posterior sampling guidance.

        Args:
            obs: observations.
            zt: current state (requires grad).
            t: current time, (B, T).
            z1: estimate of the clean state, a function of zt.
            solver: "cg" or "gmres" for the linear system in observation space.
            iterations: solver iterations.
        """
        if solver == "cg":
            solve = partial(cg, iterations=iterations)
        elif solver == "gmres":
            solve = partial(gmres, iterations=iterations)

        with torch.enable_grad():
            y_hat = self.observation_fn(z1)

        def A(v):
            return torch.func.jvp(self.observation_fn, (z1.detach(),), (v,))[1]

        def At(v):
            return torch.autograd.grad(y_hat, z1, v, retain_graph=True)[0]

        # fmt: off
        def cov_x(v):
            Jv = torch.autograd.grad(z1, zt, v, retain_graph=True)[0]
            beta_t = schedule.beta(t).clamp_min(1e-6)
            scale = (schedule.alpha(t)**2 / beta_t)[..., None, None, None]
            return scale * Jv
        # fmt: on

        def cov_y(v):
            return self.obs_sigma**2 * v + A(cov_x(At(v)))

        grad = obs - y_hat
        grad = solve(A=cov_y, b=grad)
        grad = torch.autograd.grad(y_hat, zt, grad)[0]

        return grad
