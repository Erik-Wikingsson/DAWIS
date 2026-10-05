# Flow Matching model for time series forecasting
# Third-party
import torch
import wandb
import matplotlib.pyplot as plt

from plotting import plotting
from metrics import metrics
from forecasting.models.ar_prob_model import ARProbModel
from networks.diffusion_networks import SongUNet


class FMUncond(ARProbModel):
    """Unconditional flow-matching model of single states."""

    def __init__(self, args):
        super().__init__(args)

        self.backbone = SongUNet(
            # Wrap-around padding only for periodic domains.
            circular_padding=all(self.md.periodic),
            img_resolution=torch.as_tensor(args.nx),
            in_channels=self.ch.state,
            out_channels=self.ch.state,
            embedding_type="fourier",
            model_channels=args.hidden_dim,
            resample_filter=args.resample_filter,
            channel_mult=args.channel_mult,
            encoder_type=args.encoder_type,
            attn_resolutions=args.attn_resolutions,
            channel_mult_emb=args.channel_mult_emb,
            channel_mult_noise=args.channel_mult_noise,
        )

    def predict_step_train(self, z0, t, z1):
        """Flow-matching training step from noise z0 to data z1.

        Args:
            z0: (B, d_state, X, Y) noise.
            t: unused; a fresh t ~ U(0, 1) is drawn.
            z1: (B, d_state, X, Y) data.

        Returns:
            pred_state: (B, d_state, X, Y), one-step estimate of z1.
            loss: (B,)
        """

        z0 = z0
        t = torch.rand([z1.shape[0], 1, 1, 1],
                       device=z1.device)
        zt = (1 - t) * z0 + t * z1

        # Shape (B, d_state, N_x, N_y)
        pred_drift = self.backbone(zt, t.flatten())

        # This predicts the drift b
        drift = z1 - z0

        loss = self.loss(
            pred_drift,
            drift,
        )  # (B)

        # This is the predicted E[z1 | zt]
        pred_state = zt + pred_drift * (1-t)

        return pred_state, loss

    def predict_step(self, latents=None):
        """Sample a state from latents (B, d_state, X, Y), drawn if None.

        Returns:
            (B, d_state, X, Y)
        """
        if latents is None:
            latents = torch.randn_like(self.args.batch_size,
                                       2,
                                       self.args.nx,
                                       self.args.nx)  # (B, d_state, X, Y)

        # Run through sampler
        if self.args.sampler == "heun":
            next_state = self.heun_sampler(
                latents=latents,
            )
        elif self.args.sampler == "stochastic":
            next_state = self.stochastic_sampler(
                latents=latents,
            )

        return next_state

    # Samplers
    def heun_sampler(
        self, latents, class_labels=None, num_steps=20, tmin=0.0, tmax=1.0
    ):

        # Time step discretization.
        t_steps = torch.linspace(
            tmin, tmax, num_steps+1, device=self.device)

        # Main sampling loop.
        x_next = latents
        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            t_cur = t_cur.reshape(-1, 1, 1, 1).flatten()
            t_next = t_next.reshape(-1, 1, 1, 1).flatten()

            # Euler step.
            x_cur = x_next
            d_cur = self.backbone(
                x_cur, t_cur.flatten(), class_labels)
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                d_prime = self.backbone(
                    x_next, t_next.flatten(), class_labels)
                x_next = x_cur + (t_next - t_cur) * \
                    (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def stochastic_sampler(
        self, latents, class_labels=None, num_steps=100, eps=1.0, tmin=0.0, tmax=1.0
    ):

        # Time step discretization.
        ts = torch.linspace(tmin, tmax, num_steps+1, device=self.device)[:-1]
        dt = (tmax - tmin) / num_steps

        # Main sampling loop.
        zt = latents  # Initialize with noise
        for t in ts:
            t = t.reshape(-1, 1, 1, 1).flatten()
            beta_t = t
            alpha_t = 1 - t
            gamma_t = 1
            beta_dot_t = 1
            alpha_dot_t = -1
            eps_t = eps * alpha_t

            b = self.backbone(zt, t.flatten(), class_labels)
            s = (beta_t * b - beta_dot_t * zt) / \
                (alpha_t * gamma_t)  # s = (t * b - zt) / (1 - t)
            dz = b + eps_t * s

            dW = torch.randn_like(zt) * torch.sqrt(2*dt * eps_t)

            zt = zt + dz * dt + dW

        return zt  # (B, d_state, X, Y)
