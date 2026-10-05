# Deterministic U-Net model for time series forecasting
import torch

from forecasting.models.ar_model import ARModel
from networks.diffusion_networks import SongUNet


class UNET(ARModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = SongUNet(
            # Wrap-around padding only for periodic domains.
            circular_padding=all(self.md.periodic),
            img_resolution=torch.as_tensor(args.nx),
            in_channels=self.ch.cond,
            out_channels=self.ch.state,
            embedding_type="fourier",
            model_channels=args.hidden_dim,
            resample_filter=args.resample_filter,
            channel_mult=args.channel_mult,
            encoder_type=args.encoder_type,
            attn_resolutions=args.attn_resolutions,
            channel_mult_emb=0,
            channel_mult_noise=0,
        )

    def predict_step(self, init_states):
        """Predict X_{t+1} (B, d_state, X, Y) from init_states (B, init_states, d_state, X, Y)."""
        input_grid = init_states.reshape(init_states.shape[0],
                                         -1,
                                         init_states.shape[3],
                                         init_states.shape[4])

        next_state = self.backbone(input_grid)

        # Add residual if needed
        if self.args.pred_residual:
            # TODO: Enable support for pred_residual
            next_state = (next_state * self.diff_std) + \
                self.diff_mean  # Unormalize residual
            next_state = init_states[:, -1] + next_state

        return next_state

    def predict_step_train(self, init_states, true_state):
        """Predict X_{t+1} during training.

        Args:
            init_states: (B, N_steps, d_state, X, Y)
            true_state: (B, d_state, X, Y)

        Returns:
            next_state: (B, d_state, X, Y)
            loss: (B,)
        """

        input_grid = init_states.reshape(init_states.shape[0],
                                         -1,
                                         init_states.shape[3],
                                         init_states.shape[4])

        next_state = self.backbone(input_grid)

        # Add residual if needed
        if self.args.pred_residual:
            # TODO: Enable support for pred_residual
            next_state = (next_state * self.diff_std) + \
                self.diff_mean  # Unormalize residual
            next_state = init_states[:, -1] + next_state

        loss = self.loss(next_state, true_state)

        return next_state, loss
