"""FlowDAS's pretrained stochastic-interpolant forecaster, for any dataset.

Reference implementation: https://github.com/umjiayx/FlowDAS. The SQG and SEVIR
checkpoints differ in architecture and normalization, which are configured in
`FLOWDAS_PRESETS`. Inputs are in the run's assimilation units and are mapped to
the checkpoint's training units by `_resolve_affine`.
"""

from __future__ import annotations

import math
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn

from forecasting.flowdas_models import DriftModel, Interpolant

#: Interpolant configuration of the released checkpoints (`sigma_coef=1`).
DEFAULT_INTERP = dict(
    sigma_coef=1.0,
    beta_fn="t",
    t_min_train=0.0,
    t_max_train=1.0,
    t_min_sampling=0.001,
    t_max_sampling=0.999,
    EM_sample_steps=200,
)

#: Backbone settings shared by both checkpoints.
_SHARED_ARCH = dict(
    unet_resnet_block_groups=8,
    unet_learned_sinusoidal_dim=32,
    unet_attn_dim_head=64,
    unet_attn_heads=4,
    unet_learned_sinusoidal_cond=True,
    unet_random_fourier_features=False,
    use_classes=False,
)

#: dataset -> architecture and normalization of its FlowDAS checkpoint.
FLOWDAS_PRESETS: dict[str, dict] = {
    "SQG": dict(
        # 2 channels, 6 conditioning frames -> init_conv in_channels 14.
        num_channels=2,
        cond_frames=6,
        unet_channels=64,
        unet_dim_mults=(1, 2, 2),
        # Raw PV -> drift units: divide by 2660, no offset.
        ckpt_mean=0.0,
        ckpt_std=2660.0,
        path_var="FLOWDAS_MODEL_PATH",
        path_default="SQG/models/flowdas/flowdas_sqg_3hrly.pt",
        # `{"model": state_dict, "opt": ..., "step": ...}`.
        state_keys=("model",),
        key_prefix=None,
    ),
    "SEVIR": dict(
        # 1 channel (VIL), 6 conditioning frames -> init_conv in_channels 7.
        num_channels=1,
        cond_frames=6,
        unet_channels=128,
        unet_dim_mults=(1, 2, 2, 2),
        # Raw VIL -> drift units: (vil/255 - 0.5) / 0.1 == (vil - 127.5) / 25.5.
        ckpt_mean=127.5,
        ckpt_std=25.5,
        path_var="FLOWDAS_SEVIR_MODEL_PATH",
        path_default="SQG/models/flowdas/flowdas_sevir.pt",
        # `{"model_state_dict": ..., "optimizer_state_dict": ..., "step": ...}`.
        state_keys=("model_state_dict",),
        # Upstream names the UNet `_arch`; DriftModel here calls it `net`.
        key_prefix=("_arch.", "net."),
    ),
}


class FlowDASModel(nn.Module):
    """FlowDAS's pretrained drift + interpolant as a one-step ensemble forecaster.

    Args:
        device: torch device.
        dataset: key into `FLOWDAS_PRESETS`.
        path: checkpoint override; defaults to the preset's path.
        md: DatasetMetadata, used to map assimilation units to the checkpoint's
            units. Without it, inputs are assumed to be in raw units.
        forward_norm: named normalization the checkpoint was trained in.
    """

    def __init__(self, device, dataset="SQG", path=None, md=None,
                 forward_norm=None):
        super().__init__()
        if dataset not in FLOWDAS_PRESETS:
            raise ValueError(
                f"no FlowDAS checkpoint is configured for dataset {dataset!r}; "
                f"available: {sorted(FLOWDAS_PRESETS)}. Add a preset to "
                f"FLOWDAS_PRESETS with the architecture read off the checkpoint."
            )
        preset = FLOWDAS_PRESETS[dataset]
        self.dataset = dataset
        self.preset = preset

        from data.paths import checkpoint

        ckpt_path = path or checkpoint(preset["path_var"], preset["path_default"])

        C = int(preset["num_channels"])
        cond_channels = int(preset["cond_frames"]) * C

        data_cfg = SimpleNamespace(C=C)
        model_cfg = SimpleNamespace(
            unet_channels=preset["unet_channels"],
            unet_dim_mults=tuple(preset["unet_dim_mults"]),
            **_SHARED_ARCH,
        )
        interp_cfg = SimpleNamespace(**DEFAULT_INTERP)

        drift = DriftModel(data_cfg=data_cfg, model_cfg=model_cfg,
                           cond_channels=cond_channels).to(device)
        drift.eval()
        drift.load_state_dict(self._load_state(ckpt_path, preset, device),
                              strict=True)

        interp = Interpolant(interp_cfg)

        self.sampler = partial(
            self.em_sample_uncond,
            drift=drift, interp=interp,
            steps=interp_cfg.EM_sample_steps,
            t_min=interp_cfg.t_min_sampling,
            t_max=interp_cfg.t_max_sampling,
        )

        self.device = device
        # Exposed for the guided FlowDAS assimilation method.
        self.drift = drift
        self.interp = interp
        self.interp_cfg = interp_cfg
        self.num_channels = C
        self.cond_frames = int(preset["cond_frames"])

        # Assimilation units -> drift units: `z = (x - norm_mean) / norm_std`.
        self.norm_mean, self.norm_std = self._resolve_affine(
            preset, md, forward_norm=forward_norm)
        print(f"FlowDASModel[{dataset}]: {ckpt_path}\n"
              f"  channels={C} cond_frames={self.cond_frames} "
              f"assim_space={getattr(md, 'assim_space', 'physical (no md)')} "
              f"norm=(x - {self.norm_mean:g}) / {self.norm_std:g}", flush=True)

    @staticmethod
    def _resolve_affine(preset, md, forward_norm=None):
        """Return `(mean, std)` mapping assimilation units to the drift's units.

        Composes assim -> raw (`raw = v * s_a + m_a`) with raw -> drift
        (`z = (raw - c_mean) / c_std`), giving `mean = (c_mean - m_a) / s_a` and
        `std = c_std / s_a`.
        """
        c_mean = float(preset["ckpt_mean"])
        c_std = float(preset["ckpt_std"])
        named = None if md is None else md.named_norm(forward_norm)
        if named is not None:
            c_mean, c_std = named
        if md is None:
            # No metadata: assume raw units.
            return c_mean, c_std
        if getattr(md, "assim_space", "physical") == "normalized":
            md_mean, md_std = md.norm.mean_std(per_channel=False)
            s_a = float(md_std[0])
            m_a = float(md_mean[0])
        else:
            s_a = 1.0 / float(md.unit_factor)
            m_a = 0.0
        return (c_mean - m_a) / s_a, c_std / s_a

    @staticmethod
    def _load_state(path, preset, device):
        """Extract the drift's state dict from the checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt
        if isinstance(ckpt, dict):
            for key in preset["state_keys"]:
                if key in ckpt:
                    state = ckpt[key]
                    break
        prefix = preset.get("key_prefix")
        if prefix:
            old, new = prefix
            state = {(new + k[len(old):] if k.startswith(old) else k): v
                     for k, v in state.items()}
        return state

    def to_norm(self, x):
        return (x - self.norm_mean) / self.norm_std

    def from_norm(self, z):
        return z * self.norm_std + self.norm_mean

    @torch.no_grad()
    def em_sample_uncond(self, drift, interp, base, cond, steps=300,
                         t_min=0.0, t_max=0.999):
        """Unguided Euler-Maruyama from `base` (z0) to t=1, conditioned on `cond`."""
        device = base.device
        ts = torch.linspace(t_min, t_max, steps, device=device)
        dt = float(ts[1] - ts[0])
        xt = base.clone()

        for t in ts:
            tb = t.repeat(xt.shape[0]).to(device)         # (B,)
            bF = drift(xt, tb, cond=cond)                 # (B,C,H,W)
            sig = interp.sigma(tb)                        # (B,1,1,1)
            mu = xt + bF * dt
            xt = mu + sig * torch.randn_like(mu) * math.sqrt(dt)

        return xt  # (B,C,H,W)

    def build_cond_z0_z1(self, x):
        """`(B, T, C, H, W)` with `T >= 2` -> `(cond, z0)`.

        cond = the whole window flattened over time, `(B, T*C, H, W)`;
        z0 = the most recent frame, `(B, C, H, W)`.
        """
        assert x.dim() == 5 and x.shape[1] >= 2
        B, T, C, H, W = x.shape
        cond = x.reshape(B, T * C, H, W).contiguous()
        z0 = x[:, -1]
        return cond, z0

    def forward(self, xens):
        """Forecast one step ahead, X_t -> X_t+1.

        Args:
            xens: `(window, n_ens, C, H, W)` in the run's assimilation units.

        Returns:
            `(n_ens, C, H, W)` numpy array in the run's assimilation units.
        """
        x = self.to_norm(torch.as_tensor(xens, device=self.device))
        # (T, B, C, H, W) -> (B, T, C, H, W)
        x = x.permute(1, 0, 2, 3, 4)
        cond, z0 = self.build_cond_z0_z1(x)
        pred = self.sampler(base=z0, cond=cond)
        return self.from_norm(pred).detach().cpu().numpy()
