"""A trained forecasting checkpoint used as a one-step ensemble forecaster.

Gives datasets without numerical dynamics (e.g. SEVIR) a forward model with the
same `forward(xens)` interface as `SQGModel`. The architecture is read from the
checkpoint. `forward` works in assimilation units and converts internally.
"""

from __future__ import annotations

import numpy as np
import torch

from forecasting.ckpt_args import (
    apply_checkpoint_arch,
    arch_from_checkpoint,
    check_nx,
    read_checkpoint,
    resolved_model_cls,
    warn,
)


class LearnedForecaster:
    """One-step ensemble forecaster backed by a trained model.

    Args:
        args: parsed arguments; `forward_model_path` names the checkpoint, whose
            recorded architecture is what the model is built with.
        md: DatasetMetadata, for the physical<->normalized conversion.
        device: torch device.
        model_cls: model class to build. Defaults to what the checkpoint says.
        label: what to call this model in the architecture banner.
    """

    def __init__(self, args, md, device, model_cls=None, label="propagator"):
        path = getattr(args, "forward_model_path", None)
        if not path:
            raise ValueError(
                f"--forward_model {getattr(args, 'forward_model', 'learned')} needs "
                "--forward_model_path pointing at a trained forecasting checkpoint "
                "(a Lightning .ckpt, or a raw state_dict)."
            )

        self.md = md
        self.device = device

        info = read_checkpoint(path, map_location=device)
        arch = arch_from_checkpoint(info)
        check_nx(arch, md, label=label)

        # Build from a copy of `args` carrying the checkpoint architecture.
        from assimilation.parser import arch_defaults

        # A propagator differing from the run's --init_states is expected.
        model_args = apply_checkpoint_arch(
            args, info, label=label, defaults=arch_defaults(),
            quiet={"init_states"})

        # Conditioning length: checkpoint, else --forward_model_init_states,
        # else --init_states.
        requested_init = getattr(args, "forward_model_init_states", None)
        if "init_states" in arch:
            self.init_states = int(arch["init_states"])
            if requested_init and int(requested_init) != self.init_states:
                warn(f"--forward_model_init_states {requested_init} ignored: "
                     f"{path} was trained with init_states={self.init_states}.")
        else:
            self.init_states = int(
                requested_init or getattr(args, "init_states", 1) or 1)
        model_args.init_states = self.init_states

        # Fill attributes FMW/ARModel need that the assimilation parser lacks.
        from forecasting.models.ar_model import apply_inference_defaults
        apply_inference_defaults(model_args)

        cls = model_cls or resolved_model_cls(info, args)
        self.model = cls(model_args).to(device)
        try:
            self.model.load_state_dict(info.state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{cls.__name__} does not match the checkpoint at {path}.\n"
                f"  Built with init_states={self.init_states}, "
                f"hidden_dim={getattr(model_args, 'hidden_dim', None)}, "
                f"nx={getattr(model_args, 'nx', None)}.\n"
                f"  These came from the checkpoint itself"
                + ("" if arch else " -- except that this checkpoint records no "
                                   "architecture (a bare state_dict), so they came "
                                   "from the command line")
                + ". A mismatch therefore means the file is not what it claims to "
                f"be, or --ignore_ckpt_arch is in effect.\n"
                f"  Original error: {exc}"
            ) from exc
        self.model.eval()
        self.arch = arch

        # Assimilation -> network units; `shift` is nonzero only with --forward_norm.
        self.scale = md.assim_scale(self.model.data_std)
        self.scale, self.shift = self._resolve_affine(
            md, getattr(args, "forward_norm", None))
        print(
            f"LearnedForecaster: {cls.__name__} from {path}\n"
            f"  init_states={self.init_states} "
            f"scale={torch.as_tensor(self.scale).flatten().tolist()} "
            f"shift={self.shift:g}",
            flush=True,
        )

    def _resolve_affine(self, md, forward_norm):
        """Return `(scale, shift)` mapping assimilation units to the checkpoint's.

        With `--forward_norm`, `z = (v - (c_mean - m_a) / s_a) / (c_std / s_a)`;
        otherwise shift is 0.
        """
        named = md.named_norm(forward_norm)
        if named is None:
            return self.scale, 0.0
        if getattr(md, "assim_space", "physical") != "normalized":
            raise ValueError(
                f"--forward_norm {forward_norm!r} needs a dataset that "
                f"assimilates in model space (assim_space 'normalized'); "
                f"{md.name}/{md.variant} assimilates in '{md.assim_space}', "
                f"where the checkpoint's own data_std is already the bridge."
            )
        c_mean, c_std = named
        # A named norm is scalar, so the bridge is scalar too.
        m_a, s_a = (float(t[0]) for t in md.norm.mean_std(per_channel=False))
        scale = torch.full_like(
            torch.as_tensor(self.scale, dtype=torch.float32), c_std / s_a)
        return scale, (c_mean - m_a) / s_a

    def _to_normalized(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) / torch.as_tensor(self.scale).to(x.device, x.dtype)

    def _to_physical(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.as_tensor(self.scale).to(x.device, x.dtype) + self.shift

    def _as_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input to `(n_ens, init_states, C, ny, nx)`.

        Accepts `(n_ens, C, ny, nx)` or `(window, n_ens, C, ny, nx)`; short inputs
        are front-padded by repeating the oldest state.
        """
        if x.dim() == 4:                      # (n_ens, C, ny, nx)
            cond = x.unsqueeze(1)             # -> (n_ens, 1, C, ny, nx)
        elif x.dim() == 5:                    # (window, n_ens, C, ny, nx)
            cond = x.permute(1, 0, 2, 3, 4)   # -> (n_ens, window, C, ny, nx)
        else:
            raise ValueError(
                f"expected a 4-D or 5-D conditioning tensor, got {tuple(x.shape)}"
            )

        if cond.shape[1] > self.init_states:
            cond = cond[:, -self.init_states:]
        elif cond.shape[1] < self.init_states:
            pad = cond[:, :1].expand(-1, self.init_states - cond.shape[1], -1, -1, -1)
            cond = torch.cat([pad, cond], dim=1)
        return cond

    def advance(self, x, timesteps=None):
        """Differentiable one-step forecast in normalized units.

        Same signature as `SQGTorch.advance`. `timesteps` is ignored. The caller applies the unit scaling (requires
        `shift == 0`).

        Args:
            x: `(n_ens, C, ny, nx)` (or a 5-D window) in normalized units.

        Returns:
            `(n_ens, C, ny, nx)`, normalized units, with gradient history.
        """
        cond = self._as_conditioning(torch.as_tensor(x))
        # Cast to the model dtype (float32).
        param_dtype = next(self.model.parameters()).dtype
        pred = self.model.predict_step(cond.to(device=self.device, dtype=param_dtype))
        if pred.dim() == 5:                   # (B, 1, C, ny, nx)
            pred = pred[:, -1]
        return pred.to(dtype=cond.dtype)

    @torch.no_grad()
    def forward(self, xens):
        """Advance an ensemble one assimilation step.

        Args:
            xens: `(n_ens, C, ny, nx)` or `(window, n_ens, C, ny, nx)`, physical units.

        Returns:
            `(n_ens, C, ny, nx)` numpy array in physical units.
        """
        x = torch.as_tensor(np.asarray(xens), dtype=torch.float32, device=self.device)
        cond = self._as_conditioning(x)
        pred = self.model.predict_step(self._to_normalized(cond))
        if pred.dim() == 5:                   # (B, 1, C, ny, nx)
            pred = pred[:, -1]
        return self._to_physical(pred).cpu().numpy()
