"""Backbone construction for the unconditional generative model.

Used both for training (`unconditional_generation/trainer.py`) and for loading
checkpoints (`assimilation/methods/daisi.py`); the shape comes from dataset
metadata.
"""

from __future__ import annotations

from networks.diffusion_networks import SongUNet

# Architecture of the released DAISI checkpoints.
DEFAULT_ARCH = dict(
    embedding_type="fourier",
    encoder_type="residual",
    decoder_type="standard",
    channel_mult_noise=2,
    resample_filter=[1, 3, 3, 1],
    channel_mult=[2, 2, 2],
    attn_resolutions=[32],
)


def build_backbone(md, args=None, **overrides) -> SongUNet:
    """Build the unconditional SongUNet backbone for a dataset.

    Shape comes from `md`; architecture from DEFAULT_ARCH, updated by `args`
    where set, then by `overrides`. Circular padding is used only if the grid
    is periodic (`md.periodic`).
    """
    arch = dict(DEFAULT_ARCH)
    if args is not None:
        for key in ("channel_mult", "attn_resolutions", "resample_filter",
                    "embedding_type", "encoder_type", "decoder_type",
                    "channel_mult_noise", "num_blocks", "dropout"):
            value = getattr(args, key, None)
            if value is not None:
                arch[key] = value
    arch.update(overrides)

    model_channels = arch.pop("model_channels", None)
    if model_channels is None:
        model_channels = getattr(args, "channels", None) or 32

    if md.ny != md.nx:
        raise NotImplementedError(
            f"{md.name}/{md.variant}: SongUNet takes a single img_resolution, "
            f"but the grid is non-square {md.grid}"
        )

    return SongUNet(
        img_resolution=md.nx,
        in_channels=md.num_channels,
        out_channels=md.num_channels,
        model_channels=model_channels,
        circular_padding=all(md.periodic),
        **arch,
    )
