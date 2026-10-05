"""Read model architecture from a checkpoint's saved training args.

Lightning checkpoints store the training namespace at
`ckpt["hyper_parameters"]["args"]`. Only `ARCH_KEYS` (what sizes the weights) are
taken from it; sampling, guidance and dataset settings stay as given on the
command line. `merge_arch` returns a copy, so several models with different
architectures can coexist in one run.
"""

from __future__ import annotations

import copy
import importlib
import sys
from dataclasses import dataclass

import torch

#: Keys that determine the weight shapes (and, for `schedule`, the interpolant).
ARCH_KEYS = (
    "nx",
    "init_states",
    "hidden_dim",
    "channel_mult",
    "channel_mult_emb",
    "channel_mult_noise",
    "attn_resolutions",
    "resample_filter",
    "encoder_type",
    "noise_embedding",
    "fm_loss",
    "schedule",
    "alpha_beta_mult",
    "alpha_beta_spatial",
    "fm_forecast",
    "fm_uncond",
    "pred_residual",
    "spatial_noise_dim",
    "noise_dim",
    "latent_channels",
)

#: Keys whose recorded `None` is a real training value and is adopted.
NONE_IS_MEANINGFUL = frozenset({"alpha_beta_mult", "latent_channels", "time_dropout"})

#: Upper-cased `args.model` -> class to rebuild.
MODEL_CLASSES = {
    "FMW": "forecasting.models.fmw:FMW",
    "UNET": "forecasting.models.unet:UNET",
}


def warn(message: str) -> None:
    """Print a `WARNING:` line to stderr."""
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class CheckpointInfo:
    """Contents of a checkpoint, loaded once."""

    path: str
    #: The weights, already unwrapped from `ckpt["state_dict"]` where there is one.
    state_dict: dict
    #: `vars(hyper_parameters["args"])`, or None for a bare state_dict.
    train_args: dict | None
    #: `train_args["model"]` -- 'FMW', 'UNET', ... -- or None.
    model: str | None
    #: True for a Lightning checkpoint, False for a bare state_dict.
    is_lightning: bool


def _train_args(hyper_parameters) -> dict | None:
    """Extract the training args dict from Lightning `hyper_parameters`.

    Accepts `{"args": Namespace}`, a bare `Namespace`, or a flat dict.
    """
    if hyper_parameters is None:
        return None
    if hasattr(hyper_parameters, "__dict__") and not isinstance(hyper_parameters, dict):
        hyper_parameters = vars(hyper_parameters)
    if not isinstance(hyper_parameters, dict):
        return None

    inner = hyper_parameters.get("args")
    if inner is not None:
        if not isinstance(inner, dict):
            inner = vars(inner) if hasattr(inner, "__dict__") else None
        return inner
    # A flat dict is only usable if it actually holds architecture.
    if any(key in hyper_parameters for key in ARCH_KEYS):
        return dict(hyper_parameters)
    return None


def read_checkpoint(path, map_location="cpu") -> CheckpointInfo:
    """Load a checkpoint and return its weights and training args.

    `weights_only=False` is needed to unpickle the stored `argparse.Namespace`.
    """
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict, is_lightning = ckpt["state_dict"], True
        train_args = _train_args(ckpt.get("hyper_parameters"))
    else:
        # Bare state_dict: no architecture recorded.
        state_dict, is_lightning, train_args = ckpt, False, None

    model = None
    if train_args is not None:
        model = train_args.get("model")
    return CheckpointInfo(
        path=str(path),
        state_dict=state_dict,
        train_args=train_args,
        model=str(model) if model is not None else None,
        is_lightning=is_lightning,
    )


def arch_from_checkpoint(info: CheckpointInfo) -> dict:
    """Return the `ARCH_KEYS` recorded in the checkpoint.

    Missing keys, and `None` values outside `NONE_IS_MEANINGFUL`, are left to
    the caller.
    """
    if info.train_args is None:
        return {}
    arch = {}
    for key in ARCH_KEYS:
        if key not in info.train_args:
            continue
        value = info.train_args[key]
        if value is None and key not in NONE_IS_MEANINGFUL:
            continue
        arch[key] = value
    return arch


def _equal(a, b) -> bool:
    """Compare values, treating lists/tuples and int/float as equivalent."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:  # pragma: no cover - exotic values
        return False


def describe_arch(arch: dict) -> str:
    """The architecture as one compact line, for banners and run metadata."""
    return " ".join(f"{k}={v}" for k, v in arch.items())


def merge_arch(base_args, arch: dict, *, label: str, defaults: dict | None = None,
               path: str | None = None, quiet: frozenset | set | None = None):
    """Return a copy of `base_args` with `arch` applied, logging changes.

    Differences from parser defaults are printed as info; differences from
    explicitly chosen values are printed as warnings.

    Args:
        base_args: the run's namespace. Never mutated.
        arch: from `arch_from_checkpoint`.
        label: model name for the log ("DAWIS", "propagator", ...).
        defaults: `{key: parser default}`. Without it every difference is warned.
        path: the checkpoint path, for the log.
        quiet: keys whose difference is expected and not warned about
            (e.g. `init_states` for a propagator).

    Returns:
        A shallow copy of `base_args` with the architecture applied.
    """
    merged = copy.copy(base_args)
    if not arch:
        if path is not None:
            print(f"{label}: no training args in {path}; "
                  f"using the command-line architecture", flush=True)
        return merged

    defaults = defaults or {}
    quiet = quiet or frozenset()
    overridden, filled = [], []
    for key, value in arch.items():
        current = getattr(base_args, key, None)
        setattr(merged, key, value)
        if _equal(current, value):
            continue
        if key in quiet or (key in defaults and _equal(current, defaults[key])):
            filled.append((key, value))
        else:
            overridden.append((key, current, value))

    print(f"{label}: architecture from {path or '<checkpoint>'}", flush=True)
    for key, current, value in overridden:
        warn(f"  --{key} {current} overridden by checkpoint -> {value}")
    if filled:
        print("  from checkpoint: "
              + ", ".join(f"{k}={v}" for k, v in filled), flush=True)
    if overridden:
        warn(f"  {len(overridden)} command-line value(s) overridden for {label}. "
             f"Pass --ignore_ckpt_arch to use them as given.")
    return merged


def apply_checkpoint_arch(base_args, info: CheckpointInfo, *, label: str,
                          defaults: dict | None = None,
                          quiet: frozenset | set | None = None):
    """`merge_arch` against a checkpoint, honouring `--ignore_ckpt_arch`."""
    if getattr(base_args, "ignore_ckpt_arch", False):
        arch = arch_from_checkpoint(info)
        if arch:
            warn(f"--ignore_ckpt_arch: using the command-line architecture for "
                 f"{label}; {info.path} recorded {describe_arch(arch)}")
        return copy.copy(base_args)
    return merge_arch(base_args, arch_from_checkpoint(info), label=label,
                      defaults=defaults, path=info.path, quiet=quiet)


def resolved_model_cls(info: CheckpointInfo, args=None):
    """Return the model class recorded in the checkpoint.

    `--learned_model` is only a fallback for bare state_dicts; a conflicting
    value is ignored with a warning.
    """
    requested = getattr(args, "learned_model", None) if args is not None else None
    recorded = info.model.upper() if info.model else None

    if recorded is not None and requested and requested.upper() != recorded:
        warn(f"--learned_model {requested!r} ignored: {info.path} holds a "
             f"{recorded}.")

    name = recorded or (requested.upper() if requested else "FMW")
    if name not in MODEL_CLASSES:
        raise ValueError(
            f"{info.path} was trained with model={name!r}, which cannot be rebuilt "
            f"here; known: {sorted(MODEL_CLASSES)}.")
    module_name, class_name = MODEL_CLASSES[name].split(":")
    return getattr(importlib.import_module(module_name), class_name)


def check_nx(arch: dict, md, *, label: str) -> None:
    """Warn when the checkpoint's `nx` tag differs from the dataset grid.

    The checkpoint's `nx` is still used, since it determines the parameter names.
    """
    ckpt_nx = arch.get("nx")
    data_nx = getattr(md, "nx", None)
    if ckpt_nx is None or data_nx is None or ckpt_nx == data_nx:
        return
    warn(f"{label}: checkpoint was tagged nx={ckpt_nx} but the dataset grid is "
         f"{data_nx}. Either the checkpoint's resolution tags are wrong (see "
         f"daisi_sevir_128.pth in assimilation/checkpoints.py) or it was trained "
         f"on another grid. Building at nx={ckpt_nx}, which is what its parameter "
         f"names encode.")
