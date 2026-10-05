"""Which trained checkpoint a method loads, per dataset.

Two kinds of model are involved and they are not interchangeable:

* an **unconditional prior** (DAISI), trained by `unconditional_generation/`
  and saved as a bare `state_dict`;
* an **FMW window model** (DAWIS / SDA), trained by
  `forecasting/trainer.py --model FMW` and saved as a Lightning checkpoint.

Resolution order, first hit wins: `--model_path`, the entry's environment
variable (or `.env`), the entry's path under `MODELS_ROOT`.
"""

from __future__ import annotations

import re

from data.paths import checkpoint

# The released checkpoints (huggingface.co/Erik-Wikingsson/dawis-checkpoints) are
# stored under these exact subpaths, so renaming one here breaks the download
# layout. Keep the shell tables (scripts/SQG/_models.sh, scripts/SEVIR/_common.sh)
# in step.

# dataset -> variant -> (env var, path under MODELS_ROOT).
# "default" applies to any variant without its own entry.
PRIORS: dict[str, dict[str, tuple[str, str]]] = {
    "SQG": {
        "default": ("ASSIM_MODEL_PATH", "SQG/models/daisi/daisi_64.pth"),
    },
    "SEVIR": {
        # 128x128, pairs with --variant lr_vil and --attn_resolutions 64.
        "default": ("SEVIR_PRIOR_PATH", "SQG/models/daisi/daisi_sevir_128.pth"),
    },
}


def _fmw_windows(env_prefix: str, subpaths: dict[int, str]) -> dict:
    return {w: (f"{env_prefix}_INIT_{w}", sub) for w, sub in subpaths.items()}


def _window_env(entry: dict, init_states: int) -> str:
    """The env var for `init_states` in a per-window entry, listed or not."""
    env = next(iter(entry.values()))[0]
    return re.sub(r"_INIT_\d+$", f"_INIT_{init_states}", env)


# dataset -> fm_loss -> either
#   (env var, path under MODELS_ROOT)            one checkpoint for every window, or
#   {init_states: (env var, path)}               one checkpoint per window.
# Only the released models are listed (eta01_channel, init_states 6); a window
# trained locally is found through its env var (e.g.
# ASSIM_FMW_PATH_SQG_ETA01_CHANNEL_INIT_3), any other model through --model_path.
# An FMW network is built for a fixed window T = init_states + 1 with
# hidden_dim = 32 * (init_states + 1), so it only loads under that init_states.
FMW_MODELS: dict[str, dict[str, object]] = {
    "SQG": {
        "eta01_channel": _fmw_windows("ASSIM_FMW_PATH_SQG_ETA01_CHANNEL", {
            6: "DAWIS/models/eta_channel_init_6-FMW-224-06_28_18-3280/last.ckpt",
        }),
    },
    "SEVIR": {
        # 128x128 VIL under the FlowDAS normalization (--sevir_norm flowdas).
        "eta01_channel": _fmw_windows("ASSIM_FMW_PATH_SEVIR_ETA01_CHANNEL", {
            6: "DAWIS/models_SEVIR/SEVIR_FLOWDAS_SPLIT_lr_vil_eta_channel"
               "_init_6_flowdas-FMW-224-09_12_05-1441/last.ckpt",
        }),
    },
}

DEFAULT_FM_LOSS = "eta01_channel"


class MissingCheckpoint(FileNotFoundError):
    """No checkpoint could be resolved for this dataset/method combination."""


def _explicit(args) -> str | None:
    path = getattr(args, "model_path", None)
    if path in (None, "", "None"):
        return None
    return str(path)


def resolve_prior(md, args) -> str:
    """Path to the unconditional prior for DAISI on this dataset."""
    explicit = _explicit(args)
    if explicit:
        return explicit

    table = PRIORS.get(md.name)
    if not table:
        raise MissingCheckpoint(
            f"No unconditional prior is registered for dataset {md.name!r}. "
            f"Train one with unconditional_generation/trainer.py (see "
            f"unconditional_generation/scripts/), then pass --model_path or "
            f"add an entry to PRIORS in assimilation/checkpoints.py."
        )
    env, subpath = table.get(md.variant, table["default"])
    return checkpoint(env, subpath)


def resolve_window_model(md, args) -> str:
    """Path to the FMW window model for DAWIS / SDA on this dataset."""
    explicit = _explicit(args)
    if explicit:
        return explicit

    fm_loss = getattr(args, "fm_loss", None) or DEFAULT_FM_LOSS
    train_hint = (f"Train one with forecasting/training_scripts/{md.name}/"
                  f"train_all_fmw.sh, then pass its last.ckpt via --model_path "
                  f"or add it to FMW_MODELS in assimilation/checkpoints.py.")
    table = FMW_MODELS.get(md.name, {})
    if fm_loss not in table:
        known = sorted(table)
        raise MissingCheckpoint(
            f"No FMW window model is registered for {md.name}/{fm_loss}."
            + (f" Registered for {md.name}: {known}." if known else
               f" Nothing is registered for {md.name} yet.")
            + f"\n{train_hint}"
        )

    entry = table[fm_loss]
    if isinstance(entry, dict):
        init_states = getattr(args, "init_states", None)
        if init_states is None:
            raise MissingCheckpoint(
                f"{md.name}/{fm_loss} has one FMW window model per "
                f"--init_states ({sorted(entry)}), but --init_states was not "
                f"given. Pass --init_states, or --model_path."
            )
        if init_states not in entry:
            env = _window_env(entry, init_states)
            override = checkpoint(env)
            if override:
                return override
            raise MissingCheckpoint(
                f"No FMW window model is registered for {md.name}/{fm_loss} at "
                f"--init_states {init_states}. Registered windows: "
                f"{sorted(entry)}. A window model only loads under the "
                f"--init_states it was trained with.\n{train_hint} "
                f"Or set {env} (environment or .env)."
            )
        entry = entry[init_states]

    env, subpath = entry
    return checkpoint(env, subpath)
