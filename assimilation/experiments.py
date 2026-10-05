"""Named assimilation experiment presets.

Presets name a dataset variant rather than a path; nothing here touches the
filesystem at import time.
"""

import os

# Kept for backward compatibility; data is selected by --dataset/--variant/--data_index.
DATA_PATH_64 = None
CLIM_PATH_64 = None


def trajectories(dataset="SQG", variant="base", split="test", pattern=None):
    """Trajectory files for a split, resolved lazily through the registry."""
    from data.registry import get_data_source

    return get_data_source(dataset, variant).list_trajectories(split, pattern)


def __getattr__(name):
    """Lazy module attributes (PEP 562), e.g. `nc_files_64`."""
    if name == "nc_files_64":
        return [str(p.with_suffix("")) for p in trajectories(
            "SQG", "base", "test", "sqg_N64_3hrly_steps_110_*.npy")]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


EXPERIMENTS = {
    'noisy': {
        'variant': 'base',
        'nx': 64,
        'obs_fn': 'linear',
        'hrly': 3,
        'obs_prob': 0.25,
        'obs_sigma': 5.0,
        'init_std': 1000,
        'fixed_obs': True,
        'hcovlocal_scale': 2500.0*1000,  # 2500.0*1000 for 3hrly, 1500*1000 for 12hrly
        'covinflate1': 0.5,  # 0.5 for 3hrly, 0.4 for 12hrly
    },
    'noisy-5': {
        'variant': 'base',
        'nx': 64,
        'obs_fn': 'linear',
        'hrly': 3,
        'obs_prob': 0.05,
        'obs_sigma': 5.0,
        'init_std': 1000,
        'fixed_obs': True,
        'hcovlocal_scale': 2500.0*1000,  # 2500.0*1000 for 3hrly, 1500*1000 for 12hrly
        'covinflate1': 0.5,  # 0.5 for 3hrly, 0.4 for 12hrly
    },
    'sparse': {
        'variant': 'base',
        'nx': 64,
        'obs_fn': 'linear',
        'hrly': 3,
        'obs_prob': 0.05,
        'obs_sigma': 1.0,
        'init_std': 1000,
        'fixed_obs': True,
        'hcovlocal_scale': 1500.0*1000,  # 1500*1000 for both
        'covinflate1': 0.5,  # 0.5 for 3hrly, 0.2 for 12hrly
    },
    'sparse-0': {
        'variant': 'base',
        'nx': 64,
        'obs_fn': 'linear',
        'hrly': 3,
        'obs_prob': 0.0,
        'obs_sigma': 1.0,
        'init_std': 1000,
        'fixed_obs': True,
        'hcovlocal_scale': 1500.0*1000,  # 1500*1000 for both
        'covinflate1': 0.5,  # 0.5 for 3hrly, 0.2 for 12hrly
    },
    'saturating': {
        'variant': 'base',
        'nx': 64,
        'obs_fn': 'arctan',
        'hrly': 3,
        'obs_prob': 0.25,
        'obs_sigma': 0.01,
        'init_std': 1000,
        'fixed_obs': True,
        'hcovlocal_scale': 4500.0*1000,  # 4500.0*1000 for 3hrly, 4000*1000 for 12hrly
        'covinflate1': 0.9,  # 0.9 for 3hrly, 0.8 for 12hrly
    },
    'saturating-5': {
        'variant': 'base',
        'nx': 64,
        'obs_fn': 'arctan',
        'hrly': 3,
        'obs_prob': 0.05,
        'obs_sigma': 0.01,
        'init_std': 1000,
        'fixed_obs': True,
        'hcovlocal_scale': 4500.0*1000,  # 4500.0*1000 for 3hrly, 4000*1000 for 12hrly
        'covinflate1': 0.9,  # 0.9 for 3hrly, 0.8 for 12hrly
    },
    'multimodal': {
        'variant': 'base',
        'nx': 64,
        'obs_fn': 'square_scaled',
        'hrly': 3,
        'obs_prob': 0.25,
        'obs_sigma': 1.0,
        'init_std': 1000,
        'fixed_obs': True,
        'hcovlocal_scale': 2500.0*1000,  # 2500.0*1000 for 3hrly, 1500*1000 for 12hrly
        'covinflate1': 0.5,  # 0.5 for 3hrly, 0.4 for 12hrly
    },
    'multimodal-5': {
        'variant': 'base',
        'nx': 64,
        'obs_fn': 'square_scaled',
        'hrly': 3,
        'obs_prob': 0.05,
        'obs_sigma': 1.0,
        'init_std': 1000,
        'fixed_obs': True,
        'hcovlocal_scale': 2500.0*1000,  # 2500.0*1000 for 3hrly, 1500*1000 for 12hrly
        'covinflate1': 0.5,  # 0.5 for 3hrly, 0.4 for 12hrly
    },
    'noisy-12h': {
        'variant': 'base',
        'nx': 64,
        'obs_fn': 'linear',
        'hrly': 3,
        'obs_prob': 0.25,
        'obs_sigma': 5.0,
        'init_std': 1000,
        'fixed_obs': True,
        'hcovlocal_scale': 1500.0*1000,  # 2500.0*1000 for 3hrly, 1500*1000 for 12hrly
        'covinflate1': 0.4,  # 0.5 for 3hrly, 0.4 for 12hrly
        'assim_interval': 4,
    },
    # SEVIR: the DAISI paper's observation network, started from the ground
    # truth. LOCKED: contradicting command-line values are an error, and
    # --dataset SEVIR applies it without --experiment. `obs_sigma_counts` is in
    # raw VIL counts and is converted to the run's --assim_norm units.
    'sevir': {
        'dataset': 'SEVIR',
        'obs_fn': 'linear',
        'obs_prob': 0.1,
        'obs_sigma_counts': 0.255,
        'fixed_obs': True,
        'init_std': 0.0,
        'init_state': 'GT',
    },
}

# Presets whose values may not be overridden from the command line.
LOCKED_EXPERIMENTS = {'sevir'}

# Default preset per dataset when no --experiment is passed.
DATASET_EXPERIMENT = {'SEVIR': 'sevir'}

# Preset keys that are not argparse attributes, mapped to the one they set.
_DERIVED_KEYS = {'obs_sigma_counts': 'obs_sigma'}

# Propagators that add no ensemble spread of their own.
_DETERMINISTIC_PROPAGATORS = {'unet', 'numerical', 'numerical_gpu'}


def _resolve_preset(args, config):
    """The preset with its derived keys converted to real arguments."""
    resolved = {}
    for key, value in config.items():
        if key == 'obs_sigma_counts':
            from data.registry import data_source_from_args

            # counts = a * v + b; a sigma scales by a only.
            md = data_source_from_args(args).metadata
            counts_per_unit, _ = md.output_affine('physical')
            value = value / counts_per_unit
        resolved[_DERIVED_KEYS.get(key, key)] = value
    return resolved


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return False
    return a == b


def _check_zero_spread(args):
    """Refuse method/propagator pairs that init_std 0 turns into NaN.

    With zero initial spread and a deterministic propagator, LETKF
    inflation and EnSF divide by a zero ensemble spread.
    """
    if float(args.init_std) != 0.0:
        return
    fm = getattr(args, 'forward_model', None)
    if args.method in ('LETKF', 'EnSF') and fm in _DETERMINISTIC_PROPAGATORS:
        raise ValueError(
            f"{args.method} with init_std 0 and the deterministic "
            f"--forward_model {fm}: every member stays identical, and the "
            "first analysis divides by the zero spread. Use a stochastic "
            "propagator (--forward_model flowdas or fmw).")


def apply_experiment(args, defaults=None):
    """Apply the named preset to `args` in place and return it.

    Unlocked presets overwrite `args`. A LOCKED preset raises on command-line
    values that differ from it (values equal to the parser `defaults` count
    as not passed).
    """
    if args.experiment is None:
        args.experiment = DATASET_EXPERIMENT.get(getattr(args, 'dataset', None))
    if args.experiment is None:
        return args
    if args.experiment not in EXPERIMENTS:
        raise ValueError(
            f"Experiment '{args.experiment}' not recognized. Available "
            f"experiments: {list(EXPERIMENTS)}")

    config = EXPERIMENTS[args.experiment]
    locked = args.experiment in LOCKED_EXPERIMENTS
    own = DATASET_EXPERIMENT.get(getattr(args, 'dataset', None))
    if own is not None and args.experiment != own:
        raise ValueError(
            f"--dataset {args.dataset} has exactly one experiment, '{own}'; "
            f"got --experiment {args.experiment}.")
    dataset = config.get('dataset')
    if dataset is not None and getattr(args, 'dataset', None) != dataset:
        raise ValueError(
            f"Experiment '{args.experiment}' is a {dataset} experiment, but "
            f"--dataset is {getattr(args, 'dataset', None)}.")
    if not locked:
        for key, value in config.items():
            setattr(args, key, value)
    else:
        resolved = _resolve_preset(args, config)
        if defaults is None:
            from assimilation.parser import build_parser
            defaults = build_parser().parse_args([])
        conflicts = []
        for key, value in resolved.items():
            given = getattr(args, key, None)
            if not _same(given, value) and not _same(given, getattr(defaults, key, None)):
                conflicts.append(f"--{key} {given} (the experiment is {value})")
            setattr(args, key, value)
        if conflicts:
            raise ValueError(
                f"Experiment '{args.experiment}' fixes these settings, and the "
                "command line contradicts them:\n  " + "\n  ".join(conflicts)
                + "\nEdit EXPERIMENTS in assimilation/experiments.py if the "
                "experiment itself is meant to change.")
        _check_zero_spread(args)
    print(f"Using experiment configuration for '{args.experiment}': "
          f"{resolved if locked else config}", flush=True)
    return args