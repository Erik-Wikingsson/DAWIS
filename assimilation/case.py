"""Dataset resolution for assimilation and smoothing.

Maps --data_path/--data_index to a trajectory and picks the normalization
profile. Physical model parameters travel in `AssimCase.model_params`.
"""

from __future__ import annotations

import numpy as np

from data.registry import data_source_from_args


def normalization_profile(args) -> bool:
    """Whether to use per-channel stats.

    DAWIS uses per-channel stats; the other methods (incl. the DAISI
    checkpoints) use the scalar data_std=2660.
    """
    return getattr(args, "method", None) == "DAWIS"


def load_assim_case(args):
    """Resolve (src, md, case) for an assimilation or smoothing run.

    SQG honours an explicit `--data_path` (directory, `.npy` or prefix), else
    `--data_index` in the split. SEVIR accepts `--data_index` only.
    """
    src = data_source_from_args(args)
    md = src.metadata

    split = getattr(args, "split", None) or ("test" if "test" in md.splits else md.splits[-1])
    path = getattr(args, "data_path", None)

    case = src.assim_trajectory(
        index=int(getattr(args, "data_index", 0) or 0),
        split=split,
        path=path,
    )
    print(f"Data source: {md.name}/{md.variant} split={split} "
          f"trajectory={case.source_path}", flush=True)

    # Models and methods size their grids from args.nx, so take it from the data.
    nx_arg = getattr(args, "nx", None)
    if nx_arg is not None and nx_arg != md.nx:
        raise ValueError(
            f"--nx {nx_arg} contradicts {md.name}/{md.variant} grid {md.grid}; "
            f"select the data with --variant instead."
        )
    args.nx = md.nx
    return src, md, case


def assim_scale(md, args):
    """`scale` and `init_sigma`, in the units the run assimilates in.

    `scale` divides the assimilation state to reach network units:
    `std * unit_factor` in physical space (SQG), 1 in model space (SEVIR).
    """
    per_channel = normalization_profile(args)
    std = md.assim_scale(per_channel=per_channel)
    if per_channel:
        scale = std.view(1, -1, 1, 1)
    else:
        scale = float(std[0])
    # unit_factor is 1.0 in normalized space.
    init_sigma = float(args.init_std) * md.unit_factor
    return scale, init_sigma


def initial_conditioning(case, start_time, window, n_ens, unit_factor):
    """Conditioning states for the first assimilation step.

    Shape `(window, n_ens, C, ny, nx)`, in physical units.
    """
    states = case.states[start_time - window:start_time]
    return (
        np.expand_dims(states, axis=1)
        .repeat(n_ens, axis=1)
        .astype(np.float32, copy=False)
        * unit_factor
    )
