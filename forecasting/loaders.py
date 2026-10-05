"""DataLoader construction for forecasting."""

from __future__ import annotations

import torch

from data.registry import resolve_n_workers


def build_forecast_dataset(src, split, args, *, pred_length):
    md = src.metadata
    return src.forecast_window(
        split,
        init_states=args.init_states,
        pred_length=pred_length,
        step_hours=float(getattr(args, "step_length", md.cadence_hours)),
        subset=bool(getattr(args, "subset_ds", False)),
        standardize=True,
        per_channel=True,
    )


def build_loader(src, split, args, *, pred_length, shuffle,
                 pin_memory=None, persistent_workers=None):
    """Build a DataLoader over `split`.

    `pin_memory` / `persistent_workers` override the source defaults when given.
    """
    dataset = build_forecast_dataset(src, split, args, pred_length=pred_length)

    # Per-source DataLoader defaults; an explicit --n_workers wins.
    kwargs = dict(src.metadata.dataloader_kwargs)
    n_workers = resolve_n_workers(src, args)
    kwargs["num_workers"] = n_workers

    if pin_memory is not None:
        kwargs["pin_memory"] = pin_memory
    if persistent_workers is not None:
        kwargs["persistent_workers"] = persistent_workers
    # These options are only valid with worker processes.
    if n_workers == 0:
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)

    return torch.utils.data.DataLoader(
        dataset, args.batch_size, shuffle=shuffle, **kwargs)
