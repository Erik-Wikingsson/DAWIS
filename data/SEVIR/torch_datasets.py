"""Torch Datasets over SEVIR events, in the canonical `(T, C, H, W)` layout.

Splits are a date range (handled by the loader) plus an optional
`random_part` (a seeded train/val partition of events, as in FlowDAS).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from data.SEVIR.sevir_dataloader import SEVIRDataLoader


def _downsample_dict(cfg):
    """`{var: (t, h, w)}` downsampling factors from config `downsample`, or None."""
    factors = cfg.get("downsample")
    if not factors:
        return None
    return {var: tuple(int(f) for f in factors) for var in cfg["data_types"]}


def _partition_indices(n, part, *, val_ratio, seed, per_event=1):
    """Sorted loader item indices kept by a `random_part` split, or None for all.

    FlowDAS-style `random_split(.., [1 - val_ratio, val_ratio])`, but over
    events (`per_event` consecutive items each) so train/val stay disjoint for
    any window shape.
    """
    if part is None:
        return None
    if part not in ("train", "val"):
        raise ValueError(
            f"random_part must be 'train', 'val' or absent, got {part!r}"
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    per_event = int(per_event)
    if per_event < 1:
        raise ValueError(f"per_event must be >= 1, got {per_event}")
    n_events = int(n) // per_event
    if n_events == 0:
        return []
    pieces = random_split(
        range(n_events), [1.0 - val_ratio, val_ratio],
        generator=torch.Generator().manual_seed(int(seed)),
    )
    events = sorted(pieces[0 if part == "train" else 1].indices)
    return [e * per_event + j for e in events for j in range(per_event)]


def _assim_sample_indices(n_available, cfg):
    """Split positions addressed by `--data_index`, or None to index the split directly.

    A seeded truncated permutation, so increasing `size` keeps earlier indices.
    """
    if not cfg:
        return None
    size = int(cfg.get("size", 0) or 0)
    if size <= 0:
        return None
    n_available = int(n_available)
    if n_available <= 0:
        return []
    if size > n_available:
        raise ValueError(
            f"assim_sample.size is {size} but this split has only "
            f"{n_available} events; lower `size` in data/SEVIR/config.yaml "
            f"(or export SEVIR_ASSIM_SIZE)."
        )
    seed = int(cfg.get("seed", 0) or 0)
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.permutation(n_available)[:size]]


def _check_affine(post_affine):
    """Validate `(mean, std)`; None means the identity (0.0, 1.0)."""
    if post_affine is None:
        return (0.0, 1.0)
    mean, std = (float(v) for v in post_affine)
    if not std > 0:
        raise ValueError(f"post_affine std must be positive, got {std}")
    return (mean, std)


def _close_quietly(loader):
    """Release the HDF5 handles, ignoring errors."""
    try:
        loader.close()
    except Exception:
        pass


def _apply_affine(x, post_affine):
    """`(x - mean) / std`, skipped entirely when it is the identity."""
    mean, std = post_affine
    if mean == 0.0 and std == 1.0:
        return x
    return (x - mean) / std


def _loader(cfg, *, seq_len, stride, split_dates, shuffle=False):
    start, end = split_dates
    return SEVIRDataLoader(
        data_types=list(cfg["data_types"]),
        seq_len=seq_len,
        raw_seq_len=int(cfg["raw_seq_len"]),
        sample_mode="sequent",
        stride=stride,
        batch_size=1,
        layout="NTCHW",          # -> (1, T, C, H, W); squeeze(0) is canonical
        sevir_catalog=cfg["catalog"],
        sevir_data_dir=cfg["data_dir"],
        start_date=start,
        end_date=end,
        shuffle=shuffle,
        output_type=np.float32,
        preprocess=True,          # applies PREPROCESS_SCALE/OFFSET
        rescale_method=cfg.get("rescale_method", "01"),
        downsample_dict=_downsample_dict(cfg),
        verbose=False,
    )


class _SevirBase(Dataset):
    #: Number of items kept by `subset=True` (debugging).
    SUBSET_LEN = 8

    def __init__(self, cfg, *, seq_len, stride, split_dates, var, subset=False,
                 post_affine=None, random_part=None):
        self.cfg = cfg
        self.var = var
        self.subset = subset
        self.post_affine = _check_affine(post_affine)
        self.loader = _loader(cfg, seq_len=seq_len, stride=stride,
                              split_dates=split_dates)
        #: Loader item indices this split keeps, or None for all.
        self.indices = _partition_indices(
            len(self.loader), random_part,
            val_ratio=float(cfg.get("val_ratio", 0.1)),
            seed=int(cfg.get("split_seed", 0)),
            per_event=self.loader.num_seq_per_event,
        )

    def __len__(self):
        n = len(self.loader) if self.indices is None else len(self.indices)
        return min(n, self.SUBSET_LEN) if self.subset else n

    def _item(self, idx):
        """Map this dataset's `idx` to the loader's item index."""
        return idx if self.indices is None else self.indices[idx]

    def _window(self, idx) -> torch.Tensor:
        """(T, C, H, W) float32, normalized."""
        sample = self.loader._idx_sample(self._item(idx))
        x = sample[self.var]
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x))
        x = x.squeeze(0).to(torch.float32)      # drop the batch axis
        return _apply_affine(x, self.post_affine)

    def close(self):
        _close_quietly(self.loader)


class SEVIRStateDataset(_SevirBase):
    """i.i.d. frames for unconditional generation. Items: (C, H, W)."""

    def __init__(self, cfg, *, split_dates, var, subset=False, post_affine=None,
                 random_part=None):
        super().__init__(cfg, seq_len=1, stride=1, split_dates=split_dates,
                         var=var, subset=subset, post_affine=post_affine,
                         random_part=random_part)

    def __getitem__(self, idx):
        return self._window(idx)[0]


class SEVIRWindowDataset(_SevirBase):
    """Forecast windows. Items: ((T_in, C, H, W), (T_out, C, H, W))."""

    def __init__(self, cfg, *, split_dates, var, init_states, pred_length,
                 subsample_step=1, subset=False, post_affine=None,
                 random_part=None):
        self.init_states = init_states
        self.pred_length = pred_length
        self.subsample_step = subsample_step
        seq_len = 1 + (init_states - 1 + pred_length) * subsample_step \
            if init_states else pred_length * subsample_step
        super().__init__(cfg, seq_len=seq_len, stride=seq_len,
                         split_dates=split_dates, var=var, subset=subset,
                         post_affine=post_affine, random_part=random_part)

    def __getitem__(self, idx):
        x = self._window(idx)[::self.subsample_step]
        return x[:self.init_states], x[self.init_states:]


class SEVIRTrajectory:
    """One event as a `(T, C, H, W)` array indexed by time, normalized by `post_affine`."""

    def __init__(self, cfg, *, split_dates, var, index, post_affine=None,
                 random_part=None, sample=True):
        loader = _loader(cfg, seq_len=int(cfg["raw_seq_len"]),
                         stride=int(cfg["raw_seq_len"]),
                         split_dates=split_dates)
        # Apply the same partition as the Datasets so `index` counts within the split.
        indices = _partition_indices(
            len(loader), random_part,
            val_ratio=float(cfg.get("val_ratio", 0.1)),
            seed=int(cfg.get("split_seed", 0)),
            per_event=loader.num_seq_per_event,
        )
        pool = list(range(len(loader))) if indices is None else list(indices)
        #: Position (in time order) within the split that `index` resolved to.
        self.split_position = None
        #: The `assim_sample` draw for this split, or None (`sample=False` indexes the split).
        self.sample = (_assim_sample_indices(len(pool), cfg.get("assim_sample"))
                       if sample else None)
        index = int(index)
        limit = len(pool) if self.sample is None else len(self.sample)
        if not 0 <= index < limit:
            what = "events in this split" if self.sample is None else (
                f"sampled events (assim_sample.size in data/SEVIR/config.yaml; "
                f"this split has {len(pool)} events in total)")
            raise IndexError(
                f"--data_index {index} is out of range: there are {limit} {what}."
            )
        self.split_position = index if self.sample is None else self.sample[index]
        item = pool[self.split_position]
        sample = loader._idx_sample(item)
        x = sample[var]
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x))
        x = _apply_affine(x.squeeze(0).to(torch.float32),
                          _check_affine(post_affine))
        self._data = x.numpy()
        _close_quietly(loader)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
