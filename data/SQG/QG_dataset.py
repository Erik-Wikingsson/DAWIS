"""Torch Datasets over SQG trajectory files.

Paths and normalization statistics are injected by `SQGDataSource`
(data/SQG/source.py).
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _as_file_list(files):
    """Accept a list of paths, or a single directory/glob for convenience."""
    if isinstance(files, (str, os.PathLike)):
        path = str(files)
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "*.npy")))
        else:
            found = sorted(glob.glob(path))
        if not found:
            raise ValueError(f"No .npy files found for {path!r}")
        return found
    files = [str(f) for f in files]
    if not files:
        raise ValueError("Empty trajectory file list")
    return files


class SQGStateDataset(Dataset):
    """i.i.d. snapshots for unconditional generation.

    Items are `(C, H, W)` float32, **normalized**.

    Args:
        files: trajectory .npy paths (already sorted by the data source).
        mean, std: broadcastable normalization tensors, shape (C, 1, 1).
        mmap: load lazily via `np.load(mmap_mode='r')` instead of into RAM.
    """

    def __init__(self, files, mean, std, mmap=False):
        self.files = _as_file_list(files)
        self.mean = mean
        self.std = std
        self.mmap = mmap

        if mmap:
            self._arrays = [np.load(f, mmap_mode="r") for f in self.files]
            self._lengths = [a.shape[0] for a in self._arrays]
            self._offsets = np.cumsum([0] + self._lengths)
            self.data = None
        else:
            data_list = [np.load(f) for f in self.files]
            self.data = torch.tensor(
                np.concatenate(data_list, axis=0), dtype=torch.float32
            )

    def __len__(self):
        if self.mmap:
            return int(self._offsets[-1])
        return len(self.data)

    def __getitem__(self, idx):
        if self.mmap:
            file_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
            local = idx - int(self._offsets[file_idx])
            x = torch.tensor(
                np.asarray(self._arrays[file_idx][local]), dtype=torch.float32
            )
        else:
            x = self.data[idx]
        # Normalize the data
        x = (x - self.mean) / self.std
        return x


class SQGAssimDataset(Dataset):
    """One trajectory, indexed by time, in raw (unnormalized) units."""

    def __init__(self, data_path):
        path = str(data_path)
        if path.endswith(".npy"):
            path = path[:-4]
        self.source_path = path + ".npy"
        self.data = np.load(self.source_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x


class SQGForecastDataset(Dataset):
    """Non-overlapping windows for forecasting training and evaluation.

    Items are `(init_states, C, H, W)`, `(pred_length, C, H, W)` float32,
    **normalized**.

    `max_length` sets `trajectories_per_file = max_length // sample_length`
    (the epoch size); `trajectory_length` bounds the sample length.
    """

    def __init__(self,
                 files,
                 init_states=2,
                 pred_length=1,
                 subsample_step=1,
                 max_length=100,
                 trajectory_length=None,
                 random_subsample=False,
                 mean=None,
                 std=None,
                 subset_ds=False,
                 ):
        self.pred_length = pred_length
        self.init_states = init_states
        self.subsample_step = subsample_step
        self.max_length = max_length
        self.trajectory_length = (
            max_length if trajectory_length is None else trajectory_length
        )
        self.random_subsample = random_subsample

        self.standardize = mean is not None and std is not None
        self.data_mean = mean
        self.data_std = std

        self.subset_ds = subset_ds
        self.trajectory_files = _as_file_list(files)

        if init_states == 0:
            self.sample_length = pred_length * subsample_step
        else:
            self.sample_length = 1 + \
                (init_states-1 + pred_length) * subsample_step

        assert self.sample_length <= self.trajectory_length, (
            f"Requesting too long time series samples. Requested length "
            f"({self.sample_length}) exceeds trajectory length "
            f"({self.trajectory_length})."
        )

        self.trajectories_per_file = self.max_length // self.sample_length

        if subset_ds:
            # Limit to 1 file
            self.trajectory_files = self.trajectory_files[:1]
            self.trajectories_per_file = min(
                4, self.trajectories_per_file)  # Only 4 samples per file

    def __len__(self):
        return len(self.trajectory_files) * self.trajectories_per_file

    def __getitem__(self, idx):
        # We want to find non-overlapping trajectories
        file_idx = idx // self.trajectories_per_file
        start_idx = idx - file_idx * self.trajectories_per_file
        # Check if we can move forward
        # If we are on the last possible trajectory, start_idx = 0
        if self.random_subsample:
            # Check if start_idx + sample_length * subsample_step exceeds max_length, only include valid start indices
            end_idx = start_idx + self.sample_length
            overflow = end_idx - self.max_length
            max_start_idx = start_idx - overflow if overflow < 0 else start_idx
            max_start_idx = min(max_start_idx, start_idx + self.sample_length)
            if max_start_idx > start_idx:
                start_idx = torch.randint(
                    start_idx, max_start_idx, ()).item()
        sample_path = self.trajectory_files[file_idx]
        try:
            full_sample = torch.tensor(
                np.load(sample_path), dtype=torch.float32
            )
        except ValueError:
            raise ValueError(f"Failed to load {sample_path}")

        sample = full_sample[start_idx: start_idx +
                             self.sample_length: self.subsample_step]

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std

        init_states = sample[:self.init_states]
        target_states = sample[self.init_states:]

        # B, T, C, H, W
        return init_states, target_states
