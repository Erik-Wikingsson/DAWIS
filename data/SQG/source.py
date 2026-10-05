"""SQG data source.

Wraps the Dataset classes in data/SQG/QG_dataset.py behind the `DataSource`
interface, and derives `scalefact`.
"""

from __future__ import annotations

import functools
import glob as _glob
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data.base import (
    ASSIM_TRAJECTORY,
    FORECAST_WINDOW,
    SINGLE_STATE,
    AssimCase,
    DataSource,
    DatasetMetadata,
    NormStats,
    VariableSpec,
)

# Scalar .nc attributes passed to the numerical model via AssimCase.model_params.
NC_SCALAR_ATTRS = (
    "r", "f", "U", "L", "H", "g", "theta0", "nsq", "tdiab",
    "dt", "diff_efold", "diff_order", "symmetric", "dealias",
)
NC_VARIABLES = ("x", "y", "z", "t")


def derive_unit_factor(nc) -> float:
    """scalefact = f * theta0 / g, from an open netCDF4 Dataset.

    Each attribute is cast to Python float first, for exact reproducibility.
    """
    return float(nc.f) * float(nc.theta0) / float(nc.g)


class SQGDataSource(DataSource):
    def __init__(self, config: Mapping[str, Any], root: Path, variant: str):
        super().__init__(config, root, variant)
        self._splits = dict(config.get("splits") or {})

    def _split_cfg(self, split: str) -> dict:
        if split not in self._splits:
            raise KeyError(
                f"SQG/{self.variant} has no split {split!r}; "
                f"available: {sorted(self._splits)}"
            )
        return dict(self._splits[split] or {})

    def split_dir(self, split: str) -> Path:
        return self.root / self._split_cfg(split)["dir"]

    def list_trajectories(self, split: str, pattern: str | None = None) -> list[Path]:
        cfg = self._split_cfg(split)
        pattern = pattern or cfg.get("glob", "*.npy")
        found = sorted(self.split_dir(split).glob(pattern))
        if not found:
            raise FileNotFoundError(
                f"SQG/{self.variant}: no files matching {pattern!r} in "
                f"{self.split_dir(split)}"
            )
        return found

    def trajectory_length(self, split: str) -> int:
        """Declared length, cross-checked against the data the first time."""
        cfg = self._split_cfg(split)
        declared = cfg.get("trajectory_length")
        if declared is not None:
            return int(declared)
        first = self.list_trajectories(split)[0]
        return int(np.load(first, mmap_mode="r").shape[0])

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._splits)

    def is_available(self) -> tuple[bool, str]:
        problem = self._root_problem()
        if problem:
            return False, problem
        for split in self._splits:
            directory = self.split_dir(split)
            if not directory.is_dir():
                return False, f"split dir {directory} does not exist"
        return True, ""

    def _climatology_source(self) -> "SQGDataSource":
        """The source whose .nc supplies model parameters (may be another variant)."""
        clim = dict(self.config.get("climatology") or {})
        other = clim.get("variant")
        if other and other != self.variant:
            from data.registry import get_data_source

            return get_data_source("SQG", other, root=str(self.root))
        return self

    def climatology_nc_path(self) -> Path:
        src = self._climatology_source()
        clim = dict(src.config.get("climatology") or self.config.get("climatology") or {})
        split = clim.get("split", "train")
        index = int(clim.get("index", 0))
        npy = src.list_trajectories(split)[index]
        nc = npy.with_suffix(".nc")
        if not nc.is_file():
            raise FileNotFoundError(
                f"SQG/{self.variant}: climatology netCDF {nc} not found "
                f"(needed for scalefact and numerical-model parameters)"
            )
        return nc

    def model_params(self) -> dict:
        """Scalar attrs + coordinate variables from the climatology .nc."""
        from netCDF4 import Dataset as NetCDFDataset

        path = self.climatology_nc_path()
        nc = NetCDFDataset(str(path), "r")
        try:
            params: dict[str, Any] = {
                key: nc.getncattr(key) for key in NC_SCALAR_ATTRS if hasattr(nc, key)
            }
            for name in NC_VARIABLES:
                if name in nc.variables:
                    # Kept as MaskedArray for bit-exact localization.
                    params[name] = nc.variables[name][:]
            params["scalefact"] = derive_unit_factor(nc)
            params["nc_path"] = str(path)
            return params
        finally:
            nc.close()

    def _norm_stats(self) -> NormStats:
        cfg = dict(self.config.get("normalization") or {})
        legacy = dict(cfg.get("legacy_scalar") or {})
        legacy_mean = float(legacy.get("mean", 0.0))
        legacy_std = float(legacy.get("std", 1.0))

        num_channels = 2
        per_channel = cfg.get("per_channel")
        source = "config:legacy_scalar"
        state_mean = torch.full((num_channels,), legacy_mean, dtype=torch.float32)
        state_std = torch.full((num_channels,), legacy_std, dtype=torch.float32)
        diff_mean = diff_std = None

        if per_channel and self._root_error is not None:
            source = "config:legacy_scalar (data root not set)"
        elif per_channel:
            stats_dir = self.root / dict(per_channel)["from_dir"]
            try:
                state_mean = self._load_stat(stats_dir / "data_mean.pt")
                state_std = self._load_stat(stats_dir / "data_std.pt")
                diff_mean = self._load_stat(stats_dir / "diff_mean.pt")
                diff_std = self._load_stat(stats_dir / "diff_std.pt")
                source = f"pt:{stats_dir}"
            except FileNotFoundError:
                # Keep shapes valid; checkpoints overwrite the values.
                source = f"config:legacy_scalar (missing .pt stats in {stats_dir})"

        return NormStats(
            state_mean=state_mean,
            state_std=state_std,
            legacy_scalar_mean=legacy_mean,
            legacy_scalar_std=legacy_std,
            diff_mean=diff_mean,
            diff_std=diff_std,
            source=source,
        )

    @staticmethod
    def _load_stat(path: Path) -> torch.Tensor:
        if not path.is_file():
            raise FileNotFoundError(path)
        return torch.load(path, weights_only=True).flatten().to(torch.float32)

    def has_per_channel_stats(self) -> bool:
        cfg = dict(self.config.get("normalization") or {})
        per_channel = cfg.get("per_channel")
        if not per_channel or self._root_error is not None:
            return False
        stats_dir = self.root / dict(per_channel)["from_dir"]
        return (stats_dir / "data_mean.pt").is_file()

    def require_per_channel_stats(self) -> None:
        """Raise unless per-channel .pt stats are available (none exist for `hires`)."""
        if not self.has_per_channel_stats():
            raise FileNotFoundError(
                f"SQG/{self.variant} has no per-channel standardization stats. "
                f"Generate them with `python data/compute_data_stats.py "
                f"--dataset SQG --variant {self.variant}`, or use the "
                f"legacy scalar profile."
            )

    @functools.cached_property
    def metadata(self) -> DatasetMetadata:
        cfg = self.config
        variables = tuple(
            VariableSpec(
                name=str(v["name"]),
                units=str(v.get("units", "")),
                plot_range=(
                    tuple(float(x) for x in v["plot_range"])
                    if v.get("plot_range") is not None
                    else None
                ),
                cmap=str(v.get("cmap", "jet")),
            )
            for v in cfg["variables"]
        )
        return DatasetMetadata(
            name="SQG",
            variant=self.variant,
            grid=tuple(int(v) for v in cfg["grid"]),
            num_channels=len(variables),
            variables=variables,
            norm=self._norm_stats(),
            unit_factor=self._unit_factor(),
            unit_name=str(cfg.get("unit_name", "K")),
            cadence_hours=float(cfg["cadence_hours"]),
            periodic=tuple(bool(v) for v in cfg.get("periodic", (True, True))),
            max_window=int(cfg.get("max_window", 100)),
            splits=self.splits,
            supports=(SINGLE_STATE, FORECAST_WINDOW, ASSIM_TRAJECTORY),
            raw_dtype=str(cfg.get("raw_dtype", "float64")),
            dataloader_kwargs=dict(cfg.get("dataloader_kwargs") or {}),
        )

    def _unit_factor(self) -> float:
        """Derive from the climatology .nc, asserting against the config value."""
        physical = dict(self.config.get("physical") or {})
        declared = physical.get("value")
        declared = None if declared is None else float(declared)
        tol = float(physical.get("tol", 0.0))

        if physical.get("derive") != "netcdf_attrs":
            if declared is None:
                raise ValueError(
                    f"SQG/{self.variant}: physical.value required when "
                    f"physical.derive is not 'netcdf_attrs'"
                )
            return declared

        try:
            from netCDF4 import Dataset as NetCDFDataset

            nc = NetCDFDataset(str(self.climatology_nc_path()), "r")
            try:
                derived = derive_unit_factor(nc)
            finally:
                nc.close()
        except (FileNotFoundError, OSError, ImportError):
            if declared is None:
                raise
            return declared

        if declared is not None and abs(derived - declared) > tol:
            raise ValueError(
                f"SQG/{self.variant}: scalefact derived from "
                f"{self.climatology_nc_path()} is {derived!r} but config "
                f"declares {declared!r} (tol={tol}). One of them is wrong; "
                f"refusing to guess."
            )
        return derived

    def single_state(self, split: str, *, per_channel: bool = False,
                     mmap: bool = False, **kwargs):
        """i.i.d. snapshots, normalized (legacy scalar profile by default)."""
        from data.SQG.QG_dataset import SQGStateDataset

        self.metadata.require(SINGLE_STATE)
        if per_channel:
            self.require_per_channel_stats()
        mean, std = self.metadata.norm.view(3, per_channel=per_channel)
        return SQGStateDataset(
            files=self.list_trajectories(split),
            mean=mean,
            std=std,
            mmap=mmap,
        )

    def forecast_window(self, split: str, *, init_states: int, pred_length: int,
                        step_hours: float | None = None, subset: bool = False,
                        standardize: bool = True, per_channel: bool = True,
                        subsample_step: int | None = None,
                        random_subsample: bool | None = None, **kwargs):
        """Windows, normalized per-channel by default.

        `random_subsample` (jittered window starts) defaults to True on the
        train split only.
        """
        from data.SQG.QG_dataset import SQGForecastDataset

        md = self.metadata
        md.require(FORECAST_WINDOW)
        if subsample_step is None:
            subsample_step = (
                1 if step_hours is None else md.subsample_step(step_hours)
            )
        md.check_window(
            init_states=init_states,
            pred_length=pred_length,
            subsample_step=subsample_step,
        )
        if standardize and per_channel:
            self.require_per_channel_stats()
        mean = std = None
        if standardize:
            mean, std = md.norm.view(4, per_channel=per_channel)
        return SQGForecastDataset(
            files=self.list_trajectories(split),
            init_states=init_states,
            pred_length=pred_length,
            subsample_step=subsample_step,
            max_length=md.max_window,
            trajectory_length=self.trajectory_length(split),
            random_subsample=((split == "train") if random_subsample is None
                              else random_subsample),
            mean=mean,
            std=std,
            subset_ds=subset,
        )

    def assim_trajectory(self, *, index: int = 0, split: str = "test",
                         path: str | os.PathLike | None = None) -> AssimCase:
        """One trajectory in raw units, plus numerical-model parameters."""
        from data.SQG.QG_dataset import SQGAssimDataset

        md = self.metadata
        md.require(ASSIM_TRAJECTORY)

        if path is not None:
            resolved = Path(str(path))
            if resolved.suffix == ".npy":
                resolved = resolved.with_suffix("")
            if resolved.is_dir():
                candidates = sorted(resolved.glob("*.npy"))
                if not candidates:
                    raise FileNotFoundError(f"no .npy files in {resolved}")
                resolved = candidates[index].with_suffix("")
            source_path = resolved.with_suffix(".npy")
        else:
            source_path = self.list_trajectories(split)[index]

        states = SQGAssimDataset(source_path)

        try:
            params = self.model_params()
        except FileNotFoundError:
            params = None

        domain = None
        dt = None
        if params is not None:
            if "L" in params:
                domain = (float(params["L"]), float(params["L"]))
            if "dt" in params:
                dt = float(params["dt"])

        geometry = None
        if params is not None:
            import numpy as _np

            # Vertical localization length (Rossby radius) between the two levels.
            vloc = None
            if md.num_channels > 1 and {"nsq", "H", "f"} <= set(params):
                # NOTE: kept float32 (no float()) for bit-exact LETKF results.
                vloc = _np.sqrt(params["nsq"]) * params["H"] / params["f"]
            geometry = {
                "x": params.get("x"),
                "y": params.get("y"),
                "Lx": float(params["L"]) if "L" in params else None,
                "Ly": float(params["L"]) if "L" in params else None,
                "rossby_radius_m": vloc,
                "periodic": tuple(md.periodic),
            }

        return AssimCase(
            states=states,
            metadata=md,
            source_path=source_path,
            model_params=params,
            grid_geometry=geometry,
            domain_size_m=domain,
            dt=dt,
            assim_interval_hours=md.cadence_hours,
        )
