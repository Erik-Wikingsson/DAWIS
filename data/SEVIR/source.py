"""SEVIR data source."""

from __future__ import annotations

import datetime
import functools
import warnings
from pathlib import Path
from typing import Any, Mapping

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


def _as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day)
    return datetime.datetime(*[int(v) for v in value])


#: Model-space standardization as (mean, std) in raw VIL counts: z = (raw - mean) / std.
#:   01        vil/255 (default)
#:   flowdas   (vil/255 - 0.5) / 0.1, the FlowDAS/DAISI convention (uint8 -> [-5, 5])
#:   standard  published SEVIR mean/std; overridable via `norm_mean`/`norm_std`
NORM_MODES = {
    "01": (0.0, 255.0),
    "flowdas": (127.5, 25.5),
    "standard": (33.44, 47.54),
}


def _norm_mode_constants(name: str, cfg: Mapping[str, Any]) -> tuple[float, float]:
    """`NORM_MODES[name]` in raw VIL counts, with the config override for `standard`."""
    mean, std = NORM_MODES[name]
    if name == "standard":
        mean = float(cfg.get("norm_mean", mean))
        std = float(cfg.get("norm_std", std))
    if not std > 0:
        raise ValueError(f"norm_std for mode {name!r} must be positive, got {std}")
    return float(mean), float(std)


class SEVIRDataSource(DataSource):
    def __init__(self, config: Mapping[str, Any], root: Path, variant: str):
        super().__init__(config, root, variant)
        self._splits = dict(config.get("splits") or {})

    @property
    def catalog(self) -> Path:
        self.root  # raises if SEVIR_ROOT is unset
        return Path(self.config["catalog"])

    @property
    def data_dir(self) -> Path:
        self.root  # raises if SEVIR_ROOT is unset
        return Path(self.config["data_dir"])

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._splits)

    def split_dir(self, split: str) -> Path:
        """SEVIR splits by date, not by directory, so every split shares one."""
        if split not in self._splits:
            raise KeyError(
                f"SEVIR/{self.variant} has no split {split!r}; "
                f"available: {sorted(self._splits)}"
            )
        return self.data_dir

    def split_dates(self, split: str):
        cfg = dict(self._splits[split] or {})
        return _as_date(cfg.get("start")), _as_date(cfg.get("end"))

    def split_part(self, split: str) -> str | None:
        """Part of the seeded random partition (`train`/`val`) `split` takes, or None.

        Train and val share a date range and differ only in this; see
        `torch_datasets._partition_indices`.
        """
        cfg = dict(self._splits[split] or {})
        part = cfg.get("random_part")
        return None if part is None else str(part)

    def list_trajectories(self, split: str, pattern: str | None = None) -> list[Path]:
        """Every HDF5 shard; SEVIR splits by date inside the files, not by file."""
        self.split_dir(split)          # validates the split name
        found = sorted(self.data_dir.rglob(pattern or "*.h5"))
        if not found:
            raise FileNotFoundError(
                f"SEVIR/{self.variant}: no HDF5 files under {self.data_dir}"
            )
        return found

    def trajectory_length(self, split: str) -> int:
        """Frames per event (the same for every split)."""
        self.split_dir(split)          # validates the split name
        return int(self.config["raw_seq_len"])

    def is_available(self) -> tuple[bool, str]:
        problem = self._root_problem()
        if problem:
            return False, problem
        if not self.catalog.is_file():
            return False, f"catalog {self.catalog} not found"
        if not self.data_dir.is_dir():
            return False, f"data dir {self.data_dir} not found"
        try:
            import h5py  # noqa: F401
        except ImportError:
            return False, "h5py is not installed (add it to environment.yaml)"
        return True, ""

    @property
    def norm_mode(self) -> str:
        """The NORM_MODES entry in use (`norm_mode:` in config, i.e. $SEVIR_NORM)."""
        mode = str(self.config.get("norm_mode", "01"))
        if mode not in NORM_MODES:
            raise ValueError(
                f"SEVIR/{self.variant}: unknown norm_mode {mode!r}; "
                f"available: {sorted(NORM_MODES)}. Set it with $SEVIR_NORM, "
                f"--sevir_norm, or `norm_mode:` in data/SEVIR/config.yaml."
            )
        return mode

    def _target_affine(self) -> tuple[float, float]:
        """(mean, std) in raw VIL counts of the target normalization."""
        types = list(self.config["data_types"])
        if len(types) != 1:
            raise ValueError(
                f"SEVIR/{self.variant}: norm_mode gives one (mean, std) pair, "
                f"so it cannot normalize {len(types)} data_types {types}. Add "
                f"per-channel constants before using a multi-channel variant."
            )
        return _norm_mode_constants(self.norm_mode, self.config)

    def _loader_affine(self) -> tuple[float, float]:
        """(mean, std) in raw counts already applied by SEVIRDataLoader.

        `scale * (data + offset)` is equivalent to mean = -offset, std = 1/scale.
        """
        from data.SEVIR.preprocess import (
            PREPROCESS_OFFSET_01,
            PREPROCESS_OFFSET_SEVIR,
            PREPROCESS_SCALE_01,
            PREPROCESS_SCALE_SEVIR,
        )

        method = self.config.get("rescale_method", "01")
        scale_map = PREPROCESS_SCALE_01 if method == "01" else PREPROCESS_SCALE_SEVIR
        offset_map = PREPROCESS_OFFSET_01 if method == "01" else PREPROCESS_OFFSET_SEVIR
        var = self._var
        return -float(offset_map[var]), 1.0 / float(scale_map[var])

    def raw_affine(self) -> tuple[float, float]:
        """The `(mean, std)` that maps the loader's output back to raw counts."""
        loader_mean, loader_std = self._loader_affine()
        return (-loader_mean / loader_std, 1.0 / loader_std)

    def post_affine(self) -> tuple[float, float]:
        """The residual `(x - m) / s` applied after the loader to reach model space.

        With loader output x = (raw - m_l) / s_l and target z = (raw - M) / S,
        returns m = (M - m_l) / s_l, s = S / s_l.
        """
        target_mean, target_std = self._target_affine()
        loader_mean, loader_std = self._loader_affine()
        return ((target_mean - loader_mean) / loader_std, target_std / loader_std)

    def _norm_stats(self) -> NormStats:
        """Total raw-counts -> model-space affine (loader and norm_mode combined)."""
        mean_value, std_value = self._target_affine()
        n = len(list(self.config["data_types"]))
        mean = torch.full((n,), mean_value, dtype=torch.float32)
        std = torch.full((n,), std_value, dtype=torch.float32)
        return NormStats(
            state_mean=mean,
            state_std=std,
            legacy_scalar_mean=float(mean[0]),
            legacy_scalar_std=float(std[0]),
            # No one-step-difference statistics, so --pred_residual fails.
            diff_mean=None,
            diff_std=None,
            source=(f"sevir:{self.norm_mode}"
                    f"(mean={mean_value:g},std={std_value:g})"),
        )

    @functools.cached_property
    def metadata(self) -> DatasetMetadata:
        cfg = self.config
        cmaps = self._cmaps()
        variables = tuple(
            VariableSpec(
                name=str(v["name"]),
                units=str(v.get("units", "")),
                plot_range=(
                    tuple(float(x) for x in v["plot_range"])
                    if v.get("plot_range") is not None else None
                ),
                cmap=str(v.get("cmap", "viridis")),
                norm=cmaps.get(str(v["name"])),
            )
            for v in cfg["variables"]
        )
        return DatasetMetadata(
            name="SEVIR",
            variant=self.variant,
            grid=tuple(int(v) for v in cfg["grid"]),
            num_channels=int(cfg["num_channels"]),
            variables=variables,
            norm=self._norm_stats(),
            unit_factor=float(cfg.get("unit_factor", 1.0)),
            unit_name=str(cfg.get("unit_name", "")),
            cadence_hours=float(cfg["cadence_hours"]),
            periodic=tuple(bool(v) for v in cfg.get("periodic", (False, False))),
            max_window=int(cfg["max_window"]),
            assim_space=str(cfg.get("assim_space", "physical")),
            splits=self.splits,
            supports=tuple(cfg.get("supports",
                                   (SINGLE_STATE, FORECAST_WINDOW, ASSIM_TRAJECTORY))),
            raw_dtype=str(cfg.get("raw_dtype", "float32")),
            dataloader_kwargs=dict(cfg.get("dataloader_kwargs") or {}),
            # All modes, so --forward_norm can name a propagator's own normalization.
            norm_modes={name: (self._target_affine() if name == self.norm_mode
                               else _norm_mode_constants(name, cfg))
                        for name in NORM_MODES},
        )

    @staticmethod
    def _cmaps() -> dict:
        """Matplotlib norms for non-linear colour scales; also registers `sevir:vil`."""
        try:
            from data.SEVIR.cmap import register_cmaps, vil_cmap

            register_cmaps()
            return {"vil": vil_cmap()[1]}
        except Exception as exc:
            warnings.warn(f"SEVIR colour maps unavailable ({exc}); "
                          "plots using cmap 'sevir:vil' will fail")
            return {}

    @property
    def _var(self) -> str:
        return list(self.config["data_types"])[0]

    def single_state(self, split: str, **kwargs):
        from data.SEVIR.torch_datasets import SEVIRStateDataset
        self.root  # raises if SEVIR_ROOT is unset

        self.metadata.require(SINGLE_STATE)
        return SEVIRStateDataset(self.config, split_dates=self.split_dates(split),
                                 var=self._var, subset=bool(kwargs.get("subset", False)),
                                 post_affine=self.post_affine(),
                                 random_part=self.split_part(split))

    def forecast_window(self, split: str, *, init_states: int, pred_length: int,
                        step_hours: float | None = None, subset: bool = False,
                        standardize: bool = True, per_channel: bool = True,
                        subsample_step: int | None = None, **kwargs):
        """Windows normalized by `norm_mode`.

        `standardize=False` returns the loader's output (vil/255 under
        `rescale_method: "01"`) without the `norm_mode` affine. `per_channel`
        is a no-op (single channel).
        """
        from data.SEVIR.torch_datasets import SEVIRWindowDataset
        self.root  # raises if SEVIR_ROOT is unset

        md = self.metadata
        md.require(FORECAST_WINDOW)
        if subsample_step is None:
            subsample_step = 1 if step_hours is None else md.subsample_step(step_hours)
        md.check_window(init_states=init_states, pred_length=pred_length,
                        subsample_step=subsample_step)
        return SEVIRWindowDataset(
            self.config, split_dates=self.split_dates(split), var=self._var,
            init_states=init_states, pred_length=pred_length,
            subsample_step=subsample_step, subset=subset,
            post_affine=self.post_affine() if standardize else None,
            random_part=self.split_part(split),
        )

    def assim_trajectory(self, *, index: int = 0, split: str = "test",
                         path=None, sample: bool = True) -> AssimCase:
        from data.SEVIR.torch_datasets import SEVIRTrajectory
        self.root  # raises if SEVIR_ROOT is unset

        md = self.metadata
        md.require(ASSIM_TRAJECTORY)
        if path is not None:
            # SEVIR trajectories are selected by --data_index, not by file.
            raise ValueError(
                "SEVIR does not support --data_path: a trajectory is selected "
                f"with --data_index (got --data_path {path!r}). "
                "Use --data_index N --split <split> instead."
            )
        # "normalized": full model normalization (offset included);
        # "physical": raw VIL counts.
        affine = (self.post_affine() if md.assim_space == "normalized"
                  else self.raw_affine())
        states = SEVIRTrajectory(self.config, split_dates=self.split_dates(split),
                                 var=self._var, index=index,
                                 post_affine=affine,
                                 random_part=self.split_part(split),
                                 sample=sample)
        pixel = self.config.get("pixel_size_m")
        domain = None if pixel is None else (
            float(pixel) * md.ny, float(pixel) * md.nx)
        geometry = None
        if pixel is not None:
            import numpy as _np

            # Regular grid in metres, for covariance localization.
            geometry = {
                "x": _np.arange(md.nx, dtype=_np.float64) * float(pixel),
                "y": _np.arange(md.ny, dtype=_np.float64) * float(pixel),
                "Lx": float(pixel) * md.nx,
                "Ly": float(pixel) * md.ny,
                # Single variable: nothing to localize vertically.
                "rossby_radius_m": None,
                "periodic": tuple(md.periodic),
            }

        return AssimCase(
            states=states,
            metadata=md,
            source_path=self.catalog,
            grid_geometry=geometry,
            # No numerical dynamics: LETKF raises a clear error.
            model_params=None,
            domain_size_m=domain,
            dt=None,
            assim_interval_hours=md.cadence_hours,
        )
