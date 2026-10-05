"""Dataset-agnostic interface for the data sources under `data/`.

A data source provides metadata and datasets; observation operators and
forward models live in `assimilation/` and `forecasting/`. `data/` never
imports from those packages.

Unit systems:
    raw          as stored on disk (SQG: potential vorticity)
    physical     raw * unit_factor, used by assimilation methods and observers
    normalized   (raw - mean) / std, used by the networks

Layouts: state (C, H, W), temporal (T, C, H, W), batched (B, T, C, H, W).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

# Canonical access-pattern names, used by `DatasetMetadata.supports`.
SINGLE_STATE = "single_state"
FORECAST_WINDOW = "forecast_window"
ASSIM_TRAJECTORY = "assim_trajectory"
ACCESS_PATTERNS = (SINGLE_STATE, FORECAST_WINDOW, ASSIM_TRAJECTORY)

# Axis of the channel/variable dimension in every tensor this interface returns.
CHANNEL_AXIS = -3

# Continuous colormap for spread/error panels of variables with a discrete cmap.
SPREAD_CMAP = "magma"


@dataclass(frozen=True)
class VariableSpec:
    """One physical variable (one channel)."""

    name: str
    units: str
    plot_range: tuple[float, float] | None = None  # None -> autoscale
    cmap: str = "jet"
    # Matplotlib Normalize for non-linear colour scales (e.g. SEVIR BoundaryNorm).
    norm: Any | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("VariableSpec.name must be non-empty")
        if self.plot_range is not None:
            lo, hi = self.plot_range
            if not lo < hi:
                raise ValueError(
                    f"VariableSpec({self.name}).plot_range must be increasing, "
                    f"got {self.plot_range}"
                )


@dataclass(frozen=True)
class NormStats:
    """Per-channel affine normalization statistics.

    Two conventions are kept: per-channel (forecasting, DAWIS) and a legacy
    scalar (unconditional generation and DAISI checkpoints). `diff_*` are
    statistics of one-step differences of normalized data, or None if not
    computed.
    """

    state_mean: torch.Tensor  # (C,) float32 cpu
    state_std: torch.Tensor  # (C,) float32 cpu
    legacy_scalar_mean: float
    legacy_scalar_std: float
    diff_mean: torch.Tensor | None = None  # (C,)
    diff_std: torch.Tensor | None = None  # (C,)
    source: str = "config"  # provenance, logged on every run

    def __post_init__(self) -> None:
        if self.state_mean.shape != self.state_std.shape:
            raise ValueError(
                f"state_mean {tuple(self.state_mean.shape)} and state_std "
                f"{tuple(self.state_std.shape)} must have the same shape"
            )
        if self.state_std.ndim != 1:
            raise ValueError("state_mean/state_std must be 1-D, shape (C,)")
        if not bool(torch.all(self.state_std > 0)):
            raise ValueError(f"state_std must be positive, got {self.state_std}")
        if self.legacy_scalar_std <= 0:
            raise ValueError(
                f"legacy_scalar_std must be positive, got {self.legacy_scalar_std}"
            )
        for name in ("diff_mean", "diff_std"):
            value = getattr(self, name)
            if value is not None and value.shape != self.state_mean.shape:
                raise ValueError(
                    f"{name} {tuple(value.shape)} must match state_mean "
                    f"{tuple(self.state_mean.shape)}"
                )

    @property
    def has_diff(self) -> bool:
        return self.diff_mean is not None and self.diff_std is not None

    def mean_std(self, *, per_channel: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """The (mean, std) pair for the requested convention, both shape (C,)."""
        if per_channel:
            return self.state_mean, self.state_std
        c = self.state_mean.numel()
        return (
            torch.full((c,), float(self.legacy_scalar_mean), dtype=torch.float32),
            torch.full((c,), float(self.legacy_scalar_std), dtype=torch.float32),
        )

    def view(self, ndim: int, *, per_channel: bool = True):
        """(mean, std) reshaped to broadcast against an ndim-D tensor.

        ndim=3 -> (C,1,1);  ndim=4 -> (1,C,1,1);  ndim=5 -> (1,1,C,1,1).
        """
        if ndim < 3:
            raise ValueError(f"need at least 3 dims (C,H,W), got ndim={ndim}")
        mean, std = self.mean_std(per_channel=per_channel)
        shape = [1] * ndim
        shape[CHANNEL_AXIS] = mean.numel()
        return mean.view(shape), std.view(shape)

    def normalize(self, x, *, per_channel: bool = True):
        mean, std = self.view(x.ndim, per_channel=per_channel)
        return (x - mean.to(x.device, x.dtype)) / std.to(x.device, x.dtype)

    def denormalize(self, z, *, per_channel: bool = True):
        mean, std = self.view(z.ndim, per_channel=per_channel)
        return z * std.to(z.device, z.dtype) + mean.to(z.device, z.dtype)


@dataclass(frozen=True)
class DatasetMetadata:
    """Static description of a dataset variant: grid, variables, units, norms."""

    name: str
    variant: str
    grid: tuple[int, int]  # (ny, nx), not necessarily square
    num_channels: int
    variables: tuple[VariableSpec, ...]
    norm: NormStats
    # raw -> physical factor (SQG `scalefact`; 1.0 for SEVIR).
    unit_factor: float = 1.0
    unit_name: str = ""
    # Hours between consecutive time indices (SEVIR-LR: 1/6).
    cadence_hours: float = 1.0
    # Per-axis periodicity, passed to SongUNet(circular_padding=...).
    periodic: tuple[bool, bool] = (False, False)
    # Units of assimilation runs: "physical" (raw * unit_factor, SQG) or
    # "normalized" (model space, SEVIR). See `assim_scale`.
    assim_space: str = "physical"
    # Longest trajectory window the source can serve.
    max_window: int = 1
    splits: tuple[str, ...] = ()
    supports: tuple[str, ...] = ACCESS_PATTERNS
    raw_dtype: str = "float32"
    # Per-source DataLoader defaults; an explicit --n_workers overrides.
    dataloader_kwargs: Mapping[str, Any] = field(default_factory=dict)
    # Named normalizations as (mean, std) in raw units, for `--forward_norm`.
    norm_modes: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.grid) != 2 or not all(
            isinstance(v, int) and v > 0 for v in self.grid
        ):
            raise ValueError(f"grid must be two positive ints (ny, nx), got {self.grid}")
        if self.num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {self.num_channels}")
        if len(self.variables) != self.num_channels:
            raise ValueError(
                f"{self.name}/{self.variant}: {len(self.variables)} VariableSpecs "
                f"for num_channels={self.num_channels}"
            )
        if self.norm.state_mean.numel() != self.num_channels:
            raise ValueError(
                f"{self.name}/{self.variant}: norm stats have "
                f"{self.norm.state_mean.numel()} channels, expected "
                f"{self.num_channels}"
            )
        if not isinstance(self.unit_factor, float) or not self.unit_factor > 0:
            raise ValueError(
                f"unit_factor must be a positive float, got {self.unit_factor!r}"
            )
        if not isinstance(self.cadence_hours, float) or not self.cadence_hours > 0:
            raise ValueError(
                f"cadence_hours must be a positive float, got {self.cadence_hours!r}"
            )
        if len(self.periodic) != 2 or not all(
            isinstance(v, bool) for v in self.periodic
        ):
            raise ValueError(f"periodic must be two bools, got {self.periodic}")
        if self.max_window < 1:
            raise ValueError(f"max_window must be >= 1, got {self.max_window}")
        if self.assim_space not in ("physical", "normalized"):
            raise ValueError(
                f"assim_space must be 'physical' or 'normalized', "
                f"got {self.assim_space!r}"
            )
        unknown = set(self.supports) - set(ACCESS_PATTERNS)
        if unknown:
            raise ValueError(f"unknown access patterns {sorted(unknown)}")
        if not self.supports:
            raise ValueError("supports must name at least one access pattern")

    @property
    def ny(self) -> int:
        return self.grid[0]

    @property
    def nx(self) -> int:
        return self.grid[1]

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return (self.num_channels, self.ny, self.nx)

    @property
    def is_square(self) -> bool:
        return self.ny == self.nx

    @property
    def variable_names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def units(self) -> list[str]:
        return [v.units for v in self.variables]

    def variable(self, var: int | str) -> VariableSpec:
        if isinstance(var, int):
            return self.variables[var]
        for spec in self.variables:
            if spec.name == var:
                return spec
        raise KeyError(f"{self.name}/{self.variant} has no variable {var!r}")

    def plot_range(self, var: int | str) -> tuple[float | None, float | None]:
        spec = self.variable(var)
        return (None, None) if spec.plot_range is None else spec.plot_range

    def imshow_kwargs(
        self, var: int | str, *, spread: bool = False
    ) -> dict[str, Any]:
        """`imshow` colour kwargs for one variable, in physical units.

        A variable with a `norm` gets only the norm; otherwise `plot_range`
        sets vmin/vmax. `spread=True` (std or error maps) autoscales and uses
        `SPREAD_CMAP` for discrete colormaps.
        """
        spec = self.variable(var)
        if spread:
            discrete = spec.norm is not None
            return {"cmap": SPREAD_CMAP if discrete else spec.cmap}
        kwargs: dict[str, Any] = {"cmap": spec.cmap}
        if spec.norm is not None:
            kwargs["norm"] = spec.norm
        elif spec.plot_range is not None:
            kwargs["vmin"], kwargs["vmax"] = spec.plot_range
        return kwargs

    def physical_std(self, *, per_channel: bool = True) -> torch.Tensor:
        """`std * unit_factor`, shape (C,): normalized -> physical scale.

        The legacy scalar profile is computed in float64, the per-channel one
        in float32, to reproduce existing results exactly.
        """
        if not per_channel:
            value = float(self.norm.legacy_scalar_std) * float(self.unit_factor)
            return torch.full((self.num_channels,), value, dtype=torch.float64)
        _, std = self.norm.mean_std(per_channel=True)
        return std * self.unit_factor

    def assim_scale(self, data_std=None, *, per_channel: bool = True):
        """Divisor mapping assimilation units to a network's normalized units.

        Methods use `z = x / scale` and `x = z * scale`. For "physical" data
        (SQG) the scale is `data_std * unit_factor`; for "normalized" data
        (SEVIR) the loader already applied the full affine, so it is 1.

        Args:
            data_std: optional checkpoint std; defaults to the dataset's stats.
            per_channel: normalization profile used when `data_std` is None.

        Returns:
            A (C,) tensor, or `data_std`-shaped if given.
        """
        if self.assim_space == "normalized":
            if data_std is None:
                return torch.ones(self.num_channels, dtype=torch.float32)
            return torch.ones_like(torch.as_tensor(data_std))
        if data_std is None:
            return self.physical_std(per_channel=per_channel)
        return torch.as_tensor(data_std) * self.unit_factor

    def named_norm(self, name: str | None) -> tuple[float, float] | None:
        """One of `norm_modes` as `(mean, std)` in raw units, or None if `name` is None.

        Used when a checkpoint was trained under a different normalization
        than the run assimilates in. Unknown names raise.
        """
        if name is None:
            return None
        if not self.norm_modes:
            raise ValueError(
                f"{self.name}/{self.variant} defines no named normalizations, "
                f"so --forward_norm {name!r} cannot be resolved. It applies to "
                f"datasets with more than one convention (SEVIR); leave it "
                f"unset elsewhere."
            )
        if name not in self.norm_modes:
            raise ValueError(
                f"{self.name}/{self.variant}: unknown normalization {name!r}; "
                f"available: {sorted(self.norm_modes)}."
            )
        mean, std = self.norm_modes[name]
        return float(mean), float(std)

    #: `--output_norm` value for the dataset's own physical units.
    OUTPUT_PHYSICAL = "physical"

    def _assim_to_raw(self) -> tuple[float, float]:
        """`(mean, std)` with `raw = v * std + mean`, v in assimilation units."""
        if self.assim_space == "normalized":
            mean, std = self.norm.mean_std(per_channel=False)
            return float(mean[0]), float(std[0])
        # physical: v = raw * unit_factor, so raw = v / unit_factor.
        return 0.0, 1.0 / float(self.unit_factor)

    def _raw_to(self, norm: str) -> tuple[float, float]:
        """`(mean, std)` with `out = (raw - mean) / std` for an output space."""
        if norm == self.OUTPUT_PHYSICAL:
            return 0.0, 1.0 / float(self.unit_factor)
        return self.named_norm(norm)

    def output_affine(self, output_norm: str | None) -> tuple[float, float]:
        """`(a, b)` with `out = a * v + b`, v in assimilation units.

        Maps results into the `--output_norm` space so metrics are comparable
        across methods. `None` or "assim" is the identity.
        """
        if output_norm in (None, "assim"):
            return 1.0, 0.0
        m_a, s_a = self._assim_to_raw()
        m_o, s_o = self._raw_to(output_norm)
        # out = ((v * s_a + m_a) - m_o) / s_o
        return s_a / s_o, (m_a - m_o) / s_o

    def stored_to_physical_affine(self, stored_norm: str | None) -> tuple[float, float]:
        """`(a, b)` with `physical = a * s + b`, s as stored in a result file.

        `None` or "assim" means the file is in assimilation units.
        """
        u = float(self.unit_factor)
        if stored_norm == self.OUTPUT_PHYSICAL:
            return 1.0, 0.0
        if stored_norm in (None, "assim"):
            if self.assim_space != "normalized":
                return 1.0, 0.0
            mean, std = self.norm.mean_std(per_channel=False)
            return float(std[0]) * u, float(mean[0]) * u
        m_n, s_n = self.named_norm(stored_norm)
        return s_n * u, m_n * u

    def to_physical(self, x):
        return x * self.unit_factor

    def from_physical(self, x):
        return x / self.unit_factor

    def format_lead_time(self, n_steps: int) -> str:
        """Human-readable lead time, e.g. 't=2 (6 h)' or 't=2 (20 min)'."""
        hours = n_steps * self.cadence_hours
        if hours and hours < 1.0:
            return f"t={n_steps} ({round(hours * 60)} min)"
        pretty = f"{hours:g}"
        return f"t={n_steps} ({pretty} h)"

    def subsample_step(self, step_hours: float) -> int:
        """Index stride realizing a requested wall-clock step."""
        step = round(step_hours / self.cadence_hours)
        if step < 1:
            raise ValueError(
                f"{self.name}/{self.variant}: requested step {step_hours} h is "
                f"shorter than the data cadence {self.cadence_hours} h"
            )
        return step

    def check_window(self, *, init_states: int, pred_length: int,
                     subsample_step: int = 1) -> int:
        """Validate and return the sample length a request needs."""
        if init_states == 0:
            length = pred_length * subsample_step
        else:
            length = 1 + (init_states - 1 + pred_length) * subsample_step
        if length > self.max_window:
            raise ValueError(
                f"{self.name}/{self.variant}: requested window of {length} steps "
                f"(init_states={init_states}, pred_length={pred_length}, "
                f"subsample_step={subsample_step}) exceeds max_window="
                f"{self.max_window}"
            )
        return length

    def supports_pattern(self, pattern: str) -> bool:
        return pattern in self.supports

    def require(self, pattern: str) -> None:
        if not self.supports_pattern(pattern):
            raise NotImplementedError(
                f"{self.name}/{self.variant} does not support {pattern!r}; "
                f"it supports {list(self.supports)}"
            )

    def describe(self) -> dict:
        """Flat, string-valued summary for netCDF attrs and W&B config."""
        return {
            "dataset": self.name,
            "variant": self.variant,
            "grid": list(self.grid),
            "num_channels": self.num_channels,
            "variable_names": list(self.variable_names),
            "units": list(self.units),
            "unit_factor": self.unit_factor,
            "unit_name": self.unit_name,
            "cadence_hours": self.cadence_hours,
            "norm_source": self.norm.source,
        }


@dataclass
class AssimCase:
    """One assimilation trajectory, plus what a forward model needs.

    `states` yields raw units (see `metadata.assim_space`). `model_params` is
    None when the dataset has no numerical dynamics (SEVIR).
    """

    states: Any  # states[t] -> (C, ny, nx); states[a:b] -> (b-a, C, ny, nx)
    metadata: DatasetMetadata
    source_path: Path
    # Numerical dynamics parameters, or None.
    model_params: dict | None = None
    # {"x", "y", "Lx", "Ly" (metres), "vertical_localization"}, for localization.
    grid_geometry: dict | None = None
    domain_size_m: tuple[float, float] | None = None
    dt: float | None = None
    assim_interval_hours: float | None = None

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

    @property
    def n_times(self) -> int:
        return len(self.states)

    @property
    def has_numerical_model(self) -> bool:
        return self.model_params is not None

    @property
    def has_grid_geometry(self) -> bool:
        return self.grid_geometry is not None


class DataSource(abc.ABC):
    """One dataset family + variant: metadata and datasets.

    Construction must be cheap and touch no files; use `is_available()` to
    check whether the data is present.
    """

    def __init__(self, config: Mapping, root: Path | Exception, variant: str):
        # `root` is the MissingRootError when the root variable is unset: the
        # source can still describe itself, and raises on first use of `.root`.
        self.config = config
        self._root_error = root if isinstance(root, Exception) else None
        self._root = None if self._root_error else Path(root)
        self.variant = variant

    @property
    def root(self) -> Path:
        if self._root_error is not None:
            raise self._root_error
        return self._root

    def _root_problem(self) -> str | None:
        """Why the root is unusable, or None."""
        if self._root_error is not None:
            return str(self._root_error).splitlines()[0]
        if not self._root.exists():
            return f"root {self._root} does not exist"
        return None

    @property
    @abc.abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Cached; must not read data files."""

    def is_available(self) -> tuple[bool, str]:
        """(available, reason). Only `os.path.exists`-style checks."""
        problem = self._root_problem()
        if problem:
            return False, problem
        return True, ""

    def has_per_channel_stats(self) -> bool:
        """Whether per-channel `state_mean`/`state_std` are available."""
        return True

    def require_per_channel_stats(self) -> None:
        """Raise unless per-channel statistics are available."""
        if not self.has_per_channel_stats():
            raise FileNotFoundError(
                f"{self.metadata.name}/{self.variant} has no per-channel "
                f"standardization statistics."
            )

    @abc.abstractmethod
    def split_dir(self, split: str) -> Path: ...

    @abc.abstractmethod
    def list_trajectories(self, split: str, pattern: str | None = None) -> list[Path]:
        """Sorted trajectory files for a split (`--data_index` indexes this list)."""

    def trajectory_length(self, split: str) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def single_state(self, split: str, **kwargs):
        """i.i.d. snapshots. Items: (C, ny, nx) float32, **normalized**."""

    @abc.abstractmethod
    def forecast_window(self, split: str, *, init_states: int, pred_length: int,
                        step_hours: float | None = None, subset: bool = False,
                        **kwargs):
        """Windows. Items: ((init_states,C,ny,nx), (pred_length,C,ny,nx))
        float32, **normalized**."""

    @abc.abstractmethod
    def assim_trajectory(self, *, index: int = 0, split: str = "test") -> AssimCase:
        """One trajectory in **raw** units, plus optional model_params."""

    def close(self) -> None:
        """Release any open file handles. Overridden by sources that hold them."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.metadata.name!r}, variant={self.variant!r})"
