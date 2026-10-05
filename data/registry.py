"""Discovery and construction of data sources.

Any `data/<name>/config.yaml` registers a source. Nothing here touches a data
root at import time.
"""

from __future__ import annotations

import functools
import importlib
import os
from pathlib import Path
import warnings
from typing import Any, Mapping

import yaml

from data.base import DataSource
from data.paths import MissingRootError, expand, resolve_root

DATA_DIR = Path(__file__).resolve().parent

# argparse defaults, kept here so every entry point agrees.
DEFAULT_DATASET = "SQG"
DEFAULT_VARIANT = None  # None -> the config's `default_variant`, else the first


class UnknownDataSource(KeyError):
    pass


def _config_paths() -> dict[str, Path]:
    """{name: config.yaml} for every data/*/config.yaml, without importing sources."""
    found: dict[str, Path] = {}
    for candidate in sorted(DATA_DIR.glob("*/config.yaml")):
        try:
            with candidate.open(encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except Exception as exc:
            # Warn rather than let a broken config vanish silently.
            warnings.warn(f"skipping unreadable data config {candidate}: {exc}")
            continue
        name = raw.get("name") or candidate.parent.name
        found[str(name)] = candidate
    return found


def available_sources() -> dict[str, Path]:
    return _config_paths()


def available() -> list[str]:
    return sorted(_config_paths())


def _load_raw(name: str) -> tuple[Mapping[str, Any], Path]:
    paths = _config_paths()
    if name not in paths:
        raise UnknownDataSource(
            f"unknown data source {name!r}; available: {sorted(paths)}"
        )
    path = paths[name]
    with path.open(encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}), path


def _merge_variant(raw: Mapping[str, Any], variant: str | None,
                   context: str) -> tuple[str, dict]:
    variants = raw.get("variants") or {}
    if not variants:
        raise ValueError(f"{context}: no `variants:` block")
    if variant is None:
        variant = raw.get("default_variant") or next(iter(variants))
    if variant not in variants:
        raise UnknownDataSource(
            f"{context}: unknown variant {variant!r}; "
            f"available: {sorted(variants)}"
        )
    merged = dict(variants[variant] or {})
    # Top-level keys act as defaults for every variant.
    for key, value in raw.items():
        if key in ("variants", "defaults", "default_variant"):
            continue
        merged.setdefault(key, value)
    return variant, merged


def _resolve_class(spec: str):
    """Import a `pkg.module:Class` spec."""
    if ":" not in spec:
        raise ValueError(
            f"source_class must be 'pkg.module:Class', got {spec!r}"
        )
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@functools.lru_cache(maxsize=None)
def _get_cached(name: str, variant: str | None, root: str | None) -> DataSource:
    raw, path = _load_raw(name)
    context = str(path.relative_to(DATA_DIR.parent))
    variant, merged = _merge_variant(raw, variant, context)

    source_class = merged.get("source_class")
    if not source_class:
        raise ValueError(f"{context}: missing `source_class:`")

    try:
        resolved_root = resolve_root(
            str(merged.get("root", "")), cli_override=root, context=context
        )
    except MissingRootError as err:
        # Metadata needs no data; the error is raised when `.root` is used.
        resolved_root = err
        merged = {**merged, "root": "<unset>"}
    # `${root}` is not an env var, so substitute it before expand().
    merged = _substitute_root(
        merged, "<unset>" if isinstance(resolved_root, Exception) else resolved_root)
    merged = expand(merged, context=context)

    cls = _resolve_class(source_class)
    return cls(merged, resolved_root, variant)


def _substitute_root(value, root: Path):
    if isinstance(value, str):
        return value.replace("${root}", str(root))
    if isinstance(value, dict):
        return {k: _substitute_root(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_root(v, root) for v in value]
    return value


def get_data_source(name: str = DEFAULT_DATASET, variant: str | None = None,
                    *, root: str | None = None) -> DataSource:
    """Build (or return a cached) data source. Touches no data files."""
    return _get_cached(name, variant, root)


def all_variants(name: str | None = None) -> list[tuple[str, str]]:
    """[(name, variant), ...] over every registered source, for parametrized tests."""
    out: list[tuple[str, str]] = []
    for source_name in ([name] if name else available()):
        raw, _ = _load_raw(source_name)
        for variant, body in (raw.get("variants") or {}).items():
            if (body or {}).get("available") is False:
                continue
            out.append((source_name, variant))
    return out


def _apply_sevir_norm(args, dataset: str) -> None:
    """Validate the SEVIR norm flags and route --sevir_norm into $SEVIR_NORM."""
    norm = getattr(args, "sevir_norm", None)
    forward = getattr(args, "forward_norm", None)
    output = getattr(args, "output_norm", None)
    # 'physical' and 'assim' are valid for every dataset.
    if output in ("physical", "assim"):
        output = None
    if not norm and not forward and not output:
        return
    for flag, value in (("--assim_norm", norm), ("--forward_norm", forward),
                        ("--output_norm", output)):
        if value and dataset != "SEVIR":
            raise ValueError(
                f"{flag} {value!r} only applies to --dataset SEVIR, got "
                f"{dataset!r}. Remove it, or select the SEVIR data source."
            )
    # Lazy import: sources are never imported at module import time.
    from data.SEVIR.source import NORM_MODES

    for flag, value in (("--assim_norm", norm), ("--forward_norm", forward),
                        ("--output_norm", output)):
        if value and value not in NORM_MODES:
            raise ValueError(
                f"{flag} {value!r} is not a known SEVIR normalization; "
                f"available: {sorted(NORM_MODES)}"
            )
    if not norm:
        return
    if os.environ.get("SEVIR_NORM") != norm:
        os.environ["SEVIR_NORM"] = norm
        # The source cache key does not include the normalization.
        _get_cached.cache_clear()


def data_source_from_args(args) -> DataSource:
    """Build the data source named by an argparse or checkpoint Namespace.

    Missing `dataset`/`variant`/`data_root`/`sevir_norm` fall back to defaults.
    """
    dataset = getattr(args, "dataset", None) or DEFAULT_DATASET
    _apply_sevir_norm(args, dataset)
    return get_data_source(
        dataset,
        getattr(args, "variant", None) or getattr(args, "dataset_variant", None),
        root=getattr(args, "data_root", None),
    )


#: Fallback --n_workers when neither the flag nor the source sets it.
DEFAULT_N_WORKERS = 4


def resolve_n_workers(src, args) -> int:
    """DataLoader worker count: `--n_workers` if given, else the source's default."""
    requested = getattr(args, "n_workers", None)
    if requested is not None:
        return int(requested)
    default = src.metadata.dataloader_kwargs.get("num_workers", DEFAULT_N_WORKERS)
    return int(default)


def add_dataset_args(parser, *, default_dataset: str = DEFAULT_DATASET,
                     default_variant: str | None = None):
    """Add --dataset/--variant/--data_root to an ArgumentParser."""
    parser.add_argument(
        "--dataset", type=str, default=default_dataset,
        help=f"Data source name under data/ (default: {default_dataset})",
    )
    parser.add_argument(
        "--variant", type=str, default=default_variant,
        help="Data source variant (default: the config's default_variant)",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Override the data source's root path",
    )
    parser.add_argument(
        "--sevir_norm", "--assim_norm", type=str, default=None,
        dest="sevir_norm",
        help="SEVIR only: the ASSIMILATION normalization -- the units the run "
             "itself works in, and so the units of the state, the "
             "observations, --obs_sigma, --init_std, the saved .nc and every "
             "logged metric. '01' = vil/255, 'flowdas' = (vil/255 - 0.5)/0.1 "
             "as in the FlowDAS/DAISI paper, 'standard' = (vil - 33.44)/47.54. "
             "See NORM_MODES in data/SEVIR/source.py. Set it to whatever the "
             "network doing the ASSIMILATION step was trained in (the DAISI "
             "prior is '01'; the DAWIS/FMW window models and the FlowDAS "
             "drift are 'flowdas'); a propagator trained in a different one is "
             "bridged by --forward_norm. Equivalent to exporting $SEVIR_NORM; "
             "unset leaves data/SEVIR/config.yaml in charge. --assim_norm is "
             "the same option under the name the run_all scripts use.",
    )
    parser.add_argument(
        "--forward_norm", type=str, default=None,
        help="SEVIR only: the normalization the FORWARD PROPAGATOR checkpoint "
             "was trained in, when that differs from --assim_norm. The "
             "forecaster converts assimilation units into it on the way in and "
             "back on the way out, so the state stays in --assim_norm "
             "everywhere else. Needed because the checkpoints disagree: the "
             "pretrained FlowDAS drift and the SEVIR FMW models are 'flowdas' "
             "while the DAISI prior is '01', so e.g. DAISI assimilates in '01' "
             "and propagates with --forward_norm flowdas. Unset means 'the "
             "same as the assimilation norm', which reproduces the previous "
             "behaviour exactly.",
    )
    parser.add_argument(
        "--output_norm", type=str, default=None,
        help="Units the RESULTS are written in: the saved .nc, every logged "
             "metric and the console RMSE line. 'physical' is the dataset's "
             "own unit (VIL counts on SEVIR, K on SQG) and is what the "
             "run_all scripts set, so that two methods are comparable even "
             "when they had to assimilate in different spaces. Any name from "
             "NORM_MODES also works -- '01' gives vil/255, which is what the "
             "FlowDAS and DAISI papers report in. Unset means 'the same as "
             "--assim_norm', the behaviour before this option, and the file "
             "records which was used as `nc.output_norm`. This changes only "
             "how results are REPORTED; the run still computes in "
             "--assim_norm, and --obs_sigma/--init_std are still given in it.",
    )
    return parser
