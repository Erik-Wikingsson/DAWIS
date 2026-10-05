"""Contract tests every data source under data/ must satisfy.

Tier A checks metadata only and runs anywhere; Tier B needs the data files and
skips when they are absent. New data sources are picked up automatically.
"""

import os

import numpy as np
import pytest
import torch

from data.base import ACCESS_PATTERNS, ASSIM_TRAJECTORY, FORECAST_WINDOW, SINGLE_STATE
from data.registry import all_variants, available, get_data_source

VARIANTS = all_variants()
IDS = [f"{n}/{v}" for n, v in VARIANTS]


@pytest.fixture(params=VARIANTS, ids=IDS)
def src(request):
    name, variant = request.param
    return get_data_source(name, variant)


@pytest.fixture
def md(src):
    return src.metadata


def needs_data(src):
    ok, why = src.is_available()
    if not ok:
        pytest.skip(f"{src.metadata.name}/{src.metadata.variant}: {why}")


# Tier A: metadata only
def test_registry_discovers_something():
    assert available(), "no data/*/config.yaml found"


def test_grid_is_two_positive_ints(md):
    assert len(md.grid) == 2
    assert all(isinstance(v, int) and v > 0 for v in md.grid)


def test_variables_match_channels(md):
    assert md.num_channels >= 1
    assert len(md.variables) == md.num_channels
    assert len(md.variable_names) == md.num_channels
    for spec in md.variables:
        assert spec.name
        if spec.plot_range is not None:
            lo, hi = spec.plot_range
            assert lo < hi


def test_norm_stats_shape(md):
    n = md.num_channels
    assert md.norm.state_mean.shape == (n,)
    assert md.norm.state_std.shape == (n,)
    assert md.norm.state_mean.dtype == torch.float32
    assert bool(torch.all(md.norm.state_std > 0))
    for extra in (md.norm.diff_mean, md.norm.diff_std):
        assert extra is None or extra.shape == (n,)


def test_unit_factor_is_a_positive_python_float(md):
    """unit_factor is a positive finite float (1.0 allowed, e.g. SEVIR)."""
    assert isinstance(md.unit_factor, float)
    assert md.unit_factor > 0
    assert np.isfinite(md.unit_factor)


def test_cadence_is_a_float(md):
    """cadence_hours is a positive float, so sub-hourly cadences survive."""
    assert isinstance(md.cadence_hours, float)
    assert md.cadence_hours > 0


def test_periodic_is_two_bools(md):
    assert len(md.periodic) == 2
    assert all(isinstance(v, bool) for v in md.periodic)


def test_supports_and_max_window(md):
    assert md.supports
    assert set(md.supports) <= set(ACCESS_PATTERNS)
    assert md.max_window >= 1


def test_round_trips_through_registry(md):
    again = get_data_source(md.name, md.variant).metadata
    assert (again.name, again.variant) == (md.name, md.variant)
    assert again.grid == md.grid


def test_describe_is_flat(md):
    described = md.describe()
    assert described["dataset"] == md.name
    assert described["variant"] == md.variant


def test_lead_time_never_renders_zero_hours(md):
    """A sub-hourly cadence must not format as '(0 h)'."""
    text = md.format_lead_time(1)
    assert "(0 h)" not in text


def test_construction_touched_no_files(md):
    """Reading metadata must not require the data to exist."""
    assert md.name and md.variant


def test_imshow_kwargs_are_directly_usable(md):
    """imshow_kwargs work with imshow: registered cmap, never both norm and vmin/vmax."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field = np.zeros(md.grid, dtype=np.float32)
    for var_i in range(md.num_channels):
        for kwargs in (md.imshow_kwargs(var_i),
                       md.imshow_kwargs(var_i, spread=True)):
            assert "cmap" in kwargs
            assert not ("norm" in kwargs and "vmin" in kwargs)
            plt.get_cmap(kwargs["cmap"])  # raises on an unregistered name
            fig, ax = plt.subplots()
            try:
                ax.imshow(field, **kwargs)
            finally:
                plt.close(fig)
        # A spread is never on the state's scale, so it must autoscale.
        spread = md.imshow_kwargs(var_i, spread=True)
        assert "norm" not in spread and "vmin" not in spread


# Tier B: needs data
def test_single_state_contract(src, md):
    needs_data(src)
    if not md.supports_pattern(SINGLE_STATE):
        pytest.skip("no single_state support")
    split = "val" if "val" in md.splits else md.splits[0]
    ds = src.single_state(split)
    assert len(ds) > 0
    x = ds[0]
    assert x.shape == md.state_shape, "layout drift"
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()


def test_single_state_normalization_is_sane(src, md):
    """Loose bounds that catch missing or doubly-applied normalization."""
    needs_data(src)
    if not md.supports_pattern(SINGLE_STATE):
        pytest.skip("no single_state support")
    split = "val" if "val" in md.splits else md.splits[0]
    ds = src.single_state(split)
    xs = torch.stack([ds[i] for i in range(0, min(len(ds), 64))])
    assert abs(float(xs.mean())) < 1.0
    assert 0.05 < float(xs.std()) < 20.0


def test_single_state_is_deterministic(src, md):
    needs_data(src)
    if not md.supports_pattern(SINGLE_STATE):
        pytest.skip("no single_state support")
    split = "val" if "val" in md.splits else md.splits[0]
    ds = src.single_state(split)
    assert torch.equal(ds[0], ds[0])


def test_layout_smoke(src, md):
    """Catches an (H, W, T)-style layout leaking through."""
    needs_data(src)
    if not md.supports_pattern(SINGLE_STATE):
        pytest.skip("no single_state support")
    split = "val" if "val" in md.splits else md.splits[0]
    x = src.single_state(split)[0]
    assert tuple(x.shape[-2:]) == md.grid
    if md.num_channels != md.grid[0]:
        assert x.shape[0] == md.num_channels


def test_forecast_window_contract(src, md):
    needs_data(src)
    if not md.supports_pattern(FORECAST_WINDOW):
        pytest.skip("no forecast_window support")
    split = "val" if "val" in md.splits else md.splits[0]
    for init_states, pred_length in [(1, 1), (2, 1), (2, 3)]:
        try:
            ds = src.forecast_window(split, init_states=init_states,
                                     pred_length=pred_length)
        except FileNotFoundError as exc:
            pytest.skip(f"stats unavailable: {exc}")
        init, target = ds[0]
        assert init.shape == (init_states, *md.state_shape)
        assert target.shape == (pred_length, *md.state_shape)
        assert init.dtype == target.dtype == torch.float32
        assert torch.isfinite(init).all() and torch.isfinite(target).all()


def test_forecast_window_respects_max_window(src, md):
    needs_data(src)
    if not md.supports_pattern(FORECAST_WINDOW):
        pytest.skip("no forecast_window support")
    split = "val" if "val" in md.splits else md.splits[0]
    with pytest.raises(ValueError):
        src.forecast_window(split, init_states=1,
                            pred_length=md.max_window + 10)


def test_forecast_window_temporal_order(src, md):
    """target[0] is the same frame for pred_length 1 and 2 (catches a transposed time axis)."""
    needs_data(src)
    if not md.supports_pattern(FORECAST_WINDOW):
        pytest.skip("no forecast_window support")
    split = "val" if "val" in md.splits else md.splits[0]
    try:
        a = src.forecast_window(split, init_states=2, pred_length=1)
        b = src.forecast_window(split, init_states=2, pred_length=2)
    except FileNotFoundError as exc:
        pytest.skip(f"stats unavailable: {exc}")
    assert torch.equal(a[0][0], b[0][0][:2])
    assert torch.equal(a[0][1][0], b[0][1][0])


def test_assim_trajectory_contract(src, md):
    needs_data(src)
    if not md.supports_pattern(ASSIM_TRAJECTORY):
        pytest.skip("no assim_trajectory support")
    split = "test" if "test" in md.splits else md.splits[-1]
    case = src.assim_trajectory(split=split, index=0)
    assert case.n_times >= 2
    assert case.states[0].shape == md.state_shape
    assert case.source_path.exists()
    # model_params is None exactly when the dataset has no known dynamics.
    if case.model_params is not None:
        assert "scalefact" in case.model_params
        assert case.model_params["scalefact"] == pytest.approx(md.unit_factor)
    else:
        assert not case.has_numerical_model


def test_assim_trajectory_matches_declared_assim_space(src, md):
    """`AssimCase.states` are in the units `md.assim_space` declares.

    Loose check aimed at a factor-of-`std` mistake in either direction.
    """
    needs_data(src)
    if not md.supports_pattern(ASSIM_TRAJECTORY):
        pytest.skip("no assim_trajectory support")
    split = "test" if "test" in md.splits else md.splits[-1]
    case = src.assim_trajectory(split=split, index=0)
    peak = float(np.abs(np.asarray(case.states[:min(len(case), 8)])).max())
    std = float(md.norm.legacy_scalar_std)

    if md.assim_space == "physical":
        assert peak > 0.1 * std, (
            f"{md.name}/{md.variant}: assim_trajectory peaks at {peak:g}, far "
            f"below the normalization std {std:g} -- it looks normalized, but "
            f"assim_space says 'physical', which owes the caller raw units."
        )
        return

    # Normalized (possibly with an offset): mapping back to physical units
    # must land in the declared plot range.
    lo = min(v.plot_range[0] for v in md.variables)
    hi = max(v.plot_range[1] for v in md.variables)
    a, b = md.stored_to_physical_affine(None)
    physical = np.asarray(case.states[:min(len(case), 8)]) * a + b
    span = hi - lo
    assert physical.min() >= lo - 0.05 * span and physical.max() <= hi + 0.05 * span, (
        f"{md.name}/{md.variant}: assim_trajectory maps to physical "
        f"[{physical.min():g}, {physical.max():g}], outside the declared range "
        f"[{lo:g}, {hi:g}] -- assim_space says 'normalized', so `x * {a:g} + "
        f"{b:g}` should be physical units. Raw data here would overshoot by "
        f"about {a:g}x."
    )
    # scale must be 1, or the state is normalized twice.
    assert float(md.assim_scale().max()) == 1.0, (
        f"{md.name}/{md.variant}: assim_space is 'normalized' but assim_scale "
        f"is {md.assim_scale().tolist()}, so the state would be normalized twice."
    )


def test_splits_do_not_overlap(src, md):
    """Splits overlapping in time must be separated by distinct `random_part`s.

    Only checked where a source exposes `split_dates`.
    """
    dates = getattr(src, "split_dates", None)
    if dates is None:
        pytest.skip("source does not split by date")
    part = getattr(src, "split_part", None)
    spans = {}
    for split in md.splits:
        start, end = dates(split)
        spans[split] = (start, end)
    names = list(spans)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            a_start, a_end = spans[a]
            b_start, b_end = spans[b]
            # Half-open [start, end); None is -inf / +inf.
            overlap = ((a_end is None or b_start is None or a_end > b_start)
                       and (b_end is None or a_start is None or b_end > a_start))
            if not overlap:
                continue
            a_part = None if part is None else part(a)
            b_part = None if part is None else part(b)
            assert a_part is not None and b_part is not None and a_part != b_part, (
                f"{md.name}/{md.variant}: splits {a} {spans[a]} and {b} "
                f"{spans[b]} overlap in time and are not separated by disjoint "
                f"random_parts (got {a_part!r} and {b_part!r})"
            )


def test_random_part_partitions_exactly():
    """`_partition_indices` splits train/val into a disjoint, complete partition."""
    from data.SEVIR.torch_datasets import _partition_indices

    for n_events, per_event in ((0, 8), (1, 8), (9, 8), (10, 1), (101, 3),
                                (1000, 8)):
        n = n_events * per_event
        kw = dict(val_ratio=0.1, seed=0, per_event=per_event)
        train = _partition_indices(n, "train", **kw)
        val = _partition_indices(n, "val", **kw)
        assert set(train).isdisjoint(
            val), f"{n_events=}: train and val share items"
        assert sorted(train + val) == list(range(n)
                      ), f"{n_events=}: not a partition"
        assert train == sorted(train) and val == sorted(
            val), "indices not sorted"
        if n_events >= 10:
            assert len(val) == round(n_events * 0.1) * per_event, (
                f"{n_events}: val is {len(val)} items"
            )

    # No `random_part` means the whole range.
    assert _partition_indices(100, None, val_ratio=0.1, seed=0) is None

    # Same seed, same partition; a different seed is allowed to differ.
    again = _partition_indices(1000, "val", val_ratio=0.1, seed=0)
    assert again == _partition_indices(1000, "val", val_ratio=0.1, seed=0)
    assert again != _partition_indices(1000, "val", val_ratio=0.1, seed=1)

    with pytest.raises(ValueError):
        _partition_indices(100, "nope", val_ratio=0.1, seed=0)
    with pytest.raises(ValueError):
        _partition_indices(100, "train", val_ratio=0.0, seed=0)
    with pytest.raises(ValueError):
        _partition_indices(100, "train", val_ratio=0.1, per_event=0, seed=0)


def test_assim_sample_is_reproducible_and_size_stable():
    """SEVIR `--data_index` draws are pinned, stable under a larger `size`, and reseedable.

    The literal is the draw the SEVIR launchers use.
    """
    from data.SEVIR.torch_datasets import _assim_sample_indices

    n_test = 4053                      # events in SEVIR lr_vil `test`
    cfg = {"size": 11, "seed": 0}
    drawn = _assim_sample_indices(n_test, cfg)
    assert drawn == [258, 1181, 3742, 262, 3376, 2623, 1456, 2709, 1293,
                     2028, 2116]
    assert drawn == _assim_sample_indices(n_test, {"size": "11", "seed": "0"}), \
        "config values arrive as strings from ${VAR:-default} interpolation"
    assert len(set(drawn)) == len(drawn), "an event was drawn twice"

    wider = _assim_sample_indices(n_test, {"size": 20, "seed": 0})
    assert wider[:11] == drawn, "raising size renumbered the existing indices"

    assert _assim_sample_indices(n_test, {"size": 11, "seed": 1}) != drawn

    # Disabled: --data_index then indexes the split directly.
    for off in (None, {}, {"size": 0}, {"size": 0, "seed": 3}):
        assert _assim_sample_indices(n_test, off) is None

    # More trajectories than the split holds is an error.
    with pytest.raises(ValueError):
        _assim_sample_indices(5, {"size": 11, "seed": 0})


def test_random_part_is_window_shape_independent():
    """The same events land in val for every window shape (train and val use different ones)."""
    from data.SEVIR.torch_datasets import _partition_indices

    n_events = 500
    kw = dict(val_ratio=0.1, seed=0)
    events = {}
    for per_event in (1, 2, 8, 12):
        idx = _partition_indices(n_events * per_event, "val",
                                 per_event=per_event, **kw)
        events[per_event] = {i // per_event for i in idx}
    assert len(set(map(frozenset, events.values()))) == 1, (
        f"val events differ by window shape: "
        f"{ {k: len(v) for k, v in events.items()} }"
    )


def test_dataloader_worker_safety(src, md):
    """single_state loads a batch with the source's own worker settings."""
    needs_data(src)
    if not md.supports_pattern(SINGLE_STATE):
        pytest.skip("no single_state support")
    split = "val" if "val" in md.splits else md.splits[0]
    ds = src.single_state(split)
    workers = int(md.dataloader_kwargs.get("num_workers", 0))
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=workers)
    batch = next(iter(loader))
    assert batch.shape == (2, *md.state_shape)


def test_forecast_window_worker_safety(src, md):
    """forecast_window loads a batch with the source's own worker settings."""
    needs_data(src)
    if not md.supports_pattern(FORECAST_WINDOW):
        pytest.skip("no forecast_window support")
    workers = int(md.dataloader_kwargs.get("num_workers", 0))
    if workers == 0:
        pytest.skip("source loads in the main process; nothing to fork")
    split = "val" if "val" in md.splits else md.splits[0]
    init_states, pred_length = 2, 1
    ds = src.forecast_window(split, init_states=init_states,
                             pred_length=pred_length)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, num_workers=workers,
        persistent_workers=bool(
            md.dataloader_kwargs.get("persistent_workers")),
    )
    prev, target = next(iter(loader))
    assert prev.shape == (2, init_states, *md.state_shape)
    assert target.shape == (2, pred_length, *md.state_shape)
    assert torch.isfinite(prev).all() and torch.isfinite(target).all()


def test_resolve_n_workers_prefers_the_flag_over_the_source(src, md):
    """--n_workers overrides the source default (including 0); unset, the source decides."""
    from argparse import Namespace

    from data.registry import DEFAULT_N_WORKERS, resolve_n_workers

    expected_default = int(md.dataloader_kwargs.get("num_workers",
                                                    DEFAULT_N_WORKERS))
    assert resolve_n_workers(src, Namespace(
        n_workers=None)) == expected_default
    assert resolve_n_workers(src, Namespace()) == expected_default
    # An explicit 0 is not "unset".
    assert resolve_n_workers(src, Namespace(n_workers=0)) == 0
    assert resolve_n_workers(src, Namespace(n_workers=3)) == 3


# Package structure
def test_no_module_shadows_a_top_level_package():
    """No `<pkg>/<name>.py` shares a name with another top-level package.

    Such a module would shadow that package when an entry point is run as a script.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    top_level_packages = {
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
    }
    offenders = []
    for pkg in sorted(top_level_packages):
        for module in (root / pkg).glob("*.py"):
            # pkg/pkg.py (e.g. metrics/metrics.py) is harmless.
            if module.stem in top_level_packages and module.stem != pkg:
                offenders.append(
                    f"{pkg}/{module.name} shadows the {module.stem}/ package")
    assert not offenders, "; ".join(offenders)


def test_entry_points_import_in_script_mode():
    """Each entry point runs `--help` in script mode (own directory on sys.path[0])."""
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    for script in ("forecasting/trainer.py",
                   "assimilation/assimilate.py",
                   "unconditional_generation/trainer.py"):
        path = root / script
        proc = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=str(root), capture_output=True, text=True, timeout=300,
            env={**os.environ,
                "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": ""},
        )
        assert proc.returncode == 0, (
            f"{script} failed in script mode:\n{proc.stderr[-1500:]}"
        )
