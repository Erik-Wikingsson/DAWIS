"""Pinned SQG metadata, statistics and observations that published results depend on.

Values are literals on purpose: changing any of them must be deliberate.
"""

import pathlib

import numpy as np
import pytest
import torch

from data.registry import get_data_source

# Pinned values
SCALEFACT = 0.003061224412462883
LEGACY_MEAN = 0.0
LEGACY_STD = 2660.0
PER_CHANNEL_MEAN = [-0.00041767506627365947, -0.00041766747017391026]
PER_CHANNEL_STD = [2663.8447265625, 2665.251708984375]
DIFF_MEAN = [5.7777649851986634e-12, 4.023168083400197e-12]
DIFF_STD = [0.47019970417022705, 0.46922051906585693]
VAR_NAMES = ["Level 0", "Level 1"]


@pytest.fixture(scope="module")
def src():
    return get_data_source("SQG", "base")


@pytest.fixture(scope="module")
def md(src):
    return src.metadata


def _needs_data(src):
    ok, why = src.is_available()
    if not ok:
        pytest.skip(f"SQG data not present: {why}")


# Metadata (no data files needed)
def test_grid_and_channels(md):
    assert md.grid == (64, 64)
    assert md.num_channels == 2
    assert md.variable_names == VAR_NAMES
    assert md.cadence_hours == 3.0
    assert md.periodic == (True, True)
    assert md.max_window == 100, (
        "max_window feeds trajectories_per_file and therefore epoch size; "
        "changing it changes training"
    )


def test_legacy_scalar_normalization(md):
    """The profile every DAISI checkpoint was trained under."""
    assert md.norm.legacy_scalar_mean == LEGACY_MEAN
    assert md.norm.legacy_scalar_std == LEGACY_STD


def test_plot_range_is_physical_units(md):
    assert md.plot_range(0) == (-20.0, 20.0)
    assert md.plot_range(1) == (-20.0, 20.0)


def test_scalefact_is_bit_exact(md):
    """unit_factor derived from the .nc equals the pinned literal exactly.

    It differs slightly from the naive 1e-4*300/9.8 because nc.f is stored as float32.
    """
    assert md.unit_factor == SCALEFACT
    naive = 1e-4 * 300 / 9.8
    assert naive != SCALEFACT, "guard: the naive formula is genuinely different"


# Statistics and datasets (need data)
def test_per_channel_stats(src, md):
    _needs_data(src)
    assert md.norm.state_mean.tolist() == PER_CHANNEL_MEAN
    assert md.norm.state_std.tolist() == PER_CHANNEL_STD
    assert md.norm.diff_mean.tolist() == DIFF_MEAN
    assert md.norm.diff_std.tolist() == DIFF_STD


def test_two_profiles_stay_distinct(src, md):
    """They differ by ~0.15%; collapsing them would invalidate checkpoints."""
    _needs_data(src)
    assert md.norm.legacy_scalar_std != md.norm.state_std[0].item()
    assert abs(md.norm.state_std[0].item() / md.norm.legacy_scalar_std - 1) < 0.01


def test_physical_std_bridge(src, md):
    """`scale` = std * unit_factor, the value assimilation applies."""
    _needs_data(src)
    assert md.physical_std(per_channel=False)[0].item() == pytest.approx(
        LEGACY_STD * SCALEFACT, rel=1e-6)


def test_hires_refuses_missing_per_channel_stats():
    """No .pt stats exist at nx=256: asking must raise, not silently fall back."""
    hires = get_data_source("SQG", "hires")
    assert not hires.has_per_channel_stats()
    with pytest.raises(FileNotFoundError):
        hires.require_per_channel_stats()


def test_data_index_mapping_is_stable(src):
    """--data_index maps to the sorted test files: 0 = tuning trajectory, 1-10 = evaluation.

    Index 10 is named `_x10_` so that sorting keeps indices 0-9 in place.
    """
    _needs_data(src)
    files = src.list_trajectories("test")
    assert len(files) == 11
    assert files == sorted(files)
    assert files[0].name == "sqg_N64_3hrly_steps_110_0_c874.npy"
    # The `_x10_` trajectory is --data_index 10.
    assert files[10].name.startswith("sqg_N64_3hrly_steps_110_x10_")
    # Indices 0-9 map to versions 0-9.
    for i in range(10):
        assert files[i].name.startswith(f"sqg_N64_3hrly_steps_110_{i}_")


def test_state_dataset_contract(src, md):
    _needs_data(src)
    ds = src.single_state("val")
    x = ds[0]
    assert x.shape == md.state_shape
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()


def test_forecast_dataset_epoch_size_unchanged(src):
    """Epoch size is trajectories_per_file = 100 // sample_length."""
    _needs_data(src)
    for init_states, pred_length, expected in [
        (2, 1, 33), (1, 1, 50), (2, 3, 20), (0, 1, 100),
    ]:
        ds = src.forecast_window("val", init_states=init_states,
                                 pred_length=pred_length)
        assert len(ds) == expected, (
            f"init_states={init_states} pred_length={pred_length}")


def test_forecast_window_rejects_oversized_request(src, md):
    _needs_data(src)
    with pytest.raises(ValueError, match="max_window"):
        src.forecast_window("val", init_states=2, pred_length=200)


def test_assim_trajectory_is_raw_units(src, md):
    """Raw PV, NOT normalized: the methods apply unit_factor themselves."""
    _needs_data(src)
    case = src.assim_trajectory(split="test", index=0)
    x = case.states[0]
    assert x.shape == md.state_shape
    assert np.abs(x).max() > 100.0, "looks normalized; should be raw PV"
    assert case.has_numerical_model
    assert case.model_params["scalefact"] == SCALEFACT
    assert case.domain_size_m == (20000000.0, 20000000.0)


def test_observer_is_unchanged(md):
    """GridObserver reproduces the pinned SQG observations for a fixed seed."""
    from assimilation.observers.obsop import OBS_FNS
    from assimilation.observers.observer import GridObserver

    rng = np.random.RandomState(7)
    x = rng.randn(2, 64, 64) * 3000.0
    obs, mask, sigma = GridObserver(
        obs_fun=OBS_FNS["linear"], obs_prob=0.25, obs_sigma=5.0,
        stationary_obs=True, random_seed=42, nx=64, ny=64, device="cpu",
        num_channels=md.num_channels, unit_factor=md.unit_factor,
    ).observe(x, t=0)
    assert mask.shape == (1, 2, 64, 64)
    assert obs.shape == (1, 2 * 1024)
    assert sigma == 5.0
    # Pinned reference values for this seed.
    assert float(obs[0, 0]) == pytest.approx(2.913081169128418, abs=1e-6)
    assert float(obs[0, -1]) == pytest.approx(-19.340652465820312, abs=1e-6)


def test_letkf_scalefact_keeps_float64_promotion():
    """`LETKF.scalefact` is a np.float64 so that, under NEP 50, dividing float32 data promotes to float64."""
    import numpy as np

    probe32 = np.ones((2, 2), dtype=np.float32)
    assert (probe32 / np.float64(3.0)).dtype == np.float64
    assert (probe32 / 3.0).dtype == np.float32, "NEP 50 weak-scalar rule changed"

    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "assimilation" / "methods" / "letkf.py"
    ).read_text()
    assert 'self.scalefact = np.float64(' in source, (
        "LETKF.scalefact must be built with np.float64(...) to preserve "
        "float64 promotion in assimilate()"
    )


def test_physical_std_scalar_profile_is_float64():
    """The scalar physical_std (2660 * scalefact) is computed in float64."""
    md = get_data_source("SQG", "base").metadata
    scale = md.physical_std(per_channel=False)
    assert scale.dtype == torch.float64
    assert float(scale[0]) == LEGACY_STD * SCALEFACT
