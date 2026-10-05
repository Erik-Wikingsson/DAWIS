"""Tests for which `--forward_model` each method accepts, and for `_resolve_window`.

Unsupported method/propagator combinations must be refused up front with a clear error.
"""
import pytest

from assimilation.assimilate import _resolve_window, check_forward_model_supported
from assimilation.parser import parse_args
from assimilation.registry import FORWARD_MODELS, forward_model_frames


def _args(method, forward_model, **kw):
    argv = ["--method", method, "--forward_model", forward_model]
    for key, value in kw.items():
        argv += [f"--{key}", str(value)]
    return parse_args(argv)


class _FlowDAS:
    """Stands in for FlowDASModel: conditions on exactly 6 frames."""
    cond_frames = 6


class _Learned:
    """Stands in for LearnedForecaster built from a 1-state checkpoint."""
    def __init__(self, init_states=1):
        self.init_states = init_states


# Vocabulary
def test_every_dataset_offers_unet_fmw_and_none():
    for dataset, table in FORWARD_MODELS.items():
        assert {"unet", "fmw", "none"} <= set(table), dataset


def test_sevir_still_has_no_numerical_model():
    assert "numerical" not in FORWARD_MODELS["SEVIR"]


@pytest.mark.parametrize("method", ["DAISI", "EnSF", "LETKF", "DAWIS"])
@pytest.mark.parametrize("forward_model", ["unet", "flowdas"])
def test_filters_accept_the_learned_propagators(method, forward_model):
    check_forward_model_supported(_args(method, forward_model))


@pytest.mark.parametrize("method", ["DAISI", "EnSF", "LETKF"])
def test_fmw_with_a_path_is_open_to_every_filter(method):
    args = _args(method, "fmw", forward_model_path="/some/fmw.ckpt")
    check_forward_model_supported(args)


# Refusals
def test_fmw_without_a_path_is_dawis_only():
    """`fmw` without a checkpoint path is accepted only for DAWIS."""
    check_forward_model_supported(_args("DAWIS", "fmw"))
    with pytest.raises(ValueError, match="only DAWIS"):
        check_forward_model_supported(_args("LETKF", "fmw"))


@pytest.mark.parametrize("method", ["SDA", "GIBBS"])
def test_window_denoisers_take_no_forecast(method):
    check_forward_model_supported(_args(method, "none"))
    with pytest.raises(ValueError, match="takes no forecast"):
        check_forward_model_supported(_args(method, "unet"))


def test_a_refusal_names_what_to_use_instead():
    with pytest.raises(ValueError) as excinfo:
        check_forward_model_supported(_args("SDA", "flowdas"))
    assert "--forward_model none" in str(excinfo.value)


# Window vs. propagator depth
def test_frames_are_read_off_the_propagator():
    assert forward_model_frames(_FlowDAS()) == 6
    assert forward_model_frames(_Learned(1)) == 1
    assert forward_model_frames(_Learned(6)) == 6
    assert forward_model_frames(object()) == 1  # the SQG solver takes one state


def test_dawis_keeps_its_own_window_for_a_shallower_propagator(capsys):
    """DAWIS with --init_states 6 and a 1-state U-Net keeps a 6-deep window."""
    args = _args("DAWIS", "unet", init_states=6)
    window, start_time = _resolve_window(args, _Learned(1), 6, 6)
    assert (window, start_time) == (6, 6)
    assert "uses the newest 1" in capsys.readouterr().out


def test_dawis_refuses_a_propagator_that_needs_more_frames():
    args = _args("DAWIS", "flowdas", init_states=2)
    with pytest.raises(ValueError, match="conditions on 6 frames"):
        _resolve_window(args, _FlowDAS(), 2, 6)


def test_window_is_derived_for_a_method_that_does_not_pin_one():
    args = _args("LETKF", "flowdas")
    window, start_time = _resolve_window(args, _FlowDAS(), 1, 6)
    assert window == 6
    # The buffer is filled from the states before start_time.
    assert start_time >= 6


def test_start_time_is_raised_to_fit_the_buffer():
    args = _args("LETKF", "flowdas")
    _, start_time = _resolve_window(args, _FlowDAS(), 1, 2)
    assert start_time == 6


def test_an_explicit_window_that_is_replaced_warns(capsys):
    args = _args("LETKF", "unet", window=4)
    window, _ = _resolve_window(args, _Learned(1), 4, 6)
    assert window == 1
    assert "--window 4 replaced by 1" in capsys.readouterr().err
