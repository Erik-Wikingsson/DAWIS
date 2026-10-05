"""Tests for `DAWIS._step_tmin` and the `--tmin_init` schedule.

Without `--tmin_init` the per-slot depths equal `tmin`; with it, the last entry
tracks the slot holding the initial condition. Instances are built with `__new__`.
"""
import pytest
import torch

from assimilation.methods.dawis import DAWIS


def _model(tmin, tmin_init, init_states):
    """A DAWIS with only the attributes `_step_tmin` touches (no checkpoint load)."""
    m = DAWIS.__new__(DAWIS)
    m.tmin = tmin
    m.tmin_init = tmin_init
    m.init_states = init_states
    return m


# tmin_init absent

@pytest.mark.parametrize("tmin, init_states", [
    ([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], 6),   # linspace-shaped
    ([0.3] * 7, 6),                             # flat
    ([1.0] + [0.3] * 6, 6),                     # clean context + inverted target
    ([0.3, 0.3], 1),                            # degenerate single-state window
])
def test_no_tmin_init_is_identical_to_tmin(tmin, init_states):
    """With tmin_init=None every step returns `tmin`."""
    m = _model(tmin, None, init_states)
    for ntime in range(20):  # well past the window
        assert m._step_tmin(ntime) == tmin, f"ntime={ntime}"


def test_no_tmin_init_with_tensor_tmin():
    """A tensor `tmin` is returned as a list of floats with the same values."""
    tmin = torch.linspace(0.9, 0.3, 7)
    m = _model(tmin, None, 6)
    out = m._step_tmin(0)
    assert all(isinstance(t, float) for t in out)
    assert out == pytest.approx(tmin.tolist())


# tmin_init present

# tmin_init -> expected step_tmin for ntime = 0 ... w, with tmin flat at 0.3.
TABLES = {
    # scenario 1: pin the IC, standard depth for the noise-fill slots
    (0.3, 0.3, 0.3, 0.3, 0.3, 1.0): [
        [0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 0.3],
        [0.3, 0.3, 0.3, 0.3, 1.0, 0.3, 0.3],
        [0.3, 0.3, 0.3, 1.0, 0.3, 0.3, 0.3],
        [0.3, 0.3, 1.0, 0.3, 0.3, 0.3, 0.3],
        [0.3, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3],
        [1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],  # ntime = w: IC has left the window
    ],
    # scenario 2: pin the IC, fully re-invert the noise-fill slots
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0): [
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.3],
        [0.0, 0.0, 0.0, 1.0, 0.3, 0.3, 0.3],
        [0.0, 0.0, 1.0, 0.3, 0.3, 0.3, 0.3],
        [0.0, 1.0, 0.3, 0.3, 0.3, 0.3, 0.3],
        [1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    ],
    # scenario 3: intermediate depths, IC slot shallow but NOT pinned
    (0.1, 0.1, 0.1, 0.1, 0.1, 0.9): [
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.3],
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.3, 0.3],
        [0.1, 0.1, 0.1, 0.9, 0.3, 0.3, 0.3],
        [0.1, 0.1, 0.9, 0.3, 0.3, 0.3, 0.3],
        [0.1, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.9, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    ],
}


@pytest.mark.parametrize("tmin_init, expected", TABLES.items())
def test_step_tmin_table(tmin_init, expected):
    m = _model([0.3] * 7, list(tmin_init), 6)
    for ntime, row in enumerate(expected):
        assert m._step_tmin(ntime) == pytest.approx(row), f"ntime={ntime}"
    assert m._step_tmin(9) == pytest.approx([0.3] * 7)  # standard forever after


@pytest.mark.parametrize("tmin_init", TABLES.keys())
def test_ic_entry_walks_to_slot_zero(tmin_init):
    """The last tmin_init entry is at slot w - 1 - ntime while the IC is in the window."""
    w = 6
    ic_depth = tmin_init[-1]
    m = _model([0.3] * (w + 1), list(tmin_init), w)
    for ntime in range(w):
        assert m._step_tmin(ntime)[w - 1 - ntime] == pytest.approx(ic_depth)


def test_target_slot_is_never_overridden():
    """The target slot always keeps tmin[-1]."""
    tmin = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.42]
    m = _model(tmin, [0.0] * 5 + [1.0], 6)
    for ntime in range(12):
        assert m._step_tmin(ntime)[-1] == pytest.approx(0.42), f"ntime={ntime}"


def test_tensor_tmin_composes_with_tmin_init():
    """A tensor tmin combines with a list tmin_init and yields plain floats."""
    tmin = torch.linspace(0.9, 0.3, 7)  # as assimilate.py builds it
    m = _model(tmin, [0.0] * 5 + [1.0], 6)
    out = m._step_tmin(2)
    # No 0-d tensors may reach _make_schedules.
    assert all(isinstance(t, float) for t in out)
    # Slots 0..2 from tmin_init, slot 3 the IC, slots 4..6 the untouched linspace tail.
    assert out == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.5, 0.4, 0.3])


def test_step_tmin_does_not_mutate_stored_schedules():
    """_step_tmin does not alias or mutate self.tmin / self.tmin_init."""
    tmin = [0.3] * 7
    tmin_init = [0.0] * 5 + [1.0]
    m = _model(tmin, tmin_init, 6)
    m._step_tmin(0)[0] = 0.123
    m._step_tmin(3)
    assert m.tmin == [0.3] * 7
    assert m.tmin_init == [0.0] * 5 + [1.0]


# Validation

def test_validate_tmin_init_none_is_disabled():
    assert DAWIS._validate_tmin_init(None, 6) is None


@pytest.mark.parametrize("value", [
    [0.3] * 7,      # length init_states + 1
    [0.3] * 5,
    [],
])
def test_validate_tmin_init_rejects_wrong_length(value):
    with pytest.raises(ValueError, match="length"):
        DAWIS._validate_tmin_init(value, 6)


def test_validate_tmin_init_rejects_scalar():
    with pytest.raises(ValueError, match="per-slot list"):
        DAWIS._validate_tmin_init(0.3, 6)


@pytest.mark.parametrize("value", [
    [0.3, 0.3, 0.3, 0.3, 0.3, 1.5],
    [-0.1, 0.3, 0.3, 0.3, 0.3, 1.0],
])
def test_validate_tmin_init_rejects_out_of_range(value):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DAWIS._validate_tmin_init(value, 6)


def test_validate_tmin_init_accepts_tensor_and_returns_floats():
    out = DAWIS._validate_tmin_init(torch.tensor([0.1, 0.1, 0.9]), 3)
    assert out == pytest.approx([0.1, 0.1, 0.9])
    assert all(isinstance(t, float) for t in out)
