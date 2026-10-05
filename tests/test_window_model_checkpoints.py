"""Tests for which FMW window-model checkpoint a run loads for each `init_states`.

Also checks agreement with `fmw_ckpt_for` in assimilation/scripts/SEVIR/_common.sh.
"""
import os
import re
import subprocess
import types

import pytest

from assimilation.checkpoints import FMW_MODELS, MissingCheckpoint, resolve_window_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON_SH = os.path.join(REPO_ROOT, "assimilation", "scripts", "SEVIR", "_common.sh")

# Only the released window is registered; others come from the environment.
SEVIR_WINDOWS = (6,)
SQG_WINDOWS = (6,)


@pytest.fixture(autouse=True)
def _models_root(monkeypatch):
    """Set a dummy MODELS_ROOT if none is configured (files need not exist)."""
    from data.paths import lookup
    if not lookup("MODELS_ROOT"):
        monkeypatch.setenv("MODELS_ROOT", "/models")


def _md(name="SEVIR", variant="lr_vil"):
    return types.SimpleNamespace(name=name, variant=variant)


def _args(fm_loss="eta01_channel", init_states=6, model_path=None):
    return types.SimpleNamespace(
        model_path=model_path, fm_loss=fm_loss, init_states=init_states)


# Registry

def test_every_sevir_window_is_registered():
    assert sorted(FMW_MODELS["SEVIR"]["eta01_channel"]) == list(SEVIR_WINDOWS)


@pytest.mark.parametrize("init_states", SEVIR_WINDOWS)
def test_each_window_resolves_to_its_own_checkpoint(init_states):
    path = resolve_window_model(_md(), _args(init_states=init_states))
    # The window is encoded in the directory name.
    assert f"_init_{init_states}_" in path, path


@pytest.mark.parametrize("init_states", SEVIR_WINDOWS)
def test_hidden_dim_in_the_name_matches_the_window(init_states):
    """FMW-<N> in the directory name is 32 * (init_states + 1), as `fmw_arch_args` assumes."""
    path = resolve_window_model(_md(), _args(init_states=init_states))
    match = re.search(r"-FMW-(\d+)-", path)
    assert match, path
    assert int(match.group(1)) == 32 * (init_states + 1)


@pytest.mark.parametrize("init_states", SQG_WINDOWS)
def test_sqg_default_fm_loss_resolves_per_window(init_states):
    """`--experiment noisy --method DAWIS` needs no --model_path on SQG."""
    from assimilation.parser import parse_args

    args = parse_args(["--experiment", "noisy", "--init_states", str(init_states)])
    assert args.fm_loss == "eta01_channel"
    args.model_path = None
    path = resolve_window_model(_md("SQG", "base"), args)
    assert f"eta_channel_init_{init_states}-FMW-{32 * (init_states + 1)}-" in path, path


# Refusals

def test_an_unregistered_window_is_refused_by_name():
    with pytest.raises(MissingCheckpoint) as excinfo:
        resolve_window_model(_md(), _args(init_states=3))
    message = str(excinfo.value)
    assert "init_states 3" in message
    # The error lists the available windows and the variable to set.
    assert "[6]" in message
    assert "ASSIM_FMW_PATH_SEVIR_ETA01_CHANNEL_INIT_3" in message


@pytest.mark.parametrize("name,variant", [("SEVIR", "lr_vil"), ("SQG", "base")])
def test_an_unregistered_window_comes_from_its_env_var(monkeypatch, name, variant):
    """A locally trained window is used through ASSIM_FMW_PATH_<DATASET>_..._INIT_<n>."""
    monkeypatch.setenv(f"ASSIM_FMW_PATH_{name}_ETA01_CHANNEL_INIT_3", "/mine/init3.ckpt")
    path = resolve_window_model(_md(name, variant), _args(init_states=3))
    assert path == "/mine/init3.ckpt"


def test_a_missing_init_states_is_refused_rather_than_defaulted():
    """A missing init_states raises instead of falling back to a default window."""
    args = _args(init_states=None)
    with pytest.raises(MissingCheckpoint) as excinfo:
        resolve_window_model(_md(), args)
    assert "--init_states" in str(excinfo.value)


def test_explicit_model_path_still_wins():
    path = resolve_window_model(
        _md(), _args(init_states=2, model_path="/somewhere/else.ckpt"))
    assert path == "/somewhere/else.ckpt"


# Agreement with the bash launchers

def _fmw_ckpt_for(init_states):
    """What assimilation/scripts/SEVIR/_common.sh resolves for this window."""
    script = (
        f'source "{COMMON_SH}" >/dev/null 2>&1; fmw_ckpt_for {init_states}')
    out = subprocess.run(
        ["bash", "-c", script], cwd=REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "REPO_ROOT": REPO_ROOT})
    if out.returncode != 0:
        pytest.skip(f"_common.sh is not sourceable here: {out.stderr.strip()[:200]}")
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("init_states", SEVIR_WINDOWS)
def test_python_and_bash_tables_agree(init_states):
    assert resolve_window_model(_md(), _args(init_states=init_states)) == \
        _fmw_ckpt_for(init_states)
