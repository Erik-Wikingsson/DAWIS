"""Tests for reading model architecture out of a checkpoint.

Unit tests on the merge use synthetic namespaces; the checkpoint tests need
files under $MODELS_ROOT and skip when they are absent.
"""
from argparse import Namespace

import pytest
import torch

from forecasting.ckpt_args import (
    ARCH_KEYS,
    CheckpointInfo,
    apply_checkpoint_arch,
    arch_from_checkpoint,
    merge_arch,
    read_checkpoint,
    resolved_model_cls,
)

# Paths under MODELS_ROOT.
FMW6 = ("DAWIS/models_SEVIR/"
        "SEVIR_FLOWDAS_SPLIT_lr_vil_eta_channel_init_6_flowdas-FMW-224-09_12_05-1441/"
        "last.ckpt")


@pytest.fixture(autouse=True)
def _contain_the_sevir_norm():
    """Restore `$SEVIR_NORM` and the data-source cache after each test.

    `--assim_norm flowdas` sets it process-wide, which would leak into other test files.
    """
    import os

    from data.registry import _get_cached

    before = os.environ.get("SEVIR_NORM")
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("SEVIR_NORM", None)
        else:
            os.environ["SEVIR_NORM"] = before
        _get_cached.cache_clear()


def _info(train_args, *, model=None, lightning=True):
    if train_args is not None and model is not None:
        train_args = {**train_args, "model": model}
    return CheckpointInfo(path="<test>", state_dict={}, train_args=train_args,
                          model=model, is_lightning=lightning)


# arch_from_checkpoint
def test_only_arch_keys_are_taken():
    """Only architecture keys are taken; sampling/guidance/training knobs are ignored."""
    info = _info({"hidden_dim": 224, "guide_method": None, "sampler": "heun",
                  "sevir_norm": "flowdas", "lr": 1e-3})
    arch = arch_from_checkpoint(info)
    assert arch == {"hidden_dim": 224}
    assert set(arch) <= set(ARCH_KEYS)


def test_absent_key_is_left_to_the_command_line():
    """A key missing from the checkpoint keeps its command-line value."""
    merged = merge_arch(Namespace(hidden_dim=32, alpha_beta_mult=2),
                        arch_from_checkpoint(_info({"hidden_dim": 128})),
                        label="t")
    assert merged.hidden_dim == 128
    assert merged.alpha_beta_mult == 2


def test_none_is_adopted_only_where_it_is_meaningful():
    arch = arch_from_checkpoint(_info({"alpha_beta_mult": None, "nx": None,
                                       "hidden_dim": 64}))
    assert "alpha_beta_mult" in arch and arch["alpha_beta_mult"] is None
    # A None nx would crash SongUNet.
    assert "nx" not in arch
    assert arch["hidden_dim"] == 64


def test_zero_and_false_are_real_values():
    """0.0 and False are adopted, not treated as missing."""
    arch = arch_from_checkpoint(_info({"channel_mult_emb": 0.0,
                                       "fm_forecast": False}))
    assert arch == {"channel_mult_emb": 0.0, "fm_forecast": False}


def test_bare_state_dict_yields_no_architecture():
    assert arch_from_checkpoint(_info(None, lightning=False)) == {}


# merge_arch
def test_base_args_are_never_mutated():
    """merge_arch returns a new namespace and leaves the base untouched."""
    base = Namespace(hidden_dim=32, init_states=6, channel_mult=[2, 2, 2])
    before = vars(base).copy()
    merged = merge_arch(base, {"hidden_dim": 64, "init_states": 1}, label="t")
    assert vars(base) == before
    assert (merged.hidden_dim, merged.init_states) == (64, 1)


def test_list_and_float_spelling_do_not_count_as_differences(capsys):
    """[2,2,2] vs (2,2,2) and 4 vs 4.0 are not reported as differences."""
    merge_arch(Namespace(channel_mult=[2, 2, 2], channel_mult_emb=4),
               {"channel_mult": (2, 2, 2), "channel_mult_emb": 4.0},
               label="t", defaults={"channel_mult": [9], "channel_mult_emb": 9})
    out = capsys.readouterr()
    assert "WARNING" not in out.err
    assert "channel_mult" not in out.out


def test_no_architecture_says_so(capsys):
    merge_arch(Namespace(hidden_dim=32), {}, label="t", path="/tmp/bare.pth")
    assert "no training args" in capsys.readouterr().out


# Reporting
def test_parser_default_is_filled_in_quietly(capsys):
    merge_arch(Namespace(hidden_dim=32), {"hidden_dim": 224}, label="DAWIS",
               defaults={"hidden_dim": 32}, path="/ckpt")
    out = capsys.readouterr()
    assert "WARNING" not in out.err
    assert "from checkpoint: hidden_dim=224" in out.out


def test_a_chosen_value_warns_on_stderr(capsys):
    merge_arch(Namespace(hidden_dim=128), {"hidden_dim": 224}, label="DAWIS",
               defaults={"hidden_dim": 32}, path="/ckpt")
    out = capsys.readouterr()
    assert "--hidden_dim 128 overridden by checkpoint -> 224" in out.err
    assert "1 command-line value(s) overridden" in out.err
    assert "--ignore_ckpt_arch" in out.err


def test_agreement_is_not_reported(capsys):
    merge_arch(Namespace(hidden_dim=224), {"hidden_dim": 224}, label="DAWIS",
               defaults={"hidden_dim": 32}, path="/ckpt")
    out = capsys.readouterr()
    assert "WARNING" not in out.err
    assert "hidden_dim" not in out.out


def test_quiet_keys_do_not_warn(capsys):
    """Overrides of keys listed in `quiet` are reported without a warning."""
    merge_arch(Namespace(init_states=6), {"init_states": 1}, label="propagator",
               defaults={"init_states": 1}, path="/ckpt",
               quiet={"init_states"})
    out = capsys.readouterr()
    assert "WARNING" not in out.err
    assert "init_states=1" in out.out


def test_ignore_ckpt_arch_keeps_the_command_line(capsys):
    base = Namespace(hidden_dim=128, ignore_ckpt_arch=True)
    merged = apply_checkpoint_arch(base, _info({"hidden_dim": 224}), label="t")
    assert merged.hidden_dim == 128
    assert "--ignore_ckpt_arch" in capsys.readouterr().err


# Model class
def test_model_class_comes_from_the_checkpoint():
    cls = resolved_model_cls(_info({"hidden_dim": 64}, model="UNET"),
                             Namespace(learned_model="fmw"))
    assert cls.__name__ == "UNET"


def test_learned_model_is_the_fallback_for_a_bare_state_dict():
    cls = resolved_model_cls(_info(None, lightning=False),
                             Namespace(learned_model="unet"))
    assert cls.__name__ == "UNET"


def test_unknown_recorded_model_is_named():
    with pytest.raises(ValueError, match="CVAE"):
        resolved_model_cls(_info({"hidden_dim": 8}, model="CVAE"), None)


# Real checkpoints
def _require(subpath):
    """The checkpoint's full path under MODELS_ROOT, or skip if it is absent."""
    import os

    from data.paths import lookup
    root = lookup("MODELS_ROOT")
    if not root:
        pytest.skip("MODELS_ROOT is not set")
    path = os.path.join(root, subpath)
    if not os.path.isfile(path):
        pytest.skip(f"checkpoint not on this filesystem: {path}")
    return path


def _sevir_args(**overrides):
    """A SEVIR run namespace with no architecture flags."""
    from assimilation.parser import parse_args

    args = parse_args(["--dataset", "SEVIR", "--variant", "lr_vil",
                       "--assim_norm", "flowdas"])
    args.device = torch.device("cpu")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _build(path, args):
    """Build the model this checkpoint describes and load it strictly."""
    from assimilation.parser import arch_defaults
    from forecasting.models.ar_model import apply_inference_defaults

    info = read_checkpoint(path)
    merged = apply_checkpoint_arch(args, info, label="t",
                                   defaults=arch_defaults())
    apply_inference_defaults(merged)
    model = resolved_model_cls(info, args)(merged)
    missing, unexpected = model.load_state_dict(info.state_dict, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model, merged


def test_checkpoint_architecture_loads_strictly():
    """With no architecture flags, the checkpoint loads with strict=True."""
    path = _require(FMW6)
    _build(path, _sevir_args())


@pytest.mark.parametrize("init_states,path", [
    (6, "SEVIR_FLOWDAS_SPLIT_lr_vil_eta_channel_init_6_flowdas-FMW-224-09_12_05-1441"),
])
def test_checkpoint_agrees_with_what_the_launchers_used_to_pass(init_states, path):
    """The recorded architecture equals the explicit launcher flags (`fmw_arch_args`)."""
    from assimilation.parser import parse_args

    full = _require("DAWIS/models_SEVIR/"
            f"{path}/last.ckpt")
    launcher = parse_args([
        "--dataset", "SEVIR", "--variant", "lr_vil", "--assim_norm", "flowdas",
        "--init_states", str(init_states),
        "--fm_loss", "eta01_channel",
        "--hidden_dim", str(32 * (init_states + 1)),
        "--alpha_beta_mult", "2",
        "--schedule", "linear_scalar",
        "--noise_embedding", "positional",
    ])
    launcher.nx = 128

    arch = arch_from_checkpoint(read_checkpoint(full))
    for key, recorded in arch.items():
        passed = getattr(launcher, key, None)
        if passed is None:
            # Not an assimilation-parser flag, so no launcher could set it.
            continue
        if isinstance(passed, (list, tuple)):
            passed, recorded = list(passed), list(recorded)
        assert passed == recorded, f"{key}: launcher={passed} checkpoint={recorded}"
