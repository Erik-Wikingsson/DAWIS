"""Copy the released checkpoints into a staging directory, ready for upload.

Lightning checkpoints are reduced to the two keys `forecasting/ckpt_args.py`
reads (`state_dict` and `hyper_parameters`); optimizer state, loops and
callbacks are dropped. The training args are stored as a plain dict rather than
an `argparse.Namespace`. Bare state_dicts (the DAISI priors) are copied as they
are. Files keep their subpath under MODELS_ROOT, so the staged directory can be
used directly as MODELS_ROOT.

    python scripts/prepare_checkpoint_release.py --out /path/to/staging
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assimilation.checkpoints import FMW_MODELS, PRIORS  # noqa: E402
from data.paths import load_env_file  # noqa: E402

KEEP_KEYS = ("state_dict", "hyper_parameters")


def released_subpaths(windows: set[int]) -> list[str]:
    """DAISI priors, plus the eta01_channel window models for `windows`."""
    subpaths = [sub for variants in PRIORS.values() for _, sub in variants.values()]
    for dataset, losses in FMW_MODELS.items():
        per_window = losses["eta01_channel"]
        for w in sorted(windows):
            if w not in per_window:
                raise SystemExit(f"{dataset}: no eta01_channel model for init_states={w}")
            subpaths.append(per_window[w][1])
    return subpaths


def strip_lightning(ckpt: dict, dst: Path) -> None:
    out = {key: ckpt[key] for key in KEEP_KEYS}
    hp = dict(out["hyper_parameters"])
    if isinstance(hp.get("args"), argparse.Namespace):
        hp["args"] = vars(hp["args"])
    out["hyper_parameters"] = hp
    torch.save(out, dst)

    # The stripped file must load as plain data and keep every weight.
    reloaded = torch.load(dst, map_location="cpu", weights_only=True)
    assert reloaded["state_dict"].keys() == ckpt["state_dict"].keys()
    for key, value in ckpt["state_dict"].items():
        assert torch.equal(reloaded["state_dict"][key], value), key


def main() -> None:
    load_env_file()
    import os

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=os.environ.get("MODELS_ROOT"),
                        help="MODELS_ROOT holding the original checkpoints")
    parser.add_argument("--out", type=Path, required=True, help="staging directory")
    parser.add_argument("--windows", type=int, nargs="+", default=[6],
                        help="init_states of the window models to release")
    args = parser.parse_args()
    if args.src is None:
        parser.error("--src not given and MODELS_ROOT is not set")

    for sub in released_subpaths(set(args.windows)):
        src, dst = args.src / sub, args.out / sub
        if dst.exists():
            print(f"exists, skipping  {sub}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            strip_lightning(ckpt, dst)
        else:
            shutil.copy2(src, dst)
        print(f"{src.stat().st_size / 1e6:8.0f} MB -> {dst.stat().st_size / 1e6:6.0f} MB  {sub}")


if __name__ == "__main__":
    main()
