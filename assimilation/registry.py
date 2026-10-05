"""Dataset-keyed registries for observers, forward models and obs functions.

They live here rather than in `data/` to avoid a data -> forecasting import
cycle. A missing key raises `UnsupportedForDataset`.
"""

from __future__ import annotations

from typing import Callable

from assimilation.observers.observer import AvgObserver, GridObserver


class UnsupportedForDataset(RuntimeError):
    """The requested component does not exist for this dataset."""

    def __init__(self, dataset: str, kind: str, name: str, available):
        super().__init__(
            f"{kind} {name!r} is not available for dataset {dataset!r}. "
            f"Available: {sorted(available)}"
        )
        self.dataset, self.kind, self.name = dataset, kind, name


# Observers
def _build_grid_observer(md, args, device, *, obs_fn, obs_prob=None, **kw):
    return GridObserver(
        obs_fun=obs_fn,
        obs_prob=args.obs_prob if obs_prob is None else obs_prob,
        obs_sigma=args.obs_sigma,
        obs_mask=None,  # TODO: Add support for manual observation mask
        stationary_obs=args.fixed_obs,
        random_seed=42,
        nx=md.nx,
        ny=md.ny,
        device=device,
        radar=bool(getattr(args, "radar", False)),
        radar_width=getattr(args, "radar_width", 4) or 4,
        radar_dx=getattr(args, "radar_dx", 4) or 4,
        num_channels=md.num_channels,
        unit_factor=md.unit_factor,
        **kw,
    )


def _build_avg_observer(md, args, device, *, obs_fn=None, obs_prob=None, **kw):
    return AvgObserver(
        obs_fun=None,
        obs_prob=args.obs_prob if obs_prob is None else obs_prob,
        obs_sigma=args.obs_sigma,
        obs_mask=None,  # TODO: Add support for manual observation mask
        stationary_obs=args.fixed_obs,
        random_seed=42,
        nx=md.nx,
        ny=md.ny,
        device=device,
        lres=args.lres,
        num_channels=md.num_channels,
        unit_factor=md.unit_factor,
        **kw,
    )


# dataset -> observation-network kind -> factory
OBSERVERS: dict[str, dict[str, Callable]] = {
    "SQG": {"grid": _build_grid_observer, "avg": _build_avg_observer},
    "SEVIR": {"grid": _build_grid_observer, "avg": _build_avg_observer},
}


def observer_kind(args) -> str:
    return "avg" if getattr(args, "obs_fn", None) == "avg" else "grid"


def build_observer(args, md, device, *, obs_prob=None):
    """Build (observer, obs_fn) for a dataset. `obs_prob` overrides --obs_prob."""
    from assimilation.observers.obsop import OBS_FNS

    table = OBSERVERS.get(md.name)
    if table is None:
        raise UnsupportedForDataset(md.name, "observer", "any", OBSERVERS)
    kind = observer_kind(args)
    if kind not in table:
        raise UnsupportedForDataset(md.name, "observer", kind, table)

    if kind == "avg":
        observer = table[kind](md, args, device, obs_prob=obs_prob)
        return observer, observer.obs_fn

    name = args.obs_fn
    if name not in OBS_FNS:
        raise UnsupportedForDataset(md.name, "obs_fn", name, OBS_FNS)
    obs_fn = OBS_FNS[name]

    # The radar network's density is set by its width, not by --obs_prob.
    prob = obs_prob
    if prob is None and getattr(args, "radar", False):
        prob = args.radar_width / md.nx

    observer = table[kind](md, args, device, obs_fn=obs_fn, obs_prob=prob)
    return observer, obs_fn


# Forward models
def _numerical(md, args, case, init_states, *, gpu=False):
    from forecasting.sqg_model import SQGModel, SQGModelGPU

    if not case.has_numerical_model:
        raise UnsupportedForDataset(
            md.name, "forward_model", "numerical_gpu" if gpu else "numerical",
            ["fmw", "dawis", "none"],
        )
    cls = SQGModelGPU if gpu else SQGModel
    return cls(args, init_states)


def _flowdas(md, args, case, init_states):
    """FlowDAS's pretrained interpolant as an unguided one-step forecaster."""
    from forecasting.flowdas_forecaster import FlowDASModel

    return FlowDASModel(args.device, dataset=md.name, md=md,
                        forward_norm=getattr(args, "forward_norm", None))


def _learned(md, args, case, init_states):
    """A trained forecasting checkpoint as a standalone one-step forecaster.

    Used for `unet`, `learned` and `fmw` with a path; the architecture comes
    from the checkpoint. `fmw` is stochastic, `unet` deterministic.
    """
    from forecasting.learned_forecaster import LearnedForecaster

    return LearnedForecaster(args, md, args.device,
                             label=f"propagator ({args.forward_model})")


def _fmw(md, args, case, init_states):
    """An FMW window model as the propagator.

    With `--forward_model_path` it is a learned forecaster usable by any
    method. Without one, returns None: the method (DAWIS/SDA) is its own
    forward model, which is required for `--guide_forecast`.
    """
    if getattr(args, "forward_model_path", None):
        return _learned(md, args, case, init_states)
    return None


#: dataset -> forward-model name -> factory. `None` means the assimilation
#: method provides the forecast. `learned` is an alias of `unet`.
FORWARD_MODELS: dict[str, dict[str, Callable | None]] = {
    "SQG": {
        "numerical": lambda md, a, c, x: _numerical(md, a, c, x, gpu=False),
        "numerical_gpu": lambda md, a, c, x: _numerical(md, a, c, x, gpu=True),
        "flowdas": _flowdas,
        "learned": _learned,
        "unet": _learned,
        "fmw": _fmw,
        "dawis": None,
        "none": None,
    },
    # SEVIR has no numerical dynamics; LETKF needs a learned propagator.
    "SEVIR": {
        "learned": _learned,
        "unet": _learned,
        "fmw": _fmw,
        # Unguided; `--method FlowDAS` is the guided version (--forward_model none).
        "flowdas": _flowdas,
        "dawis": None,
        "none": None,
    },
}

#: Forward models that build a standalone propagator object.
LEARNED_FORWARD_MODELS = frozenset({"learned", "unet", "fmw"})


def build_forward_model(args, md, case, init_states):
    table = FORWARD_MODELS.get(md.name)
    if table is None:
        raise UnsupportedForDataset(md.name, "forward_model", "any", FORWARD_MODELS)
    name = getattr(args, "forward_model", "none")
    if name not in table:
        raise UnsupportedForDataset(md.name, "forward_model", name, table)
    factory = table[name]
    return None if factory is None else factory(md, args, case, init_states)


def forward_model_frames(forward_model) -> int:
    """How many conditioning frames a built propagator needs.

    `cond_frames` for FlowDAS, `init_states` for learned forecasters, else 1.
    """
    for attr in ("cond_frames", "init_states"):
        value = getattr(forward_model, attr, None)
        if value:
            return int(value)
    return 1


def require_numerical_model(args, md, case, method: str) -> None:
    """Guard for methods that need the numerical model and its .nc parameters.

    Without numerical dynamics, a learned propagator is accepted instead.
    """
    if case.has_numerical_model:
        return
    # Any real forecast operator will do; `fmw` needs a checkpoint path.
    substitutes = {"learned", "unet", "fmw", "flowdas"}
    name = getattr(args, "forward_model", None)
    if name in substitutes and (name != "fmw"
                                or getattr(args, "forward_model_path", None)):
        return
    raise UnsupportedForDataset(
        md.name, "method", method,
        ["DAISI", "DAWIS", "EnSF",
         *(f"or --forward_model {s}" for s in sorted(substitutes))],
    )
