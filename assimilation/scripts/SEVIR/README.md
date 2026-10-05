# Assimilation on SEVIR

SEVIR-LR: 1 channel (VIL, 0-255), 128x128, 25 frames per event at a 10-minute
cadence (see `data/README.md` for obtaining the data). There is one
experiment, `sevir` (the DAISI paper's observation network: linear
observations of 10% of the pixels, noise 0.255 VIL counts, start from the true
state). It is defined in `_common.sh` and locked by the `sevir` preset in
`assimilation/experiments.py`, which refuses contradicting `OBS_*` / `INIT_*`
values.

## How the launchers work

Each method has a launcher `run_all_<METHOD>.sh` and a worker
`run_<METHOD>.sh`. The launcher sources `_launcher.sh`, loops over the arrays at
its top and submits the worker with the settings as environment variables; the
worker sources `_common.sh` and prints its resolved configuration into the
SLURM log. Any variable exported in your shell reaches every job, so
`OBS_SIGMA=...` or `N_ENS=...` overrides a whole sweep (and a stale export
changes it silently).

```bash
DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_DAWIS.sh --sweep false    # print the plan
DATA_INDICES="1 2 3 4 5 6 7 8 9 10" \
    bash assimilation/scripts/SEVIR/run_all_DAWIS.sh --sweep false          # submit
```

`DRY_RUN`, `--sweep true|false`, `DATA_INDICES`, `INIT_STATES_VALUES` and
`ASSIM_INTERVAL` behave as described in `../README.md`. SEVIR-specific knobs:

| variable | default | effect |
| --- | --- | --- |
| `NODE_TYPE` | per launcher | passed as `sbatch -C`: `fat` (80 GB A100) for the window methods and FlowDAS; `thin` (40 GB) for LETKF, EnSF, DAISI |
| `TIME_LIMIT` | per launcher | wall-clock limit (most launchers) |
| `N_ENS`, `BATCH_SIZE` | 20, per launcher | ensemble size; members per GPU call (memory only) |
| `START_TIME`, `N_TIMES` | 6, 19 | `START_TIME + N_TIMES` must fit in the 25-frame event |
| `LOG_MEDIA` | `false` (EnSF: `true`) | upload wandb videos and field plots |
| `ONLINE_METRICS` | `true` | stream per-step metrics to wandb (filters only) |
| `WANDB_MODE`, `WANDB_PROJECT` | `online`, `ScoreDA_SEVIR` | wandb settings |

## Trajectories

`--data_index` indexes a seeded 11-event sample of the `test` split: index 0 is
the tuning trajectory, 1-10 the evaluation set. The launchers ship different
lists: `(1 2 3 4 5 6 7 8 9 10)` in LETKF, EnSF, DAISI, FlowDAS,
GuidedForecast, ForcingDAS and Gibbs; `(0)` in DAWIS, SDA and SDA_filter. Set `DATA_INDICES` explicitly rather than
relying on the shipped list.

## Three normalizations

Each launcher sets three knobs, all overridable from the environment:

| variable | meaning |
| --- | --- |
| `ASSIM_NORM` | units the run computes in: state, observations, `--obs_sigma`, `--init_std`. Must match the network that does the assimilation step. |
| `FORWARD_NORM` | units the forward-propagator checkpoint was trained in; the forecaster converts to and from it. |
| `OUTPUT_NORM` | units of the saved `.nc` and every logged metric: `physical` (VIL counts, the default), `assim`, or a mode name such as `01` (vil/255, the unit of the DAISI and FlowDAS papers). |

Modes: `01` = vil/255, `flowdas` = `(vil/255 - 0.5)/0.1`, `standard` =
`(vil - 33.44)/47.54`. The pretrained SEVIR window models and the FlowDAS
checkpoint use `flowdas`; the DAISI prior was trained on `01`. The launchers
therefore use `ASSIM_NORM=flowdas` for the window-model methods (DAWIS,
GuidedForecast, ForcingDAS, SDA, SDA_filter, Gibbs) and `ASSIM_NORM=01` for
DAISI, FlowDAS, LETKF and EnSF, with
`FORWARD_NORM=flowdas` throughout. Each result file records its units in the
`output_norm` attribute.

## Forward models

SEVIR has no numerical model, so methods that propagate an ensemble use a
learned one, chosen with `FORWARD_MODEL` (LETKF: `PROPAGATOR`):

| value | propagator | used by |
| --- | --- | --- |
| `flowdas` | FlowDAS's pretrained interpolant; stochastic; needs a 6-frame window | DAWIS Filter, DAISI, LETKF, EnSF (defaults) |
| `unet` | deterministic U-Net (`UNET_CKPT`, trained with `forecasting/trainer.py --model UNET`); adds no spread | optional |
| `fmw` | an FMW window model as propagator (`FMW_CKPT`, or the checkpoint for `FWD_INIT_STATES`) | optional |
| `dawis` | the window model generates the new state itself | DAWIS-Joint (`run_DAWIS.sh` only) |
| `self` | the run's own window model as a separate forecast step | `run_DAWIS.sh` only |
| `none` | no forecast | GuidedForecast, ForcingDAS, SDA_filter, SDA, Gibbs |

With the zero-spread start of the `sevir` preset, LETKF and EnSF refuse a
deterministic propagator.

## Checkpoints

Resolved under `MODELS_ROOT` by `_common.sh` (shell) and
`assimilation/checkpoints.py` (Python). A propagator's architecture is read
from its checkpoint; the window model's flags are derived by `fmw_arch_args` in
`_common.sh`. Per-run overrides:

| variable | model | default under `MODELS_ROOT` |
| --- | --- | --- |
| `FMW_CKPT` | FMW window model, all windows (`ASSIM_FMW_PATH_SEVIR_ETA01_CHANNEL_INIT_<n>` for one window) | `DAWIS/models_SEVIR/..._init_6_flowdas-FMW-.../last.ckpt` |
| `UNET_CKPT` | U-Net forecaster | none (not released; train with `forecasting/trainer.py --model UNET`) |
| `PRIOR_CKPT` (or `SEVIR_PRIOR_PATH`) | DAISI prior, 128x128 | `SQG/models/daisi/daisi_sevir_128.pth` |
| `FLOWDAS_CKPT` (or `FLOWDAS_SEVIR_MODEL_PATH`) | FlowDAS SEVIR checkpoint (not in the Hugging Face download; get FlowDAS's `latest.pt`, see the top-level README) | `SQG/models/flowdas/flowdas_sevir.pt` |

A window model is released for `init_states` 6 (`hidden_dim = 32 *
(init_states + 1)`). Train others with
`forecasting/training_scripts/SEVIR/train_all_fmw.sh` and set
`ASSIM_FMW_PATH_SEVIR_ETA01_CHANNEL_INIT_<n>`.

## Methods

| paper method | launcher | notes |
| --- | --- | --- |
| DAWIS Filter / Lagged Smoother | `run_all_DAWIS.sh` | `FORWARD_MODEL=flowdas` |
| DAWIS-Joint Filter / Lagged Smoother | `run_all_DAWIS.sh` | `FORWARD_MODEL=dawis`; `--sweep false` runs both |
| Joint AR 1\|W, W\|1 | `run_all_GuidedForecast.sh` | context modes `past`, `future` |
| SDA-Filter | `run_all_SDA_filter.sh` | |
| ForcingDAS-Pyr | `run_all_ForcingDAS.sh` | |
| SDA | `run_all_SDA.sh` | even `init_states` |
| Block Gibbs / DAWIS Block / Soft Block Smoother | `run_all_Gibbs.sh` | variants `Gibbs` / `DAWIS` / `Soft` (`Soft` is not in the default list) |
| DAISI, FlowDAS, LETKF, EnSF | `run_all_DAISI.sh`, `run_all_FlowDAS.sh`, `run_all_LETKF.sh`, `run_all_EnSF.sh` | published settings; no sweep |

GuidedForecast, ForcingDAS, SDA_filter and the DAWIS launchers all submit
`run_DAWIS.sh`.

Notes:

* **Gibbs** refines a finished run: `TRAJECTORY_PATH` names a result `.nc`, or
  `INITIAL_TRAJECTORY` (default `DAWISdawis`) is looked up in `PAPER_RUNS_ROOT`
  as `<METHOD>_SEVIR_run_<index>_<suffix>.nc`.
* **EnSF** needs `GUIDANCE_STRENGTH=0.01` (the launcher's default); larger
  values make the explicit reverse SDE unstable.
* **FlowDAS**: its checkpoint was trained on the pre-2019-06-01 pool, so report
  `test` results only.
