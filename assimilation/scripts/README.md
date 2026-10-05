# Assimilation launchers

SLURM scripts that run `assimilation/assimilate.py` (filters and fixed-lag
smoothers) and `assimilation/smoothing.py` (SDA, block smoothers) for the
experiments in the paper. One directory per dataset:

```
assimilation/scripts/
  SQG/                        SQG launchers (.bash)
  SEVIR/                      SEVIR launchers (.sh); see SEVIR/README.md
  _sweep_arg.sh               --sweep true|false
  _dry_run.sh                 DRY_RUN=1
  _data_indices.sh            DATA_INDICES
  _init_states.sh             INIT_STATES_VALUES
```

## How they work

A **launcher** (`run_all_<METHOD>.bash` / `.sh`) loops over experiments,
trajectories and hyperparameters and submits one `sbatch` job per combination
to a **worker**: `SQG/run_experiment_<METHOD>.bash` (the launcher passes a
`--flag` list) or `SEVIR/run_<METHOD>.sh` (the launcher passes environment
variables). The worker activates the conda environment and runs python.

Everything machine-specific comes from the repo `.env` (template:
`.env.example`), loaded by `scripts/load_env.sh`: data roots, `MODELS_ROOT`
for the pretrained checkpoints (subpaths in `assimilation/checkpoints.py`,
`SQG/_models.sh` and `SEVIR/_common.sh`), `RESULTS_ROOT` for the output
`<exp_name>.nc` files, `CONDA_ENV` / `MAMBA_MODULE`, `WANDB_ENTITY` and the job
mail settings. Launchers are run from anywhere; they `cd` to the repo root.

```bash
DRY_RUN=1 bash assimilation/scripts/SQG/run_all_DAWIS.bash --sweep false      # print the plan
DATA_INDICES="1 2 3 4 5 6 7 8 9 10" \
    bash assimilation/scripts/SQG/run_all_DAWIS.bash --sweep false            # submit
```

## Shared controls

**`DRY_RUN=1`** prints one line per job instead of submitting it. It works by
shadowing `sbatch` (`_dry_run.sh`) in every `run_all_*` launcher. Always do a
dry run first: a sweep is experiments x trajectories x every swept value.

**`--sweep true|false`** (or `SWEEP=true|false`; default `true`). `true` runs
the hyperparameter grid written at the top of the launcher; `false` runs the
tuned configuration used for the paper. On SQG only `run_all_DAWIS.bash`,
`run_all_DAISI.bash` and `run_all_SDA_filter.bash` have a separate
`--sweep false` block; the other SQG launchers already ship their tuned
configuration and accept the flag without effect. `--sweep` does **not**
change which trajectories run.

**`DATA_INDICES="1 2 3 4 5 6 7 8 9 10"`** selects the trajectories (the
`--data_index` of each job). Index 0 is the tuning trajectory; 1-10 are the
evaluation trajectories reported in the paper. Each launcher ships its own
`data_indices` list (often `(0)` or `(0 10)`), so set `DATA_INDICES` explicitly
when reproducing results. It is applied after any `--sweep false` block;
empty or non-numeric values are an error.

Other knobs, all environment variables:

| variable | effect | launchers |
| --- | --- | --- |
| `INIT_STATES_VALUES="2 4 6"` | window depth `--init_states` (window of `init_states + 1` states); refused if no checkpoint exists (window 6 is released; others need `ASSIM_FMW_PATH_<SQG|SEVIR>_ETA01_CHANNEL_INIT_<n>`). Non-default windows add `_init<W>` to the run name. | DAWIS, GuidedForecast, SDA_filter, ForcingDAS, SDA, Gibbs |
| `ASSIM_INTERVAL=k` | observe every `k`-th step (1 = every step, the paper setting; -1 = never). `k != 1` adds `_ai<k>` to the run name. | all |
| `FORWARD_MODELS="numerical"` | subset of `dawis numerical` | `SQG/run_all_DAWIS.bash` |
| `TRAJECTORY_PATH`, `INITIAL_TRAJECTORY`, `TRAJECTORY_VAR` | warm start for the block smoothers (below) | Gibbs |
| `TIME_LIMIT`, `BATCH_SIZE`, `SWEEP_LOGGING` | wall-clock limit, members per GPU call, per-sweep logging | `SQG/run_all_Gibbs.bash` |
| `MAIL_TYPE`, `MAIL_USER` | SLURM job mail (from `.env`; default none) | all |

SDA and the Gibbs smoothers need an even `init_states`.

## Paper methods (SQG)

All launchers are in `SQG/`. The experiments are `noisy`, `sparse`,
`multimodal` and `saturating` (`--experiment`, defined in
`assimilation/experiments.py`).

| task | paper method | launcher | notes |
| --- | --- | --- | --- |
| Filtering | EnSF | `run_all_EnSF.bash` | |
| | FlowDAS | `run_all_FlowDAS.bash` | 6-state conditioning window |
| | LETKF | `run_all_LETKF.bash` | NumPy backend on CPU |
| | DAISI | `run_all_DAISI.bash` | |
| | SDA-Filter | `run_all_SDA_filter.bash` | |
| | Joint AR 1\|W, Joint AR W\|1 | `run_all_GuidedForecast.bash` | `context_modes` `past` = 1\|W, `future` = W\|1 |
| | DAWIS Filter | `run_all_DAWIS.bash` | `--forward_model numerical` |
| | DAWIS-Joint Filter | `run_all_DAWIS.bash` | `--forward_model dawis` |
| Fixed-lag smoothing | SDA-Filter, Joint AR W\|1 | as above | |
| | ForcingDAS-Pyr | `run_all_ForcingDAS.bash` | `--pyramid` |
| | DAWIS / DAWIS-Joint Lagged Smoother | `run_all_DAWIS.bash` | same runs as the filters |
| Joint smoothing | SDA | `run_all_SDA.bash` | |
| | Block Gibbs Smoother, DAWIS Block Smoother | `run_all_Gibbs.bash` | `gibbs_variants` `Gibbs` / `DAWIS` |
| Appendix | DAWIS Soft Block Smoother | `run_all_Gibbs.bash` | `gibbs_variants` `Soft` |

**Filter and lagged smoother from one run.** A DAWIS run writes the filter
estimate (last window slot) as `x_assim` and the lagged-smoother estimate
(window slot 0) as `x_smooth` in the same `.nc`. Setting
`save_window_states=true` in the launcher adds `--save_window_states`, which
also stores every slot of every window as `x_window`.

**Block smoothers start from a finished run.** Either set `TRAJECTORY_PATH` to
a result `.nc` (used for every job the launcher submits, so restrict
`DATA_INDICES` and `experiments` accordingly), or set `PAPER_RUNS_ROOT` to a
directory of runs named `<METHOD>_<Experiment>_run_<index>_<suffix>.nc`,
looked up by `INITIAL_TRAJECTORY` (default `DAWISnumerical`, e.g.
`DAWISnumerical_Noisy_run_3_<suffix>.nc`). `TRAJECTORY_VAR` picks `x_smooth`
(default) or `x_assim`.

## Running on another cluster

The `#SBATCH` headers of the workers and the resource arguments in the
launchers contain Berzelius-specific settings: node types `-C fat` / `-C thin`
(80 GB / 40 GB A100) and the CPU partition `-p berzelius-cpu`
(`SQG/run_all_LETKF.bash`, `data/SQG/gen_data.sh`). Adjust them for your
cluster. The SEVIR
launchers pass `-C "$NODE_TYPE"` on the command line; see `SEVIR/README.md`.

`#SBATCH` lines cannot read variables, so job mail is added on the command
line from `MAIL_TYPE` / `MAIL_USER` in `.env` (`scripts/sbatch_opts.sh`). The
launchers do this themselves; submit a single job script with
`scripts/submit.sh` to get the same options:

```bash
scripts/submit.sh -t 04:00:00 unconditional_generation/scripts/SQG/train_64.sh
```
