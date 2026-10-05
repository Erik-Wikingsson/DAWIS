# Data sources

Every dataset lives in its own directory under `data/` and is discovered
automatically: add a `config.yaml` and a `DataSource` subclass, and
`unconditional_generation`, `forecasting` and `assimilation` can all use it.

```
data/
  base.py               DataSource ABC, DatasetMetadata, NormStats, VariableSpec
  paths.py              path-variable resolution (environment > repo .env)
  registry.py           get_data_source(name, variant), available(), all_variants()
  compute_data_stats.py per-channel normalization statistics
  SQG/                  2-level surface quasi-geostrophic PV (generated)
  SEVIR/                SEVIR-LR radar mosaics (VIL)
```

## Configuring where the data is

Data roots are read from the repo `.env` (copy `.env.example` to `.env` and
edit it); a variable already set in the environment takes precedence.

| variable | contents |
| --- | --- |
| `SQG_ROOT` | `train/`, `val/`, `test/`, each with `<nx>_3h/` subdirectories |
| `SEVIR_ROOT` | `sevir_lr/CATALOG.csv` and `sevir_lr/data/vil/<year>/*.h5` |

Every entry point takes `--dataset`, `--variant` and `--data_root`
(`data.registry.add_dataset_args`); `--data_root` overrides the root for that
run.

### SQG

The SQG trajectories are generated with the bundled model
(`data/SQG/sqg_nature_run.py`):

```bash
bash data/SQG/generate_splits.sh --jobs 8               # base variant, 64x64
bash data/SQG/generate_splits.sh --variant hires        # 256x256
bash data/SQG/generate_splits.sh --help                 # all options
```

Options: `--variant base|hires`, `--splits "train val test test_long"`,
`--n_train`, `--n_val`, `--n_test` (default 11: index 0 is the tuning
trajectory, 1-10 the evaluation set), `--jobs`, `--seed`, `--no_stats`. The
base variant writes

```
$SQG_ROOT/train/64_3h/sqg_N64_3hrly_steps_100_<i>_<id>.{nc,npy}            101 frames
$SQG_ROOT/val/64_3h/sqg_N64_3hrly_steps_100_<i>_<id>.{nc,npy}              101 frames
$SQG_ROOT/test/64_3h/sqg_N64_3hrly_steps_110_<i>_<id>.{nc,npy}             111 frames
$SQG_ROOT/test/64_3h_1000step/sqg_N64_3hrly_steps_1000_0_<id>.{nc,npy}     1001 frames
$SQG_ROOT/train/64_3h/{data,diff}_{mean,std}.pt
```

The script refuses to write into a directory that already holds trajectories.
The `.nc` files carry the model parameters that the numerical forward model
and LETKF read.

### SEVIR

The code uses SEVIR-LR, the low-resolution SEVIR distributed by
[Earthformer](https://github.com/amazon-science/earth-forecasting-transformer):
VIL only, 128x128, 25 frames per event at a 10-minute cadence.
[Download it here](https://www.dropbox.com/scl/fi/h83pp33jx5gz62gk0gncs/sevir_lr.zip?rlkey=dtnnk6x4af0hhrneugijhq60s&st=ux0ud8pz&dl=0) (3.8 GB) and unzip it into `$SEVIR_ROOT`:

```bash
cd "$SEVIR_ROOT"
wget -O sevir_lr.zip "https://www.dropbox.com/scl/fi/h83pp33jx5gz62gk0gncs/sevir_lr.zip?rlkey=dtnnk6x4af0hhrneugijhq60s&st=ux0ud8pz&dl=1"
unzip sevir_lr.zip && rm sevir_lr.zip
```

which gives:

```
$SEVIR_ROOT/sevir_lr/CATALOG.csv
$SEVIR_ROOT/sevir_lr/data/vil/{2017,2018,2019}/*.h5
```

Variants: `lr_vil` (128x128, the default) and `lr_vil_64` (downsampled 2x).
The splits follow FlowDAS: `test` is every event from 2019-06-01 on, and
`train`/`val` are a seeded partition of the events before it. For
assimilation, `--data_index` indexes a seeded 11-event sample of the split
(`assim_sample` in `data/SEVIR/config.yaml`), so index 0 (tuning) and 1-10
(evaluation) mean the same as on SQG.

The model-space normalization is `norm_mode`, set with `--sevir_norm` /
`--assim_norm` or `$SEVIR_NORM`: `01` (vil/255, the config default), `flowdas`
(`(vil/255 - 0.5)/0.1`, what the pretrained SEVIR window models and the
FlowDAS checkpoint use) or `standard`. See `NORM_MODES` in
`data/SEVIR/source.py`.

## Using one

```python
from data.registry import get_data_source

src = get_data_source("SQG", "base")      # cheap; touches no files
md  = src.metadata

md.grid            # (ny, nx)          -- never assume square
md.num_channels    # variables per grid point
md.unit_factor     # raw -> physical   (SQG: scalefact; SEVIR: exactly 1.0)
md.cadence_hours   # float, so a 10-minute cadence does not truncate to 0
md.periodic        # -> SongUNet(circular_padding=...)
md.max_window      # longest window the source can serve

ok, why = src.is_available()             # os.path.exists-level check only
```

## The three access patterns

Canonical layout is time-major with the channel axis at `-3`: a single state is
`(C, H, W)` and anything temporal is `(T, C, H, W)`. A source converts its
native layout internally (SEVIR is stored `NHWT`).

| method | per-item | dtype | units |
| --- | --- | --- | --- |
| `single_state(split)` | `(C, ny, nx)` | float32 | normalized |
| `forecast_window(split, init_states=, pred_length=)` | `((T_in,C,ny,nx), (T_out,C,ny,nx))` | float32 | normalized |
| `assim_trajectory(index=, split=)` | `AssimCase`; `.states[t]` -> `(C,ny,nx)` | float32 | **`md.assim_space`** |

### `assim_space`

`assim_trajectory` is the one access pattern whose units a source chooses, and
`DatasetMetadata.assim_space` declares which:

* `"physical"` (SQG, and the default): raw units. The assimilation methods
  state their contract in physical units and apply `unit_factor` themselves.
* `"normalized"` (SEVIR): the network's own space, offset included. Every
  method's `x / scale` bridge is then the identity (`md.assim_scale()` is 1),
  and no offset has to be threaded through the assimilation code. Set
  `SEVIR_ASSIM_SPACE=physical` to get raw VIL counts instead.

`--obs_sigma` and `--init_std` are stated in whichever unit the run works in, so
getting this wrong is a silent factor-of-`std` error.
`tests/test_data_sources.py::test_assim_trajectory_matches_declared_assim_space`
guards it in both directions.

`AssimCase.model_params` carries the numerical-model parameters, or is `None`
for a dataset with no known dynamics (SEVIR). Then `--forward_model numerical`
raises `UnsupportedForDataset`, and LETKF needs a learned propagator.

## Normalization

`NormStats` keeps two conventions, named, because for SQG they are different
numbers (scalar `std=2660` versus per-channel values):

* `legacy_scalar_mean/std`: unconditional generation (the DAISI prior) and the
  non-DAWIS assimilation methods
* `state_mean/state_std`: forecasting and DAWIS (`data_mean.pt` / `data_std.pt`)
* `diff_mean/diff_std`: `--pred_residual`; `None` where never computed, in
  which case `--pred_residual` fails rather than guessing

The per-channel files are computed from the training split with

```bash
PYTHONPATH=$(pwd) python data/compute_data_stats.py \
    --dataset SQG --variant base --split train --in_place
```

`generate_splits.sh` runs this automatically. Without `--in_place` the files
go to a `stats/` subdirectory of the split directory (or `--out_dir`), so
statistics that existing checkpoints were trained against are not overwritten.

## Adding a data source

1. `mkdir data/<NAME>/`
2. Write `config.yaml` with `name`, `source_class` (`pkg.module:Class`), `root`
   (may use `${VAR}` or `${VAR:-default}`), and a `variants:` block. Anything
   shared goes in a `defaults: &defaults` anchor.
3. Subclass `DataSource` in `source.py`: implement `metadata`, `split_dir`,
   `list_trajectories`, and the three access patterns. **Constructing it must
   not read data files**: `metadata` has to work on a machine with no data, so
   keep heavy imports inside methods.
4. Register a bespoke observer or forward model in `assimilation/registry.py`
   if you need one. They live there, not under `data/`, to avoid an import
   cycle with `forecasting/`.
5. Run the contract tests.

## Contract tests

```bash
PYTHONPATH=$(pwd) python -m pytest tests/test_data_sources.py -q
```

They parametrize over every registered variant, assert the layout, dtype and
unit contracts above, and skip cleanly when the data is not on the machine.

Conventions they rely on:

* `unit_factor` may be exactly `1.0`; every `* unit_factor` site must be a
  harmless no-op for such a dataset.
* `list_trajectories` is always `sorted()`, since `--data_index` indexes it.
* Nothing under `data/` imports from `assimilation/`, `forecasting/`,
  `unconditional_generation/` or `plotting/`.
* Splits must be disjoint. A date-ranged source exposes `split_dates(split)`;
  two splits may share a date range only if they take disjoint pieces of a
  seeded partition, exposed as `split_part(split)`. `test_splits_do_not_overlap`
  checks both.
