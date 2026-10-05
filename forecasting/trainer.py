# Standard library
import json
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from pytorch_lightning.callbacks import LearningRateMonitor

# First-party
from forecasting.models.unet import UNET
from forecasting.models.fmw import FMW
from data.registry import (
    add_dataset_args,
    data_source_from_args,
    resolve_n_workers,
)
from forecasting.loaders import build_loader
import utils

MODELS = {
    "UNET": UNET,
    "FMW": FMW,
}


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def limit_batches(arg):
    """Parse a Lightning `limit_*_batches` value: int = batch count, float = fraction."""
    return float(arg) if "." in arg or "e" in arg.lower() else int(arg)


def main(input_args=None):
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    # Data is selected by --dataset/--variant
    add_dataset_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        default="FMW",
        choices=sorted(MODELS),
        help="Model architecture to train/evaluate (default: FMW)",
    )
    parser.add_argument(
        "--subset_ds",
        action="store_true",
        help="Use only a small subset of the dataset, for debugging"
        "(default: false)",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of workers in data loader. Unset, the data source's own "
             "default applies (SEVIR 16, everything else 4); pass a number to "
             "override it, e.g. 0 to load in the main process for debugging.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="upper epoch limit (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size (default: 32)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--backbone_ckpt_path",
        type=str,
        help="Path to load backbone model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        action="store_true",
        help="If optimizer state should be restored with model "
        "(default: false)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )
    parser.add_argument(
        "--num_sanity_steps",
        type=int,
        default=2,
        help="Number of sanity checking validation steps to run before starting"
        " training (default: 2)",
    )

    # Model architecture
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Dimensionality of all hidden representations (default: 32)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="heun",
        help="The sampler to use when generating trajectories with a diffusion model"
        "(heun/edm) (default: heun)",
    ),

    # Training options
    parser.add_argument(
        "--init_states",
        type=int,
        default=2,
        help="Number of initial states "
        "(default: 2)",
    )
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss "
        "(default: 1)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Loss function to use, see metric.py (default: mse)",
    )
    parser.add_argument(
        "--fm_loss",
        type=str,
        default="eta01_channel",
        help="FMW loss (b/eta0/eta1/eta01_label/eta01_model/eta01_channel) "
             "(default: eta01_channel)",
    )
    parser.add_argument(
        "--fm_forecast",
        action="store_true",
        help="If true, only predict the target state channels, setting t=0 for all other channels (default: False)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, help="minimum learning rate for cosine annealing (default: 0.00001)"
    )
    parser.add_argument(
        "--lr_ref_batches",
        type=float,
        default=5e3,
        help="Reference number of batches for learning rate scheduling (default: 15000)",
    )
    parser.add_argument(
        "--lr_rampup_Mimg",
        type=float,
        default=0.01,
        help="Ramp-up Mimg for learning rate scheduling (default: 0.01)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=limit_batches,
        default=1.0,
        help="How much of the val split each validation run uses: an int is a "
        "number of batches, a float is a fraction (default: 1.0, the whole "
        "split). Handed straight to pl.Trainer(limit_val_batches=...), so the "
        "batches are always the *first* N of the split -- the val loader is "
        "not shuffled, which is what makes the val curve comparable across "
        "epochs and across runs. Note a count is per rank: under DDP every "
        "gpu runs that many batches, so N validates on N * batch_size * gpus "
        "windows. Worth setting on datasets where a full "
        "validation run costs more than the training epoch it follows -- see "
        "the ensemble sampling in ARProbModel.validation_step, which makes one "
        "SEVIR val batch ~40-65x a train batch.",
    )
    parser.add_argument(
        "--pred_residual",
        action="store_true",
        help="If the model should predict residuals instead of absolute values",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for training. (default: 0.01)",
    )

    # Backbone / UNET options
    parser.add_argument(
        "--resample_filter",
        type=list_of_ints,
        default="1,3,3,1",
        help="Resample filter for backbone unet (default: 1,1 or 1,3,3,1)",
    )
    parser.add_argument(
        "--channel_mult",
        type=list_of_ints,
        default="2,2,2",
        help="Channel multiplier for backbone unet (depth and width of UNET) (default: 2,2,2,2)",
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="residual",
        help="Type of encoder to use in backbone unet (standard/residual/skip)"
        "(default: 'standard')",
    )

    parser.add_argument(
        "--attn_resolutions",
        type=list_of_ints,
        default="32",
        help="Resolutions to apply attention to in edm model (default: '1')",
    )
    parser.add_argument(
        "--sdl_res_dec",
        type=list_of_ints,
        default="1",
        help="Resolutions to apply sdl noise adapters to in unet model decoder (default: '1')",
    )
    parser.add_argument(
        "--sdl_res_enc",
        type=list_of_ints,
        default="1",
        help="Resolutions to apply sdl noise adapters to in unet model encoder (default: '1')",
    )

    parser.add_argument(
        "--noise_embedding",
        type=str,
        default="fourier",
        help="Type of encoder to use in edm model (positional/fourier/linear)"
        "(default: 'fourier')",
    )
    parser.add_argument(
        "--channel_mult_emb",
        type=float,
        default=4,
        help="Channel multiplier for noise embedding MLP (default: 4)",
    )
    parser.add_argument(
        "--channel_mult_noise",
        type=float,
        default=1,
        help="Channel multiplier for noise level MLP (default: 1)",
    )

    # Noise inputs (recorded in checkpoints, see forecasting/ckpt_args.py)
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=32,
        help="Dimension of the noise vector z (default: 32)",
    )
    parser.add_argument(
        "--spatial_noise_dim",
        type=int,
        default=0,
        help="Number of spatial noise channels to concatenate to the input "
        "of the model (default: 0)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--ar_steps_eval",
        type=int,
        default=50,
        help="Number of steps to unroll prediction for in evaluation "
        "(default: 50)",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during val/test "
        "(default: 1)",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of ensemble members during evaluation (default: 5)",
    )
    parser.add_argument(
        "--sampler_steps",
        type=int,
        default=100,
        help="Number of sampling steps during inference (default: 20)",
    )
    parser.add_argument(
        "--sampler_eps",
        type=float,
        default=1e-3,
        help="Epsilon parameter for samplers during inference (default: 1e-3)",
    )

    # FMW options
    parser.add_argument(
        "--schedule",
        type=str,
        default="linear_scalar",
        help="The alpha, beta schedule to use (default: linear_scalar)"
    )
    parser.add_argument(
        "--guide_method",
        type=str,
        default=None,
        help="The guidance method to use (MMPS/DPS), (default: None)"
    )
    parser.add_argument(
        "--corr_noise",
        type=float,
        default=0.,
        help="The correlation structure for the noise at diffusion time 0 (default: 0.)"
    )
    parser.add_argument(
        "--alpha_beta_spatial",
        action="store_true",
        help="If the alpha, beta values should be concatented as spatial input channels (default: False)"
    )
    parser.add_argument(
        "--time_dropout",
        type=float,
        default=None,
        help="Dropout probability for the time dimension (default: None)"
    )
    parser.add_argument(
        "--fm_uncond",
        action="store_true",
        help="If we also want to train a unconditional FM model at the same time (default: False)"
    )
    parser.add_argument(
        "--alpha_beta_mult",
        type=int,
        default=None,
        help="Channel multiplier for alpha, beta channels (default: None -> will scale linearly with init_states)"
    )

    # System options
    parser.add_argument(
        "--step_length",
        type=float,
        default=None,
        help="Step length in hours. Defaults to the dataset's own cadence "
             "(3 h for SQG base, 10 min for SEVIR-LR); the old fixed default "
             "of 3 h is meaningless for a sub-hourly dataset.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=None,
        help="Spatial resolution. Derived from --variant when omitted; "
             "passing it asserts agreement (many launcher scripts still do).")

    # Logger Settings
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="SQG",
        help="Wandb run project (default: 'SQG')",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="Wandb run name (default: '')",
    )
    parser.add_argument(
        "--val_steps_to_log",
        type=list_of_ints,
        default="1,25,50",
        help="Steps to log val loss for (default: [1, 25, 50])",
    )
    parser.add_argument(
        "--metrics_watch",
        nargs="+",
        default=["val_rmse"],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )

    args = parser.parse_args(input_args)

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Load data; paths, grid size and cadence come from the data source
    src = data_source_from_args(args)
    md = src.metadata
    print(f"Data source: {md.name}/{md.variant} grid={md.grid} "
          f"channels={md.num_channels} cadence={md.cadence_hours}h "
          f"norm={md.norm.source}", flush=True)
    if args.nx is not None and args.nx != md.nx:
        raise ValueError(
            f"--nx {args.nx} contradicts {md.name}/{md.variant} grid {md.grid}; "
            f"select the data with --variant instead."
        )
    # Models read args.nx for img_resolution, so fill it in from metadata.
    args.nx = md.nx
    # Resolve before the wandb logger snapshots args
    args.n_workers = resolve_n_workers(src, args)
    print(f"DataLoader workers: {args.n_workers}", flush=True)
    # Default the forecast step to the data's own cadence
    if args.step_length is None:
        args.step_length = md.cadence_hours
    print(f"Forecast step: {args.step_length} h "
          f"(subsample_step={md.subsample_step(args.step_length)})", flush=True)

    train_loader = build_loader(src, "train", args,
                                pred_length=args.ar_steps, shuffle=True)
    val_loader = build_loader(src, "val", args,
                              pred_length=args.ar_steps_eval, shuffle=False)

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        # Allows using Tensor Cores on A100s
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"

    args.device = device_name

    model_class = MODELS[args.model]
    model = model_class(args)

    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"

    prefix = f"{args.wandb_run_name}-{prefix}" if args.wandb_run_name else prefix
    run_name = (
        f"{prefix}{args.model}-{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    # Callbacks for saving model checkpoint
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_mean_loss",
            mode="min",
            save_last=True,
        )
    )

    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="last_epoch",
            monitor="epoch",
            save_on_train_epoch_end=True,
            save_top_k=1,
            every_n_epochs=1,
            save_last=True,  # Optionally also save the last epoch
        )
    )

    # Saving the evaluation results
    wandb_project = args.wandb_project if args.eval is None else f"{args.wandb_project}_eval"

    if args.wandb_entity is not None:
        logger = pl.loggers.WandbLogger(
            entity=args.wandb_entity, project=wandb_project, name=run_name, config=args
        )
    else:
        logger = pl.loggers.WandbLogger(
            project=wandb_project, name=run_name, config=args
        )

    # Training strategy
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy="ddp",
        accelerator=device_name,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        limit_val_batches=args.limit_val_batches,
        precision=args.precision,
        num_sanity_val_steps=args.num_sanity_steps,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(
            logger, args.val_steps_to_log
        )  # Do after wandb.init

    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            eval_loader = build_loader(
                src, "test", args, pred_length=args.ar_steps_eval,
                shuffle=False, pin_memory=True, persistent_workers=True)
        print(f"Running evaluation on {args.eval}")
        trainer.test(model=model, dataloaders=eval_loader, ckpt_path=args.load)
        print(f"Evaluated model trained for {trainer.current_epoch} epochs.")
    else:
        print("Starting training")
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.load,
        )


if __name__ == "__main__":
    main()
