"""
A script to train a diffusion model for unconditional generation.
"""

import os
import csv
import math
import torch
import wandb
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import DataLoader


# Local imports
from data.paths import lookup
from data.registry import (
    add_dataset_args,
    data_source_from_args,
    resolve_n_workers,
)
from metrics.metrics import power_spectrum, radial_average
from unconditional_generation.backbone import build_backbone
from plotting.plotting import save_spectrum_series
from networks.diffusion_networks import SongUNet


def etaz_loss(z0, z1): return z0
def eta1_loss(z0, z1): return z1
def b_loss(z0, z1): return z1 - z0


TARGET_FNS = {
    'etaz_loss': etaz_loss,
    'eta1_loss': eta1_loss,
    'b_loss': b_loss
}


def loss_fn(model, batch, target_fn):
    """Stochastic interpolant loss on zt = (1 - t) z0 + t z1, z0 ~ N(0, I), t ~ U(0, 1).

    Args:
        model: callable (zt, t) -> (B, C, H, W) prediction.
        batch: (B, C, H, W) data samples z1.
        target_fn: callable (z0, z1) -> (B, C, H, W) regression target.

    Returns:
        Scalar loss.
    """
    z0 = torch.randn_like(batch, device=batch.device)
    z1 = batch

    t = torch.rand(batch.shape[0], device=batch.device)
    zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1

    pred = model(zt, t)
    target = target_fn(z0, z1)
    loss = torch.mean(0.5 * pred ** 2 - target * pred)

    return loss


def parse_args():
    parser = ArgumentParser(
        description="Train a diffusion model for unconditional generation.")

    # Logging arguments
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=lookup("WANDB_ENTITY"),
        help="Wandb entity (default: $WANDB_ENTITY, else your default entity)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ScoreDA",
        help="Wandb run project",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="Wandb run name",
    )
    parser.add_argument(
        "--target_fn",
        type=str,
        default="b_loss",
        choices=["b_loss", "etaz_loss", "eta1_loss"],
        help="Which target function to use for the loss",
    )

    # Data arguments
    add_dataset_args(parser)
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of dataloader workers. Unset, the data "
                             "source's own default applies (SEVIR 16, "
                             "everything else 4); pass a number to override "
                             "it, e.g. 0 to load in the main process.")
    parser.add_argument("--plot_channel", type=int, default=0,
                        help="Which channel to show in the sample plot")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float,
                        default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-4, help="Weight decay")
    parser.add_argument("--num_epochs", type=int,
                        default=50, help="Number of epochs")
    parser.add_argument("--start_factor", type=float,
                        default=0.001, help="Warmup scheduler start factor")
    parser.add_argument("--end_factor", type=float,
                        default=1.0, help="Warmup scheduler end factor")
    parser.add_argument("--total_iters", type=int,
                        default=1000, help="Warmup scheduler total iters")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Batch size for training dataloader")
    parser.add_argument("--channels", type=int, default=32,
                        help="Number of channels in the model")

    return parser.parse_args()


def train(args):
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config=vars(args)
    )

    result_path = Path(f'results/{run.name}')
    result_path.mkdir(parents=True, exist_ok=True)

    # All shapes and constants come from the data source metadata.
    src = data_source_from_args(args)
    md = src.metadata
    print(f"Data source: {md.name}/{md.variant} "
          f"grid={md.grid} channels={md.num_channels} "
          f"norm={md.norm.source}", flush=True)

    # Source defaults, overridden by --n_workers; persistent_workers and
    # prefetch_factor require workers > 0.
    loader_kwargs = dict(pin_memory=True)
    loader_kwargs.update(md.dataloader_kwargs)
    args.n_workers = resolve_n_workers(src, args)
    loader_kwargs["num_workers"] = args.n_workers
    if args.n_workers == 0:
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("prefetch_factor", None)
    print(f"DataLoader workers: {args.n_workers}", flush=True)
    # Record the resolved worker count.
    run.config.update({"n_workers": args.n_workers}, allow_val_change=True)

    dataset = src.single_state("train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            **loader_kwargs)

    print("Number of training batches:", len(dataloader), flush=True)

    val_dataset = src.single_state("val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, **loader_kwargs)

    model = build_backbone(md, args)
    print("Num params: ", sum(p.numel()
          for p in model.parameters()), flush=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model.to(device)

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    target_fn = TARGET_FNS[args.target_fn]

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.start_factor,
        end_factor=args.end_factor,
        total_iters=args.total_iters
    )

    loss_values = []
    val_loss_values = []
    best_val_loss = float('inf')

    # Setup for logging
    log_file_path = result_path / f'training_log.csv'
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss'])

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_train_loss = 0
        for step, image in enumerate(dataloader, start=1):
            image = image.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model, image, target_fn)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            warmup_scheduler.step()

            wandb.log(
                {"train/loss_step": loss.item(),
                 "train/global_step": epoch * len(dataloader) + step}
            )

        avg_train_loss = total_train_loss / len(dataloader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for image in tqdm(val_dataloader):
                image = image.to(device)
                loss = loss_fn(model, image, target_fn)
                total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), result_path/f'best_model.pth')

        scheduler.step()

        # Generate sample
        sampler = Sampler(device=device, members=20, eps=lambda t: 0.03 *
                          (1-t), steps=100, model=model, debug=False,
                          state_shape=md.state_shape)
        with torch.no_grad():
            sample = sampler.sample().cpu()

        # Plot sample in physical units.
        channel = min(args.plot_channel, md.num_channels - 1)
        physical = md.to_physical(md.norm.denormalize(sample, per_channel=False))
        colours = md.imshow_kwargs(channel)
        fig, axs = plt.subplots(2, 3, figsize=(6, 4), constrained_layout=True)
        axs = axs.flatten()
        for i in range(6):
            axs[i].imshow(physical.numpy()[i, channel], **colours)
            axs[i].axis('off')
        plt.savefig(result_path/f'sample_epoch_{epoch+1}.png')
        plt.close(fig)

        # Plot spectrum
        truth = next(iter(val_dataloader))
        save_spectrum(sample, truth, result_path, epoch)

        loss_values.append([avg_train_loss])
        val_loss_values.append(avg_val_loss)

        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])

        print(
            f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}', flush=True)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss_epoch": avg_train_loss,
            "val/loss_epoch": avg_val_loss,
            "sample": wandb.Image(str(result_path/f"sample_epoch_{epoch+1}.png")),
            "spectrum": wandb.Image(str(result_path/f"spectrum_epoch_{epoch+1}.png")),
        })

        torch.save(model.state_dict(), result_path/f'final_model.pth')

    run.finish()


class Sampler():
    def __init__(self, device, members, eps, steps, model, debug=False,
                 state_shape=None):
        self.model = model
        self.device = device
        self.members = members
        # (C, H, W) of a single state.
        self.state_shape = tuple(state_shape) if state_shape is not None else (2, 64, 64)

        self.beta = lambda t: t
        self.alpha = lambda t: 1 - t  # Adjusted alpha to account for prior noise
        self.beta_dot = lambda t: 1  # Derivative of beta with respect to t
        self.alpha_dot = lambda t: - 1  # Derivative of alpha with respect to t
        self.gamma = lambda t: self.alpha(
            t) * self.beta_dot(t) - self.beta(t) * self.alpha_dot(t)

        self.invert_eps = eps
        self.invert_steps = steps
        self.eps = eps
        self.steps = steps

        self.debug = debug

    # Euler-Maruyama sampling with learned b
    def sample(self, z0=None):
        eps = self.eps
        steps = self.steps

        if z0 is None:
            z0 = torch.randn((self.members, *self.state_shape), device=self.device)

        with torch.no_grad():
            tmin, tmax = 0, 1
            zt = z0

            ts = torch.linspace(tmin, tmax, steps+1, device=self.device)[:-1]
            dt = (tmax - tmin) / steps

            if self.debug:
                enum = tqdm(ts)
            else:
                enum = ts

            zs = [zt.clone().cpu()] if self.debug else None

            for t in enum:
                beta_t, alpha_t = self.beta(t), self.alpha(t)
                beta_dot_t, gamma_t = self.beta_dot(t), self.gamma(t)

                eps_t = eps(t)

                t_tensor = torch.ones((self.members,), device=self.device) * t
                b = self.model(zt, t_tensor)

                if t < 1.0:
                    s = (beta_t * b - beta_dot_t * zt) / \
                        (alpha_t * gamma_t)  # s = (t * b - zt) / (1 - t)
                else:
                    s, eps_t = 0, 0

                dz = (b + s * eps_t) * dt
                dW = torch.randn_like(zt) * math.sqrt(2*math.fabs(dt) * eps_t)

                zt = zt + dz + dW

                if self.debug:
                    zs.append(zt.clone().cpu())

            if self.debug:
                return zt, torch.stack(zs, dim=1)
            else:
                return zt


def save_spectrum(sample, truth, result_path, epoch):
    """Plot the radial power spectrum of samples against a validation batch."""
    rad_truth = radial_average(power_spectrum(truth))
    rad_forecast = radial_average(power_spectrum(sample))
    return save_spectrum_series(
        {"Truth": rad_truth, "Sample": rad_forecast},
        result_path / f"spectrum_epoch_{epoch+1}.png",
        members=rad_forecast,
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)
