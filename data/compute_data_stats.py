"""Pre-compute per-channel normalization statistics for a data source.

Writes `data_mean.pt`, `data_std.pt`, `diff_mean.pt`, `diff_std.pt`, the
per-channel profile used by forecasting and DAWIS. By default writes to a
`stats/` subdirectory so existing files are not overwritten.

Usage:
    PYTHONPATH=$(pwd) python data/compute_data_stats.py \
        --dataset SQG --variant base --split train --out_dir /tmp/stats
"""

# Standard library
import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from data.registry import add_dataset_args, get_data_source


def save_stats(out_dir, means, squares, filename_prefix):
    means = torch.stack(means) if len(means) > 1 else means[0]
    squares = torch.stack(squares) if len(squares) > 1 else squares[0]
    print(f"means shape: {means.shape}, squares shape: {squares.shape}")
    mean = torch.mean(means, dim=0)              # (d_features,)
    second_moment = torch.mean(squares, dim=0)   # (d_features,)
    std = torch.sqrt(second_moment - mean**2)    # (d_features,)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving computed stats to {out_dir}...")
    print(f"{filename_prefix} mean: {mean}")
    print(f"{filename_prefix} std.: {std}")
    torch.save(mean.cpu(), os.path.join(out_dir, f"{filename_prefix}_mean.pt"))
    torch.save(std.cpu(), os.path.join(out_dir, f"{filename_prefix}_std.pt"))
    return mean, std


def main():
    parser = ArgumentParser(description="Compute dataset normalization statistics")
    add_dataset_args(parser)
    parser.add_argument("--split", type=str, default="train",
                        help="Split to compute statistics over (default: train)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to write the .pt files. Defaults to a "
                             "'stats' subdirectory of the split dir, so the "
                             "existing files are never overwritten in place.")
    parser.add_argument("--in_place", action="store_true",
                        help="Write directly into the split directory, "
                             "overwriting existing statistics. Every existing "
                             "checkpoint's buffers came from those files.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    src = get_data_source(args.dataset, args.variant, root=args.data_root)
    md = src.metadata
    split_dir = src.split_dir(args.split)
    out_dir = args.out_dir or (str(split_dir) if args.in_place
                               else os.path.join(str(split_dir), "stats"))
    print(f"{md.name}/{md.variant} split={args.split} -> {out_dir}")

    # Pass 1: raw statistics over consecutive pairs; no random subsampling so
    # the result is deterministic.
    ds = src.forecast_window(args.split, init_states=1, pred_length=1,
                             standardize=False, random_subsample=False)
    loader = torch.utils.data.DataLoader(
        ds, args.batch_size, shuffle=False, num_workers=args.n_workers)

    print("Computing mean and std.-dev. for parameters...")
    means, squares = [], []
    for init_batch, target_batch in tqdm(loader):
        batch = torch.cat((init_batch, target_batch), dim=1)   # (B, T, C, H, W)
        means.append(torch.mean(batch, dim=(1, 3, 4)).cpu())    # (B, C)
        squares.append(torch.mean(batch**2, dim=(1, 3, 4)).cpu())
    save_stats(out_dir, [torch.cat(means, dim=0)],
               [torch.cat(squares, dim=0)], "data")

    # Pass 2: one-step differences of the standardized data, using the stats
    # just written.
    print("Computing mean and std.-dev. for one-step differences...")
    data_mean = torch.load(os.path.join(out_dir, "data_mean.pt"),
                           weights_only=True).view(1, 1, -1, 1, 1)
    data_std = torch.load(os.path.join(out_dir, "data_std.pt"),
                          weights_only=True).view(1, 1, -1, 1, 1)

    diff_means, diff_squares = [], []
    for init_batch, target_batch in tqdm(loader):
        batch = torch.cat((init_batch, target_batch), dim=1)
        batch = (batch - data_mean) / data_std
        batch_diffs = batch[:, 1:] - batch[:, :-1]
        diff_means.append(torch.mean(batch_diffs, dim=(1, 3, 4)).cpu())
        diff_squares.append(torch.mean(batch_diffs**2, dim=(1, 3, 4)).cpu())
    save_stats(out_dir, [torch.cat(diff_means, dim=0)],
               [torch.cat(diff_squares, dim=0)], "diff")


if __name__ == "__main__":
    main()
