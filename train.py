import os
import csv
import time
import argparse
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return (not is_distributed()) or get_rank() == 0


# -------------------------
# DDP Setup (DDP-ready, but safe for single-process)
# -------------------------
def ddp_setup(backend: str):
    if dist.is_initialized():
        return

    # Force localhost init to avoid docker/kubernetes env interference on Windows
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29501")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Dataset
# -------------------------
class NBATeamGameDataset(Dataset):
    """
    Reads a CSV of team-game rows.
    Features: numeric columns (float32)
    Label: win (0/1)
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        # Basic safety checks
        required = {"win"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"CSV must include columns: {required}")

        self.y = df["win"].astype(np.float32).to_numpy(copy=True).reshape(-1, 1)

        # Everything except 'win' is treated as a feature
        feature_cols = [c for c in df.columns if c != "win"]
        self.X = df[feature_cols].astype(np.float32).to_numpy(copy=True)

        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y


# -------------------------
# Simple model
# -------------------------
class WinPredictor(nn.Module):
    """
    Simple MLP for tabular data → outputs win probability (logit).
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Metrics logging
# -------------------------
def init_metrics_file(path: str):
    if is_main_process():
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss", "effective_batch_size", "world_size", "accum_steps", "timestamp"])


def append_metrics(path: str, step: int, loss: float, eff_bs: int, world_size: int, accum_steps: int):
    if is_main_process():
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([step, loss, eff_bs, world_size, accum_steps, int(time.time())])


# -------------------------
# Train
# -------------------------
def train(args):
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device(f"cuda:{get_local_rank()}" if use_cuda else "cpu")
    backend = "nccl" if use_cuda else "gloo"

    if is_distributed():
        ddp_setup(backend)

    set_seed(args.seed + get_rank())

    dataset = NBATeamGameDataset(args.data_path)

    sampler = None
    shuffle = True
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            seed=args.seed,
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )

    model = WinPredictor(in_dim=dataset.X.shape[1]).to(device)

    # Wrap with DDP only if distributed
    if is_distributed():
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[get_local_rank()] if use_cuda else None,
            output_device=get_local_rank() if use_cuda else None,
        )

    # Binary classification loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    init_metrics_file(args.metrics_path)

    eff_batch = args.micro_batch_size * args.accum_steps * get_world_size()

    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            # accumulation: scale loss so gradients match large batch average
            (loss / args.accum_steps).backward()

            if (i + 1) % args.accum_steps == 0:
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # Average loss across processes for cleaner logging
                loss_value = loss.detach()
                if is_distributed():
                    dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)

                append_metrics(
                    args.metrics_path,
                    global_step,
                    float(loss_value.item()),
                    eff_batch,
                    get_world_size(),
                    args.accum_steps,
                )

                if is_main_process() and global_step % args.log_every == 0:
                    print(f"step={global_step} loss={float(loss_value.item()):.4f} eff_batch={eff_batch}")

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

    if is_main_process():
        print("Done.")
        print(f"metrics written to: {args.metrics_path}")

    if is_distributed():
        ddp_cleanup()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/nba_team_games.csv")
    p.add_argument("--metrics_path", type=str, default="metrics.csv")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=50)

    p.add_argument("--micro_batch_size", type=int, default=2)
    p.add_argument("--accum_steps", type=int, default=4)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=5)

    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
