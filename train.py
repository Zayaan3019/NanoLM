"""
Training entry point.

Quick start — CPU (Intel Core Ultra 7 / any machine)
─────────────────────────────────────────────────────
python prepare_data.py --dataset shakespeare --vocab_size 4096
python train.py --preset nano

Single GPU
──────────
python train.py --preset gpt2 --dataset openwebtext --device cuda

Multi-GPU (4 GPUs)
──────────────────
torchrun --nproc_per_node=4 train.py --preset gpt2 --dataset openwebtext

Resume from checkpoint
──────────────────────
python train.py --preset nano --resume checkpoints/best.pt

Override any config value
─────────────────────────
python train.py --preset nano --max_steps 10000 --batch_size 16 --log_interval 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

from nanolm.model.config import ModelConfig, TrainConfig
from nanolm.model.gpt import GPT
from nanolm.data.dataset import create_dataloaders
from nanolm.training.trainer import Trainer, get_device, setup_ddp, teardown_ddp
from nanolm.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a GPT language model from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Preset
    p.add_argument(
        "--preset",
        choices=["nano", "small", "gpt2", "gpt2_medium"],
        default="nano",
        help="Named model + training preset.",
    )

    # Device
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda / cpu / xpu / mps.  Default: auto-detect.",
    )

    # Resume
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )

    # ---- TrainConfig overrides (common ones exposed as CLI flags) --------
    p.add_argument("--dataset",         type=str,   default=None)
    p.add_argument("--data_dir",        type=str,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--grad_accum_steps",type=int,   default=None)
    p.add_argument("--max_steps",       type=int,   default=None)
    p.add_argument("--max_lr",          type=float, default=None)
    p.add_argument("--warmup_steps",    type=int,   default=None)
    p.add_argument("--weight_decay",    type=float, default=None)
    p.add_argument("--grad_clip",       type=float, default=None)
    p.add_argument("--eval_interval",   type=int,   default=None)
    p.add_argument("--save_interval",   type=int,   default=None)
    p.add_argument("--eval_steps",      type=int,   default=None)
    p.add_argument("--log_interval",    type=int,   default=None)
    p.add_argument("--checkpoint_dir",  type=str,   default=None)
    p.add_argument("--log_dir",         type=str,   default=None)
    p.add_argument("--dtype",           type=str,   default=None,
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--use_wandb",       action="store_true", default=None)
    p.add_argument("--wandb_project",   type=str,   default=None)
    p.add_argument("--num_workers",     type=int,   default=None)

    return p.parse_args()


def _apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Apply any non-None CLI arguments on top of the preset config."""
    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k in TrainConfig.__dataclass_fields__
    }
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Main train function (single process or one DDP worker)
# ---------------------------------------------------------------------------

def train(
    rank: int = 0,
    world_size: int = 1,
    args: argparse.Namespace = None,
) -> None:
    is_ddp    = world_size > 1
    is_master = rank == 0

    if is_ddp:
        setup_ddp(rank, world_size)

    # ---- Build configs ---------------------------------------------------
    model_preset_fn = {
        "nano":       ModelConfig.nano,
        "small":      ModelConfig.small,
        "gpt2":       ModelConfig.gpt2,
        "gpt2_medium":ModelConfig.gpt2_medium,
    }
    train_preset_fn = {
        "nano":       TrainConfig.nano,
        "small":      TrainConfig.nano,    # override max_steps / lr via CLI
        "gpt2":       TrainConfig.gpt2,
        "gpt2_medium":TrainConfig.gpt2,
    }

    model_cfg = model_preset_fn[args.preset]()
    train_cfg = train_preset_fn[args.preset]()
    train_cfg = _apply_overrides(train_cfg, args)

    # seq_len must match model max_seq_len
    train_cfg.seq_len = model_cfg.max_seq_len

    if is_master:
        print(model_cfg)
        print(train_cfg)

    # ---- Seed --------------------------------------------------------------
    # Same seed value on every rank: makes model init deterministic and
    # identical across ranks (defense in depth alongside DDP's own
    # parameter broadcast), and is required by create_dataloaders' shared
    # DistributedSampler shuffle permutation below. Must run BEFORE the
    # model is constructed.
    set_seed(train_cfg.seed)

    # ---- Device ----------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif is_ddp:
        device = torch.device(f"cuda:{rank}")
    else:
        device = get_device()

    if is_master:
        print(f"Device: {device}")

    # ---- Verify data exists ---------------------------------------------
    data_path = Path(train_cfg.data_dir) / train_cfg.dataset / "train.bin"
    if not data_path.exists():
        print(
            f"\nERROR: Training data not found at {data_path}\n"
            f"Run first:  python prepare_data.py --dataset {train_cfg.dataset}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Data loaders ---------------------------------------------------
    # rank/world_size/seed enable DistributedSampler-based sharding when
    # is_ddp — without it every rank would redundantly iterate the full
    # dataset with no partitioning (see create_dataloaders docstring).
    train_loader, val_loader = create_dataloaders(
        data_dir=train_cfg.data_dir,
        dataset=train_cfg.dataset,
        seq_len=train_cfg.seq_len,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        rank=rank,
        world_size=world_size,
        seed=train_cfg.seed,
    )

    if is_master:
        print(
            f"Train batches/epoch: {len(train_loader):,}  |  "
            f"Val batches: {len(val_loader):,}"
        )

    # ---- Model ----------------------------------------------------------
    model = GPT(model_cfg)

    # ---- Trainer --------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    trainer.train(resume_from=args.resume)

    if is_ddp:
        teardown_ddp()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Detect DDP via torchrun environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK",       0))

    if world_size > 1:
        # torchrun sets these; run as a single DDP worker
        train(rank=rank, world_size=world_size, args=args)
    else:
        train(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
