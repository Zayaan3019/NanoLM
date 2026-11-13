"""
Training loop with:
  - Gradient accumulation  (effective large batch on limited VRAM / CPU RAM)
  - Mixed-precision AMP    (float16 / bfloat16 — auto-disabled on CPU)
  - Gradient norm clipping (prevents exploding gradients)
  - Cosine LR schedule     (linear warmup + cosine decay)
  - Periodic evaluation    (val loss + perplexity)
  - Best-model checkpointing
  - Periodic step checkpoints
  - TensorBoard logging    (loss, lr, grad norm, tokens/sec)
  - Optional W&B logging
  - DDP support            (multi-GPU via torch.distributed)
  - Intel Arc / XPU support via IPEX (optional dependency)

The training loop intentionally avoids an epoch abstraction: we train for
`max_steps` gradient steps regardless of dataset size, using an infinite
DataLoader wrapper.  This matches GPT-2/3/4 and modern LLM training practice.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from nanolm.model.config import ModelConfig, TrainConfig
from nanolm.model.gpt import GPT
from nanolm.training.scheduler import get_lr
from nanolm.data.dataset import InfiniteDataLoader
from nanolm.utils.checkpoint import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Auto-detect the best available compute device.

    Priority: CUDA > Intel XPU (Arc) > MPS (Apple Silicon) > CPU

    Intel Arc users: install Intel Extension for PyTorch first:
        pip install intel-extension-for-pytorch
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Intel Arc via IPEX
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore  # noqa: F401
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    except ImportError:
        pass

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Self-contained training engine for the GPT language model.

    Designed to run on a single device (CPU / GPU / XPU) or across multiple
    GPUs via PyTorch DDP.  In the DDP case, rank-0 handles logging and
    checkpointing; all ranks participate in the forward/backward passes.

    Args:
        model:        Initialised GPT model (not yet on device).
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        train_cfg:    Training hyperparameters.
        device:       Target device.  None = auto-detect.
        rank:         DDP rank (0 for single-device training).
        world_size:   Total number of DDP processes (1 for single-device).
    """

    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainConfig,
        device: Optional[torch.device] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.cfg        = train_cfg
        self.rank       = rank
        self.world_size = world_size
        self.is_master  = rank == 0

        self.device = device or get_device()

        # ---- Model -------------------------------------------------------
        self.model = model.to(self.device)

        # Optional Intel Extension for PyTorch optimisations on XPU/CPU
        if self.device.type == "xpu":
            try:
                import intel_extension_for_pytorch as ipex  # type: ignore
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
            except ImportError:
                pass

        if world_size > 1:
            # device_ids is this process's own GPU (its local index), and must
            # be None on CPU; the global rank is wrong on any node but the first.
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
            )

        self._raw_model: GPT = self.model.module if world_size > 1 else self.model  # type: ignore

        # ---- Optimiser ---------------------------------------------------
        self.optimizer = self._raw_model.configure_optimizer(
            weight_decay=train_cfg.weight_decay,
            lr=train_cfg.max_lr,
            betas=(train_cfg.beta1, train_cfg.beta2),
            device_type=self.device.type,
        )

        # ---- Mixed precision ---------------------------------------------
        _dtype_map = {
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.amp_dtype = _dtype_map[train_cfg.dtype]
        self.use_amp   = (
            train_cfg.use_amp
            and self.device.type in ("cuda", "xpu")
        )
        # GradScaler is only needed for float16 (bfloat16 doesn't overflow)
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=(self.use_amp and self.amp_dtype == torch.float16),
        )

        # ---- Data --------------------------------------------------------
        self.train_iter = InfiniteDataLoader(train_loader)
        self.val_loader = val_loader

        # ---- Logging -----------------------------------------------------
        self.writer = None
        if self.is_master and train_cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                Path(train_cfg.log_dir).mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=train_cfg.log_dir)
            except Exception as e:
                print(f"[warn] TensorBoard unavailable ({type(e).__name__}) — skipping TB logging")

        self.wandb_run = None
        if self.is_master and train_cfg.use_wandb:
            try:
                import wandb  # type: ignore
                self.wandb_run = wandb.init(
                    project=train_cfg.wandb_project,
                    config={
                        **self._raw_model.config.to_dict(),
                        **train_cfg.to_dict(),
                    },
                )
            except ImportError:
                print("[warn] wandb not installed — skipping W&B logging")

        # ---- State -------------------------------------------------------
        self.step           = 0
        self.best_val_loss  = float("inf")
        Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Run the model over `eval_steps` validation batches and return the
        mean loss.

        Under DDP, `self.val_loader` is sharded across ranks (see
        create_dataloaders), so each rank only sees 1/world_size of the
        validation batches. We all-reduce (sum + count) across ranks so the
        reported val_loss reflects the full validation set every rank
        contributed to, not just this rank's local slice — otherwise ranks
        would log different "val_loss" values and only rank 0's (partial)
        number would ever be checkpointed against.
        """
        self.model.eval()
        loss_sum   = 0.0
        loss_count = 0

        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.cfg.eval_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                _, loss, _ = self.model(x, y)
            loss_sum   += loss.item()
            loss_count += 1

        if self.world_size > 1:
            stats = torch.tensor([loss_sum, float(loss_count)], device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            loss_sum, loss_count = stats[0].item(), stats[1].item()

        self.model.train()
        return loss_sum / loss_count if loss_count > 0 else float("inf")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Run the training loop.

        Args:
            resume_from: Path to a checkpoint file.  If given, the model
                         weights, optimizer state, and step counter are
                         restored before training continues.
        """
        if resume_from is not None:
            self.step = load_checkpoint(
                resume_from, self._raw_model, self.optimizer
            )
            if self.is_master:
                print(f"Resumed from step {self.step:,} ({resume_from})")

        self.model.train()

        n_params = self._raw_model.count_params(non_embedding=True)
        eff_batch_tokens = self.cfg.batch_size * self.cfg.grad_accum_steps * self.cfg.seq_len

        if self.is_master:
            print("=" * 70)
            print(self._raw_model)
            print(f"  Total trainable params : {n_params / 1e6:.2f} M (non-embedding)")
            print(f"  Training steps         : {self.cfg.max_steps:,}")
            print(f"  Micro-batch size       : {self.cfg.batch_size}")
            print(f"  Gradient accum steps   : {self.cfg.grad_accum_steps}")
            print(f"  Effective tokens/step  : {eff_batch_tokens:,}")
            print(f"  Device                 : {self.device}")
            print(f"  AMP dtype              : {self.cfg.dtype} (enabled={self.use_amp})")
            print("=" * 70)

        t0 = time.perf_counter()

        while self.step < self.cfg.max_steps:

            # ---- LR schedule -------------------------------------------
            lr = get_lr(
                self.step,
                self.cfg.warmup_steps,
                self.cfg.max_steps,
                self.cfg.max_lr,
                self.cfg.min_lr,
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # ---- Gradient accumulation loop ----------------------------
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(self.cfg.grad_accum_steps):
                x, y = next(self.train_iter)
                x, y = x.to(self.device), y.to(self.device)

                # In DDP, synchronise gradients only on the last micro-step
                # (avoids expensive all-reduce on every accumulation step)
                sync = _should_sync_grads(micro_step, self.cfg.grad_accum_steps)
                ctx  = (
                    torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    )
                )
                if self.world_size > 1:
                    with ctx:
                        with (self.model.no_sync() if not sync else _null_ctx()):
                            _, loss, _ = self.model(x, y)
                            loss = loss / self.cfg.grad_accum_steps
                else:
                    with ctx:
                        _, loss, _ = self.model(x, y)
                        loss = loss / self.cfg.grad_accum_steps

                accum_loss += loss.item()
                self.scaler.scale(loss).backward()

            # ---- Gradient clipping + optimiser step --------------------
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            ).item()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1

            # ---- Logging -----------------------------------------------
            if self.is_master and self.step % self.cfg.log_interval == 0:
                t1   = time.perf_counter()
                dt   = t1 - t0
                tps  = eff_batch_tokens * self.cfg.log_interval / dt

                print(
                    f"step {self.step:>6,} | loss {accum_loss:.4f} | "
                    f"lr {lr:.2e} | gnorm {grad_norm:.3f} | {tps:,.0f} tok/s"
                )

                if self.writer:
                    self.writer.add_scalar("train/loss",        accum_loss, self.step)
                    self.writer.add_scalar("train/lr",          lr,          self.step)
                    self.writer.add_scalar("train/grad_norm",   grad_norm,   self.step)
                    self.writer.add_scalar("train/tokens_per_s", tps,        self.step)

                if self.wandb_run:
                    self.wandb_run.log({
                        "train/loss": accum_loss, "train/lr": lr,
                        "train/grad_norm": grad_norm, "step": self.step,
                    })

                t0 = time.perf_counter()

            # ---- Evaluation --------------------------------------------
            if self.step % self.cfg.eval_interval == 0:
                val_loss = self.evaluate()
                val_ppl  = torch.exp(torch.tensor(val_loss)).item()

                if self.is_master:
                    print(
                        f"  >> [eval] step {self.step:,} | val_loss {val_loss:.4f} "
                        f"| val_ppl {val_ppl:.2f}"
                    )
                    if self.writer:
                        self.writer.add_scalar("val/loss",       val_loss, self.step)
                        self.writer.add_scalar("val/perplexity", val_ppl,  self.step)
                    if self.wandb_run:
                        self.wandb_run.log({"val/loss": val_loss, "val/perplexity": val_ppl})

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(
                            path=Path(self.cfg.checkpoint_dir) / "best.pt",
                            model=self._raw_model,
                            optimizer=self.optimizer,
                            step=self.step,
                            val_loss=val_loss,
                            model_config=self._raw_model.config,
                            train_config=self.cfg,
                        )
                        print(
                            f"  >> [ckpt] Best model saved  "
                            f"(val_loss={val_loss:.4f})"
                        )

            # ---- Periodic checkpoint -----------------------------------
            if self.is_master and self.step % self.cfg.save_interval == 0:
                save_checkpoint(
                    path=Path(self.cfg.checkpoint_dir) / f"step_{self.step:07d}.pt",
                    model=self._raw_model,
                    optimizer=self.optimizer,
                    step=self.step,
                    val_loss=self.best_val_loss,
                    model_config=self._raw_model.config,
                    train_config=self.cfg,
                )

        # ---- Finalise -----------------------------------------------
        if self.is_master:
            print("=" * 70)
            print(f"Training complete at step {self.step:,}.")
            print(f"Best val loss: {self.best_val_loss:.4f}")
            if self.writer:
                self.writer.close()
            if self.wandb_run:
                self.wandb_run.finish()


# ---------------------------------------------------------------------------
# DDP setup / teardown helpers
# ---------------------------------------------------------------------------

def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    local_rank: Optional[int] = None,
) -> torch.device:
    """
    Initialise the process group and return the device this process uses.

    rank:       GLOBAL rank, 0..world_size-1 across every machine.
    local_rank: index of this process on its own machine (torchrun's
                LOCAL_RANK). That, not the global rank, picks the GPU: with
                two 8-GPU nodes, global rank 9 is cuda:1 on the second node,
                and cuda:9 does not exist. Defaults to $LOCAL_RANK, then rank.
    backend:    "nccl" for GPUs, "gloo" for CPU. gloo is what the test suite
                uses to run a real two-process job on a laptop.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError(
            "dist_backend='nccl' needs CUDA GPUs; use 'gloo' to run DDP on CPU"
        )
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def teardown_ddp() -> None:
    """Clean up the process group after DDP training."""
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# DDP gradient-sync predicate (pure function — unit-testable without DDP)
# ---------------------------------------------------------------------------

def _should_sync_grads(micro_step: int, grad_accum_steps: int) -> bool:
    """
    True only on the LAST micro-step of a gradient-accumulation window.

    Every other micro-step should run inside `model.no_sync()` so DDP does
    not all-reduce gradients on each micro-batch — only once per optimiser
    step, after accumulation completes. Wrong on either end wastes
    bandwidth (over-sync) or silently drops gradient contributions
    (under-sync, if no_sync() were applied on the final step).
    """
    return micro_step == grad_accum_steps - 1


# ---------------------------------------------------------------------------
# Null context manager (for DDP no_sync path on single-device)
# ---------------------------------------------------------------------------

class _null_ctx:
    def __enter__(self) -> "_null_ctx":
        return self

    def __exit__(self, *args: object) -> None:
        pass
