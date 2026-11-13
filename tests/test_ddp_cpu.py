"""
A real two-process DDP run on CPU (gloo), through the actual Trainer.

The other DDP tests check the pieces in isolation: the no_sync() predicate and
the DistributedSampler sharding. This one runs the whole thing: two processes,
a real process group, the Trainer's own gradient-accumulation loop with
no_sync() on every micro-step but the last, gradient clipping and AdamW.

What it proves, on the real code path:
  1. the two replicas end with bit-identical weights (DDP kept them in sync);
  2. those weights equal a single process trained on both ranks' batches
     concatenated, i.e. DDP averaged the gradients correctly, and no
     micro-step's contribution was dropped or double-counted by no_sync().

It does not prove anything about NCCL or GPUs; it proves the training logic.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from nanolm.model.config import ModelConfig, TrainConfig
from nanolm.model.gpt import GPT
from nanolm.training.trainer import Trainer, setup_ddp, teardown_ddp

WORLD = 2
STEPS = 3
ACCUM = 2          # two micro-steps: exercises the no_sync() path
BATCH = 2          # per rank, per micro-step
SEQ = 16
VOCAB = 64

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with gloo is not available",
)


def _model_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=VOCAB, max_seq_len=SEQ, n_layers=2, n_heads=2,
                       d_model=32, dropout=0.0)


def _train_cfg(tmp: str, batch: int) -> TrainConfig:
    cfg = TrainConfig()
    cfg.batch_size = batch
    cfg.grad_accum_steps = ACCUM
    cfg.seq_len = SEQ
    cfg.max_steps = STEPS
    cfg.warmup_steps = 1
    cfg.max_lr = 1e-2            # big enough that a wrong gradient shows
    cfg.min_lr = 1e-3
    cfg.grad_clip = 1.0
    cfg.use_amp = False
    cfg.dtype = "float32"
    cfg.eval_interval = 10**9
    cfg.save_interval = 10**9
    cfg.log_interval = 10**9
    cfg.eval_steps = 1
    cfg.use_tensorboard = False
    cfg.use_wandb = False
    cfg.checkpoint_dir = os.path.join(tmp, "ckpt")
    cfg.log_dir = os.path.join(tmp, "logs")
    cfg.dist_backend = "gloo"
    return cfg


def _micro_batches(rank: int) -> torch.Tensor:
    """STEPS*ACCUM micro-batches of (BATCH, SEQ+1) tokens, different per rank."""
    g = torch.Generator().manual_seed(100 + rank)
    return torch.randint(0, VOCAB, (STEPS * ACCUM, BATCH, SEQ + 1), generator=g)


class _Fixed(Dataset):
    """Yields the given sequences in order, as (x, y) pairs."""

    def __init__(self, seqs: torch.Tensor) -> None:
        self.seqs = seqs.reshape(-1, SEQ + 1)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int):
        s = self.seqs[i]
        return s[:-1], s[1:]


def _loader(seqs: torch.Tensor, batch: int) -> DataLoader:
    return DataLoader(_Fixed(seqs), batch_size=batch, shuffle=False)


def _worker(rank: int, port: int, tmp: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = setup_ddp(rank, WORLD, backend="gloo")
    try:
        torch.manual_seed(0)
        model = GPT(_model_cfg()).double()
        loader = _loader(_micro_batches(rank), BATCH)
        trainer = Trainer(model, loader, loader, _train_cfg(tmp, BATCH),
                          device=device, rank=rank, world_size=WORLD)
        trainer.train()
        torch.save(trainer._raw_model.state_dict(), os.path.join(tmp, f"rank{rank}.pt"))
    finally:
        teardown_ddp()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def ddp_run(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("ddp"))
    mp.spawn(_worker, args=(_free_port(), tmp), nprocs=WORLD, join=True)
    return tmp, [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(WORLD)]


def test_replicas_stay_identical(ddp_run):
    _, (w0, w1) = ddp_run
    for name in w0:
        assert torch.equal(w0[name], w1[name]), f"{name} diverged between ranks"


def test_ddp_equals_one_process_on_the_combined_batch(ddp_run, tmp_path):
    """Gradient averaging across ranks == one process on both ranks' data."""
    _, (w0, _) = ddp_run
    # Micro-batch k of the single process = rank 0's micro-batch k followed by
    # rank 1's, so its mean loss is the mean of the two ranks' mean losses.
    combined = torch.cat([_micro_batches(0), _micro_batches(1)], dim=1)
    ref = _single_process(combined, 2 * BATCH, tmp_path / "both")
    for name in ref:
        # float64 end to end, so the only difference left is summation order.
        torch.testing.assert_close(w0[name], ref[name], rtol=1e-9, atol=1e-11,
                                   msg=lambda m, n=name: f"{n}: {m}")

    # Controls, so a pass means something. Training moved the weights...
    torch.manual_seed(0)
    init = GPT(_model_cfg()).double().state_dict()
    assert max((ref[n] - init[n]).abs().max().item() for n in ref) > 1e-3
    # ...and the comparison can tell "averaged both ranks" from "used one rank":
    # a run on rank 0's data alone must NOT match.
    only_rank0 = _single_process(_micro_batches(0), BATCH, tmp_path / "one")
    gap = max((w0[n] - only_rank0[n]).abs().max().item() for n in w0)
    assert gap > 1e-4


def _single_process(seqs: torch.Tensor, batch: int, tmp) -> dict:
    torch.manual_seed(0)
    model = GPT(_model_cfg()).double()
    trainer = Trainer(model, _loader(seqs, batch), _loader(seqs, batch),
                      _train_cfg(str(tmp), batch),
                      device=torch.device("cpu"), rank=0, world_size=1)
    trainer.train()
    return trainer._raw_model.state_dict()


def test_nccl_without_gpus_fails_with_a_clear_message():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(RuntimeError, match="gloo"):
        setup_ddp(0, 1, backend="nccl")
