"""
Numerical correctness tests for the training loop (07-nanolm.md Priority 3).

Tests verify:
  - Gradient accumulation: accumulated micro-batch grads (loss / accum_steps)
    are numerically equivalent to a single full-batch backward pass. If the
    division were missing or wrong, the effective learning rate would
    silently scale with grad_accum_steps.
  - Clipping order: scaler.unscale_() is called exactly once per optimiser
    step, before clip_grad_norm_(), and only after the full accumulation
    loop (not once per micro-step).
  - DDP gradient-sync predicate: no_sync() must be active on every
    micro-step except the last one in the accumulation window.
  - DDP data sharding: ranks must see disjoint, non-overlapping shards of
    the dataset, and per-rank RNG must differ for data order while model
    init stays identical across ranks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from nanolm.data.dataset import TokenizedDataset, create_dataloaders
from nanolm.model.config import ModelConfig, TrainConfig
from nanolm.model.gpt import GPT
from nanolm.training.trainer import Trainer, _should_sync_grads
from nanolm.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------

class TestGradientAccumulation:
    def test_accumulated_grad_matches_single_batch_grad(self):
        """
        Splitting a batch into N equal-sized microbatches, dividing each
        microbatch's mean loss by N before backward, and summing the
        resulting .grad tensors must reproduce the gradient of a single
        backward pass over the full batch with mean-reduction loss.

        Full-batch grad:  (1/(B*T)) * sum_i dL_i/dw
        Accumulated grad: sum_{m=1..N} (1/N) * (1/(micro_bs*T)) * sum_{i in m} dL_i/dw
                         = (1/(N*micro_bs*T)) * sum_i dL_i/dw = (1/(B*T)) * sum_i dL_i/dw
        (since B = N * micro_bs). These must match to floating-point precision.
        """
        cfg = ModelConfig(
            vocab_size=64, max_seq_len=16, n_layers=2, n_heads=2, d_model=16,
            dropout=0.0, bias=False,
        )
        torch.manual_seed(7)
        model_full = GPT(cfg)
        model_accum = GPT(cfg)
        model_accum.load_state_dict(model_full.state_dict())

        accum_steps, micro_bs, T = 4, 3, 8
        B = accum_steps * micro_bs
        torch.manual_seed(99)
        idx = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))

        # ---- Ground truth: single full-batch backward ----
        model_full.zero_grad(set_to_none=True)
        _, loss_full, _ = model_full(idx, targets)
        loss_full.backward()

        # ---- Accumulated micro-batch backward, mirroring Trainer.train() ----
        model_accum.zero_grad(set_to_none=True)
        for m in range(accum_steps):
            xb = idx[m * micro_bs:(m + 1) * micro_bs]
            yb = targets[m * micro_bs:(m + 1) * micro_bs]
            _, loss, _ = model_accum(xb, yb)
            (loss / accum_steps).backward()

        for (n1, p1), (n2, p2) in zip(
            model_full.named_parameters(), model_accum.named_parameters()
        ):
            assert p1.grad is not None and p2.grad is not None, f"missing grad at {n1}"
            max_diff = (p1.grad - p2.grad).abs().max().item()
            assert torch.allclose(p1.grad, p2.grad, atol=1e-4, rtol=1e-3), (
                f"Gradient mismatch at {n1}: max abs diff {max_diff:.3e}. "
                f"Accumulated gradients are not equivalent to a single full-batch "
                f"backward — the loss/accum_steps division is missing or wrong, "
                f"so the effective LR silently scales with grad_accum_steps."
            )

    def test_missing_division_would_be_caught(self):
        """
        Negative control: confirm the test above actually has teeth by
        showing that omitting the /accum_steps division produces a grad
        that is ~accum_steps times too large (and the assertion would fail).
        """
        cfg = ModelConfig(
            vocab_size=64, max_seq_len=16, n_layers=2, n_heads=2, d_model=16,
            dropout=0.0, bias=False,
        )
        torch.manual_seed(7)
        model_full = GPT(cfg)
        model_buggy = GPT(cfg)
        model_buggy.load_state_dict(model_full.state_dict())

        accum_steps, micro_bs, T = 4, 3, 8
        B = accum_steps * micro_bs
        torch.manual_seed(99)
        idx = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))

        model_full.zero_grad(set_to_none=True)
        _, loss_full, _ = model_full(idx, targets)
        loss_full.backward()

        model_buggy.zero_grad(set_to_none=True)
        for m in range(accum_steps):
            xb = idx[m * micro_bs:(m + 1) * micro_bs]
            yb = targets[m * micro_bs:(m + 1) * micro_bs]
            _, loss, _ = model_buggy(xb, yb)
            loss.backward()   # BUG: no division by accum_steps

        p_full = dict(model_full.named_parameters())
        p_buggy = dict(model_buggy.named_parameters())
        name = "transformer.h.0.attn.c_proj.weight"
        ratio = (p_buggy[name].grad.norm() / p_full[name].grad.norm()).item()
        assert ratio > 2.0, (
            "expected the undivided grad to be much larger than the correct one "
            f"(got ratio {ratio:.2f}); the equivalence test above would not have "
            "caught a missing division"
        )


# ---------------------------------------------------------------------------
# Clipping order
# ---------------------------------------------------------------------------

class TestClippingOrder:
    def _make_trainer(self, tmp_path: Path, grad_accum_steps: int) -> Trainer:
        cfg = ModelConfig(
            vocab_size=64, max_seq_len=16, n_layers=2, n_heads=2, d_model=16,
            dropout=0.0, bias=False,
        )
        model = GPT(cfg)

        n_tokens = 2000
        rng = np.random.default_rng(0)
        ids = rng.integers(0, cfg.vocab_size, size=n_tokens, dtype=np.uint16)
        data_dir = tmp_path / "ddata"
        data_dir.mkdir()
        ids[: int(0.9 * n_tokens)].tofile(data_dir / "train.bin")
        ids[int(0.9 * n_tokens):].tofile(data_dir / "val.bin")

        from nanolm.data.dataset import create_dataloaders
        train_loader, val_loader = create_dataloaders(
            data_dir=str(tmp_path), dataset="ddata", seq_len=16, batch_size=2,
        )

        train_cfg = TrainConfig(
            max_steps=1, batch_size=2, grad_accum_steps=grad_accum_steps,
            seq_len=16, eval_interval=10_000, save_interval=10_000,
            log_interval=10_000, use_amp=False, dtype="float32",
            use_tensorboard=False, use_wandb=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        return Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            train_cfg=train_cfg, device=torch.device("cpu"),
        )

    def test_unscale_called_once_before_clip_after_full_accumulation(self, tmp_path):
        call_order: list[str] = []

        orig_unscale = torch.amp.GradScaler.unscale_
        orig_clip = torch.nn.utils.clip_grad_norm_

        def spy_unscale(self, optimizer):
            call_order.append("unscale")
            return orig_unscale(self, optimizer)

        def spy_clip(*args, **kwargs):
            call_order.append("clip")
            return orig_clip(*args, **kwargs)

        grad_accum_steps = 3
        trainer = self._make_trainer(tmp_path, grad_accum_steps)

        import nanolm.training.trainer as trainer_mod
        with (
            patch.object(torch.amp.GradScaler, "unscale_", spy_unscale),
            patch.object(trainer_mod.nn.utils, "clip_grad_norm_", spy_clip),
        ):
            trainer.train()

        # Exactly one optimiser step happened (max_steps=1) => exactly one
        # unscale + one clip, NOT one per micro-step (there were 3 micro-steps).
        assert call_order.count("unscale") == 1, call_order
        assert call_order.count("clip") == 1, call_order
        assert call_order == ["unscale", "clip"], (
            f"expected unscale_() strictly before clip_grad_norm_(), got {call_order}"
        )


# ---------------------------------------------------------------------------
# DDP gradient-sync predicate
# ---------------------------------------------------------------------------

class TestDDPSyncPredicate:
    def test_only_last_microstep_syncs(self):
        accum = 5
        results = [_should_sync_grads(m, accum) for m in range(accum)]
        assert results == [False, False, False, False, True]

    def test_single_microstep_always_syncs(self):
        # grad_accum_steps=1: the only micro-step is also the last one.
        assert _should_sync_grads(0, 1) is True

    def test_never_syncs_before_final_step(self):
        accum = 8
        for m in range(accum - 1):
            assert _should_sync_grads(m, accum) is False
        assert _should_sync_grads(accum - 1, accum) is True


# ---------------------------------------------------------------------------
# DDP data sharding (no overlap across ranks) + seeding
# ---------------------------------------------------------------------------

def _make_token_bin(tmp_path: Path, n_tokens: int, vocab_size: int) -> Path:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    path = tmp_path / "train.bin"
    ids.tofile(path)
    return path


class TestDDPDataSharding:
    def test_distributed_sampler_shards_do_not_overlap(self, tmp_path):
        """
        Simulates world_size=4 by instantiating one DistributedSampler per
        rank (all sharing the SAME seed, as required) against the same
        dataset, and asserts every index is assigned to exactly one rank —
        i.e. the shards partition the dataset with zero overlap.
        """
        bin_path = _make_token_bin(tmp_path, n_tokens=2000, vocab_size=64)
        ds = TokenizedDataset(bin_path, seq_len=16)

        world_size = 4
        seed = 1337
        per_rank_indices = []
        for rank in range(world_size):
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank,
                shuffle=True, seed=seed, drop_last=True,
            )
            per_rank_indices.append(set(sampler))

        # No pairwise overlap between any two ranks' index sets.
        for i in range(world_size):
            for j in range(i + 1, world_size):
                overlap = per_rank_indices[i] & per_rank_indices[j]
                assert not overlap, (
                    f"rank {i} and rank {j} shards overlap at indices {overlap} — "
                    f"DDP ranks would train on duplicate data"
                )

        # Every rank gets a (roughly) equal, non-empty shard.
        sizes = [len(s) for s in per_rank_indices]
        assert all(s > 0 for s in sizes)
        assert max(sizes) - min(sizes) <= 1

    def test_distributed_sampler_requires_matching_seed_across_ranks(self, tmp_path):
        """
        If ranks used DIFFERENT seeds for the sampler (the wrong reading of
        "per-rank seeding"), the shared permutation each rank partitions
        from would differ, and shards would no longer reliably tile the
        dataset without gaps/overlap. This is a negative control showing
        why create_dataloaders always passes one shared `seed` to every
        rank's DistributedSampler.
        """
        bin_path = _make_token_bin(tmp_path, n_tokens=2000, vocab_size=64)
        ds = TokenizedDataset(bin_path, seq_len=16)
        world_size = 4

        same_seed_indices = set()
        for rank in range(world_size):
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank,
                shuffle=True, seed=1337, drop_last=True,
            )
            same_seed_indices |= set(sampler)

        mismatched_seed_indices = set()
        for rank in range(world_size):
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank,
                shuffle=True, seed=1337 + rank, drop_last=True,   # BUG: differs per rank
            )
            mismatched_seed_indices |= set(sampler)

        # With a shared seed, the union of shards covers (~all of) the dataset.
        assert len(same_seed_indices) >= len(ds) - world_size

        # With mismatched per-rank seeds each rank re-derives an
        # INDEPENDENT full permutation before slicing its own 1/world_size
        # stride — some dataset indices land in every rank's shard
        # (duplicated) while others land in none, unlike the shared-seed
        # case which tiles the dataset exactly once.
        overlap_count = 0
        seen = set()
        for rank in range(world_size):
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank,
                shuffle=True, seed=1337 + rank, drop_last=True,
            )
            idxs = set(sampler)
            overlap_count += len(idxs & seen)
            seen |= idxs
        assert overlap_count > 0, (
            "expected mismatched per-rank seeds to produce overlapping shards"
        )

    def test_create_dataloaders_ddp_mode_shards_train_and_val(self, tmp_path):
        """End-to-end: create_dataloaders(world_size=2) yields disjoint
        train batches for rank 0 vs rank 1 across a full pass."""
        base = tmp_path / "shard_ds"
        base.mkdir()
        bin_path = _make_token_bin(base, n_tokens=3000, vocab_size=64)
        # val.bin required by create_dataloaders
        np.array(
            np.random.default_rng(1).integers(0, 64, size=500), dtype=np.uint16
        ).tofile(base / "val.bin")

        loader0, val0 = create_dataloaders(
            data_dir=tmp_path, dataset="shard_ds", seq_len=16, batch_size=4,
            rank=0, world_size=2, seed=42,
        )
        loader1, val1 = create_dataloaders(
            data_dir=tmp_path, dataset="shard_ds", seq_len=16, batch_size=4,
            rank=1, world_size=2, seed=42,
        )

        assert isinstance(loader0.sampler, DistributedSampler)
        idx0 = set(loader0.sampler)
        idx1 = set(loader1.sampler)
        assert not (idx0 & idx1), "rank 0 and rank 1 train shards overlap"

        val_idx0 = set(val0.sampler)
        val_idx1 = set(val1.sampler)
        assert not (val_idx0 & val_idx1), "rank 0 and rank 1 val shards overlap"


class TestSeedReproducibility:
    def test_same_seed_gives_identical_model_init(self):
        """
        Model init must match across ranks. Calling set_seed with the same
        value before constructing each model (as train.py does identically
        on every DDP rank) must produce byte-identical initial weights.
        """
        cfg = ModelConfig(vocab_size=64, max_seq_len=16, n_layers=2, n_heads=2, d_model=16)

        set_seed(1337)
        model_rank0 = GPT(cfg)

        set_seed(1337)
        model_rank1 = GPT(cfg)

        for (n0, p0), (n1, p1) in zip(
            model_rank0.named_parameters(), model_rank1.named_parameters()
        ):
            assert torch.equal(p0, p1), f"Mismatch at {n0}: seeded init not reproducible"

    def test_different_seed_gives_different_model_init(self):
        cfg = ModelConfig(vocab_size=64, max_seq_len=16, n_layers=2, n_heads=2, d_model=16)

        set_seed(1337)
        model_a = GPT(cfg)
        set_seed(999)
        model_b = GPT(cfg)

        any_diff = any(
            not torch.equal(p_a, p_b)
            for p_a, p_b in zip(model_a.parameters(), model_b.parameters())
        )
        assert any_diff, "different seeds unexpectedly produced identical init"


class TestInfiniteDataLoaderEpochAdvance:
    def test_set_epoch_called_on_distributed_sampler_wraparound(self, tmp_path):
        """
        InfiniteDataLoader must call sampler.set_epoch() every time it
        wraps around, so the shared shuffle-then-partition scheme keeps
        producing non-overlapping shards (and fresh shuffles) on every
        pass, not just the first.
        """
        from nanolm.data.dataset import InfiniteDataLoader

        bin_path = _make_token_bin(tmp_path, n_tokens=200, vocab_size=64)
        ds = TokenizedDataset(bin_path, seq_len=8)
        sampler = DistributedSampler(
            ds, num_replicas=2, rank=0, shuffle=True, seed=1, drop_last=True,
        )
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=2, sampler=sampler, drop_last=True)

        inf_loader = InfiniteDataLoader(loader)
        n_batches_per_pass = len(loader)

        assert sampler.epoch == 0
        for _ in range(n_batches_per_pass + 1):   # force one wraparound
            next(inf_loader)
        assert sampler.epoch == 1, "sampler.set_epoch() was not advanced on wraparound"
