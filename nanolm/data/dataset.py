"""
Dataset and DataLoader utilities.

Data pipeline:
  1. `prepare_data.py` downloads raw text, trains a BPE tokenizer,
     tokenises the corpus, and saves two memory-mapped binary files:
       data/<dataset>/train.bin  — uint16 token ids
       data/<dataset>/val.bin    — uint16 token ids

  2. `TokenizedDataset` memory-maps those files and serves (x, y) pairs
     where x = tokens[i : i+seq_len] and y = tokens[i+1 : i+seq_len+1].
     Using np.memmap means only the pages actually accessed are loaded —
     the full dataset never sits in RAM, enabling training on corpora
     much larger than available memory.

Token id dtype:
  uint16 supports up to 65535 distinct tokens, which covers GPT-2's
  50257-token vocabulary with room to spare.  For vocabularies > 65535
  change to uint32 in both prepare_data.py and here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenizedDataset(Dataset):
    """
    Memory-mapped dataset of pre-tokenised token ids.

    Each call to __getitem__ returns a random (x, y) pair of length seq_len
    sampled from the memory-mapped file without loading the entire array.
    """

    def __init__(self, data_path: str | Path, seq_len: int) -> None:
        self.seq_len = seq_len
        # mode='r' = read-only memmap — safe for multiprocessing workers
        self.data = np.memmap(str(data_path), dtype=np.uint16, mode="r")

        if len(self.data) <= seq_len:
            raise ValueError(
                f"Dataset at {data_path} has {len(self.data):,} tokens, "
                f"which is not enough for a single sequence of length {seq_len + 1}."
            )

    def __len__(self) -> int:
        # Every starting index from 0 to len-seq_len-1 is valid
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Grab seq_len+1 tokens; split into input and target (shifted by 1)
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_dir: str | Path,
    dataset: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 1337,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        data_dir:    Root data directory (contains <dataset>/train.bin etc.)
        dataset:     Dataset name subfolder (e.g. "shakespeare", "tinystories")
        seq_len:     Sequence length (tokens).
        batch_size:  Micro-batch size per forward pass.
        num_workers: DataLoader worker processes (0 = main process only).
        pin_memory:  Pin memory for faster CPU→GPU transfer.
        rank:        DDP rank of this process (0 for single-device training).
        world_size:  Total number of DDP processes (1 for single-device).
        seed:        Shared shuffle seed. MUST be identical across all ranks —
                     DistributedSampler derives one shuffle permutation from
                     it and then partitions that permutation by rank, which
                     is what guarantees ranks see disjoint, non-overlapping
                     shards. A per-rank-varying seed here would NOT shard
                     correctly (it would risk overlap/gaps between ranks).

    Returns:
        (train_loader, val_loader)

    DDP note: without an explicit DistributedSampler, every rank would
    iterate over the ENTIRE dataset independently — no sharding, `world_size`×
    redundant computation, and no actual speedup from adding GPUs. When
    world_size > 1 both loaders use DistributedSampler(num_replicas=world_size,
    rank=rank, drop_last=True) so each rank sees a disjoint 1/world_size
    slice of the data every pass.
    """
    base = Path(data_dir) / dataset

    train_ds = TokenizedDataset(base / "train.bin", seq_len)
    val_ds   = TokenizedDataset(base / "val.bin",   seq_len)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,             # keeps every batch the same size
        persistent_workers=num_workers > 0,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=seed, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=True,
        )
        train_loader = DataLoader(train_ds, sampler=train_sampler, **common)
        val_loader   = DataLoader(val_ds,   sampler=val_sampler,   **common)
    else:
        train_loader = DataLoader(train_ds, shuffle=True,  **common)
        val_loader   = DataLoader(val_ds,   shuffle=False, **common)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Infinite iterator wrapper
# ---------------------------------------------------------------------------

class InfiniteDataLoader:
    """
    Wraps a DataLoader to yield batches indefinitely.

    The training loop uses a step counter rather than epochs, so we need
    an iterator that restarts automatically at dataset end.

    If the underlying loader uses a DistributedSampler, each restart calls
    `sampler.set_epoch(n)` before re-iterating. This both reshuffles the
    data on every pass (otherwise the identical permutation would repeat
    forever) and — critically for DDP — advances the RNG identically on
    every rank each time, so the shared shuffle-then-partition scheme that
    guarantees non-overlapping shards keeps holding pass after pass, not
    just on the first one.
    """

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._epoch = 0
        self._iter  = iter(loader)

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            sampler = getattr(self.loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(self._epoch)
            self._iter = iter(self.loader)
            return next(self._iter)

    def __iter__(self) -> "InfiniteDataLoader":
        return self
