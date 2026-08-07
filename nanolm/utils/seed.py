"""
Global RNG seeding for reproducible runs.

Usage in DDP: call `set_seed(cfg.seed)` with the SAME seed value on every
rank, before constructing the model. This makes each rank's model
initialisation deterministic and identical (defense in depth — DDP's
constructor also broadcasts rank 0's parameters to every other rank, but
relying on that alone is implicit and silently breaks if a model is ever
used before being wrapped in DDP, e.g. for a pre-DDP sanity forward pass).

Per-rank data order is handled separately by `create_dataloaders`, which
uses `DistributedSampler(..., seed=cfg.seed)` — the *same* seed across
ranks is required there too, since DistributedSampler derives one shared
shuffle permutation from it and then partitions that permutation by rank.
Using the same seed for both purposes is intentional, not an oversight:
identical seeds + rank-based partitioning is what "no overlap" requires.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (CPU + all CUDA devices) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
