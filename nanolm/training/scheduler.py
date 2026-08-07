"""
Learning rate schedule: linear warmup + cosine decay.

This is the schedule used in GPT-3 (Brown et al., 2020) and most
subsequent large language model work:

  1. Linear warmup from 0 → max_lr over `warmup_steps` steps.
  2. Cosine decay from max_lr → min_lr over the remaining steps.
  3. Constant at min_lr for any steps beyond max_steps (prevents the
     scheduler from driving lr all the way to zero mid-experiment).

Typical values (from the GPT-3 paper):
  max_lr  = 6e-4   (for 125M params; scale down for larger models)
  min_lr  = 6e-5   = 10% of max_lr
  warmup  = 375 M tokens  ≈ 2000 steps at batch ×480K tokens/step
"""

from __future__ import annotations

import math


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Compute the learning rate for a given training step.

    Args:
        step:         Current optimiser step (0-indexed).
        warmup_steps: Number of linear warmup steps.
        max_steps:    Total training steps (decay reaches min_lr here).
        max_lr:       Peak learning rate.
        min_lr:       Minimum learning rate (floor of cosine decay).

    Returns:
        Learning rate for this step.
    """
    # Phase 1: linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Phase 3: past the decay horizon — hold at floor
    if step >= max_steps:
        return min_lr

    # Phase 2: cosine decay
    progress   = (step - warmup_steps) / (max_steps - warmup_steps)   # ∈ [0, 1]
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))           # ∈ [1, 0]
    return min_lr + cos_factor * (max_lr - min_lr)


class CosineWarmupScheduler:
    """
    Thin wrapper around `get_lr` that integrates with PyTorch's optimizer.

    Usage:
        scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_steps=5000,
                                          max_lr=1e-3, min_lr=1e-4)
        for step in range(max_steps):
            scheduler.step()          # sets lr on all param groups
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        optimizer: "torch.optim.Optimizer",
        warmup: int,
        max_steps: int,
        max_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer  = optimizer
        self.warmup     = warmup
        self.max_steps  = max_steps
        self.max_lr     = max_lr
        self.min_lr     = min_lr
        self._step      = 0

    def step(self) -> float:
        """Advance one step and update all param-group learning rates."""
        lr = get_lr(self._step, self.warmup, self.max_steps, self.max_lr, self.min_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._step += 1
        return lr

    @property
    def current_lr(self) -> float:
        return get_lr(self._step, self.warmup, self.max_steps, self.max_lr, self.min_lr)
