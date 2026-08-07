"""
Checkpoint save / load utilities.

Every checkpoint is a self-describing dict containing:
  - model state dict
  - optimizer state dict
  - training step
  - best validation loss
  - ModelConfig  (as dict — for exact architecture reconstruction)
  - TrainConfig  (as dict — for exact hyperparameter reconstruction)

This means a checkpoint alone is sufficient to resume training or
load a model for inference, with no external config files required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from nanolm.model.config import ModelConfig, TrainConfig


def save_checkpoint(
    path: str | Path,
    model: "GPT",
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> None:
    """
    Save a full training checkpoint.

    Args:
        path:         Destination file path (.pt).
        model:        The raw (non-DDP-wrapped) GPT model.
        optimizer:    AdamW optimizer with current state.
        step:         Current training step.
        val_loss:     Best (or current) validation loss.
        model_config: Model architecture config.
        train_config: Training hyperparameter config.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step":                 step,
            "val_loss":             val_loss,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config":         model_config.to_dict(),
            "train_config":         train_config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: "GPT",
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """
    Load a checkpoint into `model` (and optionally `optimizer`).

    Args:
        path:         Path to the .pt checkpoint file.
        model:        The model to load weights into.
                      Must have been constructed with the same ModelConfig
                      that was used when saving.
        optimizer:    If given, load optimizer state too (for resuming training).
        map_location: Where to initially load tensors (default: CPU, then let
                      the caller move the model to the target device).

    Returns:
        The training step at which the checkpoint was saved.
    """
    # weights_only=True restricts unpickling to tensors/dicts/primitives —
    # checkpoints saved by save_checkpoint() contain nothing else, so this
    # is safe and avoids arbitrary code execution when loading a checkpoint
    # from an untrusted source (e.g. downloaded, or shared by a teammate).
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt.get("step", 0)


def load_model_from_checkpoint(path: str | Path) -> tuple["GPT", ModelConfig, int]:
    """
    Reconstruct a GPT model entirely from a checkpoint (no external config needed).

    Args:
        path: Path to the .pt checkpoint file.

    Returns:
        (model, model_config, step)
    """
    from nanolm.model.gpt import GPT

    ckpt         = torch.load(path, map_location="cpu", weights_only=True)
    model_config = ModelConfig.from_dict(ckpt["model_config"])
    model        = GPT(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    step         = ckpt.get("step", 0)
    return model, model_config, step
