"""
Model and training configuration dataclasses.

All hyperparameters live here so checkpoints are fully self-describing:
every saved .pt file contains both configs, enabling exact reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """GPT architecture hyperparameters."""

    # Vocabulary & sequence
    vocab_size: int = 50304      # Padded to nearest multiple of 64 (hardware efficiency)
    max_seq_len: int = 1024

    # Depth & width
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768           # Must be divisible by n_heads

    # Regularisation
    dropout: float = 0.0         # 0 during pre-training; 0.1 for fine-tuning

    # Architecture choices
    bias: bool = False           # Omitting bias is modern best practice (faster, negligible quality loss)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        """Approximate parameter count (wpe excluded; wte/lm_head counted once due to tying)."""
        # Each layer: QKV proj + out proj + 2 × MLP + 2 × LN
        per_layer = (
            3 * self.d_model * self.d_model          # QKV
            + self.d_model * self.d_model            # out proj
            + self.d_model * 4 * self.d_model        # MLP fc
            + 4 * self.d_model * self.d_model        # MLP proj
            + 2 * self.d_model                       # two LayerNorms
        )
        # wte/lm_head weight is shared (weight tying) — counted once
        return self.n_layers * per_layer + self.vocab_size * self.d_model

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def nano(cls) -> ModelConfig:
        """~3.5 M params — trains on Shakespeare in < 1 hour on a single CPU."""
        return cls(vocab_size=4096, max_seq_len=256, n_layers=6, n_heads=6, d_model=192)

    @classmethod
    def small(cls) -> ModelConfig:
        """~41 M params — single GPU, trains on TinyStories."""
        return cls(vocab_size=50304, max_seq_len=1024, n_layers=12, n_heads=12, d_model=384)

    @classmethod
    def gpt2(cls) -> ModelConfig:
        """~124 M params — GPT-2 scale, trains on OpenWebText."""
        return cls(vocab_size=50304, max_seq_len=1024, n_layers=12, n_heads=12, d_model=768)

    @classmethod
    def gpt2_medium(cls) -> ModelConfig:
        """~345 M params — GPT-2 Medium scale."""
        return cls(vocab_size=50304, max_seq_len=1024, n_layers=24, n_heads=16, d_model=1024)

    def __repr__(self) -> str:
        return (
            f"ModelConfig(n_layers={self.n_layers}, n_heads={self.n_heads}, "
            f"d_model={self.d_model}, vocab_size={self.vocab_size}, "
            f"max_seq_len={self.max_seq_len}, ~{self.n_params / 1e6:.1f}M non-emb params)"
        )


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # ---- Optimiser -------------------------------------------------------
    max_lr: float = 6e-4
    min_lr: float = 6e-5           # Cosine decay floor — GPT-3 paper uses ~10% of max_lr
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95            # Higher beta2 than default (0.999) — GPT-3 recommendation
    grad_clip: float = 1.0

    # ---- LR schedule -----------------------------------------------------
    warmup_steps: int = 2000
    max_steps: int = 600_000
    decay_lr: bool = True

    # ---- Batch & context -------------------------------------------------
    batch_size: int = 12           # Per-device micro-batch size
    grad_accum_steps: int = 5      # Effective batch = batch_size × grad_accum_steps
    seq_len: int = 1024            # Must equal ModelConfig.max_seq_len

    # ---- Data ------------------------------------------------------------
    data_dir: str = "data"
    dataset: str = "shakespeare"   # "shakespeare" | "tinystories" | "openwebtext"
    num_workers: int = 0

    # ---- Precision -------------------------------------------------------
    use_amp: bool = True
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

    # ---- Checkpointing & logging -----------------------------------------
    checkpoint_dir: str = "checkpoints"
    eval_interval: int = 500
    save_interval: int = 5_000
    eval_steps: int = 200
    log_dir: str = "logs"
    log_interval: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "llm-from-scratch"

    # ---- Distributed training (DDP) --------------------------------------
    distributed: bool = False
    dist_backend: str = "nccl"     # "nccl" for GPU, "gloo" for CPU

    # ---- Reproducibility ---------------------------------------------------
    # Same value MUST be used on every DDP rank: it seeds model init
    # (defense in depth alongside DDP's own parameter broadcast) AND seeds
    # DistributedSampler's shared shuffle permutation, which is then
    # partitioned per-rank to guarantee non-overlapping shards.
    seed: int = 1337

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def nano(cls) -> TrainConfig:
        """Trains nano model to convergence on Shakespeare in ~1 h on CPU."""
        return cls(
            max_lr=1e-3,
            min_lr=1e-4,
            weight_decay=0.1,
            warmup_steps=100,
            max_steps=5_000,
            batch_size=12,
            grad_accum_steps=1,
            seq_len=256,
            dataset="shakespeare",
            eval_interval=200,
            save_interval=1_000,
            eval_steps=50,
            use_amp=False,      # CPU — AMP not supported
            dtype="float32",
            log_interval=10,
        )

    @classmethod
    def gpt2(cls) -> TrainConfig:
        """GPT-2 scale on OpenWebText — requires 1× A100 80 GB or 4× A100 40 GB."""
        return cls(
            max_lr=6e-4,
            min_lr=6e-5,
            weight_decay=0.1,
            warmup_steps=2_000,
            max_steps=600_000,
            batch_size=12,
            grad_accum_steps=40,
            seq_len=1024,
            dataset="openwebtext",
            eval_interval=1_000,
            save_interval=10_000,
            eval_steps=200,
            use_amp=True,
            dtype="bfloat16",
        )

    def __repr__(self) -> str:
        eff_batch = self.batch_size * self.grad_accum_steps
        eff_tokens = eff_batch * self.seq_len
        return (
            f"TrainConfig(max_steps={self.max_steps:,}, max_lr={self.max_lr}, "
            f"effective_batch={eff_batch} seqs ({eff_tokens:,} tokens/step), "
            f"dataset={self.dataset!r}, dtype={self.dtype!r})"
        )
