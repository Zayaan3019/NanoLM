"""
NanoLM — GPT-style language model trained from scratch.

Core modules:
    nanolm.model.gpt         — GPT architecture (Flash Attention, KV-cache)
    nanolm.model.config      — ModelConfig and TrainConfig dataclasses
    nanolm.tokenizer.bpe     — Byte-level BPE tokenizer from scratch
    nanolm.data.dataset      — Memory-mapped dataset and DataLoaders
    nanolm.data.prepare      — Dataset download and pre-processing
    nanolm.training.trainer  — Full training loop (AMP, gradient accum, DDP)
    nanolm.training.scheduler— Cosine LR schedule with warmup
    nanolm.utils.checkpoint  — Checkpoint save / load utilities
"""

__version__ = "0.1.0"
__author__  = "Mohamed Zayaan S"

from nanolm.model.gpt    import GPT
from nanolm.model.config import ModelConfig, TrainConfig

__all__ = ["GPT", "ModelConfig", "TrainConfig", "__version__"]
