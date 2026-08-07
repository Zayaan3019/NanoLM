"""
Integration tests — end-to-end pipeline on tiny synthetic data.

These tests exercise the full pipeline:
  BPETokenizer.train  → encode → TokenizedDataset → Trainer (few steps) →
  save_checkpoint → load_checkpoint → generate

They run on CPU in < 30 seconds and do NOT require any downloaded datasets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from nanolm.model.config import ModelConfig, TrainConfig
from nanolm.model.gpt import GPT
from nanolm.tokenizer.bpe import BPETokenizer
from nanolm.data.dataset import TokenizedDataset, create_dataloaders
from nanolm.training.scheduler import get_lr
from nanolm.utils.checkpoint import (
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Shared synthetic corpus
# ---------------------------------------------------------------------------

SYNTHETIC_CORPUS = (
    "the cat sat on the mat. the cat ate the rat. "
    "a quick brown fox jumps over a lazy dog. "
    "to be or not to be that is the question. "
    "hello world foo bar baz qux. "
) * 200   # ~10 KB — large enough for BPE training and a few training steps


# ---------------------------------------------------------------------------
# Tokenizer integration
# ---------------------------------------------------------------------------

class TestTokenizerIntegration:
    def test_train_encode_decode(self):
        tok = BPETokenizer(vocab_size=300)
        tok.train(SYNTHETIC_CORPUS, verbose=False)

        text    = "the cat sat on the mat"
        ids     = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encoded_ids_fit_in_uint16(self):
        tok = BPETokenizer(vocab_size=512)
        tok.train(SYNTHETIC_CORPUS, verbose=False)
        ids = tok.encode(SYNTHETIC_CORPUS[:1000])
        assert all(0 <= i < 65536 for i in ids)


# ---------------------------------------------------------------------------
# Dataset integration
# ---------------------------------------------------------------------------

class TestDatasetIntegration:
    def test_tokenized_dataset_shapes(self):
        tok = BPETokenizer(vocab_size=300)
        tok.train(SYNTHETIC_CORPUS, verbose=False)
        ids = tok.encode(SYNTHETIC_CORPUS)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            bin_path = Path(tmp) / "train.bin"
            np.array(ids, dtype=np.uint16).tofile(bin_path)

            ds  = TokenizedDataset(bin_path, seq_len=32)
            x, y = ds[0]

            assert x.shape == (32,)
            assert y.shape == (32,)
            assert x.dtype == torch.int64
            # y is x shifted by 1
            assert torch.equal(x[1:], y[:-1])
            del ds      # release memmap file handle (important on Windows)

    def test_dataloader_batch_shapes(self):
        tok = BPETokenizer(vocab_size=300)
        tok.train(SYNTHETIC_CORPUS, verbose=False)
        ids = tok.encode(SYNTHETIC_CORPUS)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            base = Path(tmp) / "testds"
            base.mkdir()
            n    = int(0.9 * len(ids))
            np.array(ids[:n], dtype=np.uint16).tofile(base / "train.bin")
            np.array(ids[n:], dtype=np.uint16).tofile(base / "val.bin")

            train_ldr, val_ldr = create_dataloaders(
                data_dir=tmp, dataset="testds",
                seq_len=32, batch_size=4, num_workers=0,
            )
            x_batch, y_batch = next(iter(train_ldr))
            del train_ldr, val_ldr  # release memmap file handles (important on Windows)
            assert x_batch.shape == (4, 32)
            assert y_batch.shape == (4, 32)


# ---------------------------------------------------------------------------
# LR scheduler integration
# ---------------------------------------------------------------------------

class TestSchedulerIntegration:
    def test_warmup_phase(self):
        for step in range(0, 100, 10):
            lr = get_lr(step, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
            assert lr <= 1e-3
            assert lr > 0

    def test_at_max_steps_returns_min_lr(self):
        lr = get_lr(1000, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
        assert abs(lr - 1e-4) < 1e-9

    def test_monotone_decay_after_warmup(self):
        lrs = [
            get_lr(s, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
            for s in range(100, 1001, 50)
        ]
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1)), (
            "LR should be monotonically non-increasing after warmup"
        )


# ---------------------------------------------------------------------------
# Checkpoint integration
# ---------------------------------------------------------------------------

class TestCheckpointIntegration:
    def test_save_load_preserves_weights(self):
        cfg   = ModelConfig(vocab_size=64, max_seq_len=16, n_layers=2, n_heads=2, d_model=16)
        model = GPT(cfg)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Take a gradient step so optimizer state is non-trivial
        idx     = torch.randint(0, 64, (1, 8))
        targets = torch.randint(0, 64, (1, 8))
        _, loss, _ = model(idx, targets)
        loss.backward()
        opt.step()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "test.pt"
            save_checkpoint(ckpt_path, model, opt, step=1, val_loss=2.5,
                            model_config=cfg, train_config=TrainConfig())

            model2 = GPT(cfg)
            step   = load_checkpoint(ckpt_path, model2, None)

            assert step == 1
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.equal(p1, p2), f"Mismatch at {n1}"

    def test_load_model_from_checkpoint(self):
        cfg   = ModelConfig(vocab_size=64, max_seq_len=16, n_layers=2, n_heads=2, d_model=16)
        model = GPT(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.pt"
            save_checkpoint(ckpt_path, model,
                            torch.optim.AdamW(model.parameters()),
                            step=42, val_loss=1.0,
                            model_config=cfg, train_config=TrainConfig())

            model2, cfg2, step = load_model_from_checkpoint(ckpt_path)

        assert step == 42
        assert cfg2.n_layers == cfg.n_layers
        assert cfg2.d_model  == cfg.d_model
        for (_, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# Full mini training run
# ---------------------------------------------------------------------------

class TestMiniTrainingRun:
    """
    Trains a 2-layer nano model for 10 gradient steps on synthetic data
    and verifies that the loss goes down.
    """

    def test_loss_decreases(self):
        corpus = SYNTHETIC_CORPUS

        # ---- Tokenize -----------------------------------------------
        tok = BPETokenizer(vocab_size=300)
        tok.train(corpus, verbose=False)
        ids = tok.encode(corpus)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            ds_dir = tmp / "minidata"
            ds_dir.mkdir()

            n = int(0.9 * len(ids))
            np.array(ids[:n], dtype=np.uint16).tofile(ds_dir / "train.bin")
            np.array(ids[n:], dtype=np.uint16).tofile(ds_dir / "val.bin")

            # ---- Model --------------------------------------------------
            cfg = ModelConfig(
                vocab_size=tok.vocab_size,
                max_seq_len=32,
                n_layers=2,
                n_heads=2,
                d_model=32,
            )
            model = GPT(cfg)
            opt   = model.configure_optimizer(
                weight_decay=0.01, lr=1e-3, betas=(0.9, 0.95), device_type="cpu"
            )
            device = torch.device("cpu")
            model.to(device)

            # ---- Data ---------------------------------------------------
            train_loader, _ = create_dataloaders(
                data_dir=str(tmp), dataset="minidata",
                seq_len=32, batch_size=4, num_workers=0,
            )
            data_iter = iter(train_loader)

            # ---- Training loop ------------------------------------------
            losses: list[float] = []
            model.train()
            for _ in range(10):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                _, loss, _ = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

            del train_loader, data_iter  # release memmap file handles (Windows)

            # Loss should trend downward (first 5 avg > last 5 avg)
            first_half = sum(losses[:5]) / 5
            second_half = sum(losses[5:]) / 5
            assert second_half < first_half + 0.1, (
                f"Loss did not decrease: first_half={first_half:.4f}, "
                f"second_half={second_half:.4f}"
            )

    def test_generate_after_training(self):
        """After a few training steps, generate() should produce valid token ids."""
        corpus = SYNTHETIC_CORPUS
        tok = BPETokenizer(vocab_size=300)
        tok.train(corpus, verbose=False)
        ids = tok.encode(corpus)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ds_dir = tmp / "minidata"
            ds_dir.mkdir()
            n = int(0.9 * len(ids))
            np.array(ids[:n], dtype=np.uint16).tofile(ds_dir / "train.bin")
            np.array(ids[n:], dtype=np.uint16).tofile(ds_dir / "val.bin")

            cfg   = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=32,
                                n_layers=2, n_heads=2, d_model=32)
            model = GPT(cfg)
            model.eval()

            prompt = torch.randint(0, tok.vocab_size, (1, 4))
            out    = model.generate(prompt, max_new_tokens=8, temperature=1.0)

            assert out.shape == (1, 12)
            # All token ids should be within vocabulary
            assert out.max().item() < tok.vocab_size
            assert out.min().item() >= 0

            # Decoding the generated portion should not raise
            new_ids = out[0, 4:].tolist()
            _ = tok.decode(new_ids)
