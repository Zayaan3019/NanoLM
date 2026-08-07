"""
Data preparation entry point.

Downloads a dataset, trains a BPE tokenizer from scratch on it, tokenises
the corpus, and saves binary train/val splits ready for the training loop.

Usage
-----
# Tiny Shakespeare — runs in < 2 minutes on CPU, good for the nano model
python prepare_data.py --dataset shakespeare --vocab_size 4096

# TinyStories — ~10-minute tokenizer training on CPU
python prepare_data.py --dataset tinystories --vocab_size 10000

# OpenWebText — requires `pip install datasets`; downloads ~38 GB
python prepare_data.py --dataset openwebtext --vocab_size 50257
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanolm.data.prepare import (
    prepare_openwebtext,
    prepare_shakespeare,
    prepare_tinystories,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and pre-process a text dataset for NanoLM training."
    )
    p.add_argument(
        "--dataset",
        choices=["shakespeare", "tinystories", "openwebtext"],
        default="shakespeare",
        help="Dataset to prepare (default: shakespeare)",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory (default: data/)",
    )
    p.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help=(
            "BPE vocabulary size.  Defaults: shakespeare=4096, "
            "tinystories=10000, openwebtext=50257"
        ),
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to an existing tokenizer.json to reuse instead of training a new one.",
    )
    p.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of parallel processes for openwebtext preparation (default: 4)",
    )
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data_dir)

    defaults = {"shakespeare": 4096, "tinystories": 10000, "openwebtext": 50257}
    vocab_size = args.vocab_size or defaults[args.dataset]

    tok_path = Path(args.tokenizer) if args.tokenizer else None

    print(f"\n{'=' * 60}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Vocab size : {vocab_size:,}")
    print(f"  Output dir : {data_dir / args.dataset}")
    print(f"{'=' * 60}\n")

    if args.dataset == "shakespeare":
        prepare_shakespeare(data_dir, vocab_size=vocab_size, tokenizer_path=tok_path)
    elif args.dataset == "tinystories":
        prepare_tinystories(data_dir, vocab_size=vocab_size, tokenizer_path=tok_path)
    elif args.dataset == "openwebtext":
        prepare_openwebtext(
            data_dir, vocab_size=vocab_size,
            tokenizer_path=tok_path, num_proc=args.num_proc,
        )

    print("\nData preparation complete.")
    print(f"Train your model with:  python train.py --preset nano\n")


if __name__ == "__main__":
    main()
