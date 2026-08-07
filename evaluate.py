"""
Model evaluation — perplexity on held-out data.

Perplexity (PPL) is the standard intrinsic metric for language models:
  PPL = exp(mean cross-entropy loss over all tokens)

Lower perplexity = better fit to the data distribution.

Reference numbers:
  GPT-2 small (124M) on Penn Treebank: PPL ≈ 35
  GPT-2 small (124M) on WebText:       PPL ≈ 18
  Nano model  (~3M)  on Shakespeare:   PPL ≈ 3–5 (small dataset, can overfit)

Usage
-----
python evaluate.py \\
    --checkpoint checkpoints/best.pt \\
    --tokenizer  data/shakespeare/tokenizer.json \\
    --data       data/shakespeare/val.bin

# Evaluate on a custom text file
python evaluate.py \\
    --checkpoint checkpoints/best.pt \\
    --tokenizer  data/shakespeare/tokenizer.json \\
    --text_file  my_text.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from nanolm.tokenizer.bpe import BPETokenizer
from nanolm.utils.checkpoint import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a NanoLM model — reports perplexity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--tokenizer",  required=True, help="Path to tokenizer.json.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data",      help="Path to pre-tokenised .bin file (val.bin etc.)")
    src.add_argument("--text_file", help="Path to a raw UTF-8 text file to evaluate on.")

    p.add_argument("--seq_len",    type=int,   default=None,  help="Evaluation sequence length (default: model max).")
    p.add_argument("--batch_size", type=int,   default=8,     help="Evaluation batch size.")
    p.add_argument("--max_batches",type=int,   default=None,  help="Cap the number of batches evaluated.")
    p.add_argument("--device",     type=str,   default=None,  help="Device (default: auto).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_tokens(
    model:      "GPT",
    token_ids:  np.ndarray,
    seq_len:    int,
    batch_size: int,
    device:     torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """
    Compute mean cross-entropy loss and perplexity over token_ids.

    Returns:
        (loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    # Build non-overlapping windows of length seq_len
    n_windows = (len(token_ids) - 1) // seq_len
    if n_windows == 0:
        raise ValueError(
            f"Not enough tokens ({len(token_ids)}) for seq_len={seq_len}. "
            "Lower --seq_len or provide more text."
        )

    indices = list(range(n_windows))

    for batch_start in tqdm(
        range(0, len(indices), batch_size),
        desc="Evaluating",
        unit="batch",
        file=sys.stderr,
    ):
        if max_batches is not None and n_batches >= max_batches:
            break

        batch_idx = indices[batch_start : batch_start + batch_size]
        xs, ys = [], []

        for i in batch_idx:
            start = i * seq_len
            chunk = token_ids[start : start + seq_len + 1].astype(np.int64)
            xs.append(torch.from_numpy(chunk[:-1]))
            ys.append(torch.from_numpy(chunk[1:]))

        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)

        _, loss, _ = model(x, y)
        total_loss += loss.item()
        n_batches  += 1

    mean_loss   = total_loss / n_batches
    perplexity  = math.exp(mean_loss)
    return mean_loss, perplexity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Device ----------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ---- Load model ------------------------------------------------------
    print(f"Loading model from {args.checkpoint} …")
    model, model_cfg, step = load_model_from_checkpoint(args.checkpoint)
    model = model.to(device)
    print(f"  {model}  (step {step:,})")

    seq_len = args.seq_len or model_cfg.max_seq_len

    # ---- Load / tokenise data -------------------------------------------
    if args.data:
        print(f"Loading token ids from {args.data} …")
        token_ids = np.memmap(args.data, dtype=np.uint16, mode="r")
        print(f"  {len(token_ids):,} tokens")
    else:
        print(f"Tokenising {args.text_file} …")
        tokenizer = BPETokenizer.load(args.tokenizer)
        text      = Path(args.text_file).read_text(encoding="utf-8")
        ids       = tokenizer.encode(text)
        token_ids = np.array(ids, dtype=np.uint16)
        print(f"  {len(token_ids):,} tokens")

    # ---- Evaluate --------------------------------------------------------
    loss, ppl = evaluate_tokens(
        model, token_ids, seq_len, args.batch_size, device, args.max_batches
    )

    print(f"\n{'-'*40}")
    print(f"  Mean loss   : {loss:.4f}")
    print(f"  Perplexity  : {ppl:.2f}")
    print(f"{'-'*40}\n")


if __name__ == "__main__":
    main()
