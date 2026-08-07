"""
Text generation from a trained NanoLM checkpoint.

Usage
-----
# Generate 200 tokens with the default settings
python generate.py --checkpoint checkpoints/best.pt \\
                   --tokenizer  data/shakespeare/tokenizer.json \\
                   --prompt     "To be or not to be"

# Nucleus sampling (recommended for diversity)
python generate.py --checkpoint checkpoints/best.pt \\
                   --tokenizer  data/shakespeare/tokenizer.json \\
                   --prompt     "HAMLET:" \\
                   --max_tokens 500 --temperature 0.8 --top_p 0.9

# Greedy decoding (deterministic, temperature=0 → argmax)
python generate.py --checkpoint checkpoints/best.pt \\
                   --tokenizer  data/shakespeare/tokenizer.json \\
                   --prompt     "KING LEAR:" \\
                   --temperature 0.0

# Multiple samples from the same prompt
python generate.py --checkpoint checkpoints/best.pt \\
                   --tokenizer  data/shakespeare/tokenizer.json \\
                   --prompt     "Once upon a time" \\
                   --num_samples 5
"""

from __future__ import annotations

import argparse
import sys

import torch

from nanolm.tokenizer.bpe import BPETokenizer
from nanolm.utils.checkpoint import load_model_from_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate text from a trained NanoLM model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,  help="Path to .pt checkpoint file.")
    p.add_argument("--tokenizer",    required=True,  help="Path to tokenizer.json file.")
    p.add_argument("--prompt",       default="",     help="Prompt text.  Empty = unconditional.")
    p.add_argument("--max_tokens",   type=int,   default=200,  help="Maximum new tokens to generate.")
    p.add_argument("--temperature",  type=float, default=1.0,  help="Sampling temperature (0 = greedy).")
    p.add_argument("--top_k",        type=int,   default=None, help="Top-k truncation (e.g. 50).")
    p.add_argument("--top_p",        type=float, default=None, help="Nucleus sampling (e.g. 0.9).")
    p.add_argument("--num_samples",  type=int,   default=1,    help="Number of independent samples.")
    p.add_argument("--device",       default=None, help="Device: cuda / cpu / mps (default: auto).")
    p.add_argument("--seed",         type=int,   default=None, help="Random seed for reproducibility.")
    p.add_argument("--no_kv_cache",  action="store_true", help="Disable KV-cache (slower but less memory).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Seed ------------------------------------------------------------
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ---- Device ----------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}", file=sys.stderr)

    # ---- Load model ------------------------------------------------------
    print(f"Loading model from {args.checkpoint} …", file=sys.stderr)
    model, model_cfg, step = load_model_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  {model}  (trained {step:,} steps)", file=sys.stderr)

    # ---- Load tokenizer --------------------------------------------------
    print(f"Loading tokenizer from {args.tokenizer} …", file=sys.stderr)
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"  {tokenizer}", file=sys.stderr)

    # ---- Encode prompt ---------------------------------------------------
    if args.prompt:
        prompt_ids = tokenizer.encode(args.prompt)
    else:
        # Unconditional generation: start with <|endoftext|>
        prompt_ids = [tokenizer.eot_id]

    prompt_tensor = torch.tensor(
        [prompt_ids], dtype=torch.long, device=device
    )   # (1, T_prompt)

    # ---- Generation settings --------------------------------------------
    gen_kwargs: dict = dict(
        max_new_tokens=args.max_tokens,
        use_kv_cache=not args.no_kv_cache,
    )

    if args.temperature == 0.0:
        # Greedy: temperature=0 → argmax.  We use a tiny ε to avoid /0.
        gen_kwargs["temperature"] = 1e-10
        gen_kwargs["top_k"]       = 1
    else:
        gen_kwargs["temperature"] = args.temperature
        if args.top_k is not None:
            gen_kwargs["top_k"] = args.top_k
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p

    # ---- Generate -------------------------------------------------------
    for sample_idx in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n{'-'*60}")
            print(f"  Sample {sample_idx + 1}/{args.num_samples}")
            print(f"{'-'*60}")

        with torch.no_grad():
            out_ids = model.generate(
                prompt_tensor.clone(),
                **gen_kwargs,
            )

        # Decode only the newly generated tokens (exclude prompt)
        new_ids  = out_ids[0, len(prompt_ids):].tolist()
        new_text = tokenizer.decode(new_ids)

        # Print prompt in bold (if terminal supports it), then generation
        if sys.stdout.isatty() and args.prompt:
            print(f"\033[1m{args.prompt}\033[0m", end="")
        elif args.prompt:
            print(args.prompt, end="")

        print(new_text)

    if args.num_samples > 1:
        print()


if __name__ == "__main__":
    main()
