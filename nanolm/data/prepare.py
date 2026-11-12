"""
Dataset download and pre-processing.

Supported datasets
──────────────────
shakespeare   ~1 MB   — tiny, ideal for CPU demo
tinystories   ~2 GB   — children's stories, good for small model
openwebtext   ~38 GB  — web text, GPT-2 training distribution

Each dataset is:
  1. Downloaded to  data/<name>/raw/
  2. Tokenised with a BPE tokenizer (trained on the same corpus if using
     a from-scratch tokenizer, or loaded from an existing file)
  3. Saved as two memory-mapped binary files:
       data/<name>/train.bin  (uint16, 90% of tokens)
       data/<name>/val.bin    (uint16, 10% of tokens)
  4. Tokenizer saved to data/<name>/tokenizer.json

Usage:
    from nanolm.data.prepare import prepare_shakespeare
    prepare_shakespeare(Path("data"), vocab_size=4096)
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from nanolm.tokenizer.bpe import BPETokenizer


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

class _DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest`, showing a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


# ---------------------------------------------------------------------------
# Tokenise + save helpers
# ---------------------------------------------------------------------------

def _tokenise_and_save(
    text: str,
    tokenizer: BPETokenizer,
    out_dir: Path,
    val_fraction: float = 0.1,
) -> tuple[int, int]:
    """
    Tokenise `text`, split into train/val, save as uint16 memmaps.

    Returns:
        (n_train_tokens, n_val_tokens)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Tokenising… ", end="", flush=True)
    ids = tokenizer.encode(text, add_special=False)
    print(f"{len(ids):,} tokens total")

    split = int(len(ids) * (1 - val_fraction))
    train_ids = np.array(ids[:split],  dtype=np.uint16)
    val_ids   = np.array(ids[split:],  dtype=np.uint16)

    train_path = out_dir / "train.bin"
    val_path   = out_dir / "val.bin"

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"  train: {len(train_ids):>12,} tokens  ->  {train_path}")
    print(f"  val:   {len(val_ids):>12,} tokens  ->  {val_path}")

    return len(train_ids), len(val_ids)


# ---------------------------------------------------------------------------
# Shakespeare
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def prepare_shakespeare(
    data_dir: Path,
    vocab_size: int = 4096,
    tokenizer_path: Optional[Path] = None,
) -> BPETokenizer:
    """
    Download Tiny Shakespeare and prepare train/val splits.

    Args:
        data_dir:       Root data directory.  Output goes to data_dir/shakespeare/.
        vocab_size:     BPE vocabulary size.  4096 is plenty for this ~1 MB corpus.
        tokenizer_path: If given, load an existing tokenizer instead of training one.

    Returns:
        The fitted BPETokenizer.
    """
    out_dir  = data_dir / "shakespeare"
    raw_path = out_dir / "raw" / "input.txt"

    # ---- Download ---------------------------------------------------------
    if not raw_path.exists():
        print(f"Downloading Shakespeare from {SHAKESPEARE_URL}")
        _download(SHAKESPEARE_URL, raw_path)
    else:
        print(f"Found existing raw file: {raw_path}")

    text = raw_path.read_text(encoding="utf-8")
    print(f"Corpus: {len(text):,} characters")

    # ---- Tokenizer --------------------------------------------------------
    tok_path = out_dir / "tokenizer.json"
    if tokenizer_path is not None and Path(tokenizer_path).exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    elif tok_path.exists():
        print(f"Loading existing tokenizer from {tok_path}")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print(f"Training BPE tokenizer (vocab_size={vocab_size}) on Shakespeare…")
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(text, verbose=True)
        tokenizer.save(tok_path)
        print(f"Tokenizer saved to {tok_path}")

    # ---- Tokenise & save --------------------------------------------------
    _tokenise_and_save(text, tokenizer, out_dir)

    return tokenizer


# ---------------------------------------------------------------------------
# TinyStories
# ---------------------------------------------------------------------------

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VAL_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


def prepare_tinystories(
    data_dir: Path,
    vocab_size: int = 10000,
    tokenizer_path: Optional[Path] = None,
    train_tokenizer_on_fraction: float = 0.1,
) -> BPETokenizer:
    """
    Download TinyStories and prepare train/val splits.

    The tokenizer is trained on a fraction of the training data to keep
    BPE training fast (~2–5 min on CPU for 10% of TinyStories).

    Args:
        data_dir:                     Root data directory.
        vocab_size:                   BPE vocabulary size.
        tokenizer_path:               Load an existing tokenizer instead of training.
        train_tokenizer_on_fraction:  Fraction of training text to use for BPE training.

    Returns:
        The fitted BPETokenizer.
    """
    out_dir   = data_dir / "tinystories"
    train_raw = out_dir / "raw" / "train.txt"
    val_raw   = out_dir / "raw" / "val.txt"

    # ---- Download ---------------------------------------------------------
    if not train_raw.exists():
        print("Downloading TinyStories training set…")
        _download(TINYSTORIES_URL, train_raw)
    if not val_raw.exists():
        print("Downloading TinyStories validation set…")
        _download(TINYSTORIES_VAL_URL, val_raw)

    train_text = train_raw.read_text(encoding="utf-8")
    val_text   = val_raw.read_text(encoding="utf-8")
    print(f"Train corpus: {len(train_text):,} characters")
    print(f"Val   corpus: {len(val_text):,} characters")

    # ---- Tokenizer --------------------------------------------------------
    tok_path = out_dir / "tokenizer.json"
    if tokenizer_path is not None and Path(tokenizer_path).exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
    elif tok_path.exists():
        print(f"Loading existing tokenizer from {tok_path}")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        frac_end  = int(len(train_text) * train_tokenizer_on_fraction)
        tok_text  = train_text[:frac_end]
        print(
            f"Training BPE tokenizer on {len(tok_text):,} chars "
            f"({train_tokenizer_on_fraction*100:.0f}% of training data)…"
        )
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(tok_text, verbose=True)
        tokenizer.save(tok_path)
        print(f"Tokenizer saved to {tok_path}")

    # ---- Tokenise & save (train) ------------------------------------------
    print("\nTokenising training set…")
    train_ids = tokenizer.encode(train_text, add_special=False)
    print("Tokenising validation set…")
    val_ids   = tokenizer.encode(val_text, add_special=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.array(train_ids, dtype=np.uint16).tofile(out_dir / "train.bin")
    np.array(val_ids,   dtype=np.uint16).tofile(out_dir / "val.bin")

    print(f"  train: {len(train_ids):>12,} tokens")
    print(f"  val:   {len(val_ids):>12,} tokens")

    return tokenizer


# ---------------------------------------------------------------------------
# OpenWebText  (requires datasets library: pip install datasets)
# ---------------------------------------------------------------------------

def prepare_openwebtext(
    data_dir: Path,
    vocab_size: int = 50257,
    tokenizer_path: Optional[Path] = None,
    num_proc: int = 4,
) -> BPETokenizer:
    """
    Prepare OpenWebText using the HuggingFace `datasets` library.

    Requires:  pip install datasets

    The tokenizer is trained on a 1% sample of the dataset (~380 MB)
    to approximate GPT-2's tokenizer without needing the original.

    This function will download ~38 GB of data on first run.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for OpenWebText preparation.\n"
            "Install it with:  pip install datasets"
        ) from exc

    out_dir  = data_dir / "openwebtext"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer --------------------------------------------------------
    tok_path = out_dir / "tokenizer.json"
    if tokenizer_path is not None and Path(tokenizer_path).exists():
        tokenizer = BPETokenizer.load(tokenizer_path)
    elif tok_path.exists():
        print(f"Loading existing tokenizer from {tok_path}")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print("Loading 1% sample of OpenWebText for tokenizer training…")
        ds_sample = load_dataset("openwebtext", split="train[:1%]", num_proc=num_proc)
        sample_text = "\n\n".join(ds_sample["text"])
        print(f"  sample size: {len(sample_text):,} characters")

        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(sample_text, verbose=True)
        tokenizer.save(tok_path)
        print(f"Tokenizer saved to {tok_path}")

    # ---- Load full dataset ------------------------------------------------
    print("Loading full OpenWebText dataset (this may take a while)…")
    dataset = load_dataset("openwebtext", num_proc=num_proc)
    split   = dataset["train"].train_test_split(test_size=0.0005, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    # ---- Tokenise in parallel --------------------------------------------
    eot = tokenizer.eot_id

    def _process(example: dict) -> dict:
        ids = tokenizer.encode(example["text"])
        ids.append(eot)    # document separator
        return {"ids": ids, "len": len(ids)}

    print("Tokenising training split…")
    train_tokenised = train_ds.map(
        _process, remove_columns=["text"],
        desc="train", num_proc=num_proc,
    )
    print("Tokenising validation split…")
    val_tokenised = val_ds.map(
        _process, remove_columns=["text"],
        desc="val", num_proc=num_proc,
    )

    # ---- Write memmaps ---------------------------------------------------
    for split_name, split_data in [("train", train_tokenised), ("val", val_tokenised)]:
        n_tokens = sum(split_data["len"])
        arr = np.memmap(out_dir / f"{split_name}.bin", dtype=np.uint16, mode="w+", shape=(n_tokens,))
        offset = 0
        for ids in tqdm(split_data["ids"], desc=f"Writing {split_name}.bin"):
            arr[offset : offset + len(ids)] = ids
            offset += len(ids)
        arr.flush()
        print(f"  {split_name}: {n_tokens:,} tokens -> {out_dir / f'{split_name}.bin'}")

    return tokenizer
