"""
Byte-Pair Encoding (BPE) tokenizer — implemented from scratch.

Algorithm (Sennrich et al., 2016):
  1. Represent the corpus as a sequence of raw bytes (0–255).
  2. Count the frequency of every adjacent byte pair.
  3. Merge the most frequent pair into a new token.
  4. Repeat for `num_merges` steps.
  5. Vocabulary = 256 base bytes + merged tokens + special tokens.

Encoding:
  Convert text -> bytes -> apply merges in priority order (lowest merge
  index = highest priority, processed first).  Uses a priority-queue
  approach: O(N log N) per sequence instead of the naïve O(N × merges).

This implementation is byte-level (like GPT-2), meaning it handles any
UTF-8 text without unknown tokens.

Vocabulary size:
  vocab_size = 256 (bytes) + num_merges + len(special_tokens)
  Default special token: <|endoftext|>  (used as document separator)
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Byte-level BPE tokenizer trained from scratch.

    Usage — train from text:
        tok = BPETokenizer(vocab_size=4096)
        tok.train(open("corpus.txt").read())
        tok.save("tokenizer.json")

    Usage — load and encode:
        tok = BPETokenizer.load("tokenizer.json")
        ids = tok.encode("Hello, world!")
        text = tok.decode(ids)
    """

    ENDOFTEXT = "<|endoftext|>"

    def __init__(self, vocab_size: int = 50257) -> None:
        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257 (256 bytes + 1 special token)")
        self.vocab_size = vocab_size

        # Core state — populated by train() or load()
        # merges: (token_a, token_b) -> new_token_id, ordered by discovery
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: token_id -> bytes
        self.vocab:  dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # special tokens: string -> token_id
        self.special_tokens: dict[str, int] = {}
        # Reverse map: bytes -> token_id (for fast encode)
        self._bytes_to_id: dict[bytes, int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, verbose: bool = True) -> None:
        """
        Learn BPE merge rules from a text corpus.

        The tokenizer is trained entirely in-memory.  For corpora larger
        than a few hundred MB you should pre-split into chunks and train
        on a representative sample.

        Args:
            text:    Training corpus as a UTF-8 string.
            verbose: Print progress every 500 merges.
        """
        num_merges = self.vocab_size - 256 - 1   # Reserve 1 slot for <|endoftext|>

        # Convert text to byte sequence
        ids: list[int] = list(text.encode("utf-8"))

        if verbose:
            print(f"Training BPE: vocab_size={self.vocab_size}, "
                  f"num_merges={num_merges:,}, corpus={len(ids):,} bytes")

        for merge_idx in range(num_merges):
            stats = _count_pairs(ids)
            if not stats:
                if verbose:
                    print(f"  Stopping early at merge {merge_idx}: no more pairs.")
                break

            best_pair = max(stats, key=lambda p: stats[p])
            new_id    = 256 + merge_idx

            self.merges[best_pair] = new_id
            self.vocab[new_id]     = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            ids = _merge_sequence(ids, best_pair, new_id)

            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"  merge {merge_idx + 1:>{len(str(num_merges))}}/{num_merges}  "
                      f"'{self._render(best_pair[0])}' + '{self._render(best_pair[1])}' "
                      f"-> token {new_id}  (freq={stats[best_pair]:,})")

        # Add the end-of-text special token at the end of the vocab
        self._register_special_token(self.ENDOFTEXT)
        self._build_reverse_map()

        if verbose:
            print(f"Training complete. Actual vocab size: {len(self.vocab):,}")

    def _register_special_token(self, token: str) -> int:
        """Add a special token and return its id."""
        if token in self.special_tokens:
            return self.special_tokens[token]
        new_id = len(self.vocab)
        self.special_tokens[token] = new_id
        self.vocab[new_id] = token.encode("utf-8")
        return new_id

    def _build_reverse_map(self) -> None:
        self._bytes_to_id = {v: k for k, v in self.vocab.items()}

    def _render(self, token_id: int) -> str:
        """Human-readable token preview for logging."""
        raw = self.vocab.get(token_id, b"?")
        return raw.decode("utf-8", errors="replace").replace("\n", "\\n").replace("\t", "\\t")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """
        Encode a string to a list of token ids.

        Uses a priority-queue approach: we maintain all adjacent pairs in
        a min-heap keyed by merge rank, so each merge costs O(log N) and
        the total encode cost is O(N log N).

        Args:
            text:        Input string (UTF-8).
            add_special: Prepend <|endoftext|> before the text.
        """
        ids: list[int] = list(text.encode("utf-8"))

        if len(ids) < 2:
            if add_special and self.ENDOFTEXT in self.special_tokens:
                return [self.special_tokens[self.ENDOFTEXT]] + ids
            return ids

        # Doubly-linked list for O(1) merges:  prev[i], next_[i]
        # We use simple arrays here; None marks head/tail sentinels.
        n = len(ids)
        prev_  = list(range(-1, n - 1))   # prev_[0] = -1 (no predecessor)
        next_  = list(range(1, n + 1))    # next_[n-1] = n (no successor)
        active = [True] * n

        # Priority queue: (merge_rank, left_idx)
        # Lower merge rank = earlier merge = higher priority
        heap: list[tuple[int, int]] = []

        def _push_pair(i: int, j: int) -> None:
            pair = (ids[i], ids[j])
            rank = self.merges.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        # Initialise heap with all adjacent pairs
        for i in range(n - 1):
            _push_pair(i, i + 1)

        while heap:
            rank, i = heapq.heappop(heap)

            # Skip stale heap entries
            if not active[i]:
                continue
            j = next_[i]
            if j >= n or not active[j]:
                continue
            if self.merges.get((ids[i], ids[j])) != rank:
                continue

            # Apply merge: replace ids[i] with new token, deactivate ids[j]
            new_id = rank               # merges maps pair -> new_id, which IS the rank+offset
            # Actually: merges[(a,b)] = new_token_id, rank is also the new_token_id value
            # We stored rank as merge rank but value is also new_token_id (they're the same thing
            # since new_id = 256 + merge_index and merge rank = merge_index via insertion order)
            ids[i]     = self.merges[(ids[i], ids[j])]   # re-fetch to be safe
            active[j]  = False

            # Relink linked list
            k = next_[j]
            next_[i] = k
            if k < n:
                prev_[k] = i

            # Push new pairs involving position i
            if prev_[i] >= 0:
                _push_pair(prev_[i], i)
            if next_[i] < n:
                _push_pair(i, next_[i])

        result = [ids[i] for i in range(n) if active[i]]

        if add_special and self.ENDOFTEXT in self.special_tokens:
            result = [self.special_tokens[self.ENDOFTEXT]] + result

        return result

    def encode_batch(
        self,
        texts: list[str],
        pad: bool = False,
        add_special: bool = False,
    ) -> list[list[int]]:
        """Encode a list of strings.  Optionally right-pad to the same length."""
        encoded = [self.encode(t, add_special=add_special) for t in texts]
        if pad and encoded:
            max_len = max(len(e) for e in encoded)
            pad_id  = self.special_tokens.get("<|pad|>", 0)
            encoded = [e + [pad_id] * (max_len - len(e)) for e in encoded]
        return encoded

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token ids back to a string.

        Args:
            ids: List of token ids (may include special tokens, which are
                 rendered as their string representation).
        """
        byte_chunks: list[bytes] = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                raise ValueError(f"Unknown token id: {token_id}")
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Special token helpers
    # ------------------------------------------------------------------

    @property
    def eot_id(self) -> int:
        """End-of-text token id."""
        return self.special_tokens[self.ENDOFTEXT]

    def add_special_token(self, token: str) -> int:
        """Register an additional special token (e.g. <|user|>, <|assistant|>)."""
        return self._register_special_token(token)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save tokenizer to a JSON file.

        Format:
          {
            "vocab_size":    int,
            "merges":        {"a,b": new_id, ...},   # key = "id_a,id_b"
            "vocab":         {"id": [byte, ...]},
            "special_tokens": {"<|endoftext|>": id, ...}
          }
        """
        data = {
            "vocab_size":     self.vocab_size,
            "merges":         {f"{a},{b}": c for (a, b), c in self.merges.items()},
            "vocab":          {str(k): list(v) for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer previously saved with `.save()`."""
        data = json.loads(Path(path).read_text())
        tok  = cls(vocab_size=data["vocab_size"])

        tok.merges = {
            (int(k.split(",")[0]), int(k.split(",")[1])): v
            for k, v in data["merges"].items()
        }
        tok.vocab  = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.special_tokens = data["special_tokens"]
        tok._build_reverse_map()
        return tok

    # ------------------------------------------------------------------
    # Properties & dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"actual={len(self.vocab)}, merges={len(self.merges):,})"
        )


# ---------------------------------------------------------------------------
# Internal helpers (module-level for speed)
# ---------------------------------------------------------------------------

def _count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count frequency of every adjacent (token_a, token_b) pair."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_sequence(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id` in a single pass."""
    result: list[int] = []
    i = 0
    a, b = pair
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result
