"""
Reference-tokenizer comparison (07-nanolm.md Priority 4, item 11).

Compares NanoLM's from-scratch BPE tokenizer against tiktoken's GPT-2
encoding on a fixed corpus and reports:

  1. Token-ID exact-match rate — the fraction of lines where our token ID
     sequence is byte-for-byte identical to tiktoken's. This is EXPECTED
     to be near 0%: our tokenizer is trained from scratch on a small
     Shakespeare corpus with vocab_size=4096, while tiktoken's "gpt2"
     encoding was trained on WebText with vocab_size=50257. Different
     training corpora + different vocab sizes -> different merge tables
     -> different token IDs, even for identical input text. This is not a
     bug; it is stated explicitly here so the divergence is known rather
     than silently assumed away.

  2. Round-trip correctness for BOTH tokenizers on the same corpus — this
     IS a meaningful apples-to-apples check, since both are byte-level BPE
     and both should losslessly round-trip any UTF-8 text regardless of
     vocab.

  3. Compression ratio (chars per token) for both, as a sanity metric —
     not a pass/fail gate, just a reported number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanolm.tokenizer.bpe import BPETokenizer

tiktoken = pytest.importorskip("tiktoken", reason="tiktoken not installed")

REPO_ROOT = Path(__file__).resolve().parent.parent
SHAKESPEARE_RAW = REPO_ROOT / "data" / "shakespeare" / "raw" / "input.txt"
SHAKESPEARE_TOK = REPO_ROOT / "data" / "shakespeare" / "tokenizer.json"


def _fixed_corpus_lines(n_lines: int = 200) -> list[str]:
    if not SHAKESPEARE_RAW.exists():
        pytest.skip("data/shakespeare/raw/input.txt not present — run prepare_data.py first")
    text = SHAKESPEARE_RAW.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[:n_lines]


class TestTiktokenReferenceAgreement:
    def test_report_exact_match_rate_and_roundtrip(self):
        if not SHAKESPEARE_TOK.exists():
            pytest.skip("data/shakespeare/tokenizer.json not present — run prepare_data.py first")

        our_tok = BPETokenizer.load(SHAKESPEARE_TOK)
        ref_enc = tiktoken.get_encoding("gpt2")

        lines = _fixed_corpus_lines(n_lines=200)
        assert len(lines) > 0

        exact_matches = 0
        our_roundtrip_ok = 0
        ref_roundtrip_ok = 0
        our_token_total = 0
        ref_token_total = 0
        char_total = 0

        for line in lines:
            our_ids = our_tok.encode(line)
            ref_ids = ref_enc.encode(line)

            if our_ids == ref_ids:
                exact_matches += 1

            if our_tok.decode(our_ids) == line:
                our_roundtrip_ok += 1
            if ref_enc.decode(ref_ids) == line:
                ref_roundtrip_ok += 1

            our_token_total += len(our_ids)
            ref_token_total += len(ref_ids)
            char_total += len(line)

        n = len(lines)
        exact_match_rate = exact_matches / n
        our_roundtrip_rate = our_roundtrip_ok / n
        ref_roundtrip_rate = ref_roundtrip_ok / n
        our_chars_per_token = char_total / our_token_total
        ref_chars_per_token = char_total / ref_token_total

        print(f"\n[tokenizer reference comparison] corpus: {n} lines from Shakespeare, "
              f"{char_total:,} chars")
        print(f"  vocab sizes         : ours={our_tok.vocab_size:,}  tiktoken(gpt2)={ref_enc.n_vocab:,}")
        print(f"  token-ID exact match: {exact_matches}/{n} = {exact_match_rate:.1%}  "
              f"(expected ~0% -- different training corpora/vocab sizes, NOT a bug)")
        print(f"  round-trip (ours)   : {our_roundtrip_ok}/{n} = {our_roundtrip_rate:.1%}")
        print(f"  round-trip (tiktoken): {ref_roundtrip_ok}/{n} = {ref_roundtrip_rate:.1%}")
        print(f"  compression (ours)  : {our_chars_per_token:.2f} chars/token "
              f"({our_token_total:,} tokens)")
        print(f"  compression (tiktoken): {ref_chars_per_token:.2f} chars/token "
              f"({ref_token_total:,} tokens)")

        # The meaningful, apples-to-apples assertions: both tokenizers must
        # losslessly round-trip this corpus, regardless of their differing
        # vocabularies/token IDs.
        assert our_roundtrip_rate == 1.0, "our tokenizer failed to round-trip some lines"
        assert ref_roundtrip_rate == 1.0, "tiktoken failed to round-trip some lines (unexpected)"

        # Document (not gate on) the expected divergence in token IDs.
        assert exact_match_rate < 0.05, (
            "token-ID exact match rate unexpectedly high — investigate whether "
            "the vocab/corpus assumption above still holds"
        )
