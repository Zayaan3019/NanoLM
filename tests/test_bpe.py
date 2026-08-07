"""
Unit tests for the BPE tokenizer.

Tests verify:
  - Round-trip encode→decode is lossless
  - Vocabulary size is respected
  - Merge table is built correctly
  - Special tokens are handled
  - Save/load preserves tokenizer state exactly
  - Edge cases: empty string, single char, non-ASCII (UTF-8), very long text
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nanolm.tokenizer.bpe import BPETokenizer, _count_pairs, _merge_sequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS = (
    "aaabdaaabac " * 100
    + "the quick brown fox jumps over the lazy dog " * 50
    + "Hello world! こんにちは 🌍 " * 10
)


@pytest.fixture(scope="module")
def trained_tokenizer() -> BPETokenizer:
    tok = BPETokenizer(vocab_size=512)
    tok.train(CORPUS, verbose=False)
    return tok


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_count_pairs_empty(self):
        assert _count_pairs([]) == {}

    def test_count_pairs_single(self):
        assert _count_pairs([1]) == {}

    def test_count_pairs_basic(self):
        result = _count_pairs([1, 2, 1, 2, 3])
        assert result[(1, 2)] == 2
        assert result[(2, 1)] == 1
        assert result[(2, 3)] == 1

    def test_merge_sequence_no_match(self):
        ids = [1, 2, 3]
        assert _merge_sequence(ids, (4, 5), 99) == [1, 2, 3]

    def test_merge_sequence_full_replace(self):
        ids    = [1, 2, 1, 2]
        result = _merge_sequence(ids, (1, 2), 99)
        assert result == [99, 99]

    def test_merge_sequence_partial(self):
        ids    = [1, 2, 3, 1, 2]
        result = _merge_sequence(ids, (1, 2), 99)
        assert result == [99, 3, 99]

    def test_merge_sequence_overlap(self):
        # When pair overlaps, only non-overlapping matches should be merged
        ids    = [1, 1, 1]
        result = _merge_sequence(ids, (1, 1), 99)
        # First (1,1) merged → [99, 1], not [99, 99]
        assert result == [99, 1]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_base_vocab_size(self):
        tok = BPETokenizer(vocab_size=300)
        # Before training, vocab should be 256 bytes
        assert len(tok.vocab) == 256

    def test_min_vocab_size(self):
        with pytest.raises(ValueError):
            BPETokenizer(vocab_size=100)   # must be >= 257

    def test_trained_vocab_size(self, trained_tokenizer):
        # If the corpus was rich enough for all merges, actual == vocab_size.
        # If the corpus exhausted unique pairs early (early stopping), actual < vocab_size.
        assert len(trained_tokenizer) <= trained_tokenizer.vocab_size
        assert len(trained_tokenizer) >= 257   # at least 256 bytes + endoftext

    def test_vocab_contains_base_bytes(self, trained_tokenizer):
        for i in range(256):
            assert i in trained_tokenizer.vocab
            assert trained_tokenizer.vocab[i] == bytes([i])

    def test_special_token_registered(self, trained_tokenizer):
        assert BPETokenizer.ENDOFTEXT in trained_tokenizer.special_tokens
        eot_id = trained_tokenizer.special_tokens[BPETokenizer.ENDOFTEXT]
        assert eot_id in trained_tokenizer.vocab


# ---------------------------------------------------------------------------
# Round-trip encode / decode
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("text", [
        "Hello, world!",
        "the quick brown fox",
        "aabaabaab",
        "123 + 456 = 789",
        "こんにちは世界",            # Japanese
        "مرحبا بالعالم",              # Arabic
        "🚀🌍💡",                     # Emoji
        "",                           # Empty string
        "a",                          # Single character
        "  \n\t  ",                   # Whitespace only
    ])
    def test_roundtrip(self, trained_tokenizer, text):
        ids      = trained_tokenizer.encode(text)
        decoded  = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_roundtrip_long(self, trained_tokenizer):
        text    = CORPUS[:500]
        ids     = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_encode_returns_list_of_ints(self, trained_tokenizer):
        ids = trained_tokenizer.encode("hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_ids_within_vocab_range(self, trained_tokenizer):
        ids = trained_tokenizer.encode("Hello, world! こんにちは")
        for i in ids:
            assert 0 <= i < trained_tokenizer.vocab_size


# ---------------------------------------------------------------------------
# Adversarial round-trip (07-nanolm.md Priority 4, item 10)
# ---------------------------------------------------------------------------

class TestAdversarialRoundTrip:
    """
    decode(encode(x)) == x on inputs deliberately chosen to stress the
    byte-level BPE implementation: multi-codepoint unicode, emoji (which are
    often surrogate-pair / multi-byte / ZWJ-sequence in UTF-8), runs of
    whitespace, leading/trailing whitespace, and the tokenizer's own special
    token appearing as LITERAL text in the input (not via add_special=True).
    """

    ADVERSARIAL_CASES = [
        # Whitespace runs and boundaries
        "     ",
        "\n\n\n\n\n",
        "\t\t\t",
        " leading space",
        "trailing space ",
        "  both  sides  ",
        "line1\nline2\r\nline3\rline4",
        "tab\tseparated\tvalues",
        "mixed \t\n \r\n whitespace",
        # Unicode: combining diacritics, multi-codepoint graphemes
        "é́́",              # e + combining accents stacked
        "ñö",                    # combining tilde/diaeresis
        "​‌‍",                # zero-width space/non-joiner/joiner
        # Emoji: multi-byte, ZWJ sequences, skin-tone modifiers, flags
        "👨‍👩‍👧‍👦",                              # family (ZWJ sequence)
        "👍🏽",                                # thumbs up + skin tone modifier
        "🇺🇸🇯🇵",                              # regional indicator flags
        "🚀🌍💡🔥✨🎉",
        # Mixed scripts / bidi text
        "Hello مرحبا こんにちは 你好 Привет",
        "English mixed مع العربية and 日本語",
        # Literal special-token text (NOT via add_special=True)
        "<|endoftext|>",
        "text before <|endoftext|> text after",
        "<|endoftext|><|endoftext|><|endoftext|>",
        # Repeats / degenerate patterns
        "a" * 300,
        "ab" * 200,
        " " * 50 + "x" + " " * 50,
        # Punctuation-heavy / symbols
        "!@#$%^&*()_+-=[]{}|;:'\",.<>/?`~",
        # Empty-ish edge cases
        "\x00",       # NUL byte (valid single UTF-8 byte)
        " ",     # non-breaking space
    ]

    @pytest.mark.parametrize("text", ADVERSARIAL_CASES)
    def test_adversarial_roundtrip(self, trained_tokenizer, text):
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text, (
            f"Round-trip failed for {text!r}: got {decoded!r} "
            f"(ids={ids})"
        )

    def test_adversarial_roundtrip_combined_corpus(self, trained_tokenizer):
        """All adversarial cases concatenated into one corpus (stresses
        merge-boundary interactions between adjacent adversarial spans)."""
        text = "".join(self.ADVERSARIAL_CASES)
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_adversarial_roundtrip_on_real_shakespeare_tokenizer(self):
        """
        Same adversarial suite against the tokenizer actually shipped with
        the repo (data/shakespeare/tokenizer.json), not just a test-only
        tokenizer trained on a synthetic corpus.
        """
        from pathlib import Path
        tok_path = Path(__file__).resolve().parent.parent / "data" / "shakespeare" / "tokenizer.json"
        if not tok_path.exists():
            pytest.skip("data/shakespeare/tokenizer.json not present — run prepare_data.py first")

        tok = BPETokenizer.load(tok_path)
        failures = []
        for text in self.ADVERSARIAL_CASES:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            if decoded != text:
                failures.append((text, decoded))

        assert not failures, f"Round-trip failures on shipped tokenizer: {failures}"


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

class TestSpecialTokens:
    def test_eot_id_property(self, trained_tokenizer):
        eot_id = trained_tokenizer.eot_id
        assert isinstance(eot_id, int)
        assert eot_id == trained_tokenizer.special_tokens[BPETokenizer.ENDOFTEXT]

    def test_add_special_token(self, trained_tokenizer):
        # Use a fresh copy to avoid polluting the module-scoped fixture
        tok2 = BPETokenizer.load(self._save_and_load(trained_tokenizer))
        new_id = tok2.add_special_token("<|user|>")
        assert "<|user|>" in tok2.special_tokens
        assert tok2.special_tokens["<|user|>"] == new_id

    def test_encode_with_special(self, trained_tokenizer):
        ids = trained_tokenizer.encode("hello", add_special=True)
        assert ids[0] == trained_tokenizer.eot_id

    @staticmethod
    def _save_and_load(tok: BPETokenizer) -> str:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        tok.save(path)
        return path


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

class TestBatchEncode:
    def test_batch_same_as_single(self, trained_tokenizer):
        texts = ["hello world", "foo bar baz", "abc"]
        batch = trained_tokenizer.encode_batch(texts)
        for text, ids in zip(texts, batch):
            assert ids == trained_tokenizer.encode(text)

    def test_batch_padding(self, trained_tokenizer):
        texts   = ["a", "hello world", "x"]
        padded  = trained_tokenizer.encode_batch(texts, pad=True)
        lengths = [len(p) for p in padded]
        assert len(set(lengths)) == 1   # all equal


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            trained_tokenizer.save(path)

            loaded = BPETokenizer.load(path)

            # Same merges
            assert trained_tokenizer.merges == loaded.merges
            # Same vocab
            assert trained_tokenizer.vocab == loaded.vocab
            # Same special tokens
            assert trained_tokenizer.special_tokens == loaded.special_tokens

    def test_loaded_tokenizer_produces_same_ids(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            trained_tokenizer.save(path)
            loaded = BPETokenizer.load(path)

            text = "Hello world, this is a test of the BPE tokenizer."
            assert trained_tokenizer.encode(text) == loaded.encode(text)

    def test_saved_file_is_valid_json(self, trained_tokenizer):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            trained_tokenizer.save(path)
            data = json.loads(path.read_text())
            assert "merges" in data
            assert "vocab" in data
            assert "special_tokens" in data
            assert "vocab_size" in data


# ---------------------------------------------------------------------------
# Merge quality
# ---------------------------------------------------------------------------

class TestMergeQuality:
    def test_frequent_substrings_merged(self):
        """
        On a corpus of "aaab" repeated many times, 'aa' should be one of
        the first merge rules.
        """
        tok = BPETokenizer(vocab_size=260)
        tok.train("aaab" * 1000, verbose=False)
        # The pair (97, 97) = ('a', 'a') should appear in merges
        merged = tok.merges.get((97, 97))
        assert merged is not None, "Expected ('a','a') to be merged on 'aaab' corpus"

    def test_encode_shorter_than_chars(self):
        """Encoding a repetitive string should produce fewer tokens than characters."""
        tok = BPETokenizer(vocab_size=512)
        tok.train("hello world " * 500, verbose=False)
        text = "hello world hello world hello world"
        ids  = tok.encode(text)
        assert len(ids) < len(text.encode("utf-8")), (
            "BPE should compress repetitive text below byte-level length"
        )
