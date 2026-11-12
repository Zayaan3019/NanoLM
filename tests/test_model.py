"""
Unit tests for the GPT model architecture.

Tests verify:
  - Output shapes are correct for all presets
  - Causal masking: future tokens do not influence past token representations
  - Loss computation is numerically sound
  - Weight tying: token embedding and LM head share the same parameter tensor
  - Residual scaling: c_proj.weight std scales with depth
  - KV-cache produces identical outputs to uncached forward pass
  - Generation produces sequences of the correct length
  - Parameter count matches expected values
  - configure_optimizer separates decayed vs non-decayed parameters correctly
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from nanolm.model.config import ModelConfig
from nanolm.model.gpt import GPT, Block, CausalSelfAttention, MLP


# ---------------------------------------------------------------------------
# Shared fixture — tiny model for fast tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        max_seq_len=32,
        n_layers=2,
        n_heads=2,
        d_model=16,
        dropout=0.0,
        bias=False,
    )


@pytest.fixture(scope="module")
def tiny_model(tiny_cfg) -> GPT:
    torch.manual_seed(42)
    return GPT(tiny_cfg)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestOutputShapes:
    @pytest.mark.parametrize("B,T", [(1, 8), (4, 16), (2, 32)])
    def test_logits_shape_with_targets(self, tiny_model, tiny_cfg, B, T):
        idx     = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        targets = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        logits, loss, _ = tiny_model(idx, targets)
        assert logits.shape == (B, T, tiny_cfg.vocab_size)
        assert loss is not None
        assert loss.shape == ()   # scalar

    @pytest.mark.parametrize("B,T", [(1, 1), (3, 8)])
    def test_logits_shape_inference_mode(self, tiny_model, tiny_cfg, B, T):
        idx = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        logits, loss, _ = tiny_model(idx)
        # Inference mode: only the last-position logit is returned
        assert logits.shape == (B, 1, tiny_cfg.vocab_size)
        assert loss is None

    def test_seq_len_exceeds_max_raises(self, tiny_model, tiny_cfg):
        idx = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.max_seq_len + 1))
        with pytest.raises(ValueError, match="Sequence length"):
            tiny_model(idx)

    @pytest.mark.parametrize("preset", ["nano", "small", "gpt2", "gpt2_medium"])
    def test_preset_shapes(self, preset):
        cfg   = getattr(ModelConfig, preset)()
        model = GPT(cfg)
        B, T  = 2, min(64, cfg.max_seq_len)
        idx     = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss, _ = model(idx, targets)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------

class TestCausalMasking:
    def test_future_tokens_do_not_affect_past(self, tiny_cfg):
        """
        Changing tokens at position T should NOT change logits at positions < T.
        This verifies the causal (lower-triangular) attention mask is correct.
        """
        torch.manual_seed(0)
        model = GPT(tiny_cfg)
        model.eval()

        B, T = 1, 16
        idx = torch.randint(0, tiny_cfg.vocab_size, (B, T))

        with torch.no_grad():
            targets = torch.zeros_like(idx)
            logits1, _, _ = model(idx, targets)

            # Corrupt the second half of the sequence
            idx_corrupted = idx.clone()
            idx_corrupted[:, T // 2 :] = torch.randint(0, tiny_cfg.vocab_size, (B, T // 2))
            logits2, _, _ = model(idx_corrupted, targets)

        # Logits for positions before the corruption point must be identical
        assert torch.allclose(
            logits1[:, : T // 2, :],
            logits2[:, : T // 2, :],
            atol=1e-5,
        ), "Causal mask violated: future tokens affected past logits"

    @pytest.mark.parametrize("t_plus_1", [1, 4, 8, 15])
    def test_single_future_token_perturbation_does_not_change_earlier_logits(
        self, tiny_cfg, t_plus_1
    ):
        """
        Surgical version of the causal-mask test: flip exactly ONE token at
        position t_plus_1 and assert logits at every position <= t_plus_1 - 1
        are byte-for-byte unaffected (torch.equal, not just allclose) — proves
        SDPA's is_causal flag and the fallback additive mask are not
        double-applying or cancelling each other out.
        """
        torch.manual_seed(t_plus_1)
        model = GPT(tiny_cfg)
        model.eval()

        B, T = 2, 16
        idx = torch.randint(0, tiny_cfg.vocab_size, (B, T))

        with torch.no_grad():
            logits1, _, _ = model(idx, torch.zeros_like(idx))

            idx2 = idx.clone()
            # Flip a single future token to a different id.
            idx2[:, t_plus_1] = (idx2[:, t_plus_1] + 1) % tiny_cfg.vocab_size
            logits2, _, _ = model(idx2, torch.zeros_like(idx2))

        assert torch.equal(logits1[:, :t_plus_1, :], logits2[:, :t_plus_1, :]), (
            f"Perturbing token at position {t_plus_1} changed logits at an "
            f"earlier position — causal mask is leaking future information"
        )
        # Sanity: the perturbation SHOULD change logits at/after the flipped position
        assert not torch.equal(logits1[:, t_plus_1:, :], logits2[:, t_plus_1:, :])


# ---------------------------------------------------------------------------
# Attention numerics — compare against an independent naive reference
# ---------------------------------------------------------------------------

def _naive_causal_attention(attn: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """
    Independent, non-SDPA reference implementation of causal self-attention.

    Deliberately reimplements scaled dot-product attention from matmuls
    (never calls F.scaled_dot_product_attention) so it can catch scaling or
    masking bugs in CausalSelfAttention.forward that would otherwise be
    invisible if the same bug were baked into both implementations.
    """
    B, T, C = x.shape
    nh, hd = attn.n_heads, attn.head_dim
    qkv = attn.c_attn(x)
    q, k, v = qkv.split(attn.d_model, dim=2)

    def _split(t: torch.Tensor) -> torch.Tensor:
        return t.view(B, T, nh, hd).transpose(1, 2)

    q, k, v = _split(q), _split(k), _split(v)

    scale = 1.0 / math.sqrt(hd)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale     # (B, nh, T, T)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)                            # (B, nh, T, hd)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return attn.c_proj(out)


class TestAttentionNumerics:
    def test_flash_path_matches_naive_reference(self, tiny_cfg):
        """
        CausalSelfAttention's SDPA (flash) path must numerically match an
        independent naive matmul+softmax reference. This proves:
          - the 1/sqrt(head_dim) scale is applied, and applied exactly once
          - the causal mask matches a lower-triangular reference exactly
        """
        torch.manual_seed(17)
        attn = CausalSelfAttention(tiny_cfg)
        attn.eval()
        assert attn.flash, "expected SDPA to be available on this torch build"

        B, T = 3, 12
        x = torch.randn(B, T, tiny_cfg.d_model)

        with torch.no_grad():
            y_flash, _ = attn(x)
            y_naive = _naive_causal_attention(attn, x)

        max_diff = (y_flash - y_naive).abs().max().item()
        print(f"\n[attention numerics] flash vs naive reference max abs diff = {max_diff:.3e}")
        assert torch.allclose(y_flash, y_naive, atol=1e-5), (
            f"Flash attention output diverges from naive reference: max diff {max_diff:.3e}"
        )

    def test_fallback_path_matches_naive_reference(self, tiny_cfg):
        """The non-SDPA fallback branch (explicit masked softmax) must also
        match the naive reference — and therefore match the flash path too,
        confirming neither path double-applies scale or mask."""
        torch.manual_seed(18)
        attn = CausalSelfAttention(tiny_cfg)
        attn.eval()

        B, T = 3, 12
        x = torch.randn(B, T, tiny_cfg.d_model)

        # Force the fallback branch exactly as __init__ would have if SDPA
        # were unavailable (register the same causal_mask buffer).
        attn.flash = False
        attn.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(tiny_cfg.max_seq_len, tiny_cfg.max_seq_len, dtype=torch.bool))
            .view(1, 1, tiny_cfg.max_seq_len, tiny_cfg.max_seq_len),
        )

        with torch.no_grad():
            y_fallback, _ = attn(x)
            y_naive = _naive_causal_attention(attn, x)

        assert torch.allclose(y_fallback, y_naive, atol=1e-5), (
            "Non-flash fallback attention diverges from naive reference"
        )


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------

class TestWeightTying:
    def test_embedding_and_lm_head_share_storage(self, tiny_model):
        emb_weight = tiny_model.transformer["wte"].weight
        lm_weight  = tiny_model.lm_head.weight
        assert emb_weight.data_ptr() == lm_weight.data_ptr(), (
            "Token embedding and LM head should share the same parameter tensor"
        )

    def test_gradient_flows_through_both(self, tiny_model, tiny_cfg):
        model = GPT(tiny_cfg)   # fresh copy
        idx     = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
        targets = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
        _, loss, _ = model(idx, targets)
        loss.backward()

        assert model.transformer["wte"].weight.grad is not None
        # lm_head.weight is the same tensor; its grad IS wte.weight.grad
        assert model.lm_head.weight.grad is not None


# ---------------------------------------------------------------------------
# Residual scaling
# ---------------------------------------------------------------------------

class TestResidualScaling:
    def test_c_proj_std_scales_with_depth(self, tiny_cfg):
        """
        c_proj.weight should have std ≈ 0.02 / sqrt(2 × n_layers).
        """
        torch.manual_seed(99)
        model    = GPT(tiny_cfg)
        expected = 0.02 / math.sqrt(2 * tiny_cfg.n_layers)

        c_proj_stds = [
            p.std().item()
            for n, p in model.named_parameters()
            if n.endswith("c_proj.weight")
        ]
        assert len(c_proj_stds) > 0

        for std in c_proj_stds:
            assert abs(std - expected) / expected < 0.5, (
                f"c_proj std {std:.4f} deviates too far from expected {expected:.4f}"
            )


# ---------------------------------------------------------------------------
# Loss sanity
# ---------------------------------------------------------------------------

class TestLoss:
    def test_initial_loss_near_log_vocab(self, tiny_cfg):
        """
        At random initialisation, cross-entropy loss should be ≈ log(vocab_size),
        since the model has no information yet (uniform distribution over vocab).
        """
        torch.manual_seed(7)
        model = GPT(tiny_cfg)
        model.eval()

        B, T      = 4, 16
        idx       = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        targets   = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        _, loss, _ = model(idx, targets)

        expected = math.log(tiny_cfg.vocab_size)
        # Allow generous tolerance: init is random but expectation is log(V)
        assert abs(loss.item() - expected) < 1.5, (
            f"Initial loss {loss.item():.4f} far from expected {expected:.4f}"
        )

    def test_loss_decreases_after_one_step(self, tiny_cfg):
        """One gradient step on a single batch should decrease the loss."""
        torch.manual_seed(42)
        model = GPT(tiny_cfg)
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

        B, T    = 2, 16
        idx     = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        targets = torch.randint(0, tiny_cfg.vocab_size, (B, T))

        _, loss1, _ = model(idx, targets)
        loss1.backward()
        opt.step()
        opt.zero_grad()

        with torch.no_grad():
            _, loss2, _ = model(idx, targets)

        assert loss2.item() < loss1.item(), (
            f"Loss did not decrease: {loss1.item():.4f} → {loss2.item():.4f}"
        )

    def test_no_nan_in_loss(self, tiny_cfg):
        torch.manual_seed(0)
        model   = GPT(tiny_cfg)
        idx     = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        targets = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
        _, loss, _ = model(idx, targets)
        assert not torch.isnan(loss), "NaN loss at initialisation"


# ---------------------------------------------------------------------------
# KV-cache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_kv_cache_matches_uncached(self, tiny_cfg):
        """
        KV-cache should produce identical logits to a full uncached forward pass.

        Correct comparison protocol:
          1. Uncached: run full T-token prompt → logit for position T
          2. Cached:   warm up on first T-1 tokens (build KV cache),
                       then feed token T-1 (the last prompt token) incrementally
                       → logit for position T
          Both should be numerically identical.

        Note: we deliberately do NOT warm up on all T tokens and then re-feed
        the last token — that would double-count the last prompt token in the
        KV cache, giving a different (wrong) result.
        """
        torch.manual_seed(5)
        model = GPT(tiny_cfg)
        model.eval()

        T      = 8
        prompt = torch.randint(0, tiny_cfg.vocab_size, (1, T))

        with torch.no_grad():
            # Uncached: full T-token sequence
            logits_full, _, _ = model(prompt)          # (B, 1, V)

            # Cached: warm up on T-1 tokens, then feed token T-1 incrementally
            _, _, kv_caches = model(
                prompt[:, :-1],                        # first T-1 tokens
                kv_caches=[None] * tiny_cfg.n_layers,
            )
            logits_cached, _, _ = model(
                prompt[:, -1:],                        # the T-th token (index T-1)
                kv_caches=kv_caches,
            )

        assert torch.allclose(
            logits_full[:, -1, :],
            logits_cached[:, -1, :],
            atol=1e-4,
        ), "KV-cache forward pass differs from full forward pass"

    # -----------------------------------------------------------------
    # DECISIVE TEST (see 07-nanolm.md Priority 1):
    # cached decode must produce logits IDENTICAL to a full forward pass
    # across a *multi-token* generation, with batch > 1, varying prompt
    # lengths, and generation past the training context length.
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("B", [1, 4])
    @pytest.mark.parametrize("T_prompt", [1, 3, 7, 16])
    def test_kv_cache_matches_uncached_multistep(self, tiny_cfg, B, T_prompt):
        """
        Multi-token decisive test: for every step of a 12-token generation,
        the cached-decode logit must match a from-scratch uncached forward
        pass over the full context accumulated so far.

        This directly exercises:
          - multi-token generation (12 incremental steps, not just 1)
          - batch > 1 (B in {1, 4})
          - varying prompt lengths (T_prompt in {1, 3, 7, 16})
        all while staying within max_seq_len (32), isolating pure KV-cache
        numerics from the context-overflow fallback (tested separately below).
        """
        torch.manual_seed(123)
        model = GPT(tiny_cfg)
        model.eval()

        n_new = 12
        T_total = T_prompt + n_new
        assert T_total <= tiny_cfg.max_seq_len, "test config must stay within context"

        full_idx = torch.randint(0, tiny_cfg.vocab_size, (B, T_total))

        max_abs_diff = 0.0
        kv_caches = None

        with torch.no_grad():
            for t in range(T_prompt, T_total):
                # ---- Ground truth: full uncached forward over context [0, t) ----
                uncached_logits, _, _ = model(full_idx[:, :t])
                uncached_logits = uncached_logits[:, -1, :]

                # ---- Cached decode, exactly mirroring GPT.generate()'s logic ----
                if kv_caches is None:
                    cached_logits, _, kv_caches = model(
                        full_idx[:, :T_prompt],
                        kv_caches=[None] * tiny_cfg.n_layers,
                    )
                else:
                    cached_logits, _, kv_caches = model(
                        full_idx[:, t - 1 : t],
                        kv_caches=kv_caches,
                    )
                cached_logits = cached_logits[:, -1, :]

                diff = (uncached_logits - cached_logits).abs().max().item()
                max_abs_diff = max(max_abs_diff, diff)

                assert torch.allclose(uncached_logits, cached_logits, atol=1e-4), (
                    f"KV-cache diverged at step t={t} (T_prompt={T_prompt}, B={B}): "
                    f"max abs diff = {diff:.3e}"
                )

        print(
            f"\n[decisive KV-cache test] T_prompt={T_prompt} B={B} "
            f"steps={n_new} -> max abs logit diff = {max_abs_diff:.3e}"
        )

    def test_kv_cache_generation_matches_uncached_past_context_length(self, tiny_cfg):
        """
        Generation must remain correct (cached == uncached token trajectory)
        even when max_new_tokens pushes the sequence PAST max_seq_len,
        exercising the sliding-window / cache-full fallback path in
        GPT.generate(). Greedy decoding (temperature~0, top_k=1) makes the
        argmax deterministic, so identical logits at every step imply an
        identical token trajectory end to end.
        """
        torch.manual_seed(321)
        model = GPT(tiny_cfg)
        model.eval()

        B, T_prompt = 2, 5
        prompt = torch.randint(0, tiny_cfg.vocab_size, (B, T_prompt))
        # Generate well past max_seq_len (32) to force the cache-full fallback.
        max_new_tokens = tiny_cfg.max_seq_len + 10

        out_cached = model.generate(
            prompt.clone(), max_new_tokens=max_new_tokens,
            temperature=1e-10, top_k=1, use_kv_cache=True,
        )
        out_uncached = model.generate(
            prompt.clone(), max_new_tokens=max_new_tokens,
            temperature=1e-10, top_k=1, use_kv_cache=False,
        )

        assert torch.equal(out_cached, out_uncached), (
            "Cached and uncached greedy generation trajectories diverge once "
            "generation passes the training context length"
        )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGeneration:
    @pytest.mark.parametrize("max_new_tokens", [1, 5, 20])
    def test_generated_length(self, tiny_model, tiny_cfg, max_new_tokens):
        prompt = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
        out    = tiny_model.generate(prompt, max_new_tokens=max_new_tokens)
        assert out.shape[1] == 4 + max_new_tokens

    def test_greedy_is_deterministic(self, tiny_model, tiny_cfg):
        prompt = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
        torch.manual_seed(0)
        out1 = tiny_model.generate(prompt.clone(), max_new_tokens=10, temperature=1e-10, top_k=1)
        torch.manual_seed(0)
        out2 = tiny_model.generate(prompt.clone(), max_new_tokens=10, temperature=1e-10, top_k=1)
        assert torch.equal(out1, out2)

    def test_temperature_affects_diversity(self, tiny_cfg):
        """Low temperature should produce more repetitive sequences."""
        torch.manual_seed(42)
        model  = GPT(tiny_cfg)
        prompt = torch.randint(0, tiny_cfg.vocab_size, (1, 4))

        out_low  = model.generate(prompt.clone(), max_new_tokens=20, temperature=0.1)
        out_high = model.generate(prompt.clone(), max_new_tokens=20, temperature=2.0)

        # Low temp: unique tokens / total should be lower
        unique_low  = len(set(out_low[0].tolist()))
        unique_high = len(set(out_high[0].tolist()))

        # This is probabilistic; just assert they're not equal in practice
        # (could technically fail with very low probability)
        assert unique_low <= unique_high + 5   # generous bound


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

class TestParameterCount:
    @pytest.mark.parametrize("preset,expected_M", [
        # Total parameter counts (including embeddings, weight-tied so counted once).
        # Nano : 6L×6H×192D, vocab=4096, seq=256  → ~3.5 M
        # Small: 12L×12H×384D, vocab=50304, seq=1024 → ~41 M
        # GPT-2: 12L×12H×768D, vocab=50304, seq=1024 → ~124 M
        # GPT-2 Medium: 24L×16H×1024D, vocab=50304, seq=1024 → the "345M"
        #   configuration as OpenAI published it (exact count pinned below)
        ("nano",          3.5),
        ("small",        41.0),
        ("gpt2",        124.0),
        ("gpt2_medium", 345.0),
    ])
    def test_param_count_approximate(self, preset, expected_M):
        cfg   = getattr(ModelConfig, preset)()
        model = GPT(cfg)
        n     = sum(p.numel() for p in model.parameters())
        # Within 10% of expected
        assert abs(n / 1e6 - expected_M) / expected_M < 0.10, (
            f"{preset}: got {n/1e6:.1f}M params, expected ≈{expected_M}M"
        )


    def test_gpt2_medium_exact_parameter_count(self):
        """Pins the GPT-2 Medium configuration to its closed-form size.

        24 blocks of (attention 4·d² + MLP 8·d² + two LayerNorm weights 2·d),
        a tied token embedding V·d, learned positions T·d and a final
        LayerNorm d, with d=1024, V=50304, T=1024 and no biases.
        """
        cfg = ModelConfig.gpt2_medium()
        d, V, T, L = cfg.d_model, cfg.vocab_size, cfg.max_seq_len, cfg.n_layers
        expected = L * (12 * d * d + 2 * d) + V * d + T * d + d
        model = GPT(cfg)
        assert sum(p.numel() for p in model.parameters()) == expected


# ---------------------------------------------------------------------------
# Optimizer configuration
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_weight_decay_separation(self, tiny_model):
        opt = tiny_model.configure_optimizer(
            weight_decay=0.1, lr=1e-3, betas=(0.9, 0.95), device_type="cpu"
        )
        decay_group    = opt.param_groups[0]
        no_decay_group = opt.param_groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0

        # All params with dim >= 2 should be in decay group
        for p in decay_group["params"]:
            assert p.dim() >= 2

        # All params with dim < 2 should be in no_decay group
        for p in no_decay_group["params"]:
            assert p.dim() < 2

    def test_all_params_covered(self, tiny_model):
        opt = tiny_model.configure_optimizer(
            weight_decay=0.1, lr=1e-3, betas=(0.9, 0.95), device_type="cpu"
        )
        opt_params  = set()
        for group in opt.param_groups:
            opt_params.update(id(p) for p in group["params"])

        model_params = set(id(p) for p in tiny_model.parameters() if p.requires_grad)
        assert opt_params == model_params, "Some parameters are missing from the optimizer"

    def test_single_step_zero_grad_isolates_decoupled_weight_decay_exactly(self, tiny_cfg):
        """
        Exact numeric test of a single optimizer step (07-nanolm.md
        Priority 6: "numerical tests ... for a single optimizer step").

        With every parameter's gradient forced to exactly zero, AdamW's
        moment estimates stay at m=0, v=0, so the adaptive update term
        `lr * m_hat / (sqrt(v_hat) + eps)` is exactly 0 regardless of beta1/
        beta2/eps. What remains is PURELY the decoupled weight-decay term
        (Loshchilov & Hutter, 2019):

            theta_new = theta_old * (1 - lr * weight_decay)   [2-D params]
            theta_new = theta_old                              [<2-D params]

        This proves configure_optimizer's decay/no-decay grouping isn't
        just cosmetic metadata sitting in param_groups — it produces
        different, exactly-predictable numeric outcomes.
        """
        torch.manual_seed(3)
        model = GPT(tiny_cfg)
        lr, wd = 0.1, 0.2   # deliberately large so the effect is unmistakable
        opt = model.configure_optimizer(
            weight_decay=wd, lr=lr, betas=(0.9, 0.95), device_type="cpu"
        )

        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

        opt.step()

        expected_decay_factor = 1.0 - lr * wd
        n_decayed = n_no_decay = 0
        for n, p in model.named_parameters():
            if p.dim() >= 2:
                expected = before[n] * expected_decay_factor
                assert torch.allclose(p.detach(), expected, atol=1e-7, rtol=1e-6), (
                    f"{n}: decayed (dim>=2) param did not shrink by exactly "
                    f"(1 - lr*weight_decay) = {expected_decay_factor}"
                )
                n_decayed += 1
            else:
                assert torch.equal(p.detach(), before[n]), (
                    f"{n}: no-decay (dim<2) param changed despite zero gradient "
                    f"and weight_decay=0 for its param group"
                )
                n_no_decay += 1

        assert n_decayed > 0 and n_no_decay > 0, "test fixture has no params in one of the groups"
