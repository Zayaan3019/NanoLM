"""
GPT language model — implemented from scratch.

Design choices and rationale:
- Pre-LayerNorm (GPT-J / PaLM style): more stable gradients than post-LN
- Flash Attention via torch.nn.functional.scaled_dot_product_attention (PyTorch >= 2.0):
  O(N) memory vs O(N²) naive, 2–4× faster on modern hardware
- Fused QKV projection: single matmul for Q, K, V — cache-friendly
- Weight-tying between token embedding and LM head (Press & Wolf, 2017):
  reduces parameters by vocab_size × d_model without quality loss
- Scaled residual init (GPT-2 paper): residual branch weights initialised
  with std = 0.02 / sqrt(2 × n_layers) to prevent variance explosion at depth
- KV-cache for autoregressive inference: O(1) per new token instead of O(N)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanolm.model.config import ModelConfig


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses PyTorch 2.0 scaled_dot_product_attention (Flash Attention kernel)
    when available; falls back to explicit masked attention otherwise.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        # Fused QKV: one matmul instead of three
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Fallback: register a persistent causal mask buffer
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
                .view(1, 1, config.max_seq_len, config.max_seq_len),
            )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x:        (B, T, C)
            kv_cache: Optional cached (K, V) tensors for incremental decoding.
                      Shape: (B, n_heads, T_past, head_dim) each.
        Returns:
            y:         (B, T, C)
            new_cache: updated (K, V) cache if kv_cache was provided, else None
        """
        B, T, C = x.shape

        # ---- QKV projection -----------------------------------------------
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to (B, n_heads, T, head_dim)
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)

        # ---- KV cache (inference only) ------------------------------------
        # Concatenate past keys/values when present (incremental decoding).
        # Always return the current (k, v) so the caller can cache them —
        # on the first (warm-up) pass kv_cache=None but we still need to
        # return the prompt's K,V for subsequent single-token decoding steps.
        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
        new_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = (k, v)

        # ---- Attention ----------------------------------------------------
        if self.flash:
            # is_causal=True only when we are not using a KV-cache
            # (with cache the sequence is a single new token — no future to mask)
            is_causal = kv_cache is None
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            T_k = k.size(2)
            attn = (q @ k.transpose(-2, -1)) * scale           # (B, nh, T, T_k)
            if kv_cache is None:
                attn = attn.masked_fill(
                    ~self.causal_mask[:, :, :T, :T_k], float("-inf")
                )
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v                                         # (B, nh, T, head_dim)

        # ---- Merge heads & project ----------------------------------------
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, new_cache


# ---------------------------------------------------------------------------
# Feed-forward MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Position-wise feed-forward network.

    Uses the 4× expansion ratio from the original Transformer paper and
    GELU activation (Hendrycks & Gimpel, 2016) as used in GPT-2/3.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = 4 * config.d_model
        self.c_fc   = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.d_model, bias=config.bias)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Pre-norm Transformer block: LN → Attention → residual, LN → MLP → residual.

    Pre-norm (vs. GPT-2's post-norm) improves gradient flow at depth and
    eliminates the need for learning rate warmup in some settings.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    Decoder-only GPT language model.

    Architecture follows GPT-2 (Radford et al., 2019) with several
    modern improvements:
      - Pre-norm instead of post-norm
      - Flash Attention (Dao et al., 2022) via PyTorch 2.0 SDPA
      - No bias terms in linear layers
      - Weight tying between token embedding and LM head
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte  = nn.Embedding(config.vocab_size, config.d_model),   # token embedding
                wpe  = nn.Embedding(config.max_seq_len, config.d_model),  # position embedding
                drop = nn.Dropout(config.dropout),
                h    = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f = nn.LayerNorm(config.d_model, bias=config.bias),
            )
        )
        # LM head: projects d_model → vocab_size for next-token prediction
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share parameters between token embedding and LM head
        # (reduces params by vocab_size × d_model with no quality loss)
        self.transformer["wte"].weight = self.lm_head.weight

        # Initialise all weights
        self.apply(self._init_weights)

        # Scale residual branch projections to prevent variance blow-up at depth
        # (GPT-2 paper §2.3: std = 0.02 / sqrt(2 × n_layers))
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Args:
            idx:       Token indices (B, T).
            targets:   Target token indices (B, T) for computing cross-entropy loss.
            kv_caches: Per-layer KV caches for incremental decoding.

        Returns:
            logits:       (B, T, vocab_size)  — full sequence if targets given,
                          (B, 1, vocab_size)  — last position only during generation
            loss:         Scalar cross-entropy loss (None if targets not given)
            new_caches:   Updated per-layer KV caches (None if not using cache)
        """
        B, T = idx.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds model maximum {self.config.max_seq_len}"
            )

        # Token + positional embeddings.
        # During incremental KV-cache decoding each new token must use its
        # absolute position in the full sequence, not position 0.
        # kv_caches[0] is None during the warm-up pass (first pass on the full
        # prompt); it becomes (K, V) on subsequent single-token passes.
        if kv_caches is not None and kv_caches[0] is not None:
            T_past = kv_caches[0][0].size(2)    # tokens already in the cache
        else:
            T_past = 0
        pos = torch.arange(T_past, T_past + T, dtype=torch.long, device=idx.device)
        x = self.transformer["drop"](
            self.transformer["wte"](idx) + self.transformer["wpe"](pos)
        )

        # Transformer blocks
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.transformer["h"]):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cache)
            if kv_caches is not None:
                new_caches.append(new_cache)

        x = self.transformer["ln_f"](x)

        if targets is not None:
            # Training: compute loss over all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only project the last token (saves compute)
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_caches if kv_caches is not None else None

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        KV-cache flow (step == 0 is special):
          step 0 : run the full prompt through the model, building the KV cache
                   for all prompt positions. The logit at the last position is
                   used to sample the first new token.
          step 1+: feed only the single last-generated token; the KV cache
                   carries all prior context so attention is O(1) per step.

        Without KV cache the full context window is re-processed each step.

        Args:
            idx:            Prompt token ids (B, T).
            max_new_tokens: Number of new tokens to generate.
            temperature:    Softmax temperature (< 1 = focused, > 1 = random).
            top_k:          Top-k logit truncation before sampling.
            top_p:          Nucleus sampling threshold.
            use_kv_cache:   Use KV-cache for O(1) per-token generation.

        Returns:
            idx: (B, T + max_new_tokens) token ids.
        """
        self.eval()
        kv_caches: Optional[list] = None

        for step in range(max_new_tokens):
            # If the KV-cache is full (all max_seq_len positions occupied) we can no
            # longer assign a valid position embedding to the next token, so fall back
            # to the sliding-window (no-cache) path for the remainder of generation.
            cache_full = (
                kv_caches is not None
                and kv_caches[0] is not None
                and kv_caches[0][0].size(2) >= self.config.max_seq_len
            )
            if use_kv_cache and not cache_full:
                if step == 0:
                    # Warm-up: run full prompt, populate KV cache, get logit for t_{T}
                    idx_in = idx[:, -self.config.max_seq_len:]
                    logits_out, _, kv_caches = self.forward(
                        idx_in, kv_caches=[None] * self.config.n_layers
                    )
                else:
                    # Incremental: feed only the last generated token
                    idx_in = idx[:, -1:]
                    logits_out, _, kv_caches = self.forward(idx_in, kv_caches=kv_caches)
            else:
                idx_in = idx[:, -self.config.max_seq_len:]
                logits_out, _, _ = self.forward(idx_in)

            logits = logits_out[:, -1, :]       # (B, vocab_size)

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k truncation
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = logits.topk(k).values[..., -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_params(self, non_embedding: bool = True) -> int:
        """
        Parameter count.

        Args:
            non_embedding: If True, exclude embedding table weights (matches
                           GPT-3 paper convention for comparing model sizes).
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer["wpe"].weight.numel()
        return n

    def configure_optimizer(
        self,
        weight_decay: float,
        lr: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """
        Create AdamW with decoupled weight decay.

        Only 2-D parameters (weight matrices) receive weight decay.
        Biases, LayerNorm parameters, and embeddings are excluded.
        """
        decay:    dict[str, torch.Tensor] = {}
        no_decay: dict[str, torch.Tensor] = {}

        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay[pn] = p
            else:
                no_decay[pn] = p

        param_groups = [
            {"params": list(decay.values()),    "weight_decay": weight_decay},
            {"params": list(no_decay.values()), "weight_decay": 0.0},
        ]

        # Fused AdamW kernel — available in PyTorch >= 2.0 on CUDA
        extra: dict = {}
        try:
            if device_type == "cuda":
                extra["fused"] = True
        except Exception:
            pass

        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, **extra)
        return optimizer

    def __repr__(self) -> str:  # pragma: no cover
        n_params = self.count_params(non_embedding=True)
        return (
            f"GPT | {self.config.n_layers}L × {self.config.n_heads}H × "
            f"{self.config.d_model}D | {n_params / 1e6:.2f}M params (non-emb)"
        )
