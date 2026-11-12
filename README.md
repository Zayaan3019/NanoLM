# NanoLM — GPT Language Model from Scratch

A complete, production-grade implementation of a GPT-style autoregressive language model,
built entirely from first principles — BPE tokenizer, Flash Attention, gradient accumulation,
mixed-precision training, KV-cache inference, and distributed training (DDP).

The codebase is designed to be readable, well-tested, and scalable from a 3.5 M-parameter
CPU demo all the way to a 345 M-parameter GPT-2 Medium replica on multi-GPU clusters.

---

## Architecture

```
Input tokens (B, T)
      │
      ▼
Token Embedding (vocab_size → d_model)   ← weight-tied with LM head
      +
Position Embedding (T → d_model)         ← learned (not sinusoidal)
      │
      ▼
  ┌───────────────────────────────────┐
  │   Transformer Block × n_layers    │
  │                                   │
  │  ┌─ LayerNorm (pre-norm)          │
  │  │                                │
  │  └→ CausalSelfAttention           │
  │       ├── Fused QKV projection    │
  │       ├── Flash Attention (SDPA)  │
  │       └── Output projection       │
  │                     │             │
  │             + residual            │
  │                                   │
  │  ┌─ LayerNorm (pre-norm)          │
  │  │                                │
  │  └→ MLP (4× expansion, GELU)     │
  │                     │             │
  │             + residual            │
  └───────────────────────────────────┘
      │
      ▼
  Final LayerNorm
      │
      ▼
  LM Head: Linear(d_model → vocab_size)   ← weights shared with token embedding
      │
      ▼
  Cross-entropy loss (training)  /  Softmax + sample (inference)
```

### Key design decisions

| Choice | Rationale |
|--------|-----------|
| Pre-LayerNorm | More stable gradients at depth than GPT-2's post-norm (Xiong et al., 2020) |
| Flash Attention via PyTorch SDPA | O(N) memory, 2–4× faster than naive attention |
| Fused QKV projection | One matmul → better memory bandwidth utilisation |
| Weight tying (emb ↔ LM head) | Reduces params by `vocab × d_model` with no quality loss (Press & Wolf, 2017) |
| Scaled residual init | `std = 0.02 / √(2 × n_layers)` prevents variance blow-up at depth (GPT-2 paper §2.3) |
| No bias in linear layers | Modern best practice — faster, negligible quality difference |
| AdamW β₂ = 0.95 | Recommended in GPT-3 paper over default 0.999 |
| KV-cache | O(1) per token during inference instead of O(N) |
| Byte-level BPE | Handles any UTF-8 text without unknown tokens |

---

## Model Presets

| Preset | Layers | Heads | d_model | Params | Context | Target hardware |
|--------|--------|-------|---------|--------|---------|-----------------|
| `nano` | 6 | 6 | 192 | ~3.5 M | 256 | CPU (any laptop) |
| `small` | 12 | 12 | 384 | ~41 M | 1024 | Single GPU (4 GB) |
| `gpt2` | 12 | 12 | 768 | ~124 M | 1024 | Single A100 / 4× V100 |
| `gpt2_medium` | 24 | 16 | 1024 | ~345 M | 1024 | 4× A100 |

> **What's actually been trained here:** only `nano`, on Shakespeare, on CPU
> (see [Reproducing the reported numbers](#reproducing-the-reported-numbers)
> below for the exact command and measured results). `small`, `gpt2`, and
> `gpt2_medium` are supported configurations, covered by shape/param-count
> tests, and architecturally identical to `nano` at larger scale — but they
> have not been trained end-to-end in this repo. "Scales to 345M" and
> "trained a 345M model" are different claims; only the first is made here.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or editable install:
pip install -e .
```

### 2. Prepare data

```bash
# Shakespeare (~1 MB) — recommended for first run
python prepare_data.py --dataset shakespeare --vocab_size 4096

# TinyStories (~2 GB)
python prepare_data.py --dataset tinystories --vocab_size 10000

# OpenWebText (~38 GB) — requires: pip install datasets
python prepare_data.py --dataset openwebtext --vocab_size 50257
```

### 3. Train

```bash
# Nano model on Shakespeare — trains in ~45 min on CPU (Intel Core Ultra 7)
python train.py --preset nano

# GPT-2 scale on OpenWebText (single GPU)
python train.py --preset gpt2 --dataset openwebtext --device cuda

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train.py --preset gpt2 --dataset openwebtext

# Resume from checkpoint
python train.py --preset nano --resume checkpoints/best.pt
```

### 4. Generate text

```bash
python generate.py \
    --checkpoint checkpoints/best.pt \
    --tokenizer  data/shakespeare/tokenizer.json \
    --prompt     "HAMLET:" \
    --max_tokens 300 --temperature 0.8 --top_p 0.9
```

### 5. Evaluate perplexity

```bash
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --tokenizer  data/shakespeare/tokenizer.json \
    --data       data/shakespeare/val.bin
```

---

## Intel Arc GPU Support (Windows)

This project runs on CPU by default and automatically uses the Intel Arc GPU if
Intel Extension for PyTorch (IPEX) is installed:

```bash
pip install intel-extension-for-pytorch
python train.py --preset nano    # auto-detects XPU
```

The Arc GPU with shared memory (128 MB VRAM) is best used for inference only.
For training, CPU mode with the nano preset is recommended on this hardware.

---

## Repository Structure

```
llm-from-scratch/
│
├── nanolm/                      # Core library
│   ├── model/
│   │   ├── config.py            # ModelConfig + TrainConfig dataclasses
│   │   └── gpt.py               # GPT, Block, CausalSelfAttention, MLP
│   ├── tokenizer/
│   │   └── bpe.py               # Byte-level BPE tokenizer (from scratch)
│   ├── data/
│   │   ├── dataset.py           # Memory-mapped TokenizedDataset
│   │   └── prepare.py           # Download + tokenise Shakespeare/TinyStories/OWT
│   ├── training/
│   │   ├── trainer.py           # Trainer (AMP, gradient accum, DDP, logging)
│   │   └── scheduler.py         # Cosine LR schedule with warmup
│   └── utils/
│       └── checkpoint.py        # Save / load model checkpoints
│
├── train.py                     # Training entry point
├── generate.py                  # Text generation
├── prepare_data.py              # Data preparation entry point
├── evaluate.py                  # Perplexity evaluation
│
├── tests/
│   ├── test_bpe.py                   # BPE unit tests + adversarial round-trip
│   ├── test_model.py                 # Model, causal mask, KV-cache, attention numerics, optimizer
│   ├── test_training.py              # Grad accumulation, clip order, DDP sync/sharding, seeding
│   ├── test_tokenizer_reference.py   # tiktoken reference-agreement comparison
│   └── test_integration.py           # End-to-end pipeline tests
│
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## BPE Tokenizer — Implementation Details

The tokenizer is a from-scratch implementation of Byte-Pair Encoding (Sennrich et al., 2016)
operating at the byte level (like GPT-2), so it handles any UTF-8 text without unknown tokens.

**Training algorithm:**
```
vocab = {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}
corpus_bytes = corpus.encode('utf-8')
ids = list(corpus_bytes)

for merge_idx in range(num_merges):
    pair  = most_frequent_adjacent_pair(ids)
    new_id = 256 + merge_idx
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    ids = replace_all(ids, pair, new_id)
```

**Encoding — priority queue approach:**
Naïve O(N × merges) encoding is replaced with a doubly-linked list + min-heap
keyed by merge rank, achieving O(N log N) per sequence.

**Vocabulary layout:**
```
 0 – 255      : individual bytes (base vocabulary)
 256 – V-2    : merged tokens (V = vocab_size)
 V-1          : <|endoftext|> special token
```

---

## Training Details

### Loss function
Standard cross-entropy over next-token prediction:
```
L = -1/T Σ log P(token_t | token_1, ..., token_{t-1})
```

### Optimiser
AdamW with decoupled weight decay (Loshchilov & Hutter, 2019):
- Weight decay applied only to 2-D parameters (weight matrices)
- Biases, LayerNorm γ/β, and embeddings are excluded

### LR schedule (cosine with warmup)
```
           max_lr × step / warmup_steps        if step < warmup_steps
lr(step) = min_lr + 0.5(1 + cos(π×p)) × Δlr  if warmup ≤ step ≤ max_steps
           min_lr                               if step > max_steps

where p = (step - warmup) / (max_steps - warmup)  ∈ [0, 1]
      Δlr = max_lr - min_lr
```

### Gradient accumulation
Effective batch = `batch_size × grad_accum_steps`.  For the nano model
on CPU this is 12 × 1 = 12 sequences; for GPT-2 scale it is 12 × 40 = 480
sequences ≈ 491K tokens per step (matches the GPT-3 batch schedule).

### Mixed precision
`torch.amp.autocast` with `bfloat16` on CUDA/XPU for ~2× training throughput.
Automatically disabled on CPU (no hardware support).

---

## Scaling Laws (Chinchilla)

Training compute budget C ≈ 6 × N × D, where N = model parameters, D = training tokens.
For optimal performance per FLOP (Hoffmann et al., 2022):
```
N_opt ∝ C^0.50
D_opt ∝ C^0.50
```

| Model | Params | Optimal tokens | Training steps (batch=480K) |
|-------|--------|----------------|------------------------------|
| nano  | 3.5 M  | ~70 M          | ~146 |
| small | 41 M   | ~820 M         | ~1,710 |
| gpt2  | 124 M  | ~2.5 B         | ~5,200 |
| gpt2_medium | 345 M | ~6.9 B | ~14,400 |

The nano model on Shakespeare (266K training tokens from ~1.1M characters) is intentionally undertrained by Chinchilla
standards — this is fine for a demo; the model still learns valid language patterns.

---

## Sample Output (Nano Model — Shakespeare)

Generated from `checkpoints/best.pt` (`python generate.py --checkpoint checkpoints/best.pt
--tokenizer data/shakespeare/tokenizer.json --prompt "HAMLET:" --max_tokens 80 --temperature 0.8
--top_p 0.9 --seed 42`), best val_loss 5.57 / val_ppl 263 at step 1,200 — see
[Reproducing the reported numbers](#reproducing-the-reported-numbers):

```
HAMLET: general ther
As he deed is munruituddance.
For this vextremesome one another at leasle:
Show he kept that movedness
Which thoughts, and then were won to take
There is no good o'er them enjour bornenary me: overthunderstands do confess and stume,
But
```

This is genuinely what the `nano` preset produces on Shakespeare with a 4096-token
vocabulary and 266K training tokens: recognizable structure (character names, verse-like
line breaks, period-appropriate vocabulary) but not fluent, grammatical text. Do not expect
coherent Shakespeare — see the honest numbers below.

---

## Reproducing the reported numbers

Every number below was produced by actually running the commands shown, on this exact
hardware, and is regenerable by anyone with this repo. No number in this README is
asserted without a command that reproduces it.

**Hardware / config for every number below:** Intel Core Ultra 7 155H (CPU-only, no CUDA/XPU),
16 logical threads, PyTorch 2.4.1, `dtype=float32` (AMP disabled — unsupported on CPU),
micro-batch size 12, `grad_accum_steps=1` (effective batch 12 sequences), `seq_len=256`.

```bash
python prepare_data.py --dataset shakespeare --vocab_size 4096
python train.py --preset nano
```

**Validation loss / perplexity — full 5,000-step run:**

| Step | val_loss | val_ppl | |
|-----:|---------:|--------:|---|
| 200 | 6.44 | 629 | |
| 600 | 5.73 | 308 | |
| **1,200** | **5.57** | **263** | ← best checkpoint (`checkpoints/best.pt`) |
| 2,000 | 5.86 | 351 | |
| 3,000 | 6.77 | 875 | |
| 4,000 | 7.70 | 2,214 | |
| 5,000 | 8.18 | 3,551 | (severely overfit — see below) |

**This is the actual, measured behavior — not "val loss ≈ 1.3, ppl ≈ 3.7" as an earlier
version of this README claimed.** That number was never reproducible from this codebase
and has been corrected. What actually happens: the model overfits. Train loss falls
smoothly and monotonically to 0.76 by step 5,000 while val loss falls only until step
~1,200 and then rises for the rest of the run — on a 266K-token training split with
`dropout=0.0` and no early stopping, 5,000 steps × 3,072 tokens is about 58 passes over the data.
The `Trainer` already checkpoints `best.pt` on every val-loss improvement (see
`nanolm/training/trainer.py`), so `checkpoints/best.pt` correctly holds the step-1,200
model, not the overfit step-5,000 one — **always generate/evaluate from `best.pt`,
not a later `step_*.pt` checkpoint.** If you want a better held-out model, the direct
levers are `--max_steps 1200` (stop near the actual optimum), adding dropout, or more
training data.

**Cross-check with `evaluate.py`** (independent of the training loop):

```bash
python evaluate.py --checkpoint checkpoints/best.pt --tokenizer data/shakespeare/tokenizer.json \
    --data data/shakespeare/val.bin --seq_len 256 --batch_size 12
# Mean loss: 6.03  |  Perplexity: 415.84
```

This is in the same ballpark as the training log's 5.57 / 263 but not identical — expected,
not a bug: the training loop's periodic eval (`Trainer.evaluate`) samples up to `eval_steps`
batches of *random, overlapping* windows from `val.bin`, while `evaluate.py` does one full
pass over *every non-overlapping* window in the file. Different sampling of the same
underlying validation set will not produce bit-identical numbers; both are legitimate, and
neither should be read as more "official" than the other. State which one you're citing.

**Throughput:** measured 700–7,100 tok/s across several runs on this machine, with the
low end coming from runs that shared the CPU with other processes (this measurement is
noisy on a general-purpose laptop, not a dedicated benchmark box) and the high end from
an otherwise-idle machine after a few hundred steps of warm-up. Report any single number
("7.5k tok/s") only alongside the hardware/batch/seq_len/precision it was measured
under — a bare number is not meaningful.

**What has and hasn't actually been trained:** only `nano` on Shakespeare, as above.
`small`/`gpt2`/`gpt2_medium` are supported configs exercised by shape and parameter-count
tests (`tests/test_model.py::TestParameterCount`, `TestOutputShapes`) but have not been
trained end-to-end in this repo — "scales to 345M" and "trained a 345M model" are
different claims, and only the first is made here.

---

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=nanolm --cov-report=term-missing

# Specific test file
pytest tests/test_bpe.py -v
pytest tests/test_model.py -v
pytest tests/test_training.py -v
pytest tests/test_tokenizer_reference.py -v
pytest tests/test_integration.py -v
```

---

## References

1. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
2. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* (GPT-2)
3. Brown et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS 2020. (GPT-3)
4. Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention.* NeurIPS 2022.
5. Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL 2016. (BPE)
6. Press & Wolf (2017). *Using the Output Embedding to Improve Language Models.* EACL 2017. (Weight tying)
7. Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models.* NeurIPS 2022. (Chinchilla)
8. Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. (AdamW)
9. Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture.* ICML 2020. (Pre-LN)

---

## License

MIT
