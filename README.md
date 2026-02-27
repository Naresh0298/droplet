# Droplet 1B

A 1.1 billion parameter language model trained from scratch, designed for **structured reasoning** and **tool use**. Built on a £100 budget using 4× RTX 4090 GPUs.

Most 1B models can complete text. Droplet can think step-by-step, call functions, and chain multi-step tool use — capabilities typically reserved for models 10-100× its size.

```
User: What's the weather in London? Email it to my boss.

Droplet:
<think>Two steps needed: 1) get weather, 2) email the results.</think>
<tool_call>{"name": "get_weather", "params": {"location": "London"}}</tool_call>
<tool_result>Cloudy, 12°C, light rain</tool_result>
<tool_call>{"name": "send_email", "params": {"to": "boss", "body": "London: Cloudy, 12°C"}}</tool_call>
Done. I've checked the weather and emailed your boss.
```

## Architecture

Droplet combines cutting-edge techniques from DeepSeek, Meta, and Mistral into a compact 1.1B model:

| Component | Technique | Inspired By |
|-----------|-----------|-------------|
| Attention | Multi-Head Latent Attention (MLA) | DeepSeek-V2 |
| FFN | SwiGLU activation | LLaMA, PaLM |
| Positions | Rotary Position Embeddings (RoPE) | RoFormer |
| Normalization | RMSNorm (pre-norm) | LLaMA |
| Pre-training | Multi-Token Prediction (MTP) | DeepSeek, Meta |
| Post-training | GRPO (Group Relative Policy Optimization) | DeepSeek-R1 |

### Model Specifications

```
Parameters:        1.1B
Hidden dim:        2048
Layers:            22
Attention heads:   32
KV compression:    512 (4× reduction via MLA)
FFN intermediate:  5632
Vocab size:        32,000 (Llama 2 tokenizer)
Context length:    2048
Precision:         BF16
```

## Training Pipeline

Three-phase training following the DeepSeek R1 methodology:

```
Phase 1: Pre-Training          Phase 2: SFT                Phase 3: GRPO
─────────────────────          ──────────────               ─────────────
10B tokens                     37K conversations            Reinforcement learning
6 datasets                     Cold-start → Full SFT        Reasoning + tool-use RL
35 hours on 4× RTX 4090       4 hours                      10 hours

Output: Base model             Output: Instruct model       Output: Reasoning agent
Can: Complete text             Can: Follow instructions     Can: Think + use tools
Can't: Follow instructions     Can't: Self-verify           Can: Self-verify
```

### Data Mix (Pre-Training)

| Dataset | Weight | Purpose |
|---------|--------|---------|
| FineWeb-Edu | 50% | General knowledge, grammar, reasoning |
| StarCoder (Python + JS) | 20% | Code understanding, structured output |
| Wikipedia | 15% | Factual knowledge, entities |
| UltraData-Math | 10% | Numerical reasoning |
| OpenWebMath | 5% | Mathematical proofs, LaTeX |

### SFT Data

| Source | Examples | Purpose |
|--------|----------|---------|
| OpenHermes 2.5 | ~15K | General instruction following |
| GSM8K | ~7K | Math with chain-of-thought |
| glaive-function-calling | ~5K | Tool/function calling |
| Synthetic (Claude API) | ~5K | `<think>` tag reasoning traces |
| Capybara | ~5K | Multi-turn conversation |

## Project Structure

```
droplet/
├── data/
│   ├── prepare_pretrain.py        # Tokenize 10B tokens from 6 datasets
│   ├── prepare_sft.py             # Format SFT data with chat template
│   └── generate_synthetic.py      # Claude API → reasoning traces
├── model/
│   ├── normalization.py           # RMSNorm
│   ├── feedforward.py             # SwiGLU FFN
│   ├── attention.py               # Multi-Head Latent Attention + RoPE
│   ├── architecture.py            # Full model + Multi-Token Prediction
│   ├── tokenizer.py               # Tokenizer wrapper
│   └── __init__.py
├── training/
│   ├── data_loader.py             # Distributed memmap data loader
│   ├── pretrain.py                # FSDP pre-training loop
│   ├── sft.py                     # Two-stage SFT (cold-start + full)
│   ├── grpo.py                    # Group Relative Policy Optimization
│   ├── rewards.py                 # Reward functions for GRPO
│   └── __init__.py
├── scripts/
│   ├── setup_runpod.sh            # One-command environment setup
│   ├── launch_pretrain.sh         # torchrun launcher for pre-training
│   └── launch_sft.sh             # Two-stage SFT launcher
├── evaluation/
│   ├── benchmarks.py              # MMLU, ARC, GSM8K evaluation
│   └── tool_use_eval.py           # Function calling benchmarks
└── configs/
```

## Quick Start

### 1. Data Preparation (CPU pod)

```bash
bash scripts/setup_runpod.sh cpu

python data/prepare_pretrain.py \
    --total_tokens 10_000_000_000 \
    --output_dir /workspace/data \
    --tokenizer_name NousResearch/Llama-2-7b-hf
```

### 2. Pre-Training (4× RTX 4090)

```bash
bash scripts/setup_runpod.sh gpu

# Test first (5 minutes)
bash scripts/launch_pretrain.sh --test

# Full training (35 hours)
tmux new -s train
bash scripts/launch_pretrain.sh
```

### 3. SFT (4× RTX 4090)

```bash
# Generate synthetic reasoning data (local machine)
export ANTHROPIC_API_KEY="sk-ant-..."
python data/generate_synthetic.py --num_examples 5000

# Prepare and run SFT
python data/prepare_sft.py
bash scripts/launch_sft.sh both
```

### 4. GRPO (4× RTX 4090)

```bash
bash scripts/launch_grpo.sh
```

## Key Technical Decisions

**Why MLA over standard Multi-Head Attention?**
MLA compresses KV cache from 2048-dim to 512-dim (4× reduction). At 1.1B parameters, this lets us use more attention heads within the same memory budget, improving quality without increasing cost.

**Why Multi-Token Prediction?**
MTP predicts tokens t+1 AND t+2 simultaneously during pre-training. Meta and DeepSeek showed this improves sample efficiency by 10-15% — critical when you only have 10B tokens and a £100 budget.

**Why GRPO over PPO?**
PPO requires a separate critic model (doubles memory). GRPO compares multiple completions from the same model — no critic needed. This is how DeepSeek R1 achieved reasoning in small models.

**Why Llama 2's tokenizer (32K) over Llama 3's (128K)?**
128K vocab × 2048 hidden dim = 262M embedding parameters — that's 26% of a 1B model wasted on embeddings. 32K vocab keeps embeddings at ~65M params (6.5%).

## Expected Performance

| Benchmark | Droplet 1B | SmolLM2 1.3B | Notes |
|-----------|-----------|--------------|-------|
| MMLU | ~25-30% | ~25-30% | Similar general knowledge |
| GSM8K | ~15-20% | ~10-15% | Better due to reasoning training |
| IFEval | ~35-40% | ~30-35% | Better instruction following |
| BFCL (Function Calling) | ~40-50% | ~10% | **Primary differentiator** |
| Tool Use Accuracy | ~35-45% | ~10% | **Primary differentiator** |

The model's edge is structured reasoning and tool use, not raw knowledge.

## Built From Scratch

Every component is implemented from first principles:

- **RMSNorm** — not `nn.LayerNorm`, hand-written with rsqrt
- **RoPE** — complex-number rotation matrices, not a library call
- **SwiGLU** — gated FFN with SiLU activation
- **Multi-Head Latent Attention** — KV compression with learned down/up projections
- **Multi-Token Prediction** — auxiliary heads sharing the transformer backbone
- **FSDP training** — distributed across 4 GPUs with gradient accumulation
- **Data pipeline** — streaming tokenization, packing, memmap storage

No HuggingFace `Trainer`. No `transformers.AutoModel`. No shortcuts.

## References

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — MLA architecture
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs](https://arxiv.org/abs/2501.12948) — GRPO, cold-start SFT
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) — MTP
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — SwiGLU, RMSNorm, RoPE
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE

## Author

**Naresh Mahendhar** — AI Engineer

Building production-grade AI systems from first principles.
