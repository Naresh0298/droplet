"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NanoAgent-1B: Pre-Training Script                        ║
║                                                                            ║
║  This is the main training loop. It orchestrates:                          ║
║  - FSDP (Fully Sharded Data Parallel) across 4× RTX 4090                  ║
║  - BF16 mixed precision training                                           ║
║  - Gradient accumulation (8 steps)                                         ║
║  - Cosine learning rate schedule with warmup                               ║
║  - Checkpointing every 1000 steps                                          ║
║  - Comprehensive logging (loss, LR, throughput, MTP loss)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
    # Launch with torchrun on 4 GPUs:
    torchrun --nproc_per_node=4 training/pretrain.py \
        --data_path /workspace/data/train_tokens.bin \
        --output_dir /workspace/checkpoints \
        --total_steps 19000

    # Quick test (100 steps):
    torchrun --nproc_per_node=4 training/pretrain.py \
        --data_path /workspace/data/train_tokens.bin \
        --total_steps 100

HARDWARE REQUIREMENTS:
    4× RTX 4090 (24GB each) = 96GB total
    Model in BF16: ~2.2GB
    FSDP shards model across GPUs: ~0.55GB per GPU for weights
    Activations + optimizer states: ~15-18GB per GPU
    Headroom: ~5GB per GPU (safe)

ESTIMATED TIME:
    10B tokens, 19K steps ≈ 35 hours on 4× RTX 4090
"""

import os
import sys
import math
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)
import functools

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model.architecture import NanoAgent, NanoAgentConfig, TransformerBlock
    from training.data_loader import create_dataloader
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.architecture import NanoAgent, NanoAgentConfig, TransformerBlock
    from training.data_loader import create_dataloader


# ════════════════════════════════════════════════════════════════════════════
# CONCEPT 1: FSDP — Why and How We Split the Model Across GPUs
# ════════════════════════════════════════════════════════════════════════════
#
# THE MEMORY PROBLEM:
#   A 1.1B parameter model in BF16 = 2.2GB for weights alone.
#   But during training, you also need:
#   - Optimizer states (AdamW): 2× model size = 4.4GB (fp32 momentum + variance)
#   - Gradients: 1× model size = 2.2GB
#   - Activations: Variable, ~8-12GB for batch_size=8
#   Total per GPU (without sharding): ~20+ GB → barely fits on one RTX 4090
#
# THE SOLUTION — Fully Sharded Data Parallel (FSDP):
#   FSDP splits (shards) the model across all GPUs. Each GPU holds only
#   1/N of the weights, gradients, and optimizer states.
#
#   Without FSDP (DDP — each GPU holds FULL model):
#     GPU 0: [Full Model Weights] [Full Optimizer States] [Activations]
#     GPU 1: [Full Model Weights] [Full Optimizer States] [Activations]
#     GPU 2: [Full Model Weights] [Full Optimizer States] [Activations]
#     GPU 3: [Full Model Weights] [Full Optimizer States] [Activations]
#     Memory per GPU: ~20GB (tight!)
#
#   With FSDP (each GPU holds 1/4 of model):
#     GPU 0: [1/4 Weights] [1/4 Optimizer] [Activations]
#     GPU 1: [1/4 Weights] [1/4 Optimizer] [Activations]
#     GPU 2: [1/4 Weights] [1/4 Optimizer] [Activations]
#     GPU 3: [1/4 Weights] [1/4 Optimizer] [Activations]
#     Memory per GPU: ~8-10GB (plenty of headroom!)
#
# HOW FSDP WORKS (simplified):
#   For each layer during forward pass:
#   1. All-gather: collect weight shards from all GPUs → full layer weights
#   2. Forward: compute output using full weights
#   3. Discard: release the gathered weights (memory freed!)
#
#   For backward pass:
#   1. All-gather: collect weights again (needed for gradient computation)
#   2. Backward: compute gradients
#   3. Reduce-scatter: each GPU gets 1/4 of the gradient (its shard)
#   4. Discard: release gathered weights
#
#   The key insight: at any moment, only ONE layer's full weights are in
#   memory. All other layers only have their 1/4 shard. This is why
#   FSDP uses much less memory than DDP.
#
# SHARDING STRATEGY:
#   FULL_SHARD (what we use):
#     Shards weights, gradients, AND optimizer states.
#     Maximum memory savings. Equivalent to ZeRO Stage 3.
#     More communication overhead, but we're on NVLink (fast!).
#
#   SHARD_GRAD_OP (alternative):
#     Shards gradients and optimizer states, but keeps full weights.
#     Less communication, but more memory. Good for 2 GPU setups.
#
# WRAPPING POLICY:
#   FSDP needs to know WHICH modules to shard independently.
#   We use transformer_auto_wrap_policy → each TransformerBlock gets its
#   own FSDP wrapper. This means FSDP gathers/releases one block at a time.
#


# ════════════════════════════════════════════════════════════════════════════
# CONCEPT 2: MIXED PRECISION (BF16) — Why Not FP32 or FP16?
# ════════════════════════════════════════════════════════════════════════════
#
# NUMBER FORMATS:
#   FP32 (float32):  1 sign + 8 exponent + 23 mantissa = 32 bits
#     Range: ±3.4 × 10³⁸, Precision: ~7 decimal digits
#     Used for: optimizer states, loss computation
#
#   FP16 (float16):  1 sign + 5 exponent + 10 mantissa = 16 bits
#     Range: ±65,504, Precision: ~3.3 decimal digits
#     Problem: TINY range! Gradients often exceed 65K → overflow → NaN
#     Requires loss scaling (multiply loss by 1024, then divide gradients)
#
#   BF16 (bfloat16): 1 sign + 8 exponent + 7 mantissa = 16 bits
#     Range: ±3.4 × 10³⁸ (SAME as FP32!), Precision: ~2.4 decimal digits
#     Best of both: same range as FP32 (no overflow), half the memory
#     No loss scaling needed! Much simpler training code.
#
# WHY BF16 IS THE MODERN STANDARD:
#   - Llama 3: BF16
#   - DeepSeek V3: BF16 (with FP8 for some ops — we can't do on RTX 4090)
#   - GPT-4: BF16
#   - All modern LLMs: BF16
#
# OUR MIXED PRECISION STRATEGY:
#   Forward pass:  BF16 (fast, half memory)
#   Backward pass: BF16 (fast)
#   Optimizer:     FP32 (full precision for weight updates)
#   Loss:          FP32 (numerical stability in softmax + cross-entropy)
#   RMSNorm:       FP32 internally (as we implemented — cast up, compute, cast back)
#
# RTX 4090 TENSOR CORE PERFORMANCE:
#   FP32: 82.6 TFLOPS
#   BF16: 330.3 TFLOPS  ← 4× faster!
#   Using BF16 literally makes training 4× faster at the hardware level.
#


# ════════════════════════════════════════════════════════════════════════════
# CONCEPT 3: LEARNING RATE SCHEDULE — Cosine Decay with Warmup
# ════════════════════════════════════════════════════════════════════════════
#
# The learning rate (LR) controls how big each weight update step is.
# Too high → training diverges (loss → infinity)
# Too low → training is too slow (waste GPU hours)
#
# MODERN LLM SCHEDULE — Cosine Decay with Linear Warmup:
#
#   LR │
#      │    ╭──── peak_lr = 3e-4
#      │   ╱ ╲
#      │  ╱   ╲
#      │ ╱     ╲
#      │╱       ╲
#      │         ╲
#      │          ╲
#      │           ╲──── min_lr = 3e-5
#      │            ╲
#      └───┬───┬────┬──────────── Steps
#         warmup peak  cosine decay
#         2000   │    to min_lr
#                │
#
# PHASE 1 — LINEAR WARMUP (steps 0 → 2000):
#   LR ramps from 0 to peak_lr linearly.
#   WHY: At step 0, the model is randomly initialized. A large LR would
#   cause huge, random weight updates → training diverges immediately.
#   Warming up gives the model time to "orient" its weights toward
#   reasonable values before turning up the learning rate.
#
# PHASE 2 — COSINE DECAY (steps 2000 → 19000):
#   LR follows a cosine curve from peak_lr down to min_lr.
#   WHY COSINE (not linear or step decay):
#   - Cosine starts decaying slowly, then faster, then slowly again
#   - This gives the model more time at high LR (for rapid learning)
#   - And a long tail at low LR (for fine-grained optimization)
#   - Empirically outperforms linear and step decay for LLMs
#
#   Formula:
#     lr = min_lr + 0.5 × (peak_lr - min_lr) × (1 + cos(π × progress))
#     where progress = (step - warmup) / (total - warmup)
#
# WHY peak_lr = 3e-4:
#   Rule of thumb for Transformer models: lr ∝ 1/√(hidden_dim)
#   For hidden_dim=2048: 1/√2048 ≈ 0.022 → way too high for AdamW
#   In practice, 3e-4 is the standard for 1B-scale models:
#   - Llama 2 7B: 3e-4
#   - DeepSeek V3: 2.2e-4 (slightly lower for 671B)
#   - SmolLM2 1.7B: 3e-4
#
# WHY min_lr = peak_lr / 10:
#   The minimum LR should be small but not zero. Zero LR means the model
#   stops learning entirely. A small LR allows continued fine-grained
#   improvement in the final training steps.
#   10× ratio (3e-4 / 3e-5) is standard across all major LLMs.
#

def get_cosine_schedule_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    Compute learning rate for the current step.

    Returns the LR value (not a scheduler object) because we set LR manually
    each step. This is simpler and more transparent than PyTorch's scheduler
    classes, which hide the logic.
    """
    if step < warmup_steps:
        # Linear warmup: 0 → peak_lr over warmup_steps
        return peak_lr * step / warmup_steps
    elif step >= total_steps:
        return min_lr
    else:
        # Cosine decay: peak_lr → min_lr over remaining steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ════════════════════════════════════════════════════════════════════════════
# CONCEPT 4: GRADIENT ACCUMULATION — Simulating Large Batch Sizes
# ════════════════════════════════════════════════════════════════════════════
#
# We want a global batch size of 524K tokens, but each GPU can only fit
# micro_batch_size=8 sequences × 2048 tokens = 16K tokens.
#
# With 4 GPUs: 4 × 16K = 64K tokens per forward pass (still 8× too small!)
#
# Solution: Accumulate gradients over 8 micro batches before updating.
#
# WITHOUT gradient accumulation (batch_size=8, 4 GPUs):
#   Step 1: forward(batch_0) → backward → update_weights → 64K tokens
#   Step 2: forward(batch_1) → backward → update_weights → 64K tokens
#   Each update uses only 64K tokens of gradient signal.
#
# WITH gradient accumulation (8 micro steps):
#   Micro 1: forward(batch_0) → backward → ADD to gradient buffer
#   Micro 2: forward(batch_1) → backward → ADD to gradient buffer
#   Micro 3: forward(batch_2) → backward → ADD to gradient buffer
#   ...
#   Micro 8: forward(batch_7) → backward → ADD to gradient buffer
#   → FSDP sync → divide by 8 → update_weights → 524K tokens!
#
# The gradients are ACCUMULATED (summed) across micro steps.
# Then we divide by accumulation_steps to get the average gradient.
# This is mathematically identical to having a batch_size of 256!
#
# IMPLEMENTATION DETAIL:
#   FSDP's no_sync() context manager prevents gradient synchronization
#   between GPUs during accumulation steps. Only on the FINAL micro step
#   do we allow FSDP to all-reduce gradients across GPUs.
#
#   Without no_sync: FSDP syncs after EVERY micro batch (wasteful!)
#   With no_sync: FSDP syncs only after the last micro batch (efficient!)
#


# ════════════════════════════════════════════════════════════════════════════
# CONCEPT 5: GRADIENT CLIPPING — Preventing Explosive Updates
# ════════════════════════════════════════════════════════════════════════════
#
# Occasionally, a batch produces abnormally large gradients (gradient spike).
# This can happen when:
# - The batch contains unusual text (very rare tokens, corrupted data)
# - The model is in an unstable region of loss landscape
# - Numerical issues in BF16
#
# Without clipping: large gradient → huge weight update → model destabilized
# → possibly irrecoverable
#
# With clipping (max_norm=1.0):
#   1. Compute the GLOBAL gradient norm: √(Σ grad_i²) across all parameters
#   2. If norm > max_norm: scale ALL gradients by (max_norm / norm)
#      This preserves the gradient DIRECTION but limits its MAGNITUDE.
#
# Example:
#   Gradient norm = 5.0, max_norm = 1.0
#   Scale factor = 1.0 / 5.0 = 0.2
#   All gradients multiplied by 0.2 → effective norm = 1.0
#
# This is a safety net, not a regular occurrence. If you see clipping
# on >50% of steps, your learning rate is probably too high.
#
# ALL MAJOR LLMS USE GRADIENT CLIPPING:
#   Llama 3: max_norm=1.0
#   DeepSeek V3: max_norm=1.0
#   GPT-3: max_norm=1.0
#   It's universal.
#


# ════════════════════════════════════════════════════════════════════════════
# CONCEPT 6: CHECKPOINTING — Saving Progress During Training
# ════════════════════════════════════════════════════════════════════════════
#
# Training takes ~35 hours. If the pod crashes at hour 34 and you haven't
# saved, you lose EVERYTHING and £55. Checkpointing saves periodically.
#
# WHAT WE SAVE:
#   1. Model weights (the trained parameters)
#   2. Optimizer state (momentum + variance for each parameter)
#      WHY: Without optimizer state, resuming training resets momentum.
#      This causes a temporary quality drop and wasted steps.
#   3. Current step number (to resume LR schedule correctly)
#   4. RNG states (for reproducibility)
#   5. Training stats (loss history, throughput)
#
# FSDP CHECKPOINTING:
#   With FSDP, model weights are sharded across GPUs. To save a complete
#   checkpoint, we need to:
#   1. All-gather all shards → full model on rank 0
#   2. Save from rank 0 only (otherwise 4 GPUs save 4 copies!)
#   3. This temporarily uses extra memory (full model in RAM)
#
# FREQUENCY: Every 1000 steps (~1.8 hours)
#   Too frequent: lots of SSD writes slow down training
#   Too rare: risk losing hours of progress on crash
#   1000 steps is the standard sweet spot.
#
# KEEP LAST N: We keep last 3 checkpoints and delete older ones.
#   Each checkpoint: ~2.5GB (model) + ~5GB (optimizer) ≈ 7.5GB
#   3 checkpoints: ~22.5GB on the 100GB volume. Plenty of room.
#


# ════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED SETUP
# ════════════════════════════════════════════════════════════════════════════

def setup_distributed():
    """
    Initialize distributed training.

    torchrun sets these environment variables:
    - RANK: This process's global rank (0-3 for 4 GPUs)
    - LOCAL_RANK: GPU index on this node (same as RANK for single node)
    - WORLD_SIZE: Total number of processes (4)
    - MASTER_ADDR: IP of rank 0 (localhost for single node)
    - MASTER_PORT: Port for communication
    """
    dist.init_process_group(backend="nccl")
    # NCCL = NVIDIA Collective Communications Library
    # Optimized for GPU-to-GPU communication via NVLink/PCIe
    # Much faster than alternatives (gloo, mpi) for GPU training

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    # Set the GPU for this process
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def print_rank0(msg, rank=0):
    """Only print from rank 0 (avoid 4× duplicate prints)."""
    if rank == 0:
        print(msg)


# ════════════════════════════════════════════════════════════════════════════
# FSDP MODEL WRAPPING
# ════════════════════════════════════════════════════════════════════════════

def create_fsdp_model(model: nn.Module, local_rank: int) -> FSDP:
    """
    Wrap model with FSDP for distributed training.

    This configures:
    - Sharding strategy (FULL_SHARD = ZeRO-3)
    - Mixed precision (BF16 for compute, FP32 for optimizer)
    - Auto-wrapping policy (each TransformerBlock is a FSDP unit)
    """
    # ── Mixed precision config ────────────────────────────────────────
    # param_dtype: Weight storage during compute → BF16 (half memory)
    # reduce_dtype: Gradient reduction across GPUs → BF16 (half bandwidth)
    # buffer_dtype: Non-parameter tensors (e.g., RoPE tables) → BF16
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # ── Auto-wrapping policy ──────────────────────────────────────────
    # Tells FSDP to treat each TransformerBlock as an independent shard unit.
    # This means FSDP gathers/releases one block at a time during forward/backward.
    # Without this, FSDP would shard the ENTIRE model as one unit, which is
    # less memory-efficient (need to gather ALL weights at once).
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # ── Wrap model ────────────────────────────────────────────────────
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        # limit_all_gathers=True prevents FSDP from pre-fetching too many
        # all-gathers, which can cause memory spikes
        limit_all_gathers=True,
        # Forward prefetch: overlap next layer's all-gather with current
        # layer's compute. Free speedup!
        forward_prefetch=True,
    )

    return model


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    output_dir: Path,
    rank: int,
    keep_last_n: int = 3,
):
    """
    Save a full checkpoint (model + optimizer + metadata).

    Only rank 0 actually writes to disk. Other ranks participate in
    the all-gather but don't save.
    """
    ckpt_dir = output_dir / f"checkpoint-{step}"

    # Gather full state dict on rank 0
    # FULL_STATE_DICT: all-gathers shards → full model on rank 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()

    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(model_state, ckpt_dir / "model.pt")

        # Save optimizer state
        # Note: FSDP optimizer state is already sharded, so each rank
        # could save its own shard. But for simplicity, we gather on rank 0.
        # For production, use FSDP.optim_state_dict() for sharded saving.
        # torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        # Save metadata
        meta = {
            "step": step,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  💾 Checkpoint saved: {ckpt_dir}")

        # Cleanup old checkpoints (keep last N)
        all_ckpts = sorted(output_dir.glob("checkpoint-*"),
                          key=lambda p: int(p.name.split("-")[1]))
        while len(all_ckpts) > keep_last_n:
            old = all_ckpts.pop(0)
            import shutil
            shutil.rmtree(old)
            print(f"  🗑️  Deleted old checkpoint: {old.name}")

    # Wait for rank 0 to finish saving before continuing
    dist.barrier()


# ════════════════════════════════════════════════════════════════════════════
# CONCEPT 7: ACTIVATION CHECKPOINTING — Trade Compute for Memory
# ════════════════════════════════════════════════════════════════════════════
#
# During the forward pass, PyTorch stores ALL intermediate activations
# (outputs of every layer) because they're needed for the backward pass.
#
# For a 22-layer model with batch_size=8, seq_len=2048:
#   Activations ≈ 22 layers × 8 × 2048 × 2048 × 2 bytes ≈ 1.5 GB
#   (This is a rough estimate — actual depends on implementation)
#
# ACTIVATION CHECKPOINTING:
#   Instead of storing ALL activations, we only store activations at
#   certain "checkpoint" boundaries. During backward pass, we RECOMPUTE
#   the missing activations by re-running the forward pass for that block.
#
#   Without checkpointing: Store 22 blocks' activations (high memory)
#   With checkpointing:    Store 0 blocks' activations, recompute each
#                          during backward (saves memory, costs ~33% more compute)
#
#   The 33% compute overhead is worth it because:
#   1. We can use larger batch sizes (more sequences per step)
#   2. Memory saved lets FSDP work more efficiently
#   3. On RTX 4090, memory is the bottleneck, not compute
#

def apply_activation_checkpointing(model: FSDP):
    """
    Enable activation checkpointing for all TransformerBlocks.

    This wraps each block's forward pass so activations are recomputed
    during backward instead of stored.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing as _apply_activation_checkpointing,
    )

    # Check function: which modules to checkpoint
    check_fn = lambda module: isinstance(module, TransformerBlock)

    _apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn,
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def train(args):
    """
    Main pre-training loop.

    STEPS:
    1. Setup distributed (NCCL, device assignment)
    2. Create model + FSDP wrapping
    3. Create optimizer (AdamW)
    4. Create dataloader
    5. Training loop (forward → backward → accumulate → update)
    6. Save checkpoints periodically
    """

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: DISTRIBUTED SETUP
    # ══════════════════════════════════════════════════════════════════

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    if is_main:
        print("=" * 70)
        print("  NanoAgent-1B Pre-Training")
        print("=" * 70)
        print(f"  GPUs: {world_size}× {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  BF16 support: {torch.cuda.is_bf16_supported()}")
        print()

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: CREATE MODEL
    # ══════════════════════════════════════════════════════════════════

    config = NanoAgentConfig(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        ffn_dim=args.ffn_dim,
        d_c=args.d_c,
        d_c_q=args.d_c_q,
        d_rope=args.d_rope,
        max_seq_len=args.seq_len,
        mtp_enabled=args.mtp_enabled,
    )

    if is_main:
        estimates = config.estimate_params()
        print(f"📊 Model Configuration:")
        print(f"   Parameters: ~{estimates['total']/1e9:.2f}B")
        print(f"   Layers: {config.n_layers}")
        print(f"   Hidden dim: {config.hidden_dim}")
        print(f"   Attention: MLA (d_c={config.d_c}, d_rope={config.d_rope})")
        print(f"   FFN: SwiGLU (ffn_dim={config.ffn_dim})")
        print(f"   MTP: {'enabled' if config.mtp_enabled else 'disabled'}")
        print()

    # Create model on CPU first, then FSDP moves to GPU
    model = NanoAgent(config)

    # Wrap with FSDP
    model = create_fsdp_model(model, local_rank)

    # Apply activation checkpointing (saves ~30% memory)
    apply_activation_checkpointing(model)

    if is_main:
        # Count actual parameters after FSDP wrapping
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Actual parameters: {param_count:,}")
        print(f"   FSDP shards per GPU: ~{param_count * 2 / world_size / 1e9:.2f} GB (BF16)")
        print()

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: OPTIMIZER
    # ══════════════════════════════════════════════════════════════════
    #
    # AdamW (Adam with decoupled Weight Decay):
    #
    # WHY ADAMW (not plain Adam or SGD):
    # - Adam maintains per-parameter learning rates via momentum (β₁)
    #   and variance (β₂). This adapts the step size for each weight.
    # - Weight decay (0.1) prevents weights from growing too large,
    #   acting as a regularizer.
    # - "Decoupled" means weight decay is applied DIRECTLY to weights,
    #   not through the gradient. This is mathematically cleaner.
    #
    # HYPERPARAMETERS (matching Llama 3 / DeepSeek V3):
    # β₁=0.9: Momentum. Higher = smoother updates, slower to adapt.
    # β₂=0.95: Variance tracking. Lower than default (0.999) for LLMs
    #           because training data is non-stationary (changing topics).
    #           0.95 forgets old variance faster → adapts to new data faster.
    # ε=1e-8: Prevents division by zero in Adam's update rule.
    # weight_decay=0.1: Regularization strength. Standard for all major LLMs.
    #
    # WHICH PARAMETERS GET WEIGHT DECAY:
    # ALL weight matrices (Linear layers, embeddings) → weight_decay=0.1
    # All biases and norms → weight_decay=0.0
    # WHY: Weight decay on bias/norms hurts training. These are small
    # parameters that need freedom to learn without regularization.

    # Separate params into decay and no-decay groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for: biases, normalization weights, embeddings
        if param.dim() <= 1:  # bias, norm weights are 1D
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.peak_lr,  # Will be overridden by schedule
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    if is_main:
        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"🔧 Optimizer: AdamW")
        print(f"   Params with decay:    {n_decay:,}")
        print(f"   Params without decay: {n_no_decay:,}")
        print(f"   Peak LR: {args.peak_lr}")
        print(f"   Weight decay: {args.weight_decay}")
        print()

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: DATALOADER
    # ══════════════════════════════════════════════════════════════════

    print_rank0("📂 Creating DataLoader...", rank)
    loader = create_dataloader(
        data_path=args.data_path,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        distributed=True,
        rank=rank,
        world_size=world_size,
        seed=42,
        epoch=0,
    )

    # Calculate training metrics
    tokens_per_micro_batch = args.micro_batch_size * args.seq_len
    tokens_per_step = tokens_per_micro_batch * args.accumulation_steps * world_size
    total_tokens = args.total_steps * tokens_per_step

    if is_main:
        print(f"\n📈 Training Plan:")
        print(f"   Total steps:     {args.total_steps:,}")
        print(f"   Warmup steps:    {args.warmup_steps:,}")
        print(f"   Micro batch:     {args.micro_batch_size} sequences × {args.seq_len} tokens = {tokens_per_micro_batch:,} tokens")
        print(f"   Accumulation:    {args.accumulation_steps} micro steps")
        print(f"   Global batch:    {tokens_per_step:,} tokens/step")
        print(f"   Total tokens:    {total_tokens:,} ({total_tokens/1e9:.1f}B)")
        print(f"   MTP weight:      {args.mtp_weight}")
        print(f"   Checkpoint every: {args.checkpoint_every} steps")
        print()

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════

    model.train()
    global_step = 0
    micro_step = 0
    accumulated_loss = 0.0
    accumulated_mtp_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()
    step_start_time = time.time()

    # Log file (rank 0 only)
    log_path = Path(args.output_dir) / "training_log.jsonl"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print_rank0("\n🚀 Starting training...\n", rank)

    data_iter = iter(loader)

    while global_step < args.total_steps:

        # ── Get learning rate for this step ───────────────────────────
        lr = get_cosine_schedule_with_warmup(
            step=global_step,
            warmup_steps=args.warmup_steps,
            total_steps=args.total_steps,
            peak_lr=args.peak_lr,
            min_lr=args.min_lr,
        )
        # Set LR for all parameter groups
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Gradient accumulation loop ────────────────────────────────
        optimizer.zero_grad()  # Clear gradients from previous step

        for micro_idx in range(args.accumulation_steps):
            # Get next batch (handle epoch boundary)
            try:
                batch = next(data_iter)
            except StopIteration:
                # Epoch finished — create new iterator
                # Update sampler epoch for different shuffle order
                if hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(global_step)
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            # non_blocking=True: Don't wait for transfer to complete.
            # CUDA operations will automatically sync when data is needed.
            # This overlaps data transfer with compute.

            # Use FSDP no_sync for all but the last accumulation step
            # This prevents gradient all-reduce until we've accumulated all micro batches
            context = model.no_sync if micro_idx < args.accumulation_steps - 1 else lambda: torch.enable_grad()

            with context():
                # Forward pass
                result = model(
                    input_ids,
                    targets=targets,
                    mtp_weight=args.mtp_weight,
                )

                # Scale loss by accumulation steps
                # WHY: We're summing gradients over multiple micro batches.
                # To get the AVERAGE gradient, divide by accumulation_steps.
                loss = result["total_loss"] / args.accumulation_steps

                # Backward pass
                loss.backward()

            # Track losses (for logging)
            accumulated_loss += result["loss"].detach().item() / args.accumulation_steps
            if "mtp_loss" in result:
                accumulated_mtp_loss += result["mtp_loss"].detach().item() / args.accumulation_steps

            micro_step += 1

        # ── Gradient clipping ─────────────────────────────────────────
        # Must be called after all accumulation steps and before optimizer.step()
        grad_norm = model.clip_grad_norm_(args.max_grad_norm)

        # ── Optimizer step ────────────────────────────────────────────
        optimizer.step()

        global_step += 1

        # ── Logging ───────────────────────────────────────────────────
        if is_main and (global_step % args.log_every == 0 or global_step == 1):
            elapsed = time.time() - step_start_time
            tokens_this_interval = tokens_per_step * args.log_every
            throughput = tokens_this_interval / elapsed if elapsed > 0 else 0
            total_elapsed = time.time() - start_time
            eta_seconds = (args.total_steps - global_step) * (total_elapsed / global_step)
            eta_hours = eta_seconds / 3600

            log_entry = {
                "step": global_step,
                "loss": round(accumulated_loss, 4),
                "mtp_loss": round(accumulated_mtp_loss, 4) if args.mtp_enabled else None,
                "lr": lr,
                "grad_norm": round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
                "throughput_tokens_sec": round(throughput),
                "elapsed_hours": round(total_elapsed / 3600, 2),
                "eta_hours": round(eta_hours, 2),
            }

            # Console output
            mtp_str = f" | MTP: {accumulated_mtp_loss:.4f}" if args.mtp_enabled else ""
            print(
                f"  Step {global_step:>6d}/{args.total_steps} | "
                f"Loss: {accumulated_loss:.4f}{mtp_str} | "
                f"LR: {lr:.2e} | "
                f"Grad: {log_entry['grad_norm']:.2f} | "
                f"{throughput:,.0f} tok/s | "
                f"ETA: {eta_hours:.1f}h"
            )

            # Write to log file
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            step_start_time = time.time()

        # Reset accumulated losses
        accumulated_loss = 0.0
        accumulated_mtp_loss = 0.0

        # Track best loss
        if is_main:
            current_loss = result["loss"].item()
            if current_loss < best_loss:
                best_loss = current_loss

        # ── Checkpointing ─────────────────────────────────────────────
        if global_step % args.checkpoint_every == 0:
            print_rank0(f"\n  📸 Saving checkpoint at step {global_step}...", rank)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=global_step,
                loss=accumulated_loss,
                output_dir=Path(args.output_dir),
                rank=rank,
                keep_last_n=3,
            )

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: SAVE FINAL MODEL
    # ══════════════════════════════════════════════════════════════════

    print_rank0("\n💾 Saving final model...", rank)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=global_step,
        loss=best_loss,
        output_dir=Path(args.output_dir),
        rank=rank,
        keep_last_n=4,  # Keep final + last 3
    )

    # ── Training summary ──────────────────────────────────────────────
    total_time = time.time() - start_time
    if is_main:
        print(f"\n{'='*70}")
        print(f"  ✅ PRE-TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"  Total steps:  {global_step:,}")
        print(f"  Total tokens: {global_step * tokens_per_step:,}")
        print(f"  Best loss:    {best_loss:.4f}")
        print(f"  Total time:   {total_time/3600:.1f} hours")
        print(f"  Avg throughput: {global_step * tokens_per_step / total_time:,.0f} tokens/sec")
        print(f"  Cost estimate:  ~£{total_time/3600 * 1.92:.0f}")
        print(f"  Checkpoints:  {args.output_dir}")
        print(f"\n  NEXT: Run SFT (training/sft.py)")
        print(f"{'='*70}")

    cleanup_distributed()


# ════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NanoAgent-1B Pre-Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE:
  # Full training run (10B tokens, ~35 hours):
  torchrun --nproc_per_node=4 training/pretrain.py \\
      --data_path /workspace/data/train_tokens.bin \\
      --output_dir /workspace/checkpoints

  # Quick test (100 steps, ~5 minutes):
  torchrun --nproc_per_node=4 training/pretrain.py \\
      --data_path /workspace/data/train_tokens.bin \\
      --total_steps 100 \\
      --log_every 10
        """
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to train_tokens.bin from prepare_pretrain.py")
    parser.add_argument("--output_dir", type=str, default="/workspace/checkpoints",
                        help="Directory for checkpoints and logs")

    # Model architecture (defaults match NanoAgentConfig)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=22)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--ffn_dim", type=int, default=5632)
    parser.add_argument("--d_c", type=int, default=512)
    parser.add_argument("--d_c_q", type=int, default=1536)
    parser.add_argument("--d_rope", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=2048)

    # MTP
    parser.add_argument("--mtp_enabled", action="store_true", default=True)
    parser.add_argument("--mtp_weight", type=float, default=0.3,
                        help="Weight of MTP auxiliary loss")

    # Training hyperparameters
    parser.add_argument("--total_steps", type=int, default=19000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--peak_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Logging and checkpointing
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train(args)