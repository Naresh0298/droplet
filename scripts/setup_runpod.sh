#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                NanoAgent-1B: RunPod Setup Script                        ║
# ║                                                                         ║
# ║  Run this ONCE when you first start a RunPod GPU pod.                   ║
# ║  It installs everything needed for training.                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# USAGE:
#   Phase 1 (Data Prep — CPU Pod, $0.012/hr):
#     1. Create a RunPod CPU pod
#     2. Attach a Network Volume (100GB, "nanoagent-vol")
#     3. Clone your repo or upload files
#     4. Run: bash scripts/setup_runpod.sh cpu
#     5. Run: python data/prepare_pretrain.py --total_tokens 10_000_000_000
#
#   Phase 1 (Pre-Training — 4× RTX 4090 Pod, $1.60/hr):
#     1. Create a RunPod GPU pod (4× RTX 4090)
#     2. Attach SAME network volume ("nanoagent-vol")
#     3. Clone your repo or upload files
#     4. Run: bash scripts/setup_runpod.sh gpu
#     5. Run: bash scripts/launch_pretrain.sh
#
# NETWORK VOLUME:
#   RunPod Network Volumes persist across pods. This is critical!
#   - Data prep writes train_tokens.bin to the volume
#   - GPU pod reads from the SAME volume
#   - Checkpoints are saved TO the volume (survive pod crashes)
#   - Cost: ~$7/month for 100GB
#
#   Path: /workspace/ is the network volume mount point on RunPod

set -e  # Exit on any error

MODE=${1:-gpu}  # "cpu" or "gpu"

echo "════════════════════════════════════════════════════════════"
echo "  NanoAgent-1B Setup — Mode: $MODE"
echo "════════════════════════════════════════════════════════════"

# ── Directory structure ───────────────────────────────────────────────────
echo ""
echo "📁 Creating directory structure on network volume..."
mkdir -p /workspace/data          # Tokenized training data
mkdir -p /workspace/checkpoints   # Model checkpoints
mkdir -p /workspace/logs          # Training logs
mkdir -p /workspace/models        # Final saved models

# ── System packages ───────────────────────────────────────────────────────
echo ""
echo "📦 Installing system packages..."
apt-get update -qq
apt-get install -y -qq htop nvtop tmux wget git > /dev/null 2>&1
# htop: Monitor CPU/RAM usage
# nvtop: Monitor GPU usage (like htop but for GPUs)
# tmux: Keep training running after SSH disconnect
# wget: Download files
# git: Version control

# ── Python packages ───────────────────────────────────────────────────────
echo ""
echo "🐍 Installing Python packages..."

if [ "$MODE" = "cpu" ]; then
    # CPU pod: Only needs tokenizer and data processing
    pip install -q \
        sentencepiece \
        datasets \
        numpy \
        tqdm
    echo "  ✅ CPU packages installed (sentencepiece, datasets, numpy)"

elif [ "$MODE" = "gpu" ]; then
    # GPU pod: Full training stack
    # PyTorch should already be installed on RunPod GPU pods
    pip install -q \
        sentencepiece \
        datasets \
        numpy \
        tqdm \
        wandb \
        tensorboard
    echo "  ✅ GPU packages installed"

    # Verify CUDA
    echo ""
    echo "🔍 Checking GPU setup..."
    python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.version.cuda}')
print(f'  GPUs:     {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}:    {props.name} ({props.total_mem / 1e9:.0f} GB)')
print(f'  BF16:     {torch.cuda.is_bf16_supported()}')
print(f'  NCCL:     {torch.distributed.is_nccl_available()}')
"
fi

# ── Download Llama 2 tokenizer ────────────────────────────────────────────
echo ""
echo "📥 Setting up Llama 2 tokenizer..."
TOKENIZER_DIR="/workspace/tokenizer"
if [ -f "$TOKENIZER_DIR/tokenizer.model" ]; then
    echo "  ✅ Tokenizer already exists at $TOKENIZER_DIR"
else
    mkdir -p "$TOKENIZER_DIR"
    # The tokenizer.model file is included in the repo
    # If not, download from HuggingFace:
    echo "  ⚠️  Tokenizer not found. You need to provide tokenizer.model"
    echo "     Option 1: Copy from your local machine"
    echo "     Option 2: Download from HuggingFace (requires auth):"
    echo "       huggingface-cli login"
    echo "       python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf'); t.save_pretrained('$TOKENIZER_DIR')\""
fi

# ── Verify network volume ────────────────────────────────────────────────
echo ""
echo "💾 Network volume status:"
df -h /workspace 2>/dev/null || echo "  ⚠️  /workspace not mounted (no network volume?)"
echo "  Contents:"
ls -la /workspace/ 2>/dev/null || echo "  (empty)"

# ── Print next steps ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo "════════════════════════════════════════════════════════════"

if [ "$MODE" = "cpu" ]; then
    echo ""
    echo "  NEXT STEPS (Data Preparation):"
    echo "  ─────────────────────────────────"
    echo ""
    echo "  1. Make sure tokenizer is at /workspace/tokenizer/tokenizer.model"
    echo ""
    echo "  2. Test run (2 minutes, verifies everything works):"
    echo "     python data/prepare_pretrain.py \\"
    echo "       --total_tokens 100_000_000 \\"
    echo "       --output_dir /workspace/data \\"
    echo "       --tokenizer_path /workspace/tokenizer/tokenizer.model"
    echo ""
    echo "  3. Full run (2-3 hours, creates 20GB training data):"
    echo "     python data/prepare_pretrain.py \\"
    echo "       --total_tokens 10_000_000_000 \\"
    echo "       --output_dir /workspace/data \\"
    echo "       --tokenizer_path /workspace/tokenizer/tokenizer.model"
    echo ""
    echo "  4. When done, verify:"
    echo "     ls -lh /workspace/data/train_tokens.bin"
    echo "     # Should be ~20GB"
    echo ""
    echo "  5. Shut down CPU pod (data persists on network volume)"
    echo ""
    echo "  ESTIMATED COST: ~£0.30"

elif [ "$MODE" = "gpu" ]; then
    echo ""
    echo "  NEXT STEPS (Pre-Training):"
    echo "  ─────────────────────────────"
    echo ""
    echo "  1. Verify data exists:"
    echo "     ls -lh /workspace/data/train_tokens.bin"
    echo ""
    echo "  2. Quick test (5 minutes, 100 steps):"
    echo "     bash scripts/launch_pretrain.sh --test"
    echo ""
    echo "  3. Full training (35 hours):"
    echo "     tmux new -s train"
    echo "     bash scripts/launch_pretrain.sh"
    echo "     # Ctrl+B, D to detach (training continues in background)"
    echo "     # tmux attach -t train  to reconnect"
    echo ""
    echo "  4. Monitor:"
    echo "     nvtop                    # GPU utilization"
    echo "     tail -f /workspace/logs/training_log.jsonl"
    echo ""
    echo "  ESTIMATED COST: ~£55 (35 hours × £1.60/hr)"
fi
echo ""