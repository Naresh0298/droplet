
set -e

# ── Configuration ─────────────────────────────────────────────────────────
DATA_PATH="/workspace/data/train_tokens.bin"
OUTPUT_DIR="/workspace/checkpoints"
LOG_DIR="/workspace/logs"
NUM_GPUS=4

# Check if test mode
if [ "$1" = "--test" ]; then
    echo "🧪 TEST MODE — 100 steps only"
    TOTAL_STEPS=100
    LOG_EVERY=10
    CHECKPOINT_EVERY=50
    WARMUP_STEPS=20
else
    echo "🚀 FULL TRAINING MODE — 19,000 steps"
    TOTAL_STEPS=19000
    LOG_EVERY=10
    CHECKPOINT_EVERY=1000
    WARMUP_STEPS=2000
fi

# ── Pre-flight checks ────────────────────────────────────────────────────
echo ""
echo "🔍 Pre-flight checks..."

# Check data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "  ❌ Data file not found: $DATA_PATH"
    echo "     Run data preparation first:"
    echo "     python data/prepare_pretrain.py --total_tokens 10_000_000_000"
    exit 1
fi

DATA_SIZE=$(du -h "$DATA_PATH" | cut -f1)
echo "  ✅ Data file: $DATA_PATH ($DATA_SIZE)"

# Check GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "  ✅ GPUs available: $GPU_COUNT"

if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "  ⚠️  Expected $NUM_GPUS GPUs, found $GPU_COUNT. Adjusting..."
    NUM_GPUS=$GPU_COUNT
fi

# Check NCCL
python -c "import torch; assert torch.distributed.is_nccl_available(), 'NCCL not available!'" 2>/dev/null
echo "  ✅ NCCL available"

# Create output dirs
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── Print training plan ──────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Training Configuration"
echo "════════════════════════════════════════════════════════════"
echo "  Data:           $DATA_PATH ($DATA_SIZE)"
echo "  GPUs:           $NUM_GPUS× $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
echo "  Total steps:    $TOTAL_STEPS"
echo "  Warmup:         $WARMUP_STEPS steps"
echo "  Micro batch:    8 sequences × 2048 tokens"
echo "  Accumulation:   8 steps"
echo "  Global batch:   $(( 8 * 8 * NUM_GPUS * 2048 )) tokens/step"
echo "  Checkpoints:    $OUTPUT_DIR (every $CHECKPOINT_EVERY steps)"
echo "  Logs:           $LOG_DIR"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check for existing checkpoints to resume from
RESUME_STEP=""
LATEST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    RESUME_STEP=$(basename "$LATEST_CKPT" | sed 's/checkpoint-//')
    echo "  📸 Found checkpoint: $LATEST_CKPT (step $RESUME_STEP)"
    echo "  Resuming from step $RESUME_STEP..."
    echo ""
fi

# ── Launch training ──────────────────────────────────────────────────────
echo "🚀 Launching training with torchrun ($NUM_GPUS GPUs)..."
echo ""

# NCCL environment variables for optimal performance
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0            # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=2         # GPU Direct RDMA level
export OMP_NUM_THREADS=4            # OpenMP threads per process
export TOKENIZERS_PARALLELISM=false # Avoid HuggingFace tokenizer warnings

# torchrun handles distributed process spawning:
# - Starts NUM_GPUS copies of the script
# - Sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
# - Each process gets assigned to a different GPU
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    training/pretrain.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --total_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --log_every $LOG_EVERY \
    --checkpoint_every $CHECKPOINT_EVERY \
    --micro_batch_size 8 \
    --accumulation_steps 8 \
    --seq_len 2048 \
    --peak_lr 3e-4 \
    --min_lr 3e-5 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --mtp_enabled \
    --mtp_weight 0.3 \
    --num_workers 4 \
    2>&1 | tee "$LOG_DIR/pretrain_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Training complete!"
echo "  Checkpoints: $OUTPUT_DIR"
echo "  Logs: $LOG_DIR"
echo "════════════════════════════════════════════════════════════"