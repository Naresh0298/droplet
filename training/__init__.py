"""
NanoAgent-1B Training Pipeline.

- data_loader.py: Memory-mapped dataset + distributed DataLoader
- pretrain.py: Main pre-training loop with FSDP on 4× RTX 4090
"""
# Step 1: CPU pod → prepare 10B tokens (~2 hrs, £0.30)
bash scripts/setup_runpod.sh cpu
python data/prepare_pretrain.py --total_tokens 10_000_000_000

# Step 2: GPU pod (4× RTX 4090) → train
bash scripts/setup_runpod.sh gpu
bash scripts/launch_pretrain.sh --test    # 5 min sanity check first!
bash scripts/launch_pretrain.sh           # full 35hr run (in tmux!)