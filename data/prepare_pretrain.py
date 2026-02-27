"""
droplet-1B: Pre-training Data Preparation Pipeline

Streams text from 6 HuggingFace datasets, tokenizes with Llama 2 tokenizer,
packs into fixed-length sequences, and saves as a memory-mapped binary file.

USAGE:
    # Test run (2 minutes)
    python prepare_pretrain.py --total_tokens 100_000_000 --output_dir /workspace/data

    # Full run (2-3 hours)
    python prepare_pretrain.py --total_tokens 10_000_000_000 --output_dir /workspace/data
"""

import os
import argparse
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


# ── Dataset Configuration ────────────────────────────────────────────

DATASET_CONFIGS = [
    {
        "name": "FineWeb-Edu",
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_name": "sample-10BT",
        "hf_kwargs": {},
        "text_field": "text",
        "weight": 0.50,
    },
    {
        "name": "StarCoder-Python",
        "hf_path": "bigcode/starcoderdata",
        "hf_name": None,
        "hf_kwargs": {"data_dir": "python"},
        "text_field": "content",
        "weight": 0.12,
    },
    {
        "name": "StarCoder-JavaScript",
        "hf_path": "bigcode/starcoderdata",
        "hf_name": None,
        "hf_kwargs": {"data_dir": "javascript"},
        "text_field": "content",
        "weight": 0.08,
    },
    {
        "name": "Wikipedia-EN",
        "hf_path": "wikimedia/wikipedia",
        "hf_name": "20231101.en",
        "hf_kwargs": {},
        "text_field": "text",
        "weight": 0.15,
    },
    {
        "name": "UltraData-Math",
        "hf_path": "openbmb/UltraData-Math",
        "hf_name": "UltraData-Math-L1",
        "hf_kwargs": {},
        "text_field": "content",
        "weight": 0.10,
    },
    {
        "name": "OpenWebMath",
        "hf_path": "open-web-math/open-web-math",
        "hf_name": None,
        "hf_kwargs": {},
        "text_field": "text",
        "weight": 0.05,
    },
]


# ── TokenBuffer: Packs tokens into fixed-length chunks ───────────────

class TokenBuffer:
    """
    Accumulates tokens from multiple documents and yields fixed-length chunks.
    Adds EOS token between documents so the model learns document boundaries.
    """

    def __init__(self, seq_len, eos_token_id):
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.buffer = []
        self.chunks_produced = 0

    def add_document(self, token_ids):
        """Add a tokenized document to the buffer with EOS separator."""
        self.buffer.extend(token_ids)
        self.buffer.append(self.eos_token_id)

    def get_chunks(self):
        """Extract all complete fixed-length chunks from the buffer."""
        chunks = []
        while len(self.buffer) >= self.seq_len:
            chunk = self.buffer[:self.seq_len]
            self.buffer = self.buffer[self.seq_len:]
            chunks.append(np.array(chunk, dtype=np.uint16))
            self.chunks_produced += 1
        return chunks

    @property
    def pending_tokens(self):
        return len(self.buffer)


# ── DatasetProcessor: Handles one dataset stream ─────────────────────

class DatasetProcessor:
    """
    Processes a single streaming dataset: loads it, tokenizes text,
    and produces token chunks through a TokenBuffer.
    """

    def __init__(self, config, tokenizer, seq_len):
        self.config = config
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer = TokenBuffer(seq_len, tokenizer.eos_token_id)
        self.docs_processed = 0
        self.tokens_produced = 0

    def stream_dataset(self):
        """Open a streaming connection to the HuggingFace dataset."""
        load_kwargs = {
            "path": self.config["hf_path"],
            "split": "train",
            "streaming": True,
        }
        if self.config["hf_name"]:
            load_kwargs["name"] = self.config["hf_name"]
        load_kwargs.update(self.config["hf_kwargs"])
        return load_dataset(**load_kwargs)

    def tokenize_text(self, text):
        """Convert raw text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def process_documents(self, target_tokens):
        """
        Process documents until we've produced enough token chunks.
        Streams data, tokenizes, packs into chunks, stops at target.
        """
        all_chunks = []
        target_chunks = target_tokens // self.seq_len
        dataset = self.stream_dataset()

        print(f"\n  Streaming {self.config['name']}...")
        print(f"  Target: {target_tokens:,} tokens ({target_chunks:,} chunks of {self.seq_len})")

        pbar = tqdm(total=target_chunks, desc=f"  {self.config['name']}", unit="chunks")

        for example in dataset:
            text = example.get(self.config["text_field"], "")

            # Skip empty or very short documents
            if not text or len(text) < 50:
                continue

            # Truncate extremely long documents
            if len(text) > 100_000:
                text = text[:100_000]

            token_ids = self.tokenize_text(text)

            # Skip if tokenization produced very few tokens
            if len(token_ids) < 10:
                continue

            # Add to buffer and extract any complete chunks
            self.buffer.add_document(token_ids)
            chunks = self.buffer.get_chunks()

            if chunks:
                all_chunks.extend(chunks)
                self.docs_processed += 1
                self.tokens_produced += sum(len(c) for c in chunks)
                pbar.update(len(chunks))

            # Check if we've reached our target
            if len(all_chunks) >= target_chunks:
                break

        pbar.close()

        # Trim to exactly the target number of chunks
        all_chunks = all_chunks[:target_chunks]

        print(f"  Produced {len(all_chunks):,} chunks "
              f"({len(all_chunks) * self.seq_len:,} tokens) "
              f"from {self.docs_processed:,} documents")

        return all_chunks


# ── Main Pipeline ────────────────────────────────────────────────────

def prepare_data(args):
    """
    Main data preparation pipeline.
    1. Initialize tokenizer
    2. Calculate token targets per dataset
    3. Process each dataset (stream -> tokenize -> pack)
    4. Shuffle chunks
    5. Write to memory-mapped binary file
    6. Save metadata and verify
    """

    print("=" * 70)
    print("  droplet-1B Data Preparation Pipeline")
    print("=" * 70)

    # Step 1: Initialize tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    assert vocab_size <= 65535, (
        f"Vocab size {vocab_size} exceeds uint16 max (65535). "
        f"Use int32 dtype instead."
    )
    print(f"  Vocab fits in uint16 (max 65535)")

    # Step 2: Calculate token targets
    print(f"\nToken allocation ({args.total_tokens:,} total):")
    for config in DATASET_CONFIGS:
        config["target_tokens"] = int(args.total_tokens * config["weight"])
        print(f"  {config['name']:25s}: {config['target_tokens']:>15,} tokens ({config['weight']*100:.0f}%)")

    total_weight = sum(c["weight"] for c in DATASET_CONFIGS)
    assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, not 1.0"

    # Step 3: Process each dataset
    print(f"\n{'='*70}")
    print(f"  Processing {len(DATASET_CONFIGS)} datasets...")
    print(f"{'='*70}")

    all_chunks = []
    dataset_stats = {}
    start_time = time.time()

    for config in DATASET_CONFIGS:
        ds_start = time.time()
        processor = DatasetProcessor(config, tokenizer, args.seq_len)
        chunks = processor.process_documents(config["target_tokens"])
        all_chunks.extend(chunks)

        ds_time = time.time() - ds_start
        dataset_stats[config["name"]] = {
            "chunks": len(chunks),
            "tokens": len(chunks) * args.seq_len,
            "docs": processor.docs_processed,
            "time_seconds": ds_time,
        }

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time/60:.1f} minutes")

    # Step 4: Shuffle chunks
    print(f"\nShuffling {len(all_chunks):,} chunks...")
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(all_chunks))
    all_chunks = [all_chunks[i] for i in shuffle_indices]
    print(f"  Shuffled with seed=42 (reproducible)")

    # Step 5: Write memory-mapped binary file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "train_tokens.bin"
    total_tokens_actual = len(all_chunks) * args.seq_len

    print(f"\nWriting {total_tokens_actual:,} tokens to {output_file}")
    print(f"  File size: ~{total_tokens_actual * 2 / 1e9:.1f} GB")

    fp = np.memmap(
        str(output_file),
        dtype=np.uint16,
        mode='w+',
        shape=(total_tokens_actual,)
    )

    offset = 0
    for chunk in tqdm(all_chunks, desc="  Writing"):
        fp[offset:offset + args.seq_len] = chunk
        offset += args.seq_len

    fp.flush()
    del fp

    print(f"  Written to {output_file}")

    # Step 6: Save metadata
    metadata = {
        "total_tokens": total_tokens_actual,
        "total_chunks": len(all_chunks),
        "seq_len": args.seq_len,
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "tokenizer_name": args.tokenizer_name,
        "datasets": dataset_stats,
        "file_size_bytes": total_tokens_actual * 2,
        "file_size_gb": total_tokens_actual * 2 / 1e9,
        "shuffle_seed": 42,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_file = output_dir / "data_config.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved to {metadata_file}")

    # Step 7: Verification
    print(f"\nVerifying output file...")
    verify = np.memmap(str(output_file), dtype=np.uint16, mode='r')

    assert len(verify) == total_tokens_actual, \
        f"Size mismatch: {len(verify)} != {total_tokens_actual}"

    max_token = verify.max()
    assert max_token < vocab_size, \
        f"Found token ID {max_token} >= vocab size {vocab_size}"

    print(f"  File size: {len(verify):,} tokens")
    print(f"  Max token ID: {max_token} (vocab: {vocab_size})")

    sample_start = np.random.randint(0, len(verify) - 200)
    sample_tokens = verify[sample_start:sample_start + 50].astype(np.int64)
    sample_text = tokenizer.decode(sample_tokens)
    print(f"\n  Random sample (tokens {sample_start}-{sample_start+50}):")
    print(f'  "{sample_text[:200]}..."')

    del verify

    # Final summary
    print(f"\n{'='*70}")
    print(f"  DATA PREPARATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Output file:     {output_file}")
    print(f"  Total tokens:    {total_tokens_actual:,}")
    print(f"  Total chunks:    {len(all_chunks):,}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  File size:       {total_tokens_actual * 2 / 1e9:.1f} GB")
    print(f"  Time taken:      {total_time/60:.1f} minutes")
    print(f"")
    print(f"  Dataset breakdown:")
    for name, stats in dataset_stats.items():
        print(f"    {name:25s}: {stats['tokens']:>12,} tokens from {stats['docs']:,} docs")
    print(f"")
    print(f"  NEXT STEP:")
    print(f"  1. Stop this CPU pod")
    print(f"  2. Launch 4x RTX 4090 GPU pod with same network volume")
    print(f"  3. Run: bash scripts/launch_pretrain.sh")
    print(f"{'='*70}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="droplet-1B: Pre-training Data Preparation",
    )

    parser.add_argument(
        "--output_dir", type=str, default="/workspace/data",
        help="Directory to save tokenized data (default: /workspace/data)"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="NousResearch/Llama-2-7b-hf",
        help="HuggingFace tokenizer name (default: NousResearch/Llama-2-7b-hf)"
    )
    parser.add_argument(
        "--total_tokens", type=int, default=10_000_000_000,
        help="Total tokens to prepare (default: 10B)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2048,
        help="Sequence length for packed chunks (default: 2048)"
    )

    args = parser.parse_args()

    assert args.total_tokens > 0, "total_tokens must be positive"
    assert args.seq_len > 0, "seq_len must be positive"
    assert args.seq_len <= 8192, "seq_len > 8192 is unusual for pre-training"

    prepare_data(args)