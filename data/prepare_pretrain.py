"""
droplet-1B: Pre-training Data Preparation Pipeline (Memory-Efficient)

Streams text from 6 HuggingFace datasets, tokenizes with Llama 2 tokenizer,
packs into fixed-length sequences, and writes directly to a memory-mapped
binary file. Never holds more than ~200MB in RAM.

USAGE:
    python prepare_pretrain.py --total_tokens 100_000_000 --output_dir /workspace/data
    python prepare_pretrain.py --total_tokens 10_000_000_000 --output_dir /workspace/data
"""

import os
import argparse
import json
import time
import gc
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
    def __init__(self, seq_len, eos_token_id):
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.buffer = []
        self.chunks_produced = 0

    def add_document(self, token_ids):
        self.buffer.extend(token_ids)
        self.buffer.append(self.eos_token_id)

    def get_chunks(self):
        chunks = []
        while len(self.buffer) >= self.seq_len:
            chunk = self.buffer[:self.seq_len]
            self.buffer = self.buffer[self.seq_len:]
            chunks.append(np.array(chunk, dtype=np.uint16))
            self.chunks_produced += 1
        return chunks


# ── Process one dataset, writing directly to memmap ──────────────────

def process_dataset(config, tokenizer, seq_len, fp, start_offset):
    """
    Stream one dataset, tokenize, pack, and write chunks directly to
    the memmap file. Returns the number of chunks written.
    """
    buffer = TokenBuffer(seq_len, tokenizer.eos_token_id)
    target_chunks = config["target_chunks"]

    load_kwargs = {
        "path": config["hf_path"],
        "split": "train",
        "streaming": True,
    }
    if config["hf_name"]:
        load_kwargs["name"] = config["hf_name"]
    load_kwargs.update(config["hf_kwargs"])
    dataset = load_dataset(**load_kwargs)

    print(f"\n  Streaming {config['name']}...")
    print(f"  Target: {config['target_tokens']:,} tokens ({target_chunks:,} chunks)")

    pbar = tqdm(total=target_chunks, desc=f"  {config['name']}", unit="chunks")

    chunks_written = 0
    docs_processed = 0
    errors = 0
    offset = start_offset

    try:
        for example in dataset:
            try:
                text = example.get(config["text_field"], "")

                if not text or len(text) < 50:
                    continue
                if len(text) > 100_000:
                    text = text[:100_000]

                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if len(token_ids) < 10:
                    continue

                buffer.add_document(token_ids)
                chunks = buffer.get_chunks()

                for chunk in chunks:
                    if chunks_written >= target_chunks:
                        break
                    fp[offset:offset + seq_len] = chunk
                    offset += seq_len
                    chunks_written += 1
                    pbar.update(1)

                if chunks:
                    docs_processed += 1

                # Periodic garbage collection and flush
                if docs_processed % 10000 == 0 and docs_processed > 0:
                    fp.flush()
                    gc.collect()

                if chunks_written >= target_chunks:
                    break

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\n  Warning: Error processing doc #{docs_processed}: {e}")
                if errors > 1000:
                    print(f"\n  Too many errors ({errors}), stopping this dataset")
                    break
                continue

    except Exception as e:
        print(f"\n  Dataset stream error: {e}")
        print(f"  Got {chunks_written:,} chunks before error")

    pbar.close()
    fp.flush()

    print(f"  Produced {chunks_written:,} chunks "
          f"({chunks_written * seq_len:,} tokens) "
          f"from {docs_processed:,} documents"
          f" ({errors} errors skipped)")

    return chunks_written


# ── Main Pipeline ────────────────────────────────────────────────────

def prepare_data(args):
    print("=" * 70)
    print("  droplet-1B Data Preparation Pipeline")
    print("=" * 70)

    # Step 1: Initialize tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    assert vocab_size <= 65535, f"Vocab size {vocab_size} exceeds uint16 max"

    # Step 2: Calculate token targets
    print(f"\nToken allocation ({args.total_tokens:,} total):")
    total_chunks = 0
    for config in DATASET_CONFIGS:
        config["target_tokens"] = int(args.total_tokens * config["weight"])
        config["target_chunks"] = config["target_tokens"] // args.seq_len
        total_chunks += config["target_chunks"]
        print(f"  {config['name']:25s}: {config['target_tokens']:>15,} tokens ({config['weight']*100:.0f}%)")

    total_tokens_planned = total_chunks * args.seq_len

    # Step 3: Pre-allocate memmap file on disk
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train_tokens.bin"

    print(f"\nPre-allocating {output_file}")
    print(f"  {total_tokens_planned:,} tokens = {total_tokens_planned * 2 / 1e9:.1f} GB")

    fp = np.memmap(
        str(output_file),
        dtype=np.uint16,
        mode='w+',
        shape=(total_tokens_planned,)
    )

    # Step 4: Process each dataset, writing directly to memmap
    print(f"\n{'='*70}")
    print(f"  Processing {len(DATASET_CONFIGS)} datasets...")
    print(f"{'='*70}")

    dataset_stats = {}
    start_time = time.time()
    offset = 0

    for config in DATASET_CONFIGS:
        ds_start = time.time()

        chunks_written = process_dataset(
            config, tokenizer, args.seq_len, fp, offset
        )

        offset += chunks_written * args.seq_len
        ds_time = time.time() - ds_start

        dataset_stats[config["name"]] = {
            "chunks": chunks_written,
            "tokens": chunks_written * args.seq_len,
            "time_seconds": round(ds_time, 1),
        }

        # Force cleanup between datasets
        gc.collect()

    fp.flush()
    del fp

    total_time = time.time() - start_time
    total_tokens_actual = offset
    n_chunks = total_tokens_actual // args.seq_len
    print(f"\nTotal processing time: {total_time/60:.1f} minutes")

    # Handle case where we got fewer tokens than planned
    if total_tokens_actual < total_tokens_planned:
        print(f"\nNote: Got {total_tokens_actual:,} tokens (planned {total_tokens_planned:,})")
        print(f"  Truncating file to actual size...")
        # Rewrite the file with correct size
        old_fp = np.memmap(str(output_file), dtype=np.uint16, mode='r')
        tmp_file = output_dir / "train_tokens_tmp.bin"
        new_fp = np.memmap(str(tmp_file), dtype=np.uint16, mode='w+', shape=(total_tokens_actual,))
        # Copy in chunks to save memory
        copy_chunk = 1_000_000
        for i in range(0, total_tokens_actual, copy_chunk):
            end = min(i + copy_chunk, total_tokens_actual)
            new_fp[i:end] = old_fp[i:end]
        new_fp.flush()
        del old_fp
        del new_fp
        os.remove(str(output_file))
        os.rename(str(tmp_file), str(output_file))
        print(f"  Truncated to {total_tokens_actual:,} tokens")

    # Step 5: Shuffle on disk using Fisher-Yates
    print(f"\nShuffling {n_chunks:,} chunks on disk...")

    fp = np.memmap(str(output_file), dtype=np.uint16, mode='r+')
    seq = args.seq_len

    rng = np.random.RandomState(42)
    temp = np.empty(seq, dtype=np.uint16)

    for i in tqdm(range(n_chunks - 1, 0, -1), desc="  Shuffling", unit="swaps",
                  mininterval=2.0):
        j = rng.randint(0, i + 1)
        if i != j:
            i_start = i * seq
            j_start = j * seq
            temp[:] = fp[i_start:i_start + seq]
            fp[i_start:i_start + seq] = fp[j_start:j_start + seq]
            fp[j_start:j_start + seq] = temp

    fp.flush()
    del fp
    print(f"  Shuffled with seed=42")

    # Step 6: Save metadata
    metadata = {
        "total_tokens": total_tokens_actual,
        "total_chunks": n_chunks,
        "seq_len": args.seq_len,
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "tokenizer_name": args.tokenizer_name,
        "datasets": dataset_stats,
        "file_size_bytes": total_tokens_actual * 2,
        "file_size_gb": round(total_tokens_actual * 2 / 1e9, 2),
        "shuffle_seed": 42,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    metadata_file = output_dir / "data_config.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_file}")

    # Step 7: Verify
    print(f"\nVerifying...")
    verify = np.memmap(str(output_file), dtype=np.uint16, mode='r')
    assert len(verify) == total_tokens_actual
    max_token = int(verify.max())
    assert max_token < vocab_size, f"Token {max_token} >= vocab {vocab_size}"
    print(f"  File: {len(verify):,} tokens, max_id={max_token}")

    sample_start = np.random.randint(0, len(verify) - 200)
    sample_tokens = verify[sample_start:sample_start + 50].astype(np.int64)
    sample_text = tokenizer.decode(sample_tokens)
    print(f'  Sample: "{sample_text[:150]}..."')
    del verify

    # Final summary
    print(f"\n{'='*70}")
    print(f"  DATA PREPARATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Output file:     {output_file}")
    print(f"  Total tokens:    {total_tokens_actual:,}")
    print(f"  File size:       {total_tokens_actual * 2 / 1e9:.1f} GB")
    print(f"  Time taken:      {total_time/60:.1f} minutes")
    print()
    for name, stats in dataset_stats.items():
        print(f"    {name:25s}: {stats['tokens']:>12,} tokens ({stats['time_seconds']:.0f}s)")
    print()
    print(f"  NEXT: Stop CPU pod -> Launch 4x RTX 4090 -> bash scripts/launch_pretrain.sh")
    print(f"{'='*70}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="droplet-1B: Pre-training Data Preparation")
    parser.add_argument("--output_dir", type=str, default="/workspace/data")
    parser.add_argument("--tokenizer_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--total_tokens", type=int, default=10_000_000_000)
    parser.add_argument("--seq_len", type=int, default=2048)

    args = parser.parse_args()
    assert args.total_tokens > 0
    assert args.seq_len > 0

    prepare_data(args)