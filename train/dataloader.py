"""
FlightMind Training Data Loader
=================================
Converts cleaned JSONL documents into packed token sequences ready for training.

The key challenge: LLM training operates on fixed-length sequences (2048 tokens),
but our documents vary wildly in length (a METAR observation is ~30 tokens,
an NTSB report can be 5000+ tokens). We need to efficiently pack documents
into sequences without wasting tokens on padding.

Pipeline:
    JSONL docs -> Tokenize -> Pack into sequences -> Batch -> Training

Packing Strategy:
    Documents are concatenated with <|endoftext|> separators. When we hit
    the sequence length, we cut and start a new sequence. This means some
    sequences will contain multiple short documents, and long documents
    will span multiple sequences. No tokens are wasted on padding.

    Example (seq_len=10, simplified):
    Doc A: [1, 2, 3]  Doc B: [4, 5, 6, 7, 8]  Doc C: [9, 10, 11, 12]

    Sequence 1: [1, 2, 3, EOT, 4, 5, 6, 7, 8, EOT]   <- A + B packed
    Sequence 2: [9, 10, 11, 12, EOT, ...]              <- C starts here

    This is how GPT-2, LLaMA, and all modern LLMs handle pretraining data.

Data Mixing:
    We maintain the 70/30 general/aviation ratio by interleaving documents
    from each source pool according to their target ratio.

Usage:
    python train/dataloader.py                  # Prepare training data
    python train/dataloader.py --seq-len 2048   # Custom sequence length
"""

import argparse
import json
import random
import struct
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"

# Source classification for mixing
AVIATION_JSONL = [
    "ntsb.jsonl", "handbooks.jsonl", "regulations.jsonl", "metar.jsonl",
    "wikipedia.jsonl", "hf_datasets.jsonl", "openap.jsonl",
]
GENERAL_JSONL = [
    "fineweb.jsonl",
]


def load_tokenizer():
    """Load the trained FlightMind BPE tokenizer."""
    from tokenizers import Tokenizer

    tokenizer_path = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run scripts/train_tokenizer.py first."
        )
    return Tokenizer.from_file(str(tokenizer_path))


def tokenize_jsonl(
    jsonl_path: Path,
    tokenizer,
    eot_id: int,
) -> np.ndarray:
    """Tokenize all documents in a JSONL file into a single token array.

    Each document is tokenized and terminated with <|endoftext|>.
    Returns a flat numpy array of token IDs (uint16 since vocab < 65536).

    WHY uint16?
    Our vocab is 32768 < 65536, so uint16 (2 bytes/token) is sufficient.
    This halves memory vs int32, which matters when we have hundreds of
    millions of tokens. It also halves disk space for the cached files.
    """
    all_ids = []
    doc_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            text = doc["text"]

            # Tokenize
            encoded = tokenizer.encode(text)
            ids = encoded.ids

            # Append tokens + end-of-text marker
            all_ids.extend(ids)
            all_ids.append(eot_id)
            doc_count += 1

    tokens = np.array(all_ids, dtype=np.uint16)
    return tokens, doc_count


def pack_sequences(
    tokens: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """Pack a flat token array into fixed-length training sequences.

    WHY fixed-length?
    GPUs are most efficient when processing uniform tensor shapes.
    Variable-length sequences require padding (wasted compute) or
    complex bucketing. Fixed-length packing wastes at most seq_len-1
    tokens at the very end (negligible for millions of tokens).

    The last partial sequence is dropped (not padded). With millions
    of tokens, losing up to 2047 tokens is insignificant.

    Returns: (n_sequences, seq_len) array of token IDs
    """
    n_tokens = len(tokens)
    n_sequences = n_tokens // seq_len

    # Trim to exact multiple of seq_len
    tokens = tokens[:n_sequences * seq_len]

    # Reshape into sequences
    sequences = tokens.reshape(n_sequences, seq_len)
    return sequences


def prepare_training_data(
    seq_len: int = 2048,
    aviation_ratio: float = 0.30,
    seed: int = 42,
    output_dir: Path = TOKENIZED_DIR,
):
    """Full pipeline: tokenize, mix, pack, and save training data.

    Outputs:
        data/tokenized/train.bin  - packed training sequences (memmap-friendly)
        data/tokenized/val.bin    - validation sequences (1% of data)
        data/tokenized/meta.json  - metadata (token counts, sequence counts, etc.)
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("FlightMind Training Data Preparation")
    print("=" * 60)

    # Load tokenizer
    tokenizer = load_tokenizer()
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"Tokenizer loaded (vocab={tokenizer.get_vocab_size():,}, EOT={eot_id})")
    print(f"Sequence length: {seq_len:,}")

    # Tokenize each source
    print("\nTokenizing sources...")
    aviation_tokens = []
    general_tokens = []

    for jsonl_name in AVIATION_JSONL:
        jsonl_path = CLEANED_DIR / jsonl_name
        if not jsonl_path.exists():
            continue
        t0 = time.time()
        tokens, n_docs = tokenize_jsonl(jsonl_path, tokenizer, eot_id)
        elapsed = time.time() - t0
        print(f"  {jsonl_name}: {n_docs:,} docs -> {len(tokens):,} tokens ({elapsed:.1f}s)")
        aviation_tokens.append(tokens)

    for jsonl_name in GENERAL_JSONL:
        jsonl_path = CLEANED_DIR / jsonl_name
        if not jsonl_path.exists():
            continue
        t0 = time.time()
        tokens, n_docs = tokenize_jsonl(jsonl_path, tokenizer, eot_id)
        elapsed = time.time() - t0
        print(f"  {jsonl_name}: {n_docs:,} docs -> {len(tokens):,} tokens ({elapsed:.1f}s)")
        general_tokens.append(tokens)

    # Concatenate within each pool
    if aviation_tokens:
        aviation_all = np.concatenate(aviation_tokens)
    else:
        aviation_all = np.array([], dtype=np.uint16)

    if general_tokens:
        general_all = np.concatenate(general_tokens)
    else:
        general_all = np.array([], dtype=np.uint16)

    n_aviation = len(aviation_all)
    n_general = len(general_all)
    print(f"\nAviation tokens: {n_aviation:,}")
    print(f"General tokens:  {n_general:,}")

    # Mix according to ratio
    # If we don't have enough general data to fill 70%, use what we have
    total_available = n_aviation + n_general
    target_aviation = int(total_available * aviation_ratio)
    target_general = total_available - target_aviation

    # Clamp to available data
    actual_aviation = min(target_aviation, n_aviation)
    actual_general = min(target_general, n_general)

    # If one pool is short, give the excess budget to the other
    if actual_aviation < target_aviation:
        actual_general = min(n_general, total_available - actual_aviation)
    if actual_general < target_general:
        actual_aviation = min(n_aviation, total_available - actual_general)

    # Sample
    if actual_aviation < n_aviation:
        # Subsample aviation (take a contiguous slice for data locality)
        start = random.randint(0, n_aviation - actual_aviation)
        aviation_sample = aviation_all[start:start + actual_aviation]
    else:
        aviation_sample = aviation_all

    if actual_general < n_general:
        start = random.randint(0, n_general - actual_general)
        general_sample = general_all[start:start + actual_general]
    else:
        general_sample = general_all

    # Combine and shuffle at sequence level (not token level!)
    all_tokens = np.concatenate([aviation_sample, general_sample])
    actual_ratio = len(aviation_sample) / len(all_tokens) if len(all_tokens) > 0 else 0
    print(f"\nMixed: {len(all_tokens):,} tokens "
          f"(aviation: {actual_ratio:.1%}, general: {1 - actual_ratio:.1%})")

    # Pack into sequences
    sequences = pack_sequences(all_tokens, seq_len)
    n_sequences = len(sequences)
    print(f"Packed into {n_sequences:,} sequences of {seq_len} tokens")

    # Shuffle sequences
    print("Shuffling sequences...")
    np.random.shuffle(sequences)

    # Split into train/val (99/1)
    val_size = max(1, n_sequences // 100)
    train_size = n_sequences - val_size
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]

    print(f"Train: {train_size:,} sequences ({train_size * seq_len:,} tokens)")
    print(f"Val:   {val_size:,} sequences ({val_size * seq_len:,} tokens)")

    # Save as raw binary (memmap-friendly)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    # Save as flat uint16 binary
    # To load: np.memmap(path, dtype=np.uint16).reshape(-1, seq_len)
    train_sequences.tofile(str(train_path))
    val_sequences.tofile(str(val_path))

    train_mb = train_path.stat().st_size / 1e6
    val_mb = val_path.stat().st_size / 1e6
    print(f"\nSaved: {train_path} ({train_mb:.1f} MB)")
    print(f"Saved: {val_path} ({val_mb:.1f} MB)")

    # Save metadata
    meta = {
        "seq_len": seq_len,
        "vocab_size": tokenizer.get_vocab_size(),
        "eot_id": eot_id,
        "n_train_sequences": train_size,
        "n_val_sequences": val_size,
        "n_train_tokens": train_size * seq_len,
        "n_val_tokens": val_size * seq_len,
        "aviation_tokens": int(len(aviation_sample)),
        "general_tokens": int(len(general_sample)),
        "aviation_ratio": float(actual_ratio),
        "dtype": "uint16",
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {meta_path}")

    print(f"\nDone! Ready for training.")
    return meta


class PretrainDataset:
    """Memory-mapped dataset for efficient training data loading.

    WHY memory-mapped?
    With hundreds of millions of tokens, loading everything into RAM is
    wasteful (and may not fit). Memory mapping lets the OS handle paging:
    only the sequences currently being read are loaded into physical memory.
    This is transparent to the training loop.

    WHY not a standard PyTorch Dataset?
    We could use torch.utils.data.Dataset, but our data is simple enough
    that a lightweight wrapper around memmap is cleaner and avoids the
    overhead of DataLoader's multiprocessing (which has issues on Windows).
    For distributed training, we'll add a DistributedSampler later.
    """

    def __init__(self, bin_path: Path, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.n_sequences = len(self.data) // seq_len
        # Reshape as (n_sequences, seq_len)
        self.data = self.data[:self.n_sequences * seq_len].reshape(
            self.n_sequences, seq_len
        )

    def __len__(self) -> int:
        return self.n_sequences

    def get_batch(self, batch_size: int, device: str = "cpu"):
        """Get a random batch of (input, target) pairs.

        For language modeling, target[i] = input[i+1] (predict next token).
        So we take seq_len+1 tokens: first seq_len are input, last seq_len
        are target (shifted by 1).

        Wait - our sequences are exactly seq_len tokens, not seq_len+1.
        So input = sequence[:-1] and target = sequence[1:], giving us
        seq_len-1 predictions per sequence. This wastes 1 token per
        sequence (~0.05% overhead at seq_len=2048). Acceptable.

        Actually, the standard approach is simpler: input = sequence,
        target = sequence shifted by 1. Since sequences are packed
        (continuous text with EOT separators), the "next token" for the
        last position of one sequence IS the first token of the next
        sequence in the original data stream. But since we shuffled,
        we just accept losing the last prediction. This is standard.
        """
        import torch

        # Random sequence indices
        idx = np.random.randint(0, self.n_sequences, size=batch_size)
        batch = self.data[idx]  # (batch_size, seq_len)

        # Convert to torch tensors
        x = torch.from_numpy(batch[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(batch[:, 1:].astype(np.int64)).to(device)
        return x, y


class FineWebStreamer:
    """Streams FineWeb-EDU from HuggingFace and produces packed token sequences.

    For the general-domain portion of training, we stream from HuggingFace's
    FineWeb-EDU dataset (~1.3T tokens) instead of storing it locally. Documents
    are fetched via HTTP, tokenized with our BPE tokenizer, and packed into
    fixed-length sequences with the same interface as PretrainDataset.

    In multi-GPU (DDP) mode, each rank gets a different shard of the stream
    so GPUs never see the same documents.
    """

    def __init__(self, seq_len, rank=0, world_size=1, seed=42, buffer_target=256):
        from collections import deque
        from datasets import load_dataset

        self.seq_len = seq_len
        self._buffer_target = buffer_target

        tokenizer = load_tokenizer()
        self._tokenizer = tokenizer
        self._eot_id = tokenizer.token_to_id("<|endoftext|>")

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", split="train", streaming=True
        )
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        ds = ds.shuffle(seed=seed + rank, buffer_size=10_000)

        self._iter = iter(ds)
        self._token_buf = []
        self._seq_queue = deque()
        self.tokens_streamed = 0

        self._fill()

    def _fill(self):
        """Tokenize and pack streamed documents until buffer reaches target."""
        while len(self._seq_queue) < self._buffer_target:
            # Fetch tokens from stream
            while len(self._token_buf) < self.seq_len:
                try:
                    doc = next(self._iter)
                except StopIteration:
                    return
                ids = self._tokenizer.encode(doc["text"]).ids
                self._token_buf.extend(ids)
                self._token_buf.append(self._eot_id)

            # Pack complete sequences
            while len(self._token_buf) >= self.seq_len:
                seq = np.array(self._token_buf[:self.seq_len], dtype=np.uint16)
                self._token_buf = self._token_buf[self.seq_len:]
                self._seq_queue.append(seq)
                self.tokens_streamed += self.seq_len

    def get_batch(self, batch_size, device="cpu"):
        """Get a batch of (input, target) pairs. Same interface as PretrainDataset."""
        import torch

        if len(self._seq_queue) < batch_size:
            self._fill()

        seqs = []
        for _ in range(batch_size):
            if not self._seq_queue:
                self._fill()
                if not self._seq_queue:
                    raise RuntimeError("FineWeb-EDU stream exhausted")
            seqs.append(self._seq_queue.popleft())

        # Trigger refill when buffer gets low
        if len(self._seq_queue) < self._buffer_target // 2:
            self._fill()

        batch = np.stack(seqs)
        x = torch.from_numpy(batch[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(batch[:, 1:].astype(np.int64)).to(device)
        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FlightMind training data")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--aviation-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_training_data(
        seq_len=args.seq_len,
        aviation_ratio=args.aviation_ratio,
        seed=args.seed,
    )
