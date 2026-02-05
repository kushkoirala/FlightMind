"""
FlightMind BPE Tokenizer Training
===================================
Trains a custom BPE tokenizer on cleaned aviation + general text data.

Uses HuggingFace tokenizers library for fast Rust-based BPE training.
Maintains the 70/30 general/aviation ratio from config.yaml by sampling.

Output: tokenizer/ directory with tokenizer.json + vocab files.

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 32768
    python scripts/train_tokenizer.py --sample-size 50000000  # 50M chars
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"

# Aviation sources (30% of training data)
AVIATION_SOURCES = {
    "ntsb.jsonl", "handbooks.jsonl", "regulations.jsonl", "metar.jsonl",
    "wikipedia.jsonl", "hf_datasets.jsonl", "openap.jsonl", "ntsb_csv.jsonl",
}

# General sources (70% of training data)
GENERAL_SOURCES = {
    "fineweb.jsonl",
}


def load_texts(jsonl_path: Path, max_chars: int = 0) -> list[str]:
    """Load text documents from a JSONL file."""
    texts = []
    total_chars = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            text = doc["text"]
            texts.append(text)
            total_chars += len(text)
            if max_chars > 0 and total_chars >= max_chars:
                break
    return texts


def build_training_corpus(
    aviation_ratio: float = 0.30,
    sample_chars: int = 100_000_000,  # 100M chars for tokenizer training
) -> list[str]:
    """Build a balanced training corpus respecting aviation/general ratio.

    For tokenizer training we don't need all the data - a representative
    sample is sufficient and faster to train on.
    """
    aviation_budget = int(sample_chars * aviation_ratio)
    general_budget = int(sample_chars * (1 - aviation_ratio))

    print(f"Target: {sample_chars / 1e6:.0f}M chars "
          f"({aviation_ratio:.0%} aviation / {1 - aviation_ratio:.0%} general)")

    all_texts = []

    # Load aviation data
    print("\nLoading aviation data...")
    aviation_texts = []
    aviation_chars = 0
    for jsonl_name in sorted(AVIATION_SOURCES):
        jsonl_path = CLEANED_DIR / jsonl_name
        if not jsonl_path.exists():
            continue
        texts = load_texts(jsonl_path)
        chars = sum(len(t) for t in texts)
        aviation_texts.extend(texts)
        aviation_chars += chars
        print(f"  {jsonl_name}: {len(texts):,} docs, {chars / 1e6:.1f}M chars")

    # Load general data
    print("\nLoading general data...")
    general_texts = []
    general_chars = 0
    for jsonl_name in sorted(GENERAL_SOURCES):
        jsonl_path = CLEANED_DIR / jsonl_name
        if not jsonl_path.exists():
            continue
        texts = load_texts(jsonl_path)
        chars = sum(len(t) for t in texts)
        general_texts.extend(texts)
        general_chars += chars
        print(f"  {jsonl_name}: {len(texts):,} docs, {chars / 1e6:.1f}M chars")

    print(f"\nAvailable: {aviation_chars / 1e6:.1f}M aviation, "
          f"{general_chars / 1e6:.1f}M general")

    # Sample to budget
    random.shuffle(aviation_texts)
    random.shuffle(general_texts)

    # Take aviation texts up to budget
    sampled_aviation = []
    chars_so_far = 0
    for t in aviation_texts:
        if chars_so_far >= aviation_budget:
            break
        sampled_aviation.append(t)
        chars_so_far += len(t)

    # Take general texts up to budget
    sampled_general = []
    chars_so_far = 0
    for t in general_texts:
        if chars_so_far >= general_budget:
            break
        sampled_general.append(t)
        chars_so_far += len(t)

    all_texts = sampled_aviation + sampled_general
    random.shuffle(all_texts)

    actual_aviation = sum(len(t) for t in sampled_aviation)
    actual_general = sum(len(t) for t in sampled_general)
    total = actual_aviation + actual_general

    print(f"\nTraining corpus: {len(all_texts):,} docs, {total / 1e6:.1f}M chars")
    print(f"  Aviation: {len(sampled_aviation):,} docs, "
          f"{actual_aviation / 1e6:.1f}M chars ({actual_aviation / total:.1%})")
    print(f"  General:  {len(sampled_general):,} docs, "
          f"{actual_general / 1e6:.1f}M chars ({actual_general / total:.1%})")

    return all_texts


def train_tokenizer(
    texts: list[str],
    vocab_size: int = 32768,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a BPE tokenizer on the given texts."""
    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size:,})...")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # GPT-style pre-tokenization: split on whitespace + punctuation patterns
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Special tokens
    special_tokens = [
        "<|endoftext|>",
        "<|pad|>",
        "<|begin_of_turn|>",
        "<|end_of_turn|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
    ]

    # BPE trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train from iterator (memory efficient)
    t0 = time.time()
    tokenizer.train_from_iterator(texts, trainer=trainer)
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    # Post-processing: ByteLevel decoder + add EOT token behavior
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    return tokenizer


def evaluate_tokenizer(tokenizer: Tokenizer):
    """Run basic evaluation on the trained tokenizer."""
    print("\n" + "=" * 60)
    print("Tokenizer Evaluation")
    print("=" * 60)

    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size:,}")

    # Test encoding on aviation-specific text
    test_texts = [
        "METAR KJFK 051953Z 31015G25KT 10SM FEW250 M02/M17 A3042 RMK AO2",
        "The pilot reported experiencing a loss of engine power during cruise flight at 8,500 feet MSL.",
        "14 CFR 91.113 states that aircraft on final approach have the right-of-way.",
        "Cleared ILS runway 28L approach, maintain 3,000 until established, contact tower 119.1.",
        "The Cessna 172S has a maximum takeoff weight of 2,550 pounds and a Vne of 163 KIAS.",
        "Cumulus clouds were observed building rapidly to the west with tops estimated at FL350.",
        "The aircraft's left main landing gear failed to extend on the downwind leg.",
        # General text for comparison
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large amounts of training data.",
    ]

    print(f"\nSample encodings:")
    total_chars = 0
    total_tokens = 0
    for text in test_texts:
        encoded = tokenizer.encode(text)
        n_tokens = len(encoded.ids)
        ratio = len(text) / n_tokens if n_tokens > 0 else 0
        total_chars += len(text)
        total_tokens += n_tokens
        print(f"  [{n_tokens:3d} tokens, {ratio:.1f} c/t] {text[:80]}...")

    avg_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"\nAverage chars/token: {avg_ratio:.2f}")

    # Check aviation-specific tokens exist in vocab
    aviation_terms = [
        "METAR", "runway", "altitude", "airspeed", "knots", "ILS",
        "VFR", "IFR", "ATIS", "taxiway", "clearance", "turbulence",
        "flaps", "aileron", "rudder", "elevator", "throttle",
        "Cessna", "Boeing", "Airbus", "CFR", "NTSB", "FAA",
    ]

    vocab = tokenizer.get_vocab()
    found = 0
    for term in aviation_terms:
        # Check if term appears as a complete token (with byte-level encoding prefix)
        encoded = tokenizer.encode(term)
        if len(encoded.ids) == 1:
            found += 1
            status = "single token"
        else:
            status = f"{len(encoded.ids)} tokens"
        print(f"  '{term}': {status} -> {encoded.tokens}")

    print(f"\n{found}/{len(aviation_terms)} aviation terms are single tokens")


def main():
    parser = argparse.ArgumentParser(description="Train FlightMind BPE Tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32768,
                        help="Vocabulary size (default: 32768)")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Minimum token frequency (default: 2)")
    parser.add_argument("--sample-chars", type=int, default=100_000_000,
                        help="Total chars to sample for training (default: 100M)")
    parser.add_argument("--aviation-ratio", type=float, default=0.30,
                        help="Fraction of aviation data (default: 0.30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("FlightMind BPE Tokenizer Training")
    print("=" * 60)

    # Build training corpus
    texts = build_training_corpus(
        aviation_ratio=args.aviation_ratio,
        sample_chars=args.sample_chars,
    )

    if not texts:
        print("ERROR: No training data found. Run clean_all.py first.")
        sys.exit(1)

    # Train
    tokenizer = train_tokenizer(
        texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # Save
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(output_path))
    print(f"\nTokenizer saved to {output_path}")

    # Also save config for reference
    config = {
        "type": "bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": args.min_frequency,
        "sample_chars": args.sample_chars,
        "aviation_ratio": args.aviation_ratio,
        "training_docs": len(texts),
    }
    config_path = TOKENIZER_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # Evaluate
    evaluate_tokenizer(tokenizer)

    print(f"\nDone! Tokenizer at: {TOKENIZER_DIR}")


if __name__ == "__main__":
    main()
