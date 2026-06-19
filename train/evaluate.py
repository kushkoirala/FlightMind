"""
FlightMind Evaluation & Generation
=====================================
Tools for evaluating a trained FlightMind model and generating sample text.

This script provides three main functions:

1. **Perplexity measurement**: The standard metric for language models.
   Perplexity = exp(average_cross_entropy_loss). A perplexity of 100 means
   the model is as confused as if it had to choose uniformly from 100 tokens
   at each step. Lower is better. For reference:
   - Random model (32K vocab): perplexity = 32,768
   - Decent small LLM: perplexity ≈ 20-50
   - GPT-2 (124M) on OpenWebText: perplexity ≈ 29
   - GPT-2 (1.5B) on OpenWebText: perplexity ≈ 18

2. **Sample generation**: Generate text from aviation-themed prompts to
   qualitatively assess whether the model has learned aviation language.
   This is the most intuitive evaluation — can it write about flying?

3. **Domain probing**: Test the model on aviation-specific completions
   (e.g., "The Cessna 172 has a maximum cruise speed of") to see if
   it has absorbed domain knowledge vs. just generic language patterns.

Usage:
    # Evaluate checkpoint perplexity
    python train/evaluate.py --checkpoint checkpoints/best.pt --perplexity

    # Generate samples
    python train/evaluate.py --checkpoint checkpoints/best.pt --generate

    # Full evaluation (both)
    python train/evaluate.py --checkpoint checkpoints/best.pt --all

    # Quick test with random model (no checkpoint needed)
    python train/evaluate.py --depth 4 --generate --max-tokens 50
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import FlightMindConfig, FlightMind
from train.dataloader import PretrainDataset


# ---------------------------------------------------------------------------
# Perplexity Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: FlightMind,
    dataset: PretrainDataset,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    n_batches: int = 50,
) -> dict:
    """Compute perplexity over the validation set.

    WHY perplexity?
    ----------------
    Cross-entropy loss tells us the average negative log-probability the model
    assigns to each token. Perplexity is just exp(loss), which converts this
    into a more intuitive number: "how many tokens is the model effectively
    choosing between at each step?"

    A model with perplexity 25 is roughly as uncertain as rolling a 25-sided
    die at each position. Note that this includes easy predictions (articles,
    common words) and hard ones (technical terms, names), averaged together.

    We also compute per-sequence statistics to understand variance — some
    sequences (like METAR observations with a regular format) should have
    lower perplexity than others (like narrative NTSB reports).

    Returns dict with: perplexity, avg_loss, min_loss, max_loss, n_tokens
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_losses = []

    actual_batches = min(n_batches, len(dataset) // batch_size)
    if actual_batches == 0:
        return {"perplexity": float("inf"), "avg_loss": float("inf"),
                "n_tokens": 0, "n_batches": 0}

    for i in range(actual_batches):
        x, y = dataset.get_batch(batch_size, device)
        with torch.amp.autocast(device_type=device.split(":")[0], dtype=dtype,
                                enabled=dtype != torch.float32):
            _, loss = model(x, y)

        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        all_losses.append(loss.item())

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "min_batch_loss": min(all_losses),
        "max_batch_loss": max(all_losses),
        "std_batch_loss": float(np.std(all_losses)),
        "n_tokens": total_tokens,
        "n_batches": actual_batches,
    }


# ---------------------------------------------------------------------------
# Text Generation
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Load the FlightMind BPE tokenizer."""
    from tokenizers import Tokenizer
    tokenizer_path = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def generate_samples(
    model: FlightMind,
    tokenizer,
    device: str,
    prompts: list[str],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
) -> list[dict]:
    """Generate text completions for a list of prompts.

    WHY these generation parameters?
    ----------------------------------
    - temperature=0.8: Slightly below 1.0 for more focused but not greedy
      generation. At 1.0 you get maximum diversity but sometimes incoherent
      text. At 0.0 (greedy) you get repetitive but "safe" completions.

    - top_k=50: Only sample from the 50 most likely next tokens. This
      prevents the model from sampling very unlikely tokens that would
      derail the generation. The classic paper (Fan et al., 2018) found
      top-k and nucleus (top-p) sampling both work well.

    Returns list of {prompt, completion, tokens_per_sec, total_tokens}
    """
    model.eval()
    results = []

    for prompt in prompts:
        # Tokenize prompt
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
        prompt_len = input_ids.size(1)

        # Generate
        t0 = time.time()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        elapsed = time.time() - t0

        # Decode
        generated_ids = output_ids[0].tolist()
        full_text = tokenizer.decode(generated_ids)
        completion = tokenizer.decode(generated_ids[prompt_len:])
        new_tokens = len(generated_ids) - prompt_len

        results.append({
            "prompt": prompt,
            "completion": completion,
            "full_text": full_text,
            "new_tokens": new_tokens,
            "tokens_per_sec": new_tokens / elapsed if elapsed > 0 else 0,
            "elapsed_sec": elapsed,
        })

    return results


# ---------------------------------------------------------------------------
# Domain Probes
# ---------------------------------------------------------------------------

# Aviation-themed prompts for qualitative evaluation
AVIATION_PROMPTS = [
    # General aviation knowledge
    "The primary purpose of a preflight inspection is to",
    "During a standard rate turn, the aircraft banks at",
    "VFR weather minimums for Class E airspace require",

    # Aircraft knowledge
    "The Cessna 172 is powered by a",
    "Boeing 737 MAX was grounded because",

    # Procedures
    "On approach to landing, the pilot should configure",
    "In the event of an engine failure after takeoff, the pilot must",
    "ATIS information includes the current",

    # Weather
    "A METAR report showing TSRA indicates",
    "Cumulonimbus clouds are dangerous to aviation because",

    # Regulations
    "According to 14 CFR Part 91, the minimum safe altitude over",
    "A private pilot certificate requires a minimum of",
]

# More creative/open-ended prompts
OPEN_PROMPTS = [
    "The history of aviation began when",
    "Modern fly-by-wire systems work by",
    "The future of electric aviation depends on",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(args) -> tuple:
    """Load model from checkpoint or create fresh."""
    device = args.device
    dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_bf16_supported() else torch.float32

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = FlightMindConfig(**checkpoint["config"])
        model = FlightMind(config).to(device)
        model.load_state_dict(checkpoint["model"])
        step = checkpoint.get("step", "unknown")
        train_loss = checkpoint.get("loss", "unknown")
        print(f"  Step: {step}, Training loss: {train_loss}")
    else:
        print(f"Creating fresh model (depth={args.depth}) — no training yet")
        config = FlightMindConfig(depth=args.depth, max_seq_len=args.seq_len)
        model = FlightMind(config).to(device)
        step = 0

    print(f"{config.describe()}")
    return model, config, device, dtype


def main():
    parser = argparse.ArgumentParser(description="FlightMind Evaluation")

    # Model source
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--depth", type=int, default=4,
                        help="Model depth (only if no checkpoint)")
    parser.add_argument("--seq-len", type=int, default=2048)

    # Evaluation modes
    parser.add_argument("--perplexity", action="store_true",
                        help="Compute validation perplexity")
    parser.add_argument("--generate", action="store_true",
                        help="Generate sample completions")
    parser.add_argument("--all", action="store_true",
                        help="Run all evaluations")

    # Generation params
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (overrides built-in prompts)")

    # Perplexity params
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-batches", type=int, default=50,
                        help="Number of batches for perplexity")

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not args.perplexity and not args.generate and not args.all:
        args.all = True

    model, config, device, dtype = load_model(args)

    # ---- Perplexity ----
    if args.perplexity or args.all:
        print("=" * 60)
        print("Perplexity Evaluation")
        print("=" * 60)

        val_path = PROJECT_ROOT / "data" / "tokenized" / "val.bin"
        if val_path.exists():
            val_data = PretrainDataset(val_path, config.max_seq_len)
            print(f"Validation set: {len(val_data):,} sequences")

            t0 = time.time()
            metrics = compute_perplexity(
                model, val_data, args.batch_size, device, dtype, args.n_batches,
            )
            elapsed = time.time() - t0

            print(f"\nResults ({elapsed:.1f}s):")
            print(f"  Perplexity:    {metrics['perplexity']:.2f}")
            print(f"  Avg loss:      {metrics['avg_loss']:.4f}")
            print(f"  Loss range:    [{metrics['min_batch_loss']:.4f}, "
                  f"{metrics['max_batch_loss']:.4f}]")
            print(f"  Loss std:      {metrics['std_batch_loss']:.4f}")
            print(f"  Tokens eval'd: {metrics['n_tokens']:,}")

            # Context: what does this perplexity mean?
            print(f"\n  Context:")
            ppl = metrics["perplexity"]
            if ppl > 10000:
                print(f"  -> Near random ({config.vocab_size:,} vocab). Model barely trained.")
            elif ppl > 1000:
                print(f"  -> Very early training. Model learning basic token frequencies.")
            elif ppl > 100:
                print(f"  -> Early training. Model learning common patterns.")
            elif ppl > 50:
                print(f"  -> Moderate. Model producing somewhat coherent text.")
            elif ppl > 25:
                print(f"  -> Good for a small model. Coherent, topical text.")
            else:
                print(f"  -> Excellent. Near state-of-art for this model size.")
        else:
            print(f"  No validation data at {val_path}")
            print(f"  Run: python train/dataloader.py")

    # ---- Generation ----
    if args.generate or args.all:
        print("\n" + "=" * 60)
        print("Sample Generation")
        print("=" * 60)
        print(f"Settings: temperature={args.temperature}, top_k={args.top_k}, "
              f"max_tokens={args.max_tokens}")

        tokenizer = load_tokenizer()

        # Choose prompts
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = AVIATION_PROMPTS + OPEN_PROMPTS

        results = generate_samples(
            model, tokenizer, device, prompts,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        for i, result in enumerate(results):
            print(f"\n--- Prompt {i + 1}/{len(results)} ---")
            print(f"Prompt:     {result['prompt']}")
            print(f"Completion: {result['completion'][:500]}")
            print(f"({result['new_tokens']} tokens, {result['tokens_per_sec']:.1f} tok/s)")

        # Summary stats
        avg_tps = np.mean([r["tokens_per_sec"] for r in results])
        print(f"\n--- Summary ---")
        print(f"Average generation speed: {avg_tps:.1f} tokens/sec")

    print("\nDone!")


if __name__ == "__main__":
    main()
