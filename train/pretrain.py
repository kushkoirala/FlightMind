"""
FlightMind Pretraining Loop
=============================
Full training script for pretraining FlightMind from scratch.

This script handles:
- Model construction from config
- AdamW optimizer with cosine learning rate schedule
- Gradient accumulation for large effective batch sizes
- Mixed-precision training (bfloat16/float16) for speed
- Periodic evaluation and checkpointing
- Logging to console and optional wandb

Design Decisions (educational notes throughout):

1. WHY AdamW?
   Adam with decoupled weight decay. Standard for transformer training.
   SGD requires very careful LR tuning for transformers; Adam is more
   forgiving and converges faster. Weight decay acts as L2 regularization
   on weights (but NOT on biases/norms - they shouldn't be regularized).

2. WHY cosine schedule?
   After linear warmup, LR follows cos(progress * pi/2) from peak to ~0.
   This is empirically the best schedule for LLM pretraining (vs linear
   decay, step decay, etc.). The gradual decay prevents sudden loss spikes
   that happen with step schedules.

3. WHY gradient accumulation?
   We want large batch sizes (512K tokens/step) for training stability,
   but a single GPU can only fit a few sequences in memory. Gradient
   accumulation simulates large batches by summing gradients across
   multiple micro-batches before stepping the optimizer.

4. WHY mixed precision?
   bfloat16 uses 2 bytes instead of 4 per activation, roughly doubling
   throughput and halving memory. The reduced precision is fine for
   forward/backward passes; the optimizer maintains float32 copies of
   weights for accurate updates.

Usage:
    # Local test (tiny model, CPU)
    python train/pretrain.py --depth 4 --device cpu --max-steps 100

    # GPU training (full model)
    python train/pretrain.py --depth 20 --device cuda

    # Resume from checkpoint
    python train/pretrain.py --resume checkpoints/step_10000.pt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import FlightMindConfig, FlightMind
from train.dataloader import PretrainDataset


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup.

    WHY this specific schedule?

    Phase 1 - Linear warmup (steps 0 to warmup_steps):
        LR ramps from 0 to max_lr linearly. This prevents large gradient
        updates in early training when the model's weights are random and
        loss is high. Without warmup, Adam's adaptive LR can overshoot
        catastrophically on the first few steps.

    Phase 2 - Cosine decay (warmup_steps to max_steps):
        LR follows cos() from max_lr down to min_lr. This provides:
        - Fast progress early (high LR explores loss landscape broadly)
        - Careful refinement late (low LR fine-tunes to a good minimum)
        - No abrupt transitions (unlike step decay which causes loss spikes)

    min_lr is typically max_lr / 10. Setting it to 0 works but can cause
    the model to "forget" in the very last steps. A small floor prevents this.

    Visualization:

    LR ^
       |     /\
       |    /   \
       |   /     `.
       |  /        `-.___
       | /                 min_lr
       +-------------------------> step
       0  warmup          max_steps
    """
    # Warmup phase
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay phase
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """Main training function."""
    print("=" * 60)
    print("FlightMind Pretraining")
    print("=" * 60)

    device = args.device
    dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Device: {device}, Dtype: {dtype}")

    # ---- Model ----
    config = FlightMindConfig(
        depth=args.depth,
        max_seq_len=args.seq_len,
        dropout=0.0,
    )
    print(f"\n{config.describe()}")

    model = FlightMind(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Data ----
    tokenized_dir = PROJECT_ROOT / "data" / "tokenized"
    train_data = PretrainDataset(tokenized_dir / "train.bin", args.seq_len)
    val_data = PretrainDataset(tokenized_dir / "val.bin", args.seq_len)
    print(f"\nTrain: {len(train_data):,} sequences ({len(train_data) * args.seq_len:,} tokens)")
    print(f"Val:   {len(val_data):,} sequences ({len(val_data) * args.seq_len:,} tokens)")

    # ---- Optimizer ----
    # Separate parameters: apply weight decay only to 2D params (weights),
    # not to 1D params (norms, biases). This is important because:
    # - Weight decay on norms/biases hurts: these should be free to take any value
    # - Weight decay on weight matrices is beneficial: acts as L2 regularization
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"\nOptimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"  Decayed params:    {n_decay:,}")
    print(f"  Non-decayed params: {n_nodecay:,}")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95),   # Beta2=0.95 is standard for LLM pretraining (vs 0.999 default)
        eps=1e-8,
        fused=device == "cuda",  # Fused AdamW is faster on CUDA
    )

    # ---- Gradient accumulation ----
    # tokens_per_step = batch_size * seq_len * grad_accum_steps
    # We want ~512K tokens per optimizer step
    tokens_per_micro = args.batch_size * (args.seq_len - 1)  # -1 because target is shifted
    grad_accum_steps = max(1, args.tokens_per_step // (tokens_per_micro * max(1, args.n_gpu)))
    tokens_per_step = tokens_per_micro * grad_accum_steps * max(1, args.n_gpu)
    print(f"\nGradient accumulation: {grad_accum_steps} steps")
    print(f"Effective batch: {tokens_per_step:,} tokens/step")

    # ---- Resume from checkpoint ----
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        print(f"\nResumed from {args.resume} (step {start_step})")

    # ---- Training loop ----
    print(f"\nTraining for {args.max_steps:,} steps...")
    print(f"LR schedule: warmup {args.warmup_steps} steps, cosine decay to {args.min_lr}")
    print("-" * 60)

    model.train()
    best_val_loss = float("inf")
    log_interval = args.log_every
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    t0 = time.time()
    tokens_processed = 0

    for step in range(start_step, args.max_steps):
        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_data.get_batch(args.batch_size, device)

            # Mixed precision forward pass
            with torch.amp.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=dtype != torch.float32):
                logits, loss = model(x, y)

            # Scale loss by accumulation steps (so gradients are averaged, not summed)
            loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.item()

        # Gradient clipping (prevents exploding gradients)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()
        tokens_processed += tokens_per_step

        # ---- Logging ----
        if step % log_interval == 0 or step == args.max_steps - 1:
            elapsed = time.time() - t0
            tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
            print(
                f"step {step:>6d} | loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"{tokens_per_sec:,.0f} tok/s"
            )

        # ---- Evaluation ----
        if step > 0 and step % args.eval_every == 0:
            val_loss = evaluate(model, val_data, args.batch_size, device, dtype, args.eval_steps)
            print(f"  -> val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, config, step, val_loss,
                    checkpoint_dir / "best.pt",
                )

            model.train()

        # ---- Checkpointing ----
        if step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, config, step, loss_accum,
                checkpoint_dir / f"step_{step}.pt",
            )

    # Final save
    save_checkpoint(
        model, optimizer, config, args.max_steps - 1, loss_accum,
        checkpoint_dir / "final.pt",
    )

    total_time = time.time() - t0
    print(f"\nTraining complete! {args.max_steps} steps in {total_time:.0f}s")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total tokens processed: {tokens_processed:,}")


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, dtype, n_steps=20):
    """Evaluate model on validation data.

    Returns average cross-entropy loss over n_steps batches.
    """
    model.eval()
    total_loss = 0.0

    for _ in range(n_steps):
        x, y = dataset.get_batch(batch_size, device)
        with torch.amp.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=dtype != torch.float32):
            _, loss = model(x, y)
        total_loss += loss.item()

    return total_loss / n_steps


def save_checkpoint(model, optimizer, config, step, loss, path):
    """Save training checkpoint.

    WHY save optimizer state?
    Adam maintains per-parameter momentum (m) and variance (v) estimates.
    Without these, resuming training from a checkpoint would reset the
    optimizer to "cold start" and likely cause a loss spike. Saving the
    full optimizer state allows seamless resumption.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "depth": config.depth,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
        },
        "step": step,
        "loss": loss,
    }
    torch.save(checkpoint, str(path))
    size_mb = path.stat().st_size / 1e6
    print(f"  Checkpoint saved: {path.name} ({size_mb:.0f} MB, step {step})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlightMind Pretraining")

    # Model
    parser.add_argument("--depth", type=int, default=20, help="Model depth (default: 20 = 566M)")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")

    # Training
    parser.add_argument("--max-steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size (per GPU)")
    parser.add_argument("--tokens-per-step", type=int, default=524288, help="Target tokens per optimizer step (~512K)")
    parser.add_argument("--n-gpu", type=int, default=1, help="Number of GPUs (for grad accum calculation)")

    # Optimizer
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate (end of cosine)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping")

    # Logging and checkpointing
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--eval-steps", type=int, default=20, help="Batches per evaluation")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Checkpoint every N steps")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)
