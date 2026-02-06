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
- Multi-GPU training via PyTorch DDP (DistributedDataParallel)

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

5. WHY DDP?
   DistributedDataParallel replicates the model on each GPU. Each GPU
   processes different data, computes gradients independently, then
   all-reduces gradients before the optimizer step. This gives near-linear
   scaling: 4 GPUs ≈ 4x throughput. We use no_sync() during gradient
   accumulation to avoid redundant all-reduce calls.

Usage:
    # Local test (tiny model, CPU)
    python train/pretrain.py --depth 4 --device cpu --max-steps 100

    # Single GPU training
    python train/pretrain.py --depth 20 --device cuda

    # Multi-GPU training (4 GPUs)
    torchrun --nproc_per_node=4 train/pretrain.py --depth 24

    # Multi-GPU training (8 GPUs)
    torchrun --nproc_per_node=8 train/pretrain.py --depth 32 --lr 3e-4

    # Resume from checkpoint (single or multi-GPU)
    torchrun --nproc_per_node=4 train/pretrain.py --depth 24 --resume checkpoints/step_10000.pt
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import FlightMindConfig, FlightMind
from train.dataloader import PretrainDataset


# ---------------------------------------------------------------------------
# Distributed Setup
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize distributed training if launched via torchrun.

    WHY check for RANK env var?
    torchrun automatically sets RANK, LOCAL_RANK, and WORLD_SIZE environment
    variables. If they're absent, we're running with plain `python` and should
    fall back to single-GPU mode. This makes the script backward-compatible.

    Returns:
        (rank, local_rank, world_size) tuple.
        For single-GPU: (0, 0, 1).
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


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

def log(msg, logfile=None, rank=0):
    """Print and optionally write to log file (rank 0 only)."""
    if rank != 0:
        return
    print(msg, flush=True)
    if logfile:
        logfile.write(msg + "\n")
        logfile.flush()
        os.fsync(logfile.fileno())  # Force OS to write to disk immediately


def train(args):
    """Main training function."""
    # ---- Distributed setup ----
    rank, local_rank, world_size = setup_distributed()
    is_distributed = world_size > 1
    is_main = rank == 0

    # ---- Device ----
    if is_distributed:
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    # ---- Logging (rank 0 only) ----
    logfile = None
    if is_main:
        log_path = PROJECT_ROOT / "train.log"
        logfile = open(log_path, "w", encoding="utf-8")

    log("=" * 60, logfile, rank)
    log("FlightMind Pretraining", logfile, rank)
    log("=" * 60, logfile, rank)

    dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_bf16_supported() else torch.float32
    log(f"Device: {device}, Dtype: {dtype}", logfile, rank)
    if is_distributed:
        log(f"Distributed: {world_size} GPUs (rank {rank})", logfile, rank)

    # ---- Model ----
    config = FlightMindConfig(
        depth=args.depth,
        max_seq_len=args.seq_len,
        dropout=0.0,
    )
    log(f"\n{config.describe()}", logfile, rank)

    model = FlightMind(config).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", logfile, rank)

    # ---- Data ----
    # Seed each rank differently so GPUs process different batches
    np.random.seed(args.seed + rank)

    tokenized_dir = PROJECT_ROOT / "data" / "tokenized"
    train_data = PretrainDataset(tokenized_dir / "train.bin", args.seq_len)
    val_data = PretrainDataset(tokenized_dir / "val.bin", args.seq_len)
    log(f"\nTrain: {len(train_data):,} sequences ({len(train_data) * args.seq_len:,} tokens)", logfile, rank)
    log(f"Val:   {len(val_data):,} sequences ({len(val_data) * args.seq_len:,} tokens)", logfile, rank)

    # ---- FineWeb-EDU streaming (optional) ----
    fineweb_streamer = None
    aviation_bs = args.batch_size
    fineweb_bs = 0

    if args.fineweb:
        from train.dataloader import FineWebStreamer

        log(f"\nInitializing FineWeb-EDU stream...", logfile, rank)
        t0_fw = time.time()
        fineweb_streamer = FineWebStreamer(
            seq_len=args.seq_len,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
        )
        fw_init_time = time.time() - t0_fw

        aviation_bs = max(1, round(args.batch_size * args.aviation_ratio))
        fineweb_bs = args.batch_size - aviation_bs

        log(f"FineWeb-EDU ready ({fw_init_time:.1f}s)", logfile, rank)
        log(f"  Batch split: {aviation_bs} aviation + {fineweb_bs} FineWeb = {args.batch_size} total", logfile, rank)
        log(f"  Aviation ratio: {args.aviation_ratio:.0%}", logfile, rank)

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
    log(f"\nOptimizer: AdamW (lr={args.lr}, wd={args.weight_decay})", logfile, rank)
    log(f"  Decayed params:    {n_decay:,}", logfile, rank)
    log(f"  Non-decayed params: {n_nodecay:,}", logfile, rank)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95),   # Beta2=0.95 is standard for LLM pretraining (vs 0.999 default)
        eps=1e-8,
        fused="cuda" in device,  # Fused AdamW is faster on CUDA
    )

    # ---- Gradient accumulation ----
    # tokens_per_step = batch_size * seq_len * grad_accum_steps * world_size
    # We want ~512K tokens per optimizer step
    tokens_per_micro = args.batch_size * (args.seq_len - 1)  # -1 because target is shifted
    grad_accum_steps = max(1, args.tokens_per_step // (tokens_per_micro * world_size))
    tokens_per_step = tokens_per_micro * grad_accum_steps * world_size
    log(f"\nGradient accumulation: {grad_accum_steps} steps (x{world_size} GPUs)", logfile, rank)
    log(f"Effective batch: {tokens_per_step:,} tokens/step", logfile, rank)

    # ---- Resume from checkpoint ----
    # Load BEFORE wrapping with DDP so all ranks start with identical weights
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        log(f"\nResumed from {args.resume} (step {start_step})", logfile, rank)

    # ---- Wrap model with DDP ----
    # WHY after checkpoint load? DDP wraps the model, adding a .module prefix
    # to all state dict keys. Loading a non-DDP checkpoint into a DDP model
    # would fail due to key mismatch. So we load first, then wrap.
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # For saving checkpoints, we need the unwrapped model
    raw_model = model.module if is_distributed else model

    # ---- Training loop ----
    log(f"\nTraining for {args.max_steps:,} steps...", logfile, rank)
    log(f"LR schedule: warmup {args.warmup_steps} steps, cosine decay to {args.min_lr}", logfile, rank)
    log("-" * 60, logfile, rank)

    model.train()
    best_val_loss = float("inf")
    log_interval = args.log_every
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    if is_main:
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
            if fineweb_streamer is not None:
                x_av, y_av = train_data.get_batch(aviation_bs, device)
                x_fw, y_fw = fineweb_streamer.get_batch(fineweb_bs, device)
                x = torch.cat([x_av, x_fw], dim=0)
                y = torch.cat([y_av, y_fw], dim=0)
            else:
                x, y = train_data.get_batch(args.batch_size, device)

            # WHY no_sync()?
            # DDP normally all-reduces gradients after every backward() call.
            # During gradient accumulation, we don't want that — we want to
            # accumulate local gradients and only sync on the final micro-step.
            # no_sync() disables the all-reduce hook, and the final micro-step
            # (without no_sync) triggers the actual gradient synchronization.
            # This reduces inter-GPU communication by grad_accum_steps times.
            is_last_micro = (micro_step == grad_accum_steps - 1)
            sync_ctx = nullcontext() if (not is_distributed or is_last_micro) else model.no_sync()

            with sync_ctx:
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

        # ---- Logging (rank 0 only) ----
        if step % log_interval == 0 or step == args.max_steps - 1:
            elapsed = time.time() - t0
            tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
            log(
                f"step {step:>6d} | loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"{tokens_per_sec:,.0f} tok/s",
                logfile,
                rank,
            )

        # ---- Evaluation (rank 0 only) ----
        if step > 0 and step % args.eval_every == 0:
            if is_main:
                val_loss = evaluate(raw_model, val_data, args.batch_size, device, dtype, args.eval_steps)
                log(f"  -> val_loss: {val_loss:.4f}", logfile, rank)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        raw_model, optimizer, config, step, val_loss,
                        checkpoint_dir / "best.pt",
                    )

            # All ranks wait for rank 0 to finish eval + checkpoint
            if is_distributed:
                dist.barrier()

            model.train()

        # ---- Checkpointing (rank 0 only) ----
        if step > 0 and step % args.checkpoint_every == 0:
            if is_main:
                save_checkpoint(
                    raw_model, optimizer, config, step, loss_accum,
                    checkpoint_dir / f"step_{step}.pt",
                )
            if is_distributed:
                dist.barrier()

    # Sync all ranks before final save
    if is_distributed:
        dist.barrier()

    # Final save
    if is_main:
        save_checkpoint(
            raw_model, optimizer, config, args.max_steps - 1, loss_accum,
            checkpoint_dir / "final.pt",
        )

    if is_distributed:
        dist.barrier()

    total_time = time.time() - t0
    log(f"\nTraining complete! {args.max_steps} steps in {total_time:.0f}s", logfile, rank)
    log(f"Best val loss: {best_val_loss:.4f}", logfile, rank)
    log(f"Total tokens processed: {tokens_processed:,}", logfile, rank)

    if logfile:
        logfile.close()

    # Cleanup distributed
    if is_distributed:
        dist.destroy_process_group()


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
    print(f"  Checkpoint saved: {path.name} ({size_mb:.0f} MB, step {step})", flush=True)


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
    parser.add_argument("--seed", type=int, default=42, help="Random seed (each GPU gets seed + rank)")

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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for single-GPU mode (ignored when using torchrun)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # FineWeb-EDU streaming
    parser.add_argument("--fineweb", action="store_true", help="Stream FineWeb-EDU for general data (70/30 mix)")
    parser.add_argument("--aviation-ratio", type=float, default=0.30, help="Fraction of batch from aviation data (default: 0.30)")

    args = parser.parse_args()
    train(args)
