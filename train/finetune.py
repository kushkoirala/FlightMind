"""
FlightMind Instruction Fine-tuning with LoRA
===============================================

Fine-tunes a pretrained FlightMind model on instruction-following data
using Low-Rank Adaptation (LoRA) for parameter-efficient training.

Background: Why LoRA?
---------------------
Full fine-tuning updates all model parameters, requiring full optimizer
states in memory (~12 bytes/param for AdamW). For a 956M-param d24 model,
that's ~11.5 GB just for optimizer states — exceeding our RTX 4060's 8 GB.

LoRA (Hu et al., 2021) freezes the pretrained weights and injects small
trainable rank-decomposition matrices into each linear layer:

    W_new = W_frozen + (alpha/r) * B @ A

Where:
    W_frozen: (d_out, d_in) — original pretrained weights, frozen
    A: (r, d_in)  — trainable, initialized from N(0, sigma)
    B: (d_out, r) — trainable, initialized to zero
    r: rank (typically 8-32) — controls capacity vs. efficiency

At rank r=16, a d24 model (956M params) adds only ~10M trainable params
(~1% of total). This reduces optimizer memory from 11.5 GB to ~120 MB,
making fine-tuning feasible on an 8 GB GPU.

The key insight: task-specific adaptations (like command parsing) live in
a low-dimensional subspace of the full weight space. We don't need to
modify all 956M parameters — the few million that matter for our task
can be captured by low-rank updates.

Data Format
-----------
Each training example is a JSON object with three fields:
    {
        "system": "You are FlightMind, an aviation AI copilot...",
        "user": "PHASE: cruise | ALT: 4500ft | SPD: 110kt | HDG: 270\\nturn heading 360",
        "assistant": '{"action": "heading", "value": 360}\\nRoger, heading 360.'
    }

These are formatted into a single sequence with special tokens:
    <|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>{assistant}<|end|>

The model is trained to predict only the assistant tokens (the system and
user tokens are masked in the loss computation).

Usage
-----
    # Fine-tune d8 on local GPU (test run)
    python train/finetune.py --checkpoint checkpoints/best.pt --device cuda

    # Fine-tune d24 with LoRA rank 16
    python train/finetune.py --checkpoint checkpoints/d24_pretrained.pt \\
        --lora-rank 16 --lr 2e-4 --max-steps 3000

    # Resume fine-tuning
    python train/finetune.py --resume checkpoints/finetune_step_1000.pt
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import FlightMindConfig, FlightMind


# ---------------------------------------------------------------------------
# LoRA Implementation
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Replaces a frozen linear layer with:
        output = W_frozen(x) + (alpha/r) * B(A(x))

    The scaling factor alpha/r controls the magnitude of the LoRA update
    relative to the pretrained weights. Higher alpha means stronger
    adaptation; lower means more conservative.

    Initialization:
        A ~ N(0, 1/r)  — small random init
        B = 0           — LoRA output starts at zero (no change to pretrained)

    This zero-init is crucial: at the start of fine-tuning, the model
    behaves exactly like the pretrained model. The LoRA adapters gradually
    learn task-specific modifications during training.
    """

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = original.in_features
        d_out = original.out_features

        # Freeze original weights
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        # LoRA matrices (on same device as original weights)
        device = original.weight.device
        self.lora_A = nn.Parameter(torch.randn(rank, d_in, device=device) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=device))

    def forward(self, x):
        # Original frozen forward pass + low-rank update
        base = self.original(x)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora

    @property
    def weight(self):
        """Merged weight for inference (no LoRA overhead)."""
        return self.original.weight + (self.lora_B @ self.lora_A) * self.scaling


def apply_lora(model: FlightMind, rank: int = 16, alpha: float = 32.0) -> dict:
    """Apply LoRA adapters to all attention and MLP linear layers.

    WHY these specific layers?
    - Attention Q, K, V, O projections: These determine what the model
      attends to. For command parsing, we need it to attend to command
      keywords differently than during pretraining.
    - MLP gate, up, down projections: These determine the model's
      "thinking" at each position. Fine-tuning these helps the model
      learn to produce structured JSON output.

    Returns dict of {layer_name: LoRALinear} for saving/loading adapters.
    """
    lora_layers = {}

    for name, module in model.named_modules():
        # Target: attention projections and MLP projections
        if isinstance(module, nn.Linear) and any(
            target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ):
            # Get parent module and attribute name
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                attr_name = name

            lora_linear = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora_linear)
            lora_layers[name] = lora_linear

    return lora_layers


def get_lora_params(model: FlightMind) -> list:
    """Get only the trainable LoRA parameters."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def save_lora_checkpoint(model, optimizer, config, step, loss, path, lora_rank, lora_alpha):
    """Save only LoRA weights + optimizer state (much smaller than full checkpoint)."""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[name] = {
                "lora_A": module.lora_A.data,
                "lora_B": module.lora_B.data,
            }

    checkpoint = {
        "lora_state": lora_state,
        "optimizer": optimizer.state_dict(),
        "config": {
            "depth": config.depth,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
        },
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "step": step,
        "loss": loss,
    }
    torch.save(checkpoint, str(path))
    size_mb = path.stat().st_size / 1e6
    print(f"  LoRA checkpoint saved: {path.name} ({size_mb:.1f} MB, step {step})", flush=True)


def merge_lora_weights(model: FlightMind):
    """Merge LoRA weights into base model for efficient inference.

    After fine-tuning, we can merge the low-rank updates into the original
    weights: W_merged = W_frozen + (alpha/r) * B @ A

    This eliminates the LoRA overhead at inference time — the model runs
    at the same speed as the original, with the adapted behavior baked in.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.original.weight.add_(
                    (module.lora_B @ module.lora_A) * module.scaling
                )


# ---------------------------------------------------------------------------
# Instruction Data Loading
# ---------------------------------------------------------------------------

class InstructionDataset:
    """Loads instruction-following data from JSONL files.

    Each example is tokenized into a single sequence:
        <|system|>{system}<|end|><|user|>{user}<|end|><|assistant|>{assistant}<|end|>

    The loss mask ensures we only train on the assistant's response,
    not on the system prompt or user input. This teaches the model
    *when* to generate (after <|assistant|>) and *what* to generate
    (structured commands + acknowledgements).
    """

    def __init__(self, data_paths: list[Path], tokenizer, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.examples = []

        for path in data_paths:
            if not path.exists():
                print(f"  Warning: {path} not found, skipping")
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        random.shuffle(self.examples)
        print(f"  Loaded {len(self.examples):,} instruction examples")

    def __len__(self):
        return len(self.examples)

    def format_example(self, ex: dict) -> str:
        """Format a single example into the chat template."""
        return (
            f"<|system|>{ex['system']}<|end|>"
            f"<|user|>{ex['user']}<|end|>"
            f"<|assistant|>{ex['assistant']}<|end|>"
        )

    def get_batch(self, batch_size: int, device: str):
        """Get a batch of tokenized instruction examples.

        Returns:
            input_ids: (batch_size, seq_len) — input tokens
            targets: (batch_size, seq_len) — target tokens (shifted by 1)
            loss_mask: (batch_size, seq_len) — 1.0 for assistant tokens, 0.0 otherwise
        """
        batch_texts = []
        for _ in range(batch_size):
            ex = random.choice(self.examples)
            batch_texts.append(self.format_example(ex))

        # Tokenize
        input_ids_list = []
        targets_list = []
        mask_list = []

        for text in batch_texts:
            encoded = self.tokenizer.encode(text)
            ids = encoded.ids[:self.max_seq_len]

            # Find assistant response start (after <|assistant|>)
            # We mask everything before the assistant's response
            text_before_assistant = text.split("<|assistant|>")[0] + "<|assistant|>"
            assistant_start = len(self.tokenizer.encode(text_before_assistant).ids)

            # Pad or truncate to max_seq_len
            if len(ids) < self.max_seq_len:
                pad_len = self.max_seq_len - len(ids)
                ids = ids + [0] * pad_len  # Pad with 0

            # Create targets (shifted by 1) and loss mask
            input_tokens = ids[:-1]
            target_tokens = ids[1:]

            # Loss mask: only compute loss on assistant tokens
            mask = [0.0] * len(target_tokens)
            for i in range(len(target_tokens)):
                if i >= assistant_start - 1:  # -1 because of the shift
                    mask[i] = 1.0
                if target_tokens[i] == 0:  # Don't compute loss on padding
                    mask[i] = 0.0

            input_ids_list.append(input_tokens)
            targets_list.append(target_tokens)
            mask_list.append(mask)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        targets = torch.tensor(targets_list, dtype=torch.long, device=device)
        loss_mask = torch.tensor(mask_list, dtype=torch.float32, device=device)

        return input_ids, targets, loss_mask


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine LR schedule with warmup (same as pretraining)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def log(msg, logfile=None):
    print(msg, flush=True)
    if logfile:
        logfile.write(msg + "\n")
        logfile.flush()
        os.fsync(logfile.fileno())


def train(args):
    """Main fine-tuning function."""
    log_path = PROJECT_ROOT / "finetune.log"
    logfile = open(log_path, "w", encoding="utf-8")

    log("=" * 60, logfile)
    log("FlightMind Instruction Fine-tuning (LoRA)", logfile)
    log("=" * 60, logfile)

    device = args.device
    dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_bf16_supported() else torch.float32
    log(f"Device: {device}, Dtype: {dtype}", logfile)

    # ---- Load pretrained model ----
    log(f"\nLoading pretrained model: {args.checkpoint}", logfile)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = FlightMindConfig(**checkpoint["config"])
    model = FlightMind(config).to(device)
    model.load_state_dict(checkpoint["model"])
    log(f"{config.describe()}", logfile)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Pretrained parameters: {total_params:,}", logfile)

    # ---- Apply LoRA ----
    log(f"\nApplying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})", logfile)
    lora_layers = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    lora_params = get_lora_params(model)
    n_lora = sum(p.numel() for p in lora_params)
    log(f"LoRA layers: {len(lora_layers)}", logfile)
    log(f"Trainable LoRA params: {n_lora:,} ({n_lora/total_params*100:.2f}% of total)", logfile)

    # ---- Load tokenizer ----
    from tokenizers import Tokenizer
    tokenizer_path = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    log(f"Tokenizer: {tokenizer_path.name} (vocab={tokenizer.get_vocab_size():,})", logfile)

    # ---- Load instruction data ----
    log(f"\nLoading instruction data...", logfile)
    finetune_dir = PROJECT_ROOT / "data" / "finetune"
    data_paths = [
        finetune_dir / "aida_intent_pairs.jsonl",
        finetune_dir / "synthetic_commands.jsonl",
        finetune_dir / "xc_telemetry_pairs.jsonl",
    ]
    dataset = InstructionDataset(data_paths, tokenizer, max_seq_len=args.seq_len)
    log(f"Total examples: {len(dataset):,}", logfile)

    # ---- Optimizer (only LoRA params) ----
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    log(f"\nOptimizer: AdamW (lr={args.lr})", logfile)
    log(f"Optimizer memory: ~{n_lora * 12 / 1e6:.1f} MB (vs {total_params * 12 / 1e6:.0f} MB for full fine-tune)", logfile)

    # ---- Resume from LoRA checkpoint ----
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        log(f"\nResuming from: {resume_path}", logfile)
        resume_ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)

        # Restore LoRA weights
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear) and name in resume_ckpt["lora_state"]:
                module.lora_A.data.copy_(resume_ckpt["lora_state"][name]["lora_A"])
                module.lora_B.data.copy_(resume_ckpt["lora_state"][name]["lora_B"])

        # Restore optimizer state
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])

        start_step = resume_ckpt.get("step", 0)
        best_loss_resumed = resume_ckpt.get("loss", float("inf"))
        log(f"  Restored LoRA weights + optimizer from step {start_step} (loss {best_loss_resumed:.4f})", logfile)
        log(f"  Continuing from step {start_step} to {args.max_steps}", logfile)

    # ---- Training loop ----
    log(f"\nFine-tuning for {args.max_steps:,} steps...", logfile)
    log(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}", logfile)
    log("-" * 60, logfile)

    model.train()
    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if not any(lora_name in name for lora_name in ["lora_A", "lora_B"]):
            param.requires_grad = False

    best_loss = float("inf")
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    t0 = time.time()
    tokens_processed = 0

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(args.grad_accum):
            input_ids, targets, loss_mask = dataset.get_batch(args.batch_size, device)

            with torch.amp.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=dtype != torch.float32):
                logits, _ = model(input_ids, targets=targets)

                # Compute masked loss (only on assistant tokens)
                loss_all = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                )
                loss_all = loss_all.reshape(targets.shape)
                loss = (loss_all * loss_mask).sum() / (loss_mask.sum() + 1e-8)

            loss = loss / args.grad_accum
            loss.backward()
            loss_accum += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        optimizer.step()

        tokens_per_step = args.batch_size * args.grad_accum * (args.seq_len - 1)
        tokens_processed += tokens_per_step

        # ---- Logging ----
        if step % args.log_every == 0 or step == args.max_steps - 1:
            elapsed = time.time() - t0
            tok_s = tokens_processed / elapsed if elapsed > 0 else 0
            log(
                f"step {step:>5d} | loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"{tok_s:,.0f} tok/s",
                logfile,
            )

        # ---- Checkpointing ----
        if step > 0 and step % args.checkpoint_every == 0:
            save_lora_checkpoint(
                model, optimizer, config, step, loss_accum,
                checkpoint_dir / f"finetune_step_{step}.pt",
                args.lora_rank, args.lora_alpha,
            )

            if loss_accum < best_loss:
                best_loss = loss_accum
                save_lora_checkpoint(
                    model, optimizer, config, step, loss_accum,
                    checkpoint_dir / "finetune_best.pt",
                    args.lora_rank, args.lora_alpha,
                )

    # ---- Final save ----
    # Save merged model (LoRA weights baked into base)
    log(f"\nMerging LoRA weights into base model...", logfile)
    merge_lora_weights(model)

    merged_checkpoint = {
        "model": model.state_dict(),
        "config": {
            "depth": config.depth,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
        },
        "step": args.max_steps - 1,
        "loss": loss_accum,
        "finetuned": True,
        "lora_rank": args.lora_rank,
    }
    merged_path = checkpoint_dir / "finetune_merged.pt"
    torch.save(merged_checkpoint, str(merged_path))
    size_mb = merged_path.stat().st_size / 1e6
    log(f"Merged model saved: {merged_path.name} ({size_mb:.0f} MB)", logfile)

    total_time = time.time() - t0
    log(f"\nFine-tuning complete! {args.max_steps} steps in {total_time:.0f}s", logfile)
    log(f"Best loss: {best_loss:.4f}", logfile)
    logfile.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlightMind LoRA Fine-tuning")

    # Model source
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint (.pt)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Max sequence length (shorter than pretraining for efficiency)")

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (8=minimal, 16=default, 32=high capacity)")
    parser.add_argument("--lora-alpha", type=float, default=32.0,
                        help="LoRA scaling factor")

    # Training
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Total fine-tuning steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate (higher than pretraining)")
    parser.add_argument("--min-lr", type=float, default=2e-5,
                        help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="LR warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Logging
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=500)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a LoRA checkpoint (e.g. checkpoints/finetune_step_1000.pt)")

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    train(args)
