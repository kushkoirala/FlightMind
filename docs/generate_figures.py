"""
Generate publication-quality figures for FlightMind documentation.

Parses training logs and project metadata to create:
  1. Training loss curve (train + validation)
  2. Learning rate schedule
  3. Gradient norm stability
  4. Perplexity progression
  5. Architecture scaling (depth parameterization)
  6. Data composition (pretrain + finetune)

Usage:
    python docs/generate_figures.py
"""

import re
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "train": "#2563eb",
    "val": "#dc2626",
    "lr": "#7c3aed",
    "grad": "#059669",
    "ppl": "#ea580c",
    "accent1": "#0891b2",
    "accent2": "#4f46e5",
    "accent3": "#be185d",
}

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

LOG_PATH = Path(__file__).resolve().parents[1] / "train.log"


# ── Parse training log ────────────────────────────────────────────
def parse_train_log(path: Path) -> dict:
    """Extract metrics from train.log."""
    steps, losses, lrs, grad_norms, throughputs = [], [], [], [], []
    val_steps, val_losses = [], []

    step_re = re.compile(
        r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+lr\s+([\d.e+-]+)\s+\|\s+grad_norm\s+([\d.]+)\s+\|\s+([\d,]+)\s+tok/s"
    )
    val_re = re.compile(r"->\s+val_loss:\s+([\d.]+)")

    last_step = 0
    with open(path, "r") as f:
        for line in f:
            m = step_re.search(line)
            if m:
                last_step = int(m.group(1))
                steps.append(last_step)
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                grad_norms.append(float(m.group(4)))
                throughputs.append(int(m.group(5).replace(",", "")))

            m = val_re.search(line)
            if m:
                val_steps.append(last_step)
                val_losses.append(float(m.group(1)))

    return {
        "steps": np.array(steps),
        "losses": np.array(losses),
        "lrs": np.array(lrs),
        "grad_norms": np.array(grad_norms),
        "throughputs": np.array(throughputs),
        "val_steps": np.array(val_steps),
        "val_losses": np.array(val_losses),
    }


# ── Figure 1: Training Loss ──────────────────────────────────────
def plot_training_loss(data: dict):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(data["steps"], data["losses"],
            color=COLORS["train"], alpha=0.4, linewidth=0.8, label="Train loss (raw)")

    # Smoothed train loss (exponential moving average)
    alpha_ema = 0.15
    smoothed = np.zeros_like(data["losses"])
    smoothed[0] = data["losses"][0]
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha_ema * data["losses"][i] + (1 - alpha_ema) * smoothed[i - 1]
    ax.plot(data["steps"], smoothed,
            color=COLORS["train"], linewidth=2.0, label="Train loss (smoothed)")

    # Validation loss
    ax.plot(data["val_steps"], data["val_losses"],
            "o-", color=COLORS["val"], markersize=6, linewidth=1.5, label="Val loss")
    for vs, vl in zip(data["val_steps"], data["val_losses"]):
        ax.annotate(f"{vl:.3f}", (vs, vl), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color=COLORS["val"])

    # Warmup region
    ax.axvspan(0, 200, alpha=0.06, color="orange", label="Warmup (200 steps)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("FlightMind-d8 (50M) — Pretraining Loss")
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(data["steps"][-1], 1000))
    ax.set_ylim(0, 11)

    # Right-side perplexity axis
    ax2 = ax.twinx()
    ax2.set_ylabel("Perplexity", color=COLORS["ppl"])
    ax2.set_ylim(np.exp(0), np.exp(11))
    ax2.set_yscale("log")
    ppl_ticks = [1, 3, 10, 30, 100, 300, 1000, 10000, 50000]
    ax2.set_yticks(ppl_ticks)
    ax2.set_yticklabels([str(t) for t in ppl_ticks])
    ax2.tick_params(axis="y", labelcolor=COLORS["ppl"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COLORS["ppl"])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_loss.png")
    plt.close(fig)
    print(f"  training_loss.png")


# ── Figure 2: Learning Rate Schedule ─────────────────────────────
def plot_lr_schedule(data: dict):
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(data["steps"], data["lrs"] * 1e4,
            color=COLORS["lr"], linewidth=2.0)
    ax.axvspan(0, 200, alpha=0.06, color="orange")
    ax.axhline(y=6.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.6, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.annotate("Peak LR = 6e-4", xy=(200, 6.0), xytext=(300, 6.3),
                fontsize=9, color=COLORS["lr"])
    ax.annotate("Min LR = 6e-5", xy=(5000, 0.6), xytext=(700, 1.2),
                fontsize=9, color="gray")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate (×10⁻⁴)")
    ax.set_title("Cosine Learning Rate Schedule with Warmup")
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "lr_schedule.png")
    plt.close(fig)
    print(f"  lr_schedule.png")


# ── Figure 3: Gradient Norm ──────────────────────────────────────
def plot_gradient_norm(data: dict):
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(data["steps"], data["grad_norms"],
            color=COLORS["grad"], alpha=0.5, linewidth=0.8)

    # Smoothed
    alpha_ema = 0.2
    smoothed = np.zeros_like(data["grad_norms"])
    smoothed[0] = data["grad_norms"][0]
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha_ema * data["grad_norms"][i] + (1 - alpha_ema) * smoothed[i - 1]
    ax.plot(data["steps"], smoothed,
            color=COLORS["grad"], linewidth=2.0, label="Grad norm (EMA)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("Gradient Norm Stability During Training")
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(data["steps"][-1], 1000))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "gradient_norm.png")
    plt.close(fig)
    print(f"  gradient_norm.png")


# ── Figure 4: Architecture Scaling ───────────────────────────────
def plot_architecture_scaling():
    """Plot depth parameterization: model size vs depth."""
    depths = np.array([4, 8, 12, 16, 20, 24, 28, 32])
    vocab = 32768

    # Compute parameters for each depth
    params = []
    for d in depths:
        n_embd = d * 64
        n_layer = d
        # Embedding
        emb = vocab * n_embd
        # Per-layer: attention (4 projections) + MLP (3 matrices) + 2 layer norms
        attn = 4 * n_embd * n_embd  # wq, wk, wv, wo
        mlp = 3 * n_embd * (4 * n_embd)  # w1, w2, w3
        ln = 2 * n_embd  # rms norm weights
        per_layer = attn + mlp + ln
        # Final norm + output head (tied with embedding)
        final_ln = n_embd
        total = emb + n_layer * per_layer + final_ln
        params.append(total / 1e6)  # millions
    params = np.array(params)

    # VRAM estimates (training: ~12 bytes/param, inference bf16: 2 bytes/param)
    vram_train = params * 12 / 1000  # GB
    vram_infer = params * 2 / 1000   # GB

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Parameter count bars
    bars = ax1.bar(depths, params, width=2.5, color=COLORS["accent2"],
                   alpha=0.7, label="Parameters (M)", zorder=3)
    for bar, p, d in zip(bars, params, depths):
        label = f"{p:.0f}M"
        if p >= 1000:
            label = f"{p / 1000:.1f}B"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_xlabel("Depth Parameter (d)")
    ax1.set_ylabel("Parameters (millions)")
    ax1.set_title("FlightMind Depth Parameterization — Single Integer Controls All Dimensions")
    ax1.set_xticks(depths)
    ax1.set_xticklabels([f"d{d}" for d in depths])

    # VRAM overlay
    ax2 = ax1.twinx()
    ax2.plot(depths, vram_train, "s--", color=COLORS["accent3"],
             markersize=5, linewidth=1.5, label="Training VRAM (GB)")
    ax2.plot(depths, vram_infer, "o--", color=COLORS["accent1"],
             markersize=5, linewidth=1.5, label="Inference VRAM (GB)")
    ax2.axhline(y=8, color="red", linestyle=":", linewidth=1, alpha=0.7)
    ax2.annotate("RTX 4060 (8 GB)", xy=(30, 8), fontsize=8, color="red", alpha=0.8)
    ax2.set_ylabel("VRAM (GB)")
    ax2.set_ylim(0, max(vram_train) * 1.15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax2.spines["right"].set_visible(True)

    # Highlight current model
    ax1.annotate("← Current model",
                 xy=(8, params[1]), xytext=(12, params[1] + 200),
                 arrowprops=dict(arrowstyle="->", color=COLORS["train"]),
                 fontsize=9, color=COLORS["train"], fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "architecture_scaling.png")
    plt.close(fig)
    print(f"  architecture_scaling.png")


# ── Figure 5: Data Composition ───────────────────────────────────
def plot_data_composition():
    """Two-panel: pretrain corpus (bar chart) and finetune data (donut)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # ── Left panel: Pretrain corpus as horizontal bar chart ──
    pretrain_labels = ["NTSB Reports", "METAR Weather", "HuggingFace Aviation",
                       "OpenAP Aircraft", "14 CFR Regulations", "FAA Handbooks",
                       "Wikipedia Aviation"]
    pretrain_tokens = [135, 33, 15, 3, 2.7, 2.7, 1.5]  # millions
    pretrain_colors = ["#2563eb", "#0891b2", "#7c3aed", "#be185d", "#059669", "#ea580c", "#6b7280"]

    # Sort by size (largest at top)
    order = np.argsort(pretrain_tokens)
    pretrain_labels = [pretrain_labels[i] for i in order]
    pretrain_tokens = [pretrain_tokens[i] for i in order]
    pretrain_colors = [pretrain_colors[i] for i in order]

    y_pos = np.arange(len(pretrain_labels))
    bars = ax1.barh(y_pos, pretrain_tokens, color=pretrain_colors, edgecolor="white",
                    linewidth=1.5, height=0.65)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(pretrain_labels, fontsize=9)
    ax1.set_xlabel("Tokens (millions)")
    ax1.set_title(f"Pretraining Corpus ({sum(pretrain_tokens):.0f}M tokens)", fontweight="bold")

    # Add value labels on bars
    for bar, val in zip(bars, pretrain_tokens):
        width = bar.get_width()
        label = f"{val:.0f}M" if val >= 1 else f"{val:.1f}M"
        pct = val / sum(pretrain_tokens) * 100
        ax1.text(width + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{label} ({pct:.0f}%)", va="center", fontsize=8, fontweight="bold")

    ax1.set_xlim(0, max(pretrain_tokens) * 1.25)
    ax1.invert_yaxis()

    # ── Right panel: Finetune data as donut ──
    finetune_labels = ["AIDA Intent\nObservations", "Synthetic\nCommands", "XC Flight\nTelemetry"]
    finetune_counts = [244228, 55000, 3945]
    finetune_colors = ["#2563eb", "#7c3aed", "#ea580c"]

    wedges2, texts2, autotexts2 = ax2.pie(
        finetune_counts, labels=finetune_labels,
        autopct=lambda p: f"{p * sum(finetune_counts) / 100:,.0f}",
        colors=finetune_colors, startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
    )
    for t in autotexts2:
        t.set_fontsize(8)
    for t in texts2:
        t.set_fontsize(9)
    ax2.set_title(f"Fine-tuning Dataset\n({sum(finetune_counts):,} instruction pairs)", fontweight="bold")

    fig.suptitle("FlightMind Data Pipeline", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "data_composition.png")
    plt.close(fig)
    print(f"  data_composition.png")


# ── Figure 6: Closed-Loop Architecture Diagram ──────────────────
def plot_closed_loop_diagram():
    """Create the AIDA ↔ FlightMind closed-loop diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_aspect("equal")

    # Boxes
    boxes = {
        "AIDA\nAutonomous\nFlight System":    (1.5, 5.5, "#2563eb"),
        "Flight\nRecordings\n& Observations": (5.0, 5.5, "#059669"),
        "FlightMind\nTraining\nPipeline":      (8.5, 5.5, "#7c3aed"),
        "FlightMind\nLLM\n(Trained)":          (8.5, 1.5, "#ea580c"),
        "Inference\nEngine":                    (5.0, 1.5, "#be185d"),
        "Pilot\nInterface":                     (1.5, 1.5, "#0891b2"),
    }

    box_patches = {}
    for label, (cx, cy, color) in boxes.items():
        w, h = 2.2, 1.6
        rect = plt.Rectangle((cx - w/2, cy - h/2), w, h,
                              facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=2,
                              joinstyle="round", zorder=2)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color, zorder=3)
        box_patches[label] = (cx, cy, w, h)

    # Arrows (from → to)
    arrow_style = dict(arrowstyle="->, head_width=0.15, head_length=0.1",
                       color="#374151", linewidth=1.8)

    # Top row: AIDA → Data → Training
    ax.annotate("", xy=(3.9, 5.5), xytext=(2.6, 5.5),
                arrowprops=arrow_style)
    ax.text(3.25, 5.85, "Generates", ha="center", fontsize=8, style="italic", color="#374151")

    ax.annotate("", xy=(7.4, 5.5), xytext=(6.1, 5.5),
                arrowprops=arrow_style)
    ax.text(6.75, 5.85, "Trains on", ha="center", fontsize=8, style="italic", color="#374151")

    # Right side: Training → Trained model
    ax.annotate("", xy=(8.5, 2.3), xytext=(8.5, 4.7),
                arrowprops=arrow_style)
    ax.text(8.9, 3.5, "Produces", ha="left", fontsize=8, style="italic", color="#374151", rotation=90)

    # Bottom row: Trained → Inference → Pilot → AIDA
    ax.annotate("", xy=(6.1, 1.5), xytext=(7.4, 1.5),
                arrowprops=arrow_style)
    ax.text(6.75, 1.15, "Powers", ha="center", fontsize=8, style="italic", color="#374151")

    ax.annotate("", xy=(2.6, 1.5), xytext=(3.9, 1.5),
                arrowprops=arrow_style)
    ax.text(3.25, 1.15, "Serves", ha="center", fontsize=8, style="italic", color="#374151")

    # Left side: Pilot → AIDA (closing the loop)
    ax.annotate("", xy=(1.5, 4.7), xytext=(1.5, 2.3),
                arrowprops=dict(arrowstyle="->, head_width=0.15, head_length=0.1",
                                color="#dc2626", linewidth=2.5, linestyle="--"))
    ax.text(0.7, 3.5, "Commands\n& feedback", ha="center", fontsize=8,
            style="italic", color="#dc2626", fontweight="bold", rotation=90)

    # Title
    ax.text(5.0, 6.8, "Closed-Loop Training Paradigm", ha="center",
            fontsize=14, fontweight="bold", color="#111827")
    ax.text(5.0, 6.45, "AIDA generates flight data → trains FlightMind → FlightMind powers AIDA → better data",
            ha="center", fontsize=9, color="#6b7280", style="italic")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "closed_loop_architecture.png")
    plt.close(fig)
    print(f"  closed_loop_architecture.png")


# ── Figure 7: Training Metrics Dashboard ─────────────────────────
def plot_metrics_dashboard(data: dict):
    """4-panel dashboard: loss, perplexity, grad norm, throughput."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel 1: Loss
    ax = axes[0, 0]
    ax.plot(data["steps"], data["losses"], color=COLORS["train"], alpha=0.3, linewidth=0.6)
    # Smoothed
    ema = np.zeros_like(data["losses"])
    ema[0] = data["losses"][0]
    for i in range(1, len(ema)):
        ema[i] = 0.15 * data["losses"][i] + 0.85 * ema[i-1]
    ax.plot(data["steps"], ema, color=COLORS["train"], linewidth=2, label="Train (EMA)")
    ax.plot(data["val_steps"], data["val_losses"], "o-", color=COLORS["val"],
            markersize=5, label="Validation")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("(a) Training & Validation Loss")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: Perplexity
    ax = axes[0, 1]
    ppl = np.exp(ema)
    val_ppl = np.exp(data["val_losses"])
    ax.plot(data["steps"], ppl, color=COLORS["ppl"], linewidth=2, label="Train PPL")
    ax.plot(data["val_steps"], val_ppl, "o-", color=COLORS["val"],
            markersize=5, label="Val PPL")
    for vs, vp in zip(data["val_steps"], val_ppl):
        ax.annotate(f"{vp:.1f}", (vs, vp), textcoords="offset points",
                    xytext=(8, 5), fontsize=8, color=COLORS["val"])
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    ax.set_title("(b) Perplexity Progression")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3: Gradient norm
    ax = axes[1, 0]
    ax.plot(data["steps"], data["grad_norms"], color=COLORS["grad"], alpha=0.4, linewidth=0.6)
    gn_ema = np.zeros_like(data["grad_norms"])
    gn_ema[0] = data["grad_norms"][0]
    for i in range(1, len(gn_ema)):
        gn_ema[i] = 0.2 * data["grad_norms"][i] + 0.8 * gn_ema[i-1]
    ax.plot(data["steps"], gn_ema, color=COLORS["grad"], linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("(c) Gradient Norm Stability")

    # Panel 4: Throughput
    ax = axes[1, 1]
    ax.plot(data["steps"], data["throughputs"] / 1000, color=COLORS["accent1"], linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Throughput (K tok/s)")
    ax.set_title("(d) Training Throughput")
    avg_tp = np.mean(data["throughputs"][2:]) / 1000  # skip first couple for warmup
    ax.axhline(y=avg_tp, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.annotate(f"Avg: {avg_tp:.1f}K tok/s", xy=(data["steps"][-1] * 0.5, avg_tp + 0.2),
                fontsize=8, color="gray")

    for ax in axes.flat:
        ax.set_xlim(0, max(data["steps"][-1], 1000))

    fig.suptitle("FlightMind-d8 (50M) — Training Metrics Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_dashboard.png")
    plt.close(fig)
    print(f"  training_dashboard.png")


# ── Main ──────────────────────────────────────────────────────────
def main():
    print(f"Generating figures in {OUT_DIR}/\n")

    # Parse training log
    if LOG_PATH.exists():
        data = parse_train_log(LOG_PATH)
        print(f"Parsed {len(data['steps'])} training steps, {len(data['val_steps'])} val evals")

        plot_training_loss(data)
        plot_lr_schedule(data)
        plot_gradient_norm(data)
        plot_metrics_dashboard(data)
    else:
        print(f"WARNING: {LOG_PATH} not found — skipping training plots")

    # These don't need training data
    plot_architecture_scaling()
    plot_data_composition()
    plot_closed_loop_diagram()

    print(f"\nDone! {len(list(OUT_DIR.glob('*.png')))} figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
