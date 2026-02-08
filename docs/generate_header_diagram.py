"""Generate the AIDA + FlightMind architecture header diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_header_diagram(output_path="docs/figures/aida_flightmind_header.png"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#0a0e1a")

    # Color palette
    bg_dark = "#0a0e1a"
    box_pilot = "#1a3a5c"
    box_fm = "#1e5a3a"
    box_aida = "#5a1e3a"
    box_cessna = "#3a3a1e"
    border_pilot = "#4a8abf"
    border_fm = "#4abf7a"
    border_aida = "#bf4a7a"
    border_cessna = "#bfbf4a"
    text_main = "#e8e8e8"
    text_sub = "#a0a8b8"
    arrow_color = "#5580aa"
    feedback_color = "#aa7755"
    title_color = "#ffffff"

    # --- Title ---
    ax.text(7, 6.55, "AIDA  +  FlightMind", fontsize=22, fontweight="bold",
            color=title_color, ha="center", va="center",
            fontfamily="monospace")
    ax.text(7, 6.15, "Autonomous Flight with a Self-Training Language Model",
            fontsize=11, color=text_sub, ha="center", va="center",
            fontfamily="sans-serif", style="italic")

    # Divider line
    ax.plot([1.5, 12.5], [5.85, 5.85], color="#2a3a4a", linewidth=1.0)

    # --- Main flow boxes ---
    box_y = 4.0
    box_h = 2.0
    box_w = 2.4
    box_r = 0.15

    boxes = [
        # (x_center, label, sublabel1, sublabel2, fill, border)
        (2.0, "Pilot", '"turn heading\nthree six zero"', "", box_pilot, border_pilot),
        (5.2, "FlightMind", "d24  956M params", "BPE → Transformer → JSON", box_fm, border_fm),
        (8.4, "AIDA", "Flight Controller", "PID + Bayesian + CBF", box_aida, border_aida),
        (11.6, "Cessna 172", "X-Plane 12 Sim", "Real-time telemetry", box_cessna, border_cessna),
    ]

    for (cx, label, sub1, sub2, fill, border) in boxes:
        x = cx - box_w / 2
        y = box_y - box_h / 2
        rect = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle=f"round,pad={box_r}",
            facecolor=fill, edgecolor=border, linewidth=1.8,
            alpha=0.95,
        )
        ax.add_patch(rect)

        # Main label
        ax.text(cx, box_y + 0.45, label, fontsize=13, fontweight="bold",
                color=text_main, ha="center", va="center",
                fontfamily="monospace")

        # Sub-labels
        if sub1:
            ax.text(cx, box_y - 0.15, sub1, fontsize=8, color=text_sub,
                    ha="center", va="center", fontfamily="sans-serif")
        if sub2:
            ax.text(cx, box_y - 0.55, sub2, fontsize=7.5, color="#808890",
                    ha="center", va="center", fontfamily="sans-serif")

    # --- Forward arrows ---
    arrow_y = box_y + 0.1
    arrow_kw = dict(
        arrowstyle="->,head_width=0.25,head_length=0.15",
        color=arrow_color, linewidth=2.0,
        connectionstyle="arc3,rad=0",
    )

    for x_start, x_end in [(3.25, 3.95), (6.45, 7.15), (9.65, 10.35)]:
        ax.annotate("", xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
                    arrowprops=arrow_kw)

    # Arrow labels
    ax.text(3.6, arrow_y + 0.35, "voice", fontsize=7, color=text_sub,
            ha="center", va="center", fontfamily="sans-serif")
    ax.text(6.8, arrow_y + 0.35, "~0.1s", fontsize=7, color=text_sub,
            ha="center", va="center", fontfamily="sans-serif")
    ax.text(10.0, arrow_y + 0.35, "controls", fontsize=7, color=text_sub,
            ha="center", va="center", fontfamily="sans-serif")

    # --- Feedback loop (bottom) ---
    fb_y = 1.65
    fb_kw = dict(
        arrowstyle="->,head_width=0.2,head_length=0.12",
        color=feedback_color, linewidth=1.8,
        connectionstyle="arc3,rad=0",
        linestyle="--",
    )

    # Right side down arrow from Cessna
    ax.annotate("", xy=(11.6, fb_y + 0.25), xytext=(11.6, box_y - box_h / 2 - 0.1),
                arrowprops=dict(arrowstyle="-", color=feedback_color,
                                linewidth=1.8, linestyle="--"))

    # Horizontal feedback line
    ax.annotate("", xy=(5.2, fb_y + 0.25), xytext=(11.6, fb_y + 0.25),
                arrowprops=fb_kw)

    # Up arrow to FlightMind
    ax.annotate("", xy=(5.2, box_y - box_h / 2 - 0.1),
                xytext=(5.2, fb_y + 0.25),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.12",
                                color=feedback_color,
                                linewidth=1.8, linestyle="--"))

    # Feedback label
    ax.text(8.4, fb_y - 0.15,
            "closed loop: flight telemetry generates training data",
            fontsize=8.5, color=feedback_color, ha="center", va="center",
            fontfamily="sans-serif", style="italic")

    # --- Spec badges along bottom ---
    badge_y = 0.55
    badges = [
        "4x H100 SXM",
        "175K tok/s",
        "~$506 total",
        "30% aviation / 70% FineWeb-EDU",
        "bfloat16",
    ]
    total_badges = len(badges)
    badge_spacing = 10.0 / total_badges
    badge_start = 2.0

    for i, text in enumerate(badges):
        bx = badge_start + i * badge_spacing
        badge = FancyBboxPatch(
            (bx - len(text) * 0.065, badge_y - 0.22),
            len(text) * 0.13, 0.44,
            boxstyle="round,pad=0.08",
            facecolor="#141e2e", edgecolor="#2a3a5a", linewidth=0.8,
        )
        ax.add_patch(badge)
        ax.text(bx, badge_y, text, fontsize=7, color="#7090b0",
                ha="center", va="center", fontfamily="monospace")

    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, facecolor=bg_dark, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    draw_header_diagram()
