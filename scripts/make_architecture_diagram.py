"""Generate architecture diagram PNG for the handover package."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "resource material" / "application_screenshots" / "00_architecture_diagram.png"


def box(ax, x, y, w, h, text, color="#ffffff", edge="#1d1d1f"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="#1d1d1f", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#3a3a3c",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=160)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title(
        "AgenticRLHF V0.5 — Architecture (Live Loop vs Offline Training)",
        fontsize=13,
        pad=12,
        color="#1d1d1f",
    )

    # Live path (top)
    box(ax, 0.3, 5.4, 1.8, 0.9, "User Prompt\n(scenario / custom)", "#e8f1fb", "#0071e3")
    box(ax, 2.5, 5.4, 1.9, 0.9, "Target LLM\nOllama qwen2.5:1.5b\n(no tools)", "#fff7e8", "#9a6700")
    box(ax, 4.8, 5.4, 2.2, 0.9, "Supervisory Agent\nOllama judge\n(use_tool | finish)", "#eaf8f0", "#0b6e4f")
    box(ax, 7.4, 5.4, 2.0, 0.9, "Tools\nrubric / retrieve /\nanalyze_behavior…", "#f5f5f7", "#1d1d1f")
    box(ax, 9.8, 5.4, 2.2, 0.9, "Feedback → Revise\n→ Verify", "#eaf8f0", "#0b6e4f")

    arrow(ax, 2.1, 5.85, 2.5, 5.85)
    arrow(ax, 4.4, 5.85, 4.8, 5.85)
    arrow(ax, 7.0, 5.85, 7.4, 5.85)
    arrow(ax, 9.4, 5.85, 9.8, 5.85)

    # Supporting stores
    box(ax, 4.8, 3.7, 2.2, 1.0, "Chroma + nomic-embed\nSafety principles &\nexamples", "#e8f1fb", "#0071e3")
    box(ax, 7.4, 3.7, 2.0, 1.0, "SQLite Memory\nevals / prefs /\nconversation", "#e8f1fb", "#0071e3")
    arrow(ax, 5.9, 5.4, 5.9, 4.7)
    arrow(ax, 8.4, 5.4, 8.4, 4.7)

    # Offline training
    box(ax, 0.3, 1.5, 2.4, 1.1, "Preference pairs\n(rejected=original,\nchosen=revised)\n+ gates/quarantine", "#fff7e8", "#9a6700")
    box(ax, 3.2, 1.5, 2.6, 1.1, "DPO + LoRA (TRL/PEFT)\nHF SmolLM2-135M\nCPU training path", "#feeceb", "#b42318")
    box(ax, 6.3, 1.5, 2.4, 1.1, "Adapter checkpoint\ncheckpoints/dpo_lora/\nfinal_adapter", "#f5f5f7", "#1d1d1f")
    box(ax, 9.2, 1.5, 2.8, 1.1, "Evaluation tab\nbaseline vs adapter\n(HF path only)", "#eaf8f0", "#0b6e4f")

    arrow(ax, 10.9, 5.4, 10.9, 4.2)
    arrow(ax, 10.9, 4.2, 1.5, 2.6)
    arrow(ax, 2.7, 2.05, 3.2, 2.05)
    arrow(ax, 5.8, 2.05, 6.3, 2.05)
    arrow(ax, 8.7, 2.05, 9.2, 2.05)

    ax.text(
        0.3,
        6.7,
        "LIVE INFERENCE CYCLE — weights of the playground target are NOT updated",
        fontsize=9,
        color="#0b6e4f",
        fontweight="bold",
    )
    ax.text(
        0.3,
        2.9,
        "OFFLINE TRAINING PATH — separate from the live agentic cycle; HF adapter cannot load into Ollama GGUF",
        fontsize=9,
        color="#b42318",
        fontweight="bold",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
