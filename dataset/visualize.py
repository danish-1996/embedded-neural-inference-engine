"""
visualize.py
------------
Visualizes sample signals from each class and plots dataset statistics.
Run after generate_dataset.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "dataset.npz")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "signals_plot.png")
CLASS_NAMES = {0: "CLEAN", 1: "NOISY", 2: "DEFECTIVE"}
COLORS      = {0: "#3ecf8e", 1: "#f6c90e", 2: "#f66151"}


def plot_dataset(n_examples: int = 3) -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}")
        print("Run generate_dataset.py first.")
        return

    data = np.load(DATA_PATH)
    X, y = data["X"], data["y"]

    fig = plt.figure(figsize=(14, 8), facecolor="#0f1117")
    fig.suptitle("ENIE — Synthetic Sensor Signal Dataset",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(len(CLASS_NAMES), n_examples + 1,
                           figure=fig, width_ratios=[1.2] + [1] * n_examples)

    for cls, name in CLASS_NAMES.items():
        cls_signals = X[y == cls]
        color = COLORS[cls]

        # Left: overlay plot of 50 signals (shows distribution)
        ax_overlay = fig.add_subplot(gs[cls, 0])
        for sig in cls_signals[:50]:
            ax_overlay.plot(sig, color=color, alpha=0.1, linewidth=0.5)
        ax_overlay.plot(cls_signals[0], color=color, linewidth=1.5, label="Sample 1")
        ax_overlay.set_facecolor("#1c2333")
        ax_overlay.set_title(f"Class {cls}: {name}\n(50 overlaid)",
                              color="white", fontsize=9)
        ax_overlay.tick_params(colors="gray", labelsize=7)
        for spine in ax_overlay.spines.values():
            spine.set_edgecolor("#2a3244")
        ax_overlay.set_ylabel("Amplitude", color="gray", fontsize=7)

        # Right: individual example signals
        for i in range(n_examples):
            ax = fig.add_subplot(gs[cls, i + 1])
            ax.plot(cls_signals[i], color=color, linewidth=1.2)
            ax.set_facecolor("#1c2333")
            ax.set_title(f"Example {i+1}", color="white", fontsize=8)
            ax.tick_params(colors="gray", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a3244")
            ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"Plot saved to: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    plot_dataset()