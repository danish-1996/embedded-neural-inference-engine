"""
evaluate.py
-----------
Evaluates the trained model on the test set.
Produces:
  - Per-class accuracy
  - Confusion matrix
  - Saves confusion matrix plot to checkpoints/confusion_matrix.png
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Reuse dataset/split logic from train.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train import SensorDataset, SensorCNN, SEED, TRAIN_SPLIT, VAL_SPLIT, BATCH_SIZE

DATA_PATH      = os.path.join(os.path.dirname(__file__), "..", "dataset", "data", "dataset.npz")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CLASS_NAMES    = ["CLEAN", "NOISY", "DEFECTIVE"]


def plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0f1117")
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, color="white")
    ax.set_yticklabels(CLASS_NAMES, color="white")
    ax.set_xlabel("Predicted",  color="white")
    ax.set_ylabel("Actual",     color="white")
    ax.set_title("Confusion Matrix", color="white", fontweight="bold")
    ax.set_facecolor("#1c2333")
    ax.tick_params(colors="white")

    total = cm.sum(axis=1, keepdims=True)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            pct  = cm[i, j] / total[i, 0] * 100
            text = f"{cm[i,j]}\n({pct:.1f}%)"
            color = "white" if cm[i, j] < cm.max() * 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=9)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = os.path.join(CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"Confusion matrix saved to: {out}")
    plt.show()


def main():
    print("=" * 55)
    print("ENIE — Model Evaluation")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test split (must match train.py split logic exactly)
    data    = np.load(DATA_PATH)
    dataset = SensorDataset(data["X"], data["y"])
    n_total = len(dataset)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val   = int(n_total * VAL_SPLIT)
    n_test  = n_total - n_train - n_val

    _, _, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load best model
    model_path = os.path.join(CHECKPOINT_DIR, "model_float32.pt")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run train.py first.")
        return

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Collect predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model(X).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    overall = (all_preds == all_labels).mean()
    print(f"\nOverall Test Accuracy: {overall:.1%}")

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        acc  = (all_preds[mask] == all_labels[mask]).mean()
        print(f"  {name:12s}: {acc:.1%}  ({mask.sum()} samples)")

    # Confusion matrix
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    header = f"{'':12s}" + "".join(f"{n:>12s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:12s}" + "".join(f"{cm[i,j]:>12d}" for j in range(n))
        print(row)

    plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()