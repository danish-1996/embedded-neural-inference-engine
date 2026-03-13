"""
quantize.py
-----------
Reports float32 model size and estimates int8 size reduction.
Actual int8 quantization is handled manually in export_weights.py
using symmetric per-tensor Q7 quantization.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import SensorCNN, SEED, TRAIN_SPLIT, VAL_SPLIT, BATCH_SIZE, SensorDataset
from torch.utils.data import DataLoader, random_split

DATA_PATH      = os.path.join(os.path.dirname(__file__), "..", "dataset", "data", "dataset.npz")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


@torch.no_grad()
def evaluate_accuracy(model, loader) -> float:
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        preds    = model(X).argmax(1)
        correct += (preds == y).sum().item()
        total   += len(y)
    return correct / total


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 55)
    print("ENIE — Quantization Report (float32 → int8 estimate)")
    print("=" * 55)

    # Load test split
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

    # Load model
    model_path = os.path.join(CHECKPOINT_DIR, "model_float32.pt")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    acc        = evaluate_accuracy(model, test_loader)
    n_params   = count_parameters(model)
    float_kb   = (n_params * 4) / 1024   # 4 bytes per float32
    int8_kb    = (n_params * 1) / 1024   # 1 byte per int8

    print(f"\nModel parameters  : {n_params:,}")
    print(f"\nFloat32:")
    print(f"  Accuracy        : {acc:.2%}")
    print(f"  Weight memory   : {float_kb:.1f} KB")
    print(f"\nInt8 (estimated):")
    print(f"  Accuracy        : ~{acc:.2%} (verified after export)")
    print(f"  Weight memory   : {int8_kb:.1f} KB")
    print(f"  Size reduction  : {(1 - int8_kb/float_kb)*100:.0f}%")

    # Save report
    report = (
        f"ENIE Quantization Report\n"
        f"{'='*40}\n"
        f"Parameters       : {n_params:,}\n"
        f"Float32 Accuracy : {acc:.4%}\n"
        f"Float32 Size     : {float_kb:.2f} KB\n"
        f"Int8 Size (est.) : {int8_kb:.2f} KB\n"
        f"Size Reduction   : {(1 - int8_kb/float_kb)*100:.0f}%\n"
    )
    report_path = os.path.join(CHECKPOINT_DIR, "quant_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    print("\nNext step: run export_weights.py to generate C headers.")


if __name__ == "__main__":
    main()