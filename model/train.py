"""
train.py
--------
Trains a compact 1D-CNN to classify sensor signals into three classes:
  0 - CLEAN
  1 - NOISY
  2 - DEFECTIVE

Architecture is deliberately small — designed to fit in the L1/L2 cache
of an ARM Cortex-A53 and be exportable to a C inference engine.

Output:
  checkpoints/model_float32.pt   — full precision trained model
  checkpoints/model_state.pt     — state dict only (for quantization)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ── Configuration ──────────────────────────────────────────────────────────────
SIGNAL_LENGTH  = 128
N_CLASSES      = 3
BATCH_SIZE     = 64
EPOCHS         = 40
LEARNING_RATE  = 1e-3
TRAIN_SPLIT    = 0.8
VAL_SPLIT      = 0.1
# TEST_SPLIT   = 0.1

SEED           = 42
DATA_PATH      = os.path.join(os.path.dirname(__file__), "..", "dataset", "data", "dataset.npz")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CLASS_NAMES    = ["CLEAN", "NOISY", "DEFECTIVE"]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Dataset ────────────────────────────────────────────────────────────────────

class SensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Shape: (N, 1, L) — add channel dim for Conv1d
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    data    = np.load(DATA_PATH)
    X, y    = data["X"], data["y"]
    dataset = SensorDataset(X, y)

    n_total = len(dataset)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val   = int(n_total * VAL_SPLIT)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ── Model ──────────────────────────────────────────────────────────────────────

class SensorCNN(nn.Module):
    """
    Compact 1D-CNN for sensor signal classification.

    Architecture (designed for C export):
      Conv1 : 1  → 8  channels, kernel=7, stride=1, padding=3
      ReLU
      MaxPool: kernel=2 (L: 128 → 64)

      Conv2 : 8  → 16 channels, kernel=5, stride=1, padding=2
      ReLU
      MaxPool: kernel=2 (L: 64 → 32)

      Conv3 : 16 → 16 channels, kernel=3, stride=1, padding=1
      ReLU
      MaxPool: kernel=2 (L: 32 → 16)

      Flatten → FC1: 256 → 32 → ReLU
      FC2: 32 → 3 (logits)

    Total parameters: ~7,500 — fits entirely in 30KB of RAM.
    """
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(1,  8,  kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(8,  16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("ENIE — 1D-CNN Training")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = load_data()

    # Build model
    model = SensorCNN().to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Architecture:\n{model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    best_val_acc  = 0.0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_loss:>8.4f} | {val_acc:>6.1%}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, os.path.join(CHECKPOINT_DIR, "model_float32.pt"))
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_state.pt"))

    # Final test evaluation
    print("\n" + "=" * 55)
    print("Loading best model for test evaluation...")
    model = torch.load(os.path.join(CHECKPOINT_DIR, "model_float32.pt"),
                       map_location=device, weights_only=False)
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Accuracy : {test_acc:.1%}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Best Val Acc  : {best_val_acc:.1%}")
    print(f"\nCheckpoints saved to: {os.path.abspath(CHECKPOINT_DIR)}")
    print("\nNext step: run evaluate.py for confusion matrix.")


if __name__ == "__main__":
    main()