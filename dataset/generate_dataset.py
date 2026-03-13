"""
generate_dataset.py
-------------------
Generates a synthetic dataset of displacement sensor signals
mimicking output from a white-light interferometer.

Three signal classes:
  0 - CLEAN:     Smooth sinusoidal displacement, low noise
  1 - NOISY:     Clean signal with high-frequency Gaussian noise
  2 - DEFECTIVE: Signal with surface defect artifacts (spikes + drift)

Output:
  data/signals.csv   — (N, SIGNAL_LENGTH + 1) array, last column = label
  data/dataset.npz   — numpy archive for fast loading during training
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt

# ── Configuration ──────────────────────────────────────────────────────────────
SIGNAL_LENGTH   = 128       # Samples per signal (power of 2, fits in L1 cache)
SAMPLES_PER_CLASS = 3334    # ~10k total across 3 classes
SAMPLE_RATE     = 1000      # Hz — typical for displacement sensors
SEED            = 42

OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "data")

CLASS_NAMES = {0: "CLEAN", 1: "NOISY", 2: "DEFECTIVE"}


# ── Signal Generators ──────────────────────────────────────────────────────────

def _time_axis():
    """Return time array for one signal."""
    return np.linspace(0, SIGNAL_LENGTH / SAMPLE_RATE, SIGNAL_LENGTH)


def generate_clean(rng: np.random.Generator) -> np.ndarray:
    """
    Clean displacement signal: low-frequency sinusoid with minimal noise.
    Represents a flat, defect-free surface measurement.
    """
    t    = _time_axis()
    freq = rng.uniform(5, 30)          # Dominant displacement frequency (Hz)
    amp  = rng.uniform(0.5, 2.0)       # Amplitude (micrometers)
    phase = rng.uniform(0, 2 * np.pi)

    signal = amp * np.sin(2 * np.pi * freq * t + phase)

    # Add a secondary harmonic (realistic for real sensors)
    signal += rng.uniform(0.05, 0.15) * amp * np.sin(4 * np.pi * freq * t)

    # Very low noise floor
    signal += rng.normal(0, 0.02 * amp, SIGNAL_LENGTH)

    return signal.astype(np.float32)


def generate_noisy(rng: np.random.Generator) -> np.ndarray:
    """
    Noisy signal: clean signal corrupted by high-frequency interference.
    Represents EMI or vibration contamination in the measurement chain.
    """
    signal = generate_clean(rng)
    amp    = np.max(np.abs(signal))

    # High-frequency Gaussian noise
    noise_level = rng.uniform(0.2, 0.5)
    signal += rng.normal(0, noise_level * amp, SIGNAL_LENGTH)

    # Random high-frequency oscillation burst
    t = _time_axis()
    burst_freq  = rng.uniform(200, 400)
    burst_amp   = rng.uniform(0.1, 0.3) * amp
    burst_start = rng.integers(0, SIGNAL_LENGTH // 2)
    burst_len   = rng.integers(10, 30)
    burst = np.zeros(SIGNAL_LENGTH)
    burst[burst_start:burst_start + burst_len] = burst_amp * np.sin(
        2 * np.pi * burst_freq * t[burst_start:burst_start + burst_len]
    )
    signal += burst

    return signal.astype(np.float32)


def generate_defective(rng: np.random.Generator) -> np.ndarray:
    """
    Defective surface signal: clean signal with spike artifacts and drift.
    Represents a surface scratch, pit, or bump in the measurement area.
    """
    signal = generate_clean(rng)
    amp    = np.max(np.abs(signal))

    # Surface defect spikes (1-3 sharp discontinuities)
    n_spikes = rng.integers(1, 4)
    for _ in range(n_spikes):
        pos        = rng.integers(10, SIGNAL_LENGTH - 10)
        spike_amp  = rng.uniform(1.5, 4.0) * amp
        spike_sign = rng.choice([-1, 1])
        width      = rng.integers(1, 4)
        signal[pos:pos + width] += spike_sign * spike_amp

    # Baseline drift (low-frequency tilt — simulates surface slope)
    drift_amp = rng.uniform(0.3, 0.8) * amp
    drift     = np.linspace(0, drift_amp, SIGNAL_LENGTH)
    signal   += drift

    # Moderate noise
    signal += rng.normal(0, 0.05 * amp, SIGNAL_LENGTH)

    return signal.astype(np.float32)


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to [-1, 1] range."""
    max_abs = np.max(np.abs(signal))
    if max_abs < 1e-8:
        return signal
    return signal / max_abs


# ── Dataset Builder ────────────────────────────────────────────────────────────

def build_dataset(n_per_class: int = SAMPLES_PER_CLASS) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced dataset of sensor signals.

    Returns:
        X: (N, SIGNAL_LENGTH) float32 array of signals
        y: (N,) int array of class labels
    """
    rng = np.random.default_rng(SEED)

    generators = [generate_clean, generate_noisy, generate_defective]
    signals, labels = [], []

    for label, gen_fn in enumerate(generators):
        print(f"  Generating {n_per_class} {CLASS_NAMES[label]} signals...")
        for _ in range(n_per_class):
            sig = gen_fn(rng)
            sig = normalize(sig)
            signals.append(sig)
            labels.append(label)

    X = np.array(signals, dtype=np.float32)
    y = np.array(labels,  dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ── Save Outputs ───────────────────────────────────────────────────────────────

def save_dataset(X: np.ndarray, y: np.ndarray) -> None:
    """Save dataset as both .npz and .csv."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # .npz — fast numpy format for training
    npz_path = os.path.join(OUTPUT_DIR, "dataset.npz")
    np.savez(npz_path, X=X, y=y)
    print(f"  Saved: {npz_path}  ({X.shape[0]} samples)")

    # .csv — human readable, first N_CSV rows only
    csv_path = os.path.join(OUTPUT_DIR, "signals_preview.csv")
    header   = ",".join([f"t{i}" for i in range(SIGNAL_LENGTH)]) + ",label"
    preview  = np.column_stack([X[:500], y[:500]])
    np.savetxt(csv_path, preview, delimiter=",", header=header, comments="")
    print(f"  Saved: {csv_path}  (500 sample preview)")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("ENIE — Synthetic Sensor Dataset Generator")
    print("=" * 55)
    print(f"\nConfig:")
    print(f"  Signal length   : {SIGNAL_LENGTH} samples")
    print(f"  Samples/class   : {SAMPLES_PER_CLASS}")
    print(f"  Total samples   : {SAMPLES_PER_CLASS * 3}")
    print(f"  Classes         : {list(CLASS_NAMES.values())}")
    print()

    print("Generating signals...")
    X, y = build_dataset()

    print("\nDataset stats:")
    for cls, name in CLASS_NAMES.items():
        count = np.sum(y == cls)
        print(f"  Class {cls} ({name:10s}): {count} samples")
    print(f"  Signal shape    : {X.shape}")
    print(f"  Value range     : [{X.min():.3f}, {X.max():.3f}]")

    print("\nSaving...")
    save_dataset(X, y)

    print("\nDone! Run visualize.py to inspect the signals.")