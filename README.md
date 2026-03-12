# Embedded Neural Network Inference Engine (ENIE)

A lightweight 1D-CNN inference engine written in pure C for real-time sensor signal classification on ARM Cortex-A53 (Raspberry Pi 3B).

## Overview

This project implements a complete pipeline for deploying a neural network on an embedded ARM processor **without any ML frameworks**. It targets the class of problems found in precision sensor systems — classifying whether a displacement sensor signal is clean, noisy, or indicative of a defective surface.

The inference engine is written entirely in C using fixed-point (int8) arithmetic, static memory allocation, and no external dependencies. Power consumption and latency are benchmarked at float32, int8, and int4 precision levels.

## Pipeline

```
Python (offline)                    C (on-device)
─────────────────                   ──────────────────────────────
Dataset Generator                   Sensor Signal Input
      ↓                                     ↓
  PyTorch 1D-CNN                     Fixed-Point Inference Engine
  Training + Export                        ↓
      ↓                             Signal Classification Output
  Weight Quantizer                         ↓
  (float32 → int8)              Latency + Power Benchmarks
      ↓
  C Header Generator
  (weights as .h files)
```

## Project Structure

```
embedded-neural-inference-engine/
│
├── dataset/                    # Synthetic sensor signal generation
│   ├── generate_dataset.py     # Generates clean/noisy/defective waveforms
│   ├── visualize.py            # Signal visualization and dataset stats
│   └── data/                   # Generated .csv dataset (gitignored)
│
├── model/                      # PyTorch model training
│   ├── train.py                # 1D-CNN training script
│   ├── evaluate.py             # Accuracy, confusion matrix
│   ├── quantize.py             # float32 → int8 post-training quantization
│   ├── export_weights.py       # Export weights as C header files
│   └── checkpoints/            # Saved model weights (gitignored)
│
├── engine/                     # Pure C inference engine
│   ├── include/
│   │   ├── layers.h            # Layer definitions (conv, relu, pool, fc)
│   │   ├── fixed_point.h       # Fixed-point arithmetic macros
│   │   ├── model.h             # Model architecture constants
│   │   └── weights/            # Auto-generated weight headers
│   │       ├── conv1_weights.h
│   │       ├── conv2_weights.h
│   │       └── fc_weights.h
│   ├── src/
│   │   ├── layers.c            # Layer implementations
│   │   ├── inference.c         # Forward pass orchestration
│   │   └── fixed_point.c       # Fixed-point math utilities
│   ├── main.c                  # Entry point: load signal, run inference
│   └── Makefile                # Build system for ARM target
│
├── benchmark/                  # Performance measurement
│   ├── bench_latency.c         # Inference time measurement (clock_gettime)
│   ├── bench_accuracy.c        # Run full test set, report accuracy
│   └── results/                # Benchmark output logs
│
├── scripts/                    # Utility scripts
│   └── deploy.sh               # SCP build + run on Raspberry Pi
│
├── requirements.txt            # Python dependencies
├── README.md
└── .gitignore
```

## Milestones

- [ ] M1: Synthetic dataset generation (3 signal classes, 10k samples)
- [ ] M2: PyTorch 1D-CNN training and evaluation
- [ ] M3: Post-training quantization (float32 → int8) and weight export to C headers
- [ ] M4: Pure C inference engine (conv, relu, pool, fc layers in fixed-point)
- [ ] M5: Integration — run full inference pipeline on Raspberry Pi 3B
- [ ] M6: Benchmarking — latency, memory footprint, accuracy at float32 / int8 / int4
- [ ] M7: GitHub polish and results documentation

## Tech Stack

| Component | Technology |
|---|---|
| Signal generation | Python, NumPy, SciPy |
| Model training | PyTorch |
| Quantization | PyTorch static quantization |
| Inference engine | Pure C (C99), no external libs |
| Arithmetic | Fixed-point int8 (Q7 format) |
| Target hardware | Raspberry Pi 3B (ARM Cortex-A53) |
| Build system | GCC, Makefile |
| Benchmarking | clock_gettime, /proc/stat |

## Benchmark Results

*To be filled after M6.*

| Precision | Accuracy | Latency (ms) | Memory (KB) |
|---|---|---|---|
| float32 | - | - | - |
| int8 | - | - | - |
| int4 | - | - | - |

## Why No ML Frameworks?

Frameworks like TensorFlow Lite or ONNX Runtime abstract away the hardware. This project deliberately avoids them to demonstrate understanding of how neural network inference actually works at the arithmetic level — the kind of knowledge required when deploying models on proprietary sensor hardware with no OS or framework support.