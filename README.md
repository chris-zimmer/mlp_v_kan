# MLP vs KAN

A comparison of **Multi-Layer Perceptrons (MLPs)** and **Kolmogorov-Arnold Networks (KANs)** on standard benchmarks.

## Project Structure

```
src/
  main.py              # Entry point: trains MLP and KAN on MNIST, prints per-model results
  data/                # MNIST data (gitignored)
  models/
    mlp.py             # MLP model definition
    kan.py             # KAN builder (pykan)
notebooks/
  kan_ex.ipynb         # KAN experimentation notebook
```

## Current Status

Both models train end-to-end on MNIST via `uv run python src/main.py`, which loads each model from `src/models/` and prints per-model parameter counts, training time, and test accuracy. Both go through the same `train_and_report` path — full 28×28 MNIST, Adam + cross-entropy, batch size 64, 1 epoch — so the comparison is like-for-like.

- **MLP**: 2-layer (128 hidden, ReLU). 101,770 params, ~96% test accuracy, ~4s to train.
- **KAN**: `width=[784, 10, 10]` (grid=5, k=3) via [pykan](https://github.com/KindXiaoming/pykan). 136,648 params, ~92% test accuracy, ~10s to train. pykan's symbolic branch and activation caching are turned off (`symbolic_enabled=False`, `save_act=False`); they serve plotting and pruning, and leaving them on costs ~18× per training step. Pinned to CPU, which benchmarks slightly faster than MPS at this size.

At matched input size the KAN is the larger model, takes about twice the wall time, and scores a few points lower.

Note that pykan's default `grid_range=[-1, 1]` does not cover normalized MNIST (~[-0.42, 2.82]), so most inputs fall outside the spline grid; widening the range or calling `update_grid_from_samples` would likely improve the KAN's accuracy.

## Setup

Requires Python 3.9+. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

PyTorch is installed platform-specifically:
- **macOS**: CPU build
- **Linux/Windows**: CUDA 12.8 build

## Usage

Train and evaluate both models on MNIST:

```bash
uv run python src/main.py
```

This prints parameter counts, training time, and test accuracy for the MLP and KAN in sequence. The MLP uses the best available device (CUDA → MPS → CPU); the KAN runs on CPU.

MNIST data is expected in `src/data/`. The script does not download it automatically (`download=False`).

Tune the KAN's capacity via `build_kan`'s `hidden_size`, `grid`, and `k` arguments in `main`, and the training length of either model via the `epochs` argument to `train_and_report`.