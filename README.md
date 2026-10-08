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

Tune, train, and evaluate both models on MNIST:

```bash
uv run python src/main.py                       # 20 trials per model, 1 epoch per run
uv run python src/main.py --trials 50 --epochs 3
```

For each model, an [Optuna](https://optuna.org) study (TPE sampler) searches hyperparameters by accuracy on a 5,000-example validation split held out from the training set. The best configuration is then retrained on the full training set and evaluated once on the test set, which is never seen during tuning. The script prints the best validation accuracy and params, then parameter count, training time, and test accuracy, for the MLP and KAN in sequence.

| Model | Search space |
|-------|--------------|
| Both  | `lr` ∈ [1e-4, 1e-2] (log), `batch_size` ∈ {32, 64, 128} |
| MLP   | `hidden_size` ∈ {64, 128, 256, 512} |
| KAN   | `hidden_size` ∈ {5, 10, 20}, `grid` ∈ {3, 5, 8}, `k` ∈ {2, 3} | The MLP uses the best available device (CUDA → MPS → CPU); the KAN runs on CPU.

MNIST data is expected in `src/data/`. The script does not download it automatically (`download=False`).

Edit the search spaces in `build_mlp`, `build_tuned_kan`, and `suggest_training` in `src/main.py`.