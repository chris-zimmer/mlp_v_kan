# MLP vs KAN

A comparison of **Multi-Layer Perceptrons (MLPs)** and **Kolmogorov-Arnold Networks (KANs)**.

## Project Structure

```
src/
  main.py                # Entry point: loads config, tunes, trains, and evaluates MLP and KAN on the configured dataset
  config.yaml            # Dataset choice, paths, validation size, and random seed
  data/
    loaders.py           # MNIST, Fashion-MNIST, and Covertype loading; train/validation split
  models/
    mlp.py               # MLP model definition
    kan.py               # KAN builder (pykan)
  training/
    trainer.py           # Device selection, training loop, prediction, accuracy
  tuning/
    search_spaces.py     # Optuna search spaces and model builders
    study.py             # Optuna hyperparameter search
  evaluation/
    report.py            # Final retrain, test metrics, and PDF report
data/                    # Downloaded datasets (gitignored)
notebooks/
  kan_ex.ipynb           # KAN experimentation notebook
```

## Current Status

Both models are tuned, trained, and evaluated end-to-end on MNIST via `uv run python src/main.py` (see [Usage](#usage)). Both go through the same path — full 28×28 MNIST, Adam + cross-entropy, the same tuning budget and search over learning rate and batch size — so the comparison is like-for-like.

The figures below are a pre-tuning baseline (batch size 64, Adam's default learning rate, 1 epoch) and have not yet been updated with tuned results:

- **MLP**: 2-layer (128 hidden, ReLU). 101,770 params, ~96% test accuracy, ~4s to train.
- **KAN**: `width=[784, 10, 10]` (grid=5, k=3) via [pykan](https://github.com/KindXiaoming/pykan). 136,648 params, ~92% test accuracy, ~10s to train. pykan's symbolic branch and activation caching are turned off (`symbolic_enabled=False`, `save_act=False`); they serve plotting and pruning, and leaving them on costs ~18× per training step. Pinned to CPU, which benchmarks slightly faster than MPS at this size.

At matched input size the KAN is the larger model, takes about twice the wall time, and scores a few points lower.

Note that pykan's default `grid_range=[-1, 1]` does not cover normalized MNIST (~[-0.42, 2.82]), so most inputs fall outside the spline grid; widening the range or calling `update_grid_from_samples` would likely improve the KAN's accuracy.

## Setup

Uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. Once that is installed on your machine, simply run the following command at the root of this repository:

```bash
uv sync
```

PyTorch is installed platform-specifically:
- **macOS**: CPU build
- **Linux/Windows**: CUDA 12.8 build

## Usage

Tune, train, and evaluate both models on the dataset set by `dataset` in `src/config.yaml`:

```bash
uv run python src/main.py                       # 20 trials per model, 1 epoch per run
uv run python src/main.py --trials 50 --epochs 3
```

For each model, an [Optuna](https://optuna.org) study (TPE sampler) searches hyperparameters by accuracy on a 5,000-example validation split held out from the training set. The best configuration is then retrained on the full training set and evaluated once on the test set, which is never seen during tuning. The script prints the best validation accuracy and params, then parameter count, training time, test accuracy, and a per-class classification report (precision, recall, F1), for the MLP and KAN in sequence. If `save_results` is `true` in `src/config.yaml`, each run also saves a PDF report to `results/<date>_<dataset>_<short uuid>.pdf` (e.g. `results/2026-10-13_mnist_3f9a1c2e.pdf`, gitignored): a summary page comparing test accuracy, parameter count, and training time, then one page per model with its hyperparameters, classification report, and test-set confusion matrix.

| Model | Search space |
|-------|--------------|
| Both  | `lr` ∈ [1e-4, 1e-2] (log), `batch_size` ∈ {32, 64, 128} |
| MLP   | `hidden_size` ∈ {64, 128, 256, 512} |
| KAN   | `hidden_size` ∈ {5, 10, 20}, `grid` ∈ {3, 5, 8}, `k` ∈ {2, 3} |

Edit the search spaces in `build_mlp`, `build_tuned_kan`, and `suggest_training` in `src/tuning/search_spaces.py`.

The MLP uses the best available device (CUDA → MPS → CPU); the KAN runs on CPU. Expect tuning to take several minutes: each KAN run takes ~10s or more per epoch, and the default budget is 20 trials per model.

| `dataset` | Data | Inputs | Classes | Train / test |
|-----------|------|--------|---------|--------------|
| `mnist` | 28×28 handwritten digits | 784 | 10 | 60,000 / 10,000 |
| `fashion_mnist` | 28×28 clothing images | 784 | 10 | 60,000 / 10,000 |
| `covtype` | [Covertype](https://archive.ics.uci.edu/dataset/31/covertype) forest cover (tabular) | 54 | 7 | 464,809 / 116,203 |

Datasets are downloaded into `data/` at the repository root on first use. Covertype has no official test split, so 20% is held out (stratified, seeded by `seed`), and its 10 continuous features are standardized using training-set statistics; the 44 one-hot features are left as is.

### Why Optuna

- **Adaptive search.** The TPE sampler uses earlier trials to choose the next settings, which wastes fewer runs than grid or random search. This matters for the KAN, where each run is slow.
- **Search spaces in plain Python.** Each model's space is a small function, and the same function rebuilds the best model for the final run (via `optuna.trial.FixedTrial`).
- **Works with a plain PyTorch loop.** No trainer framework or model wrappers are needed.
- **Lightweight.** No server or cluster setup.

Alternatives considered: scikit-learn's search classes (would need the PyTorch models wrapped, and offer only grid/random search), Ray Tune (built for distributed runs, heavier than needed here), and Hyperopt (similar algorithm, clunkier API, less actively maintained). With small budgets, TPE's edge over random search is modest, since its first 10 trials are random.
