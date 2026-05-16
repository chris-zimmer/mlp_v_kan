# MLP vs KAN

A comparison of **Multi-Layer Perceptrons (MLPs)** and **Kolmogorov-Arnold Networks (KANs)** on standard benchmarks.

## Project Structure

```
src/
  main.py              # Entry point (MPS/GPU availability check)
  models/
    mlp.py             # MLP model with MNIST training loop
    kan.py             # KAN model stub (pykan)
notebooks/
  kan_ex.ipynb         # KAN experimentation notebook
```

## Current Status

Both models train end-to-end on MNIST via `python src/main.py`, which loads each model from `src/models/` and prints per-model parameter counts, training time, and test accuracy.

- **MLP**: 2-layer (128 hidden, ReLU), trained with Adam + cross-entropy on full 28×28 MNIST. ~97% test accuracy after 1 epoch.
- **KAN**: `width=[64, 10, 10]` (grid=5, k=3) via [pykan](https://github.com/KindXiaoming/pykan), trained with LBFGS + cross-entropy on 8×8 downsampled MNIST (1000 train / 1000 test). ~81% test accuracy after 20 steps. Pinned to CPU because pykan's grid ops aren't reliably MPS-compatible.

## Setup

Requires Python 3.9+. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

PyTorch is installed platform-specifically:
- **macOS**: CPU build
- **Linux/Windows**: CUDA 12.8 build

## Usage

Train the MLP on MNIST:

```bash
python src/models/mlp.py
```

MNIST data is expected in `./data/`. The training script does not download it automatically (`download=False`).

## Development

```bash
ruff check .          # lint
ruff format .         # format
mypy src/             # type check
pytest                # run tests
```
