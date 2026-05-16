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

Train and evaluate both models on MNIST:

```bash
python src/main.py
```

This prints parameter counts, training time, and test accuracy for the MLP and KAN in sequence. The MLP uses the best available device (CUDA → MPS → CPU); the KAN runs on CPU.

MNIST data is expected in `src/data/`. The script does not download it automatically (`download=False`).

Tune the KAN's training cost via the constants at the top of `src/main.py` (`KAN_INPUT_SIZE`, `KAN_TRAIN_SAMPLES`, `KAN_TEST_SAMPLES`) and the `steps` argument to `run_kan`.

## Development

```bash
ruff check .          # lint
ruff format .         # format
mypy src/             # type check
pytest                # run tests
```
