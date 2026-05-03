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

- **MLP**: Working end-to-end on MNIST (2-layer, 128 hidden units, ReLU, trained with Adam).
- **KAN**: Skeletal — model instantiated via [pykan](https://github.com/KindXiaoming/pykan) but not yet connected to a dataset or training loop.

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
