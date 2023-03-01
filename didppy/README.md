# DIDPPy -- Python Interface for DyPDL

DIDPPy is a Python interface for DyPDL, implemented in Rust with PyO3.

## Quick Start

If you just want to use DIDPPy, install it from PyPI.

```bash
pip install didppy
```

There are some examples in [`examples`](https://github.com/domain-independent-dp/didp-rs/didppy/examples).

## Development

If you want to develop DIDPPy, clone this repository.

```bash
git clone https://github.com/domain-independent-dp/didp-rs
cd didp-rs/didppy
```

### Create Python Environment

```bash
python3 -m venv .venv 
source .venv/bin/activate
pip install maturin
```

### Build Development Version

```bash
maturin develop
```

`didppy` will be installed in `.venv`.

### Run Test

```bash
cargo test --no-default-features
pytest
```

## Build Release Version

```bash
maturin build --release
```

This will create the Python wheel. Install the wheel in a Python environment you want to use (this should be different from `.venv`).

```bash
pip install --force-reinstall ../target/wheels/didppy-{x}.whl
```

`{x}` depends on your environment.
