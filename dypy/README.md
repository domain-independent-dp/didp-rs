# DyPy -- Python Interface for DyPDL

## Development

### Python Environment
```bash
python3 -m venv .venv 
source .venv/bin/activate
pip install maturin
pip install -r docs/requirements.txt
```

### Build Development Version
```bash
maturin develop
```
`dypy` will be installed in `.venv`.

### Build Docs
Replace `{x}` with the version of Python.

```bash
sphinx-apidoc -e -f -o ./docs .venv/lib/python{x}/site-packages/dypy
sphinx-build ./docs/ ./docs/_build/
```

This will generates the API reference to `./docs/_build/index.html`.

## Release
```bash
maturin build --release
```

This will creates the Python wheel. Install the wheel in a Python environment you want to use (this may be different from `.venv`).

```bash
pip install --force-reinstall ../target/wheels/dypy-{x}.whl
```

`{x}` depends on your environment.


