# DyPy -- Python Interface for DyPDL

## Development
```bash
python3 -m venv .venv 
pip install -r docs/requirements.txt
```

### Build Docs
Replace `{x}` with the version of Python.

```bash
sphinx-apidoc -e -f -o ./docs .venv/lib/python{x}/site-packages/dypy
sphinx-build ./docs/ ./docs/_build/
```