# Domain-Independent Dynamic Programming (DIDP)

This repository is a Rust implementation of Dynamic Programming Description Language (DyPDL) and its solvers for Domain-Independent Dynamic Programming (DIDP).

## Packages

- [`dypdl`](./dypdl): a library for DyPDL.
- [`dypdl-heuristic-search`](./dypdl-heuristic-search): a heuristic search algorithm library for DyPDL.
- [`didp-yaml`](./didp-yaml): a YAML interface for DyPDL.
- [`DIDPPy`](./didppy): a Python interface for DyPDL.

## Quick Start

If you want to use DIDP, we recommend using the Python interface, DIDPPy.

## Development

If you want to develop the DyPDL library, solvers, and interfaces, install Rust and clone this repository.

```bash
git clone https://github.com/domain-independent-dp/didp-rs
cd didp-rs
```

### Install Rust

Follow the instruction on the official webpage: <https://www.rust-lang.org/tools/install>

### Run Tests

```
cargo test --no-default-features
```

### Development Environment

- I recommend using VSCode as an editor: <https://code.visualstudio.com/>
- To develop a Rust project with VSCode, this guide may be helpful: <https://code.visualstudio.com/docs/languages/rust>
- I recommend using `clippy` for linting as described here: <https://code.visualstudio.com/docs/languages/rust#_linting>

### Learn Rust

- The official tutorial is very helpful: <https://doc.rust-lang.org/stable/book/>
- If you want to learn by example, see this document: <https://doc.rust-lang.org/stable/rust-by-example/>
