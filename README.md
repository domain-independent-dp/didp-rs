# Domain-Independent Dynamic Programming (DIDP)

## Packages

- [`dypdl`](./dypdl): a library for DyPDL.
- [`dypdl-heuristic-search`](./dypdl-heuristic-search): a heuristic algorithm library for DyPDL.
- [`didp-yaml`](./didp-yaml): a YAML interface for DyPDL.
- [`didppy`](./didppy): a Python interface for DyPDL.

## Build

### Install Rust

Follow the instruction on the official webpage: <https://www.rust-lang.org/tools/install>

For Linux or Mac OS, you will just run the following command.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build

Clone this repository and run `cargo build`.

```bash
git clone https://github.com/Kurorororo/didp-rs
cd didp-rs
cargo build --release
```

For debug build,

```bash
cargo build
```

To run tests,

```bash
cargo test --no-default-features
```

## Run a Solver from the YAML interface

Run a solver using the YAML interface following [this document](./didp-yaml/README.md).

## Develop DIDP

### Development Environment

- I recommend using VSCode as an editor: <https://code.visualstudio.com/>
- To develop a Rust project with VSCode, this guide may be helpful: <https://code.visualstudio.com/docs/languages/rust>
- I recommend using `clippy` for linting as described here: <https://code.visualstudio.com/docs/languages/rust#_linting>

### Learn Rust

- The official tutorial is very helpful: <https://doc.rust-lang.org/stable/book/>
- If you want to learn by example, see this document: <https://doc.rust-lang.org/stable/rust-by-example/>
