# Domain-Independent Dynamic Programming

## Build

### Install Rust
Follow the instruction on the official webpage: https://www.rust-lang.org/tools/install

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
cargo test
```

## Run DIDP Solver
Run the following command.

```bash
cargo run --release domain.yaml problem.yaml config.yaml
```

Here, `domain.yaml` is the domain file for the problem, `problem.yaml` is the problem file for the problem, and `config.yaml` is the file specifying a search algorithm.

Instead of `cargo run`, you can directly use the binary.

```bash
./target/release/didp-search domain.yaml problem.yaml config.yaml
```

There are some examples in `./examples`. For example, you can test DIDP on TSPTW, CVRP, SALBP-1, bin packing, MOSP, and graph clear.

```
cargo run --release examples/tsptw-domain.yaml examples/tsptw-Dumas-n150w60.001.yaml examples/blind-astar.yaml
cargo run --release examples/cvrp-domain.yaml examples/cvrp-E-n13-k4.yaml examples/blind-astar.yaml
cargo run --release examples/salbp-1-domain.yaml examples/salbp-1-very-large-1.yaml examples/salbp-1-astar.yaml
cargo run --release examples/bin-packing-domain.yaml examples/bin-packing-Falkenauer_T_t60_00.yaml examples/bin-packing-astar.yaml
cargo run --release examples/mosp-domain.yaml examples/mosp-SP1.yaml examples/blind-astar.yaml
cargo run --release examples/graph-clear-domain.yaml examples/graph-clear-planar20-1.yaml examples/blind-astar.yaml
```

## Model Problems
Please read [the user guide](./user-guide.md) to learn how to model a problem.

## Develop DIDP

### Development Environment
I recommend using VSCode as an editor: https://code.visualstudio.com/

To develop a Rust project with VSCode, this guide may be helpful: https://code.visualstudio.com/docs/languages/rust

I recommend using `clippy` for linting as described here: https://code.visualstudio.com/docs/languages/rust#_linting


### Learn Rust
The official tutorial is very helpful: https://doc.rust-lang.org/stable/book/

If you want to learn by example, see this document: https://doc.rust-lang.org/stable/rust-by-example/


