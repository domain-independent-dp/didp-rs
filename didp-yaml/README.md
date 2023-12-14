[![crates.io](https://img.shields.io/crates/v/didp-yaml)](https://crates.io/crates/didp-yaml)
[![minimum rustc 1.65](https://img.shields.io/badge/rustc-1.65+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# didp-yaml

didp-yaml is a YAML interface for DyPDL modeling and solvers.

## Documents

For the syntax of the DyPDL-YAML and the solver configuration, see the [user guide](https://github.com/domain-independent-dp/didp-rs/tree/main/didp-yaml/docs)

## Installation

First, install Rust following the instruction on the official webpage: <https://www.rust-lang.org/tools/install>

Next, install `didp-yaml`.

```bash
cargo install didp-yaml
```

## Run the Solver

```bash
didp-yaml domain.yaml problem.yaml config.yaml
```

Here, `domain.yaml` is the domain file for the problem, `problem.yaml` is the problem file for the problem, and `config.yaml` is the configuration file for a solver.

There are some examples in [`examples`](https://github.com/domain-independent-dp/didp-rs/tree/main/didp-yaml/examples). For example, you can test the CABS solver on TSPTW, CVRP, SALBP-1, bin packing, MOSP, and graph clear.

```bash
didp-yaml tsptw/tsptw-domain.yaml tsptw/tsptw-problem.yaml solvers/cabs.yaml
didp-yaml cvrp/cvrp-domain.yaml cvrp/cvrp-problem.yaml solvers/cabs.yaml
didp-yaml salbp-1/salbp-1-domain.yaml salbp-1/salbp-1-problem.yaml solvers/cabs.yaml
didp-yaml bin-packing/bin-packing-domain.yaml bin-packing/bin-packing-problem.yaml solvers/cabs.yaml
didp-yaml mosp/mosp-domain.yaml mosp/mosp-problem.yaml solvers/cabs.yaml
didp-yaml graph-clear/graph-clear-domain.yaml graph-clear/graph-clear-problem.yaml solvers/cabs.yaml
```
