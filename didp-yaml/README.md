# DIDP-YAML

DIDP-YAML is a YAML interface for DyPDL modeling and solvers.

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

There are some examples in [`examples`](https://github.com/domain-independent-dp/didp-rs/didp-yaml/examples). For example, you can test the CABS solver on TSPTW, CVRP, SALBP-1, bin packing, MOSP, and graph clear.

```bash
didp-yaml tsptw/tsptw-domain.yaml tsptw/tsptw-Dumas-n20w20.001.yaml solvers/cabs.yaml
didp-yaml cvrp/cvrp-domain.yaml cvrp/cvrp-E-n13-k4.yaml solvers/cabs.yaml
didp-yaml salbp-1/salbp-1-domain.yaml salbp-1/salbp-1-small-1.yaml solvers/cabs.yaml
didp-yaml bin-packing/bin-packing-domain.yaml bin-packing/bin-packing-Falkenauer_T_t60_00.yaml solvers/cabs.yaml
didp-yaml mosp/mosp-domain.yaml mosp/mosp-GP1.yaml solvers/cabs.yaml
didp-yaml graph-clear/graph-clear-domain.yaml graph-clear/graph-clear-planar20-1.yaml solvers/cabs.yaml
```

## Documents

For the syntax of the DyPDL-YAML and the solver configuration, see the [user guide](https://github.com/domain-independent-dp/didp-rs/didp-yaml/docs)
