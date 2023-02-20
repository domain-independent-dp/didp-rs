# DIDP-YAML

DIDP-YAML is a YAML interface for DyPDL models and solvers.

## Run the Solver

Run the following command.

```bash
cargo run --release domain.yaml problem.yaml config.yaml
```

Here, `domain.yaml` is the domain file for the problem, `problem.yaml` is the problem file for the problem, and `config.yaml` is the file specifying a search algorithm.

Instead of `cargo run`, you can directly use the binary.

```bash
./target/release/didp-yaml domain.yaml problem.yaml config.yaml
```

There are some examples in `./examples`. For example, you can test DIDP on TSPTW, CVRP, SALBP-1, bin packing, MOSP, and graph clear.

```bash
cargo run --release examples/tsptw/tsptw-domain.yaml examples/tsptw/tsptw-Dumas-n20w20.001.yaml examples/solvers/cabs.yaml
cargo run --release examples/cvrp/cvrp-domain.yaml examples/cvrp/cvrp-E-n13-k4.yaml examples/solvers/cabs.yaml
cargo run --release examples/salbp-1/salbp-1-domain.yaml examples/salbp-1/salbp-1-small-1.yaml examples/solvers/cabs.yaml
cargo run --release examples/bin-packing/bin-packing-domain.yaml examples/bin-packing/bin-packing-Falkenauer_T_t60_00.yaml examples/solvers/cabs.yaml
cargo run --release examples/mosp/mosp-domain.yaml examples/mosp/mosp-GP1.yaml examples/solvers/cabs.yaml
cargo run --release examples/graph-clear/graph-clear-domain.yaml examples/graph-clear/graph-clear-planar20-1.yaml examples/solvers/cabs.yaml
```

## Modeling

See the [user guide](./dypdl-guide.md)

## Solver

See the [solver guide](./solver-guide.md)
