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
cargo run --release examples/tsptw-domain.yaml examples/tsptw-Dumas-n150w60.001.yaml examples/caasdy.yaml
cargo run --release examples/cvrp-domain.yaml examples/cvrp-E-n13-k4.yaml examples/caasdy.yaml
cargo run --release examples/salbp-1-domain.yaml examples/salbp-1-very-large-1.yaml examples/caasdy.yaml
cargo run --release examples/bin-packing-domain.yaml examples/bin-packing-Falkenauer_T_t60_00.yaml examples/caasdy.yaml
cargo run --release examples/mosp-domain.yaml examples/mosp-SP1.yaml examples/caasdy.yaml
cargo run --release examples/graph-clear-domain.yaml examples/graph-clear-planar20-1.yaml examples/caasdy.yaml
```

## Modeling
See the [user guide](./dypdl-guide.md)

## Solver
See the [solver guide](./solver-guide.md)