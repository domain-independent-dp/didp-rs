# didp-yaml

[![Actions Status](https://img.shields.io/github/actions/workflow/status/domain-independent-dp/didp-rs/didp-yaml.yaml?branch=main&logo=github&style=flat-square)](https://github.com/domain-independent-dp/didp-rs/actions)
[![crates.io](https://img.shields.io/crates/v/didp-yaml)](https://crates.io/crates/didp-yaml)
[![minimum rustc 1.85](https://img.shields.io/badge/rustc-1.85+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

didp-yaml provides a solution validator and optional solvers for models written
in the Dynamic Programming Description Language (DyPDL), using YAML files.

Use `didp-yaml-validator` to check a candidate solution and compute its objective
value. You do not need to run a solver or provide a solver configuration.
The candidate can come from your own algorithm, another solver, or be written
by hand; it must be expressed as a sequence of transitions in your DyPDL model.

## Install the Validator

Install [Rust](https://www.rust-lang.org/tools/install) 1.85 or later, then install
the published package from crates.io:

```bash
cargo install didp-yaml
```

This installs both `didp-yaml-validator` and the `didp-yaml` solver executable.
You can immediately use `didp-yaml-validator` without configuring or
running a solver. To install only the validator executable, use:

```bash
cargo install didp-yaml --bin didp-yaml-validator
```

If your shell cannot find `didp-yaml-validator` after installation, ensure Cargo's
binary directory is on your `PATH`.

### Install from Source (Alternative)

To use the development version, install from this repository instead:

```bash
git clone https://github.com/domain-independent-dp/didp-rs.git
cd didp-rs
cargo install --path didp-yaml --bin didp-yaml-validator
```

If you already have this repository checked out, run just the `cargo install`
command from its root.

## Validate a Solution

If you already have your model and candidate solution, run:

```bash
didp-yaml-validator domain.yaml problem.yaml solution.yaml
```

The three inputs are:

- `domain.yaml`: the model's variables, transitions, constraints, base cases, and
  objective expressions.
- `problem.yaml`: the instance data, including the initial (target) state.
- `solution.yaml`: the ordered transitions to check, with an optional declared
  `cost`.

The validator checks feasibility and computes the objective. If `cost` is present,
it also compares that value with the computed objective. It does not search for a
solution, prove optimality, or modify the input files.

## Try a Complete Example

Create the following three files in a new directory. This model starts with two
remaining steps; each transition decreases that number by one and adds one to
the objective. A solution must finish with zero steps remaining.

`domain.yaml`:

```yaml
cost_type: integer
reduce: min
state_variables:
  - name: remaining
    type: integer
base_cases:
  - conditions: ['(= remaining 0)']
transitions:
  - name: step
    preconditions: ['(> remaining 0)']
    effect:
      remaining: '(- remaining 1)'
    cost: '(+ cost 1)'
```

`problem.yaml`:

```yaml
target:
  remaining: 2
```

`solution.yaml`:

```yaml
transitions:
  - name: step
  - name: step
```

From the directory containing these files, run:

```bash
didp-yaml-validator domain.yaml problem.yaml solution.yaml
```

The output is:

```text
Solution is valid.
cost: 2
```

The `transitions` field is required; `cost` is not. Adding `cost: 2` at the top of
`solution.yaml` also succeeds. Adding `cost: 3` instead reports a cost mismatch.
Existing didp-yaml solution files containing both fields remain supported.

To try an infeasible candidate, remove the last `- name: step` entry and rerun the
command. The validator exits with code `1` and reports:

```text
Solution `solution.yaml` failed validation: The final state (state 1) does not satisfy any base case.
```

For parameterized transitions, provide the original transition name and its
complete parameter map, such as `{name: visit, parameters: {i: 3}}`.
See the [solution format](docs/validator-guide.md#solution-format) for details,
and the [modeling guide](docs/dypdl-guide.md) to write your own domain and problem.

## Results and Cost Tolerances

Successful validation prints the computed cost to stdout. Errors are written to
stderr. For scripts and automated checks, use the exit status:

- `0`: the solution is feasible and its declared cost, if present, matches.
- `1`: the solution is infeasible or the declared cost does not match.
- `2`: an input or evaluation error, such as an unreadable file, malformed YAML,
  an unknown transition, or a failed expression evaluation.

Cost comparison is exact by default. For a continuous-cost model, you can allow
floating-point rounding differences:

```bash
didp-yaml-validator --rel-tol 1e-9 --abs-tol 1e-9 domain.yaml problem.yaml solution.yaml
```

Integer costs are always compared exactly. Omit `cost` to check feasibility and
compute the objective without comparing a declared value.
Use `didp-yaml-validator --help` for command-line options, or read the
[validator guide](docs/validator-guide.md) for validation semantics and the Rust
library API.

## Run a Solver

The default `cargo install didp-yaml` command above also installs the solver.
If you installed only the validator and want didp-yaml to find solutions, add the
solver executable with:

```bash
cargo install didp-yaml --bin didp-yaml
```

For a source installation, use `cargo install --path didp-yaml --bin didp-yaml`
from the repository root instead.
Solving, unlike validation, requires a solver configuration file:

```bash
didp-yaml domain.yaml problem.yaml config.yaml
```

Here, `config.yaml` selects and configures a solver. See the
[solver guide](docs/solver-guide.md) for its format.

There are examples in [`examples`](examples). From the repository root, change to
`didp-yaml/examples` to try the CABS solver on TSPTW, CVRP, SALBP-1, bin packing,
MOSP, and graph clear:

```bash
cd didp-yaml/examples
didp-yaml tsptw/tsptw-domain.yaml tsptw/tsptw-problem.yaml solvers/cabs.yaml
didp-yaml cvrp/cvrp-domain.yaml cvrp/cvrp-problem.yaml solvers/cabs.yaml
didp-yaml salbp-1/salbp-1-domain.yaml salbp-1/salbp-1-problem.yaml solvers/cabs.yaml
didp-yaml bin-packing/bin-packing-domain.yaml bin-packing/bin-packing-problem.yaml solvers/cabs.yaml
didp-yaml mosp/mosp-domain.yaml mosp/mosp-problem.yaml solvers/cabs.yaml
didp-yaml graph-clear/graph-clear-domain.yaml graph-clear/graph-clear-problem.yaml solvers/cabs.yaml
```
