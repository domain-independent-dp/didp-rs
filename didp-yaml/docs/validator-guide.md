# Validating a Solution

The `didp-yaml` package includes a standalone validator executable. You can check
solutions produced by any algorithm, or written by hand, by expressing them as
transitions in the supplied DyPDL model. You do not need to use a didp-yaml solver.

Start with [validator-only installation](../README.md#install-the-validator) and
the [complete example](../README.md#try-a-complete-example) in the package README.

## Run the validator

```console
didp-yaml-validator domain.yaml problem.yaml solution.yaml
```

The arguments are the domain file (model definition), problem file (instance
data and target state), and solution file (candidate transition sequence).
No solver configuration is needed. The validator loads the model, resolves the
solution's transitions, checks feasibility, and computes the objective. When the
solution declares a cost, it also checks that cost. It does not modify any files.

Successful validation prints:

```text
Solution is valid.
cost: 2
```

Exit codes are `0` for success, `1` for an infeasible solution or a cost mismatch,
and `2` for input errors (including unknown or ambiguous transition references)
or expression evaluation errors. Failures are written to stderr with the file
and validation context. Use `--help` for usage information.

To run directly from a repository checkout without installing, use this command
from the repository root, with paths to your three input files:

```bash
cargo run --release -p didp-yaml --bin didp-yaml-validator -- domain.yaml problem.yaml solution.yaml
```

The `--bin didp-yaml-validator` option selects validation; the default `didp-yaml`
executable runs a solver instead.

## Solution format

A solution lists model transitions in the order they should be applied. You can
write this file by hand or export it from your own program. The format is also
compatible with output from the didp-yaml solver:

```yaml
cost: 2
transitions:
  - name: open-and-pack
    parameters: {i: 0}
  - name: pack
    parameters: {i: 1}
```

`transitions` is required and must be a sequence. `transitions: []` represents an
empty candidate sequence. Each entry contains a string `name` and an optional
mapping `parameters`. Parameter names and values must exactly match a grounded
forward or forward-forced transition in the model. Parameter order is irrelevant;
missing or extra parameters are errors. Parameter values must be non-negative
integer element indices. Unknown fields and duplicate mapping keys are rejected.
Exactly one YAML document is required.

For transitions without parameters, omit `parameters` or specify `{}`. Names
are interpreted literally. For a model with a parameterized `visit` transition,
use `name: visit` and `parameters: {i: 3}`. If the model instead defines the literal
name `visit i:3` without parameters (as a dumped grounded model may do), use that
literal name and no parameters. Use the identities of the supplied model.
Ambiguous identities are rejected rather than resolved by model order.

`cost` is optional. An omitted or null cost means no declared cost is checked;
the objective is still computed. Integer models require a 32-bit integer declared
cost. Continuous models accept finite integer or real YAML numbers. Quoted
numeric strings, booleans, NaN, and infinities are rejected.
Existing solver output containing both `cost` and `transitions` remains supported;
allowing the cost to be omitted does not change that format.

## Validation semantics

All transition references are checked before feasibility. An invalid reference
later in the sequence is reported before an earlier feasibility failure. Once
all references are valid, replay stops at the first feasibility failure.

The validator checks state constraints at the target and every subsequent state,
checks each transition's applicability, and requires the sequence to end at the
first base state. An empty sequence is feasible if the target is a base state
that satisfies the state constraints. The objective is computed by evaluating
the base cost and then evaluating transition cost expressions in reverse order,
using the state before each transition. Nonzero base costs and non-additive
transition costs are supported.

Forced-transition priority, transition dominance, dual bounds, and optimality are
not checked. Expression evaluation failures are reported with context, and
non-finite objectives are rejected. Arithmetic otherwise follows DyPDL's existing
expression evaluation semantics.

Human-readable transition and condition numbers start at 1. State 0 is the
target; state 1 follows transition 1. Condition numbers refer to internally
simplified or grounded conditions. YAML field paths such as
`transitions[2].parameters.i` use zero-based sequence indices.

## Continuous cost comparison

Comparison is exact by default. To allow floating-point rounding differences:

```console
didp-yaml-validator --rel-tol 1e-9 --abs-tol 1e-9 domain.yaml problem.yaml solution.yaml
```

The comparison accepts `|declared - computed|` up to
`max(abs_tol, rel_tol * max(|declared|, |computed|))`. Both tolerances must be
finite and non-negative. Integer costs are always compared exactly. Successful
validation always reports the computed cost.

## Rust library

The `didp_yaml::solution` module exposes `load_solution_from_yaml`,
`load_solution_from_str`, and `load_solution_from_file`, each taking an existing
`dypdl::Model`. They return `LoadedSolution`, which contains the optional declared
cost and resolved transitions. Loading checks the file format and transition
identities; call `validate_solution` to validate feasibility and compute the cost:

```rust,ignore
use didp_yaml::solution::{load_solution_from_file, validate_solution, CostTolerance};

let solution = load_solution_from_file(&model, "solution.yaml")?;
let cost = validate_solution(&model, &solution, CostTolerance::default())?;
println!("cost: {cost}");
```

The standalone `validate_cost(computed, declared, tolerance)` function compares
two `SolutionCost` values without evaluating transitions. The YAML validator calls
it only when a declared cost is present.

`SolutionToDump` supports `dump_to_str` and `dump_to_file`, and can be constructed
from a solver `Solution` or a `LoadedSolution`. Existing imports of
`didp_yaml::heuristic_search_solver::{CostToDump, SolutionToDump}` remain supported.

The shared engine is `dypdl::Model::validate_solution`, returning a computed cost
or a structured `SolutionValidationError`. Use `Integer` for integer models and
`OrderedContinuous` for continuous models. Rust callers with transition IDs can
use `Model::validate_solution_with_ids` to resolve forward or forward-forced IDs
directly, with the same feasibility checks. IDs are relative to the supplied
model and do not track model ownership. The YAML format continues to use names
and parameter maps, not numeric IDs. In-process callers retain their panic
hook, which can print a diagnostic even when an evaluation panic is converted
into a validation error.
