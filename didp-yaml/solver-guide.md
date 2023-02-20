# User Guide for Solver for DIDP

Different solvers have different characteristics, and some features may not be supported.
The features of solvers are summarized in the following table.

|solver|supported cost expressions|supported reduce|other restrictions|exact|anytime|
|-|-|-|-|-|-|
|[dual_bound_cabs](#caasdy)|`(+ <numeric expression> cost)`, `(max <numeric expression> cost)`|`min`, `max`||yes|yes|
|[caasdy](#caasdy)|`(+ <numeric expression> cost)`, `(max <numeric expression> cost)`|`min`, `max`||yes|no|
|[forward_recursion](#forward_recursion)|any|`min`, `max`|acyclic state space|yes|no|

## Common Config

A config YAML file must have key `solver`, whose value is the name of a solver.
Solver specific configurations are described in a map under `config` key.

- `time_limit`: if `time_limit` seconds passed, a solver returns the best solution and/or bound found so far.
- `primal_bound`: the primal bound of a problem, which can be exploited by a solver. If `cost_type` is `integer`, it must be an integer value. If `cost_type` is `continuous`, it can be a real value.

```yaml
solver: <solver name>
config:
    time_limit: <nonnegative integer>
    primal_bound: <integer or real>
```

## dual_bound_cabs

Complete anytime beam search (CABS).
It performs cost-algebraic search with a heuristic function provided by a user as a numeric expression.
The cost expressions should be in the form of `(+ <numeric expression> cost)` or `(max <numeric expression> cost)`.
It uses the maximum dual bound defined in a model as a heuristic value (h-value).

```yaml
solver: cassdy
config:
    f: <+|max>
```

## cassdy

Cost-Algebraic A*Solver for DyPDL (CAASDy).
It performs cost-algebraic A* with a heuristic function provided by a user as a numeric expression.
The cost expressions should be in the form of `(+ <numeric expression> cost)` or `(max <numeric expression> cost)`.
It uses the maximum dual bound defined in a model as a heuristic value (h-value).

```yaml
solver: cassdy
config:
    f: <+|max>
```

The value of `h` is a numeric expression defining the heuristic function.
The default value for `h` is `0`.

The value of `f` is either of `+` or `max`, and the default value is `+`.
If `+`/`max`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(s)`/`max{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
If the cost expression is in the form of `(+ <numeric expression> cost)`/`(max <numeric expression> cost)`, `+`/`max` should be used.

## forward_recursion

It computes the objective value using recursion while memoizing encountered states.
This is a naive dynamic programming algorithm.

If other algorithms are applicable, they can be more efficient.

```yaml
solver: forward_recursion
```
