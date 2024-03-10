# User Guide for Solver for DIDP

Different solvers have different characteristics, and some features may not be supported.
The features of solvers are summarized in the following table.

Solvers except for `forward_recursion` solves a DyPDL model as a generalized version of the shortest path problem, so it requires specific form of the cost expression.

We recommend using [`dual_bound_cabs`](#dual_bound_cabs) with the default parameter if it is available for your model.

|solver|supported cost expressions|supported reduce|other restrictions|exact|anytime|
|-|-|-|-|-|-|
|[forward_recursion](#forward_recursion)|any|`min`, `max`|acyclic state space|yes|no|
|[caasdy](#caasdy)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|no|
|[dual_bound_cabs](#dual_bound_cabs)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|yes|
|[dual_bound_lnbs](#dual_bound_lnbs)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|yes|
|[dual_bound_dfbb](#dual_bound_dfbb)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|yes|
|[dual_bound_cbfs](#dual_bound_cbfs)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|yes|
|[dual_bound_acps](#dual_bound_acps)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|yes|
|[dual_bound_apps](#dual_bound_apps)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|yes|
|[dual_bound_dbdfs](#dual_bound_dbdfs)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|yes|
|[dual_bound_breadth_first_search](#dual_bound_breadth_first_search)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|yes|
|[dual_bound_dd_lns](#dual_bound_dd_lns)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||yes|yes|
|[dual_bound_weighted_astar](#dual_bound_weighted_astar)|`(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, `(min <numeric_expression> cost)`|`min`, `max`||no|no|

## Common Config

A config YAML file must have key `solver`, whose value is the name of a solver.
Solver specific configurations are described in a map under `config` key, which are optional.
Also, `dump_to` is optional.

```yaml
solver: <solver name>
config:
    primal_bound: <integer or real>
    time_limit: <nonnegative integer>
    quiet: <bool>
    get_all_solutions: <bool>
dump_to: <filename>
```

- `primal_bound`: the primal bound of a problem, which can be exploited by a solver. If `cost_type` is `integer`, it must be an integer value. If `cost_type` is `continuous`, it can be a real value.
  - default: `null`
- `time_limit`: if `time_limit` seconds passed, a solver returns the best solution and/or bound found so far.
  - default: `null`
- `quiet`: if `true`, the output is suppressed.
  - default: `false`
- `get_all_solutions`: if `true` and `dump_to` is specified, all feasible solutions found are reported.
  - default: `false`
- `dump_to`: the file to dump feasible solutions and the time they are obtained. Unless `get_all_solutions: true`, only improving solutions are reported.
  - default: `null`

## forward_recursion

It computes the objective value using recursion while memoizing encountered states.
This is a naive dynamic programming algorithm.

If other algorithms are applicable, they can be more efficient.

```yaml
solver: forward_recursion
```

## caasdy

Cost-Algebraic A\* Solver for DyPDL (CAASDy).

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: cassdy
config:
    f: <+|*|max|min>
    initial_registry_capacity: <nonnegative integer>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_registry_capacity`: the initial capacity of the data structure storing all generated states.
  - default: `1000000`

Ryo Kuroiwa and J. Christopher Beck. “Domain-Independent Dynamic Programming: Generic State Space Search for Combinatorial Optimization,” Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 236-244, 2023.

Stephen Edelkamp, Shahid Jabbar, Alberto Lluch Lafuente. “Cost-Algebraic Heuristic Search,” Proceedings of the 20th National Conference on Artificial Intelligence (AAAI), pp. 1362-1367, 2005.

Peter E. Hart, Nills J. Nilsson, Bertram Raphael. “A Formal Basis for the Heuristic Determination of Minimum Cost Paths”, IEEE Transactions of Systems Science and Cybernetics, vol. SSC-4(2), pp. 100-107, 1968.

## dual_bound_cabs

Complete Anytime Beam Search (CABS).

CABS performs a sequence of beam search runs with exponentially increasing beam width.

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: dual_bound_cabs 
config:
    f: <+|*|max|min>
    initial_beam_size: <int>
    keep_all_layers: <bool>
    max_beam_size: <int>
    threads: <int>
    parallel_type: <hdbs2|hdbs1|sbs>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_beam_size`: the initial beam size.
  - default: `1`
- `keep_all_layers`: if keep all states in all layers for duplicate detection. If `false`, only states in the current layer are kept. Here, the i th layer contains states that can be reached with i transitions. `keep_all_layers: true` is recommended if one state can belong to multiple layers.
  - default: `false`
- `max_beam_size`: the maximum beam size. If `None`, it keep increasing the beam size until proving the optimality or infeasibility or reaching the time limit.
  - default: `None`
- `threads`: the number[$] of threads.
  - default: `1`
- `parallel_type`: the method for parallelization. `hdbs2` is HDBS2, `hdbs1` is HDBS1, and `sbs` is SBS. `hdbs2` is recommended.
  - default: `hdbs2`

Ryo Kuroiwa and J. Christopher Beck. “Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,” Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.

Ryo Kuroiwa and J. Christopher Beck. “Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,” Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.

Weixiong Zhang. “Complete Anytime Beam Search,” Proceedings of the 15th National Conference on Artificial Intelligence/Innovative Applications of Artificial Intelligence (AAAI/IAAI), pp. 425-430, 1998.

## dual_bound_lnbs

Large Neighborhood Beam Search (LNBS).

It improves a solution by finding a partial path using beam search. It first performs CABS to find an initial feasible solution and then performs LNBS to improve the solution.

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.  If `<numeric expression>` can be negative, please set `has_negative_cost` to `true`.

```yaml
solver: dual_bound_lnbs
config:
    f: <+|*|max|min>
    initial_beam_size: <int>
    keep_all_layers: <bool>
    max_beam_size: <int>
    seed: <int>
    has_negative_cost: <bool>
    use_cost_weight: <bool>
    no_bandit: <bool>
    no_transition_mutex: <bool>
    cabs_initial_beam_size: <int>
    cabs_max_beam_size: <int>
    threads: <int>
    parallel_type: <hdbs2|hdbs1|sbs>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_beam_size`: the initial beam size.
  - default: `1`
- `keep_all_layers`: if keep all states in all layers for duplicate detection. If `false`, only states in the current layer are kept. Here, the i th layer contains states that can be reached with i transitions. `keep_all_layers: true` is recommended if one state can belong to multiple layers.
  - default: `false`
- `max_beam_size`: the maximum beam size. If `None`, it keep increasing the beam size until proving the optimality or infeasibility or reaching the time limit.
  - default: `None`
- `seed`: random seed.
  - default: `2023`
- `has_negative_cost`: whether the cost of a transition can be negative.
  - default: `false`
- `use_cost_weight`: use weighted sampling biased by costs to select a start of a partial path.
  - default: `false`
- `no_bandit`: do not use bandit-based sampling to select the depth of a partial path.
  - default: `false`
- `no_transition_mutex`: do not remove transitions conflicting with a suffix from a partial state space.
  - default: `false`
- `cabs_initial_beam_size`: the initial beam size for CABS to find an initial feasible solution.
  - default: `1`
- `cabs_max_beam_size`: the maximum beam size for CABS to find an initial feasible solution. If `None`, it keep increasing the beam size until finding a feasible solution, proving infeasibility, or reaching the time limit.
  - default: `None`
- `threads`: the number[$] of threads.
  - default: `1`
- `parallel_type`: the method for parallelization. `hdbs2` is HDBS2, `hdbs1` is HDBS1, and `sbs` is SBS. `hdbs2` is recommended.
  - default: `hdbs2`

Ryo Kuroiwa and J. Christopher Beck. “Large Neighborhood Beam Search for Domain-Independent Dynamic Programming,” Proceedings of the 29th International Conference on Principles and Practice of Constraint Programming (CP), pp. 23:1-23:22, 2023.

Ryo Kuroiwa and J. Christopher Beck. “Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,” Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.

## dual_bound_dfbb

Depth-First Branch-and-Bound (DFBB).

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: dual_bound_dfbb
config:
    f: <+|*|max|min>
    initial_registry_capacity: <nonnegative integer>
    bfs_tie_breaking: <bool>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_registry_capacity`: the initial capacity of the data structure storing all generated states.
  - default: `1000000`
- `bfs_tie_breaking`: if sort successor states according to their f-values.
  - default: `false`

Ryo Kuroiwa and J. Christopher Beck. “Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,” Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.

## dual_bound_cbfs

Cyclic Best-First Search (CBFS).

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: dual_bound_dfbb
config:
    f: <+|*|max|min>
    initial_registry_capacity: <nonnegative integer>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_registry_capacity`: the initial capacity of the data structure storing all generated states.
  - default: `1000000`

Ryo Kuroiwa and J. Christopher Beck. “Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,” Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.

Gio K. Kao, Edward C. Sewell, and Sheldom H. Jacobson. “A Branch, Bound and Remember Algorithm for the 1|r_i|Σt_i scheduling problem,” Journal of Scheduling, vol. 12(2), pp. 163-175, 2009.

## dual_bound_acps

Anytime Column Progressive Search (ACPS).

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: dual_bound_acps
config:
    f: <+|*|max|min>
    initial_registry_capacity: <nonnegative integer>
    init: <nonnegative integer>
    step: <nonnegative integer>
    width_bound: <nonnegative integer>
    reset: <bool>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_registry_capacity`: the initial capacity of the data structure storing all generated states.
  - default: `1000000`
- `init`: the initial beam size.
  - default: `1`
- `step`: the amount of the increment of the beam size when reaching the last layer.
  - default: `1`
- `width_bound`: the maximum beam size.
  - default: `null`
- `reset`: if reset the beam size to `init` when a new improving solution is found.
  - default: `false`

Ryo Kuroiwa and J. Christopher Beck. “Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,” Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.

Sataya Gautam Vadlamudi, Piyush Gaurav, Sandip Aine, and Partha Pratim Chakrabarti. “Anytime Column Search,” Proceedings of AI 2012: Advances in Artificial Intelligence, pp. 254-255, 2012.

## dual_bound_apps

Anytime Pack Progressive Search (ACPS).

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: dual_bound_apps
config:
    f: <+|*|max|min>
    initial_registry_capacity: <nonnegative integer>
    init: <nonnegative integer>
    step: <nonnegative integer>
    width_bound: <nonnegative integer>
    reset: <bool>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_registry_capacity`: the initial capacity of the data structure storing all generated states.
  - default: `1000000`
- `init`: the initial beam size.
  - default: `1`
- `step`: the amount of the increment of the beam size when reaching the last layer.
  - default: `1`
- `width_bound`: the maximum beam size.
  - default: `null`
- `reset`: if reset the beam size to `init` when a new improving solution is found.
  - default: `false`

Ryo Kuroiwa and J. Christopher Beck. “Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,” Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.

Sataya Gautam Vadlamudi, Sandip Aine, Partha Pratim Chakrabarti. “Anytime Pack Search,” Natural Computing, vol. 15(3), pp. 395-414, 2016.

## dual_bound_dbdfs

Discrepancy-Bounded Depth-First Search (DBDFS).

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: dual_bound_dbdfs
config:
    f: <+|*|max|min>
    initial_registry_capacity: <nonnegative integer>
    width: <nonnegative integer>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_registry_capacity`: the initial capacity of the data structure storing all generated states.
  - default: `1000000`
- `width`: the width of discrepancy to search.
  - default: `1`

Ryo Kuroiwa and J. Christopher Beck. “Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,” Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), pp. 245-253, 2023.

## dual_bound_breadth_first_search

Breadth-first search.

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: dual_bound_cabs 
config:
    f: <+|*|max|min>
    initial_registry_capacity: <nonnegative integer>
    keep_all_layers: <bool>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_registry_capacity`: the initial capacity of the data structure storing all generated states.
  - default: `1000000`
- `keep_all_layers`: if keep all states in all layers for duplicate detection. If `false`, only states in the current layer are kept. Here, the i th layer contains states that can be reached with i transitions. `keep_all_layers: true` is recommended if one state can belong to multiple layers.
  - default: `false`

## dual_bound_dd_lns

Large Neighborhood Search with Decision Diagrams (DD-LNS).

This performs LNS by constructing restricted multi-valued decision diagrams (MDD). It first performs CABS to find an initial feasible solution and then performs DD-LNS to improve the solution.

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: dual_bound_dd_lns
config:
    f: <+|*|max|min>
    beam_size: <int>
    keep_probability: <float>
    keep_all_layers: <bool>
    seed: <int>
    cabs_initial_beam_size: <int>
    cabs_max_beam_size: <int>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `beam_size`: the beam size.
  - default: `1000`
- `keep_probability`: probability to keep a non-best state.
  - default: `0.1`
- `keep_all_layers`: if keep all states in all layers for duplicate detection. If `false`, only states in the current layer are kept. Here, the i th layer contains states that can be reached with i transitions. `keep_all_layers: true` is recommended if one state can belong to multiple layers.
  - default: `false`
- `seed`: random seed.
  - default: `2023`
- `cabs_initial_beam_size`: the initial beam size for CABS to find an initial feasible solution.
  - default: `1`
- `cabs_max_beam_size`: the maximum beam size for CABS to find an initial feasible solution. If `None`, it keep increasing the beam size until finding a feasible solution, proving infeasibility, or reaching the time limit.
  - default: `None`

Xavier Gillard and Pierre Schaus. “Large Neighborhood Search with Decision Diagrams,” Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI), pp. 4754-4760, 2022.

## dual_bound_weighted_astar

Weighted A\*.

It uses the dual bound defined in a model as a heuristic function (h-value).

The cost expressions should be in the form of `(+ <numeric expression> cost)`, `(* <numeric expression> cost)`, `(max <numeric expression> cost)`, or `(min <numeric expression> cost)`.

```yaml
solver: dual_bound_weighted_astar 
config:
    f: <+|*|max|min>
    initial_registry_capacity: <nonnegative integer>
    weight: <integer or real>
```

- `f`: either of `+`, `*`, `max`, and `min`. If `+`/`*`/`max`/`min`, `f(s)`, the priority of a state `S` is computed as `h(S) + g(S)`/`h(S) * G(S)`/`max{h(S), g(S)}`/`min{h(S), g(S)}`, where `h(S)` is the h-value of `S` and `g(S)` is the cost to reach `S` from the target state.
  - default: `+`
- `initial_registry_capacity`: the initial capacity of the data structure storing all generated states.
  - default: `1000000`
- `weight`: the weight of the h-value.
