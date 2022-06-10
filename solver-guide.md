# User Guide for Solver for DIDP

Different solvers have different characteristics, and some features may not be supported.
The features of solvers are summarized in the following table.

|solver|supported cost expressions|supported reduce|exact|
|-|-|-|-|
|[forward_recursion](#forwardrecursion)|any|`min`, `max`|yes|
|[expression_dfbb](#expressiondfs)|`(+ <numeric expression> cost)`, `(max <numeric expression> cost)`|`min`, `max`|yes|
|[expression_astar](#expressionastar)|`(+ <numeric expression> cost)`, `(max <numeric expression> cost)`|`min`|yes|
|[dijkstra](#dijkstra)|`(+ <numeric expression> cost)`, `(max <numeric expression> cost)`|`min`|yes|
|[lazy_dijkstra](#lazydijkstra)|`(+ <numeric expression> cost)`, `(max <numeric expression> cost)`|`min`|yes|
|[exist_dfs](#existdfs)|`(+ <numeric expression> cost)`, `(max <numeric expression> cost)`|`min`, `max`|yes|
|[expression_beam](#expressionbeam)|`(+ <numeric expression> cost)`, `(max <numeric expression> cost)`|`min`, `max`|no|

## Common Config

A config YAML file must have key `solver`, whose value is the name of a solver.
Solver specific configurations are described in a map under `config` key.
You can specify `time_limit` in all solvers.
If `time_limit` seconds passed, a solver returns the best solution and/or bound found so far.

```yaml
solver: <solver name>
config:
    time_limit: <nonnegative integer>
```

## forward_recursion
It computes the objective value using recursion while memoizing encountered states.
This is a naive dynamic programming algorithm.

If other altorithms are applicable, they can be more efficient.

```yaml
solver: forward_recursion
```

## expression_dfbb
It performs depth-first branch-and-bound with a heuristic function provided by a user as a numeric expression.
The cost expressions should be in the form of `(+ <numeric expression> cost)` or `(max <numeric expression> cost)`.
It returns the optimal solution for minimization/maximization if the heuristic function always returns a lower/upper bound.

If the problem is minimization, [expression_astar](#expressionastar) should be more efficient.

```yaml
solver: expression_astar
config:
    h: <numeric expression>
    f: <+|max>
    primal_bound: <numeric value>
```

The value of `h` is a numeric expression defining the heuristic function.
If it is not specified, no heuristic functin is used.

The value of `f` is either of `+` or `max`, and the default value is `+`.
If `+`/`max`, `f(s)`, the priority of a state S is computed as `h(S) + g(s)`/`max{h(S), g(S)}`, where `g(S)` is the cost to reach `S` from the target state.
If the cost expression is in the form of `(+ <numeric expression> cost)`/`(max <numeric expression> cost)`, `+`/`max` should be used.

`primal_bound` is an upper/lower bound for the optimal cost if minimization/maximization.
It can be used to prune states.

## expression_astar
It performs A* with a heuristic function provided by a user as a numeric expression.
The cost expressions should be in the form of `(+ <numeric expression> cost)` or `(max <numeric expression> cost)`.
It returns the optimal solution if the problem is minimization and the heuristic function always returns a lower bound.

If no heuristic function is used, [dijkstra](#dijkstra) or [lazy_dijkstra](@lazydijksra) can be more efficient.

### Config

```yaml
solver: expression_astar
config:
    h: <numeric expression>
    f: <+|max>
    primal_bound: <numeric value>
```

The value of `h` is a numeric expression defining the heuristic function.
The default value for `h` is `0`.

The value of `f` is either of `+` or `max`, and the default value is `+`.
If `+`/`max`, `f(s)`, the priority of a state S is computed as `h(S) + g(s)`/`max{h(S), g(S)}`, where `g(S)` is the cost to reach `S` from the target state.
If the cost expression is in the form of `(+ <numeric expression> cost)`/`(max <numeric expression> cost)`, `+`/`max` should be used.

`primal_bound` is an upper/lower bound for the optimal cost if minimization/maximization.
It can be used to prune states.

## dijkstra
It performs Dijkstra's algorithm.
The cost expressions should be in the form of `(+ <numeric expression> cost)` or `(max <numeric expression> cost)`.
It returns the optimal solution if the problem is minimization.

If there is no preference on element, intger and continuous variables, [lazy_dijkstra](@lazydijksra) can be more efficient.

### Config

```yaml
solver: dijkstra
config:
    primal_bound: <numeric value>
```

`primal_bound` is an upper/lower bound for the optimal cost if minimization/maximization.

## lazy dijkstra
It performs Dijkstra's algorithm.
The cost expressions should be in the form of `(+ <numeric expression> cost)` or `(max <numeric expression> cost)`.
It returns the optimal solution if the problem is minimization.

Compared to [dijkstra](#dijkstra), a state is not generated until it is expanded;
a pair of a transition and a parent state is stored in a priority queue.
When there is no resource variables, lazy_dijkstra preforms exactly the same as dijkstra while it is faster and consumes less amount of memory.

If there is preference on element, integer, and continuous variables, dijksra may be more efficient since its dominance pruning might be stronger.

### Config

```yaml
solver: lazy_dijkstra
config:
    primal_bound: <numeric value>
```

`primal_bound` is an upper/lower bound for the optimal cost if minimization/maximization.
It can be used to prune states.


### exist_dfs
It iteratedly performs depth-first search to find a solution whose cost is better than an incumbent solution.

### Config
```yaml
solver: exist_dfs
config:
    primal_bound: <numeric value>
```

`primal_bound` is an upper/lower bound for the optimal cost if minimization/maximization.

## expression_beam
It performs beam search with a evaluatoin function provided by a user as a numeric expression.
The cost expressions should be in the form of `(+ <numeric expression> cost)` or `(max <numeric expression> cost)`.
There is no gurantee of optimality.

### Config

```yaml
solver: expression_astar
config:
    g:
        <transition name 1>: <cost expression 1>
        ...
        <transition name k>: <cost expression k>
    h: <numeric expression>
    f: <+|max>
    beam_size: <nonnegative integer|list of nonnegative integers>
    maximize: <boolean>
```

The value of `g` is a map, where keys are names of transitions and values are cost expressions.
In addition to the original cost defined in a domain file, the cost defined by these cost expressions are also computed and can be accessed by `cost` in these expressions.
This is used as `g(S)` for state `S`.
If `g` is not specified, the original cost defined in a domain file is used.

The value of `h` is a numeric expression defining the heuristic function.
The default value for `h` is `0`.

The value of `f` is either of `+` or `max`, and the default value is `+`.
If `+`/`max`, `f(s)`, the priority of a state S is computed as `h(S) + g(s)`/`max{h(S), g(S)}`, where `g(S)` is the cost to reach `S` from the target state.
`f(S)` is the evaluation value of S.

The value of `beam_size` is nonnegative integer, specifying how many candidates to store at each depth.
If a list is specified, it sequentially perform beam search with specified beam sizes and returns the best solution.
The default value is `10000`.

The value of `maximize` should be boolean.
If it is `true`/`false`, top `beam_size` candidates maximizing/minimizing the f-values are stored.
The default value is `false`.

## iteratve
It sequentially uses multiple solvers.
If a solution is found, the cost of the solution is passed to the next solver as a primal bound.

### Config

```yaml
solver: iterative
config:
  solvers:
    - <solver config>
    - <solver config>
```