[![Actions Status](https://img.shields.io/github/actions/workflow/status/domain-independent-dp/didp-rs/dypdl-heuristic-search.yaml?branch=main&logo=github&style=flat-square)](https://github.com/domain-independent-dp/didp-rs/actions)
[![crates.io](https://img.shields.io/crates/v/dypdl-heuristic-search)](https://crates.io/crates/dypdl-heuristic-search)
[![minimum rustc 1.65](https://img.shields.io/badge/rustc-1.65+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# dypdl-heuristic-search

dypdl-heuristic-search is a library of heuristic search solvers for DyPDL.

[API Documentation](https://docs.rs/dypdl-heuristic-search)

## Example

```rust
use dypdl::prelude::*;
use dypdl_heuristic_search::{CabsParameters, create_dual_bound_cabs, FEvaluatorType};
use std::rc::Rc;

let mut model = Model::default();
let variable = model.add_integer_variable("variable", 0).unwrap();
model.add_base_case(
    vec![Condition::comparison_i(ComparisonOperator::Ge, variable, 1)]
).unwrap();
let mut increment = Transition::new("increment");
increment.set_cost(IntegerExpression::Cost + 1);
increment.add_effect(variable, variable + 1).unwrap();
model.add_forward_transition(increment.clone()).unwrap();
model.add_dual_bound(IntegerExpression::from(0)).unwrap();

let model = Rc::new(model);
let parameters = CabsParameters::default();
let f_evaluator_type = FEvaluatorType::Plus;

let mut solver = create_dual_bound_cabs(model, parameters, f_evaluator_type);
let solution = solver.search().unwrap();
assert_eq!(solution.cost, Some(1));
assert_eq!(solution.transitions, vec![increment]);
assert!(!solution.is_infeasible);
```
