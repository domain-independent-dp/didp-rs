[![crates.io](https://img.shields.io/crates/v/dypdl)](https://crates.io/crates/dypdl)
[![minimum rustc 1.64](https://img.shields.io/badge/rustc-1.64+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# dypdl -- DyPDL Library in Rust

dypdl is a library for DyPDL modeling in Rust.

[API Documentation](https://docs.rs/dypdl)

## Example

Modeling TSPTW using DyPDL.

```rust
use dypdl::prelude::*;

// TSPTW instance.
// 0 is the depot, and 1, 2, and 3 are customers.
let n_customers = 4;
// Beginnings of time windows.
let ready_time = vec![0, 5, 0, 8];
// Ends of time windows.
let due_date = vec![0, 16, 10, 14];
// Travel time.
let distance_matrix = vec![
    vec![0, 3, 4, 5],
    vec![3, 0, 5, 4],
    vec![4, 5, 0, 3],
    vec![5, 4, 3, 0],
];

// Minimization and integer cost by default.
let mut model = Model::default();

// Define an object type.
let customer = model.add_object_type("customer", n_customers).unwrap();

// Define state variables.
// Unvisited customers, initially 1, 2, and 3.
let unvisited = model.create_set(customer, &[1, 2, 3]).unwrap();
let unvisited = model.add_set_variable("U", customer, unvisited).unwrap();
// Current location, initially 0.
let location = model.add_element_variable("i", customer, 0).unwrap();
// Current time, less is better, initially 0.
let time = model.add_integer_resource_variable("t", true, 0).unwrap();

// Define tables of constants.
let ready_time: Table1DHandle<Integer> = model.add_table_1d("a", ready_time).unwrap();
let due_date: Table1DHandle<Integer> = model.add_table_1d("b", due_date).unwrap();
let distance: Table2DHandle<Integer> = model.add_table_2d(
    "c", distance_matrix.clone()
).unwrap();

// Define transitions.
let mut visits = vec![];

// Returning to the depot;
let mut return_to_depot = Transition::new("return to depot");
// The cost is the sum of the travel time and the cost of the next state.
return_to_depot.set_cost(distance.element(location, 0) + IntegerExpression::Cost);
// Update the current location to the depot.
return_to_depot.add_effect(location, 0).unwrap();
// Increase the current time.
return_to_depot.add_effect(time, time + distance.element(location, 0)).unwrap();
// Add the transition to the model.
// When this transition is applicable, no need to consider other transitions.
model.add_forward_forced_transition(return_to_depot.clone());
visits.push(return_to_depot);

for j in 1..n_customers {
    // Visiting each customer.
    let mut visit = Transition::new(format!("visit {}", j));
    visit.set_cost(distance.element(location, j) + IntegerExpression::Cost);
    // Remove j from the set of unvisited customers.
    visit.add_effect(unvisited, unvisited.remove(j)).unwrap();
    visit.add_effect(location, j).unwrap();
    // Wait until the ready time.
    let arrival_time = time + distance.element(location, j);
    let start_time = IntegerExpression::max(arrival_time.clone(), ready_time.element(j));
    visit.add_effect(time, start_time).unwrap();
    // The time window must be satisfied.
    visit.add_precondition(
        Condition::comparison_i(ComparisonOperator::Le, arrival_time, due_date.element(j))
    );
    // Add the transition to the model.
    model.add_forward_transition(visit.clone()).unwrap();
    visits.push(visit);
}

// Define a base case.
// If all customers are visited and the current location is the depot, the cost is 0.
let is_depot = Condition::comparison_e(ComparisonOperator::Eq, location, 0);
model.add_base_case(vec![unvisited.is_empty(), is_depot.clone()]).unwrap();

// Define redundant information, which is possibly useful for a solver.
// Define state constraints.
for j in 1..n_customers {
    // The shortest arrival time, assuming the triangle inequality.
    let arrival_time = time + distance.element(location, j);
    // The salesperson must be able to visit each unvisited customer before the deadline.
    let on_time = Condition::comparison_i(
        ComparisonOperator::Le, arrival_time, due_date.element(j)
    );
    model.add_state_constraint(!unvisited.contains(j) | on_time).unwrap();
}

// Define a dual bound.
// The minimum distance to each customer.
let min_distance_to = distance_matrix.iter()
    .enumerate()
    .map(|(j, row)| row.iter().enumerate().filter_map(|(k, d)| {
            if j == k {
               None
            } else {
               Some(*d)
            }
        })
    .min().unwrap()).collect();
let min_distance_to: Table1DHandle<Integer> = model.add_table_1d(
    "c_min", min_distance_to
).unwrap();
let to_depot: IntegerExpression = is_depot.if_then_else(0, min_distance_to.element(0));
let dual_bound: IntegerExpression = min_distance_to.sum(unvisited) + to_depot;
model.add_dual_bound(dual_bound).unwrap();

// Solution.
let solution = [visits[2].clone(), visits[3].clone(), visits[1].clone(), visits[0].clone()];
// Solution cost.
let cost = 14;
// Verify the solution.
assert!(model.validate_forward(&solution, cost, true));
```
