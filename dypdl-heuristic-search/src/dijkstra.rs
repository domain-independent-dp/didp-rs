use crate::search_node::{trace_transitions, SearchNode};
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use dypdl::variable_type;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Dijkstra's algorithm solver.
///
/// The current implementation only supports cost-algebra with minimization and non-negative edge costs.
/// E.g., the shortest path and minimizing the maximum edge cost on a path.
pub struct Dijkstra<T> {
    /// Common parameters for heuristic search solvers.
    pub parameters: solver::SolverParameters<T>,
    /// The initial capacity of the data structure storing all generated states.
    pub initial_registry_capacity: Option<usize>,
}

impl<T> solver::Solver<T> for Dijkstra<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
{
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<dypdl::Transition>::new(model, false);
        Ok(dijkstra(
            model,
            generator,
            self.parameters,
            self.initial_registry_capacity,
        ))
    }

    #[inline]
    fn set_primal_bound(&mut self, primal_bound: T) {
        self.parameters.primal_bound = Some(primal_bound)
    }

    #[inline]
    fn get_primal_bound(&self) -> Option<T> {
        self.parameters.primal_bound
    }

    #[inline]
    fn set_time_limit(&mut self, time_limit: f64) {
        self.parameters.time_limit = Some(time_limit)
    }

    #[inline]
    fn get_time_limit(&self) -> Option<f64> {
        self.parameters.time_limit
    }

    #[inline]
    fn set_quiet(&mut self, quiet: bool) {
        self.parameters.quiet = quiet
    }

    #[inline]
    fn get_quiet(&self) -> bool {
        self.parameters.quiet
    }
}

/// Performs Dijkstra's algorithm.
pub fn dijkstra<T>(
    model: &dypdl::Model,
    generator: SuccessorGenerator<dypdl::Transition>,
    parameters: solver::SolverParameters<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
{
    let time_keeper = parameters.time_limit.map_or_else(
        solver::TimeKeeper::default,
        solver::TimeKeeper::with_time_limit,
    );
    let primal_bound = parameters.primal_bound;
    let mut open = BinaryHeap::default();
    let mut registry = StateRegistry::new(model);
    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let cost = T::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let constructor = |state: StateInRegistry, cost: T, _: Option<&Rc<SearchNode<T>>>| {
        Some(Rc::new(SearchNode {
            state,
            cost,
            ..Default::default()
        }))
    };
    let (initial_node, _) = registry.insert(initial_state, cost, constructor).unwrap();
    open.push(Reverse(initial_node));
    let mut expanded = 0;
    let mut generated = 0;
    let mut cost_max = T::zero();

    while let Some(Reverse(node)) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;
        if node.cost() > cost_max {
            cost_max = node.cost();
            if !parameters.quiet {
                println!("cost = {}, expanded: {}", cost_max, expanded);
            }
        }
        if model.is_goal(node.state()) {
            return solver::Solution {
                cost: Some(node.cost()),
                is_optimal: true,
                transitions: trace_transitions(node),
                expanded,
                generated,
                time: time_keeper.elapsed_time(),
                ..Default::default()
            };
        }
        if time_keeper.check_time_limit() {
            return solver::Solution {
                best_bound: Some(cost_max),
                expanded,
                generated,
                time: time_keeper.elapsed_time(),
                ..Default::default()
            };
        }
        for transition in generator.applicable_transitions(node.state()) {
            let cost = transition.eval_cost(node.cost(), node.state(), &model.table_registry);
            if primal_bound.is_some() && cost >= primal_bound.unwrap() {
                continue;
            }
            let state = transition.apply(node.state(), &model.table_registry);
            if model.check_constraints(&state) {
                let constructor =
                    |state: StateInRegistry, cost: T, _: Option<&Rc<SearchNode<T>>>| {
                        Some(Rc::new(SearchNode {
                            state,
                            cost,
                            parent: Some(node.clone()),
                            operator: Some(transition.clone()),
                            closed: RefCell::new(false),
                        }))
                    };
                if let Some((successor, dominated)) = registry.insert(state, cost, constructor) {
                    if let Some(dominated) = dominated {
                        if !*dominated.closed.borrow() {
                            *dominated.closed.borrow_mut() = true;
                        }
                    }
                    open.push(Reverse(successor));
                    generated += 1;
                }
            }
        }
    }
    solver::Solution {
        is_infeasible: true,
        expanded,
        generated,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    }
}
