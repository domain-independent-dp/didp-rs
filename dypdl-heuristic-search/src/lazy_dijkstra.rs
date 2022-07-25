use crate::lazy_search_node::LazySearchNode;
use crate::search_node::trace_transitions;
use crate::solver;
use crate::state_registry::{StateInRegistry, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use dypdl::variable_type;
use std::cmp::{Ordering, Reverse};
use std::collections;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Lazy Dijkstra's algorithm solver.
///
/// Yet another implementation of Dijkstra's algorithm.
/// Pointers to parent nodes and transitions are stored in the open list, and a state is geenrated when it is expanded.
/// The current implementation only supports cost-algebra with minimization and non-negative edge costs.
/// E.g., the shortest path and minimizing the maximum edge cost on a path.
#[derive(Debug, PartialEq, Clone)]
pub struct LazyDijkstra<T> {
    /// Common parameters for heuristic search solvers.
    pub parameters: solver::SolverParameters<T>,
    /// The initial capacity of the data structure storing all generated states.
    pub initial_registry_capacity: Option<usize>,
}

impl<T> solver::Solver<T> for LazyDijkstra<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
{
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<dypdl::Transition>::new(model, false);
        Ok(lazy_dijkstra(
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
    fn get_time_limit(&self) -> Option<u64> {
        self.parameters.time_limit
    }

    #[inline]
    fn set_time_limit(&mut self, time_limit: u64) {
        self.parameters.time_limit = Some(time_limit)
    }

    #[inline]
    fn get_primal_bound(&self) -> Option<T> {
        self.parameters.primal_bound
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

/// Search node stored in the open list of lazy Dijkstra.
#[derive(Debug)]
struct DijkstraEdge<T: variable_type::Numeric + Ord> {
    cost: T,
    parent: Rc<LazySearchNode<T>>,
    transition: Rc<dypdl::Transition>,
}

impl<T: variable_type::Numeric + Ord> PartialEq for DijkstraEdge<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl<T: variable_type::Numeric + Ord> Eq for DijkstraEdge<T> {}

impl<T: variable_type::Numeric + Ord> Ord for DijkstraEdge<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl<T: variable_type::Numeric + Ord> PartialOrd for DijkstraEdge<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Performs lazy Dijkstra's algorithm.
///
/// Pointers to parent nodes and transitions are stored in the open list, and a state is geenrated when it is expanded.
pub fn lazy_dijkstra<T>(
    model: &dypdl::Model,
    generator: SuccessorGenerator<dypdl::Transition>,
    parameters: solver::SolverParameters<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
{
    let time_keeper = parameters.time_limit.map(solver::TimeKeeper::new);
    let primal_bound = parameters.primal_bound;
    let mut open = collections::BinaryHeap::new();
    let mut registry = StateRegistry::new(model);
    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let cost = T::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let constructor = |state: StateInRegistry, cost: T, _: Option<&Rc<LazySearchNode<T>>>| {
        Some(Rc::new(LazySearchNode {
            state,
            cost,
            ..Default::default()
        }))
    };
    let (initial_node, _) = registry.insert(initial_state, cost, constructor).unwrap();
    for transition in generator.applicable_transitions(&initial_node.state) {
        let cost = transition.eval_cost(
            initial_node.cost,
            &initial_node.state,
            &model.table_registry,
        );
        open.push(Reverse(DijkstraEdge {
            cost,
            parent: initial_node.clone(),
            transition,
        }));
    }
    let mut expanded = 0;
    let mut cost_max = T::zero();

    while let Some(Reverse(edge)) = open.pop() {
        let state = edge
            .transition
            .apply(&edge.parent.state, &model.table_registry);
        if !model.check_constraints(&state) {
            continue;
        }
        let constructor = |state: StateInRegistry, cost: T, _: Option<&Rc<LazySearchNode<T>>>| {
            Some(Rc::new(LazySearchNode {
                state,
                cost,
                parent: Some(edge.parent),
                operator: Some(edge.transition),
            }))
        };
        if let Some((node, _)) = registry.insert(state, edge.cost, constructor) {
            expanded += 1;
            if node.cost > cost_max {
                cost_max = node.cost;
                if !parameters.quiet {
                    println!("cost = {}, expanded: {}", cost_max, expanded);
                }
            }
            if model.is_goal(&node.state) {
                return solver::Solution {
                    cost: Some(node.cost),
                    is_optimal: true,
                    transitions: trace_transitions(node),
                    expanded,
                    generated: expanded,
                    ..Default::default()
                };
            }
            if time_keeper
                .as_ref()
                .map_or(false, |time_keeper| time_keeper.check_time_limit())
            {
                return solver::Solution {
                    best_bound: Some(cost_max),
                    expanded,
                    generated: expanded,
                    ..Default::default()
                };
            }
            for transition in generator.applicable_transitions(&node.state) {
                let cost = transition.eval_cost(node.cost, &node.state, &model.table_registry);
                if primal_bound.is_some() && cost >= primal_bound.unwrap() {
                    continue;
                }
                open.push(Reverse(DijkstraEdge {
                    cost,
                    parent: node.clone(),
                    transition,
                }));
            }
        }
    }
    solver::Solution {
        is_infeasible: true,
        expanded,
        generated: expanded,
        ..Default::default()
    }
}
