use super::lazy_dijkstra::DijkstraEdge;
use crate::bfs_lifo_open_list::BFSLIFOOpenList;
use crate::lazy_search_node::LazySearchNode;
use crate::search_node::trace_transitions;
use crate::solver;
use crate::state_registry::{StateInRegistry, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use dypdl::{variable_type, Continuous};
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Lazy Dijkstra's algorithm solver with bounded depth-first lookahead.
pub struct LookaheadLazyDijkstra<T> {
    /// Bound ratio
    pub bound_ratio: f64,
    /// Common parameters for heuristic search solvers.
    pub parameters: solver::SolverParameters<T>,
    /// The initial capacity of the data structure storing all generated states.
    pub initial_registry_capacity: Option<usize>,
}

impl<T> solver::Solver<T> for LookaheadLazyDijkstra<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
{
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<dypdl::Transition>::new(model, false);
        Ok(lookahead_lazy_dijkstra(
            model,
            generator,
            self.bound_ratio,
            self.parameters,
            self.initial_registry_capacity,
        ))
    }

    #[inline]
    fn set_primal_bound(&mut self, primal_bound: T) {
        self.parameters.primal_bound = Some(primal_bound)
    }

    #[inline]
    fn get_time_limit(&self) -> Option<f64> {
        self.parameters.time_limit
    }

    #[inline]
    fn set_time_limit(&mut self, time_limit: f64) {
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

/// Performs lazy Dijkstra's algorithm with bounded depth-first lookahead.
///
/// Pointers to parent nodes and transitions are stored in the open list, and a state is generated when it is expanded.
pub fn lookahead_lazy_dijkstra<T>(
    model: &dypdl::Model,
    generator: SuccessorGenerator<dypdl::Transition>,
    bound_ratio: Continuous,
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
    let mut open = BFSLIFOOpenList::default();
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
        open.push(
            cost,
            DijkstraEdge {
                cost,
                parent: initial_node.clone(),
                transition,
            },
        );
    }
    let mut dfs_open = Vec::new();
    let mut expanded = 0;
    let mut best_bound = T::zero();
    let mut weighted_bound = T::from_continuous(best_bound.to_continuous() * bound_ratio);

    while let Some(peek) = open.peek() {
        if peek.cost > best_bound {
            best_bound = peek.cost;
            weighted_bound = T::from_continuous(best_bound.to_continuous() * bound_ratio);
            if !parameters.quiet {
                println!("Best bound: {}, expanded: {}", best_bound, expanded);
                println!("Weighted bound: {}", weighted_bound);
            }
        }

        let edge = if let Some(edge) = dfs_open.pop() {
            edge
        } else {
            open.pop().unwrap()
        };
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

            if model.is_goal(&node.state) {
                return solver::Solution {
                    cost: Some(node.cost),
                    is_optimal: true,
                    transitions: trace_transitions(node),
                    expanded,
                    generated: expanded,
                    time: time_keeper.elapsed_time(),
                    ..Default::default()
                };
            }

            if time_keeper.check_time_limit() {
                return solver::Solution {
                    best_bound: Some(best_bound),
                    expanded,
                    generated: expanded,
                    time: time_keeper.elapsed_time(),
                    ..Default::default()
                };
            }

            let mut dfs_successors = Vec::new();
            for transition in generator.applicable_transitions(&node.state) {
                let cost = transition.eval_cost(node.cost, &node.state, &model.table_registry);
                if primal_bound.is_some() && cost >= primal_bound.unwrap() {
                    continue;
                }
                let successor = DijkstraEdge {
                    cost,
                    parent: node.clone(),
                    transition,
                };
                if cost <= weighted_bound {
                    dfs_successors.push(successor.clone());
                }
                open.push(cost, successor);
            }
            dfs_successors.sort_by(|a, b| b.cmp(a));
            dfs_open.append(&mut dfs_successors);
        }
    }
    solver::Solution {
        is_infeasible: true,
        expanded,
        generated: expanded,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    }
}
