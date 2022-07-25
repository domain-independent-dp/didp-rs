use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use dypdl::variable_type;
use dypdl::ReduceFunction;
use dypdl::Transition;
use std::error::Error;
use std::fmt;
use std::str;

/// Iterative Bounded Depth-First Search (IBDFS) solver.
///
/// This solver repeatedly performs depth-first search to find a solution having a better cost than the current primal bound.
/// It terminates when no solution is better than the current primal bound.
/// The current implementation only support cost minimization or maximization with associative cost computation.
/// E.g., the shortest path and the longest path.
#[derive(Debug, PartialEq, Clone)]
pub struct IBDFS<T: variable_type::Numeric> {
    /// Common parameters for heuristic search solvers.
    pub parameters: solver::SolverParameters<T>,
    /// The initial capacity of the data structure storing all generated states.
    pub initial_registry_capacity: Option<usize>,
}

impl<T> solver::Solver<T> for IBDFS<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    #[inline]
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<dypdl::Transition>::new(model, false);
        Ok(forward_ibdfs(
            model,
            &generator,
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
    fn set_time_limit(&mut self, time_limit: u64) {
        self.parameters.time_limit = Some(time_limit)
    }

    #[inline]
    fn get_time_limit(&self) -> Option<u64> {
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

/// Node for DFS.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RecursiveNode<T: variable_type::Numeric> {
    pub state: StateInRegistry,
    pub cost: T,
}

impl<T: variable_type::Numeric> StateInformation<T> for RecursiveNode<T> {
    fn cost(&self) -> T {
        self.cost
    }

    fn state(&self) -> &StateInRegistry {
        &self.state
    }
}

/// Performs iterative bounded depth-first search, which repeatedly performs depth-first search to find a better solution.
pub fn forward_ibdfs<T>(
    model: &dypdl::Model,
    generator: &SuccessorGenerator<Transition>,
    parameters: solver::SolverParameters<T>,
    capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable_type::Numeric + fmt::Display,
{
    let time_keeper = parameters.time_limit.map(solver::TimeKeeper::new);
    let quiet = parameters.quiet;
    let mut primal_bound = parameters.primal_bound;
    let mut expanded = 0;
    let mut prob = StateRegistry::new(model);
    if let Some(capacity) = capacity {
        prob.reserve(capacity);
    };
    let mut incumbent = None;
    let node = RecursiveNode {
        state: StateInRegistry::new(&model.target),
        cost: T::zero(),
    };
    loop {
        match bounded_dfs(
            node.clone(),
            model,
            generator,
            &mut prob,
            primal_bound,
            &time_keeper,
            &mut expanded,
        ) {
            (Some((new_cost, transitions)), _) => {
                if let Some(current_cost) = primal_bound {
                    match model.reduce_function {
                        dypdl::ReduceFunction::Max if new_cost > current_cost => {
                            primal_bound = Some(new_cost);
                            incumbent = Some(transitions);
                            if !quiet {
                                println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                            }
                        }
                        dypdl::ReduceFunction::Min if new_cost < current_cost => {
                            primal_bound = Some(new_cost);
                            incumbent = Some(transitions);
                            if !quiet {
                                println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                            }
                        }
                        _ => {}
                    }
                } else {
                    primal_bound = Some(new_cost);
                    incumbent = Some(transitions);
                    if !quiet {
                        println!("New primal bound: {}, expanded: {}", new_cost, expanded);
                    }
                }
            }
            (_, time_out) => {
                return incumbent.map_or_else(
                    || solver::Solution {
                        is_infeasible: true,
                        expanded,
                        generated: expanded,
                        ..Default::default()
                    },
                    |mut incumbent| solver::Solution {
                        cost: primal_bound,
                        is_optimal: !time_out,
                        transitions: {
                            incumbent.reverse();
                            incumbent
                        },
                        expanded,
                        generated: expanded,
                        ..Default::default()
                    },
                );
            }
        }
    }
}

type BoundedDFSSolution<T> = (Option<(T, Vec<Transition>)>, bool);

/// Preforms depth-first search to find a solution better than the current primal bound.
pub fn bounded_dfs<T: variable_type::Numeric>(
    node: RecursiveNode<T>,
    model: &dypdl::Model,
    generator: &SuccessorGenerator<Transition>,
    prob: &mut StateRegistry<T, RecursiveNode<T>>,
    primal_bound: Option<T>,
    time_keeper: &Option<solver::TimeKeeper>,
    expanded: &mut usize,
) -> BoundedDFSSolution<T> {
    let state = node.state;
    let cost = node.cost;
    *expanded += 1;
    if model.is_goal(&state) {
        if model.reduce_function == ReduceFunction::Max
            && primal_bound.is_some()
            && cost <= primal_bound.unwrap()
        {
            return (None, false);
        }
        return (Some((cost, Vec::new())), false);
    }
    if prob.get(&state, cost).is_some() {
        return (None, false);
    }
    if time_keeper
        .as_ref()
        .map_or(false, |time_keeper| time_keeper.check_time_limit())
    {
        return (None, true);
    }
    for transition in generator.applicable_transitions(&state) {
        let cost = transition.eval_cost(cost, &state, &model.table_registry);
        if model.reduce_function == ReduceFunction::Min
            && primal_bound.is_some()
            && cost >= primal_bound.unwrap()
        {
            continue;
        }
        let successor = transition.apply(&state, &model.table_registry);
        if model.check_constraints(&successor) {
            let node = RecursiveNode {
                state: successor,
                cost,
            };
            let result = bounded_dfs(
                node,
                model,
                generator,
                prob,
                primal_bound,
                time_keeper,
                expanded,
            );
            match result {
                (Some((cost, mut transitions)), _) => {
                    transitions.push(transition.as_ref().clone());
                    return (Some((cost, transitions)), false);
                }
                (_, true) => return (None, true),
                _ => {}
            }
        }
    }
    let constructor = |state: StateInRegistry, cost: T, _: Option<&RecursiveNode<T>>| {
        Some(RecursiveNode { state, cost })
    };
    prob.insert(state, cost, constructor);
    (None, false)
}
