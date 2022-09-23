use crate::beam::{Beam, BeamSearchNodeArgs, InBeam, PrioritizedNode};
use crate::evaluator;
use crate::search_node::{trace_transitions, DPSearchNode};
use crate::solver;
use crate::solver::Solution;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type;
use std::fmt;
use std::mem;
use std::str;

pub struct EvaluatorsForBeamSearch<H, F> {
    pub h_evaluator: H,
    pub f_evaluator: F,
}

/// Performs multiple iterations of forward beam search using given beam sizes.
pub fn iterative_beam_search<'a, T, U, V, B, C, H, F>(
    model: &'a dypdl::Model,
    generator: &'a SuccessorGenerator<'a, TransitionWithCustomCost>,
    evaluators: &EvaluatorsForBeamSearch<H, F>,
    beam_constructor: &C,
    beam_sizes: &[usize],
    maximize: bool,
    parameters: solver::SolverParameters<T>,
) -> Solution<T>
where
    T: variable_type::Numeric + fmt::Display,
    U: variable_type::Numeric + Ord,
    <U as str::FromStr>::Err: fmt::Debug,
    V: DPSearchNode<T> + InBeam + Ord + StateInformation<T> + PrioritizedNode<U>,
    B: Beam<T, U, V>,
    C: Fn(usize) -> B,
    H: evaluator::Evaluator,
    F: Fn(U, U, &StateInRegistry, &dypdl::Model) -> U,
{
    let time_keeper = parameters.time_limit.map_or_else(
        solver::TimeKeeper::default,
        solver::TimeKeeper::with_time_limit,
    );
    let quiet = parameters.quiet;
    let mut incumbent = Vec::new();
    let mut cost = None;
    let mut expanded = 0;
    let mut generated = 0;
    for beam_size in beam_sizes {
        let beam_constructor = || beam_constructor(*beam_size);
        let parameters = BeamSearchParameters {
            maximize,
            primal_bound: None,
            quiet: parameters.quiet,
        };
        let (result, time_out) = beam_search(
            model,
            generator,
            &beam_constructor,
            evaluators,
            parameters,
            &time_keeper,
        );
        expanded += result.expanded;
        generated += result.expanded;
        match result.cost {
            Some(new_cost) => {
                if let Some(current_cost) = cost {
                    match model.reduce_function {
                        dypdl::ReduceFunction::Max if new_cost > current_cost => {
                            incumbent = result.transitions;
                            cost = Some(new_cost);
                            if !quiet {
                                println!("New primal bound: {}", new_cost);
                            }
                        }
                        dypdl::ReduceFunction::Min if new_cost < current_cost => {
                            incumbent = result.transitions;
                            cost = Some(new_cost);
                            if !quiet {
                                println!("New primal bound: {}", new_cost);
                            }
                        }
                        _ => {}
                    }
                } else {
                    incumbent = result.transitions;
                    cost = Some(new_cost);
                    if !quiet {
                        println!("New primal bound: {}", new_cost);
                    }
                }
            }
            _ => {
                if !quiet {
                    println!("Failed to find a solution.")
                }
            }
        }
        if time_out {
            break;
        }
    }
    solver::Solution {
        cost,
        transitions: incumbent,
        expanded,
        generated,
        ..Default::default()
    }
}

/// Arguments for beam search.
pub struct BeamSearchParameters<T> {
    /// Maximize or not.
    pub maximize: bool,
    /// Primal bound
    pub primal_bound: Option<T>,
    /// Suppress log output or not.
    pub quiet: bool,
}

/// Performs beam search.
///
/// The f-value, the priority of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
/// At each depth, the top beam_size nodes minimizing (maximizing) the f-values are kept if maximize = false (true).
pub fn beam_search<'a, T, U, V, B, C, H, F>(
    model: &'a dypdl::Model,
    generator: &SuccessorGenerator<'a, TransitionWithCustomCost>,
    beam_constructor: &C,
    evaluators: &EvaluatorsForBeamSearch<H, F>,
    parameters: BeamSearchParameters<U>,
    time_keeper: &solver::TimeKeeper,
) -> (Solution<T>, bool)
where
    T: variable_type::Numeric,
    U: variable_type::Numeric + Ord,
    V: DPSearchNode<T> + InBeam + Ord + StateInformation<T> + PrioritizedNode<U>,
    B: Beam<T, U, V>,
    C: Fn() -> B,
    H: evaluator::Evaluator,
    F: Fn(U, U, &StateInRegistry, &dypdl::Model) -> U,
{
    let h_evaluator = &evaluators.h_evaluator;
    let f_evaluator = &evaluators.f_evaluator;
    let maximize = parameters.maximize;
    let primal_bound = parameters.primal_bound;
    let quiet = parameters.quiet;
    let mut current_beam = beam_constructor();
    let mut next_beam = beam_constructor();
    let mut registry = StateRegistry::new(model);
    registry.reserve(current_beam.capacity());

    let cost = T::zero();
    let g = U::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let h = h_evaluator.eval(&initial_state, model);
    if h.is_none() {
        return (Solution::default(), false);
    }
    let h = h.unwrap();
    let f = f_evaluator(g, h, &initial_state, model);

    if let Some(primal_bound) = primal_bound {
        match model.reduce_function {
            dypdl::ReduceFunction::Max if f <= primal_bound => {
                return (
                    Solution {
                        is_infeasible: true,
                        ..Default::default()
                    },
                    false,
                );
            }
            dypdl::ReduceFunction::Min if f >= primal_bound => {
                return (
                    Solution {
                        is_infeasible: true,
                        ..Default::default()
                    },
                    false,
                );
            }
            _ => {}
        }
    }

    let f = if maximize { -f } else { f };
    let args = BeamSearchNodeArgs {
        g,
        f,
        parent: None,
        operator: None,
    };
    current_beam.insert(&mut registry, initial_state, cost, args);
    let mut expanded = 0;
    let mut generated = 0;
    let mut pruned = false;

    while !current_beam.is_empty() {
        let mut incumbent = None;
        for node in current_beam.drain() {
            expanded += 1;
            if model.is_base(node.state()) {
                if let Some(cost) = incumbent.as_ref().map(|x: &V| x.cost()) {
                    match model.reduce_function {
                        dypdl::ReduceFunction::Max if node.cost() > cost => {
                            incumbent = Some(node);
                        }
                        dypdl::ReduceFunction::Min if node.cost() < cost => {
                            incumbent = Some(node);
                        }
                        _ => {}
                    }
                } else {
                    incumbent = Some(node);
                }
                continue;
            }
            if time_keeper.check_time_limit() {
                return (
                    incumbent.map_or_else(
                        || Solution {
                            expanded,
                            generated,
                            time: time_keeper.elapsed_time(),
                            ..Default::default()
                        },
                        |node| Solution {
                            cost: Some(node.cost()),
                            transitions: trace_transitions(node),
                            expanded,
                            generated,
                            time: time_keeper.elapsed_time(),
                            ..Default::default()
                        },
                    ),
                    true,
                );
            }
            for transition in generator.applicable_transitions(node.state()) {
                let state = transition
                    .transition
                    .apply(node.state(), &model.table_registry);
                if model.check_constraints(&state) {
                    if let Some(h) = h_evaluator.eval(&state, model) {
                        let g = transition.custom_cost.eval_cost(
                            node.g(),
                            node.state(),
                            &model.table_registry,
                        );
                        let f = f_evaluator(g, h, &state, model);

                        if let Some(primal_bound) = primal_bound {
                            match model.reduce_function {
                                dypdl::ReduceFunction::Max if f <= primal_bound => continue,
                                dypdl::ReduceFunction::Min if f >= primal_bound => continue,
                                _ => {}
                            }
                        }

                        let f = if maximize { -f } else { f };
                        let cost = transition.transition.eval_cost(
                            node.cost(),
                            node.state(),
                            &model.table_registry,
                        );
                        let args = BeamSearchNodeArgs {
                            g,
                            f,
                            operator: Some(transition),
                            parent: Some(node.clone()),
                        };
                        let inserted = next_beam.insert(&mut registry, state, cost, args);
                        if !pruned && !inserted {
                            pruned = true;
                        }
                        generated += 1;
                    }
                }
            }
        }
        if !quiet {
            println!("Expanded: {}", expanded);
        }
        if let Some(node) = incumbent {
            return (
                Solution {
                    cost: Some(node.cost()),
                    transitions: trace_transitions(node),
                    expanded,
                    generated,
                    time: time_keeper.elapsed_time(),
                    ..Default::default()
                },
                false,
            );
        }
        mem::swap(&mut current_beam, &mut next_beam);
        registry.clear();
    }
    (
        Solution {
            expanded,
            generated,
            is_infeasible: !pruned,
            ..Default::default()
        },
        false,
    )
}
