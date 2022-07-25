use crate::beam_search_node::{Beam, BeamSearchNode, BeamSearchNodeArgs};
use crate::evaluator;
use crate::search_node::trace_transitions;
use crate::solver;
use crate::solver::Solution;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type;
use std::fmt;
use std::mem;
use std::rc::Rc;
use std::str;

/// Performs multiple iterations of forward beam search using given beam sizes.
pub fn iterative_forward_beam_search<'a, T, U, H, F>(
    model: &'a dypdl::Model,
    generator: &'a SuccessorGenerator<'a, TransitionWithCustomCost>,
    h_evaluator: &H,
    f_evaluator: F,
    beam_sizes: &[usize],
    maximize: bool,
    parameters: solver::SolverParameters<T>,
) -> Solution<T>
where
    T: variable_type::Numeric + fmt::Display,
    U: variable_type::Numeric + Ord,
    <U as str::FromStr>::Err: fmt::Debug,
    H: evaluator::Evaluator,
    F: Fn(U, U, &StateInRegistry, &dypdl::Model) -> U,
{
    let time_keeper = parameters.time_limit.map(solver::TimeKeeper::new);
    let quiet = parameters.quiet;
    let mut incumbent = Vec::new();
    let mut cost = None;
    let mut expanded = 0;
    let mut generated = 0;
    for beam_size in beam_sizes {
        let parameters = BeamSearchParameters {
            beam_size: *beam_size,
            maximize,
            quiet: parameters.quiet,
        };
        let (result, time_out) = forward_beam_search(
            model,
            generator,
            h_evaluator,
            &f_evaluator,
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
pub struct BeamSearchParameters {
    /// Beam size.
    beam_size: usize,
    /// Maximize or not.
    maximize: bool,
    /// Suppress log output or not.
    quiet: bool,
}

/// Performs beam search.
///
/// The f-value, the prioarity of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
/// At each depth, the top beam_size nodes minimizing (maximizing) the f-values are kept if maximize = false (true).
pub fn forward_beam_search<'a, T, U, H, F>(
    model: &'a dypdl::Model,
    generator: &SuccessorGenerator<'a, TransitionWithCustomCost>,
    h_evaluator: &H,
    f_evaluator: &F,
    parameters: BeamSearchParameters,
    time_keeper: &Option<solver::TimeKeeper>,
) -> (Solution<T>, bool)
where
    T: variable_type::Numeric,
    U: variable_type::Numeric + Ord,
    H: evaluator::Evaluator,
    F: Fn(U, U, &StateInRegistry, &dypdl::Model) -> U,
{
    let beam_size = parameters.beam_size;
    let maximize = parameters.maximize;
    let quiet = parameters.quiet;
    let mut current_beam = Beam::new(beam_size);
    let mut next_beam = Beam::new(beam_size);
    let mut registry = StateRegistry::new(model);
    registry.reserve(beam_size);

    let cost = T::zero();
    let g = U::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let h = h_evaluator.eval(&initial_state, model);
    if h.is_none() {
        return (Solution::default(), false);
    }
    let h = h.unwrap();
    let f = f_evaluator(g, h, &initial_state, model);
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

    while !current_beam.is_empty() {
        let mut incumbent = None;
        for node in current_beam.drain() {
            expanded += 1;
            if model.is_goal(node.state()) {
                if let Some(cost) = incumbent
                    .as_ref()
                    .map(|x: &Rc<BeamSearchNode<T, U>>| x.cost)
                {
                    match model.reduce_function {
                        dypdl::ReduceFunction::Max if node.cost > cost => {
                            incumbent = Some(node);
                        }
                        dypdl::ReduceFunction::Min if node.cost < cost => {
                            incumbent = Some(node);
                        }
                        _ => {}
                    }
                } else {
                    incumbent = Some(node);
                }
                continue;
            }
            if time_keeper
                .as_ref()
                .map_or(false, |time_keeper| time_keeper.check_time_limit())
            {
                return (
                    incumbent.map_or_else(
                        || Solution {
                            expanded,
                            generated,
                            ..Default::default()
                        },
                        |node| Solution {
                            cost: Some(node.cost()),
                            transitions: trace_transitions(node),
                            expanded,
                            generated,
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
                            node.g,
                            node.state(),
                            &model.table_registry,
                        );
                        let f = f_evaluator(g, h, &state, model);
                        let f = if maximize { -f } else { f };
                        let cost = transition.transition.eval_cost(
                            node.cost,
                            node.state(),
                            &model.table_registry,
                        );
                        let args = BeamSearchNodeArgs {
                            g,
                            f,
                            operator: Some(transition),
                            parent: Some(node.clone()),
                        };
                        next_beam.insert(&mut registry, state, cost, args);
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
            ..Default::default()
        },
        false,
    )
}
