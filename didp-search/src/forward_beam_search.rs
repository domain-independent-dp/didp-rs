use crate::beam_search_node::{Beam, BeamSearchNode, BeamSearchNodeArgs};
use crate::evaluator;
use crate::search_node::trace_transitions;
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use didp_parser::variable;
use std::fmt;
use std::mem;
use std::rc::Rc;
use std::str;

pub fn iterative_forward_beam_search<'a, T, U, H, F>(
    model: &'a didp_parser::Model<T>,
    generator: SuccessorGenerator<'a, TransitionWithCustomCost<T, U>>,
    h_evaluator: H,
    f_evaluator: F,
    beam_sizes: &[usize],
    maximize: bool,
    parameters: solver::SolverParameters<T>,
) -> solver::Solution<T>
where
    T: variable::Numeric + fmt::Display,
    U: variable::Numeric + Ord,
    <U as str::FromStr>::Err: fmt::Debug,
    H: evaluator::Evaluator<U>,
    F: Fn(U, U, &StateInRegistry, &didp_parser::Model<T>) -> U,
{
    let time_keeper = parameters.time_limit.map(solver::TimeKeeper::new);
    let mut incumbent = Vec::new();
    let mut cost = None;
    for beam_size in beam_sizes {
        let (result, time_out) = forward_beam_search(
            model,
            &generator,
            &h_evaluator,
            &f_evaluator,
            *beam_size,
            maximize,
            &time_keeper,
        );
        match result {
            Some((new_cost, new_incumbent)) => {
                if let Some(current_cost) = cost {
                    match model.reduce_function {
                        didp_parser::ReduceFunction::Max if new_cost > current_cost => {
                            incumbent = new_incumbent;
                            cost = Some(new_cost);
                            println!("New primal bound: {}", new_cost);
                        }
                        didp_parser::ReduceFunction::Min if new_cost < current_cost => {
                            incumbent = new_incumbent;
                            cost = Some(new_cost);
                            println!("New primal bound: {}", new_cost);
                        }
                        _ => {}
                    }
                } else {
                    incumbent = new_incumbent;
                    cost = Some(new_cost);
                    println!("New primal bound: {}", new_cost);
                }
            }
            _ => {
                println!("Failed to find a solution;")
            }
        }
        if time_out {
            break;
        }
    }
    solver::Solution {
        cost,
        transitions: incumbent,
        ..Default::default()
    }
}

type BeamSearchResult<T> = (Option<(T, Vec<Rc<didp_parser::Transition<T>>>)>, bool);

pub fn forward_beam_search<'a, T, U, H, F>(
    model: &'a didp_parser::Model<T>,
    generator: &SuccessorGenerator<'a, TransitionWithCustomCost<T, U>>,
    h_evaluator: &H,
    f_evaluator: &F,
    beam_size: usize,
    maximize: bool,
    time_keeper: &Option<solver::TimeKeeper>,
) -> BeamSearchResult<T>
where
    T: variable::Numeric,
    U: variable::Numeric + Ord,
    H: evaluator::Evaluator<U>,
    F: Fn(U, U, &StateInRegistry, &didp_parser::Model<T>) -> U,
{
    let mut current_beam = Beam::new(beam_size);
    let mut next_beam = Beam::new(beam_size);
    let mut registry = StateRegistry::new(model);
    registry.reserve(beam_size);

    let cost = T::zero();
    let g = U::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let h = h_evaluator.eval(&initial_state, model);
    if h.is_none() {
        return (None, false);
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
                        didp_parser::ReduceFunction::Max if node.cost > cost => {
                            incumbent = Some(node);
                        }
                        didp_parser::ReduceFunction::Min if node.cost < cost => {
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
                    incumbent.map(|node| (node.cost(), trace_transitions(node))),
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
                    }
                }
            }
        }
        println!("Expanded: {}", expanded);
        if let Some(node) = incumbent {
            return (Some((node.cost(), trace_transitions(node))), false);
        }
        mem::swap(&mut current_beam, &mut next_beam);
        registry.clear();
    }
    (None, false)
}
