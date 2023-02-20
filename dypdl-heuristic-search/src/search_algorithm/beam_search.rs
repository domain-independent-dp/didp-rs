use super::data_structure::beam::{BeamInterface, InformationInBeam};
use super::data_structure::state_registry::{StateInRegistry, StateRegistry};
use super::data_structure::{
    exceed_bound, BeamSearchProblemInstance, CustomCostNodeInterface, CustomCostParent,
    TransitionWithCustomCost,
};
use super::rollout::rollout;
use super::search::Solution;
use super::util;
use dypdl::variable_type;
use std::mem;
use std::rc::Rc;

/// Parameters for beam search.
#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub struct BeamSearchParameters<T, U> {
    /// Beam size.
    pub beam_size: usize,
    /// Maximize.
    pub maximize: bool,
    /// Use f-values for pruning.
    pub f_pruning: bool,
    /// Bound for the f-value.
    pub f_bound: Option<U>,
    /// Keep nodes in all layers.
    pub keep_all_layers: bool,
    /// Common parameters.
    pub parameters: util::Parameters<T>,
}

/// Performs beam search.
///
/// The f-value, the priority of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
/// At each depth, the top beam_size nodes minimizing (maximizing) the f-values are kept if maximize = false (true).
pub fn beam_search<T, U, I, B, C, H, F>(
    problem: &BeamSearchProblemInstance<T, U>,
    beam_constructor: &C,
    h_evaluator: H,
    f_evaluator: F,
    parameters: BeamSearchParameters<T, U>,
) -> Solution<T, TransitionWithCustomCost>
where
    T: variable_type::Numeric,
    U: variable_type::Numeric + Ord,
    I: InformationInBeam<T, U> + CustomCostNodeInterface<T, U>,
    B: BeamInterface<T, U, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<U>,
    F: Fn(U, U, &StateInRegistry, &dypdl::Model) -> U,
{
    let time_keeper = parameters
        .parameters
        .time_limit
        .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
    let quiet = parameters.parameters.quiet;
    let maximize = parameters.maximize;
    let mut primal_bound = parameters.parameters.primal_bound;
    let f_bound = if parameters.f_pruning {
        parameters.f_bound
    } else {
        None
    };
    let keep_all_layers = parameters.keep_all_layers;

    let model = &problem.generator.model;
    let generator = &problem.generator;
    let mut current_beam = beam_constructor(parameters.beam_size);
    let mut next_beam = beam_constructor(parameters.beam_size);
    let mut registry = StateRegistry::new(model.clone());
    registry.reserve(current_beam.capacity());

    let cost = problem.cost;
    let g = problem.g;
    let initial_state = problem.target.clone();

    let h = h_evaluator(&initial_state, model);

    if h.is_none() {
        return Solution {
            is_infeasible: true,
            ..Default::default()
        };
    }

    let h = h.unwrap();
    let f = f_evaluator(g, h, &initial_state, model);

    if f_bound.map_or(false, |bound| {
        (maximize && f <= bound) || (!maximize && f >= bound)
    }) {
        return Solution {
            is_infeasible: true,
            ..Default::default()
        };
    }

    let (g, f) = if maximize { (-g, -f) } else { (g, f) };
    let constructor = |state, cost, _: Option<&_>| Some(I::new(g, f, state, cost, None, None));
    current_beam.insert(&mut registry, initial_state, cost, g, f, constructor);

    if !keep_all_layers {
        registry.clear();
    }

    let mut expanded = 0;
    let mut generated = 1;
    let mut pruned = false;

    while !current_beam.is_empty() {
        let mut incumbent: Option<(Rc<I>, T)> = None;

        for node in current_beam.drain() {
            if let Some(result) = rollout(node.state(), node.cost(), problem.solution_suffix, model)
            {
                if result.is_base {
                    if !exceed_bound(model, result.cost, primal_bound) {
                        primal_bound = Some(node.cost());
                        incumbent = Some((node, result.cost));
                    }

                    continue;
                }
            }

            if time_keeper.check_time_limit() {
                if !quiet {
                    println!("Reached time limit.");
                }

                return incumbent.map_or_else(
                    || Solution {
                        expanded,
                        generated,
                        time: time_keeper.elapsed_time(),
                        time_out: true,
                        ..Default::default()
                    },
                    |(node, cost)| {
                        let mut transitions = node.transitions();
                        transitions.extend_from_slice(problem.solution_suffix);
                        Solution {
                            cost: Some(cost),
                            transitions,
                            expanded,
                            generated,
                            time: time_keeper.elapsed_time(),
                            time_out: true,
                            ..Default::default()
                        }
                    },
                );
            }

            if pruned && incumbent.is_some() {
                continue;
            }

            expanded += 1;

            let parent = CustomCostParent {
                state: node.state(),
                cost: node.cost(),
                g: if maximize { -node.g() } else { node.g() },
            };

            for transition in generator.applicable_transitions(node.state()) {
                if let Some((state, cost, g, f)) = transition.generate_successor_state(
                    &parent,
                    model,
                    f_bound,
                    &h_evaluator,
                    &f_evaluator,
                    maximize,
                ) {
                    let (g, f) = if maximize { (-g, -f) } else { (g, f) };
                    let constructor = |state, cost, _: Option<&_>| {
                        Some(I::new(g, f, state, cost, Some(&node), Some(transition)))
                    };
                    let (new_generated, beam_pruning) =
                        next_beam.insert(&mut registry, state, cost, g, f, constructor);

                    if !pruned && beam_pruning {
                        pruned = true;
                    }

                    if new_generated {
                        generated += 1;
                    }
                }
            }
        }

        if !quiet {
            println!("Expanded: {}", expanded);
        }

        if let Some((node, cost)) = &incumbent {
            let mut transitions = node.transitions();
            transitions.extend_from_slice(problem.solution_suffix);

            return Solution {
                cost: Some(*cost),
                transitions,
                expanded,
                generated,
                is_optimal: !pruned && next_beam.is_empty(),
                time: time_keeper.elapsed_time(),
                ..Default::default()
            };
        }

        mem::swap(&mut current_beam, &mut next_beam);

        if !keep_all_layers {
            registry.clear();
        }
    }

    Solution {
        expanded,
        generated,
        is_infeasible: !pruned,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    }
}
