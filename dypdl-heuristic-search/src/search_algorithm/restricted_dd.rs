use crate::TransitionWithCustomCost;

use super::beam_search::BeamSearchParameters;
use super::data_structure::beam::InBeam;
use super::data_structure::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::data_structure::{
    exceed_bound, BeamSearchNode, BeamSearchProblemInstance, CustomCostNodeInterface,
    CustomCostParent, PrioritizedNode, TransitionChainInterface,
};
use super::rollout::rollout;
use super::search::Solution;
use super::util;
use dypdl::variable_type;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::cmp;
use std::fmt;
use std::mem;
use std::rc::Rc;

pub fn restricted_dd<T, H, F>(
    problem: &BeamSearchProblemInstance<T, T>,
    h_evaluator: H,
    f_evaluator: F,
    keep_probability: f64,
    best_solution: Option<&[TransitionWithCustomCost]>,
    rng: &mut Pcg64Mcg,
    parameters: BeamSearchParameters<T, T>,
) -> Solution<T, TransitionWithCustomCost>
where
    T: variable_type::Numeric + fmt::Display + Ord,
    H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
    F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
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
    let mut current_beam = Vec::<Rc<BeamSearchNode<T, T>>>::with_capacity(parameters.beam_size);
    let mut next_beam = Vec::with_capacity(parameters.beam_size);
    let mut registry = StateRegistry::new(model.clone());
    registry.reserve(current_beam.capacity());

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
    let constructor =
        |state, cost, _: Option<&_>| Some(BeamSearchNode::new(g, f, state, cost, None, None));
    let (node, _) = registry
        .insert(initial_state, problem.cost, constructor)
        .unwrap();

    current_beam.push(node);

    if !keep_all_layers {
        registry.clear();
    }

    let mut expanded = 0;
    let mut generated = 1;
    let mut pruned = false;
    let mut i = 0;

    while !current_beam.is_empty() {
        let mut incumbent: Option<(Rc<BeamSearchNode<T, T>>, T, _)> = None;

        for node in current_beam.drain(..) {
            if let Some(result) = rollout(node.state(), node.cost(), problem.solution_suffix, model)
            {
                if result.is_base {
                    if !exceed_bound(model, result.cost, primal_bound) {
                        primal_bound = Some(node.cost());
                        incumbent = Some((node, result.cost, result.transitions));
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
                    |(node, cost, suffix)| {
                        let mut transitions = node.transitions();
                        transitions.extend_from_slice(suffix);
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
                        Some(BeamSearchNode::new(
                            g,
                            f,
                            state,
                            cost,
                            Some(&node),
                            Some(transition),
                        ))
                    };

                    if let Some((successor, dominated)) = registry.insert(state, cost, constructor)
                    {
                        if let Some(dominated) = dominated {
                            if dominated.in_beam() {
                                dominated.remove_from_beam();
                            }
                        } else {
                            generated += 1;
                        }

                        next_beam.push(successor);
                    }
                }
            }
        }

        if !quiet {
            println!("Expanded: {}", expanded);
        }

        next_beam.retain(|node| node.in_beam());

        if let Some((node, cost, suffix)) = &incumbent {
            let mut transitions = node.transitions();
            transitions.extend_from_slice(suffix);

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

        if next_beam.len() > parameters.beam_size {
            let best_transition = best_solution.and_then(|solution| solution.get(i));
            restrict(
                &mut next_beam,
                best_transition,
                parameters.beam_size,
                keep_probability,
                rng,
            );

            if !pruned {
                pruned = true;
            }
        }

        i += 1;

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

fn restrict<T>(
    beam: &mut Vec<Rc<BeamSearchNode<T, T>>>,
    best_transition: Option<&TransitionWithCustomCost>,
    beam_width: usize,
    keep_probability: f64,
    rng: &mut Pcg64Mcg,
) where
    T: variable_type::Numeric + fmt::Display + Ord,
{
    let mut frontier = 0;
    let size = beam.len();

    for k in 0..size {
        let node = &beam[k];

        let must_keep = if let (Some(transitions), Some(best_transition)) =
            (&node.transitions, &best_transition)
        {
            transitions.last().id == best_transition.id
        } else {
            false
        };

        if must_keep || rng.gen::<f64>() < keep_probability {
            beam.swap(frontier, k);
            frontier += 1;
        }
    }

    let (keep, candidates) = beam.split_at_mut(frontier);
    candidates.sort();
    let len = cmp::max(keep.len(), beam_width);
    beam.truncate(len)
}
