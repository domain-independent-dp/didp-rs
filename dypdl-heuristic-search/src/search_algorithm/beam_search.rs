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
    /// Keep nodes in all layers for duplicate detection.
    ///
    /// Beam search searches layer by layer, where the i th layer contains states that can be reached with i transitions.
    /// By default, this solver only keeps states in the current layer to check for duplicates.
    /// If `keep_all_layers` is `true`, this solver keeps states in all layers to check for duplicates.
    pub keep_all_layers: bool,
    /// Common parameters.
    pub parameters: util::Parameters<T>,
}

/// Performs beam search.
///
/// This function uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// The f-value, the priority of a node, is computed by f_evaluator, which is a function of the g-value, the h-value, and the state.
/// The h-value is computed by h_evaluator.
/// At each depth, the top beam_size nodes minimizing (maximizing) the f-values are kept if `maximize == false` (`true`).
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{beam_search, BeamSearchParameters, Search};
/// use dypdl_heuristic_search::search_algorithm::data_structure::beam::Beam;
/// use dypdl_heuristic_search::search_algorithm::data_structure::{
///     BeamSearchNode, BeamSearchProblemInstance, TransitionWithCustomCost,
/// };
/// use dypdl_heuristic_search::search_algorithm::data_structure::successor_generator::{
///     SuccessorGenerator
/// };
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_variable("variable", 0).unwrap();
/// model.add_base_case(
///     vec![Condition::comparison_i(ComparisonOperator::Ge, variable, 1)]
/// ).unwrap();
/// let mut increment = Transition::new("increment");
/// increment.set_cost(IntegerExpression::Cost + 1);
/// increment.add_effect(variable, variable + 1).unwrap();
/// model.add_forward_transition(increment.clone()).unwrap();
///
/// let h_evaluator = |_: &_| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
///
/// let model = Rc::new(model);
/// let generator = SuccessorGenerator::from_model_without_custom_cost(model.clone(), false);
/// let problem = BeamSearchProblemInstance {
///     target: model.target.clone().into(),
///     generator,
///     cost: 0,
///     g: 0,
///     solution_suffix: &[],
/// };
/// let parameters = BeamSearchParameters { beam_size: 1, ..Default::default() };
/// let beam_constructor = |beam_size| Beam::<_, _, BeamSearchNode<_, _>>::new(beam_size);
/// let (solution, dual_bound) = beam_search(&problem, &beam_constructor, h_evaluator, f_evaluator, parameters);
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions.len(), 1);
/// assert_eq!(Transition::from(solution.transitions[0].clone()), increment);
/// assert!(!solution.is_infeasible);
/// assert_eq!(dual_bound, Some(1));
/// ```
pub fn beam_search<T, U, I, B, C, H, F>(
    problem: &BeamSearchProblemInstance<T, U>,
    beam_constructor: &C,
    h_evaluator: H,
    f_evaluator: F,
    parameters: BeamSearchParameters<T, U>,
) -> (Solution<T, TransitionWithCustomCost>, Option<U>)
where
    T: variable_type::Numeric,
    U: variable_type::Numeric + Ord,
    I: InformationInBeam<T, U> + CustomCostNodeInterface<T, U>,
    B: BeamInterface<T, U, I>,
    C: Fn(usize) -> B,
    H: Fn(&StateInRegistry) -> Option<U>,
    F: Fn(U, U, &StateInRegistry) -> U,
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

    let f = if let Some(f) = h_evaluator(&initial_state).and_then(|h| {
        let f = f_evaluator(g, h, &initial_state);

        if f_bound.map_or(false, |bound| {
            (maximize && f <= bound) || (!maximize && f >= bound)
        }) {
            None
        } else {
            Some(f)
        }
    }) {
        f
    } else {
        return (
            Solution {
                is_infeasible: true,
                ..Default::default()
            },
            None,
        );
    };

    let mut f_dual_bound = f;
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
        let mut incumbent: Option<(Rc<I>, T, _)> = None;
        let mut peak_f = None;

        for node in current_beam.drain() {
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

                let solution = incumbent.map_or_else(
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

                return (solution, Some(f_dual_bound));
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
                    if peak_f.map_or(true, |peak_f| {
                        (maximize && f > peak_f) || (!maximize && f < peak_f)
                    }) {
                        peak_f = Some(f);
                    }

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

        if let Some(peak_f) = peak_f {
            if !pruned
                && ((maximize && f_dual_bound > peak_f) || (!maximize && f_dual_bound < peak_f))
            {
                f_dual_bound = peak_f;
            }
        }

        if let Some((node, cost, suffix)) = &incumbent {
            let mut transitions = node.transitions();
            transitions.extend_from_slice(suffix);

            return (
                Solution {
                    cost: Some(*cost),
                    transitions,
                    expanded,
                    generated,
                    is_optimal: !pruned && next_beam.is_empty(),
                    time: time_keeper.elapsed_time(),
                    ..Default::default()
                },
                Some(f_dual_bound),
            );
        }

        mem::swap(&mut current_beam, &mut next_beam);

        if !keep_all_layers {
            registry.clear();
        }
    }

    (
        Solution {
            expanded,
            generated,
            is_infeasible: !pruned,
            time: time_keeper.elapsed_time(),
            ..Default::default()
        },
        if pruned { Some(f_dual_bound) } else { None },
    )
}
