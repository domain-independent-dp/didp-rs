use super::beam_search::BeamSearchParameters;
use super::data_structure::{
    exceed_bound, BfsNode, ParentAndChildStateFunctionCache, StateRegistry, TransitionWithId,
};
use super::rollout::get_solution_cost_and_suffix;
use super::search::{SearchInput, Solution};
use super::util;
use dypdl::{variable_type, TransitionInterface};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::cmp;
use std::fmt::Display;
use std::mem;
use std::rc::Rc;

/// Parameters for constructing a randomized restricted DD.
#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub struct RandomizedRestrictedDDParameters<'a, T, V: TransitionInterface> {
    /// Probability to keep a non-best node.
    pub keep_probability: f64,
    /// Best solution.
    pub best_solution: Option<&'a [TransitionWithId<V>]>,
    /// Parameters for beam search.
    pub beam_search_parameters: BeamSearchParameters<T>,
}

/// Constructs a randomized restricted decision diagram (DD).
///
/// This function uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It keeps the best `beam_size` nodes at each layer.
///
/// Type parameter `N` is a node type that implements `BfsNode`.
/// Type parameter `E` is a type of a function that evaluates a transition and insert a successor node into a state registry.
/// The last argument of the function is the primal bound of the solution cost.
/// Type parameter `B` is a type of a function that combines the g-value (the cost to a state) and the base cost.
/// It should be the same function as the cost expression, e.g., `cost + base_cost` for `cost + w`.
///
/// # References
///
/// Xavier Gillard and Pierre Schaus. "Large Neighborhood Search with Decision Diagrams,"
/// Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI), 2022.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::Search;
/// use dypdl_heuristic_search::search_algorithm::{
///     FNode, SearchInput, randomized_restricted_dd, RandomizedRestrictedDDParameters,
///     SuccessorGenerator, TransitionWithId,
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
/// let model = Rc::new(model);
///
/// let state = model.target.clone();
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let cost = 0;
/// let h_evaluator = |_: &_, _: &mut _| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
/// let primal_bound = None;
/// let node = FNode::<_, TransitionWithId>::generate_root_node(
///     state,
///     &mut function_cache,
///     cost,
///     &model,
///     &h_evaluator,
///     &f_evaluator,
///     primal_bound,
/// );
/// let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);
/// let input = SearchInput::<_, TransitionWithId> {
///     node,
///     generator,
///     solution_suffix: &[],
/// };
/// let transition_evaluator =
///     move |node: &FNode<_, TransitionWithId>, transition, cache: &mut _, registry: &mut _, primal_bound| {
///         node.insert_successor_node(
///             transition,
///             cache,
///             registry,
///             &h_evaluator,
///             &f_evaluator,
///             primal_bound,
///         )
///     };
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let parameters = RandomizedRestrictedDDParameters::default();
/// let mut rng = rand_pcg::Pcg64Mcg::new(0);
/// let solution = randomized_restricted_dd(
///     &input, transition_evaluator, base_cost_evaluator, parameters, &mut rng,
/// );
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions.len(), 1);
/// assert_eq!(Transition::from(solution.transitions[0].clone()), increment);
/// assert!(!solution.is_infeasible);
/// ```
pub fn randomized_restricted_dd<'a, T, N, E, B, V>(
    input: &'a SearchInput<'a, N, TransitionWithId<V>>,
    mut transition_evaluator: E,
    mut base_cost_evaluator: B,
    parameters: RandomizedRestrictedDDParameters<'a, T, V>,
    rng: &mut Pcg64Mcg,
) -> Solution<T, TransitionWithId<V>>
where
    T: variable_type::Numeric + Ord + Display,
    N: BfsNode<T, TransitionWithId<V>> + Clone,
    E: FnMut(
        &N,
        Rc<TransitionWithId<V>>,
        &mut ParentAndChildStateFunctionCache,
        &mut StateRegistry<T, N>,
        Option<T>,
    ) -> Option<(Rc<N>, bool)>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
{
    let time_keeper = parameters
        .beam_search_parameters
        .parameters
        .time_limit
        .map_or_else(util::TimeKeeper::default, util::TimeKeeper::with_time_limit);
    let quiet = parameters.beam_search_parameters.parameters.quiet;
    let mut primal_bound = parameters.beam_search_parameters.parameters.primal_bound;
    let beam_size = parameters.beam_search_parameters.beam_size;
    let keep_all_layers = parameters.beam_search_parameters.keep_all_layers;

    let model = &input.generator.model;
    let generator = &input.generator;
    let suffix = input.solution_suffix;
    let mut current_beam = Vec::<Rc<N>>::with_capacity(beam_size);
    let mut next_beam = Vec::with_capacity(beam_size);
    let mut registry = StateRegistry::new(model.clone());
    registry.reserve(current_beam.capacity());

    let node = if let Some(node) = input.node.clone() {
        node
    } else {
        return Solution {
            is_infeasible: true,
            ..Default::default()
        };
    };

    let result = registry.insert(node);

    current_beam.push(result.information.unwrap());

    if !keep_all_layers {
        registry.clear();
    }

    let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
    let mut applicable_transitions = Vec::new();

    let mut expanded = 0;
    let mut generated = 1;
    let mut pruned = false;
    let mut best_dual_bound = None;
    let mut layer_index = 0;

    while !current_beam.is_empty() {
        let mut incumbent = None;
        let mut layer_dual_bound = None;
        let previously_pruned = pruned;

        for node in current_beam.drain(..) {
            if let Some(dual_bound) = node.bound(model) {
                if exceed_bound(model, dual_bound, primal_bound) {
                    continue;
                }
            }

            function_cache.parent.clear();

            if let Some((cost, suffix)) = get_solution_cost_and_suffix(
                model,
                &*node,
                suffix,
                &mut base_cost_evaluator,
                &mut function_cache,
            ) {
                if !exceed_bound(model, cost, primal_bound) {
                    primal_bound = Some(cost);
                    incumbent = Some((node, cost, suffix));

                    if Some(cost) == best_dual_bound {
                        break;
                    }
                }
                continue;
            }

            if time_keeper.check_time_limit(quiet) {
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
                            best_bound: best_dual_bound,
                            transitions,
                            expanded,
                            generated,
                            time: time_keeper.elapsed_time(),
                            time_out: true,
                            ..Default::default()
                        }
                    },
                );

                return solution;
            }

            if pruned && incumbent.is_some() {
                continue;
            }

            expanded += 1;
            generator.generate_applicable_transitions(
                node.state(),
                &mut function_cache.parent,
                &mut applicable_transitions,
            );

            for transition in applicable_transitions.drain(..) {
                if let Some((successor, new_generated)) = transition_evaluator(
                    &node,
                    transition,
                    &mut function_cache,
                    &mut registry,
                    primal_bound,
                ) {
                    if let Some(bound) = successor.bound(model) {
                        if !exceed_bound(model, bound, layer_dual_bound) {
                            layer_dual_bound = Some(bound);
                        }
                    }

                    next_beam.push(successor);

                    if new_generated {
                        generated += 1;
                    }
                }
            }
        }

        if !quiet {
            println!(
                "Searched layer: {}, expanded: {}, elapsed time: {}",
                layer_index,
                expanded,
                time_keeper.elapsed_time()
            );
        }

        if let (false, Some(value)) = (previously_pruned, layer_dual_bound) {
            if exceed_bound(model, value, primal_bound) {
                best_dual_bound = primal_bound;
            } else if best_dual_bound.map_or(true, |bound| !exceed_bound(model, bound, Some(value)))
            {
                best_dual_bound = layer_dual_bound;
            }
        }

        if let Some((node, cost, suffix)) = &incumbent {
            let mut transitions = node.transitions();
            transitions.extend_from_slice(suffix);
            let is_optimal = (!pruned && next_beam.is_empty()) || Some(*cost) == best_dual_bound;

            return Solution {
                cost: Some(*cost),
                best_bound: if is_optimal {
                    Some(*cost)
                } else {
                    best_dual_bound
                },
                transitions,
                expanded,
                generated,
                is_optimal,
                time: time_keeper.elapsed_time(),
                ..Default::default()
            };
        }

        next_beam.retain(|node| !node.is_closed());

        if next_beam.len() > beam_size {
            let best_transition = parameters
                .best_solution
                .and_then(|solution| solution.get(layer_index));
            restrict(
                &mut next_beam,
                best_transition,
                beam_size,
                parameters.keep_probability,
                rng,
            );

            if !pruned {
                pruned = true;
            }
        }

        mem::swap(&mut current_beam, &mut next_beam);

        if !keep_all_layers {
            registry.clear();
        }

        layer_index += 1;
    }

    Solution {
        expanded,
        generated,
        is_infeasible: !pruned,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    }
}

fn restrict<T, N, V>(
    beam: &mut Vec<Rc<N>>,
    best_transition: Option<&TransitionWithId<V>>,
    beam_size: usize,
    keep_probability: f64,
    rng: &mut Pcg64Mcg,
) where
    T: variable_type::Numeric + Ord + Display,
    N: BfsNode<T, TransitionWithId<V>> + Clone,
    V: TransitionInterface + Clone + Default,
{
    let mut frontier = 0;
    let size = beam.len();

    for k in 0..size {
        let node = &beam[k];

        let must_keep =
            if let (Some(transition), Some(best_transition)) = (&node.last(), &best_transition) {
                transition.id == best_transition.id
            } else {
                false
            };

        if must_keep || rng.random::<f64>() < keep_probability {
            beam.swap(frontier, k);
            frontier += 1;
        }
    }

    let (keep, candidates) = beam.split_at_mut(frontier);
    candidates.sort_by(|a, b| b.cmp(a));
    let len = cmp::max(keep.len(), beam_size);
    beam.truncate(len)
}
