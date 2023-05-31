use super::data_structure::ConcurrentStateRegistry;
use crate::search_algorithm::data_structure::{exceed_bound, HashableSignatureVariables};
use crate::search_algorithm::util::TimeKeeper;
use crate::search_algorithm::{
    get_solution_cost_and_suffix, BeamSearchParameters, BfsNode, SearchInput, Solution,
};
use dypdl::{variable_type, Model, ReduceFunction, TransitionInterface};
use rayon::prelude::*;
use std::error::Error;
use std::fmt::Display;
use std::sync::Arc;

/// Performs shared memory beam search.
///
/// This function uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// It keeps the best `beam_size` nodes at each layer.
///
/// Type parameter `N` is a node type that implements `BfsNode`.
/// Type parameter `E` is a type of a function that evaluates a transition and generate a successor node.
/// The last argument of the function is the primal bound of the solution cost.
/// Type parameter `B` is a type of a function that combines the g-value (the cost to a state) and the base cost.
/// It should be the same function as the cost expression, e.g., `cost + base_cost` for `cost + w`.
///
/// # Panics
///
/// If it fails to create a thread pool or reserve memory for the state registry.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::Search;
/// use dypdl_heuristic_search::parallel_search_algorithm::{
///     shared_memory_beam_search, SendableFNode,
/// };
/// use dypdl_heuristic_search::search_algorithm::{
///     BeamSearchParameters, SearchInput, SuccessorGenerator,
/// };
/// use std::sync::Arc;
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
/// let model = Arc::new(model);
///
/// let state = model.target.clone();
/// let cost = 0;
/// let h_evaluator = |_: &_| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
/// let primal_bound = None;
/// let node = SendableFNode::generate_root_node(
///     state,
///     cost,
///     &model,
///     &h_evaluator,
///     &f_evaluator,
///     primal_bound,
/// );
/// let generator = SuccessorGenerator::<Transition, Arc<Transition>, Arc<_>>::from_model(
///     model.clone(), false,
/// );
/// let input = SearchInput {
///     node,
///     generator,
///     solution_suffix: &[],
/// };
/// let transition_evaluator =
///     move |node: &SendableFNode<_>, transition, primal_bound| {
///         node.generate_successor_node(
///             transition,
///             &model,
///             &h_evaluator,
///             &f_evaluator,
///             primal_bound,
///         )
/// };
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let parameters = BeamSearchParameters::default();
/// let threads = 1;
/// let solution = shared_memory_beam_search(
///     &input, transition_evaluator, base_cost_evaluator, parameters, threads,
/// ).unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions.len(), 1);
/// assert_eq!(Transition::from(solution.transitions[0].clone()), increment);
/// assert!(!solution.is_infeasible);
/// ```
pub fn shared_memory_beam_search<'a, T, N, E, B, V>(
    input: &'a SearchInput<'a, N, V, Arc<V>, Arc<Model>>,
    transition_evaluator: E,
    base_cost_evaluator: B,
    parameters: BeamSearchParameters<T>,
    threads: usize,
) -> Result<Solution<T, V>, Box<dyn Error>>
where
    T: variable_type::Numeric + Ord + Display + Send + Sync,
    N: BfsNode<T, V, Arc<HashableSignatureVariables>> + Clone + Send + Sync,
    Arc<N>: Send + Sync,
    E: Fn(&N, Arc<V>, Option<T>) -> Option<N> + Send + Sync,
    B: Fn(T, T) -> T + Send + Sync,
    V: TransitionInterface + Clone + Default + Send + Sync,
    Arc<V>: Send + Sync,
    &'a [V]: Clone + Send + Sync,
{
    let time_keeper = parameters
        .parameters
        .time_limit
        .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
    let beam_size = parameters.beam_size;
    let quiet = parameters.parameters.quiet;
    let mut primal_bound = parameters.parameters.primal_bound;

    let model = &input.generator.model;
    let generator = &input.generator;
    let suffix = input.solution_suffix;

    let mut beam = Vec::with_capacity(beam_size);
    let capacity = parameters
        .parameters
        .initial_registry_capacity
        .unwrap_or_else(|| beam.capacity());
    let shard_amount = (threads * 4).next_power_of_two();
    let mut registry = ConcurrentStateRegistry::with_capacity_and_shard_amount(
        model.clone(),
        capacity,
        shard_amount,
    );

    let node = if let Some(node) = input.node.clone() {
        node
    } else {
        return Ok(Solution {
            is_infeasible: true,
            ..Default::default()
        });
    };

    let mut best_dual_bound = node.bound(model);
    let (node, _) = registry.insert(node).unwrap();
    beam.push(node);

    if !parameters.keep_all_layers {
        registry.clear();
    }

    let mut successors = Vec::with_capacity(beam_size);
    let mut non_dominated_successors = Vec::with_capacity(beam_size);
    let mut nodes_with_goal_information = Vec::with_capacity(beam_size);

    let mut expanded = 0;
    let mut generated = 1;
    let mut pruned = false;
    let mut layer_index = 0;

    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;

    while !beam.is_empty() {
        let beam_len = beam.len();

        let goal = thread_pool.install(|| {
            nodes_with_goal_information.par_extend(beam.par_drain(..).filter_map(|node| {
                if let Some((cost, suffix)) =
                    get_solution_cost_and_suffix(model, &*node, suffix, &base_cost_evaluator)
                {
                    if !exceed_bound(model, cost, primal_bound) {
                        Some((node, Some((cost, suffix))))
                    } else {
                        None
                    }
                } else {
                    Some((node, None))
                }
            }));

            let goals = nodes_with_goal_information
                .par_iter()
                .filter_map(|(node, result)| {
                    if let Some((cost, suffix)) = result {
                        Some((node, cost, suffix))
                    } else {
                        None
                    }
                });

            if let Some((node, cost, suffix)) = if model.reduce_function == ReduceFunction::Min {
                goals.min_by_key(|goal| goal.1)
            } else {
                goals.max_by_key(|goal| goal.1)
            } {
                let mut transitions = N::transitions(node);
                transitions.extend_from_slice(suffix);
                primal_bound = Some(*cost);
                Some((*cost, transitions))
            } else {
                None
            }
        });

        if time_keeper.check_time_limit(quiet) {
            return Ok(goal.map_or_else(
                || Solution {
                    expanded,
                    generated,
                    time: time_keeper.elapsed_time(),
                    time_out: true,
                    ..Default::default()
                },
                |(cost, transitions)| Solution {
                    cost: Some(cost),
                    best_bound: best_dual_bound,
                    transitions,
                    expanded,
                    generated,
                    time: time_keeper.elapsed_time(),
                    time_out: true,
                    ..Default::default()
                },
            ));
        }

        if !pruned || goal.is_none() {
            if goal.is_none() {
                expanded += beam_len;
            } else {
                thread_pool.install(|| {
                    expanded += nodes_with_goal_information
                        .par_iter()
                        .filter(|(_, result)| result.is_none())
                        .count();
                });
            }

            thread_pool.install(|| {
                successors.par_extend(
                    nodes_with_goal_information
                        .par_iter()
                        .filter_map(|(node, result)| {
                            if result.is_none() {
                                Some(generator.applicable_transitions(node.state()).filter_map(
                                    |transition| {
                                        transition_evaluator(node, transition, primal_bound)
                                            .and_then(|successor| {
                                                registry.insert(successor).and_then(
                                                    |(successor, dominated)| {
                                                        if let Some(dominated) = dominated {
                                                            if !dominated.is_closed() {
                                                                dominated.close();
                                                            }
                                                        }

                                                        if !successor.is_closed() {
                                                            Some(successor)
                                                        } else {
                                                            None
                                                        }
                                                    },
                                                )
                                            })
                                    },
                                ))
                            } else {
                                None
                            }
                        })
                        .flatten_iter(),
                );
                nodes_with_goal_information.clear();
                non_dominated_successors
                    .par_extend(successors.par_drain(..).filter(|node| !node.is_closed()));

                generated += non_dominated_successors.len();

                if non_dominated_successors.len() > beam_size {
                    non_dominated_successors.par_sort_unstable_by(|a, b| b.cmp(a));

                    if !pruned {
                        if let (true, Some(value)) = (
                            N::ordered_by_bound(),
                            non_dominated_successors
                                .first()
                                .and_then(|node| node.bound(model)),
                        ) {
                            if best_dual_bound
                                .map_or(true, |bound| !exceed_bound(model, bound, Some(value)))
                            {
                                best_dual_bound = Some(value);
                            }
                        }

                        pruned = true;
                    }

                    beam.par_extend(non_dominated_successors.par_drain(..beam_size));
                    non_dominated_successors.clear();
                } else {
                    beam.par_extend(non_dominated_successors.par_drain(..));
                }
            })
        }

        if !quiet {
            println!(
                "Searched layer: {}, expanded: {}, elapsed time: {}",
                layer_index,
                expanded,
                time_keeper.elapsed_time()
            );
        }

        if let Some((cost, transitions)) = goal {
            let is_optimal = !pruned && beam.is_empty();

            return Ok(Solution {
                cost: Some(cost),
                best_bound: if is_optimal {
                    Some(cost)
                } else {
                    best_dual_bound
                },
                transitions,
                expanded,
                generated,
                is_optimal,
                time: time_keeper.elapsed_time(),
                ..Default::default()
            });
        }

        if !parameters.keep_all_layers {
            registry.clear();
        }

        layer_index += 1;
    }

    Ok(Solution {
        is_infeasible: !pruned,
        best_bound: if pruned { best_dual_bound } else { None },
        expanded,
        generated,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    })
}
