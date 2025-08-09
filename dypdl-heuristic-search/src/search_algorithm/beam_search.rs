use super::data_structure::{
    exceed_bound, Beam, BfsNode, ParentAndChildStateFunctionCache, StateRegistry, TransitionWithId,
};
use super::rollout::get_solution_cost_and_suffix;
use super::search::{Parameters, SearchInput, Solution};
use super::util::TimeKeeper;
use dypdl::{variable_type, TransitionInterface};
use std::fmt::Display;
use std::mem;
use std::rc::Rc;

/// Parameters for beam search.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct BeamSearchParameters<T> {
    /// Beam size, the number of nodes to keep at each layer.
    pub beam_size: usize,
    /// Keep nodes in all layers for duplicate detection.
    ///
    /// Beam search searches layer by layer, where the i th layer contains states that can be reached with i transitions.
    /// By default, beam search only keeps states in the current layer to check for duplicates.
    /// If `keep_all_layers` is `true`, beam search keeps states in all layers to check for duplicates.
    pub keep_all_layers: bool,
    /// Common parameters.
    pub parameters: Parameters<T>,
}

impl<T: Default> Default for BeamSearchParameters<T> {
    fn default() -> Self {
        Self {
            beam_size: 1,
            keep_all_layers: false,
            parameters: Parameters::default(),
        }
    }
}

/// Performs beam search.
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
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::Search;
/// use dypdl_heuristic_search::search_algorithm::{
///     beam_search, BeamSearchParameters, FNode, SearchInput, SuccessorGenerator,
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
/// let node = FNode::generate_root_node(
///     state,
///     &mut function_cache,
///     cost,
///     &model,
///     &h_evaluator,
///     &f_evaluator,
///     primal_bound,
/// );
/// let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
/// let input = SearchInput {
///     node,
///     generator,
///     solution_suffix: &[],
/// };
/// let transition_evaluator = move |node: &FNode<_>, transition, cache: &mut _, primal_bound| {
///     node.generate_successor_node(
///         transition,
///         cache,
///         &model,
///         &h_evaluator,
///         &f_evaluator,
///         primal_bound,
///     )
/// };
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let parameters = BeamSearchParameters::default();
/// let solution = beam_search(
///     &input, transition_evaluator, base_cost_evaluator, parameters,
/// );
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions.len(), 1);
/// assert_eq!(Transition::from(solution.transitions[0].clone()), increment);
/// assert!(!solution.is_infeasible);
/// ```
pub fn beam_search<'a, T, N, E, B, V>(
    input: &'a SearchInput<'a, N, V>,
    mut transition_evaluator: E,
    mut base_cost_evaluator: B,
    parameters: BeamSearchParameters<T>,
) -> Solution<T, TransitionWithId<V>>
where
    T: variable_type::Numeric + Ord + Display,
    N: BfsNode<T, TransitionWithId<V>> + Clone,
    E: FnMut(
        &N,
        Rc<TransitionWithId<V>>,
        &mut ParentAndChildStateFunctionCache,
        Option<T>,
    ) -> Option<N>,
    B: FnMut(T, T) -> T,
    V: TransitionInterface + Clone + Default,
{
    let time_keeper = parameters
        .parameters
        .time_limit
        .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
    let quiet = parameters.parameters.quiet;
    let mut primal_bound = parameters.parameters.primal_bound;

    let model = input.generator.model.clone();
    let mut generator = input.generator.clone();
    let suffix = input.solution_suffix;
    let mut current_beam = Beam::new(parameters.beam_size);
    let mut next_beam = Beam::new(parameters.beam_size);
    let mut registry = StateRegistry::new(model.clone());
    let capacity = parameters
        .parameters
        .initial_registry_capacity
        .unwrap_or_else(|| current_beam.capacity());
    registry.reserve(capacity);

    let node = if let Some(node) = input.node.clone() {
        node
    } else {
        return Solution {
            is_infeasible: true,
            ..Default::default()
        };
    };

    current_beam.insert(&mut registry, node);

    if !parameters.keep_all_layers {
        registry.clear();
    }

    let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
    let mut applicable_transitions = Vec::new();

    let mut expanded = 0;
    let mut generated = 1;
    let mut pruned = false;
    let mut best_dual_bound = None;
    let mut removed_dual_bound = None;
    let mut layer_index = 0;

    while !current_beam.is_empty() {
        let mut incumbent = None;
        let mut layer_dual_bound = removed_dual_bound;

        let iter = if parameters.keep_all_layers {
            current_beam.close_and_drain()
        } else {
            current_beam.drain()
        };

        for node in iter {
            if let Some(dual_bound) = node.bound(&model) {
                if exceed_bound(&model, dual_bound, primal_bound) {
                    continue;
                }
            }

            function_cache.parent.clear();

            if let Some((cost, suffix)) = get_solution_cost_and_suffix(
                &model,
                &*node,
                suffix,
                &mut base_cost_evaluator,
                &mut function_cache,
            ) {
                if !exceed_bound(&model, cost, primal_bound) {
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

            expanded += 1;
            generator.generate_applicable_transitions(
                node.state(),
                &mut function_cache.parent,
                &mut applicable_transitions,
            );

            for transition in applicable_transitions.drain(..) {
                if let Some(successor) =
                    transition_evaluator(&node, transition, &mut function_cache, primal_bound)
                {
                    let successor_bound = successor.bound(&model);
                    let status = next_beam.insert(&mut registry, successor);

                    if !pruned && (status.is_pruned || status.removed.is_some()) {
                        pruned = true;
                    }

                    if let Some(bound) = successor_bound {
                        if !exceed_bound(&model, bound, layer_dual_bound) {
                            layer_dual_bound = Some(bound);
                        }

                        if status.is_pruned && !exceed_bound(&model, bound, removed_dual_bound) {
                            removed_dual_bound = Some(bound);
                        }
                    }

                    if let Some(bound) = status.removed.and_then(|removed| removed.bound(&model)) {
                        if !exceed_bound(&model, bound, removed_dual_bound) {
                            removed_dual_bound = Some(bound);
                        }
                    }

                    if status.is_newly_registered {
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

        if let Some(value) = layer_dual_bound {
            if exceed_bound(&model, value, primal_bound) {
                best_dual_bound = primal_bound;
            } else if best_dual_bound
                .map_or(true, |bound| !exceed_bound(&model, bound, Some(value)))
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

        mem::swap(&mut current_beam, &mut next_beam);

        if !parameters.keep_all_layers {
            registry.clear();
        }

        layer_index += 1;
    }

    Solution {
        is_infeasible: !pruned,
        best_bound: if pruned { best_dual_bound } else { None },
        expanded,
        generated,
        time: time_keeper.elapsed_time(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::Integer;

    #[test]
    fn parameters_default() {
        let parameters = BeamSearchParameters::<Integer>::default();
        assert_eq!(parameters.beam_size, 1);
        assert!(!parameters.keep_all_layers);
        assert_eq!(parameters.parameters, Parameters::default());
    }
}
