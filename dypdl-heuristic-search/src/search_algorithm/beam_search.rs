use super::data_structure::{exceed_bound, Beam, BfsNode, StateRegistry};
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
/// let cost = 0;
/// let h_evaluator = |_: &_| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
/// let primal_bound = None;
/// let node = FNode::generate_root_node(
///     state,
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
/// let transition_evaluator = move |node: &FNode<_>, transition, primal_bound| {
///     node.generate_successor_node(
///         transition,
///         &model,
///         &h_evaluator,
///         &f_evaluator,
///         primal_bound,
///     )
/// };
/// let parameters = BeamSearchParameters::default();
/// let solution = beam_search(
///     &input, transition_evaluator, parameters,
/// );
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions.len(), 1);
/// assert_eq!(Transition::from(solution.transitions[0].clone()), increment);
/// assert!(!solution.is_infeasible);
/// ```
pub fn beam_search<'a, T, N, E, V>(
    input: &'a SearchInput<'a, N, V>,
    transition_evaluator: E,
    parameters: BeamSearchParameters<T>,
) -> Solution<T, V>
where
    T: variable_type::Numeric + Display,
    N: BfsNode<T, V> + Clone,
    E: Fn(&N, Rc<V>, Option<T>) -> Option<N>,
    V: TransitionInterface + Clone + Default,
{
    let time_keeper = parameters
        .parameters
        .time_limit
        .map_or_else(TimeKeeper::default, TimeKeeper::with_time_limit);
    let quiet = parameters.parameters.quiet;
    let mut primal_bound = parameters.parameters.primal_bound;

    let model = &input.generator.model;
    let generator = &input.generator;
    let suffix = input.solution_suffix;
    let mut current_beam = Beam::<_, _>::new(parameters.beam_size);
    let mut next_beam = Beam::<_, _, _>::new(parameters.beam_size);
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

    let mut expanded = 0;
    let mut generated = 1;
    let mut pruned = false;
    let mut best_dual_bound = None;
    let mut layer_index = 0;

    while !current_beam.is_empty() {
        let mut incumbent = None;
        let mut layer_dual_bound = None;

        for node in current_beam.drain() {
            if let Some((cost, suffix)) = get_solution_cost_and_suffix(model, &*node, suffix) {
                if !exceed_bound(model, cost, primal_bound) {
                    primal_bound = Some(cost);
                    incumbent = Some((node, cost, suffix));
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

            for transition in generator.applicable_transitions(node.state()) {
                if let Some(successor) = transition_evaluator(&node, transition, primal_bound) {
                    if let Some(bound) = successor.bound(model) {
                        if !exceed_bound(model, bound, layer_dual_bound) {
                            layer_dual_bound = Some(bound);
                        }
                    }

                    let (new_generated, beam_pruning) = next_beam.insert(&mut registry, successor);

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
            println!(
                "Searched layer: {}, expanded: {}, elapsed time: {}",
                layer_index,
                expanded,
                time_keeper.elapsed_time()
            );
        }

        if let (false, Some(value)) = (pruned, layer_dual_bound) {
            if best_dual_bound.map_or(true, |bound| !exceed_bound(model, bound, Some(value))) {
                best_dual_bound = layer_dual_bound;
            }
        }

        if let Some((node, cost, suffix)) = &incumbent {
            let mut transitions = node.transitions();
            transitions.extend_from_slice(suffix);
            let is_optimal = !pruned && next_beam.is_empty();

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
