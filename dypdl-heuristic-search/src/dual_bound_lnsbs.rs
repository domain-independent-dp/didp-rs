use crate::search_algorithm::data_structure::HashableSignatureVariables;

use super::f_evaluator_type::FEvaluatorType;
use super::parallel_search_algorithm::{shared_beam_search, SendableCostNode, SendableFNode};
use super::search_algorithm::{
    data_structure::ParentAndChildStateFunctionCache, rollout, Cabs, CabsParameters, Lnbs,
    LnbsParameters, NeighborhoodSearchInput, Search, SearchInput, StateInRegistry,
    SuccessorGenerator, TransitionMutex, TransitionWithId,
};
use super::Solution;
use dypdl::variable_type;
use dypdl::{StateFunctionCache, Transition};
use std::fmt;
use std::str;
use std::sync::Arc;

/// Creates a Large Neighborhood Shared Beam Search (LNSBS) solver using the dual bound as a heuristic function.
///
/// It performs Large Neighborhood Beam Search (LNBS), which improves a solution by finding a partial path using beam search.
/// It uses a parallel beam search algorithm, shared beam search (SBS), instead of sequential beam search.
/// It starts from a initial solution (`transitions`) and improves it.
/// If `transitions` is `None`, it first performs CABS to find an initial feasible solution and then performs DD-LNS to improve the solution.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// # Panics
///
/// If CABS to find an initial feasible solution panics.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// Ryo Kuroiwa and J. Christopher Beck. "Large Neighborhood Beam Search for Domain-Independent Dynamic Programming,"
/// Proceedings of the 29th International Conference on Principles and Practice of Constraint Programming (CP), pp. 23:1-23:22, 2023.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{
///     BeamSearchParameters, CabsParameters, create_dual_bound_lnsbs, FEvaluatorType,
///     LnbsParameters, Parameters,
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
/// model.add_dual_bound(IntegerExpression::from(0)).unwrap();
///
/// let model = Arc::new(model);
/// let parameters = LnbsParameters {
///     beam_search_parameters: BeamSearchParameters {
///         parameters: Parameters {
///             time_limit: Some(1800.0),
///             ..Default::default()
///         },
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// let cabs_parameters = CabsParameters::default();
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let threads = 1;
/// let mut solver = create_dual_bound_lnsbs(
///     model, None, parameters, cabs_parameters, f_evaluator_type, threads,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_lnsbs<T>(
    model: Arc<dypdl::Model>,
    transitions: Option<Vec<Transition>>,
    mut parameters: LnbsParameters<T>,
    cabs_parameters: CabsParameters<T>,
    f_evaluator_type: FEvaluatorType,
    threads: usize,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + Send + Sync + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::<Transition, Arc<TransitionWithId>, _>::from_model(
        model.clone(),
        false,
    );
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
    let root_cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };

    let generator_transitions = generator
        .transitions
        .iter()
        .chain(generator.forced_transitions.iter())
        .map(|t| t.as_ref().clone());

    let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
    let (solution, transition_mutex) = if let Some(transitions) = transitions {
        let solution_cost = if let Some(result) = rollout(
            &generator.model.target,
            &mut function_cache,
            root_cost,
            &transitions,
            base_cost_evaluator,
            &model,
        ) {
            if result.is_base {
                Some(result.cost)
            } else {
                if !parameters.beam_search_parameters.parameters.quiet {
                    println!("Initial solution does not result in a base state");
                }
                None
            }
        } else {
            if !parameters.beam_search_parameters.parameters.quiet {
                println!("Initial solution is infeasible");
            }
            None
        };
        let id_max = generator_transitions
            .clone()
            .map(|t| t.id)
            .max()
            .unwrap_or(0);
        let transitions = transitions
            .into_iter()
            .enumerate()
            .map(|(i, transition)| TransitionWithId {
                id: id_max + 1 + i,
                forced: false,
                transition,
            })
            .collect::<Vec<_>>();
        let solution = Solution {
            cost: solution_cost,
            transitions: transitions.clone(),
            ..Default::default()
        };

        let transitions = generator_transitions.chain(transitions).collect();

        (Some(solution), TransitionMutex::new(transitions))
    } else {
        (None, TransitionMutex::new(generator_transitions.collect()))
    };

    if model.has_dual_bounds() {
        let h_model = model.clone();
        let h_evaluator = move |state: &_, cache: &mut _| h_model.eval_dual_bound(state, cache);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let g_model = model.clone();
        let node_generator = move |state, cost| {
            let mut function_cache = StateFunctionCache::new(&g_model.state_functions);

            SendableFNode::<_, TransitionWithId>::generate_root_node(
                state,
                &mut function_cache,
                cost,
                &g_model,
                &h_evaluator,
                &f_evaluator,
                parameters.beam_search_parameters.parameters.primal_bound,
            )
        };
        let h_model = model.clone();
        let h_evaluator = move |state: &_, cache: &mut _| h_model.eval_dual_bound(state, cache);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let t_model = model.clone();
        let transition_evaluator =
            move |node: &SendableFNode<_, _>, transition, cache: &mut _, primal_bound| {
                node.generate_successor_node(
                    transition,
                    cache,
                    &t_model,
                    &h_evaluator,
                    &f_evaluator,
                    primal_bound,
                )
            };
        let beam_search = move |input: &SearchInput<_, _, _, _>, parameters| {
            shared_beam_search(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
                threads,
            )
            .unwrap()
        };

        let solution = solution.unwrap_or_else(|| {
            let input = SearchInput {
                node: node_generator(StateInRegistry::from(model.target.clone()), root_cost),
                generator: generator.clone(),
                solution_suffix: &[],
            };
            let mut cabs =
                Cabs::<_, SendableFNode<_, _>, _, _, _, _, Arc<HashableSignatureVariables>>::new(
                    input,
                    beam_search.clone(),
                    cabs_parameters,
                );
            let (solution, _) = cabs.search_inner();
            solution
        });

        parameters.beam_search_parameters.parameters.time_limit = parameters
            .beam_search_parameters
            .parameters
            .time_limit
            .map(|time_limit| {
                let remaining = time_limit - solution.time;
                if remaining < 0.0 {
                    0.0
                } else {
                    remaining
                }
            });

        let input = NeighborhoodSearchInput {
            root_cost,
            node_generator,
            successor_generator: generator,
            solution,
            phantom: Default::default(),
        };

        Box::new(Lnbs::<_, _, _, _, _, _, _, _>::new(
            input,
            beam_search,
            transition_mutex,
            parameters,
        ))
    } else {
        let g_model = model.clone();
        let node_generator =
            move |state, cost| Some(SendableCostNode::generate_root_node(state, cost, &g_model));
        let t_model = model.clone();
        let transition_evaluator = move |node: &SendableCostNode<_, _>,
                                         transition,
                                         cache: &mut ParentAndChildStateFunctionCache,
                                         _| {
            node.generate_successor_node(transition, &mut cache.parent, &t_model)
        };
        let beam_search = move |input: &SearchInput<_, _, _, _>, parameters| {
            shared_beam_search(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
                threads,
            )
            .unwrap()
        };

        let solution = solution.unwrap_or_else(|| {
            let input = SearchInput {
                node: node_generator(StateInRegistry::from(model.target.clone()), root_cost),
                generator: generator.clone(),
                solution_suffix: &[],
            };
            let mut cabs = Cabs::<_, _, _, _, _, _, Arc<HashableSignatureVariables>>::new(
                input,
                beam_search.clone(),
                cabs_parameters,
            );
            let (solution, _) = cabs.search_inner();
            solution
        });

        parameters.beam_search_parameters.parameters.time_limit = parameters
            .beam_search_parameters
            .parameters
            .time_limit
            .map(|time_limit| {
                let remaining = time_limit - solution.time;
                if remaining < 0.0 {
                    0.0
                } else {
                    remaining
                }
            });

        let input = NeighborhoodSearchInput {
            root_cost,
            node_generator,
            successor_generator: generator,
            solution,
            phantom: Default::default(),
        };

        Box::new(Lnbs::<_, _, _, _, _, _, _, _>::new(
            input,
            beam_search,
            transition_mutex,
            parameters,
        ))
    }
}
