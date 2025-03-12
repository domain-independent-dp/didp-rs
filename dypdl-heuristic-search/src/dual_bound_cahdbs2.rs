use crate::search_algorithm::data_structure::ParentAndChildStateFunctionCache;

use super::f_evaluator_type::FEvaluatorType;
use super::parallel_search_algorithm::{hd_beam_search2, CostNodeMessage, FNodeMessage};
use super::search_algorithm::{
    Cabs, CabsParameters, CostNode, FNode, Search, SearchInput, SuccessorGenerator,
};
use dypdl::{variable_type, StateFunctionCache, Transition};
use std::fmt;
use std::str;
use std::sync::Arc;

/// Creates a Complete Anytime Hash Distributed Beam Search 2 (HDCABS2) solver using the dual bound as a heuristic function.
///
/// It iterates beam search with exponentially increasing beam width.
/// `beam_size` specifies the initial beam width.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{
///     CabsParameters, create_dual_bound_cahdbs2, FEvaluatorType,
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
/// let parameters = CabsParameters::default();
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let threads = 1;
/// let mut solver = create_dual_bound_cahdbs2(model, parameters, f_evaluator_type, threads);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_cahdbs2<T>(
    model: Arc<dypdl::Model>,
    parameters: CabsParameters<T>,
    f_evaluator_type: FEvaluatorType,
    threads: usize,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + Send + Sync + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator =
        SuccessorGenerator::<Transition, Arc<Transition>, _>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
    let cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };
    let print_statistics = !parameters.beam_search_parameters.parameters.quiet;

    if model.has_dual_bounds() {
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let h_model = model.clone();
        let h_evaluator = move |state: &_, cache: &mut _| h_model.eval_dual_bound(state, cache);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let node = FNodeMessage::generate_root_node(
            model.target.clone(),
            &mut function_cache,
            cost,
            &model,
            &h_evaluator,
            &f_evaluator,
            parameters.beam_search_parameters.parameters.primal_bound,
        );
        let input = SearchInput {
            node,
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator =
            move |node: &FNode<_, _, _, _, _>, transition, cache: &mut _, primal_bound| {
                node.generate_sendable_successor_node(
                    transition,
                    cache,
                    &model,
                    &h_evaluator,
                    &f_evaluator,
                    primal_bound,
                )
            };
        let beam_search = move |input: &SearchInput<_, _, _, _>, parameters| {
            let (solution, statistics) = hd_beam_search2(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
                threads,
            )
            .unwrap();

            if print_statistics {
                println!(
                    "Searched with beam size: {}, threads: {}, kept: {}, sent: {}",
                    parameters.beam_size,
                    threads,
                    statistics.kept.iter().sum::<usize>(),
                    statistics.sent.iter().sum::<usize>(),
                );
            }

            solution
        };
        Box::new(Cabs::<_, FNodeMessage<_>, _, _, _, Arc<_>>::new(
            input,
            beam_search,
            parameters,
        ))
    } else {
        let node = CostNodeMessage::generate_root_node(model.target.clone(), cost, &model);
        let input = SearchInput {
            node: Some(node),
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator = move |node: &CostNode<_, _, _, _, _>,
                                         transition,
                                         cache: &mut ParentAndChildStateFunctionCache,
                                         _| {
            node.generate_sendable_successor_node(transition, &mut cache.parent, &model)
        };
        let beam_search = move |input: &SearchInput<_, _, _, _>, parameters| {
            let (solution, statistics) = hd_beam_search2(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
                threads,
            )
            .unwrap();

            if print_statistics {
                println!(
                    "Searched with beam size: {}, threads: {}, kept: {}, sent: {}",
                    parameters.beam_size,
                    threads,
                    statistics.kept.iter().sum::<usize>(),
                    statistics.sent.iter().sum::<usize>(),
                );
            }

            solution
        };
        Box::new(Cabs::<_, CostNodeMessage<_>, _, _, _, Arc<_>>::new(
            input,
            beam_search,
            parameters,
        ))
    }
}
