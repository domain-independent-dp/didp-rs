use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::data_structure::beam::Beam;
use super::search_algorithm::data_structure::state_registry::StateInRegistry;
use super::search_algorithm::data_structure::SuccessorGenerator;
use super::search_algorithm::util::Parameters;
use super::search_algorithm::Search;
use super::search_algorithm::{Lnbs, LnbsParameters};
use crate::search_algorithm::data_structure::BeamSearchNode;
use crate::search_algorithm::BeamSearchParameters;
use dypdl::{variable_type, ReduceFunction};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Create a Large Neighborhood Beam Search (LNBS) solver using the dual bound as a heuristic function.
///
/// It performs iterative beam search to find the initial solution.
/// `beam_size` specifies the initial beam width.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// Beam search searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If `keep_all_layers` is `true`, this solver keeps states in all layers to check for duplicates.
///
/// `parameters.time_limit` is required in this solver.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{FEvaluatorType, create_dual_bound_lnbs};
/// use dypdl_heuristic_search::search_algorithm::util::Parameters;
/// use dypdl_heuristic_search::LnbsParameters;
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
/// model.add_dual_bound(IntegerExpression::from(0)).unwrap();
///
/// let model = Rc::new(model);
/// let parameters = Parameters {
///     time_limit: Some(1800.0),
///     ..Default::default()
/// };
/// let lnbs_parameters = LnbsParameters {
///     initial_beam_size: 1,
///     has_negative_cost: false,
///     no_cost_weight: false,
///     no_bandit: false,
///     no_transition_constraints: false,
///     seed: 0,
/// };
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let mut solver = create_dual_bound_lnbs(
///     model, parameters, lnbs_parameters, f_evaluator_type, 1, false
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_lnbs<T>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    lnbs_parameters: LnbsParameters,
    f_evaluator_type: FEvaluatorType,
    beam_size: usize,
    keep_all_layers: bool,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let beam_constructor = Beam::<T, T, BeamSearchNode<T, T>>::new;
    let h_evaluator = |state: &StateInRegistry, model: &dypdl::Model| {
        Some(model.eval_dual_bound(state).unwrap_or_else(T::zero))
    };
    let (f_pruning, f_evaluator_type) = if model.has_dual_bounds() {
        (true, f_evaluator_type)
    } else {
        (false, FEvaluatorType::Plus)
    };
    let f_evaluator =
        move |g, h, _: &StateInRegistry, _: &dypdl::Model| f_evaluator_type.eval(g, h);
    let parameters = BeamSearchParameters {
        beam_size,
        maximize: model.reduce_function == ReduceFunction::Max,
        f_pruning,
        f_bound: None,
        keep_all_layers,
        parameters,
    };
    let generator = SuccessorGenerator::from_model_without_custom_cost(model, false);
    Box::new(Lnbs::new(
        generator,
        h_evaluator,
        f_evaluator,
        beam_constructor,
        lnbs_parameters,
        parameters,
    ))
}
