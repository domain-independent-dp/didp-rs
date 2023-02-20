use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::data_structure::state_registry::StateInRegistry;
use super::search_algorithm::data_structure::{FNode, SuccessorGenerator};
use super::search_algorithm::util::{ForwardSearchParameters, Parameters};
use super::search_algorithm::{BestFirstSearch, Search};
use dypdl::variable_type;
use dypdl::Continuous;
use std::fmt;
use std::rc::Rc;
use std::str;

/// Create a weighted A* solver using the dual bound as a heuristic function..
pub fn create_dual_bound_weighted_astar<T>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    weight: Continuous,
    f_evaluator_type: FEvaluatorType,
    initial_registry_capacity: Option<usize>,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::from_model(model.clone(), false);
    let parameters = ForwardSearchParameters {
        generator,
        parameters,
        initial_registry_capacity,
    };
    let h_evaluator = |state: &StateInRegistry, model: &dypdl::Model| {
        Some(model.eval_dual_bound(state).unwrap_or_else(T::zero))
    };
    let (f_pruning, f_evaluator_type) = if model.has_dual_bounds() {
        (true, f_evaluator_type)
    } else {
        (false, FEvaluatorType::Plus)
    };
    let f_evaluator = move |g, h: T, _: &StateInRegistry, _: &dypdl::Model| {
        f_evaluator_type.eval(g, T::from_continuous(weight * h.to_continuous()))
    };
    Box::new(BestFirstSearch::<_, FNode<_>, _, _>::new(
        model,
        h_evaluator,
        f_evaluator,
        f_pruning,
        true,
        weight <= 1.0,
        parameters,
    ))
}
