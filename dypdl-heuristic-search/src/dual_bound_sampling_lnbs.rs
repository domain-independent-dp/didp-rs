use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::data_structure::beam::Beam;
use super::search_algorithm::data_structure::state_registry::StateInRegistry;
use super::search_algorithm::data_structure::SuccessorGenerator;
use super::search_algorithm::util::Parameters;
use super::search_algorithm::Search;
use super::search_algorithm::{LnsParameters, SamplingLnbs};
use crate::search_algorithm::data_structure::BeamSearchNode;
use crate::search_algorithm::BeamSearchParameters;
use dypdl::{variable_type, ReduceFunction};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Create a sampling LNBS solver using the dual bound as a heuristic function.
pub fn create_dual_bound_sampling_lnbs<T>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    lns_parameters: LnsParameters,
    f_evaluator_type: FEvaluatorType,
    beam_size: usize,
    keep_all_layers: bool,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let beam_constructor = Beam::<T, T, BeamSearchNode<T, T>>::new;
    let generator = SuccessorGenerator::from_model_without_custom_cost(model.clone(), false);
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
    Box::new(SamplingLnbs::new(
        model,
        generator,
        h_evaluator,
        f_evaluator,
        beam_constructor,
        lns_parameters,
        parameters,
    ))
}
