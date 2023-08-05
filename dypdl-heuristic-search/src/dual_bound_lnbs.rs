use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::{
    beam_search, Cabs, CabsParameters, CostNode, FNode, Lnbs, LnbsParameters, Search, SearchInput,
    StateInRegistry, SuccessorGenerator, TransitionWithId,
};
use dypdl::variable_type;
use std::fmt;
use std::rc::Rc;
use std::str;

/// Creates a Large Neighborhood Beam Search (LNBS) solver using the dual bound as a heuristic function.
///
/// It performs Large Neighborhood Beam Search (LNBS), which improves a solution by finding a partial path using beam search.
/// It first performs CABS to find an initial feasible solution and then performs LNBS to improve the solution.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// # Panics
///
/// If CABS panics.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Large Neighborhood Beam Search for Domain-Independent Dynamic Programming,"
/// Proceedings of the 29th International Conference on Principles and Practice of Constraint Programming (CP), 2023.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{CabsParameters, create_dual_bound_cabs, FEvaluatorType};
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
/// let parameters = CabsParameters::default();
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let mut solver = create_dual_bound_cabs(model, parameters, f_evaluator_type);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_lnbs<T>(
    model: Rc<dypdl::Model>,
    mut parameters: LnbsParameters<T>,
    cabs_parameters: CabsParameters<T>,
    f_evaluator_type: FEvaluatorType,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
    let cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };

    if model.has_dual_bounds() {
        let h_model = model.clone();
        let h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let g_model = model.clone();
        let root_generator = move |state, cost| {
            FNode::generate_root_node(
                state,
                cost,
                &g_model,
                &h_evaluator,
                &f_evaluator,
                cabs_parameters
                    .beam_search_parameters
                    .parameters
                    .primal_bound,
            )
        };
        let node = root_generator(StateInRegistry::from(model.target.clone()), cost);
        let h_model = model.clone();
        let h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let transition_evaluator = move |node: &FNode<_, _>, transition, primal_bound| {
            node.generate_successor_node(
                transition,
                &model,
                &h_evaluator,
                &f_evaluator,
                primal_bound,
            )
        };

        let input = SearchInput {
            node,
            generator: generator.clone(),
            solution_suffix: &[],
        };
        let beam_search = move |input: &SearchInput<_, _>, parameters| {
            beam_search(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
            )
        };
        let mut cabs = Cabs::new(input, beam_search.clone(), cabs_parameters);
        let (solution, _) = cabs.search_inner();

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

        Box::new(Lnbs::new(
            generator,
            beam_search,
            root_generator,
            solution,
            parameters,
        ))
    } else {
        let g_model = model.clone();
        let root_generator =
            move |state, cost| Some(CostNode::generate_root_node(state, cost, &g_model));
        let input = SearchInput {
            node: root_generator(StateInRegistry::from(model.target.clone()), cost),
            generator: generator.clone(),
            solution_suffix: &[],
        };
        let transition_evaluator = move |node: &CostNode<_, _>, transition, _| {
            node.generate_successor_node(transition, &model)
        };
        let beam_search = move |input: &SearchInput<_, _>, parameters| {
            beam_search(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
            )
        };
        let mut cabs = Cabs::new(input, beam_search.clone(), cabs_parameters);
        let (solution, _) = cabs.search_inner();

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

        Box::new(Lnbs::new(
            generator,
            beam_search,
            root_generator,
            solution,
            parameters,
        ))
    }
}
