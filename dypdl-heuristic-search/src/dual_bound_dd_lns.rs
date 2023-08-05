use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::{
    beam_search, Cabs, CabsParameters, CostNode, DdLns, DdLnsParameters, FNode, Search,
    SearchInput, StateInRegistry, SuccessorGenerator, TransitionWithId,
};
use dypdl::variable_type;
use std::fmt;
use std::rc::Rc;
use std::str;

/// Create a Large Neighborhood Search with Decision Diagrams (DD-LNS) solver using the dual bound as a heuristic function.
///
/// It performs LNS by constructing restricted multi-valued decision diagrams (MDD).
/// It first performs CABS to find an initial feasible solution and then performs DD-LNS to improve the solution.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
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
/// use dypdl_heuristic_search::{
///     CabsParameters, create_dual_bound_dd_lns, DdLnsParameters, FEvaluatorType,
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
/// model.add_dual_bound(IntegerExpression::from(0)).unwrap();
///
/// let model = Rc::new(model);
/// let parameters = DdLnsParameters::default();
/// let cabs_parameters = CabsParameters::default();
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let mut solver = create_dual_bound_dd_lns(
///     model, parameters, cabs_parameters, f_evaluator_type,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_dd_lns<T>(
    model: Rc<dypdl::Model>,
    mut parameters: DdLnsParameters<T>,
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
        let h_model = model.clone();
        let h_evaluator = move |state: &_| h_model.eval_dual_bound(state);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let t_model = model.clone();
        let transition_evaluator = move |node: &FNode<_, _>, transition, primal_bound| {
            node.generate_successor_node(
                transition,
                &t_model,
                &h_evaluator,
                &f_evaluator,
                primal_bound,
            )
        };

        let input = SearchInput {
            node: root_generator(StateInRegistry::from(model.target.clone()), cost),
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
        let mut cabs = Cabs::new(input, beam_search, cabs_parameters);
        let (solution, _) = cabs.search_inner();

        let h_evaluator = move |state: &_| model.eval_dual_bound(state);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let transition_evaluator =
            move |node: &FNode<_, _>, transition, registry: &mut _, primal_bound| {
                node.insert_successor_node(
                    transition,
                    registry,
                    &h_evaluator,
                    &f_evaluator,
                    primal_bound,
                )
            };
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

        Box::new(DdLns::new(
            generator,
            transition_evaluator,
            base_cost_evaluator,
            root_generator,
            solution,
            parameters,
        ))
    } else {
        let g_model = model.clone();
        let root_generator =
            move |state, cost| Some(CostNode::generate_root_node(state, cost, &g_model));
        let t_model = model.clone();
        let transition_evaluator = move |node: &CostNode<_, _>, transition, _| {
            node.generate_successor_node(transition, &t_model)
        };
        let beam_search = move |input: &SearchInput<_, _>, parameters| {
            beam_search(
                input,
                &transition_evaluator,
                base_cost_evaluator,
                parameters,
            )
        };
        let input = SearchInput {
            node: root_generator(StateInRegistry::from(model.target.clone()), cost),
            generator: generator.clone(),
            solution_suffix: &[],
        };
        let mut cabs = Cabs::new(input, beam_search, cabs_parameters);
        let (solution, _) = cabs.search_inner();

        let transition_evaluator = move |node: &CostNode<_, _>, transition, registry: &mut _, _| {
            node.insert_successor_node(transition, registry)
        };
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

        Box::new(DdLns::new(
            generator,
            transition_evaluator,
            base_cost_evaluator,
            root_generator,
            solution,
            parameters,
        ))
    }
}
