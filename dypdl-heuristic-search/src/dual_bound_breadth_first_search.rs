use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::{
    BreadthFirstSearch, BrfsParameters, CostNode, FNode, Search, SearchInput, SuccessorGenerator,
};
use dypdl::{variable_type, Transition};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Creates a breadth-first search solver using the dual bound as a heuristic function.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// Breadth-first searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If `keep_all_layers` is `true`, this solver keeps states in all layers to check for duplicates.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{
///     create_dual_bound_breadth_first_search, FEvaluatorType, BrfsParameters
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
/// let parameters = BrfsParameters::default();
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let mut solver = create_dual_bound_breadth_first_search(
///     model, parameters, f_evaluator_type,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_breadth_first_search<T>(
    model: Rc<dypdl::Model>,
    parameters: BrfsParameters<T>,
    f_evaluator_type: FEvaluatorType,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);

    if model.has_dual_bounds() {
        let state = model.target.clone();
        let h_evaluator = move |state: &_| model.eval_dual_bound(state);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let node = FNode::generate_root_node(
            state,
            T::zero(),
            &generator.model,
            &h_evaluator,
            &f_evaluator,
            parameters.parameters.primal_bound,
        );
        let input = SearchInput {
            node,
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator =
            move |node: &FNode<_>, transition, registry: &mut _, primal_bound| {
                node.insert_successor_node(
                    transition,
                    registry,
                    &h_evaluator,
                    &f_evaluator,
                    primal_bound,
                )
            };

        Box::new(BreadthFirstSearch::<_, FNode<_>, _, _>::new(
            input,
            transition_evaluator,
            parameters,
        ))
    } else {
        let node = CostNode::generate_root_node(model.target.clone(), T::zero(), &model);
        let input = SearchInput {
            node: Some(node),
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator = |node: &CostNode<_>, transition, registry: &mut _, _| {
            node.insert_successor_node(transition, registry)
        };
        Box::new(BreadthFirstSearch::<_, CostNode<_>, _, _>::new(
            input,
            transition_evaluator,
            parameters,
        ))
    }
}
