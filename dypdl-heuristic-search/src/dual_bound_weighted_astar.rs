use crate::search_algorithm::data_structure::ParentAndChildStateFunctionCache;

use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::{
    BestFirstSearch, CostNode, FNodeEvaluators, Parameters, Search, SearchInput,
    SuccessorGenerator, WeightedFNode,
};
use dypdl::variable_type::{Numeric, OrderedContinuous};
use dypdl::{Continuous, StateFunctionCache, Transition};
use std::fmt;
use std::rc::Rc;
use std::str;

/// Creates a weighted A* solver using the dual bound as a heuristic function.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// It uses the dual bound defined in the DyPDL model as a heuristic function.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{create_dual_bound_weighted_astar, FEvaluatorType, Parameters};
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
/// let parameters = Parameters::default();
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let mut solver = create_dual_bound_weighted_astar(
///     model, parameters, f_evaluator_type, 1.1,
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_weighted_astar<T>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    f_evaluator_type: FEvaluatorType,
    weight: Continuous,
) -> Box<dyn Search<T>>
where
    T: Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
    let base_cost_evaluator = move |cost, base_cost| f_evaluator_type.eval(cost, base_cost);
    let cost = match f_evaluator_type {
        FEvaluatorType::Plus => T::zero(),
        FEvaluatorType::Product => T::one(),
        FEvaluatorType::Max => T::min_value(),
        FEvaluatorType::Min => T::max_value(),
        FEvaluatorType::Overwrite => T::zero(),
    };

    if model.has_dual_bounds() {
        let state = model.target.clone();
        let mut function_cache = StateFunctionCache::new(&model.state_functions);
        let h_evaluator = move |state: &_, cache: &mut _| model.eval_dual_bound(state, cache);
        let bound_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let f_evaluator = move |g: T, h: T, _: &_| {
            let g = OrderedContinuous::from(g.to_continuous());
            let h = OrderedContinuous::from(h.to_continuous() * weight);
            f_evaluator_type.eval(g, h)
        };
        let evaluators = FNodeEvaluators {
            h: &h_evaluator,
            f: &f_evaluator,
        };
        let node = WeightedFNode::generate_root_node(
            state,
            &mut function_cache,
            cost,
            &generator.model,
            evaluators,
            &bound_evaluator,
            parameters.primal_bound,
        );
        let input = SearchInput {
            node,
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator = move |node: &WeightedFNode<_, _>,
                                         transition,
                                         cache: &mut _,
                                         registry: &mut _,
                                         primal_bound| {
            let evaluators = FNodeEvaluators {
                h: &h_evaluator,
                f: &f_evaluator,
            };
            node.insert_successor_node(
                transition,
                cache,
                registry,
                evaluators,
                &bound_evaluator,
                primal_bound,
            )
        };

        Box::new(BestFirstSearch::<_, WeightedFNode<_, _>, _, _>::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            parameters,
        ))
    } else {
        let node = CostNode::generate_root_node(model.target.clone(), cost, &model);
        let input = SearchInput {
            node: Some(node),
            generator,
            solution_suffix: &[],
        };
        let transition_evaluator =
            |node: &CostNode<_>,
             transition,
             cache: &mut ParentAndChildStateFunctionCache,
             registry: &mut _,
             _| { node.insert_successor_node(transition, &mut cache.parent, registry) };
        Box::new(BestFirstSearch::<_, CostNode<_>, _, _>::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            parameters,
        ))
    }
}
