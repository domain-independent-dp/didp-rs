use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::{
    BestFirstSearch, CostNode, FNode, FNodeEvaluators, Parameters, Search, SearchInput,
    StateRegistry, SuccessorGenerator,
};
use crate::caasdy;
use crate::search_algorithm::data_structure::{
    CreateTransitionChain, RcChain, ResourceLexicographicNode, StateInRegistry, StateInformation,
    TransitionWithId,
};
use dypdl::{variable_type, ParentAndChildStateFunctionCache, StateFunctionCache, Transition};
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;
use std::str;

/// Creates a labeling solver, a resource-constrained shortest path search.
///
/// Nodes are ordered by comparing resource variables lexicographically before falling back
/// to the ordinary node ordering (dual bound based if available, cost based otherwise).
///
/// If the model has no resource variables, this falls back to CAASDy since there is nothing
/// to order lexicographically.
///
/// # References
///
/// Ryo Kuroiwa and Edward Lam. "Column Generation with Domain-Independent Dynamic Programming,"
/// 32nd International Conference on Principles and Practice of Constraint Programming (CP 2026), pp. 37:1-37:24, 2026.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{create_labeling, FEvaluatorType, Parameters};
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_resource_variable("variable", false, 0).unwrap();
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
/// let mut solver = create_labeling(model, parameters, f_evaluator_type);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_labeling<T>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    f_evaluator_type: FEvaluatorType,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let metadata = &model.state_metadata;

    if metadata.set_resource_variable_names.is_empty()
        && metadata.element_resource_variable_names.is_empty()
        && metadata.integer_resource_variable_names.is_empty()
        && metadata.continuous_resource_variable_names.is_empty()
    {
        return caasdy::create_caasdy(model, parameters, f_evaluator_type);
    }

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
        let mut cache = StateFunctionCache::new(&model.state_functions);
        let wrap_model = model.clone();
        let h_evaluator = move |state: &_, cache: &mut _| model.eval_dual_bound(state, cache);
        let f_evaluator = move |g, h, _: &_| f_evaluator_type.eval(g, h);
        let node = FNode::generate_root_node(
            state,
            &mut cache,
            cost,
            &generator.model,
            &h_evaluator,
            &f_evaluator,
            parameters.primal_bound,
        )
        .map(|node| ResourceLexicographicNode::new(node, wrap_model.clone()));
        let input = SearchInput {
            node,
            generator,
            solution_suffix: &[],
        };

        let transition_evaluator =
            move |node: &ResourceLexicographicNode<T, FNode<T, TransitionWithId>>,
                  transition: Rc<TransitionWithId>,
                  function_cache: &mut ParentAndChildStateFunctionCache,
                  registry: &mut StateRegistry<
                T,
                ResourceLexicographicNode<T, FNode<T, TransitionWithId>>,
            >,
                  primal_bound: Option<T>| {
                let inner = &node.node;
                function_cache.child.clear();
                let model = registry.model().clone();
                let (state, g) = model.generate_successor_state(
                    inner.state(),
                    function_cache,
                    inner.cost(&model),
                    transition.deref(),
                    None,
                )?;

                let evaluator_model = model.clone();
                let constructor = |state,
                                   g,
                                   other: Option<
                    &ResourceLexicographicNode<T, FNode<T, TransitionWithId>>,
                >| {
                    let evaluators = FNodeEvaluators {
                        h: &h_evaluator,
                        f: &f_evaluator,
                    };
                    let (h, f) = FNode::evaluate_state(
                        &state,
                        &mut function_cache.child,
                        g,
                        &evaluator_model,
                        evaluators,
                        primal_bound,
                        other.map(|node| &node.node),
                    )?;
                    let transition_chain =
                        Rc::new(RcChain::new(inner.transition_chain(), transition.clone()));
                    let new_node =
                        CostNode::new(state, g, &evaluator_model, Some(transition_chain));

                    Some(ResourceLexicographicNode::new(
                        FNode::with_node_and_h_and_f(new_node, h, f),
                        model.clone(),
                    ))
                };

                let result = registry.insert_with(state, g, constructor);

                for d in result.dominated.iter() {
                    if !d.is_closed() {
                        d.close();
                    }
                }

                result
                    .information
                    .map(|node| (node, result.dominated.is_empty()))
            };

        Box::new(BestFirstSearch::<
            _,
            ResourceLexicographicNode<_, FNode<_, TransitionWithId>>,
            _,
            _,
        >::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            parameters,
        ))
    } else {
        let node = CostNode::generate_root_node(model.target.clone(), cost, &model);
        let node = ResourceLexicographicNode::new(node, model.clone());
        let input = SearchInput {
            node: Some(node),
            generator,
            solution_suffix: &[],
        };

        let transition_evaluator =
            |node: &ResourceLexicographicNode<T, CostNode<T, TransitionWithId>>,
             transition: Rc<TransitionWithId>,
             function_cache: &mut ParentAndChildStateFunctionCache,
             registry: &mut StateRegistry<
                T,
                ResourceLexicographicNode<T, CostNode<T, TransitionWithId>>,
            >,
             _| {
                let inner = &node.node;
                let model = registry.model().clone();
                function_cache.child.clear();
                let (state, cost) = model.generate_successor_state(
                    inner.state(),
                    function_cache,
                    inner.cost(&model),
                    transition.deref(),
                    None,
                )?;
                let wrap_model = model.clone();
                let constructor = |state: StateInRegistry,
                                   _: T,
                                   _: Option<
                    &ResourceLexicographicNode<T, CostNode<T, TransitionWithId>>,
                >| {
                    let transition_chain =
                        Rc::new(RcChain::new(inner.transition_chain(), transition.clone()));

                    Some(ResourceLexicographicNode::new(
                        CostNode::new(state, cost, &model, Some(transition_chain)),
                        wrap_model.clone(),
                    ))
                };

                let result = registry.insert_with(state, cost, constructor);

                for d in result.dominated.iter() {
                    if !d.is_closed() {
                        d.close();
                    }
                }

                result
                    .information
                    .map(|node| (node, result.dominated.is_empty()))
            };

        Box::new(BestFirstSearch::<
            _,
            ResourceLexicographicNode<_, CostNode<_, TransitionWithId>>,
            _,
            _,
        >::new(
            input,
            transition_evaluator,
            base_cost_evaluator,
            parameters,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;

    #[test]
    fn falls_back_to_caasdy_without_resource_variables() {
        let mut model = Model::default();
        let variable = model.add_integer_variable("variable", 0).unwrap();
        model
            .add_base_case(vec![Condition::comparison_i(
                ComparisonOperator::Ge,
                variable,
                1,
            )])
            .unwrap();
        let mut increment = Transition::new("increment");
        increment.set_cost(IntegerExpression::Cost + 1);
        increment.add_effect(variable, variable + 1).unwrap();
        model.add_forward_transition(increment.clone()).unwrap();
        model.add_dual_bound(IntegerExpression::from(0)).unwrap();

        let model = Rc::new(model);
        let parameters = Parameters::default();

        let mut solver = create_labeling(model, parameters, FEvaluatorType::Plus);
        let solution = solver.search().unwrap();
        assert_eq!(solution.cost, Some(1));
        assert_eq!(solution.transitions, vec![increment]);
        assert!(!solution.is_infeasible);
    }

    #[test]
    fn labeling_with_two_resource_variables() {
        let mut model = Model::default();
        let x = model.add_integer_resource_variable("x", false, 0).unwrap();
        let y = model.add_integer_resource_variable("y", false, 0).unwrap();
        model
            .add_base_case(vec![Condition::comparison_i(ComparisonOperator::Ge, x, 2)])
            .unwrap();

        let mut increment_x = Transition::new("increment_x");
        increment_x.set_cost(IntegerExpression::Cost + 1);
        increment_x.add_effect(x, x + 1).unwrap();
        model.add_forward_transition(increment_x.clone()).unwrap();

        let mut increment_y = Transition::new("increment_y");
        increment_y.set_cost(IntegerExpression::Cost + 1);
        increment_y.add_effect(y, y + 1).unwrap();
        model.add_forward_transition(increment_y).unwrap();

        model.add_dual_bound(IntegerExpression::from(0)).unwrap();

        let model = Rc::new(model);
        let parameters = Parameters::default();

        let mut solver = create_labeling(model, parameters, FEvaluatorType::Plus);
        let solution = solver.search().unwrap();
        assert_eq!(solution.cost, Some(2));
        assert_eq!(solution.transitions, vec![increment_x.clone(), increment_x]);
        assert!(!solution.is_infeasible);
    }

    #[test]
    fn labeling_without_dual_bound() {
        let mut model = Model::default();
        let x = model.add_integer_resource_variable("x", false, 0).unwrap();
        model
            .add_base_case(vec![Condition::comparison_i(ComparisonOperator::Ge, x, 1)])
            .unwrap();

        let mut increment = Transition::new("increment");
        increment.set_cost(IntegerExpression::Cost + 1);
        increment.add_effect(x, x + 1).unwrap();
        model.add_forward_transition(increment.clone()).unwrap();

        let model = Rc::new(model);
        let parameters = Parameters::default();

        let mut solver = create_labeling(model, parameters, FEvaluatorType::Plus);
        let solution = solver.search().unwrap();
        assert_eq!(solution.cost, Some(1));
        assert_eq!(solution.transitions, vec![increment]);
        assert!(!solution.is_infeasible);
    }
}
