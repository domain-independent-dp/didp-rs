use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::data_structure::beam::Beam;
use super::search_algorithm::data_structure::state_registry::StateInRegistry;
use super::search_algorithm::data_structure::{
    BeamSearchNode, BeamSearchProblemInstance, SuccessorGenerator, TransitionWithCustomCost,
};
use super::search_algorithm::util::Parameters;
use super::search_algorithm::Search;
use super::search_algorithm::Solution;
use super::search_algorithm::{beam_search, BeamSearchParameters};
use dypdl::variable_type;
use dypdl::CostType;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

/// Beam search solver using user-defined cost functions.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
///
/// Beam search is guided by user-defined cost expressions.
/// For each transition, a user needs to define a cost expression that is used to guide the search.
/// In addition, a user can define a heuristic function.
/// The user-defined cost and the heuristic value are combined by `f_evaluator_type` to guide the search.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl::variable_type::OrderedContinuous;
/// use dypdl_heuristic_search::{FEvaluatorType, ExpressionBeamSearch, CustomExpressionParameters};
/// use dypdl_heuristic_search::search_algorithm::{BeamSearchParameters, Search};
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
/// let parameters = BeamSearchParameters::<Integer, OrderedContinuous> {
///     beam_size: 1,
///     maximize: true,
///     ..Default::default()
/// };
/// let f_evaluator_type = FEvaluatorType::Plus;
/// let custom_expression_parameters = CustomExpressionParameters {
///     custom_costs: vec![CostExpression::from(ContinuousExpression::Cost + 1.5)],
///     forced_custom_costs: Vec::default(),
///     h_expression: Some(CostExpression::from(ContinuousExpression::from(variable))),
///     custom_cost_type: CostType::Continuous,
/// };
///
/// let mut solver = ExpressionBeamSearch::new(
///     model, parameters, custom_expression_parameters, f_evaluator_type
/// );
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub struct ExpressionBeamSearch<T, U>
where
    T: variable_type::Numeric + fmt::Display + 'static,
    U: variable_type::Numeric + fmt::Display + 'static,
{
    model: Rc<dypdl::Model>,
    parameters: BeamSearchParameters<T, U>,
    custom_expression_parameters: CustomExpressionParameters,
    f_evaluator_type: FEvaluatorType,
    terminated: bool,
    solution: Solution<T>,
}

/// Parameters for custom cost expressions.
pub struct CustomExpressionParameters {
    /// Custom cost expressions for transitions.
    pub custom_costs: Vec<dypdl::CostExpression>,
    /// Custom cost expressions for forced transitions.
    pub forced_custom_costs: Vec<dypdl::CostExpression>,
    /// Expression for cost estimate .
    pub h_expression: Option<dypdl::CostExpression>,
    /// Type of the custom cost.
    pub custom_cost_type: CostType,
}

impl<T, U> ExpressionBeamSearch<T, U>
where
    T: variable_type::Numeric + fmt::Display + 'static,
    U: variable_type::Numeric + fmt::Display + 'static,
{
    /// Create a new beam search solver using user-defined cost functions.
    pub fn new(
        model: Rc<dypdl::Model>,
        parameters: BeamSearchParameters<T, U>,
        custom_expression_parameters: CustomExpressionParameters,
        f_evaluator_type: FEvaluatorType,
    ) -> ExpressionBeamSearch<T, U> {
        let f_evaluator_type = if custom_expression_parameters.h_expression.is_some() {
            f_evaluator_type
        } else {
            FEvaluatorType::Plus
        };

        ExpressionBeamSearch {
            model,
            parameters,
            custom_expression_parameters,
            f_evaluator_type,
            terminated: false,
            solution: Solution::default(),
        }
    }

    fn solve_inner<H>(
        &self,
        model: Rc<dypdl::Model>,
        generator: SuccessorGenerator<TransitionWithCustomCost>,
        h_evaluator: H,
    ) -> Solution<T>
    where
        U: variable_type::Numeric + Ord + fmt::Display + 'static,
        H: Fn(&StateInRegistry) -> Option<U>,
    {
        let beam_constructor = |beam_size| Beam::<T, U, BeamSearchNode<T, U>>::new(beam_size);
        let parameters = BeamSearchParameters {
            beam_size: self.parameters.beam_size,
            maximize: self.parameters.maximize,
            keep_all_layers: self.parameters.keep_all_layers,
            f_pruning: self.parameters.f_pruning,
            f_bound: None,
            parameters: Parameters {
                primal_bound: None,
                time_limit: self.parameters.parameters.time_limit,
                get_all_solutions: self.parameters.parameters.get_all_solutions,
                quiet: self.parameters.parameters.quiet,
            },
        };
        let f_evaluator = move |g, h, _: &StateInRegistry| self.f_evaluator_type.eval(g, h);
        let target = StateInRegistry::from(model.target.clone());
        let problem = BeamSearchProblemInstance {
            generator,
            cost: T::zero(),
            g: U::zero(),
            target,
            solution_suffix: &[],
        };
        let (solution, _) = beam_search(
            &problem,
            &beam_constructor,
            &h_evaluator,
            f_evaluator,
            parameters,
        );

        Solution {
            cost: solution.cost,
            best_bound: solution.best_bound,
            is_optimal: solution.is_optimal,
            is_infeasible: solution.is_infeasible,
            transitions: solution
                .transitions
                .into_iter()
                .map(dypdl::Transition::from)
                .collect(),
            expanded: solution.expanded,
            generated: solution.generated,
            time: solution.time,
            time_out: solution.time_out,
        }
    }
}

impl<T, U> Search<T> for ExpressionBeamSearch<T, U>
where
    T: variable_type::Numeric + fmt::Display + 'static,
    U: variable_type::Numeric + Ord + fmt::Display + 'static,
{
    fn search_next(&mut self) -> Result<(Solution<T>, bool), Box<dyn Error>> {
        if self.terminated {
            return Ok((self.solution.clone(), true));
        }

        let generator = SuccessorGenerator::from_model_with_custom_costs(
            self.model.clone(),
            &self.custom_expression_parameters.custom_costs,
            &self.custom_expression_parameters.forced_custom_costs,
            false,
        );

        let h_evaluator = |state: &StateInRegistry| {
            Some(
                self.custom_expression_parameters
                    .h_expression
                    .as_ref()
                    .map_or(U::zero(), |expression| {
                        expression.eval(state, &self.model.table_registry)
                    }),
            )
        };
        let solution = self.solve_inner(self.model.clone(), generator, h_evaluator);
        self.terminated = true;
        Ok((solution, true))
    }
}
