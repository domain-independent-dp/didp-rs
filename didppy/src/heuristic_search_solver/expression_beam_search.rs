use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::CostUnion;
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::search_algorithm::BeamSearchParameters;
use dypdl_heuristic_search::{
    CustomExpressionParameters, ExpressionBeamSearch, FEvaluatorType, Parameters, Search,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;

/// Beam search solver using expressions to compute heuristic values.
///
/// This performs beam search using user-defined cost expressions as cost and heuristic functions.
///
/// To apply this solver, the user-defined cost must be computed in the form of `x + state_cost`, `x * state_cost`, `didppy.max(x, state_cost)`,
/// or `didppy.min(x, state_cost)` where, `state_cost` is either of :func:`didppy.IntExpr.state_cost()` and :func:`didppy.FloatExpr.state_cost()`,
/// and `x` is a value independent of `state_cost`.
/// Otherwise, it cannot compute the cost correctly.
///
/// This solver does not have a guarantee for optimality.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// beam_size: int
///     Beam size.
/// h_expression: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, float, or None, default: None
///     Expression to compute an h-value.
/// custom_cost_dict: dict[str, Union[IntExpr|IntVar|IntResourceVar|FloatExpr|FloatVar|FloatResourceVar|int|float] or None, default: None
///     Expressions to compute g-values.
///     A g-value is the cost of the path from the target state to the current state.
///     A key is the name of a transition, and the value is an expression to compute a g-value.
///     An expression can use `IntExpr.state_cost()` or `FloatExpr.state_cost()`, which returns the current g-value.
///     If the name of a transition is not included, its cost expression is used.
///     If `None`, the cost expressions in the DyPDL model are used for all transitions.
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
///     If the cost is computed by `+`, this should be :attr:`~FOperator.Plus`.
///     If the cost is computed by `*`, this should be :attr:`~FOperator.Product`.
///     If the cost is computed by `max`, this should be :attr:`~FOperator.Max`.
///     If the cost is computed by `min`, this should be :attr:`~FOperator.Min`.
///     This solver keeps top `b` best nodes with regard to f-values at each depth.
/// maximize: bool, default: False
///     Maximize f-values or not.
///     Greater f-values are better if `True`, and smaller are better if `False`.
/// keep_all_layers: bool, default: False
///     Keep all layers of the search graph for duplicate detection in memory.
/// float_custom_cost: bool or None, default: None
///     Use continuous values for g-, h-, and f-values.
///     The cost type of the model is used if `None`.
/// time_limit: int, float, or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
///
/// Raises
/// ------
/// TypeError
///     If the custom cost type is int and `h_evaluator` or a value in `custom_cost_dict` is `FloatExpr`, `FloatVar`, `FloatResourceVar`, or `float`.
/// OverflowError
///     If `beam_size` is negative.
///
/// Examples
/// --------
/// Example with `+` operator.
///
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> x = model.add_int_var(target=1)
/// >>> model.add_base_case([x == 0])
/// >>> t = dp.Transition(
/// ...     name="decrement",
/// ...     cost=1 + dp.IntExpr.state_cost(),
/// ...     effects=[(x, x - 1)]
/// ... )
/// >>> model.add_transition(t)
/// >>> model.add_dual_bound(x)
/// >>> solver = dp.ExpressionBeamSearch(model, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 1
///
/// Example with `max` operator.
///
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> x = model.add_int_var(target=2)
/// >>> model.add_base_case([x == 0])
/// >>> t = dp.Transition(
/// ...     name="decrement",
/// ...     cost=dp.max(x, dp.IntExpr.state_cost()),
/// ...     effects=[(x, x - 1)]
/// ... )
/// >>> model.add_transition(t)
/// >>> model.add_dual_bound(x)
/// >>> solver = dp.ExpressionBeamSearch(model, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "ExpressionBeamSearch")]
#[pyo3(
    text_signature = "(model, beam_size, h_expression=None, custom_cost_dict=None, f_operator=0, maximize=False, keep_all_layers=False, float_custom_cost=None, time_limit=None, quiet=False)"
)]
pub struct ExpressionBeamSearchPy(
    WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>,
);

impl ExpressionBeamSearchPy {
    fn create_custom_cost_vectors(
        model: &ModelPy,
        custom_cost_type: CostType,
        custom_cost_dict: &HashMap<String, CostUnion>,
    ) -> PyResult<(Vec<CostExpression>, Vec<CostExpression>)> {
        let mut custom_costs = vec![];
        for t in &model.inner_as_ref().forward_transitions {
            let cost = if let Some(cost) = custom_cost_dict.get(&t.get_full_name()) {
                CostExpression::from(cost.clone())
            } else {
                t.cost.clone()
            };
            match (custom_cost_type, cost) {
                (CostType::Integer, CostExpression::Continuous(_)) => {
                    return Err(PyTypeError::new_err(
                        "float cost expression is given while the custom cost type is integer",
                    ))
                }
                (_, cost) => custom_costs.push(cost),
            }
        }
        let mut forced_custom_costs = vec![];
        for t in &model.inner_as_ref().forward_forced_transitions {
            let cost = if let Some(cost) = custom_cost_dict.get(&t.get_full_name()) {
                CostExpression::from(cost.clone())
            } else {
                t.cost.clone()
            };
            match (custom_cost_type, cost) {
                (CostType::Integer, CostExpression::Continuous(_)) => {
                    return Err(PyTypeError::new_err(
                        "float cost expression is given while the custom cost type is integer",
                    ))
                }
                (_, cost) => forced_custom_costs.push(cost),
            }
        }
        Ok((custom_costs, forced_custom_costs))
    }
}

#[pymethods]
impl ExpressionBeamSearchPy {
    #[new]
    #[pyo3(signature = (
        model,
        beam_size,
        custom_cost_dict = None,
        h_expression = None,
        f_operator = FOperator::Plus,
        maximize = false,
        keep_all_layers = false,
        time_limit = None,
        quiet = false,
        float_custom_cost = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        beam_size: usize,
        custom_cost_dict: Option<HashMap<String, CostUnion>>,
        h_expression: Option<CostUnion>,
        f_operator: FOperator,
        maximize: bool,
        keep_all_layers: bool,
        time_limit: Option<f64>,
        quiet: bool,
        float_custom_cost: Option<bool>,
    ) -> PyResult<ExpressionBeamSearchPy> {
        let custom_cost_type =
            float_custom_cost.map_or(model.inner_as_ref().cost_type, |float_custom_cost| {
                if float_custom_cost {
                    CostType::Continuous
                } else {
                    CostType::Integer
                }
            });
        let custom_cost_dict = custom_cost_dict.unwrap_or_default();
        let (custom_costs, forced_custom_costs) =
            Self::create_custom_cost_vectors(model, custom_cost_type, &custom_cost_dict)?;
        let h_expression = h_expression.map(|expression| {
            CostExpression::from(expression).simplify(&model.inner_as_ref().table_registry)
        });
        let f_evaluator_type = FEvaluatorType::from(f_operator);
        let custom_expression_parameters = CustomExpressionParameters {
            custom_costs,
            forced_custom_costs,
            h_expression,
            custom_cost_type,
        };

        match (model.float_cost(), custom_cost_type) {
            (true, CostType::Continuous) => {
                let parameters = BeamSearchParameters {
                    beam_size,
                    maximize,
                    f_pruning: false,
                    f_bound: None,
                    keep_all_layers,
                    parameters: Parameters::<OrderedContinuous> {
                        primal_bound: None,
                        time_limit,
                        get_all_solutions: false,
                        quiet,
                    },
                };
                let solver = Box::new(
                    ExpressionBeamSearch::<OrderedContinuous, OrderedContinuous>::new(
                        Rc::new(model.inner_as_ref().clone()),
                        parameters,
                        custom_expression_parameters,
                        f_evaluator_type,
                    ),
                );
                Ok(ExpressionBeamSearchPy(WrappedSolver::Float(solver)))
            }
            (true, CostType::Integer) => {
                let parameters = BeamSearchParameters {
                    beam_size,
                    maximize,
                    f_pruning: false,
                    f_bound: None,
                    keep_all_layers,
                    parameters: Parameters::<OrderedContinuous> {
                        primal_bound: None,
                        time_limit,
                        get_all_solutions: false,
                        quiet,
                    },
                };
                let solver = Box::new(ExpressionBeamSearch::<OrderedContinuous, Integer>::new(
                    Rc::new(model.inner_as_ref().clone()),
                    parameters,
                    custom_expression_parameters,
                    f_evaluator_type,
                ));
                Ok(ExpressionBeamSearchPy(WrappedSolver::Float(solver)))
            }
            (false, CostType::Continuous) => {
                let parameters = BeamSearchParameters {
                    beam_size,
                    maximize,
                    f_pruning: false,
                    f_bound: None,
                    keep_all_layers,
                    parameters: Parameters::<Integer> {
                        primal_bound: None,
                        time_limit,
                        get_all_solutions: false,
                        quiet,
                    },
                };
                let solver = Box::new(ExpressionBeamSearch::<Integer, OrderedContinuous>::new(
                    Rc::new(model.inner_as_ref().clone()),
                    parameters,
                    custom_expression_parameters,
                    f_evaluator_type,
                ));
                Ok(ExpressionBeamSearchPy(WrappedSolver::Int(solver)))
            }
            (false, CostType::Integer) => {
                let parameters = BeamSearchParameters {
                    beam_size,
                    maximize,
                    f_pruning: false,
                    f_bound: None,
                    keep_all_layers,
                    parameters: Parameters::<Integer> {
                        primal_bound: None,
                        time_limit,
                        get_all_solutions: false,
                        quiet,
                    },
                };
                let solver = Box::new(ExpressionBeamSearch::<Integer, Integer>::new(
                    Rc::new(model.inner_as_ref().clone()),
                    parameters,
                    custom_expression_parameters,
                    f_evaluator_type,
                ));
                Ok(ExpressionBeamSearchPy(WrappedSolver::Int(solver)))
            }
        }
    }

    /// search()
    ///
    /// Search for the optimal solution of a DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If the model is invalid.
    ///
    /// Examples
    /// --------
    /// >>> import didppy as dp
    /// >>> model = dp.Model()
    /// >>> x = model.add_int_var(target=1)
    /// >>> model.add_base_case([x == 0])
    /// >>> t = dp.Transition(
    /// ...     name="decrement",
    /// ...     cost=1 + dp.IntExpr.state_cost(),
    /// ...     effects=[(x, x - 1)]
    /// ... )
    /// >>> model.add_transition(t)
    /// >>> model.add_dual_bound(x)
    /// >>> solver = dp.ExpressionBeamSearch(model, quiet=True)
    /// >>> solution = solver.search()
    /// >>> print(solution.cost)
    /// 1
    #[pyo3(signature = ())]
    fn search(&mut self) -> PyResult<SolutionPy> {
        self.0.search()
    }
}
