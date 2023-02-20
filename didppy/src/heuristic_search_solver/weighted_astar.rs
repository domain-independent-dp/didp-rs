use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{create_dual_bound_weighted_astar, FEvaluatorType, Search};
use pyo3::prelude::*;
use std::rc::Rc;

/// Weighted A* solver.
///
/// This performs weighted A* using the dual bound as the heuristic function.
///
/// To apply this solver, the cost must be computed in the form of `x + state_cost`, `x * state_cost`, `dp.max(x, state_cost)`,
/// or `dp.min(x, state_cost)` where, `state_cost` is either of `dp.IntExpr.state_cost()` and `dp.FloatExpr.state_cost()`,
/// and `x` is a value independent of `state_cost`.
/// In addition, the model must be minimization.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// time_limit: int or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     Initial size of the data structure storing all generated states.
/// lazy: bool, default: False
///     Lazily generate a state when it is expanded.
///
/// Raises
/// ------
/// OverflowError
///     If `time_limit` or `initial_registry_capacity` is negative.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// weight: float
///     Weight of the h-value.
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
///     If the cost is computed by `+`, this should be `FOperator.Plus`.
///     If the cost is computed by `*`, this should be `FOperator.Product`.
///     If the cost is computed by `max`, this should be `FOperator.Max`.
///     If the cost is computed by `min`, this should be `FOperator.Min`.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int or None, default: None
///     Time limit.
/// get_all_solutions: bool, default: False
///     Return a solution if it is not improving when `search_next()` is called.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     Initial size of the data structure storing all generated states.
///
/// Raises
/// ------
/// TypeError
///     If the type of `primal_bound` and the cost type of `model` are different.
/// OverflowError
///     If `time_limit` or `initial_registry_capacity` is negative.
///
/// Examples
/// -------
/// Example with `+` operator.
///
/// >>> import didppy as dp
/// >>> model = dp.Model()
/// >>> x = model.add_int_var(target=1)
/// >>> model.add_base_case([x == 0])
/// >>> t = dp.Transition(name="decrement", cost=1 + dp.IntExpr.state_cost(), effects=[(x, x - 1)])
/// >>> model.add_transition(t)
/// >>> model.add_dual_bound(x)
/// >>> solver = dp.WeightedAstar(model, 1.5, quiet=True)
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
/// >>> t = dp.Transition(name="decrement", cost=dp.max(x, dp.IntExpr.state_cost()), effects=[(x, x - 1)])
/// >>> model.add_transition(t)
/// >>> model.add_dual_bound(x)
/// >>> solver = dp.WeightedAstar(model, 1.5, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "WeightedAstar")]
#[pyo3(
    text_signature = "(model, weight, f_operator=0, primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000)"
)]
pub struct WeightedAstarPy(
    WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>,
);

#[pymethods]
impl WeightedAstarPy {
    #[new]
    #[pyo3(signature = (
        model,
        weight,
        f_operator = FOperator::Plus,
        primal_bound = None,
        time_limit = None,
        get_all_solutions = false,
        quiet = false,
        initial_registry_capacity = 1000000
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        weight: Continuous,
        f_operator: FOperator,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        get_all_solutions: bool,
        quiet: bool,
        initial_registry_capacity: usize,
    ) -> PyResult<WeightedAstarPy> {
        let f_evaluator_type = FEvaluatorType::from(f_operator);

        if model.float_cost() {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            let parameters = dypdl_heuristic_search::Parameters::<OrderedContinuous> {
                primal_bound,
                time_limit,
                get_all_solutions,
                quiet,
            };
            let solver = create_dual_bound_weighted_astar(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                weight,
                f_evaluator_type,
                Some(initial_registry_capacity),
            );
            Ok(WeightedAstarPy(WrappedSolver::Float(solver)))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract::<Integer>()?)
            } else {
                None
            };
            let parameters = dypdl_heuristic_search::Parameters::<Integer> {
                primal_bound,
                time_limit,
                get_all_solutions,
                quiet,
            };
            let solver = create_dual_bound_weighted_astar(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                weight,
                f_evaluator_type,
                Some(initial_registry_capacity),
            );
            Ok(WeightedAstarPy(WrappedSolver::Int(solver)))
        }
    }

    /// search()
    ///
    /// Search for a bounded-suboptimal solution of a DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    #[pyo3(signature = ())]
    fn search(&mut self) -> PyResult<SolutionPy> {
        self.0.search()
    }

    /// search_next()
    ///
    /// Search for the next solution of a DyPDL model.
    ///
    /// Returns
    /// -------
    /// solution: Solution
    ///     Solution.
    /// terminated: bool
    ///     Whether the search is terminated.
    #[pyo3(signature = ())]
    fn search_next(&mut self) -> PyResult<(SolutionPy, bool)> {
        self.0.search_next()
    }
}
