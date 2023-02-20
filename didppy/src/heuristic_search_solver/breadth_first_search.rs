use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{create_dual_bound_breadth_first_search, FEvaluatorType, Search};
use pyo3::prelude::*;
use std::rc::Rc;

/// Breadth-first search solver.
///
/// This performs breadth-first search using the dual bound as the heuristic function.
///
/// To apply this solver, the cost must be computed in the form of `x + state_cost`, `x * state_cost`, `dp.max(x, state_cost)`,
/// or `dp.min(x, state_cost)` where, `state_cost` is either of `dp.IntExpr.state_cost()` and `dp.FloatExpr.state_cost()`,
/// and `x` is a value independent of `state_cost`.
/// Otherwise, it may not produce the optimal solution.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// f_operator: FOperator, default: FOperator.Plus
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
/// keep_all_layers: bool, default: False
///     Keep all layers of the search graph for duplicate detection in memory.
///
/// Raises
/// ------
/// TypeError
///     If the type of `primal_bound` and the cost type of `model` are different.
/// OverflowError
///     If `time_limit` is negative.
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
/// >>> solver = dp.BreadthFirstSearch(model, quiet=True)
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
/// >>> solver = dp.BreadthFirstSearch(model, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "BreadthFirstSearch")]
#[pyo3(
    text_signature = "(model, f_operator=0, primal_bound=None, time_limit=None, quiet=False, keep_all_layers=False)"
)]
pub struct BreadthFirstSearchPy(
    WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>,
);

#[pymethods]
impl BreadthFirstSearchPy {
    #[new]
    #[pyo3(signature = (
        model,
        f_operator = FOperator::Plus,
        primal_bound = None,
        time_limit = None,
        get_all_solutions = false,
        quiet = false,
        initial_registry_capacity = 1000000,
        keep_all_layers = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        f_operator: FOperator,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        get_all_solutions: bool,
        quiet: bool,
        initial_registry_capacity: usize,
        keep_all_layers: bool,
    ) -> PyResult<BreadthFirstSearchPy> {
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
            let solver = create_dual_bound_breadth_first_search::<OrderedContinuous>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                f_evaluator_type,
                keep_all_layers,
                Some(initial_registry_capacity),
            );
            Ok(BreadthFirstSearchPy(WrappedSolver::Float(solver)))
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
            let solver = create_dual_bound_breadth_first_search::<Integer>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                f_evaluator_type,
                keep_all_layers,
                Some(initial_registry_capacity),
            );
            Ok(BreadthFirstSearchPy(WrappedSolver::Int(solver)))
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
