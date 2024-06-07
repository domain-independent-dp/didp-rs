use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{
    create_dual_bound_breadth_first_search, BrfsParameters, FEvaluatorType, Parameters, Search,
};
use pyo3::prelude::*;
use std::rc::Rc;

/// Breadth-first search solver.
///
/// This performs breadth-first search using the dual bound as the heuristic function.
///
/// To apply this solver, the cost must be computed in the form of :code:`x + state_cost`, :code:`x * state_cost`, :code:`didppy.max(x, state_cost)`,
/// or :code:`didppy.min(x, state_cost)` where, :code:`state_cost` is either of :meth:`IntExpr.state_cost()` and :meth:`FloatExpr.state_cost()`,
/// and :code:`x` is a value independent of :code:`state_cost`.
/// Otherwise, it may not produce the optimal solution.
///
/// Breadth-first searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If :code:`keep_all_layers` is :code:`True`, this solver keeps states in all layers to check for duplicates.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// f_operator: FOperator, default: FOperator.Plus
///     If the cost is computed by :code:`+`, this should be :attr:`~FOperator.Plus`.
///     If the cost is computed by :code:`*`, this should be :attr:`~FOperator.Product`.
///     If the cost is computed by :code:`max`, this should be :attr:`~FOperator.Max`.
///     If the cost is computed by :code:`min`, this should be :attr:`~FOperator.Min`.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int, float, or None, default: None
///     Time limit.
/// get_all_solutions: bool, default: False
///     Return a solution if it is not improving when :code:`search_next()` is called.
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
///     If :code:`primal_bound` is :code:`float` and :code:`model` is int cost.
/// PanicException
///     If :code:`time_limit` is negative.
///
/// Examples
/// --------
/// Example with :code:`+` operator.
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
/// >>> solver = dp.BreadthFirstSearch(model, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 1
///
/// Example with :code:`max` operator.
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
/// >>> solver = dp.BreadthFirstSearch(model, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "BreadthFirstSearch")]
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
        primal_bound: Option<Bound<'_, PyAny>>,
        time_limit: Option<f64>,
        get_all_solutions: bool,
        quiet: bool,
        initial_registry_capacity: usize,
        keep_all_layers: bool,
    ) -> PyResult<BreadthFirstSearchPy> {
        if !quiet {
            println!(
                "Solver: BreadthFirstSearch from DIDPPy v{}",
                env!("CARGO_PKG_VERSION")
            );
        }

        let f_evaluator_type = FEvaluatorType::from(f_operator);

        if model.float_cost() {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            let parameters = BrfsParameters {
                keep_all_layers,
                parameters: Parameters::<OrderedContinuous> {
                    primal_bound,
                    time_limit,
                    get_all_solutions,
                    quiet,
                    initial_registry_capacity: Some(initial_registry_capacity),
                },
            };
            let solver = create_dual_bound_breadth_first_search::<OrderedContinuous>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                f_evaluator_type,
            );
            Ok(BreadthFirstSearchPy(WrappedSolver::Float(solver)))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract::<Integer>()?)
            } else {
                None
            };
            let parameters = BrfsParameters {
                keep_all_layers,
                parameters: Parameters::<Integer> {
                    primal_bound,
                    time_limit,
                    get_all_solutions,
                    quiet,
                    initial_registry_capacity: Some(initial_registry_capacity),
                },
            };
            let solver = create_dual_bound_breadth_first_search::<Integer>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                f_evaluator_type,
            );
            Ok(BreadthFirstSearchPy(WrappedSolver::Int(solver)))
        }
    }

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
    /// >>> solver = dp.BreadthFirstSearch(model, quiet=True)
    /// >>> solution = solver.search()
    /// >>> solution.cost
    /// 1
    fn search(&mut self) -> PyResult<SolutionPy> {
        self.0.search()
    }

    /// Search for the next solution of a DyPDL model.
    ///
    /// Returns
    /// -------
    /// solution: Solution
    ///     Solution.
    /// terminated: bool
    ///     Whether the search is terminated.
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
    /// >>> solver = dp.BreadthFirstSearch(model, quiet=True)
    /// >>> solution, terminated = solver.search_next()
    /// >>> solution.cost
    /// 1
    /// >>> terminated
    /// True
    fn search_next(&mut self) -> PyResult<(SolutionPy, bool)> {
        self.0.search_next()
    }
}
