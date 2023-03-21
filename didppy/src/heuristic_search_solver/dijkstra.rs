use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{Dijkstra, Search};
use pyo3::prelude::*;
use std::rc::Rc;

/// Dijkstra's algorithm solver.
///
/// This performs Dijkstra's algorithm.
///
/// To apply this solver, the cost must be computed in the form of :code:`x + state_cost`, :code:`x * state_cost`, :code:`didppy.max(x, state_cost)`,
/// or :code:`didppy.min(x, state_cost)` where, :code:`state_cost` is either of :meth:`IntExpr.state_cost()` and :meth:`FloatExpr.state_cost()`,
/// and :code:`x` is a value independent of :code:`state_cost`.
/// In addition, the model must be minimization.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// time_limit: int, float, or None, default: None
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
///     If :code:`initial_registry_capacity` is negative.
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
/// >>> solver = dp.Dijkstra(model, quiet=True)
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
/// >>> solver = dp.Dijkstra(model, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "Dijkstra")]
#[pyo3(
    text_signature = "(model, time_limit=None, quiet=False, initial_registry_capacity=1000000, lazy=False)"
)]
pub struct DijkstraPy(WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>);

#[pymethods]
impl DijkstraPy {
    #[new]
    #[pyo3(signature = (
        model,
        time_limit = None,
        quiet = false,
        initial_registry_capacity = 1000000,
        lazy = false
    ))]
    fn new(
        model: &ModelPy,
        time_limit: Option<f64>,
        quiet: bool,
        initial_registry_capacity: usize,
        lazy: bool,
    ) -> DijkstraPy {
        if model.float_cost() {
            let parameters = dypdl_heuristic_search::Parameters::<OrderedContinuous> {
                primal_bound: None,
                time_limit,
                get_all_solutions: false,
                quiet,
            };
            let solver = Box::new(Dijkstra::<OrderedContinuous>::new(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                lazy,
                Some(initial_registry_capacity),
            ));
            DijkstraPy(WrappedSolver::Float(solver))
        } else {
            let parameters = dypdl_heuristic_search::Parameters::<Integer> {
                primal_bound: None,
                time_limit,
                get_all_solutions: false,
                quiet,
            };
            let solver = Box::new(Dijkstra::<Integer>::new(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                lazy,
                Some(initial_registry_capacity),
            ));
            DijkstraPy(WrappedSolver::Int(solver))
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
    /// >>> solver = dp.Dijkstra(model, quiet=True)
    /// >>> solution = solver.search()
    /// >>> print(solution.cost)
    /// 1
    #[pyo3(signature = ())]
    fn search(&mut self) -> PyResult<SolutionPy> {
        self.0.search()
    }
}
