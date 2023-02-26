use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{ForwardRecursion, Search};
use pyo3::prelude::*;
use std::rc::Rc;

/// Forward recursion solver.
///
/// This performs forward recursion while memoizing encountered states.
///
/// This solver can handle any types of cost expressions, but the state space must be acyclic.
/// If the state space is cyclic, this solver may fall in an infinite loop.
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
///
/// Raises
/// ------
/// OverflowError
///     If `initial_registry_capacity` is negative.
#[pyclass(unsendable, name = "ForwardRecursion")]
#[pyo3(text_signature = "(model, time_limit=None, quiet=False, initial_registry_capacity=1000000)")]
pub struct ForwardRecursionPy(
    WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>,
);

#[pymethods]
impl ForwardRecursionPy {
    #[new]
    #[pyo3(signature = (
        model,
        time_limit = None,
        quiet = false,
        initial_registry_capacity = 1000000
    ))]
    fn new(
        model: &ModelPy,
        time_limit: Option<f64>,
        quiet: bool,
        initial_registry_capacity: usize,
    ) -> ForwardRecursionPy {
        if model.float_cost() {
            let parameters = dypdl_heuristic_search::Parameters::<OrderedContinuous> {
                primal_bound: None,
                time_limit,
                get_all_solutions: false,
                quiet,
            };
            let solver = Box::new(ForwardRecursion::<OrderedContinuous>::new(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                Some(initial_registry_capacity),
            ));
            ForwardRecursionPy(WrappedSolver::Float(solver))
        } else {
            let parameters = dypdl_heuristic_search::Parameters::<Integer> {
                primal_bound: None,
                time_limit,
                get_all_solutions: false,
                quiet,
            };
            let solver = Box::new(ForwardRecursion::<Integer>::new(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                Some(initial_registry_capacity),
            ));
            ForwardRecursionPy(WrappedSolver::Int(solver))
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
