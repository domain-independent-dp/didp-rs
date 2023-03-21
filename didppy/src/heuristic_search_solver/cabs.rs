use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{create_dual_bound_cabs, FEvaluatorType, Search};
use pyo3::prelude::*;
use std::rc::Rc;

/// Complete Anytime Beam Search (CABS) solver.
///
/// This performs CABS using the dual bound as the heuristic function.
///
/// To apply this solver, the cost must be computed in the form of :code:`x + state_cost`, :code:`x * state_cost`, :code:`didppy.max(x, state_cost)`,
/// or :code:`didppy.min(x, state_cost)` where, :code:`state_cost` is either of :meth:`IntExpr.state_cost()` and :meth:`FloatExpr.state_cost()`,
/// and :code:`x` is a value independent of :code:`state_cost`.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
///
/// CABS searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If :code:`keep_all_layers` is :code:`True`, CABS keeps states in all layers to check for duplicates.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
///     If the cost is computed by :code:`+`, this should be :attr:`~FOperator.Plus`.
///     If the cost is computed by :code:`*`, this should be :attr:`~FOperator.Product`.
///     If the cost is computed by :code:`max`, this should be :attr:`~FOperator.Max`.
///     If the cost is computed by :code:`min`, this should be :attr:`~FOperator.Min`.
/// initial_beam_size: int, default: 1
///     Initial beam size.
/// keep_all_layers: bool, default: False
///     Keep all layers of the search graph for duplicate detection in memory.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int, float, or None, default: None
///     Time limit.
///     The count starts when a solver is created.
/// quiet: bool, default: False
///     Suppress the log output or not.
///
/// Raises
/// ------
/// TypeError
///     If :code:`primal_bound` is :code:`float` and :code:`model` is float cost.
/// PanicException
///     If :code:`time_limit` is negative.
///
/// References
/// ----------
/// Ryo Kuroiwa and J. Christopher Beck.
/// "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), 2023.
///
/// Weixiong Zhang.
/// "Complete Anytime Beam Search,"
/// Proceedings of the 15th National Conference on Artificial Intelligence/Innovative Applications of Artificial Intelligence (AAAI/IAAI), pp. 425-430, 1998.
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
/// >>> solver = dp.CABS(model, quiet=True)
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
/// >>> solver = dp.CABS(model, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "CABS")]
#[pyo3(
    text_signature = "(model, f_operator=0, keep_all_layers=False, primal_bound=None, time_limit=None, quiet=False)"
)]
pub struct CabsPy(WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>);

#[pymethods]
impl CabsPy {
    #[new]
    #[pyo3(signature = (
        model,
        f_operator = FOperator::Plus,
        initial_beam_size = 1,
        keep_all_layers = false,
        primal_bound = None,
        time_limit = None,
        quiet = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        f_operator: FOperator,
        initial_beam_size: usize,
        keep_all_layers: bool,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
    ) -> PyResult<CabsPy> {
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
                get_all_solutions: false,
                quiet,
            };
            let solver = create_dual_bound_cabs::<OrderedContinuous>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                f_evaluator_type,
                initial_beam_size,
                keep_all_layers,
            );
            Ok(CabsPy(WrappedSolver::Float(solver)))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract::<Integer>()?)
            } else {
                None
            };
            let parameters = dypdl_heuristic_search::Parameters::<Integer> {
                primal_bound,
                time_limit,
                get_all_solutions: false,
                quiet,
            };
            let solver = create_dual_bound_cabs::<Integer>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                f_evaluator_type,
                initial_beam_size,
                keep_all_layers,
            );
            Ok(CabsPy(WrappedSolver::Int(solver)))
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
    /// >>> solver = dp.CABS(model, quiet=True)
    /// >>> solution = solver.search()
    /// >>> solution.cost
    /// 1
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
    /// >>> solver = dp.CABS(model, quiet=True)
    /// >>> solution, terminated = solver.search_next()
    /// >>> solution.cost
    /// 1
    /// >>> terminated
    /// True
    #[pyo3(signature = ())]
    fn search_next(&mut self) -> PyResult<(SolutionPy, bool)> {
        self.0.search_next()
    }
}
