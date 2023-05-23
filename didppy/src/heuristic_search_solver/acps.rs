use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{
    create_dual_bound_acps, FEvaluatorType, Parameters, ProgressiveSearchParameters, Search,
};
use pyo3::prelude::*;
use std::rc::Rc;

/// Anytime Column Progressive Search (ACPS) solver.
///
/// This performs ACPS using the dual bound as the heuristic function.
///
/// To apply this solver, the cost must be computed in the form of :code:`x + state_cost`, :code:`x * state_cost`, :code:`didppy.max(x, state_cost)`,
/// or :code:`didppy.min(x, state_cost)` where, :code:`state_cost` is either of :meth:`IntExpr.state_cost()` and :meth:`FloatExpr.state_cost()`,
/// and :code:`x` is a value independent of :code:`state_cost`.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
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
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int, float, or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     Initial size of the data structure storing all generated states.
/// width_init: int, default: 1
///     Initial value of the width.
/// width_step: int, default: 1
///     Amount of increase of the width.
/// width_bound: int or None, default: None
///     Maximum value of the width.
/// reset_width: bool, default: False
///     Reset the width to :code:`width_init` when a solution is found.
///
/// Raises
/// ------
/// TypeError
///     If the type of :code:`primal_bound` and the cost type of :code:`model` are different.
/// OverflowError
///     If :code:`initial_registry_capacity`, :code:`width_init`, :code:`width_step`, or :code:`width_bound` is negative.
/// PanicException
///     If :code:`time_limit` is negative.
///
/// References
/// ----------
/// Ryo Kuroiwa and J. Christopher Beck.
/// "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), 2023.
///
/// Sataya Gautam Vadlamudi, Piyush Gaurav, Sandip Aine, and Partha Pratim Chakrabarti.
/// "Anytime Column Search," Proceedings of AI 2012: Advances in Artificial Intelligence, pp. 254-255, 2012.
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
/// >>> solver = dp.ACPS(model, quiet=True)
/// >>> solution = solver.search()
/// >>> solution.cost
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
/// >>> solver = dp.ACPS(model, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> solution.cost
/// 2
#[pyclass(unsendable, name = "ACPS")]
#[pyo3(
    text_signature = "(model, f_operator=0, primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000, width_init=1, width_step=1, width_bound=None, reset_width=False)"
)]
pub struct AcpsPy(WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>);

#[pymethods]
impl AcpsPy {
    #[new]
    #[pyo3(signature = (
        model,
        f_operator = FOperator::Plus,
        primal_bound = None,
        time_limit = None,
        quiet = false,
        initial_registry_capacity = 1000000,
        width_init = 1,
        width_step = 1,
        width_bound = None,
        reset_width = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        f_operator: FOperator,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_registry_capacity: usize,
        width_init: usize,
        width_step: usize,
        width_bound: Option<usize>,
        reset_width: bool,
    ) -> PyResult<AcpsPy> {
        if !quiet {
            println!("Solver: ACPS from DIDPPy v{}", env!("CARGO_PKG_VERSION"));
        }

        let progressive_parameters = ProgressiveSearchParameters {
            init: width_init,
            step: width_step,
            bound: width_bound,
            reset: reset_width,
        };
        let f_evaluator_type = FEvaluatorType::from(f_operator);

        if model.float_cost() {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            let parameters = Parameters::<OrderedContinuous> {
                primal_bound,
                time_limit,
                get_all_solutions: false,
                quiet,
                initial_registry_capacity: Some(initial_registry_capacity),
            };
            let solver = create_dual_bound_acps::<OrderedContinuous>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                f_evaluator_type,
                progressive_parameters,
            );
            Ok(AcpsPy(WrappedSolver::Float(solver)))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract::<Integer>()?)
            } else {
                None
            };
            let parameters = Parameters::<Integer> {
                primal_bound,
                time_limit,
                get_all_solutions: false,
                quiet,
                initial_registry_capacity: Some(initial_registry_capacity),
            };
            let solver = create_dual_bound_acps::<Integer>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                f_evaluator_type,
                progressive_parameters,
            );
            Ok(AcpsPy(WrappedSolver::Int(solver)))
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
    /// >>> solver = dp.ACPS(model, quiet=True)
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
    /// >>> solver = dp.ACPS(model, quiet=True)
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
