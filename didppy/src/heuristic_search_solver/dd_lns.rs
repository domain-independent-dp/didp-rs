use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{create_dual_bound_dd_lns, DdLnsParameters, FEvaluatorType, Search};
use pyo3::prelude::*;
use std::rc::Rc;

/// DD-LNS solver.
///
/// This performs DD-LNS using the dual bound as the heuristic function.
///
/// To apply this solver, the cost must be computed in the form of `x + state_cost`, `x * state_cost`, `didppy.max(x, state_cost)`,
/// or `didppy.min(x, state_cost)` where, `state_cost` is either of :func:`didppy.IntExpr.state_cost()` and :func:`didppy.FloatExpr.state_cost()`,
/// and `x` is a value independent of `state_cost`.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
///
/// DD-LNS searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If `keep_all_layers` is `True`, DD-LNS keeps states in all layers to check for duplicates.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
///     If the cost is computed by `+`, this should be :attr:`~FOperator.Plus`.
///     If the cost is computed by `*`, this should be :attr:`~FOperator.Product`.
///     If the cost is computed by `max`, this should be :attr:`~FOperator.Max`.
///     If the cost is computed by `min`, this should be :attr:`~FOperator.Min`.
/// lns_initial_beam_size: int, default: 1
///     Initial beam size of beam search in neighborhoods.
/// has_negative_cost: bool, default: False
///     If the cost change can be negative.
/// no_cost_weight: bool, default: False
///     Whether to bias neighborhood selection by the cost change.
/// seed: int, default: 2023
///     Random seed.
/// initial_beam_size: int or None, default: None
///     Initial beam size of iterative beam search to find the initial feasible solution.
///     If `None`, this is set to `lns_initial_beam_size`.
/// keep_all_layers: bool, default: False
///     Keep all layers of the search graph for duplicate detection in memory.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int, float, or None
///     Time limit.
///     The count starts when a solver is created.
/// quiet: bool, default: False
///     Suppress the log output or not.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `model` is float cost.
/// OverflowError
///     If `lns_initial_beam_size`, `initial_beam_size`, or `seed` is negative.
/// PanicException
///     If `time_limit` is negative.
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
/// >>> solver = dp.DDLNS(model, quiet=True)
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
/// >>> solver = dp.DDLNS(model, time_limit=1800, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "DDLNS")]
#[pyo3(
    text_signature = "(model, f_operator=0, lns_beam_size=1000, keep_probability=0.1, seed=2023, initial_beam_size=1, keep_all_layers=False, primal_bound=None, time_limit=None, quiet=False)"
)]
pub struct DdLnsPy(WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>);

#[pymethods]
impl DdLnsPy {
    #[new]
    #[pyo3(signature = (
        model,
        f_operator = FOperator::Plus,
        lns_beam_size = 1000,
        keep_probability = 0.1,
        seed = 2023,
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
        lns_beam_size: usize,
        keep_probability: f64,
        seed: u64,
        initial_beam_size: usize,
        keep_all_layers: bool,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
    ) -> PyResult<DdLnsPy> {
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
            let dd_lns_parameters = DdLnsParameters {
                beam_size: lns_beam_size,
                keep_probability,
                seed,
            };
            let solver = create_dual_bound_dd_lns::<OrderedContinuous>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                dd_lns_parameters,
                f_evaluator_type,
                initial_beam_size,
                keep_all_layers,
            );
            Ok(DdLnsPy(WrappedSolver::Float(solver)))
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
            let dd_lns_parameters = DdLnsParameters {
                beam_size: lns_beam_size,
                keep_probability,
                seed,
            };
            let solver = create_dual_bound_dd_lns::<Integer>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                dd_lns_parameters,
                f_evaluator_type,
                initial_beam_size,
                keep_all_layers,
            );
            Ok(DdLnsPy(WrappedSolver::Int(solver)))
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
    /// >>> solver = dp.DDLNS(model, quiet=True)
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
    /// >>> solver = dp.DDLNS(model, quiet=True)
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
