use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::ModelPy;
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{create_dual_bound_lnbs, FEvaluatorType, LnbsParameters, Search};
use pyo3::prelude::*;
use std::rc::Rc;

/// Large Neighborhood Beam Search (LNBS) solver.
///
/// This performs LNBS using the dual bound as the heuristic function.
///
/// To apply this solver, the cost must be computed in the form of `x + state_cost`, `x * state_cost`, `didppy.max(x, state_cost)`,
/// or `didppy.min(x, state_cost)` where, `state_cost` is either of :func:`didppy.IntExpr.state_cost()` and :func:`didppy.FloatExpr.state_cost()`,
/// and `x` is a value independent of `state_cost`.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
///
/// LNBS searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If `keep_all_layers` is `True`, LNBS keeps states in all layers to check for duplicates.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// time_limit: int, float
///     Time limit.
///     The count starts when a solver is created.
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
/// no_bandit: bool, default: False
///     Whether to use bandit-based depth selection.
/// no_transition_constraints: bool, default: False
///     Whether to remove inapplicable transitions considering a suffix.
/// seed: int, default: 2023
///     Random seed.
/// initial_beam_size: int or None, default: None
///     Initial beam size of iterative beam search to find the initial feasible solution.
///     If `None`, this is set to `lns_initial_beam_size`.
/// keep_all_layers: bool, default: False
///     Keep all layers of the search graph for duplicate detection in memory.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
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
/// >>> solver = dp.LNBS(model, quiet=True)
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
/// >>> solver = dp.LNBS(model, time_limit=1800, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "LNBS")]
#[pyo3(
    text_signature = "(model, time_limit, f_operator=0, lns_initial_beam_size=1, has_negative_cost=False, no_cost_weight=False, no_bandit=False, no_transitoin_constraints=False, seed=2023, initial_beam_size=None,  keep_all_layers=False, primal_bound=None, quiet=False)"
)]
pub struct LnbsPy(WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>);

#[pymethods]
impl LnbsPy {
    #[new]
    #[pyo3(signature = (
        model,
        time_limit,
        f_operator = FOperator::Plus,
        lns_initial_beam_size = 1,
        has_negative_cost = false,
        no_cost_weight = false,
        no_bandit = false,
        no_transition_constraints = false,
        seed = 2023,
        initial_beam_size = None,
        keep_all_layers = false,
        primal_bound = None,
        quiet = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        time_limit: f64,
        f_operator: FOperator,
        lns_initial_beam_size: usize,
        has_negative_cost: bool,
        no_cost_weight: bool,
        no_bandit: bool,
        no_transition_constraints: bool,
        seed: u64,
        initial_beam_size: Option<usize>,
        keep_all_layers: bool,
        primal_bound: Option<&PyAny>,
        quiet: bool,
    ) -> PyResult<LnbsPy> {
        let f_evaluator_type = FEvaluatorType::from(f_operator);
        let initial_beam_size = initial_beam_size.unwrap_or(lns_initial_beam_size);

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
                time_limit: Some(time_limit),
                get_all_solutions: false,
                quiet,
            };
            let lnbs_parameters = LnbsParameters {
                initial_beam_size: lns_initial_beam_size,
                has_negative_cost,
                no_cost_weight,
                no_bandit,
                no_transition_constraints,
                seed,
            };
            let solver = create_dual_bound_lnbs::<OrderedContinuous>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                lnbs_parameters,
                f_evaluator_type,
                initial_beam_size,
                keep_all_layers,
            );
            Ok(LnbsPy(WrappedSolver::Float(solver)))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract::<Integer>()?)
            } else {
                None
            };
            let parameters = dypdl_heuristic_search::Parameters::<Integer> {
                primal_bound,
                time_limit: Some(time_limit),
                get_all_solutions: false,
                quiet,
            };
            let lnbs_parameters = LnbsParameters {
                initial_beam_size: lns_initial_beam_size,
                has_negative_cost,
                no_cost_weight,
                no_bandit,
                no_transition_constraints,
                seed,
            };
            let solver = create_dual_bound_lnbs::<Integer>(
                Rc::new(model.inner_as_ref().clone()),
                parameters,
                lnbs_parameters,
                f_evaluator_type,
                initial_beam_size,
                keep_all_layers,
            );
            Ok(LnbsPy(WrappedSolver::Int(solver)))
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
    /// >>> solver = dp.LNBS(model, quiet=True)
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
    /// >>> solver = dp.LNBS(model, quiet=True)
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
