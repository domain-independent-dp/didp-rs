use super::beam_parallelization_method::BeamParallelizationMethod;
use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::{ModelPy, TransitionPy};
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{
    create_dual_bound_lnbs, create_dual_bound_lnhdbs1, create_dual_bound_lnhdbs2,
    create_dual_bound_lnsbs, BeamSearchParameters, CabsParameters, FEvaluatorType, LnbsParameters,
    Parameters, Search,
};
use pyo3::prelude::*;
use std::rc::Rc;
use std::sync::Arc;

/// Large Neighborhood Beam Search (LNBS) solver.
///
/// This performs LNBS using the dual bound as the heuristic function.
/// LNBS is complete, i.e., eventually finds the optimal solution, but is designed to find a good solution rather than proving the optimality.
/// If you want to prove the optimality, :class:`didppy.CABS` or :class:`didppy.CAASDy` might be better.
/// LNBS typically performs well in routing and scheduling problems, where solution costs are diverse.
///
/// To apply this solver, the cost must be computed in the form of :code:`x + state_cost`, :code:`x * state_cost`, :code:`didppy.max(x, state_cost)`,
/// or :code:`didppy.min(x, state_cost)` where, :code:`state_cost` is either of :meth:`IntExpr.state_cost()` and :meth:`FloatExpr.state_cost()`,
/// and :code:`x` is a value independent of :code:`state_cost`.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
/// if :code:`x` can be negative, please set :code:`has_negative_cost` to :code:`True`.
///
/// LNBS searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If :code:`keep_all_layers` is :code:`True`, LNBS keeps states in all layers to check for duplicates.
///
/// Parameters
/// ----------
/// model: Model
///     DyPDL model to solve.
/// time_limit: int or float
///     Time limit.
///     This is required for LNBS.
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
///     If the cost is computed by :code:`+`, this should be :attr:`~FOperator.Plus`.
///     If the cost is computed by :code:`*`, this should be :attr:`~FOperator.Product`.
///     If the cost is computed by :code:`max`, this should be :attr:`~FOperator.Max`.
///     If the cost is computed by :code:`min`, this should be :attr:`~FOperator.Min`.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_solution: list of Transition or None, default: None
///     Initial feasible solution.
///     If :code:`None`, CABS is is performed to find an initial feasible solution.
/// initial_beam_size: int, default: 1
///     Initial beam size.
/// keep_all_layers: bool, default: False
///     Keep all layers of the search graph for duplicate detection in memory.
/// max_beam_size: int or None, default: None
///     Maximum beam size.
///     If :code:`None`, the beam size is kept increased until a partial state space is exhausted.
/// seed: int, default: 2023
///     Random seed.
/// has_negative_cost: bool, default: False
///     Whether the cost of a transition can be negative.
/// use_cost_weight: bool, default: False
///     Use weighted sampling biased by costs to select a start of a partial path.
///     This is not activated when :code:`has_negative_cost` is :code:`True`.
/// no_bandit: bool, default: False
///     Do not use bandit-based sampling to select the depth of a partial path.
/// no_transition_mutex: bool, default: False
///     Do not remove transitions conflicting with a suffix from a partial state space.
/// cabs_initial_beam_size: int or None, default: None
///     Initial beam size for CABS to find an initial feasible solution.
/// cabs_max_beam_size: int or None, default: None
///     Maximum beam size for CABS to find an initial feasible solution.
///     If :code:`None`, the beam size is kept increased until a feasible solution is found.
/// threads: int, default 1
///     Number of threads.
/// parallelization_method: BeamParallelizationMethod, default: BeamParallelizationMethod.Hdbs2
///     How to parallelize the search.
///     When `threads` is 1, this parameter is ignored.
///
/// Raises
/// ------
/// TypeError
///     If :code:`primal_bound` is :code:`float` and :code:`model` is int cost.
/// PanicException
///     If :code:`time_limit` is negative or CABS raises an exception when finding an initial solution.
///
/// References
/// ----------
///
/// Ryo Kuroiwa and J. Christopher Beck. "Large Neighborhood Beam Search for Domain-Independent Dynamic Programming,"
/// Proceedings of the 29th International Conference on Principles and Practice of Constraint Programming (CP), pp. 23:1-23:22, 2023.
///
/// Ryo Kuroiwa and J. Christopher Beck. "Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming,"
/// Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024.
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
/// >>> solver = dp.LNBS(model, time_limit=1800, quiet=True)
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
/// >>> solver = dp.LNBS(model, time_limit=1800, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "LNBS")]
pub struct LnbsPy(WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>);

#[pymethods]
impl LnbsPy {
    #[new]
    #[pyo3(
        text_signature = "(model, time_limit, f_operator=didppy.FOperator.Plus, primal_bound=None, quiet=False, initial_solution=None, initial_beam_size=1, keep_all_layers=False, max_beam_size=None, seed=2023, has_negative_cost=false, use_cost_weight=false, no_bandit=false, no_transition_mutex=false, cabs_initial_beam_size=None, cabs_max_beam_size=None, threads=1, parallelization_method=BeamParallelizationMethod.Hdbs2)"
    )]
    #[pyo3(signature = (
        model,
        time_limit,
        f_operator = FOperator::Plus,
        primal_bound = None,
        quiet = false,
        initial_solution = None,
        initial_beam_size = 1,
        keep_all_layers = false,
        max_beam_size = None,
        seed = 2023,
        has_negative_cost = false,
        use_cost_weight = false,
        no_bandit = false,
        no_transition_mutex = false,
        cabs_initial_beam_size = None,
        cabs_max_beam_size = None,
        threads = 1,
        parallelization_method = BeamParallelizationMethod::Hdbs2,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        time_limit: f64,
        f_operator: FOperator,
        primal_bound: Option<&PyAny>,
        quiet: bool,
        initial_solution: Option<Vec<TransitionPy>>,
        initial_beam_size: usize,
        keep_all_layers: bool,
        max_beam_size: Option<usize>,
        seed: u64,
        has_negative_cost: bool,
        use_cost_weight: bool,
        no_bandit: bool,
        no_transition_mutex: bool,
        cabs_initial_beam_size: Option<usize>,
        cabs_max_beam_size: Option<usize>,
        threads: usize,
        parallelization_method: BeamParallelizationMethod,
    ) -> PyResult<LnbsPy> {
        if !quiet {
            println!("Solver: LNBS from DIDPPy v{}", env!("CARGO_PKG_VERSION"));
        }

        let f_evaluator_type = FEvaluatorType::from(f_operator);
        let cabs_initial_beam_size = cabs_initial_beam_size.unwrap_or(initial_beam_size);
        let transitions = initial_solution
            .map(|transitions| transitions.into_iter().map(Transition::from).collect());

        if model.float_cost() {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            let base_parameters = Parameters::<OrderedContinuous> {
                primal_bound,
                time_limit: Some(time_limit),
                get_all_solutions: false,
                quiet,
                initial_registry_capacity: None,
            };
            let parameters = LnbsParameters {
                max_beam_size,
                seed,
                has_negative_cost,
                use_cost_weight,
                no_bandit,
                no_transition_mutex,
                beam_search_parameters: BeamSearchParameters {
                    beam_size: initial_beam_size,
                    keep_all_layers,
                    parameters: base_parameters,
                },
            };
            let cabs_parameters = CabsParameters {
                max_beam_size: cabs_max_beam_size,
                beam_search_parameters: BeamSearchParameters {
                    beam_size: cabs_initial_beam_size,
                    keep_all_layers,
                    parameters: base_parameters,
                },
            };
            let solver = if threads > 1 {
                let model = Arc::new(model.inner_as_ref().clone());
                match parallelization_method {
                    BeamParallelizationMethod::Hdbs2 => {
                        create_dual_bound_lnhdbs2::<OrderedContinuous>(
                            model,
                            transitions,
                            parameters,
                            cabs_parameters,
                            f_evaluator_type,
                            threads,
                        )
                    }
                    BeamParallelizationMethod::Hdbs1 => {
                        create_dual_bound_lnhdbs1::<OrderedContinuous>(
                            model,
                            transitions,
                            parameters,
                            cabs_parameters,
                            f_evaluator_type,
                            threads,
                        )
                    }
                    BeamParallelizationMethod::Sbs => create_dual_bound_lnsbs::<OrderedContinuous>(
                        model,
                        transitions,
                        parameters,
                        cabs_parameters,
                        f_evaluator_type,
                        threads,
                    ),
                }
            } else {
                create_dual_bound_lnbs::<OrderedContinuous>(
                    Rc::new(model.inner_as_ref().clone()),
                    transitions,
                    parameters,
                    cabs_parameters,
                    f_evaluator_type,
                )
            };
            Ok(LnbsPy(WrappedSolver::Float(solver)))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract::<Integer>()?)
            } else {
                None
            };
            let base_parameters = Parameters::<Integer> {
                primal_bound,
                time_limit: Some(time_limit),
                get_all_solutions: false,
                quiet,
                initial_registry_capacity: None,
            };
            let parameters = LnbsParameters {
                max_beam_size,
                seed,
                has_negative_cost,
                use_cost_weight,
                no_bandit,
                no_transition_mutex,
                beam_search_parameters: BeamSearchParameters {
                    beam_size: initial_beam_size,
                    keep_all_layers,
                    parameters: base_parameters,
                },
            };
            let cabs_parameters = CabsParameters {
                max_beam_size: cabs_max_beam_size,
                beam_search_parameters: BeamSearchParameters {
                    beam_size: cabs_initial_beam_size,
                    keep_all_layers,
                    parameters: base_parameters,
                },
            };
            let solver = if threads > 1 {
                let model = Arc::new(model.inner_as_ref().clone());
                match parallelization_method {
                    BeamParallelizationMethod::Hdbs2 => create_dual_bound_lnhdbs2::<Integer>(
                        model,
                        transitions,
                        parameters,
                        cabs_parameters,
                        f_evaluator_type,
                        threads,
                    ),
                    BeamParallelizationMethod::Hdbs1 => create_dual_bound_lnhdbs1::<Integer>(
                        model,
                        transitions,
                        parameters,
                        cabs_parameters,
                        f_evaluator_type,
                        threads,
                    ),
                    BeamParallelizationMethod::Sbs => create_dual_bound_lnsbs::<Integer>(
                        model,
                        transitions,
                        parameters,
                        cabs_parameters,
                        f_evaluator_type,
                        threads,
                    ),
                }
            } else {
                create_dual_bound_lnbs::<Integer>(
                    Rc::new(model.inner_as_ref().clone()),
                    transitions,
                    parameters,
                    cabs_parameters,
                    f_evaluator_type,
                )
            };
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
    /// >>> solver = dp.LNBS(model, time_limit=1800, quiet=True)
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
    /// >>> solver = dp.LNBS(model, time_limit=1800, quiet=True)
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
