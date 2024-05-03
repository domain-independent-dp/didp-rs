use super::f_operator::FOperator;
use super::wrapped_solver::{SolutionPy, WrappedSolver};
use crate::model::{ModelPy, TransitionPy};
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{
    create_dual_bound_dd_lns, BeamSearchParameters, CabsParameters, DdLnsParameters,
    FEvaluatorType, Parameters, Search,
};
use pyo3::prelude::*;
use std::rc::Rc;

/// Large Neighborhood Search with Decision Diagrams (DD-LNS) solver.
///
/// This performs LNS by constructing restricted multi-valued decision diagrams (MDD).
///
/// To apply this solver, the cost must be computed in the form of :code:`x + state_cost`, :code:`x * state_cost`, :code:`didppy.max(x, state_cost)`,
/// or :code:`didppy.min(x, state_cost)` where, :code:`state_cost` is either of :meth:`IntExpr.state_cost()` and :meth:`FloatExpr.state_cost()`,
/// and :code:`x` is a value independent of :code:`state_cost`.
/// Otherwise, it cannot compute the cost correctly and may not produce the optimal solution.
/// if :code:`x` can be negative, please set :code:`has_negative_cost` to :code:`True`.
///
/// DD-LNS searches layer by layer, where the i th layer contains states that can be reached with i transitions.
/// By default, this solver only keeps states in the current layer to check for duplicates.
/// If :code:`keep_all_layers` is :code:`True`, DD-LNS keeps states in all layers to check for duplicates.
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
/// initial_solution: list of Transition or None, default: None
///     Initial feasible solution.
///     If :code:`None`, CABS is is performed to find an initial feasible solution.
/// beam_size: int, default: 1000
///     Beam size.
/// keep_probability: float, default: 0.1
///     Probability to keep a non-best state.
/// keep_all_layers: bool, default: False
///     Keep all layers of the search graph for duplicate detection in memory.
/// seed: int, default: 2023
///     Random seed.
/// cabs_initial_beam_size: int, default: 1
///     Initial beam size for CABS to find an initial feasible solution.
/// cabs_max_beam_size: int or None, default: None
///     Maximum beam size for CABS to find an initial feasible solution.
///     If :code:`None`, the beam size is kept increased until a feasible solution is found.
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
/// Xavier Gillard and Pierre Schaus. "Large Neighborhood Search with Decision Diagrams,"
/// Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI), pp. 4754-4760, 2022.
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
/// >>> solver = dp.DDLNS(model, quiet=True)
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
/// >>> solver = dp.DDLNS(model, f_operator=dp.FOperator.Max, quiet=True)
/// >>> solution = solver.search()
/// >>> print(solution.cost)
/// 2
#[pyclass(unsendable, name = "DDLNS")]
pub struct DdLnsPy(WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>>);

#[pymethods]
impl DdLnsPy {
    #[new]
    #[pyo3(
        text_signature = "(model, f_operator=didppy.FOperator.Plus, primal_bound=None, time_limit=None, quiet=False, initial_solution=None, beam_size=1000, keep_probability=0.1, keep_all_layers=False, seed=2023, cabs_initial_beam_size=None, cabs_max_beam_size=None)"
    )]
    #[pyo3(signature = (
        model,
        f_operator = FOperator::Plus,
        primal_bound = None,
        time_limit = None,
        quiet = false,
        initial_solution = None,
        beam_size = 1000,
        keep_probability = 0.1,
        keep_all_layers = false,
        seed = 2023,
        cabs_initial_beam_size = 1,
        cabs_max_beam_size = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &ModelPy,
        f_operator: FOperator,
        primal_bound: Option<Bound<'_, PyAny>>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_solution: Option<Vec<TransitionPy>>,
        beam_size: usize,
        keep_probability: f64,
        keep_all_layers: bool,
        seed: u64,
        cabs_initial_beam_size: usize,
        cabs_max_beam_size: Option<usize>,
    ) -> PyResult<DdLnsPy> {
        if !quiet {
            println!("Solver: DDLNS from DIDPPy v{}", env!("CARGO_PKG_VERSION"));
        }

        let f_evaluator_type = FEvaluatorType::from(f_operator);
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
                time_limit,
                get_all_solutions: false,
                quiet,
                initial_registry_capacity: None,
            };
            let parameters = DdLnsParameters {
                keep_probability,
                seed,
                beam_search_parameters: BeamSearchParameters {
                    beam_size,
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
            let solver = create_dual_bound_dd_lns::<OrderedContinuous>(
                Rc::new(model.inner_as_ref().clone()),
                transitions,
                parameters,
                cabs_parameters,
                f_evaluator_type,
            );
            Ok(DdLnsPy(WrappedSolver::Float(solver)))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract::<Integer>()?)
            } else {
                None
            };
            let base_parameters = Parameters::<Integer> {
                primal_bound,
                time_limit,
                get_all_solutions: false,
                quiet,
                initial_registry_capacity: None,
            };
            let parameters = DdLnsParameters {
                keep_probability,
                seed,
                beam_search_parameters: BeamSearchParameters {
                    beam_size,
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
            let solver = create_dual_bound_dd_lns::<Integer>(
                Rc::new(model.inner_as_ref().clone()),
                transitions,
                parameters,
                cabs_parameters,
                f_evaluator_type,
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
