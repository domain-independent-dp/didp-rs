use crate::model::ModelPy;
use crate::transition::{CostUnion, TransitionPy};
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{
    CAASDy, Dijkstra, DualBoundDFBB, ExpressionBeamSearch, ExpressionEpsilonBeamSearch,
    ExpressionEvaluator, FEvaluatorType, ForwardRecursion, LazyDijkstra, Solution, Solver, IBDFS,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(FromPyObject, Debug, PartialEq, Clone, Copy)]
pub enum WrappedCost {
    #[pyo3(transparent, annotation = "int")]
    Int(Integer),
    #[pyo3(transparent, annotation = "float")]
    Float(Continuous),
}

impl IntoPy<Py<PyAny>> for WrappedCost {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Int(value) => value.into_py(py),
            Self::Float(value) => value.into_py(py),
        }
    }
}

/// A class representing a solution returned by a heuristic search solver.
#[pyclass(name = "Solution")]
#[derive(Debug, PartialEq, Clone)]
pub struct SolutionPy {
    /// int, float, or None : Solution cost. None if a solution is not found.
    #[pyo3(get)]
    pub cost: Option<WrappedCost>,
    /// int, float, or None : Best dual bound. None if the solver does not compute a dual bound.
    #[pyo3(get)]
    pub best_bound: Option<WrappedCost>,
    /// bool : If the solution is optimal or not.
    #[pyo3(get)]
    pub is_optimal: bool,
    /// bool : If the problem is infeasible or not.
    #[pyo3(get)]
    pub is_infeasible: bool,
    /// list of transitions: Sequence of transitions corresponding to the solution
    #[pyo3(get)]
    pub transitions: Vec<TransitionPy>,
    /// int : Number of expanded nodes.
    #[pyo3(get)]
    pub expanded: usize,
    /// int : Number of generated nodes.
    #[pyo3(get)]
    pub generated: usize,
    /// float : Elapsed time in seconds.
    #[pyo3(get)]
    pub time: f64,
}

impl From<Solution<Integer>> for SolutionPy {
    fn from(solution: Solution<Integer>) -> Self {
        Self {
            cost: solution.cost.map(WrappedCost::Int),
            best_bound: solution.best_bound.map(WrappedCost::Int),
            is_optimal: solution.is_optimal,
            is_infeasible: solution.is_infeasible,
            transitions: solution
                .transitions
                .into_iter()
                .map(TransitionPy::new)
                .collect(),
            expanded: solution.expanded,
            generated: solution.generated,
            time: solution.time,
        }
    }
}

impl From<Solution<OrderedContinuous>> for SolutionPy {
    fn from(solution: Solution<OrderedContinuous>) -> Self {
        Self {
            cost: solution.cost.map(|cost| WrappedCost::Float(cost.into())),
            best_bound: solution
                .best_bound
                .map(|best_bound| WrappedCost::Float(best_bound.into())),
            is_optimal: solution.is_optimal,
            is_infeasible: solution.is_infeasible,
            transitions: solution
                .transitions
                .into_iter()
                .map(TransitionPy::new)
                .collect(),
            expanded: solution.expanded,
            generated: solution.generated,
            time: solution.time,
        }
    }
}
enum WrappedSolver<T, U> {
    Int(T),
    Float(U),
}

impl<T: Solver<Integer>, U: Solver<OrderedContinuous>> WrappedSolver<T, U> {
    fn solve(&mut self, model: &ModelPy) -> PyResult<SolutionPy> {
        match (self, model.float_cost()) {
            (Self::Int(solver), false) => {
                let result = solver.solve(model.inner_as_ref());
                match result {
                    Ok(solution) => Ok(SolutionPy::from(solution)),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            (Self::Float(solver), true) => {
                let result = solver.solve(model.inner_as_ref());
                match result {
                    Ok(solution) => Ok(SolutionPy::from(solution)),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }
            (Self::Int(_), true) => Err(PyRuntimeError::new_err(
                "a continuous cost model is given to an integer cost solver",
            )),
            (Self::Float(_), false) => Err(PyRuntimeError::new_err(
                "an integer cost model is given to a continuous cost solver",
            )),
        }
    }

    fn set_primal_bound(&mut self, bound: &PyAny) -> PyResult<()> {
        match self {
            Self::Int(solver) => {
                let bound = bound.extract()?;
                solver.set_primal_bound(bound);
            }
            Self::Float(solver) => {
                let bound: Continuous = bound.extract()?;
                solver.set_primal_bound(OrderedContinuous::from(bound));
            }
        }
        Ok(())
    }

    fn get_primal_bound(&self) -> Option<WrappedCost> {
        match self {
            Self::Int(solver) => solver.get_primal_bound().map(WrappedCost::Int),
            Self::Float(solver) => solver
                .get_primal_bound()
                .map(|bound| WrappedCost::Float(Continuous::from(bound))),
        }
    }

    fn set_time_limit(&mut self, limit: f64) {
        match self {
            Self::Int(solver) => solver.set_time_limit(limit),
            Self::Float(solver) => solver.set_time_limit(limit),
        }
    }

    fn get_time_limit(&self) -> Option<f64> {
        match self {
            Self::Int(solver) => solver.get_time_limit(),
            Self::Float(solver) => solver.get_time_limit(),
        }
    }

    fn set_quiet(&mut self, is_quiet: bool) {
        match self {
            Self::Int(solver) => solver.set_quiet(is_quiet),
            Self::Float(solver) => solver.set_quiet(is_quiet),
        }
    }

    fn get_quiet(&self) -> bool {
        match self {
            Self::Int(solver) => solver.get_quiet(),
            Self::Float(solver) => solver.get_quiet(),
        }
    }
}

/// An enum representing an operator to compute the f-value combining an h-value and a g-value.
///
/// `Plus`: f = g + h
///
/// `Max`: f = max(g, h)
///
/// `Min`: f = min(g, h)
///
/// `Overwrite`: f = h
#[pyclass(name = "FOperator")]
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum FOperator {
    Plus,
    Max,
    Min,
    Overwrite,
}

impl From<FOperator> for FEvaluatorType {
    fn from(f_operator: FOperator) -> Self {
        match f_operator {
            FOperator::Plus => FEvaluatorType::Plus,
            FOperator::Max => FEvaluatorType::Max,
            FOperator::Min => FEvaluatorType::Min,
            FOperator::Overwrite => FEvaluatorType::Overwrite,
        }
    }
}

/// Cost-Algebraic A* Solver for DyPDL (CAASDy).
///
/// This performs cost-algebraic A* using the dual bound as the heuristic function.
/// The current implementation only supports cost-algebra with minimization and non-negative edge costs.
/// E.g., the shortest path and minimizing the maximum edge cost on a path.
///
/// Parameters
/// ----------
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     The initial size of the data structure storing all generated states.
/// float_cost: bool, default: False
///     Use continuous values for costs.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `float_cost=False`.
/// OverflowError
///     If `time_limit` or `initial_registry_capacity` is negative.
#[pyclass(name = "CAASDy")]
#[pyo3(
    text_signature = "(f_operator=FOperator.Plus, primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000, float_cost=False)"
)]
pub struct CAASDyPy(WrappedSolver<CAASDy<Integer>, CAASDy<OrderedContinuous>>);

#[pymethods]
impl CAASDyPy {
    #[new]
    #[args(
        f_operator = "FOperator::Plus",
        primal_bound = "None",
        time_limit = "None",
        quiet = "false",
        initial_registry_capacity = "1000000",
        float_cost = "false"
    )]
    fn new(
        f_operator: FOperator,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_registry_capacity: usize,
        float_cost: bool,
    ) -> PyResult<CAASDyPy> {
        let f_evaluator_type = FEvaluatorType::from(f_operator);
        if float_cost {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            Ok(Self(WrappedSolver::Float(CAASDy {
                f_evaluator_type,
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract()?)
            } else {
                None
            };
            Ok(Self(WrappedSolver::Int(CAASDy {
                f_evaluator_type,
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        }
    }

    /// solve(model)
    ///
    /// Tries to solve a DyPDL model.
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a continuous/integer cost model is given to a integer/continuous cost solver.
    #[pyo3(text_signature = "(model)")]
    fn solve(&mut self, model: &ModelPy) -> PyResult<SolutionPy> {
        self.0.solve(model)
    }

    /// int, float, or None : Primal bound.
    #[setter]
    fn set_primal_bound(&mut self, bound: &PyAny) -> PyResult<()> {
        self.0.set_primal_bound(bound)
    }

    #[getter]
    fn get_primal_bound(&self) -> Option<WrappedCost> {
        self.0.get_primal_bound()
    }

    /// int or None : Time limit.
    #[setter]
    fn set_time_limit(&mut self, limit: f64) {
        self.0.set_time_limit(limit)
    }

    #[getter]
    fn get_time_limit(&self) -> Option<f64> {
        self.0.get_time_limit()
    }

    /// bool : Suppress output or not.
    #[setter]
    fn set_quiet(&mut self, is_quiet: bool) {
        self.0.set_quiet(is_quiet)
    }

    #[getter]
    fn get_quiet(&self) -> bool {
        self.0.get_quiet()
    }
}

/// Dijkstra's algorithm solver.
///
/// The current implementation only supports cost-algebra with minimization and non-negative edge costs.
/// E.g., the shortest path and minimizing the maximum edge cost on a path.
///
/// Parameters
/// ----------
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     The initial size of the data structure storing all generated states.
/// float_cost: bool, default: False
///     Use continuous values for costs.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `float_cost=False`.
/// OverflowError
///     If `time_limit` or `initial_registry_capacity` is negative.
#[pyclass(name = "Dijkstra")]
#[pyo3(
    text_signature = "(primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000, float_cost=False)"
)]
pub struct DijkstraPy(WrappedSolver<Dijkstra<Integer>, Dijkstra<OrderedContinuous>>);

#[pymethods]
impl DijkstraPy {
    #[new]
    #[args(
        primal_bound = "None",
        time_limit = "None",
        quiet = "false",
        initial_registry_capacity = "1000000",
        float_cost = "false"
    )]
    fn new(
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_registry_capacity: usize,
        float_cost: bool,
    ) -> PyResult<DijkstraPy> {
        if float_cost {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            Ok(Self(WrappedSolver::Float(Dijkstra {
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract()?)
            } else {
                None
            };
            Ok(Self(WrappedSolver::Int(Dijkstra {
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        }
    }

    /// solve(model)
    ///
    /// Tries to solve a DyPDL model.
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a continuous/integer cost model is given to a integer/continuous cost solver.
    #[pyo3(text_signature = "(model)")]
    fn solve(&mut self, model: &ModelPy) -> PyResult<SolutionPy> {
        self.0.solve(model)
    }

    /// int, float, None : Primal bound.
    #[setter]
    fn set_primal_bound(&mut self, bound: &PyAny) -> PyResult<()> {
        self.0.set_primal_bound(bound)
    }

    #[getter]
    fn get_primal_bound(&self) -> Option<WrappedCost> {
        self.0.get_primal_bound()
    }

    /// int or None : Time limit.
    #[setter]
    fn set_time_limit(&mut self, limit: f64) {
        self.0.set_time_limit(limit)
    }

    #[getter]
    fn get_time_limit(&self) -> Option<f64> {
        self.0.get_time_limit()
    }

    /// bool : Suppress output or not.
    #[setter]
    fn set_quiet(&mut self, is_quiet: bool) {
        self.0.set_quiet(is_quiet)
    }

    #[getter]
    fn get_quiet(&self) -> bool {
        self.0.get_quiet()
    }
}

/// Lazy Dijkstra's algorithm solver.
///
/// Yet another implementation of Dijkstra's algorithm.
/// Pointers to parent nodes and transitions are stored in the open list, and a state is geenrated when it is expanded.
/// The current implementation only supports cost-algebra with minimization and non-negative edge costs.
///
/// Parameters
/// ----------
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     The initial size of the data structure storing all generated states.
/// float_cost: bool, default: False
///     Use continuous values for costs.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `float_cost=False`.
/// OverflowError
///     If `time_limit` or `initial_registry_capacity` is negative.
#[pyclass(name = "LazyDijkstra")]
#[pyo3(
    text_signature = "(primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000, float_cost=False)"
)]
pub struct LazyDijkstraPy(WrappedSolver<LazyDijkstra<Integer>, LazyDijkstra<OrderedContinuous>>);

#[pymethods]
impl LazyDijkstraPy {
    #[new]
    #[args(
        primal_bound = "None",
        time_limit = "None",
        quiet = "false",
        initial_registry_capacity = "1000000",
        float_cost = "false"
    )]
    fn new(
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_registry_capacity: usize,
        float_cost: bool,
    ) -> PyResult<LazyDijkstraPy> {
        if float_cost {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            Ok(Self(WrappedSolver::Float(LazyDijkstra {
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract()?)
            } else {
                None
            };
            Ok(Self(WrappedSolver::Int(LazyDijkstra {
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        }
    }

    /// solve(model)
    ///
    /// Tries to solve a DyPDL model.
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a continuous/integer cost model is given to a integer/continuous cost solver.
    #[pyo3(text_signature = "(model)")]
    fn solve(&mut self, model: &ModelPy) -> PyResult<SolutionPy> {
        self.0.solve(model)
    }

    /// int, float, or None : Primal bound.
    #[setter]
    fn set_primal_bound(&mut self, bound: &PyAny) -> PyResult<()> {
        self.0.set_primal_bound(bound)
    }

    #[getter]
    fn get_primal_bound(&self) -> Option<WrappedCost> {
        self.0.get_primal_bound()
    }

    /// int or None : Time limit.
    #[setter]
    fn set_time_limit(&mut self, limit: f64) {
        self.0.set_time_limit(limit)
    }

    #[getter]
    fn get_time_limit(&self) -> Option<f64> {
        self.0.get_time_limit()
    }

    /// bool : Suppress output or not.
    #[setter]
    fn set_quiet(&mut self, is_quiet: bool) {
        self.0.set_quiet(is_quiet)
    }

    #[getter]
    fn get_quiet(&self) -> bool {
        self.0.get_quiet()
    }
}

/// Depth-First Branch-and-Bound (DFBB) solver using the dual bound.
///
/// The current implementation only support cost minimization or maximization with associative cost computation.
/// E.g., the shortest path and the longest path.
///
/// Parameters
/// ----------
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     The initial size of the data structure storing all generated states.
/// float_cost: bool, default: False
///     Use continuous values for costs.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `float_cost=False`.
/// OverflowError
///     If `time_limit` or `initial_registry_capacity` is negative.
#[pyclass(name = "DualBoundDFBB")]
#[pyo3(
    text_signature = "(f_operator=FOperator.Plus, primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000, float_cost=False)"
)]
pub struct DualBoundDFBBPy(WrappedSolver<DualBoundDFBB<Integer>, DualBoundDFBB<OrderedContinuous>>);

#[pymethods]
impl DualBoundDFBBPy {
    #[new]
    #[args(
        f_operator = "FOperator::Plus",
        primal_bound = "None",
        time_limit = "None",
        quiet = "false",
        initial_registry_capacity = "1000000",
        float_cost = "false"
    )]
    fn new(
        f_operator: FOperator,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_registry_capacity: usize,
        float_cost: bool,
    ) -> PyResult<DualBoundDFBBPy> {
        let f_evaluator_type = FEvaluatorType::from(f_operator);
        if float_cost {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            Ok(Self(WrappedSolver::Float(DualBoundDFBB {
                f_evaluator_type,
                callback: Box::new(|_| {}),
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract()?)
            } else {
                None
            };
            Ok(Self(WrappedSolver::Int(DualBoundDFBB {
                f_evaluator_type,
                callback: Box::new(|_| {}),
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        }
    }

    /// solve(model)
    ///
    /// Tries to solve a DyPDL model.
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a continuous/integer cost model is given to a integer/continuous cost solver.
    #[pyo3(text_signature = "(model)")]
    fn solve(&mut self, model: &ModelPy) -> PyResult<SolutionPy> {
        self.0.solve(model)
    }

    /// int, float, None : Primal bound.
    #[setter]
    fn set_primal_bound(&mut self, bound: &PyAny) -> PyResult<()> {
        self.0.set_primal_bound(bound)
    }

    #[getter]
    fn get_primal_bound(&self) -> Option<WrappedCost> {
        self.0.get_primal_bound()
    }

    /// int or None : Time limit.
    #[setter]
    fn set_time_limit(&mut self, limit: f64) {
        self.0.set_time_limit(limit)
    }

    #[getter]
    fn get_time_limit(&self) -> Option<f64> {
        self.0.get_time_limit()
    }

    /// bool : Suppress output or not.
    #[setter]
    fn set_quiet(&mut self, is_quiet: bool) {
        self.0.set_quiet(is_quiet)
    }

    #[getter]
    fn get_quiet(&self) -> bool {
        self.0.get_quiet()
    }
}

/// Iterative Bounded Depth-First Search (IBDFS) solver.
///
/// This solver repeatedly performs depth-first search to find a solution having a better cost than the current primal bound.
/// It terminates when no solution is better than the current primal bound.
/// The current implementation only support cost minimization or maximization with associative cost computation.
/// E.g., the shortest path and the longest path.
///
/// Parameters
/// ----------
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     The initial size of the data structure storing all generated states.
/// float_cost: bool, default: False
///     Use continuous values for costs.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `float_cost=False`.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `float_cost=False`.
/// OverflowError
///     If `time_limit` or `initial_registry_capacity` is negative.
#[pyclass(name = "IBDFS")]
#[pyo3(
    text_signature = "(primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000, float_cost=False)"
)]
pub struct IBDFSPy(WrappedSolver<IBDFS<Integer>, IBDFS<OrderedContinuous>>);

#[pymethods]
impl IBDFSPy {
    #[new]
    #[args(
        primal_bound = "None",
        time_limit = "None",
        quiet = "false",
        initial_registry_capacity = "1000000",
        float_cost = "false"
    )]
    fn new(
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_registry_capacity: usize,
        float_cost: bool,
    ) -> PyResult<IBDFSPy> {
        if float_cost {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            Ok(Self(WrappedSolver::Float(IBDFS {
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract()?)
            } else {
                None
            };
            Ok(Self(WrappedSolver::Int(IBDFS {
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        }
    }

    /// solve(model)
    ///
    /// Tries to solve a DyPDL model.
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a continuous/integer cost model is given to a integer/continuous cost solver.
    #[pyo3(text_signature = "(model)")]
    fn solve(&mut self, model: &ModelPy) -> PyResult<SolutionPy> {
        self.0.solve(model)
    }

    /// int, float, or None : Primal bound.
    #[setter]
    fn set_primal_bound(&mut self, bound: &PyAny) -> PyResult<()> {
        self.0.set_primal_bound(bound)
    }

    #[getter]
    fn get_primal_bound(&self) -> Option<WrappedCost> {
        self.0.get_primal_bound()
    }

    /// int or None : Time limit.
    #[setter]
    fn set_time_limit(&mut self, limit: f64) {
        self.0.set_time_limit(limit)
    }

    #[getter]
    fn get_time_limit(&self) -> Option<f64> {
        self.0.get_time_limit()
    }

    /// bool : Suppress output or not.
    #[setter]
    fn set_quiet(&mut self, is_quiet: bool) {
        self.0.set_quiet(is_quiet)
    }

    #[getter]
    fn get_quiet(&self) -> bool {
        self.0.get_quiet()
    }
}

/// Forward recursion solver.
///
/// It performs a naive recursion while memoizing all encoutered states.
/// It works only if the state space is acyclic.
///
/// Parameters
/// ----------
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     The initial size of the data structure storing all generated states.
/// float_cost: bool, default: False
///     Use continuous values for costs.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `float_cost=False`.
/// OverflowError
///     If `time_limit` or `initial_registry_capacity` is negative.
#[pyclass(name = "ForwardRecursion")]
#[pyo3(
    text_signature = "(primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000, float_cost=False)"
)]
pub struct ForwardRecursionPy(
    WrappedSolver<ForwardRecursion<Integer>, ForwardRecursion<OrderedContinuous>>,
);

#[pymethods]
impl ForwardRecursionPy {
    #[new]
    #[args(
        primal_bound = "None",
        time_limit = "None",
        quiet = "false",
        initial_registry_capacity = "1000000",
        float_cost = "false"
    )]
    fn new(
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        initial_registry_capacity: usize,
        float_cost: bool,
    ) -> PyResult<ForwardRecursionPy> {
        if float_cost {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            Ok(Self(WrappedSolver::Float(ForwardRecursion {
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract()?)
            } else {
                None
            };
            Ok(Self(WrappedSolver::Int(ForwardRecursion {
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
                initial_registry_capacity: Some(initial_registry_capacity),
            })))
        }
    }

    /// solve(model)
    ///
    /// Tries to solve a DyPDL model.
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a continuous/integer cost model is given to a integer/continuous cost solver.
    #[pyo3(text_signature = "(model)")]
    fn solve(&mut self, model: &ModelPy) -> PyResult<SolutionPy> {
        self.0.solve(model)
    }

    /// int, float, or None : Primal bound.
    #[setter]
    fn set_primal_bound(&mut self, bound: &PyAny) -> PyResult<()> {
        self.0.set_primal_bound(bound)
    }

    #[getter]
    fn get_primal_bound(&self) -> Option<WrappedCost> {
        self.0.get_primal_bound()
    }

    /// int or None : Time limit.
    #[setter]
    fn set_time_limit(&mut self, limit: f64) {
        self.0.set_time_limit(limit)
    }

    #[getter]
    fn get_time_limit(&self) -> Option<f64> {
        self.0.get_time_limit()
    }

    /// bool : Suppress output or not.
    #[setter]
    fn set_quiet(&mut self, is_quiet: bool) {
        self.0.set_quiet(is_quiet)
    }

    #[getter]
    fn get_quiet(&self) -> bool {
        self.0.get_quiet()
    }
}

/// Beam search solver using expressions to compute heuristic values.
///
/// This solver does not have a guarantee for optimality.
///
/// Parameters
/// ----------
/// beam_sizes: list of int
///     Sequence of beam sizes.
///     The solver sequentially performs beam search with the beam size of `b` for each `b` in `beam_sizes`.
/// model: Model
///     DyPDL model to solve.
/// h_evaluator: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, float, or None, default: None
///     Expression to compute an h-value.
/// custom_cost_dict: dict[str, Union[IntExpr|IntVar|IntResourceVar|FloatExpr|FloatVar|FloatResourceVar|int|float]] or None, default: None
///     Expressions to compute g-values.
///     A g-value is the cost of the path from the target state to the current state.
///     A key is the name of a transition, and the value is an expression to comptue a g-value.
///     An expression can use `IntExpr.state_cost()` or `FloatExpr.state_cost()`, which returns the current g-value.
///     If the name of a transition is not included, its cost expression is used.
///     If `None`, the cost expressoins are used for all transitions.
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
///     This solver keeps top `b` best nodes with regard to f-values at each depth.
/// maximize: bool, default: False
///     Maximize f-values or not.
///     Greater f-values are better if `True`, and smaller are better if `False`.
/// float_custom_cost: bool, default: False
///     Use continuous values for g-, h-, and f-values.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     The initial size of the data structure storing all generated states.
/// float_cost: bool, default: False
///     Use continuous values for costs.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `float_cost=False`.
///     If `float_custom_cost=False` and `h_evaluator` or a value in `custom_cost_dict` is `FloatExpr`, `FloatVar`, `FloatResouceVar`, or `float`.
/// OverflowError
///     If a value in `beam_sizes`, `time_limit`, or `initial_registry_capacity` is negative.
#[pyclass(name = "ExpressionBeamSearch")]
#[pyo3(
    text_signature = "(beam_sizes, model, h_evaluator=None, custom_cost_dict=None, float_custom_cost=False, primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000, float_cost=False)"
)]
pub struct ExpressionBeamSearchPy(
    WrappedSolver<ExpressionBeamSearch<Integer>, ExpressionBeamSearch<OrderedContinuous>>,
);

impl ExpressionBeamSearchPy {
    fn create_custom_cost_vectors(
        model: &ModelPy,
        custom_cost_type: CostType,
        custom_cost_dict: &HashMap<String, CostUnion>,
    ) -> PyResult<(Vec<CostExpression>, Vec<CostExpression>)> {
        let mut custom_costs = vec![];
        for t in &model.inner_as_ref().forward_transitions {
            let cost = if let Some(cost) = custom_cost_dict.get(&t.get_full_name()) {
                CostExpression::from(cost.clone())
            } else {
                t.cost.clone()
            };
            match (custom_cost_type, cost) {
                (CostType::Integer, CostExpression::Continuous(_)) => {
                    return Err(PyTypeError::new_err(
                        "float cost expression is given while the custom cost type is integer",
                    ))
                }
                (_, cost) => custom_costs.push(cost),
            }
        }
        let mut forced_custom_costs = vec![];
        for t in &model.inner_as_ref().forward_forced_transitions {
            let cost = if let Some(cost) = custom_cost_dict.get(&t.get_full_name()) {
                CostExpression::from(cost.clone())
            } else {
                t.cost.clone()
            };
            match (custom_cost_type, cost) {
                (CostType::Integer, CostExpression::Continuous(_)) => {
                    return Err(PyTypeError::new_err(
                        "float cost expression is given while the custom cost type is integer",
                    ))
                }
                (_, cost) => forced_custom_costs.push(cost),
            }
        }
        Ok((custom_costs, forced_custom_costs))
    }

    fn create_h_evaluator(
        model: &ModelPy,
        custom_cost_type: CostType,
        h_evaluator: Option<CostUnion>,
    ) -> PyResult<ExpressionEvaluator> {
        if let Some(h_evaluator) = h_evaluator {
            match (custom_cost_type, h_evaluator) {
                (CostType::Integer, CostUnion::Float(_)) => Err(PyTypeError::new_err(
                    "float expression is given while the custom cost type is integer",
                )),
                (_, cost) => Ok(ExpressionEvaluator::new(
                    CostExpression::from(cost).simplify(&model.inner_as_ref().table_registry),
                )),
            }
        } else {
            match custom_cost_type {
                CostType::Integer => Ok(ExpressionEvaluator::new(CostExpression::Integer(
                    IntegerExpression::Constant(0),
                ))),
                CostType::Continuous => Ok(ExpressionEvaluator::new(CostExpression::Continuous(
                    ContinuousExpression::Constant(0.0),
                ))),
            }
        }
    }
}

#[pymethods]
impl ExpressionBeamSearchPy {
    #[new]
    #[args(
        custom_cost_dict = "None",
        h_evaluator = "None",
        f_operator = "FOperator::Plus",
        maximize = "false",
        float_custom_cost = "false",
        primal_bound = "None",
        time_limit = "None",
        quiet = "false",
        float_cost = "false"
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        beam_sizes: Vec<usize>,
        model: &ModelPy,
        custom_cost_dict: Option<HashMap<String, CostUnion>>,
        h_evaluator: Option<CostUnion>,
        f_operator: FOperator,
        maximize: bool,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        float_custom_cost: bool,
        float_cost: bool,
    ) -> PyResult<ExpressionBeamSearchPy> {
        let custom_cost_type = if float_custom_cost {
            CostType::Continuous
        } else {
            CostType::Integer
        };
        let custom_cost_dict = custom_cost_dict.unwrap_or_default();
        let (custom_costs, forced_custom_costs) =
            Self::create_custom_cost_vectors(model, custom_cost_type, &custom_cost_dict)?;
        let h_evaluator = Self::create_h_evaluator(model, custom_cost_type, h_evaluator)?;
        let f_evaluator_type = FEvaluatorType::from(f_operator);

        if float_cost {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            Ok(Self(WrappedSolver::Float(ExpressionBeamSearch {
                custom_costs,
                forced_custom_costs,
                h_evaluator,
                f_evaluator_type,
                custom_cost_type: Some(custom_cost_type),
                beam_sizes,
                maximize,
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
            })))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract()?)
            } else {
                None
            };
            Ok(Self(WrappedSolver::Int(ExpressionBeamSearch {
                custom_costs,
                forced_custom_costs,
                h_evaluator,
                f_evaluator_type,
                custom_cost_type: Some(custom_cost_type),
                beam_sizes,
                maximize,
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
            })))
        }
    }

    /// solve(model)
    ///
    /// Tries to solve a DyPDL model.
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a continuous/integer cost model is given to a integer/continuous cost solver.
    #[pyo3(text_signature = "(model)")]
    fn solve(&mut self, model: &ModelPy) -> PyResult<SolutionPy> {
        self.0.solve(model)
    }

    /// int, float, or None : Primal bound.
    #[setter]
    fn set_primal_bound(&mut self, bound: &PyAny) -> PyResult<()> {
        self.0.set_primal_bound(bound)
    }

    #[getter]
    fn get_primal_bound(&self) -> Option<WrappedCost> {
        self.0.get_primal_bound()
    }

    /// int or None : Time limit.
    #[setter]
    fn set_time_limit(&mut self, limit: f64) {
        self.0.set_time_limit(limit)
    }

    #[getter]
    fn get_time_limit(&self) -> Option<f64> {
        self.0.get_time_limit()
    }

    /// bool : Suppress output or not.
    #[setter]
    fn set_quiet(&mut self, is_quiet: bool) {
        self.0.set_quiet(is_quiet)
    }

    #[getter]
    fn get_quiet(&self) -> bool {
        self.0.get_quiet()
    }
}

/// Epsilon Beam search solver using expressions to compute heuristic values.
///
/// Epsilon beam search inserts a node into the beam regardless of its heuristic value
/// with the probability of epsilon.
/// Such a node is kept in a separated list from other nodes.
/// When the beam is full, a node having the worst heuristic value is removed.
/// If the beam only contains randomly selected nodes, a node is randomly removed.
/// This solver does not have a guarantee for optimality.
///
/// Parameters
/// ----------
/// beam_sizes: list of int
///     Sequence of beam sizes.
///     The solver sequentially performs beam search with the beam size of `b` for each `b` in `beam_sizes`.
/// epsilon: bool
///     Epsilon.
/// model: Model
///     DyPDL model to solve.
/// h_evaluator: IntExpr, IntVar, IntResourceVar, FloatExpr, FloatVar, FloatResourceVar, int, float, or None, default: None
///     Expression to compute an h-value.
/// custom_cost_dict: dict[str, Union[IntExpr|IntVar|IntResourceVar|FloatExpr|FloatVar|FloatResourceVar|int|float]] or None, default: None
///     Expressions to compute g-values.
///     A g-value is the cost of the path from the target state to the current state.
///     A key is the name of a transition, and the value is an expression to comptue a g-value.
///     An expression can use `IntExpr.state_cost()` or `FloatExpr.state_cost()`, which returns the current g-value.
///     If the name of a transition is not included, its cost expression is used.
///     If `None`, the cost expressoins are used for all transitions.
/// f_operator: FOperator, default: FOperator.Plus
///     Operator to combine a g-value and the dual bound to compute the f-value.
///     This solver keeps top `b` best nodes with regard to f-values at each depth.
/// maximize: bool, default: False
///     Maximize f-values or not.
///     Greater f-values are better if `True`, and smaller are better if `False`.
/// float_custom_cost: bool, default: False
///     Use continuous values for g-, h-, and f-values.
/// primal_bound: int, float, or None, default: None
///     Primal bound.
/// time_limit: int or None, default: None
///     Time limit.
/// quiet: bool, default: False
///     Suppress the log output or not.
/// initial_registry_capacity: int, default: 1000000
///     The initial size of the data structure storing all generated states.
/// float_cost: bool, default: False
///     Use continuous values for costs.
///
/// Raises
/// ------
/// TypeError
///     If `primal_bound` is `float` and `float_cost=False`.
///     If `float_custom_cost=False` and `h_evaluator` or a value in `custom_cost_dict` is `FloatExpr`, `FloatVar`, `FloatResouceVar`, or `float`.
/// OverflowError
///     If a value in `beam_sizes`, `time_limit`, or `initial_registry_capacity` is negative.
#[pyclass(name = "ExpressionEpsilonBeamSearch")]
#[pyo3(
    text_signature = "(beam_sizes, epsilon, model, h_evaluator=None, custom_cost_dict=None, float_custom_cost=False, primal_bound=None, time_limit=None, quiet=False, initial_registry_capacity=1000000, float_cost=False)"
)]
pub struct ExpressionEpsilonBeamSearchPy(
    WrappedSolver<
        ExpressionEpsilonBeamSearch<Integer>,
        ExpressionEpsilonBeamSearch<OrderedContinuous>,
    >,
);

#[pymethods]
impl ExpressionEpsilonBeamSearchPy {
    #[new]
    #[args(
        custom_cost_dict = "None",
        h_evaluator = "None",
        f_operator = "FOperator::Plus",
        maximize = "false",
        float_custom_cost = "false",
        primal_bound = "None",
        time_limit = "None",
        quiet = "false",
        float_cost = "false",
        seed = "42"
    )]
    #[allow(clippy::too_many_arguments)]
    fn new(
        beam_sizes: Vec<usize>,
        epsilon: f64,
        model: &ModelPy,
        custom_cost_dict: Option<HashMap<String, CostUnion>>,
        h_evaluator: Option<CostUnion>,
        f_operator: FOperator,
        maximize: bool,
        primal_bound: Option<&PyAny>,
        time_limit: Option<f64>,
        quiet: bool,
        float_custom_cost: bool,
        float_cost: bool,
        seed: u64,
    ) -> PyResult<ExpressionEpsilonBeamSearchPy> {
        let custom_cost_type = if float_custom_cost {
            CostType::Continuous
        } else {
            CostType::Integer
        };
        let custom_cost_dict = custom_cost_dict.unwrap_or_default();
        let (custom_costs, forced_custom_costs) =
            ExpressionBeamSearchPy::create_custom_cost_vectors(
                model,
                custom_cost_type,
                &custom_cost_dict,
            )?;
        let h_evaluator =
            ExpressionBeamSearchPy::create_h_evaluator(model, custom_cost_type, h_evaluator)?;
        let f_evaluator_type = FEvaluatorType::from(f_operator);

        if float_cost {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(OrderedContinuous::from(
                    primal_bound.extract::<Continuous>()?,
                ))
            } else {
                None
            };
            Ok(Self(WrappedSolver::Float(ExpressionEpsilonBeamSearch {
                custom_costs,
                forced_custom_costs,
                h_evaluator,
                f_evaluator_type,
                custom_cost_type: Some(custom_cost_type),
                beam_sizes,
                maximize,
                epsilon,
                seed,
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
            })))
        } else {
            let primal_bound = if let Some(primal_bound) = primal_bound {
                Some(primal_bound.extract()?)
            } else {
                None
            };
            Ok(Self(WrappedSolver::Int(ExpressionEpsilonBeamSearch {
                custom_costs,
                forced_custom_costs,
                h_evaluator,
                f_evaluator_type,
                custom_cost_type: Some(custom_cost_type),
                beam_sizes,
                maximize,
                epsilon,
                seed,
                parameters: dypdl_heuristic_search::SolverParameters {
                    primal_bound,
                    time_limit,
                    quiet,
                },
            })))
        }
    }

    /// solve(model)
    ///
    /// Tries to solve a DyPDL model.
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// Solution
    ///     Solution.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a continuous/integer cost model is given to a integer/continuous cost solver.
    #[pyo3(text_signature = "(model)")]
    fn solve(&mut self, model: &ModelPy) -> PyResult<SolutionPy> {
        self.0.solve(model)
    }

    /// int, float, or None : Primal bound.
    #[setter]
    fn set_primal_bound(&mut self, bound: &PyAny) -> PyResult<()> {
        self.0.set_primal_bound(bound)
    }

    #[getter]
    fn get_primal_bound(&self) -> Option<WrappedCost> {
        self.0.get_primal_bound()
    }

    /// int or None : Time limit.
    #[setter]
    fn set_time_limit(&mut self, limit: f64) {
        self.0.set_time_limit(limit)
    }

    #[getter]
    fn get_time_limit(&self) -> Option<f64> {
        self.0.get_time_limit()
    }

    /// bool : Suppress output or not.
    #[setter]
    fn set_quiet(&mut self, is_quiet: bool) {
        self.0.set_quiet(is_quiet)
    }

    #[getter]
    fn get_quiet(&self) -> bool {
        self.0.get_quiet()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use dypdl::expression::*;

    #[test]
    fn solution_from_py_int() {
        let solution = Solution {
            cost: Some(0),
            best_bound: Some(0),
            is_optimal: true,
            is_infeasible: false,
            transitions: vec![Transition::default()],
            expanded: 1,
            generated: 1,
            time: 0.0,
        };
        assert_eq!(
            SolutionPy::from(solution),
            SolutionPy {
                cost: Some(WrappedCost::Int(0)),
                best_bound: Some(WrappedCost::Int(0)),
                is_optimal: true,
                is_infeasible: false,
                transitions: vec![TransitionPy::new(Transition::default())],
                expanded: 1,
                generated: 1,
                time: 0.0,
            }
        );
    }

    #[test]
    fn solution_from_py_float() {
        let solution = Solution {
            cost: Some(OrderedContinuous::from(0.0)),
            best_bound: Some(OrderedContinuous::from(0.0)),
            is_optimal: true,
            is_infeasible: false,
            transitions: vec![Transition::default()],
            expanded: 1,
            generated: 1,
            time: 0.0,
        };
        assert_eq!(
            SolutionPy::from(solution),
            SolutionPy {
                cost: Some(WrappedCost::Float(0.0)),
                best_bound: Some(WrappedCost::Float(0.0)),
                is_optimal: true,
                is_infeasible: false,
                transitions: vec![TransitionPy::new(Transition::default())],
                expanded: 1,
                generated: 1,
                time: 0.0,
            }
        );
    }

    #[test]
    fn f_evaluator_type_from_f_operator() {
        assert_eq!(FEvaluatorType::from(FOperator::Plus), FEvaluatorType::Plus);
        assert_eq!(FEvaluatorType::from(FOperator::Max), FEvaluatorType::Max);
        assert_eq!(FEvaluatorType::from(FOperator::Min), FEvaluatorType::Min);
        assert_eq!(
            FEvaluatorType::from(FOperator::Overwrite),
            FEvaluatorType::Overwrite
        );
    }

    #[test]
    fn caasdy_new_int_ok() {
        let solver = CAASDyPy::new(FOperator::Plus, None, None, false, 1000000, false);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn caasdy_new_int_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100i32.into_py(py);
            CAASDyPy::new(
                FOperator::Plus,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                1000000,
                false,
            )
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, Some(100));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn caasdy_new_int_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.5f64.into_py(py);
            CAASDyPy::new(
                FOperator::Plus,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                1000000,
                false,
            )
        });
        assert!(solver.is_err());
    }

    #[test]
    fn caasdy_new_float_ok() {
        let solver = CAASDyPy::new(FOperator::Plus, None, None, false, 1000000, true);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn caasdy_new_float_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.0f64.into_py(py);
            CAASDyPy::new(
                FOperator::Plus,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                1000000,
                true,
            )
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(100.0))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn caasdy_new_float_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound: Py<PyAny> = (100i32, 100i32).into_py(py);
            CAASDyPy::new(
                FOperator::Plus,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                1000000,
                true,
            )
        });
        assert!(solver.is_err());
    }

    #[test]
    fn caasdy_solve_int_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = CAASDyPy(WrappedSolver::Int(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn caasdy_solve_float_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = CAASDyPy(WrappedSolver::Float(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn caasdy_solve_int_float_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = CAASDyPy(WrappedSolver::Int(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn caasdy_solve_float_int_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = CAASDyPy(WrappedSolver::Float(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn caasdy_set_primal_bound_int_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = CAASDyPy(WrappedSolver::Int(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1i32.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, Some(1));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn caasdy_set_primal_bound_float_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = CAASDyPy(WrappedSolver::Float(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(1.5))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn caasdy_set_primal_bound_err() {
        pyo3::prepare_freethreaded_python();

        let mut solver = CAASDyPy(WrappedSolver::Int(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn caasdy_get_primal_bound_int() {
        let solver = CAASDyPy(WrappedSolver::Int(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(10),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Int(10)));
    }

    #[test]
    fn caasdy_get_primal_bound_float() {
        let solver = CAASDyPy(WrappedSolver::Float(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(OrderedContinuous::from(10.0)),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Float(10.0)));
    }

    #[test]
    fn caasdy_set_time_limit_int() {
        let mut solver = CAASDyPy(WrappedSolver::Int(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn caasdy_set_time_limit_float() {
        let mut solver = CAASDyPy(WrappedSolver::Float(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn caasdy_get_time_limit_int() {
        let solver = CAASDyPy(WrappedSolver::Int(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn caasdy_get_time_limit_float() {
        let solver = CAASDyPy(WrappedSolver::Float(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn caasdy_set_quiet_int() {
        let mut solver = CAASDyPy(WrappedSolver::Int(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn caasdy_set_quiet_float() {
        let mut solver = CAASDyPy(WrappedSolver::Float(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn caasdy_get_quiet_int() {
        let solver = CAASDyPy(WrappedSolver::Int(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn caasdy_get_quiet_float() {
        let solver = CAASDyPy(WrappedSolver::Float(CAASDy {
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn dijkstra_new_int_ok() {
        let solver = DijkstraPy::new(None, None, false, 1000000, false);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dijkstra_new_int_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100i32.into_py(py);
            DijkstraPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, false)
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, Some(100));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dijkstra_new_int_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.5f64.into_py(py);
            DijkstraPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, false)
        });
        assert!(solver.is_err());
    }

    #[test]
    fn dijkstra_new_float_ok() {
        let solver = DijkstraPy::new(None, None, false, 1000000, true);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dijkstra_new_float_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.0f64.into_py(py);
            DijkstraPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, true)
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(100.0))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dijkstra_new_float_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound: Py<PyAny> = (100i32, 100i32).into_py(py);
            DijkstraPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, true)
        });
        assert!(solver.is_err());
    }

    #[test]
    fn dijkstra_solve_int_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = DijkstraPy(WrappedSolver::Int(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn dijkstra_solve_float_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = DijkstraPy(WrappedSolver::Float(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn dijkstra_solver_int_float_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = DijkstraPy(WrappedSolver::Int(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn dijkstra_solve_float_int_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = DijkstraPy(WrappedSolver::Float(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn dijkstra_set_primal_bound_int_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = DijkstraPy(WrappedSolver::Int(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1i32.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, Some(1));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dijkstra_set_primal_bound_float_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = DijkstraPy(WrappedSolver::Float(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(1.5))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dijkstra_set_primal_bound_err() {
        pyo3::prepare_freethreaded_python();

        let mut solver = DijkstraPy(WrappedSolver::Int(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn dijkstra_get_primal_bound_int() {
        let solver = DijkstraPy(WrappedSolver::Int(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(10),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Int(10)));
    }

    #[test]
    fn dijkstra_get_primal_bound_float() {
        let solver = DijkstraPy(WrappedSolver::Float(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(OrderedContinuous::from(10.0)),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Float(10.0)));
    }

    #[test]
    fn dijkstra_set_time_limit_int() {
        let mut solver = DijkstraPy(WrappedSolver::Int(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dijkstra_set_time_limit_float() {
        let mut solver = DijkstraPy(WrappedSolver::Float(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dijkstra_get_time_limit_int() {
        let solver = DijkstraPy(WrappedSolver::Int(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn dijkstra_get_time_limit_float() {
        let solver = DijkstraPy(WrappedSolver::Float(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn dijkstra_set_quiet_int() {
        let mut solver = DijkstraPy(WrappedSolver::Int(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dijkstra_set_quiet_float() {
        let mut solver = DijkstraPy(WrappedSolver::Float(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dijkstra_get_quiet_int() {
        let solver = DijkstraPy(WrappedSolver::Int(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn dijkstra_get_quiet_float() {
        let solver = DijkstraPy(WrappedSolver::Float(Dijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn lazy_dijkstra_new_int_ok() {
        let solver = LazyDijkstraPy::new(None, None, false, 1000000, false);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn lazy_dijkstra_new_int_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100i32.into_py(py);
            LazyDijkstraPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, false)
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, Some(100));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn lazy_dijkstra_new_int_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.5f64.into_py(py);
            LazyDijkstraPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, false)
        });
        assert!(solver.is_err());
    }

    #[test]
    fn lazy_dijkstra_new_float_ok() {
        let solver = LazyDijkstraPy::new(None, None, false, 1000000, true);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn lazy_dijkstra_new_float_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.0f64.into_py(py);
            LazyDijkstraPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, true)
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(100.0))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn lazy_dijkstra_new_float_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound: Py<PyAny> = (100i32, 100i32).into_py(py);
            LazyDijkstraPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, true)
        });
        assert!(solver.is_err());
    }

    #[test]
    fn lazy_dijkstra_solve_int_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = LazyDijkstraPy(WrappedSolver::Int(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn lazy_dijkstra_solve_float_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = LazyDijkstraPy(WrappedSolver::Float(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn lazy_dijkstra_solver_int_float_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = LazyDijkstraPy(WrappedSolver::Int(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn lazy_dijkstra_solve_float_int_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = LazyDijkstraPy(WrappedSolver::Float(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn lazy_dijkstra_set_primal_bound_int_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = LazyDijkstraPy(WrappedSolver::Int(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1i32.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, Some(1));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn lazy_dijkstra_set_primal_bound_float_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = LazyDijkstraPy(WrappedSolver::Float(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(1.5))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn lazy_dijkstra_set_primal_bound_err() {
        pyo3::prepare_freethreaded_python();

        let mut solver = LazyDijkstraPy(WrappedSolver::Int(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn lazy_dijkstra_get_primal_bound_int() {
        let solver = LazyDijkstraPy(WrappedSolver::Int(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(10),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Int(10)));
    }

    #[test]
    fn lazy_dijkstra_get_primal_bound_float() {
        let solver = LazyDijkstraPy(WrappedSolver::Float(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(OrderedContinuous::from(10.0)),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Float(10.0)));
    }

    #[test]
    fn lazy_dijkstra_set_time_limit_int() {
        let mut solver = LazyDijkstraPy(WrappedSolver::Int(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn lazy_dijkstra_set_time_limit_float() {
        let mut solver = LazyDijkstraPy(WrappedSolver::Float(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn lazy_dijkstra_get_time_limit_int() {
        let solver = LazyDijkstraPy(WrappedSolver::Int(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn lazy_dijkstra_get_time_limit_float() {
        let solver = LazyDijkstraPy(WrappedSolver::Float(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn lazy_dijkstra_set_quiet_int() {
        let mut solver = LazyDijkstraPy(WrappedSolver::Int(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn lazy_dijkstra_set_quiet_float() {
        let mut solver = LazyDijkstraPy(WrappedSolver::Float(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn lazy_dijkstra_get_quiet_int() {
        let solver = LazyDijkstraPy(WrappedSolver::Int(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn lazy_dijkstra_get_quiet_float() {
        let solver = LazyDijkstraPy(WrappedSolver::Float(LazyDijkstra {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn dual_bound_dfbb_new_int_ok() {
        let solver = DualBoundDFBBPy::new(FOperator::Plus, None, None, false, 1000000, false);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dual_bound_dfbb_new_int_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100i32.into_py(py);
            DualBoundDFBBPy::new(
                FOperator::Plus,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                1000000,
                false,
            )
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, Some(100));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dual_bound_dfbb_new_int_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.5f64.into_py(py);
            DualBoundDFBBPy::new(
                FOperator::Plus,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                1000000,
                false,
            )
        });
        assert!(solver.is_err());
    }

    #[test]
    fn dual_bound_dfbb_new_float_ok() {
        let solver = DualBoundDFBBPy::new(FOperator::Plus, None, None, false, 1000000, true);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dual_bound_dfbb_new_float_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.0f64.into_py(py);
            DualBoundDFBBPy::new(
                FOperator::Plus,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                1000000,
                true,
            )
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(100.0))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dual_bound_dfbb_new_float_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound: Py<PyAny> = (100i32, 100i32).into_py(py);
            DualBoundDFBBPy::new(
                FOperator::Plus,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                1000000,
                true,
            )
        });
        assert!(solver.is_err());
    }

    #[test]
    fn dual_bound_dfbb_solve_int_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = DualBoundDFBBPy(WrappedSolver::Int(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn dual_bound_dfbb_solve_float_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = DualBoundDFBBPy(WrappedSolver::Float(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn dual_bound_dfbb_solver_int_float_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = DualBoundDFBBPy(WrappedSolver::Int(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn dual_bound_dfbb_solve_float_int_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = DualBoundDFBBPy(WrappedSolver::Float(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn dual_bound_dfbb_set_primal_bound_int_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = DualBoundDFBBPy(WrappedSolver::Int(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1i32.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, Some(1));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dual_bound_dfbb_set_primal_bound_float_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = DualBoundDFBBPy(WrappedSolver::Float(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(1.5))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dual_bound_dfbb_set_primal_bound_err() {
        pyo3::prepare_freethreaded_python();

        let mut solver = DualBoundDFBBPy(WrappedSolver::Int(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn dual_bound_dfbb_get_primal_bound_int() {
        let solver = DualBoundDFBBPy(WrappedSolver::Int(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(10),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Int(10)));
    }

    #[test]
    fn dual_bound_dfbb_get_primal_bound_float() {
        let solver = DualBoundDFBBPy(WrappedSolver::Float(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(OrderedContinuous::from(10.0)),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Float(10.0)));
    }

    #[test]
    fn dual_bound_dfbb_set_time_limit_int() {
        let mut solver = DualBoundDFBBPy(WrappedSolver::Int(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dual_bound_dfbb_set_time_limit_float() {
        let mut solver = DualBoundDFBBPy(WrappedSolver::Float(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dual_bound_dfbb_get_time_limit_int() {
        let solver = DualBoundDFBBPy(WrappedSolver::Int(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn dual_bound_dfbb_get_time_limit_float() {
        let solver = DualBoundDFBBPy(WrappedSolver::Float(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn dual_bound_dfbb_set_quiet_int() {
        let mut solver = DualBoundDFBBPy(WrappedSolver::Int(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn dual_bound_dfbb_set_quiet_float() {
        let mut solver = DualBoundDFBBPy(WrappedSolver::Float(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn dual_bound_dfbb_get_quiet_int() {
        let solver = DualBoundDFBBPy(WrappedSolver::Int(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn dual_bound_dfbb_get_quiet_float() {
        let solver = DualBoundDFBBPy(WrappedSolver::Float(DualBoundDFBB {
            f_evaluator_type: FEvaluatorType::Plus,
            callback: Box::new(|_| {}),
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn ibdfs_new_int_ok() {
        let solver = IBDFSPy::new(None, None, false, 1000000, false);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn ibdfs_new_int_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100i32.into_py(py);
            IBDFSPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, false)
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, Some(100));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn ibdfs_new_int_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.5f64.into_py(py);
            IBDFSPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, false)
        });
        assert!(solver.is_err());
    }

    #[test]
    fn ibdfs_new_float_ok() {
        let solver = IBDFSPy::new(None, None, false, 1000000, true);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn ibdfs_new_float_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.0f64.into_py(py);
            IBDFSPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, true)
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(100.0))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn ibdfs_new_float_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound: Py<PyAny> = (100i32, 100i32).into_py(py);
            IBDFSPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, true)
        });
        assert!(solver.is_err());
    }

    #[test]
    fn ibdfs_solve_int_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = IBDFSPy(WrappedSolver::Int(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn ibdfs_solve_float_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = IBDFSPy(WrappedSolver::Float(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn ibdfs_solver_int_float_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = IBDFSPy(WrappedSolver::Int(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn ibdfs_solve_float_int_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = IBDFSPy(WrappedSolver::Float(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn ibdfs_set_primal_bound_int_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = IBDFSPy(WrappedSolver::Int(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1i32.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, Some(1));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn ibdfs_set_primal_bound_float_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = IBDFSPy(WrappedSolver::Float(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(1.5))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn ibdfs_set_primal_bound_err() {
        pyo3::prepare_freethreaded_python();

        let mut solver = IBDFSPy(WrappedSolver::Int(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn ibdfs_get_primal_bound_int() {
        let solver = IBDFSPy(WrappedSolver::Int(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(10),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Int(10)));
    }

    #[test]
    fn ibdfs_get_primal_bound_float() {
        let solver = IBDFSPy(WrappedSolver::Float(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(OrderedContinuous::from(10.0)),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Float(10.0)));
    }

    #[test]
    fn ibdfs_set_time_limit_int() {
        let mut solver = IBDFSPy(WrappedSolver::Int(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn ibdfs_set_time_limit_float() {
        let mut solver = IBDFSPy(WrappedSolver::Float(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn ibdfs_get_time_limit_int() {
        let solver = IBDFSPy(WrappedSolver::Int(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn ibdfs_get_time_limit_float() {
        let solver = IBDFSPy(WrappedSolver::Float(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn ibdfs_set_quiet_int() {
        let mut solver = IBDFSPy(WrappedSolver::Int(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn ibdfs_set_quiet_float() {
        let mut solver = IBDFSPy(WrappedSolver::Float(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn ibdfs_get_quiet_int() {
        let solver = IBDFSPy(WrappedSolver::Int(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn ibdfs_get_quiet_float() {
        let solver = IBDFSPy(WrappedSolver::Float(IBDFS {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn forward_recursion_new_int_ok() {
        let solver = ForwardRecursionPy::new(None, None, false, 1000000, false);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn forward_recursion_new_int_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100i32.into_py(py);
            ForwardRecursionPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, false)
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, Some(100));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn forward_recursion_new_int_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.5f64.into_py(py);
            ForwardRecursionPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, false)
        });
        assert!(solver.is_err());
    }

    #[test]
    fn forward_recursion_new_float_ok() {
        let solver = ForwardRecursionPy::new(None, None, false, 1000000, true);
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn forward_recursion_new_float_with_primal_bound_ok() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound = 100.0f64.into_py(py);
            ForwardRecursionPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, true)
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(100.0))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn forward_recursion_new_float_err() {
        pyo3::prepare_freethreaded_python();

        let solver = Python::with_gil(|py| {
            let primal_bound: Py<PyAny> = (100i32, 100i32).into_py(py);
            ForwardRecursionPy::new(Some(primal_bound.as_ref(py)), None, false, 1000000, true)
        });
        assert!(solver.is_err());
    }

    #[test]
    fn forward_recursion_solve_int_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = ForwardRecursionPy(WrappedSolver::Int(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn forward_recursion_solve_float_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = ForwardRecursionPy(WrappedSolver::Float(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn forward_recursion_solver_int_float_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = ForwardRecursionPy(WrappedSolver::Int(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn forward_recursion_solve_float_int_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = ForwardRecursionPy(WrappedSolver::Float(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn forward_recursion_set_primal_bound_int_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = ForwardRecursionPy(WrappedSolver::Int(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1i32.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, Some(1));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn forward_recursion_set_primal_bound_float_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = ForwardRecursionPy(WrappedSolver::Float(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(1.5))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn forward_recursion_set_primal_bound_err() {
        pyo3::prepare_freethreaded_python();

        let mut solver = ForwardRecursionPy(WrappedSolver::Int(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn forward_recursion_get_primal_bound_int() {
        let solver = ForwardRecursionPy(WrappedSolver::Int(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(10),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Int(10)));
    }

    #[test]
    fn forward_recursion_get_primal_bound_float() {
        let solver = ForwardRecursionPy(WrappedSolver::Float(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(OrderedContinuous::from(10.0)),
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Float(10.0)));
    }

    #[test]
    fn forward_recursion_set_time_limit_int() {
        let mut solver = ForwardRecursionPy(WrappedSolver::Int(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn forward_recursion_set_time_limit_float() {
        let mut solver = ForwardRecursionPy(WrappedSolver::Float(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn forward_recursion_get_time_limit_int() {
        let solver = ForwardRecursionPy(WrappedSolver::Int(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn forward_recursion_get_time_limit_float() {
        let solver = ForwardRecursionPy(WrappedSolver::Float(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn forward_recursion_set_quiet_int() {
        let mut solver = ForwardRecursionPy(WrappedSolver::Int(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn forward_recursion_set_quiet_float() {
        let mut solver = ForwardRecursionPy(WrappedSolver::Float(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
                assert_eq!(solver.initial_registry_capacity, Some(1000000));
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn forward_recursion_get_quiet_int() {
        let solver = ForwardRecursionPy(WrappedSolver::Int(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn forward_recursion_get_quiet_float() {
        let solver = ForwardRecursionPy(WrappedSolver::Float(ForwardRecursion {
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
            initial_registry_capacity: Some(1000000),
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn expression_beam_new_no_custom_cost_int_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            forward_transitions: vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Constant(0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let solver = ExpressionBeamSearchPy::new(
            vec![10],
            &model,
            None,
            None,
            FOperator::Plus,
            false,
            None,
            None,
            false,
            false,
            false,
        );
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(
                    solver.custom_costs,
                    vec![CostExpression::Integer(IntegerExpression::Constant(0))]
                );
                assert_eq!(
                    solver.forced_custom_costs,
                    vec![CostExpression::Integer(IntegerExpression::Constant(1))]
                );
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Integer(IntegerExpression::Constant(
                        0
                    )))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Integer));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn expression_beam_new_with_primal_bound_int_ok() {
        pyo3::prepare_freethreaded_python();

        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            forward_transitions: vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Constant(0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let solver = Python::with_gil(|py| {
            let primal_bound = 100i32.into_py(py);
            ExpressionBeamSearchPy::new(
                vec![10],
                &model,
                None,
                None,
                FOperator::Plus,
                false,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                false,
                false,
            )
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(
                    solver.custom_costs,
                    vec![CostExpression::Integer(IntegerExpression::Constant(0))]
                );
                assert_eq!(
                    solver.forced_custom_costs,
                    vec![CostExpression::Integer(IntegerExpression::Constant(1))]
                );
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Integer(IntegerExpression::Constant(
                        0
                    )))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Integer));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, Some(100));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn expression_beam_new_with_primal_bound_err() {
        pyo3::prepare_freethreaded_python();

        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            forward_transitions: vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Constant(0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let solver = Python::with_gil(|py| {
            let primal_bound = 100.5f64.into_py(py);
            ExpressionBeamSearchPy::new(
                vec![10],
                &model,
                None,
                None,
                FOperator::Plus,
                false,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                false,
                false,
            )
        });
        assert!(solver.is_err())
    }

    #[test]
    fn expression_beam_new_with_custom_cost_int_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            forward_transitions: vec![Transition {
                name: String::from("t1"),
                cost: CostExpression::Integer(IntegerExpression::Constant(0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                name: String::from("t2"),
                cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let mut custom_cost_dict = HashMap::new();
        custom_cost_dict.insert(
            String::from("t1"),
            CostUnion::Int(IntUnion::Expr(IntExprPy::new(IntegerExpression::Constant(
                2,
            )))),
        );
        custom_cost_dict.insert(
            String::from("t2"),
            CostUnion::Int(IntUnion::Expr(IntExprPy::new(IntegerExpression::Constant(
                3,
            )))),
        );
        let h_evaluator = CostUnion::Int(IntUnion::Expr(IntExprPy::new(
            IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(-1)),
            ),
        )));
        let solver = ExpressionBeamSearchPy::new(
            vec![10],
            &model,
            Some(custom_cost_dict),
            Some(h_evaluator),
            FOperator::Plus,
            false,
            None,
            None,
            false,
            false,
            false,
        );
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(
                    solver.custom_costs,
                    vec![CostExpression::Integer(IntegerExpression::Constant(2))]
                );
                assert_eq!(
                    solver.forced_custom_costs,
                    vec![CostExpression::Integer(IntegerExpression::Constant(3))]
                );
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Integer(IntegerExpression::Constant(
                        1
                    )))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Integer));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn expression_beam_new_with_custom_cost_int_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            forward_transitions: vec![Transition {
                name: String::from("t1"),
                cost: CostExpression::Integer(IntegerExpression::Constant(0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                name: String::from("t2"),
                cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let mut custom_cost_dict = HashMap::new();
        custom_cost_dict.insert(
            String::from("t1"),
            CostUnion::Int(IntUnion::Expr(IntExprPy::new(IntegerExpression::Constant(
                2,
            )))),
        );
        custom_cost_dict.insert(
            String::from("t2"),
            CostUnion::Float(FloatUnion::Expr(FloatExprPy::new(
                ContinuousExpression::Constant(3.0),
            ))),
        );
        let h_evaluator = CostUnion::Int(IntUnion::Expr(IntExprPy::new(
            IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(-1)),
            ),
        )));
        let solver = ExpressionBeamSearchPy::new(
            vec![10],
            &model,
            Some(custom_cost_dict),
            Some(h_evaluator),
            FOperator::Plus,
            false,
            None,
            None,
            false,
            false,
            false,
        );
        assert!(solver.is_err());
    }

    #[test]
    fn expression_beam_new_with_custom_cost_int_forced_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            forward_transitions: vec![Transition {
                name: String::from("t1"),
                cost: CostExpression::Integer(IntegerExpression::Constant(0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                name: String::from("t2"),
                cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let mut custom_cost_dict = HashMap::new();
        custom_cost_dict.insert(
            String::from("t1"),
            CostUnion::Float(FloatUnion::Expr(FloatExprPy::new(
                ContinuousExpression::Constant(2.0),
            ))),
        );
        custom_cost_dict.insert(
            String::from("t2"),
            CostUnion::Int(IntUnion::Expr(IntExprPy::new(IntegerExpression::Constant(
                3,
            )))),
        );
        let h_evaluator = CostUnion::Int(IntUnion::Expr(IntExprPy::new(
            IntegerExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerExpression::Constant(-1)),
            ),
        )));
        let solver = ExpressionBeamSearchPy::new(
            vec![10],
            &model,
            Some(custom_cost_dict),
            Some(h_evaluator),
            FOperator::Plus,
            false,
            None,
            None,
            false,
            false,
            false,
        );
        assert!(solver.is_err());
    }

    #[test]
    fn expression_beam_new_with_custom_cost_int_h_eval_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            forward_transitions: vec![Transition {
                name: String::from("t1"),
                cost: CostExpression::Integer(IntegerExpression::Constant(0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                name: String::from("t2"),
                cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let mut custom_cost_dict = HashMap::new();
        custom_cost_dict.insert(
            String::from("t1"),
            CostUnion::Int(IntUnion::Expr(IntExprPy::new(IntegerExpression::Constant(
                2,
            )))),
        );
        custom_cost_dict.insert(
            String::from("t2"),
            CostUnion::Int(IntUnion::Expr(IntExprPy::new(IntegerExpression::Constant(
                3,
            )))),
        );
        let h_evaluator = CostUnion::Float(FloatUnion::Expr(FloatExprPy::new(
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-1.0)),
            ),
        )));
        let solver = ExpressionBeamSearchPy::new(
            vec![10],
            &model,
            Some(custom_cost_dict),
            Some(h_evaluator),
            FOperator::Plus,
            false,
            None,
            None,
            false,
            false,
            false,
        );
        assert!(solver.is_err());
    }

    #[test]
    fn expression_beam_new_no_custom_cost_float_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            forward_transitions: vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Constant(0.0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Constant(1.0)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let solver = ExpressionBeamSearchPy::new(
            vec![10],
            &model,
            None,
            None,
            FOperator::Plus,
            false,
            None,
            None,
            false,
            true,
            true,
        );
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(
                    solver.custom_costs,
                    vec![CostExpression::Continuous(ContinuousExpression::Constant(
                        0.0
                    ))]
                );
                assert_eq!(
                    solver.forced_custom_costs,
                    vec![CostExpression::Continuous(ContinuousExpression::Constant(
                        1.0
                    ))]
                );
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Continuous(
                        ContinuousExpression::Constant(0.0)
                    ))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Continuous));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn expression_beam_new_with_primal_bound_float_ok() {
        pyo3::prepare_freethreaded_python();

        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            forward_transitions: vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Constant(0.0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Constant(1.0)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let solver = Python::with_gil(|py| {
            let primal_bound = 100.0f64.into_py(py);
            ExpressionBeamSearchPy::new(
                vec![10],
                &model,
                None,
                None,
                FOperator::Plus,
                false,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                true,
                true,
            )
        });
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(
                    solver.custom_costs,
                    vec![CostExpression::Continuous(ContinuousExpression::Constant(
                        0.0
                    ))]
                );
                assert_eq!(
                    solver.forced_custom_costs,
                    vec![CostExpression::Continuous(ContinuousExpression::Constant(
                        1.0
                    ))]
                );
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Continuous(
                        ContinuousExpression::Constant(0.0)
                    ))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Continuous));
                assert!(!solver.maximize);
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(100.0))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn expression_beam_new_with_primal_bound_float_err() {
        pyo3::prepare_freethreaded_python();

        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            forward_transitions: vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Constant(0.0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                cost: CostExpression::Continuous(ContinuousExpression::Constant(1.0)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let solver = Python::with_gil(|py| {
            let primal_bound: Py<PyAny> = (100i32, 100i32).into_py(py);
            ExpressionBeamSearchPy::new(
                vec![10],
                &model,
                None,
                None,
                FOperator::Plus,
                false,
                Some(primal_bound.as_ref(py)),
                None,
                false,
                true,
                true,
            )
        });
        assert!(solver.is_err());
    }

    #[test]
    fn expression_beam_new_with_custom_cost_float_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            forward_transitions: vec![Transition {
                name: String::from("t1"),
                cost: CostExpression::Continuous(ContinuousExpression::Constant(0.0)),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                name: String::from("t2"),
                cost: CostExpression::Continuous(ContinuousExpression::Constant(1.0)),
                ..Default::default()
            }],
            ..Default::default()
        });
        let mut custom_cost_dict = HashMap::new();
        custom_cost_dict.insert(
            String::from("t1"),
            CostUnion::Float(FloatUnion::Expr(FloatExprPy::new(
                ContinuousExpression::Constant(2.0),
            ))),
        );
        custom_cost_dict.insert(
            String::from("t2"),
            CostUnion::Float(FloatUnion::Expr(FloatExprPy::new(
                ContinuousExpression::Constant(3.0),
            ))),
        );
        let h_evaluator = CostUnion::Float(FloatUnion::Expr(FloatExprPy::new(
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-1.0)),
            ),
        )));
        let solver = ExpressionBeamSearchPy::new(
            vec![10],
            &model,
            Some(custom_cost_dict),
            Some(h_evaluator),
            FOperator::Plus,
            false,
            None,
            None,
            false,
            true,
            true,
        );
        assert!(solver.is_ok());
        match solver.unwrap().0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(
                    solver.custom_costs,
                    vec![CostExpression::Continuous(ContinuousExpression::Constant(
                        2.0
                    ))]
                );
                assert_eq!(
                    solver.forced_custom_costs,
                    vec![CostExpression::Continuous(ContinuousExpression::Constant(
                        3.0
                    ))]
                );
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Continuous(
                        ContinuousExpression::Constant(1.0)
                    ))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Continuous));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn expression_beam_search_solve_int_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Int(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn expression_beam_search_solve_float_ok() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Float(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        assert!(solver.solve(&model).is_ok());
    }

    #[test]
    fn expression_beam_search_solver_int_float_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Continuous,
            ..Default::default()
        });
        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Int(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn expression_beam_search_solve_float_int_err() {
        let model = ModelPy::new(Model {
            cost_type: CostType::Integer,
            ..Default::default()
        });
        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Float(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        assert!(solver.solve(&model).is_err());
    }

    #[test]
    fn expression_beam_search_set_primal_bound_int_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Int(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        let result = Python::with_gil(|py| {
            let bound = 1i32.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(solver.custom_costs, vec![]);
                assert_eq!(solver.forced_custom_costs, vec![]);
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Continuous(
                        ContinuousExpression::Constant(1.0)
                    ))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Continuous));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, Some(1));
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn expression_beam_search_set_primal_bound_float_ok() {
        pyo3::prepare_freethreaded_python();

        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Float(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_ok());
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(solver.custom_costs, vec![]);
                assert_eq!(solver.forced_custom_costs, vec![]);
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Continuous(
                        ContinuousExpression::Constant(1.0)
                    ))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Continuous));
                assert!(!solver.maximize);
                assert_eq!(
                    solver.parameters.primal_bound,
                    Some(OrderedContinuous::from(1.5))
                );
                assert_eq!(solver.parameters.time_limit, None);
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn expression_beam_search_set_primal_bound_err() {
        pyo3::prepare_freethreaded_python();

        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Int(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        let result = Python::with_gil(|py| {
            let bound = 1.5f64.into_py(py);
            solver.set_primal_bound(bound.as_ref(py))
        });
        assert!(result.is_err());
    }

    #[test]
    fn expression_beam_search_get_primal_bound_int() {
        let solver = ExpressionBeamSearchPy(WrappedSolver::Int(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(10),
                time_limit: None,
                quiet: false,
            },
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Int(10)));
    }

    #[test]
    fn expression_beam_search_get_primal_bound_float() {
        let solver = ExpressionBeamSearchPy(WrappedSolver::Float(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: Some(OrderedContinuous::from(10.0)),
                time_limit: None,
                quiet: false,
            },
        }));
        assert_eq!(solver.get_primal_bound(), Some(WrappedCost::Float(10.0)));
    }

    #[test]
    fn expression_beam_search_set_time_limit_int() {
        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Int(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(solver.custom_costs, vec![]);
                assert_eq!(solver.forced_custom_costs, vec![]);
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Continuous(
                        ContinuousExpression::Constant(1.0)
                    ))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Continuous));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn expression_beam_search_set_time_limit_float() {
        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Float(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        solver.set_time_limit(10.0);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(solver.custom_costs, vec![]);
                assert_eq!(solver.forced_custom_costs, vec![]);
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Continuous(
                        ContinuousExpression::Constant(1.0)
                    ))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Continuous));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, Some(10.0));
                assert!(!solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn expression_beam_search_get_time_limit_int() {
        let solver = ExpressionBeamSearchPy(WrappedSolver::Int(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn expression_beam_search_get_time_limit_float() {
        let solver = ExpressionBeamSearchPy(WrappedSolver::Float(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: Some(10.0),
                quiet: false,
            },
        }));
        assert_eq!(solver.get_time_limit(), Some(10.0));
    }

    #[test]
    fn expression_beam_search_set_quiet_int() {
        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Int(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Int(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(solver.custom_costs, vec![]);
                assert_eq!(solver.forced_custom_costs, vec![]);
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Continuous(
                        ContinuousExpression::Constant(1.0)
                    ))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Continuous));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Int but is WrappedSolver::Float"),
        }
    }

    #[test]
    fn expression_beam_search_set_quiet_float() {
        let mut solver = ExpressionBeamSearchPy(WrappedSolver::Float(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        solver.set_quiet(true);
        match solver.0 {
            WrappedSolver::Float(solver) => {
                assert_eq!(solver.beam_sizes, vec![10]);
                assert_eq!(solver.custom_costs, vec![]);
                assert_eq!(solver.forced_custom_costs, vec![]);
                assert_eq!(
                    solver.h_evaluator,
                    ExpressionEvaluator::new(CostExpression::Continuous(
                        ContinuousExpression::Constant(1.0)
                    ))
                );
                assert_eq!(solver.f_evaluator_type, FEvaluatorType::Plus);
                assert_eq!(solver.custom_cost_type, Some(CostType::Continuous));
                assert!(!solver.maximize);
                assert_eq!(solver.parameters.primal_bound, None);
                assert_eq!(solver.parameters.time_limit, None);
                assert!(solver.parameters.quiet);
            }
            _ => panic!("Expected WrappedSolver::Float but is WrappedSolver::Int"),
        }
    }

    #[test]
    fn expression_beam_search_get_quiet_int() {
        let solver = ExpressionBeamSearchPy(WrappedSolver::Int(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        assert!(!solver.get_quiet());
    }

    #[test]
    fn expression_beam_search_get_quiet_float() {
        let solver = ExpressionBeamSearchPy(WrappedSolver::Float(ExpressionBeamSearch {
            beam_sizes: vec![10],
            custom_costs: vec![],
            forced_custom_costs: vec![],
            h_evaluator: ExpressionEvaluator::new(CostExpression::Continuous(
                ContinuousExpression::Constant(1.0),
            )),
            custom_cost_type: Some(CostType::Continuous),
            maximize: false,
            f_evaluator_type: FEvaluatorType::Plus,
            parameters: dypdl_heuristic_search::SolverParameters {
                primal_bound: None,
                time_limit: None,
                quiet: false,
            },
        }));
        assert!(!solver.get_quiet());
    }
}
