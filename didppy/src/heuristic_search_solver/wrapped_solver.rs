use crate::{model::StatePy, model::TransitionPy, ModelPy};
use dypdl::prelude::*;
use dypdl::variable_type::OrderedContinuous;
use dypdl_heuristic_search::{Search, Solution};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

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

/// Solution returned by a heuristic search solver.
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
    /// list of Transition : Sequence of transitions corresponding to the solution
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
    /// bool : Whether to exceed the time limit.
    #[pyo3(get)]
    pub time_out: bool,
}

#[pymethods]
impl SolutionPy {
    /// Applies transitions in the solution to the model's target state.
    ///
    /// Parameters
    /// ----------
    /// model: Model
    ///     DyPDL model.
    ///
    /// Returns
    /// -------
    /// State
    ///    The solution state after sequentially applying the transitions.
    ///
    /// Raises
    /// ------
    /// PanicException
    ///     If preconditions are invalid.
    fn state(&self, model: &ModelPy) -> StatePy {
        let mut next = model.target_state();

        for t in self.transitions.iter() {
            next = t.apply(&mut next, model)
        }

        next
    }
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
                .map(TransitionPy::from)
                .collect(),
            expanded: solution.expanded,
            generated: solution.generated,
            time: solution.time,
            time_out: solution.time_out,
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
                .map(TransitionPy::from)
                .collect(),
            expanded: solution.expanded,
            generated: solution.generated,
            time: solution.time,
            time_out: solution.time_out,
        }
    }
}
pub enum WrappedSolver<T, U> {
    Int(T),
    Float(U),
}

impl WrappedSolver<Box<dyn Search<Integer>>, Box<dyn Search<OrderedContinuous>>> {
    pub fn search(&mut self) -> PyResult<SolutionPy> {
        match self {
            WrappedSolver::Int(solver) => match solver.search() {
                Ok(solution) => Ok(SolutionPy::from(solution)),
                Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
            },
            WrappedSolver::Float(solver) => match solver.search() {
                Ok(solution) => Ok(SolutionPy::from(solution)),
                Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
            },
        }
    }

    pub fn search_next(&mut self) -> PyResult<(SolutionPy, bool)> {
        match self {
            WrappedSolver::Int(solver) => match solver.search_next() {
                Ok((solution, terminated)) => Ok((SolutionPy::from(solution), terminated)),
                Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
            },
            WrappedSolver::Float(solver) => match solver.search_next() {
                Ok((solution, terminated)) => Ok((SolutionPy::from(solution), terminated)),
                Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            time_out: false,
        };
        assert_eq!(
            SolutionPy::from(solution),
            SolutionPy {
                cost: Some(WrappedCost::Int(0)),
                best_bound: Some(WrappedCost::Int(0)),
                is_optimal: true,
                is_infeasible: false,
                transitions: vec![TransitionPy::from(Transition::default())],
                expanded: 1,
                generated: 1,
                time: 0.0,
                time_out: false,
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
            time_out: false,
        };
        assert_eq!(
            SolutionPy::from(solution),
            SolutionPy {
                cost: Some(WrappedCost::Float(0.0)),
                best_bound: Some(WrappedCost::Float(0.0)),
                is_optimal: true,
                is_infeasible: false,
                transitions: vec![TransitionPy::from(Transition::default())],
                expanded: 1,
                generated: 1,
                time: 0.0,
                time_out: false,
            }
        );
    }
}
