use crate::caasdy::{FEvaluatorType, NonnegativeLBEvaluator};
use crate::chokudai_search;
use crate::solver;
use crate::state_registry::StateInRegistry;
use crate::successor_generator::SuccessorGenerator;
use crate::util::ForwardSearchParameters;
use dypdl::variable_type;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

/// Chokudai search solver using the dual bound.
///
/// This performs cost-algebraic A* using the dual bound as the heuristic function.
/// The current implementation only supports cost-algebra with minimization and non-negative edge costs.
/// E.g., the shortest path and minimizing the maximum edge cost on a path.
pub struct DualBoundChokudaiSearch<T: variable_type::Numeric> {
    /// How to combine the g-value and the h-value.
    pub f_evaluator_type: FEvaluatorType,
    /// Common parameters for heuristic search solvers.
    pub width: usize,
    /// Chokudai width
    pub parameters: solver::SolverParameters<T>,
    /// The initial capacity of the data structure storing all generated states.
    pub initial_registry_capacity: Option<usize>,
    /// Callback function used when a new solution is found.
    pub callback: Box<solver::Callback<T>>,
}

impl<T> solver::Solver<T> for DualBoundChokudaiSearch<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<dypdl::Transition>::new(model, false);
        let parameters = ForwardSearchParameters {
            generator,
            parameters: self.parameters,
            initial_registry_capacity: self.initial_registry_capacity,
        };
        let h_evaluator = NonnegativeLBEvaluator {};
        let solution = match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator = Box::new(|g, h, _: &StateInRegistry, _: &dypdl::Model| g + h);
                chokudai_search::chokudai_search(
                    model,
                    &h_evaluator,
                    f_evaluator,
                    self.width,
                    &mut self.callback,
                    parameters,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &dypdl::Model| cmp::max(g, h));
                chokudai_search::chokudai_search(
                    model,
                    &h_evaluator,
                    f_evaluator,
                    self.width,
                    &mut self.callback,
                    parameters,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &dypdl::Model| cmp::min(g, h));
                chokudai_search::chokudai_search(
                    model,
                    &h_evaluator,
                    f_evaluator,
                    self.width,
                    &mut self.callback,
                    parameters,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator = Box::new(|_, h, _: &StateInRegistry, _: &dypdl::Model| h);
                chokudai_search::chokudai_search(
                    model,
                    &h_evaluator,
                    f_evaluator,
                    self.width,
                    &mut self.callback,
                    parameters,
                )
            }
        };
        Ok(solution)
    }

    #[inline]
    fn set_primal_bound(&mut self, primal_bound: T) {
        self.parameters.primal_bound = Some(primal_bound)
    }

    #[inline]
    fn get_primal_bound(&self) -> Option<T> {
        self.parameters.primal_bound
    }

    #[inline]
    fn set_time_limit(&mut self, time_limit: f64) {
        self.parameters.time_limit = Some(time_limit)
    }

    #[inline]
    fn get_time_limit(&self) -> Option<f64> {
        self.parameters.time_limit
    }

    #[inline]
    fn set_quiet(&mut self, quiet: bool) {
        self.parameters.quiet = quiet
    }

    #[inline]
    fn get_quiet(&self) -> bool {
        self.parameters.quiet
    }
}