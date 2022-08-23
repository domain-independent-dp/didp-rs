use crate::evaluator::Evaluator;
use crate::forward_bfs;
use crate::solver;
use crate::state_registry::StateInRegistry;
use crate::successor_generator::SuccessorGenerator;
use dypdl::variable_type;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

/// Evaluator returning the dual bound if defined and zero otherwise.
pub struct NonnegativeLBEvaluator;

impl Evaluator for NonnegativeLBEvaluator {
    fn eval<T: variable_type::Numeric + Ord, S: dypdl::DPState>(
        &self,
        state: &S,
        model: &dypdl::Model,
    ) -> Option<T> {
        Some(model.eval_dual_bound(state).unwrap_or_else(T::zero))
    }
}

/// How to combine the g-value and the h-value to compute the f-value.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum FEvaluatorType {
    /// f = g + h.
    Plus,
    /// f = max(g, h).
    Max,
    /// f = min(g, h).
    Min,
    /// f = h.
    Overwrite,
}

impl Default for FEvaluatorType {
    fn default() -> Self {
        Self::Plus
    }
}

/// Cost-Algebraic A* Solver for DyPDL (CAASDy).
///
/// This performs cost-algebraic A* using the dual bound as the heuristic function.
/// The current implementation only supports cost-algebra with minimization and non-negative edge costs.
/// E.g., the shortest path and minimizing the maximum edge cost on a path.
#[derive(Debug, PartialEq, Clone)]
pub struct CAASDy<T: variable_type::Numeric> {
    /// How to combine the g-value and the h-value.
    pub f_evaluator_type: FEvaluatorType,
    /// Common parameters for heuristic search solvers.
    pub parameters: solver::SolverParameters<T>,
    /// The initial capacity of the data structure storing all generated states.
    pub initial_registry_capacity: Option<usize>,
}

impl<T> solver::Solver<T> for CAASDy<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<dypdl::Transition>::new(model, false);
        let h_evaluator = NonnegativeLBEvaluator {};
        let solution = match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator = Box::new(|g, h, _: &StateInRegistry, _: &dypdl::Model| g + h);
                forward_bfs::forward_bfs(
                    model,
                    generator,
                    &h_evaluator,
                    f_evaluator,
                    self.parameters,
                    self.initial_registry_capacity,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &dypdl::Model| cmp::max(g, h));
                forward_bfs::forward_bfs(
                    model,
                    generator,
                    &h_evaluator,
                    f_evaluator,
                    self.parameters,
                    self.initial_registry_capacity,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &dypdl::Model| cmp::min(g, h));
                forward_bfs::forward_bfs(
                    model,
                    generator,
                    &h_evaluator,
                    f_evaluator,
                    self.parameters,
                    self.initial_registry_capacity,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator = Box::new(|_, h, _: &StateInRegistry, _: &dypdl::Model| h);
                forward_bfs::forward_bfs(
                    model,
                    generator,
                    &h_evaluator,
                    f_evaluator,
                    self.parameters,
                    self.initial_registry_capacity,
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

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::prelude::*;

    #[test]
    fn nonnetagitve_lb_evaluator_eval() {
        let model = Model::default();
        let state = State::default();
        let evaluator = NonnegativeLBEvaluator {};
        assert_eq!(evaluator.eval(&state, &model), Some(0));
        let model = Model {
            dual_bounds: vec![CostExpression::Integer(IntegerExpression::Constant(1))],
            ..Default::default()
        };
        assert_eq!(evaluator.eval(&state, &model), Some(1));
    }
}
