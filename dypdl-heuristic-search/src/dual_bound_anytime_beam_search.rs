use crate::beam::NormalBeam;
use crate::beam_search;
use crate::caasdy::{FEvaluatorType, NonnegativeLBEvaluator};
use crate::solver;
use crate::state_registry::StateInRegistry;
use crate::successor_generator::SuccessorGenerator;
use crate::{anytime_beam_search, TransitionWithCustomCost};
use dypdl::variable_type;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

/// Anytime beam search solver.
///
/// This performs beam search while doubling the beam size starting from 1.
pub struct DualBoundAnytimeBeamSearch<T: variable_type::Numeric> {
    /// How to combine the g-value and the h-value.
    pub f_evaluator_type: FEvaluatorType,
    // Whether to perform pruning.
    pub pruning: bool,
    /// Callback function used when a new solution is found.
    pub callback: Box<solver::Callback<T>>,
    /// Common parameters for heuristic search solvers.
    pub parameters: solver::SolverParameters<T>,
}

impl<T> solver::Solver<T> for DualBoundAnytimeBeamSearch<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let beam_constructor = NormalBeam::new;
        let generator = SuccessorGenerator::<TransitionWithCustomCost>::new(model, false);
        let h_evaluator = NonnegativeLBEvaluator {};
        let solution = match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator = Box::new(|g, h, _: &StateInRegistry, _: &dypdl::Model| g + h);
                let evaluators = beam_search::EvaluatorsForBeamSearch {
                    h_evaluator,
                    f_evaluator,
                };
                anytime_beam_search::anytime_beam_search(
                    model,
                    &generator,
                    &evaluators,
                    &beam_constructor,
                    self.pruning,
                    &mut self.callback,
                    self.parameters,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &dypdl::Model| cmp::max(g, h));
                let evaluators = beam_search::EvaluatorsForBeamSearch {
                    h_evaluator,
                    f_evaluator,
                };
                anytime_beam_search::anytime_beam_search(
                    model,
                    &generator,
                    &evaluators,
                    &beam_constructor,
                    self.pruning,
                    &mut self.callback,
                    self.parameters,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateInRegistry, _: &dypdl::Model| cmp::min(g, h));
                let evaluators = beam_search::EvaluatorsForBeamSearch {
                    h_evaluator,
                    f_evaluator,
                };
                anytime_beam_search::anytime_beam_search(
                    model,
                    &generator,
                    &evaluators,
                    &beam_constructor,
                    self.pruning,
                    &mut self.callback,
                    self.parameters,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator = Box::new(|_, h, _: &StateInRegistry, _: &dypdl::Model| h);
                let evaluators = beam_search::EvaluatorsForBeamSearch {
                    h_evaluator,
                    f_evaluator,
                };
                anytime_beam_search::anytime_beam_search(
                    model,
                    &generator,
                    &evaluators,
                    &beam_constructor,
                    self.pruning,
                    &mut self.callback,
                    self.parameters,
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
