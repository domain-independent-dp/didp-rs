use super::caasdy::FEvaluatorType;
use crate::expression_evaluator::ExpressionEvaluator;
use crate::forward_beam_search;
use crate::solver;
use crate::state_registry::StateInRegistry;
use crate::successor_generator::SuccessorGenerator;
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type;
use dypdl::CostType;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

/// Beam search solver using expressions to compute heuristic values.
///
/// This solver does not have a guarantee for optimality.
#[derive(Debug, PartialEq, Clone)]
pub struct ExpressionBeamSearch<T> {
    /// Cost expressions defining how to compute g-values for each transition.
    pub custom_costs: Vec<dypdl::CostExpression>,
    /// Cost expressions defining how to compute g-values for each forced transition.
    pub forced_custom_costs: Vec<dypdl::CostExpression>,
    /// Evaluator using a used-defined expression to compute h-values.
    pub h_evaluator: ExpressionEvaluator,
    /// How to combine the g-value and h-value to compute the f-value.
    pub f_evaluator_type: FEvaluatorType,
    /// The type of g, h, and f-values.
    pub custom_cost_type: Option<dypdl::CostType>,
    /// Beam size.
    pub beam_sizes: Vec<usize>,
    /// Maximize or not.
    pub maximize: bool,
    /// Common parameters for heuristic search solvers.
    pub parameters: solver::SolverParameters<T>,
}

impl<T> solver::Solver<T> for ExpressionBeamSearch<T>
where
    T: variable_type::Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::with_custom_costs(
            model,
            &self.custom_costs,
            &self.forced_custom_costs,
            false,
        );
        match self.custom_cost_type {
            Some(CostType::Integer) => self.solve_inner::<variable_type::Integer>(model, generator),
            Some(CostType::Continuous) => {
                self.solve_inner::<variable_type::OrderedContinuous>(model, generator)
            }
            None => self.solve_inner::<T>(model, generator),
        }
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
    fn set_time_limit(&mut self, time_limit: u64) {
        self.parameters.time_limit = Some(time_limit)
    }

    #[inline]
    fn get_time_limit(&self) -> Option<u64> {
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

impl<T: variable_type::Numeric + fmt::Display> ExpressionBeamSearch<T> {
    fn solve_inner<U>(
        &self,
        model: &dypdl::Model,
        generator: SuccessorGenerator<TransitionWithCustomCost>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>>
    where
        <U as str::FromStr>::Err: fmt::Debug,
        U: variable_type::Numeric + Ord + fmt::Display,
    {
        let solution = match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator =
                    Box::new(|g: U, h: U, _: &StateInRegistry, _: &dypdl::Model| g + h);
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &generator,
                    &self.h_evaluator,
                    f_evaluator,
                    &self.beam_sizes,
                    self.maximize,
                    self.parameters,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g: U, h: U, _: &StateInRegistry, _: &dypdl::Model| cmp::max(g, h));
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &generator,
                    &self.h_evaluator,
                    f_evaluator,
                    &self.beam_sizes,
                    self.maximize,
                    self.parameters,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g: U, h: U, _: &StateInRegistry, _: &dypdl::Model| cmp::min(g, h));
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &generator,
                    &self.h_evaluator,
                    f_evaluator,
                    &self.beam_sizes,
                    self.maximize,
                    self.parameters,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator = Box::new(|_, h: U, _: &StateInRegistry, _: &dypdl::Model| h);
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &generator,
                    &self.h_evaluator,
                    f_evaluator,
                    &self.beam_sizes,
                    self.maximize,
                    self.parameters,
                )
            }
        };
        Ok(solution)
    }
}
