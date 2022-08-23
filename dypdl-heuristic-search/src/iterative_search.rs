use crate::solver;
use dypdl::variable_type;
use std::error::Error;
use std::fmt;

/// Solver that iterates multiple search solvers.
pub struct IterativeSearch<T: variable_type::Numeric> {
    /// Search solvers to use.
    pub solvers: Vec<Box<dyn solver::Solver<T>>>,
    /// Common parameters for heuristic search solvers.
    pub parameters: solver::SolverParameters<T>,
}

impl<T: variable_type::Numeric + fmt::Display> solver::Solver<T> for IterativeSearch<T> {
    fn solve(&mut self, model: &dypdl::Model) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let time_keeper = self.parameters.time_limit.map_or_else(
            solver::TimeKeeper::default,
            solver::TimeKeeper::with_time_limit,
        );
        let mut primal_bound = self.parameters.primal_bound;
        let mut solution = solver::Solution::default();
        for solver in &mut self.solvers {
            if let Some(bound) = primal_bound {
                solver.set_primal_bound(bound);
            }
            if let Some(time_limit) = time_keeper.remaining_time_limit().map(|x| x.as_secs_f64()) {
                if let Some(current_limit) = solver.get_time_limit() {
                    if time_limit < current_limit {
                        solver.set_time_limit(time_limit);
                    }
                } else {
                    solver.set_time_limit(time_limit);
                }
            }
            let result = solver.solve(model)?;
            if let Some(bound) = result.cost {
                println!("New primal bound: {}", bound);
                primal_bound = Some(bound);
                solution = result;
            } else {
                println!("Failed to find a solution");
            }
        }
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
