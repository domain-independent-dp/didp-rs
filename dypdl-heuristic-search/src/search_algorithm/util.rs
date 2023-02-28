//! A module for utility.

use super::data_structure::SuccessorGenerator;
use dypdl::variable_type;
use std::{cmp, time};

/// Data structure to maintain the elapsed time.
pub struct TimeKeeper {
    time_limit: Option<time::Duration>,
    start: time::Instant,
}

impl Default for TimeKeeper {
    fn default() -> Self {
        TimeKeeper {
            time_limit: None,
            start: time::Instant::now(),
        }
    }
}

impl TimeKeeper {
    /// Returns a time keeper with the given time limit.
    pub fn with_time_limit(time_limit: f64) -> TimeKeeper {
        TimeKeeper {
            time_limit: Some(time::Duration::from_secs_f64(time_limit)),
            start: time::Instant::now(),
        }
    }

    /// Returns the elapsed time.
    pub fn elapsed_time(&self) -> f64 {
        (time::Instant::now() - self.start).as_secs_f64()
    }

    /// Returns the remaining time.
    pub fn remaining_time_limit(&self) -> Option<f64> {
        self.time_limit.map(|time_limit| {
            let elapsed = time::Instant::now() - self.start;

            if elapsed > time_limit {
                0.0
            } else {
                (time_limit - elapsed).as_secs_f64()
            }
        })
    }

    /// Returns whether the time limit is reached.
    pub fn check_time_limit(&self) -> bool {
        self.time_limit.map_or(false, |time_limit| {
            time::Instant::now() - self.start > time_limit
        })
    }
}

/// Common parameters for heuristic search solvers.
#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub struct Parameters<T> {
    /// Primal bound.
    pub primal_bound: Option<T>,
    /// Time limit.
    pub time_limit: Option<f64>,
    /// Returns all feasible solutions found.
    pub get_all_solutions: bool,
    /// Suppress log output or not.
    pub quiet: bool,
}

/// Common parameters for forward search solver.
pub struct ForwardSearchParameters<T: variable_type::Numeric> {
    /// Successor generator.
    pub generator: SuccessorGenerator,
    /// Common parameters.
    pub parameters: Parameters<T>,
    /// Initial registry capacity.
    pub initial_registry_capacity: Option<usize>,
}

/// Parameters for progressive search.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ProgressiveSearchParameters {
    /// The initial width.
    pub init: usize,
    /// The amount of increase.
    pub step: usize,
    /// The maximum value of the width.
    pub bound: Option<usize>,
    // Whether reset the bound when a better solution is found.
    pub reset: bool,
}

impl ProgressiveSearchParameters {
    /// Returns the increased width.
    pub fn increase_width(&self, width: usize) -> usize {
        if let Some(bound) = self.bound {
            cmp::min(width + self.step, bound)
        } else {
            width + self.step
        }
    }
}

impl Default for ProgressiveSearchParameters {
    fn default() -> Self {
        Self {
            init: 1,
            step: 1,
            bound: None,
            reset: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progressive_increase_width() {
        let parameters = ProgressiveSearchParameters {
            init: 1,
            step: 2,
            bound: None,
            reset: false,
        };
        assert_eq!(parameters.increase_width(3), 5);
    }

    #[test]
    fn progressive_increase_width_bonded() {
        let parameters = ProgressiveSearchParameters {
            init: 1,
            step: 2,
            bound: Some(4),
            reset: false,
        };
        assert_eq!(parameters.increase_width(3), 4);
    }
}
