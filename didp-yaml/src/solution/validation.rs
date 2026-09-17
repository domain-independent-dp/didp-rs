use super::{LoadedSolution, SolutionCost};
use dypdl::variable_type::{Integer, OrderedContinuous};
use dypdl::{CostType, Model, SolutionValidationError};
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

/// An error loading or validating a solution.
#[derive(Debug)]
pub enum SolutionError {
    /// A file could not be read.
    Io {
        /// File that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
    /// Invalid YAML or an unresolved transition, with a path into the document.
    Format {
        /// Location of the invalid value in the YAML document.
        path: String,
        /// Reason the value is invalid.
        message: String,
    },
    /// A solution is infeasible or an expression cannot be evaluated.
    Validation(SolutionValidationError),
    /// The declared objective differs from the independently computed objective.
    CostMismatch {
        /// Objective value supplied with the solution.
        declared: SolutionCost,
        /// Objective value computed by replaying the solution.
        computed: SolutionCost,
    },
    /// The declared objective has the wrong type for the model.
    CostTypeMismatch {
        /// Cost type required by the model or computed cost.
        expected: CostType,
    },
    /// A tolerance is negative or non-finite.
    InvalidTolerance,
}

impl fmt::Display for SolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "Could not read `{}`: {source}", path.display()),
            Self::Format { path, message } => write!(f, "{path}: {message}"),
            Self::Validation(error) => error.fmt(f),
            Self::CostMismatch { declared, computed } => write!(
                f,
                "The declared cost {declared} does not match the computed cost {computed}."
            ),
            Self::CostTypeMismatch { expected } => write!(
                f,
                "The declared cost has the wrong type for the model's {expected:?} cost."
            ),
            Self::InvalidTolerance => write!(
                f,
                "rel_tol and abs_tol must be finite, non-negative numbers."
            ),
        }
    }
}

impl Error for SolutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Validation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SolutionValidationError> for SolutionError {
    fn from(error: SolutionValidationError) -> Self {
        Self::Validation(error)
    }
}

/// Tolerances for comparing a declared continuous cost with a computed cost.
///
/// Defaults to exact comparison. Integer costs are always compared exactly.
/// Continuous costs match when
/// `|declared - computed| <= max(abs_tol, rel_tol * max(|declared|, |computed|))`.
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct CostTolerance {
    /// Relative tolerance (finite and non-negative).
    pub rel_tol: f64,
    /// Absolute tolerance (finite and non-negative).
    pub abs_tol: f64,
}

impl CostTolerance {
    /// Checks that both tolerances are finite and non-negative.
    pub fn validate(&self) -> Result<(), SolutionError> {
        if self.rel_tol.is_finite()
            && self.rel_tol >= 0.0
            && self.abs_tol.is_finite()
            && self.abs_tol >= 0.0
        {
            Ok(())
        } else {
            Err(SolutionError::InvalidTolerance)
        }
    }
}

/// Validates a resolved solution and returns its computed cost.
///
/// A missing declared cost is allowed. If present it is checked using
/// `tolerance` for continuous costs and exact equality for integer costs.
/// No claim about optimality or problem infeasibility is verified.
pub fn validate_solution(
    model: &Model,
    solution: &LoadedSolution,
    tolerance: CostTolerance,
) -> Result<SolutionCost, SolutionError> {
    tolerance.validate()?;
    let computed = match model.cost_type {
        CostType::Integer => {
            SolutionCost::Integer(model.validate_solution::<Integer>(&solution.transitions)?)
        }
        CostType::Continuous => SolutionCost::Continuous(
            model
                .validate_solution::<OrderedContinuous>(&solution.transitions)?
                .into_inner(),
        ),
    };
    if let Some(declared) = solution.cost {
        validate_cost(computed, declared, tolerance)?;
    }
    Ok(computed)
}

/// Compares an already computed cost with a declared cost without evaluating a solution.
///
/// Both costs must have the same type. Integer costs are compared exactly;
/// continuous costs must be finite and match within `tolerance`.
pub fn validate_cost(
    computed: SolutionCost,
    declared: SolutionCost,
    tolerance: CostTolerance,
) -> Result<(), SolutionError> {
    tolerance.validate()?;
    let matches = match (declared, computed) {
        (SolutionCost::Integer(a), SolutionCost::Integer(b)) => a == b,
        (SolutionCost::Continuous(a), SolutionCost::Continuous(b)) => {
            a.is_finite()
                && b.is_finite()
                && (a == b
                    || (a - b).abs()
                        <= tolerance
                            .abs_tol
                            .max(tolerance.rel_tol * a.abs().max(b.abs())))
        }
        _ => {
            return Err(SolutionError::CostTypeMismatch {
                expected: match computed {
                    SolutionCost::Integer(_) => CostType::Integer,
                    SolutionCost::Continuous(_) => CostType::Continuous,
                },
            })
        }
    };
    if matches {
        Ok(())
    } else {
        Err(SolutionError::CostMismatch { declared, computed })
    }
}
