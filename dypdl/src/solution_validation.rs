use crate::variable_type::Numeric;
use crate::{
    CostType, Model, State, StateFunctionCache, StateInterface, Transition, TransitionId,
    TransitionInterface,
};
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Why a forward solution could not be validated.
///
/// Indices are zero-based. Displayed transition and constraint numbers are
/// one-based; state 0 is the target state.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SolutionValidationError {
    /// The requested Rust cost type does not match the model.
    CostTypeMismatch {
        /// Cost type required by the model.
        expected: CostType,
    },
    /// The transition is not a registered forward transition, even after simplification.
    UnknownTransition {
        /// Zero-based position in the solution's transition sequence.
        transition_index: usize,
        /// Name of the unregistered transition.
        transition_name: String,
    },
    /// A transition ID is backward or does not resolve in the supplied model.
    InvalidTransitionId {
        /// Zero-based position in the solution's transition sequence.
        transition_index: usize,
        /// ID that could not be resolved to a forward transition.
        transition_id: TransitionId,
        /// Reason the ID is invalid.
        message: String,
    },
    /// A transition contains an invalid expression or variable reference.
    InvalidTransition {
        /// Zero-based position in the solution's transition sequence.
        transition_index: usize,
        /// Name of the invalid transition.
        transition_name: String,
        /// Reason the transition is invalid.
        message: String,
    },
    /// A base state was reached with transitions remaining.
    EarlyBaseState {
        /// Number of transitions applied before reaching the base state.
        state_index: usize,
        /// Number of transitions remaining after the base state.
        remaining_transitions: usize,
    },
    /// A transition's precondition or parameter membership requirement failed.
    TransitionNotApplicable {
        /// Zero-based position in the solution's transition sequence.
        transition_index: usize,
        /// Name of the inapplicable transition.
        transition_name: String,
        /// Failed precondition or parameter membership requirement.
        reason: String,
    },
    /// A state constraint failed.
    StateConstraintViolation {
        /// Number of transitions applied before reaching the invalid state.
        state_index: usize,
        /// Zero-based index of the violated state constraint in the model.
        constraint_index: usize,
    },
    /// The last state is not a base state.
    NotBaseState {
        /// Number of transitions applied to reach the final state.
        state_index: usize,
    },
    /// An expression panicked during evaluation, or produced a non-finite cost.
    EvaluationError {
        /// Number of transitions applied to reach the state being evaluated.
        state_index: usize,
        /// Description of the expression being evaluated.
        context: String,
        /// Evaluation failure or panic message.
        message: String,
    },
}

impl fmt::Display for SolutionValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CostTypeMismatch { expected } => write!(f, "The requested cost type does not match the model's {expected:?} cost type."),
            Self::UnknownTransition { transition_index, transition_name } => write!(
                f, "Transition {} (`{transition_name}`) is not a registered transition in this model.",
                transition_index + 1,
            ),
            Self::InvalidTransitionId { transition_index, transition_id, message } => write!(
                f, "Transition {} has an invalid ID (index {}, forced={}, backward={}): {message}.",
                transition_index + 1, transition_id.id, transition_id.forced, transition_id.backward,
            ),
            Self::InvalidTransition { transition_index, transition_name, message } => write!(
                f, "Transition {} (`{transition_name}`) is invalid: {message}", transition_index + 1,
            ),
            Self::EarlyBaseState { state_index, remaining_transitions } => write!(
                f, "State {state_index} satisfies a base case, but {remaining_transitions} transitions remain.",
            ),
            Self::TransitionNotApplicable { transition_index, transition_name, reason } => write!(
                f, "Transition {} (`{transition_name}`) is not applicable in state {transition_index}: {reason}.",
                transition_index + 1,
            ),
            Self::StateConstraintViolation { state_index, constraint_index } => write!(
                f, "State {state_index} violates state constraint {}.", constraint_index + 1,
            ),
            Self::NotBaseState { state_index } => write!(
                f, "The final state (state {state_index}) does not satisfy any base case.",
            ),
            Self::EvaluationError { state_index, context, message } => write!(
                f, "Could not evaluate {context} in state {state_index}: {message}",
            ),
        }
    }
}

impl Error for SolutionValidationError {}

// Keep the legacy validator's panic behavior. The new validator adds context to
// evaluation panics; it does not change the process-wide panic hook.
fn evaluate<T>(
    checked: bool,
    state_index: usize,
    context: impl FnOnce() -> String,
    f: impl FnOnce() -> T,
) -> Result<T, SolutionValidationError> {
    if !checked {
        return Ok(f());
    }
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let message = if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_owned()
        } else {
            "expression evaluation failed".to_owned()
        };
        SolutionValidationError::EvaluationError {
            state_index,
            context: context(),
            message,
        }
    })
}

impl Model {
    /// Validates a forward solution and returns its computed objective value.
    ///
    /// Transitions must belong to this model's forward or forward-forced
    /// transitions. Original transitions passed to `add_forward_transition` are
    /// accepted after the same simplification used when registering them.
    /// For transition IDs, use [`Model::validate_solution_with_ids`].
    /// State constraints are checked at the target and every subsequent state.
    /// The sequence must stop at the first base state, whose cost is evaluated
    /// before folding the transition cost expressions in reverse order.
    ///
    /// Forced-transition priority, transition dominance, dual bounds, and
    /// optimality are not checked. No messages are printed by the validator.
    /// Use `Integer` for integer costs and `OrderedContinuous` for continuous costs.
    ///
    /// # Errors
    ///
    /// Checks membership of every transition, in sequence order, before checking
    /// any states or replaying the solution. An invalid transition therefore takes
    /// precedence over a feasibility failure, even if it occurs later in the sequence.
    /// Returns the first failure within each phase. Expression evaluation panics are
    /// converted to errors with context (the installed panic hook still runs).
    /// Non-finite costs are rejected. Arithmetic otherwise follows DyPDL's
    /// existing expression evaluation semantics.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// let mut model = Model::default();
    /// let x = model.add_integer_variable("x", 1).unwrap();
    /// model.add_base_case(vec![Condition::comparison_i(ComparisonOperator::Eq, x, 0)]).unwrap();
    /// let mut step = Transition::new("step");
    /// step.add_effect(x, x - 1).unwrap();
    /// step.set_cost(IntegerExpression::Cost + 2);
    /// model.add_forward_transition(step.clone()).unwrap();
    /// assert_eq!(model.validate_solution::<Integer>(&[step]).unwrap(), 2);
    /// ```
    pub fn validate_solution<T: Numeric + Ord + 'static>(
        &self,
        transitions: &[Transition],
    ) -> Result<T, SolutionValidationError> {
        self.check_solution_cost_type::<T>()?;
        self.check_transition_membership(transitions)?;
        self.validate_solution_inner(transitions.iter(), true)
    }

    /// Validates a forward solution specified by transition IDs and returns its cost.
    ///
    /// IDs are resolved directly against this model's registered forward or
    /// forward-forced transitions, without structural membership comparisons.
    /// IDs are model-relative: they do not record which model created them.
    /// An ID created by another model refers to the corresponding transition in
    /// this model if its index and flags are valid here.
    ///
    /// After resolving the IDs, uses the same feasibility checks and cost
    /// evaluation as [`Model::validate_solution`], including target constraints
    /// and contextual evaluation errors. No declared cost is compared.
    ///
    /// # Errors
    ///
    /// Returns [`SolutionValidationError::InvalidTransitionId`] for a backward
    /// or out-of-range ID, with its position in the sequence. Otherwise, returns
    /// the same validation errors as [`Model::validate_solution`]. All IDs are
    /// resolved in sequence order before feasibility is checked.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// let mut model = Model::default();
    /// let x = model.add_integer_variable("x", 1).unwrap();
    /// model.add_base_case(vec![Condition::comparison_i(ComparisonOperator::Eq, x, 0)]).unwrap();
    /// let mut step = Transition::new("step");
    /// step.add_effect(x, x - 1).unwrap();
    /// step.set_cost(IntegerExpression::Cost + 2);
    /// let id = model.add_forward_transition(step).unwrap();
    /// assert_eq!(model.validate_solution_with_ids::<Integer>(&[id]).unwrap(), 2);
    /// ```
    pub fn validate_solution_with_ids<T: Numeric + Ord + 'static>(
        &self,
        transition_ids: &[TransitionId],
    ) -> Result<T, SolutionValidationError> {
        self.check_solution_cost_type::<T>()?;
        let transitions = transition_ids
            .iter()
            .enumerate()
            .map(|(transition_index, id)| {
                let invalid_id = |message: &str| SolutionValidationError::InvalidTransitionId {
                    transition_index,
                    transition_id: id.clone(),
                    message: message.to_owned(),
                };
                if id.backward {
                    return Err(invalid_id(
                        "backward IDs cannot be used in a forward solution",
                    ));
                }
                self.get_transition(id)
                    .map_err(|_| invalid_id("no transition with this ID exists in this model"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.validate_solution_inner(transitions.iter().copied(), true)
    }

    fn check_solution_cost_type<T: 'static>(&self) -> Result<(), SolutionValidationError> {
        let expected_type = match self.cost_type {
            CostType::Integer => std::any::TypeId::of::<crate::Integer>(),
            CostType::Continuous => {
                std::any::TypeId::of::<crate::variable_type::OrderedContinuous>()
            }
        };
        if std::any::TypeId::of::<T>() != expected_type {
            return Err(SolutionValidationError::CostTypeMismatch {
                expected: self.cost_type,
            });
        }
        Ok(())
    }

    fn check_transition_membership(
        &self,
        transitions: &[Transition],
    ) -> Result<(), SolutionValidationError> {
        let mut registered: FxHashMap<&str, Vec<&Transition>> = FxHashMap::default();
        for transition in self
            .forward_transitions
            .iter()
            .chain(&self.forward_forced_transitions)
        {
            registered
                .entry(&transition.name)
                .or_default()
                .push(transition);
        }

        for (i, transition) in transitions.iter().enumerate() {
            let candidates = registered.get(transition.name.as_str());
            if candidates.is_some_and(|ts| ts.contains(&transition)) {
                continue;
            }
            let simplified = evaluate(
                true,
                i,
                || format!("transition {} (`{}`)", i + 1, transition.get_full_name()),
                || self.check_and_simplify_transition_inner(transition, false),
            )?
            .map_err(|error| SolutionValidationError::InvalidTransition {
                transition_index: i,
                transition_name: transition.get_full_name(),
                message: error.to_string(),
            })?;
            if !candidates.is_some_and(|ts| ts.contains(&&simplified)) {
                return Err(SolutionValidationError::UnknownTransition {
                    transition_index: i,
                    transition_name: transition.get_full_name(),
                });
            }
        }
        Ok(())
    }

    // Membership checking or ID resolution is the caller's responsibility.
    // The legacy entry point intentionally skips membership checking and uses
    // `checked = false` to retain its target-constraint and evaluation behavior.
    pub(crate) fn validate_solution_inner<'a, T: Numeric + Ord>(
        &self,
        transitions: impl ExactSizeIterator<Item = &'a Transition> + DoubleEndedIterator + Clone,
        checked: bool,
    ) -> Result<T, SolutionValidationError> {
        let mut states = Vec::with_capacity(transitions.len() + 1);
        states.push(self.target.clone());
        let mut cache = StateFunctionCache::new(&self.state_functions);

        if checked {
            self.validate_state_constraints(&states[0], &mut cache, 0, checked)?;
        }

        for (i, transition) in transitions.clone().enumerate() {
            let state = &states[i];
            if evaluate(
                checked,
                i,
                || "base cases".to_owned(),
                || self.is_base(state, &mut cache),
            )? {
                return Err(SolutionValidationError::EarlyBaseState {
                    state_index: i,
                    remaining_transitions: transitions.len() - i,
                });
            }

            let reason = evaluate(
                checked,
                i,
                || {
                    format!(
                        "preconditions of transition {} (`{}`)",
                        i + 1,
                        transition.get_full_name()
                    )
                },
                || self.inapplicable_reason(transition, state, &mut cache),
            )?;
            if let Some(reason) = reason {
                return Err(SolutionValidationError::TransitionNotApplicable {
                    transition_index: i,
                    transition_name: transition.get_full_name(),
                    reason,
                });
            }
            let next_state: State = evaluate(
                checked,
                i,
                || {
                    format!(
                        "effects of transition {} (`{}`)",
                        i + 1,
                        transition.get_full_name()
                    )
                },
                || {
                    transition.apply(
                        state,
                        &mut cache,
                        &self.state_functions,
                        &self.table_registry,
                    )
                },
            )?;
            cache.clear();
            self.validate_state_constraints(&next_state, &mut cache, i + 1, checked)?;
            states.push(next_state);
        }

        let n = transitions.len();
        let mut cost = evaluate(
            checked,
            n,
            || "base cost".to_owned(),
            || self.eval_base_cost::<T, _>(&states[n], &mut cache),
        )?
        .ok_or(SolutionValidationError::NotBaseState { state_index: n })?;
        if checked {
            check_finite_cost(cost, n, "base cost".to_owned())?;
        }
        states.pop();
        for (i, (state, transition)) in states.into_iter().zip(transitions).enumerate().rev() {
            cache.clear();
            let context = || {
                format!(
                    "cost of transition {} (`{}`)",
                    i + 1,
                    transition.get_full_name()
                )
            };
            cost = evaluate(checked, i, context, || {
                transition.eval_cost(
                    cost,
                    &state,
                    &mut cache,
                    &self.state_functions,
                    &self.table_registry,
                )
            })?;
            if checked {
                check_finite_cost(cost, i, context())?;
            }
        }
        Ok(cost)
    }

    fn validate_state_constraints(
        &self,
        state: &State,
        cache: &mut StateFunctionCache,
        state_index: usize,
        checked: bool,
    ) -> Result<(), SolutionValidationError> {
        for (constraint_index, constraint) in self.state_constraints.iter().enumerate() {
            if !evaluate(
                checked,
                state_index,
                || format!("state constraint {}", constraint_index + 1),
                || {
                    constraint.is_satisfied(
                        state,
                        cache,
                        &self.state_functions,
                        &self.table_registry,
                    )
                },
            )? {
                return Err(SolutionValidationError::StateConstraintViolation {
                    state_index,
                    constraint_index,
                });
            }
        }
        Ok(())
    }

    fn inapplicable_reason(
        &self,
        transition: &Transition,
        state: &State,
        cache: &mut StateFunctionCache,
    ) -> Option<String> {
        for (i, element) in &transition.elements_in_set_variable {
            if !state.get_set_variable(*i).contains(*element) {
                let name = self
                    .state_metadata
                    .set_variable_names
                    .get(*i)
                    .map(String::as_str)
                    .unwrap_or("<unnamed>");
                return Some(format!(
                    "parameter value {element} is not in set variable `{name}`"
                ));
            }
        }
        for (i, element) in &transition.elements_in_set_resource_variable {
            if !state.get_set_resource_variable(*i).contains(*element) {
                let name = self
                    .state_metadata
                    .set_resource_variable_names
                    .get(*i)
                    .map(String::as_str)
                    .unwrap_or("<unnamed>");
                return Some(format!(
                    "parameter value {element} is not in set resource variable `{name}`"
                ));
            }
        }
        for (i, condition) in transition.preconditions.iter().enumerate() {
            if !condition.is_satisfied(state, cache, &self.state_functions, &self.table_registry) {
                return Some(format!("precondition {} is false", i + 1));
            }
        }
        None
    }
}

fn check_finite_cost<T: Numeric>(
    cost: T,
    state_index: usize,
    context: String,
) -> Result<(), SolutionValidationError> {
    if cost.to_continuous().is_finite() {
        Ok(())
    } else {
        Err(SolutionValidationError::EvaluationError {
            state_index,
            context,
            message: "the cost is not finite".to_owned(),
        })
    }
}
