use dypdl::variable_type::Numeric;
use dypdl::{Model, State, StateInterface, TransitionInterface};

/// Result of a rollout.
#[derive(PartialEq, Clone, Debug)]
pub struct RolloutResult<'a, S, U, T>
where
    S: StateInterface + From<State>,
    U: Numeric,
    T: TransitionInterface,
{
    /// State resulting from the rollout.
    /// None if no transition is applied.
    pub state: Option<S>,
    /// Cost.
    pub cost: U,
    /// Transitions applied.
    pub transitions: &'a [T],
    /// If a base case is reached.
    pub is_base: bool,
}

/// Returns the result of a rollout.
///
/// Returns `None` if the rollout fails,
/// e.g., if a transition is not applicable or a state constraint is violated.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::rollout;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_variable("variable", 1).unwrap();
/// model.add_state_constraint(
///     Condition::comparison_i(ComparisonOperator::Ne, variable, 3)
/// ).unwrap();
/// model.add_base_case(
///     vec![Condition::comparison_i(ComparisonOperator::Ge, variable, 4)]
/// ).unwrap();
/// let state = model.target.clone();
///
/// let mut increment = Transition::new("increment");
/// increment.set_cost(IntegerExpression::Cost + 2);
/// increment.add_effect(variable, variable + 1).unwrap();
///
/// let mut double = Transition::new("increment");
/// double.set_cost(IntegerExpression::Cost + 3);
/// double.add_effect(variable, variable * 2).unwrap();
///
/// let transitions = [increment.clone(), double.clone(), increment.clone()];
/// let expected_state: State = increment.apply(&state, &model.table_registry);
/// let expected_state: State = double.apply(&expected_state, &model.table_registry);
/// let result = rollout(&state, 0, &transitions, &model).unwrap();
/// assert_eq!(result.state, Some(expected_state));
/// assert_eq!(result.cost, 5);
/// assert_eq!(result.transitions, &transitions[..2]);
/// assert!(result.is_base);
///
/// let transitions = [double.clone(), increment.clone()];
/// assert_eq!(rollout(&state, 0, &transitions, &model), None);
/// ```
pub fn rollout<'a, S, U, T>(
    state: &S,
    cost: U,
    transitions: &'a [T],
    model: &Model,
) -> Option<RolloutResult<'a, S, U, T>>
where
    S: StateInterface + From<State>,
    U: Numeric,
    T: TransitionInterface,
{
    if model.is_base(state) {
        return Some(RolloutResult {
            state: None,
            cost,
            transitions: &transitions[..0],
            is_base: true,
        });
    }

    let mut current_state;
    let mut parent_state = state;
    let mut cost = cost;

    for (i, t) in transitions.iter().enumerate() {
        if !t.is_applicable(parent_state, &model.table_registry) {
            return None;
        }

        let state = t.apply(parent_state, &model.table_registry);

        if !model.check_constraints(&state) {
            return None;
        }

        cost = t.eval_cost(cost, parent_state, &model.table_registry);

        if model.is_base(&state) {
            return Some(RolloutResult {
                state: Some(state),
                cost,
                transitions: &transitions[..i + 1],
                is_base: true,
            });
        }

        if i == transitions.len() - 1 {
            return Some(RolloutResult {
                state: Some(state),
                cost,
                transitions,
                is_base: false,
            });
        }

        current_state = state;
        parent_state = &current_state;
    }

    Some(RolloutResult {
        state: None,
        cost,
        transitions,
        is_base: false,
    })
}

/// Iterator returning the result of a transition trace.
pub struct Trace<'a, S, U, T> {
    state: S,
    cost: U,
    transitions: &'a [T],
    model: &'a Model,
    i: usize,
}

impl<'a, S, U, T> Iterator for Trace<'a, S, U, T>
where
    S: StateInterface + From<State> + Clone,
    U: Numeric,
    T: TransitionInterface,
{
    type Item = (S, U);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i > self.transitions.len() {
            return None;
        }

        let result = Some((self.state.clone(), self.cost));

        if self.i < self.transitions.len() {
            self.cost = self.transitions[self.i].eval_cost(
                self.cost,
                &self.state,
                &self.model.table_registry,
            );
            self.state = self.transitions[self.i].apply(&self.state, &self.model.table_registry);
        }

        self.i += 1;

        result
    }
}

/// Returns the states and costs of a rollout without checking state constraints and base cases.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::get_trace;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_variable("variable", 1).unwrap();
/// let state = model.target.clone();
///
/// let mut increment = Transition::new("increment");
/// increment.set_cost(IntegerExpression::Cost + 2);
/// increment.add_effect(variable, variable + 1).unwrap();
///
/// let mut double = Transition::new("double");
/// double.set_cost(IntegerExpression::Cost + 3);
/// double.add_effect(variable, variable * 2).unwrap();
///
/// let transitions = [increment.clone(), double.clone()];
/// let mut iter = get_trace(&state, 0, &transitions, &model);
///
/// let expected_state: State = increment.apply(&state, &model.table_registry);
/// assert_eq!(iter.next(), Some((expected_state.clone(), 2)));
///
/// let expected_state: State = double.apply(&expected_state, &model.table_registry);
/// assert_eq!(iter.next(), Some((expected_state, 5)));
///
/// assert_eq!(iter.next(), None);
/// ```
pub fn get_trace<'a, S, U, T>(
    state: &S,
    cost: U,
    transitions: &'a [T],
    model: &'a Model,
) -> Trace<'a, S, U, T>
where
    S: StateInterface + From<State>,
    U: Numeric,
    T: TransitionInterface,
{
    let cost = transitions[0].eval_cost(cost, state, &model.table_registry);
    let state = transitions[0].apply(state, &model.table_registry);

    Trace {
        state,
        cost,
        transitions: &transitions[1..],
        model,
        i: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::{expression::*, SignatureVariables};
    use dypdl::{BaseCase, CostExpression, Effect, GroundedCondition, Transition};

    #[test]
    fn rollout_some_without_transitions_base() {
        let model = Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            }])],
            ..Default::default()
        };
        let state = State::default();
        let transitions = Vec::<Transition>::default();
        let result = rollout(&state, 1, &transitions, &model);
        assert_eq!(
            result,
            Some(RolloutResult {
                state: None,
                cost: 1,
                transitions: &transitions,
                is_base: true,
            })
        );
    }

    #[test]
    fn rollout_some_without_transitions_not_base() {
        let model = Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            }])],
            ..Default::default()
        };
        let state = State::default();
        let transitions = Vec::<Transition>::default();
        let result = rollout(&state, 1, &transitions, &model);
        assert_eq!(
            result,
            Some(RolloutResult {
                state: None,
                cost: 1,
                transitions: &transitions,
                is_base: false,
            })
        );
    }

    #[test]
    fn rollout_some_with_transitions_base() {
        let model = Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            ..Default::default()
        };
        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let transitions = vec![
            Transition {
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                ..Default::default()
            },
            Transition {
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(2))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Constant(4)),
                ..Default::default()
            },
            Transition {
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(3))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Constant(5)),
                ..Default::default()
            },
        ];
        let result = rollout(&state, 1, &transitions, &model);
        assert_eq!(
            result,
            Some(RolloutResult {
                state: Some(State {
                    signature_variables: SignatureVariables {
                        integer_variables: vec![2],
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                cost: 4,
                transitions: &transitions[..2],
                is_base: true
            })
        );
    }

    #[test]
    fn rollout_some_with_transitions_not_base() {
        let model = Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(4)),
                ),
                ..Default::default()
            }])],
            ..Default::default()
        };
        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let transitions = vec![
            Transition {
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                ..Default::default()
            },
            Transition {
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(2))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Constant(4)),
                ..Default::default()
            },
            Transition {
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(3))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Constant(5)),
                ..Default::default()
            },
        ];
        let result = rollout(&state, 1, &transitions, &model);
        assert_eq!(
            result,
            Some(RolloutResult {
                state: Some(State {
                    signature_variables: SignatureVariables {
                        integer_variables: vec![3],
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                cost: 5,
                transitions: &transitions[..],
                is_base: false
            })
        );
    }

    #[test]
    fn rollout_not_applicable() {
        let model = Model::default();
        let state = State::default();
        let transitions = vec![Transition {
            preconditions: vec![GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            }],
            ..Default::default()
        }];
        let result = rollout(&state, 1, &transitions, &model);
        assert_eq!(result, None);
    }

    #[test]
    fn rollout_violates_constraint() {
        let model = Model {
            state_constraints: vec![GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            }],
            ..Default::default()
        };
        let state = State::default();
        let transitions = vec![Transition::default()];
        let result = rollout(&state, 1, &transitions, &model);
        assert_eq!(result, None);
    }

    #[test]
    #[should_panic]
    fn trace_without_transitions() {
        let model = Model::default();
        let state = State::default();
        let transitions = Vec::<Transition>::default();
        get_trace(&state, 1, &transitions, &model);
    }

    #[test]
    fn trace_with_transitions() {
        let model = Model {
            base_cases: vec![BaseCase::new(vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }])],
            ..Default::default()
        };
        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let transitions = vec![
            Transition {
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(1))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                ..Default::default()
            },
            Transition {
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(2))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Constant(4)),
                ..Default::default()
            },
            Transition {
                effect: Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(3))],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::Constant(5)),
                ..Default::default()
            },
        ];
        let mut result = get_trace(&state, 1, &transitions, &model);
        assert_eq!(
            result.next(),
            Some((
                State {
                    signature_variables: SignatureVariables {
                        integer_variables: vec![1],
                        ..Default::default()
                    },
                    ..Default::default()
                },
                3
            ))
        );
        assert_eq!(
            result.next(),
            Some((
                State {
                    signature_variables: SignatureVariables {
                        integer_variables: vec![2],
                        ..Default::default()
                    },
                    ..Default::default()
                },
                4
            ))
        );
        assert_eq!(
            result.next(),
            Some((
                State {
                    signature_variables: SignatureVariables {
                        integer_variables: vec![3],
                        ..Default::default()
                    },
                    ..Default::default()
                },
                5
            ))
        );
        assert_eq!(result.next(), None);
    }
}
