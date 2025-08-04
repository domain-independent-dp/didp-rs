use super::data_structure::{ParentAndChildStateFunctionCache, StateInformation};
use super::StateInRegistry;
use dypdl::variable_type::Numeric;
use dypdl::{Model, State, StateFunctionCache, StateInterface, TransitionInterface};
use std::fmt::Debug;
use std::hash::Hash;

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
/// `function_cache.parent` is not cleared and updated by `node.state()` and `function_cache.child` is cleared and used while rolling out.
///
/// Returns `None` if the rollout fails,
/// e.g., if a transition is not applicable or a state constraint is violated.
///
/// # Panics
///
/// If expressions in the model or transitions are invalid.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{
///     data_structure::ParentAndChildStateFunctionCache, rollout,
/// };
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
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let expected_state: State = increment.apply(
///     &state, &mut function_cache, &model.state_functions, &model.table_registry,
/// );
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let expected_state: State = double.apply(
///     &expected_state, &mut function_cache, &model.state_functions, &model.table_registry,
/// );
/// let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
/// let result = rollout(
///     &state, &mut function_cache, 0, &transitions, &base_cost_evaluator, &model,
/// ).unwrap();
/// assert_eq!(result.state, Some(expected_state));
/// assert_eq!(result.cost, 5);
/// assert_eq!(result.transitions, &transitions[..2]);
/// assert!(result.is_base);
///
/// let transitions = [double.clone(), increment.clone()];
/// assert_eq!(
///     rollout(&state, &mut function_cache, 0, &transitions, base_cost_evaluator, &model), None
/// );
/// ```
pub fn rollout<'a, S, U, T, B>(
    state: &S,
    function_cache: &mut ParentAndChildStateFunctionCache,
    cost: U,
    transitions: &'a [T],
    mut base_cost_evaluator: B,
    model: &Model,
) -> Option<RolloutResult<'a, S, U, T>>
where
    S: StateInterface + From<State>,
    U: Numeric + Ord,
    T: TransitionInterface,
    B: FnMut(U, U) -> U,
{
    if let Some(base_cost) = model.eval_base_cost(state, &mut function_cache.parent) {
        return Some(RolloutResult {
            state: None,
            cost: base_cost_evaluator(cost, base_cost),
            transitions: &transitions[..0],
            is_base: true,
        });
    }

    if transitions.is_empty() {
        return Some(RolloutResult {
            state: None,
            cost,
            transitions,
            is_base: false,
        });
    }

    let mut current_state;
    let mut parent_state = state;
    let mut cost = cost;
    function_cache.child.clear();

    for (i, t) in transitions.iter().enumerate() {
        if !t.is_applicable(
            parent_state,
            &mut function_cache.child,
            &model.state_functions,
            &model.table_registry,
        ) {
            return None;
        }

        let state = t.apply(
            parent_state,
            &mut function_cache.child,
            &model.state_functions,
            &model.table_registry,
        );

        if !model.check_constraints(&state, &mut function_cache.child) {
            return None;
        }

        cost = t.eval_cost(
            cost,
            parent_state,
            &mut function_cache.child,
            &model.state_functions,
            &model.table_registry,
        );
        function_cache.child.clear();

        if let Some(base_cost) = model.eval_base_cost(&state, &mut function_cache.child) {
            return Some(RolloutResult {
                state: Some(state),
                cost: base_cost_evaluator(cost, base_cost),
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

/// Get the solution cost and suffix if the rollout of the transitions from the node succeeds.
///
/// `function_cache.parent` is not cleared and updated by `node.state()` and `function_cache.child` is cleared and used while rolling out.
///
/// # Panics
///
/// If expressions in the model or transitions are invalid.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::Solution;
/// use dypdl_heuristic_search::search_algorithm::{
///     data_structure::ParentAndChildStateFunctionCache,
///     FNode, StateInRegistry, get_solution_cost_and_suffix,
/// };
/// use dypdl_heuristic_search::search_algorithm::data_structure::{
///     GetTransitions, TransitionWithId,
/// };
/// use dypdl_heuristic_search::search_algorithm::util::update_solution;
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let var = model.add_integer_variable("var", 0).unwrap();
/// model.add_base_case(vec![Condition::comparison_i(ComparisonOperator::Ge, var, 3)]).unwrap();
///
/// let mut transition = Transition::new("transition");
/// transition.add_effect(var, var + 1).unwrap();
/// transition.set_cost(IntegerExpression::Cost + 1);
/// let transition = TransitionWithId {
///     transition,
///     forced: false,
///     id: 0,
/// };
///
/// let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
/// let h_evaluator = |_: &StateInRegistry, _: &mut _| Some(0);
/// let f_evaluator = |g, h, _: &StateInRegistry| g + h;
/// let node = FNode::<_>::generate_root_node(
///     model.target.clone(),
///     &mut function_cache.parent,
///     0,
///     &model,
///     &h_evaluator,
///     &f_evaluator,
///     None,
/// ).unwrap();
/// let node = node.generate_successor_node(
///     Rc::new(transition.clone()),
///     &mut function_cache,
///     &model,
///     &h_evaluator,
///     &f_evaluator,
///     None,
/// ).unwrap();
///
/// let suffix = [transition.clone(), transition.clone()];
/// let base_cost_evaluator = |cost, base_cost| cost + base_cost;
/// function_cache.parent.clear();
/// let (cost, suffix) = get_solution_cost_and_suffix(
///     &model, &node, &suffix, base_cost_evaluator, &mut function_cache,
/// ).unwrap();
///
/// assert_eq!(cost, 3);
/// assert_eq!(suffix, &[transition.clone(), transition]);
/// ```
pub fn get_solution_cost_and_suffix<'a, N, T, U, B, K>(
    model: &Model,
    node: &N,
    transitions: &'a [T],
    base_cost_evaluator: B,
    function_cache: &mut ParentAndChildStateFunctionCache,
) -> Option<(U, &'a [T])>
where
    N: StateInformation<U, K>,
    T: TransitionInterface,
    U: Numeric + Ord,
    B: FnMut(U, U) -> U,
    K: Hash + Eq + Clone + Debug,
    StateInRegistry<K>: StateInterface + From<State>,
{
    let result = rollout(
        node.state(),
        function_cache,
        node.cost(model),
        transitions,
        base_cost_evaluator,
        model,
    )?;

    if result.is_base {
        Some((result.cost, result.transitions))
    } else {
        None
    }
}

/// Iterator returning the result of a transition trace.
pub struct Trace<'a, S, U, T> {
    state: S,
    cost: U,
    transitions: &'a [T],
    model: &'a Model,
    i: usize,
    function_cache: StateFunctionCache,
}

impl<S, U, T> Iterator for Trace<'_, S, U, T>
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
            self.function_cache.clear();

            self.cost = self.transitions[self.i].eval_cost(
                self.cost,
                &self.state,
                &mut self.function_cache,
                &self.model.state_functions,
                &self.model.table_registry,
            );
            self.state = self.transitions[self.i].apply(
                &self.state,
                &mut self.function_cache,
                &self.model.state_functions,
                &self.model.table_registry,
            );
        }

        self.i += 1;

        result
    }
}

/// Returns the states and costs of a rollout without checking state constraints and base cases.
///
/// # Panics
///
/// If `transitions` is empty.
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
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let expected_state: State = increment.apply(
///     &state, &mut function_cache, &model.state_functions, &model.table_registry,
/// );
/// assert_eq!(iter.next(), Some((expected_state.clone(), 2)));
///
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let expected_state: State = double.apply(
///     &expected_state, &mut function_cache, &model.state_functions, &model.table_registry,
/// );
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
    let mut function_cache = StateFunctionCache::new(&model.state_functions);
    let cost = transitions[0].eval_cost(
        cost,
        state,
        &mut function_cache,
        &model.state_functions,
        &model.table_registry,
    );
    let state = transitions[0].apply(
        state,
        &mut function_cache,
        &model.state_functions,
        &model.table_registry,
    );

    Trace {
        state,
        cost,
        transitions: &transitions[1..],
        model,
        i: 0,
        function_cache,
    }
}

#[cfg(test)]
mod tests {
    use super::super::data_structure::{FNode, TransitionWithId};
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;
    use dypdl::{BaseCase, Effect, GroundedCondition};
    use std::rc::Rc;

    #[test]
    fn rollout_some_without_transitions_base() {
        let model = Model {
            base_cases: vec![BaseCase::from(vec![GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            }])],
            ..Default::default()
        };
        let state = State::default();
        let transitions = Vec::<Transition>::default();
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        let mut cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let result = rollout(
            &state,
            &mut cache,
            1,
            &transitions,
            base_cost_evaluator,
            &model,
        );
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
    fn rollout_some_without_transitions_base_cost() {
        let model = Model {
            base_cases: vec![BaseCase::with_cost(
                vec![GroundedCondition {
                    condition: Condition::Constant(true),
                    ..Default::default()
                }],
                1,
            )],
            ..Default::default()
        };
        let state = State::default();
        let transitions = Vec::<Transition>::default();
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        let mut cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let result = rollout(
            &state,
            &mut cache,
            1,
            &transitions,
            base_cost_evaluator,
            &model,
        );
        assert_eq!(
            result,
            Some(RolloutResult {
                state: None,
                cost: 2,
                transitions: &transitions,
                is_base: true,
            })
        );
    }

    #[test]
    fn rollout_some_without_transitions_not_base() {
        let model = Model {
            base_cases: vec![BaseCase::from(vec![GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            }])],
            ..Default::default()
        };
        let state = State::default();
        let transitions = Vec::<Transition>::default();
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        let mut cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let result = rollout(
            &state,
            &mut cache,
            1,
            &transitions,
            base_cost_evaluator,
            &model,
        );
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
            base_cases: vec![BaseCase::from(vec![GroundedCondition {
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
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        let mut cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let result = rollout(
            &state,
            &mut cache,
            1,
            &transitions,
            base_cost_evaluator,
            &model,
        );
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
    fn rollout_some_with_transitions_base_cost() {
        let model = Model {
            base_cases: vec![BaseCase::with_cost(
                vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Ge,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(2)),
                    ),
                    ..Default::default()
                }],
                1,
            )],
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
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        let mut cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let result = rollout(
            &state,
            &mut cache,
            1,
            &transitions,
            base_cost_evaluator,
            &model,
        );
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
                cost: 5,
                transitions: &transitions[..2],
                is_base: true
            })
        );
    }

    #[test]
    fn rollout_some_with_transitions_not_base() {
        let model = Model {
            base_cases: vec![BaseCase::from(vec![GroundedCondition {
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
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        let mut cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let result = rollout(
            &state,
            &mut cache,
            1,
            &transitions,
            base_cost_evaluator,
            &model,
        );
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
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        let mut cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let result = rollout(
            &state,
            &mut cache,
            1,
            &transitions,
            base_cost_evaluator,
            &model,
        );
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
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        let mut cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let result = rollout(
            &state,
            &mut cache,
            1,
            &transitions,
            base_cost_evaluator,
            &model,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn get_solution_cost_and_suffix_some() {
        let mut model = Model::default();
        let var = model.add_integer_variable("var", 0).unwrap();
        let result = model.add_base_case(vec![Condition::comparison_i(
            ComparisonOperator::Ge,
            var,
            3,
        )]);
        assert!(result.is_ok());

        let mut transition = Transition::new("transition");
        transition.add_effect(var, var + 1).unwrap();
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            transition,
            forced: false,
            id: 0,
        };

        let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let h_evaluator = |_: &StateInRegistry, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let node = FNode::<_>::generate_root_node(
            model.target.clone(),
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let node = node.generate_successor_node(
            Rc::new(transition.clone()),
            &mut function_cache,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let suffix = [transition.clone(), transition.clone()];
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        function_cache.parent.clear();
        let result = get_solution_cost_and_suffix(
            &model,
            &node,
            &suffix,
            base_cost_evaluator,
            &mut function_cache,
        );
        assert!(result.is_some());
        let (cost, suffix) = result.unwrap();
        assert_eq!(cost, 3);
        assert_eq!(suffix, &[transition.clone(), transition]);
    }

    #[test]
    fn get_solution_cost_and_suffix_some_with_base_cost() {
        let mut model = Model::default();
        let var = model.add_integer_variable("var", 0).unwrap();
        let result = model.add_base_case_with_cost(
            vec![Condition::comparison_i(ComparisonOperator::Ge, var, 3)],
            1,
        );
        assert!(result.is_ok());

        let mut transition = Transition::new("transition");
        transition.add_effect(var, var + 1).unwrap();
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            transition,
            forced: false,
            id: 0,
        };

        let h_evaluator = |_: &StateInRegistry, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let node = FNode::<_>::generate_root_node(
            model.target.clone(),
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();
        let node = node.generate_successor_node(
            Rc::new(transition.clone()),
            &mut function_cache,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let suffix = [transition.clone(), transition.clone()];
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        function_cache.parent.clear();
        let result = get_solution_cost_and_suffix(
            &model,
            &node,
            &suffix,
            base_cost_evaluator,
            &mut function_cache,
        );
        assert!(result.is_some());
        let (cost, suffix) = result.unwrap();
        assert_eq!(cost, 4);
        assert_eq!(suffix, &[transition.clone(), transition]);
    }

    #[test]
    fn get_solution_cost_and_suffix_none() {
        let mut model = Model::default();
        let var = model.add_integer_variable("var", 0).unwrap();
        let result = model.add_base_case(vec![Condition::comparison_i(
            ComparisonOperator::Ge,
            var,
            3,
        )]);
        assert!(result.is_ok());

        let mut transition = Transition::new("transition");
        transition.add_effect(var, var + 1).unwrap();
        transition.set_cost(IntegerExpression::Cost + 1);

        let h_evaluator = |_: &StateInRegistry, _: &mut _| Some(0);
        let f_evaluator = |g, h, _: &StateInRegistry| g + h;
        let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
        let node = FNode::<_>::generate_root_node(
            model.target.clone(),
            &mut function_cache.parent,
            0,
            &model,
            &h_evaluator,
            &f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = node.unwrap();

        let suffix = [transition.clone(), transition.clone()];
        let base_cost_evaluator = |cost, base_cost| cost + base_cost;
        let result = get_solution_cost_and_suffix(
            &model,
            &node,
            &suffix,
            base_cost_evaluator,
            &mut function_cache,
        );
        assert!(result.is_none());
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
            base_cases: vec![BaseCase::from(vec![GroundedCondition {
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
