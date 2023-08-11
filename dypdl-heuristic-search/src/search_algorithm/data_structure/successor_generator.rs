//! A module for successor generators.

use core::ops::Deref;
use dypdl::{Transition, TransitionInterface};
use std::rc::Rc;

/// Generator of applicable transitions.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::SuccessorGenerator;
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_variable("variable", 0).unwrap();
///
/// let mut increment = Transition::new("increment");
/// increment.add_effect(variable, variable + 1).unwrap();
/// model.add_forward_transition(increment.clone()).unwrap();
///
/// let mut transition = Transition::new("decrement");
/// transition.add_precondition(Condition::comparison_i(ComparisonOperator::Ge, variable, 1));
/// transition.add_effect(variable, variable + 1).unwrap();
/// model.add_forward_transition(transition).unwrap();
///
/// let mut transition = Transition::new("double");
/// transition.add_precondition(Condition::comparison_i(ComparisonOperator::Eq, variable, 1));
/// transition.add_effect(variable, 2 * variable).unwrap();
/// model.add_forward_forced_transition(transition).unwrap();
///
/// let model = Rc::new(model);
/// let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
///
/// let state = model.target.clone();
/// let applicable_transitions = generator.applicable_transitions(&state).collect::<Vec<_>>();
/// let mut iter = generator.applicable_transitions(&state);
/// let transition = iter.next().unwrap();
/// assert_eq!(transition.get_full_name(), "increment");
/// assert_eq!(iter.next(), None);
///
/// let state: State = increment.apply(&state, &model.table_registry);
/// let mut iter = generator.applicable_transitions(&state);
/// let transition = iter.next().unwrap();
/// assert_eq!(transition.get_full_name(), "double");
/// assert_eq!(iter.next(), None);
/// ```
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq, Clone)]
pub struct SuccessorGenerator<T = dypdl::Transition, U = Rc<T>, R = Rc<dypdl::Model>>
where
    T: TransitionInterface,
    U: Deref<Target = T> + Clone,
    R: Deref<Target = dypdl::Model>,
{
    /// Forced transitions.
    pub forced_transitions: Vec<U>,
    /// Transitions.
    pub transitions: Vec<U>,
    /// Backward or not.
    pub backward: bool,
    /// Pointer to the model.
    pub model: R,
}

/// An iterator representing applicable transitions.
pub struct ApplicableTransitions<'a, 'b, T, U, R, S>
where
    T: TransitionInterface,
    U: Deref<Target = T> + Clone,
    R: Deref<Target = dypdl::Model>,
    S: dypdl::StateInterface,
{
    state: &'b S,
    generator: &'a SuccessorGenerator<T, U, R>,
    iter: std::slice::Iter<'a, U>,
    forced: bool,
    end: bool,
}

impl<'a, 'b, T, U, R, S: dypdl::StateInterface> Iterator
    for ApplicableTransitions<'a, 'b, T, U, R, S>
where
    T: TransitionInterface,
    U: Deref<Target = T> + Clone,
    R: Deref<Target = dypdl::Model>,
    S: dypdl::StateInterface,
{
    type Item = U;

    fn next(&mut self) -> Option<Self::Item> {
        if self.end {
            return None;
        }
        match self.iter.next() {
            Some(op) => {
                if op.is_applicable(self.state, &self.generator.model.table_registry) {
                    if self.forced {
                        self.end = true;
                    }
                    Some(op.clone())
                } else {
                    self.next()
                }
            }
            None => {
                if self.forced {
                    self.forced = false;
                    self.iter = self.generator.transitions.iter();
                    self.next()
                } else {
                    None
                }
            }
        }
    }
}

impl<T, U, R> SuccessorGenerator<T, U, R>
where
    T: TransitionInterface,
    U: Deref<Target = T> + Clone,
    R: Deref<Target = dypdl::Model>,
{
    /// Returns a new successor generator
    pub fn new(forced_transitions: Vec<U>, transitions: Vec<U>, backward: bool, model: R) -> Self {
        SuccessorGenerator {
            forced_transitions,
            transitions,
            backward,
            model,
        }
    }

    /// Returns a vector of applicable transitions.
    ///
    /// `result` is used as a buffer to avoid memory allocation.
    pub fn generate_applicable_transitions<S: dypdl::StateInterface>(
        &self,
        state: &S,
        mut result: Vec<U>,
    ) -> Vec<U> {
        result.clear();
        for op in &self.forced_transitions {
            if op.is_applicable(state, &self.model.table_registry) {
                result.push(op.clone());
                return result;
            }
        }
        for op in &self.transitions {
            if op.is_applicable(state, &self.model.table_registry) {
                result.push(op.clone());
            }
        }
        result
    }

    /// Returns applicable transitions as an iterator.
    #[inline]
    pub fn applicable_transitions<'a, 'b, S: dypdl::StateInterface>(
        &'a self,
        state: &'b S,
    ) -> ApplicableTransitions<'a, 'b, T, U, R, S>
    where
        Self: Sized,
    {
        ApplicableTransitions {
            generator: self,
            state,
            iter: self.forced_transitions.iter(),
            forced: true,
            end: false,
        }
    }
}

impl<U, R> SuccessorGenerator<Transition, U, R>
where
    U: Deref<Target = Transition> + Clone + From<Transition>,
    R: Deref<Target = dypdl::Model>,
{
    /// Returns a successor generator given a model and the direction.
    pub fn from_model(model: R, backward: bool) -> Self {
        let forced_transitions = if backward {
            &model.backward_forced_transitions
        } else {
            &model.forward_forced_transitions
        };
        let forced_transitions = forced_transitions.iter().cloned().map(U::from).collect();

        let transitions = if backward {
            &model.backward_transitions
        } else {
            &model.forward_transitions
        };
        let transitions = transitions.iter().cloned().map(U::from).collect();

        SuccessorGenerator::new(forced_transitions, transitions, backward, model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;
    use dypdl::GroundedCondition;
    use rustc_hash::FxHashMap;
    use std::rc::Rc;

    fn generate_model() -> dypdl::Model {
        dypdl::Model {
            state_metadata: dypdl::StateMetadata {
                integer_variable_names: vec![String::from("i0")],
                name_to_integer_variable: {
                    let mut name_to_integer_variable = FxHashMap::default();
                    name_to_integer_variable.insert(String::from("i0"), 0);
                    name_to_integer_variable
                },
                ..Default::default()
            },
            forward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Ge,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Ge,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(2)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Ge,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(3)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Ge,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(4)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Ge,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(5)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            backward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Le,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Le,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(2)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Le,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(3)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            backward_forced_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Le,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(0)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::ComparisonI(
                            ComparisonOperator::Le,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn generate_applicable_transitions() {
        let model = Rc::new(generate_model());
        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };

        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
        let result = Vec::new();
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 2);
        assert_eq!(*result[0], model.forward_transitions[0]);
        assert_eq!(*result[1], model.forward_transitions[1]);

        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), true);
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 2);
        assert_eq!(*result[0], model.backward_transitions[1]);
        assert_eq!(*result[1], model.backward_transitions[2]);

        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
                integer_variables: vec![5],
                ..Default::default()
            },
            ..Default::default()
        };
        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 1);
        assert_eq!(*result[0], model.forward_forced_transitions[0]);

        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), true);
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 1);
        assert_eq!(*result[0], model.backward_forced_transitions[1]);
    }

    #[test]
    fn applicable_transitions() {
        let model = Rc::new(generate_model());
        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };

        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
        let mut transitions = generator.applicable_transitions(&state);
        assert_eq!(
            transitions.next(),
            Some(Rc::new(model.forward_transitions[0].clone()))
        );
        assert_eq!(
            transitions.next(),
            Some(Rc::new(model.forward_transitions[1].clone()))
        );
        assert_eq!(transitions.next(), None);

        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), true);
        let mut transitions = generator.applicable_transitions(&state);
        assert_eq!(
            transitions.next(),
            Some(Rc::new(model.backward_transitions[1].clone()))
        );
        assert_eq!(
            transitions.next(),
            Some(Rc::new(model.backward_transitions[2].clone()))
        );
        assert_eq!(transitions.next(), None);

        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
                integer_variables: vec![5],
                ..Default::default()
            },
            ..Default::default()
        };
        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);
        let mut transitions = generator.applicable_transitions(&state);
        assert_eq!(
            transitions.next(),
            Some(Rc::new(model.forward_forced_transitions[0].clone()))
        );
        assert_eq!(transitions.next(), None);

        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), true);
        let mut transitions = generator.applicable_transitions(&state);
        assert_eq!(
            transitions.next(),
            Some(Rc::new(model.backward_forced_transitions[1].clone()))
        );
        assert_eq!(transitions.next(), None);
    }

    #[test]
    fn from_model_forward() {
        let mut model = Model::default();
        let mut transition1 = Transition::new("transition1");
        transition1.set_cost(IntegerExpression::Cost + 1);
        let result = model.add_forward_transition(transition1.clone());
        assert!(result.is_ok());
        let mut transition2 = Transition::new("transition2");
        transition2.set_cost(IntegerExpression::Cost + 2);
        let result = model.add_forward_transition(transition2.clone());
        assert!(result.is_ok());
        let mut transition3 = Transition::new("transition3");
        transition3.set_cost(IntegerExpression::Cost + 3);
        let result = model.add_forward_forced_transition(transition3.clone());
        assert!(result.is_ok());
        let mut transition4 = Transition::new("transition4");
        transition4.set_cost(IntegerExpression::Cost + 4);
        let result = model.add_forward_forced_transition(transition4.clone());
        assert!(result.is_ok());
        let mut transition5 = Transition::new("transition5");
        transition5.set_cost(IntegerExpression::Cost + 5);
        let result = model.add_backward_transition(transition5.clone());
        assert!(result.is_ok());
        let mut transition6 = Transition::new("transition6");
        transition6.set_cost(IntegerExpression::Cost + 6);
        let result = model.add_backward_forced_transition(transition6.clone());
        assert!(result.is_ok());
        let model = Rc::new(model);

        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), false);

        assert_eq!(generator.model, model);
        assert_eq!(
            generator.transitions,
            vec![Rc::new(transition1,), Rc::new(transition2,),]
        );
        assert_eq!(
            generator.forced_transitions,
            vec![Rc::new(transition3,), Rc::new(transition4,),]
        );
    }

    #[test]
    fn from_model_backward() {
        let mut model = Model::default();
        let mut transition1 = Transition::new("transition1");
        transition1.set_cost(IntegerExpression::Cost + 1);
        let result = model.add_backward_transition(transition1.clone());
        assert!(result.is_ok());
        let mut transition2 = Transition::new("transition2");
        transition2.set_cost(IntegerExpression::Cost + 2);
        let result = model.add_backward_transition(transition2.clone());
        assert!(result.is_ok());
        let mut transition3 = Transition::new("transition3");
        transition3.set_cost(IntegerExpression::Cost + 3);
        let result = model.add_backward_forced_transition(transition3.clone());
        assert!(result.is_ok());
        let mut transition4 = Transition::new("transition4");
        transition4.set_cost(IntegerExpression::Cost + 4);
        let result = model.add_backward_forced_transition(transition4.clone());
        assert!(result.is_ok());
        let mut transition5 = Transition::new("transition5");
        transition5.set_cost(IntegerExpression::Cost + 5);
        let result = model.add_forward_transition(transition5.clone());
        assert!(result.is_ok());
        let mut transition6 = Transition::new("transition6");
        transition6.set_cost(IntegerExpression::Cost + 6);
        let result = model.add_forward_forced_transition(transition6.clone());
        assert!(result.is_ok());
        let model = Rc::new(model);

        let generator = SuccessorGenerator::<Transition>::from_model(model.clone(), true);

        assert_eq!(generator.model, model);
        assert_eq!(
            generator.transitions,
            vec![Rc::new(transition1,), Rc::new(transition2,),]
        );
        assert_eq!(
            generator.forced_transitions,
            vec![Rc::new(transition3,), Rc::new(transition4,),]
        );
    }
}
