use didp_parser::variable;
use didp_parser::Transition;
use std::fmt;
use std::rc::Rc;

pub trait MaybeApplicable: fmt::Debug {
    fn is_applicable<T: didp_parser::DPState>(
        &self,
        state: &T,
        registry: &didp_parser::TableRegistry,
    ) -> bool;
}

impl<T: variable::Numeric> MaybeApplicable for Transition<T> {
    fn is_applicable<U: didp_parser::DPState>(
        &self,
        state: &U,
        registry: &didp_parser::TableRegistry,
    ) -> bool {
        Transition::<T>::is_applicable(self, state, registry)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct SuccessorGenerator<'a, T: MaybeApplicable> {
    pub forced_transitions: Vec<Rc<T>>,
    pub transitions: Vec<Rc<T>>,
    pub registry: &'a didp_parser::TableRegistry,
}

impl<'a, T: variable::Numeric> SuccessorGenerator<'a, Transition<T>> {
    pub fn new(
        model: &'a didp_parser::Model<T>,
        backward: bool,
    ) -> SuccessorGenerator<'a, Transition<T>> {
        let forced_transitions: Vec<Rc<Transition<T>>> = if backward {
            model
                .backward_forced_transitions
                .iter()
                .map(|t| Rc::new(t.clone()))
                .collect()
        } else {
            model
                .forward_forced_transitions
                .iter()
                .map(|t| Rc::new(t.clone()))
                .collect()
        };
        let transitions: Vec<Rc<Transition<T>>> = if backward {
            model
                .backward_transitions
                .iter()
                .map(|t| Rc::new(t.clone()))
                .collect()
        } else {
            model
                .forward_transitions
                .iter()
                .map(|t| Rc::new(t.clone()))
                .collect()
        };
        SuccessorGenerator {
            forced_transitions,
            transitions,
            registry: &model.table_registry,
        }
    }
}

impl<'a, T: MaybeApplicable> SuccessorGenerator<'a, T> {
    pub fn generate_applicable_transitions<U: didp_parser::DPState>(
        &self,
        state: &U,
        mut result: Vec<Rc<T>>,
    ) -> Vec<Rc<T>> {
        result.clear();
        for op in self.forced_transitions.iter() {
            if op.is_applicable(state, self.registry) {
                result.push(op.clone());
                return result;
            }
        }
        for op in self.transitions.iter() {
            if op.is_applicable(state, self.registry) {
                result.push(op.clone());
            }
        }
        result
    }

    pub fn applicable_transitions<'b, U: didp_parser::DPState>(
        &'a self,
        state: &'b U,
    ) -> ApplicableTransitions<'a, 'b, T, U> {
        ApplicableTransitions {
            state,
            generator: self,
            iter: self.forced_transitions.iter(),
            forced: true,
            end: false,
        }
    }
}

pub struct ApplicableTransitions<'a, 'b, T: MaybeApplicable, U: didp_parser::DPState> {
    state: &'b U,
    generator: &'a SuccessorGenerator<'a, T>,
    iter: std::slice::Iter<'a, Rc<T>>,
    forced: bool,
    end: bool,
}

impl<'a, 'b, T: MaybeApplicable, U: didp_parser::DPState> Iterator
    for ApplicableTransitions<'a, 'b, T, U>
{
    type Item = Rc<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.end {
            return None;
        }
        match self.iter.next() {
            Some(op) => {
                if op.is_applicable(self.state, self.generator.registry) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use didp_parser::expression::*;
    use didp_parser::variable::Integer;
    use didp_parser::GroundedCondition;
    use rustc_hash::FxHashMap;
    use std::rc::Rc;

    fn generate_model() -> didp_parser::Model<variable::Integer> {
        didp_parser::Model {
            state_metadata: didp_parser::StateMetadata {
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
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Ge,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(1),
                        ))),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Ge,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(2),
                        ))),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Ge,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(3),
                        ))),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            forward_forced_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Ge,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(4),
                        ))),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Ge,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(5),
                        ))),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            backward_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Le,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(1),
                        ))),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Le,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(2),
                        ))),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Le,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(3),
                        ))),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            backward_forced_transitions: vec![
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Le,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(0),
                        ))),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Transition {
                    preconditions: vec![GroundedCondition {
                        condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                            ComparisonOperator::Le,
                            NumericExpression::IntegerVariable(0),
                            NumericExpression::Constant(1),
                        ))),
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
        let model = generate_model();
        let state = didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };

        let generator = SuccessorGenerator::<Transition<Integer>>::new(&model, false);
        let result = Vec::new();
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 2);
        assert_eq!(*result[0], model.forward_transitions[0]);
        assert_eq!(*result[1], model.forward_transitions[1]);

        let generator = SuccessorGenerator::<Transition<Integer>>::new(&model, true);
        let result = Vec::new();
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 2);
        assert_eq!(*result[0], model.backward_transitions[1]);
        assert_eq!(*result[1], model.backward_transitions[2]);

        let state = didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                integer_variables: vec![5],
                ..Default::default()
            },
            ..Default::default()
        };
        let generator = SuccessorGenerator::<Transition<Integer>>::new(&model, false);
        let result = Vec::new();
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 1);
        assert_eq!(*result[0], model.forward_forced_transitions[0]);

        let state = didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let generator = SuccessorGenerator::<Transition<Integer>>::new(&model, true);
        let result = Vec::new();
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 1);
        assert_eq!(*result[0], model.backward_forced_transitions[1]);
    }

    #[test]
    fn applicable_transitions() {
        let model = generate_model();
        let state = didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };

        let generator = SuccessorGenerator::<Transition<Integer>>::new(&model, false);
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

        let generator = SuccessorGenerator::<Transition<Integer>>::new(&model, true);
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

        let state = didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                integer_variables: vec![5],
                ..Default::default()
            },
            ..Default::default()
        };
        let generator = SuccessorGenerator::<Transition<Integer>>::new(&model, false);
        let mut transitions = generator.applicable_transitions(&state);
        assert_eq!(
            transitions.next(),
            Some(Rc::new(model.forward_forced_transitions[0].clone()))
        );
        assert_eq!(transitions.next(), None);

        let state = didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        };
        let generator = SuccessorGenerator::<Transition<Integer>>::new(&model, true);
        let mut transitions = generator.applicable_transitions(&state);
        assert_eq!(
            transitions.next(),
            Some(Rc::new(model.backward_forced_transitions[1].clone()))
        );
        assert_eq!(transitions.next(), None);
    }
}
