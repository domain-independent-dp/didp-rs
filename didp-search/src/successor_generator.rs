use didp_parser::variable;
use didp_parser::Transition;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
pub struct SuccessorGenerator<'a, T: variable::Numeric> {
    transitions: Vec<Rc<Transition<T>>>,
    registry: &'a didp_parser::TableRegistry,
}

impl<'a, T: variable::Numeric> SuccessorGenerator<'a, T> {
    pub fn new(model: &'a didp_parser::Model<T>, backward: bool) -> SuccessorGenerator<'a, T> {
        let transitions = if backward {
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
            transitions,
            registry: &model.table_registry,
        }
    }

    pub fn generate_applicable_transitions(
        &self,
        state: &didp_parser::State,
        mut result: Vec<Rc<Transition<T>>>,
    ) -> Vec<Rc<Transition<T>>> {
        result.clear();
        for op in self.transitions.iter() {
            if op.is_applicable(state, self.registry) {
                result.push(op.clone());
            }
        }
        result
    }

    pub fn applicable_transitions<'b>(
        &'a self,
        state: &'b didp_parser::State,
    ) -> ApplicableTransitions<'a, 'b, T> {
        ApplicableTransitions {
            state,
            generator: self,
            iter: self.transitions.iter(),
        }
    }
}

pub struct ApplicableTransitions<'a, 'b, T: variable::Numeric> {
    state: &'b didp_parser::State,
    generator: &'a SuccessorGenerator<'a, T>,
    iter: std::slice::Iter<'a, Rc<Transition<T>>>,
}

impl<'a, 'b, T: variable::Numeric> Iterator for ApplicableTransitions<'a, 'b, T> {
    type Item = Rc<Transition<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(op) => {
                if op.is_applicable(self.state, self.generator.registry) {
                    Some(op.clone())
                } else {
                    self.next()
                }
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use didp_parser::expression::*;
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
            ..Default::default()
        }
    }

    #[test]
    fn generate_applicable_transitions() {
        let model = generate_model();
        let state = didp_parser::State {
            signature_variables: Rc::new(didp_parser::SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            }),
            ..Default::default()
        };

        let generator = SuccessorGenerator::new(&model, false);

        let result = Vec::new();
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 2);
        assert_eq!(*result[0], model.forward_transitions[0]);
        assert_eq!(*result[1], model.forward_transitions[1]);

        let generator = SuccessorGenerator::new(&model, true);
        let result = Vec::new();
        let result = generator.generate_applicable_transitions(&state, result);
        assert_eq!(result.len(), 2);
        assert_eq!(*result[0], model.backward_transitions[1]);
        assert_eq!(*result[1], model.backward_transitions[2]);
    }

    #[test]
    fn applicable_transitions() {
        let model = generate_model();
        let state = didp_parser::State {
            signature_variables: Rc::new(didp_parser::SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            }),
            ..Default::default()
        };
        let generator = SuccessorGenerator::new(&model, false);
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
        let generator = SuccessorGenerator::new(&model, true);
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
    }
}
