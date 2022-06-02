use crate::solver::ConfigErr;
use crate::successor_generator::{MaybeApplicable, SuccessorGenerator};
use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable::Numeric;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct TransitionWithCustomCost<T: Numeric, U: Numeric> {
    pub transition: didp_parser::Transition<T>,
    pub custom_cost: didp_parser::expression::NumericExpression<U>,
}

impl<T: Numeric, U: Numeric> MaybeApplicable for TransitionWithCustomCost<T, U> {
    fn is_applicable<S: didp_parser::DPState>(
        &self,
        state: &S,
        registry: &didp_parser::TableRegistry,
    ) -> bool {
        self.transition.is_applicable(state, registry)
    }
}

impl<'a, T: Numeric> SuccessorGenerator<'a, TransitionWithCustomCost<T, T>> {
    pub fn new(
        model: &'a didp_parser::Model<T>,
        backward: bool,
    ) -> SuccessorGenerator<'a, TransitionWithCustomCost<T, T>> {
        let forced_transitions = if backward {
            &model.backward_forced_transitions
        } else {
            &model.forward_forced_transitions
        };
        let forced_transitions = forced_transitions
            .iter()
            .map(|t| {
                Rc::new(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: t.cost.clone(),
                })
            })
            .collect();
        let transitions = if backward {
            &model.backward_transitions
        } else {
            &model.forward_transitions
        };
        let transitions = transitions
            .iter()
            .map(|t| {
                Rc::new(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: t.cost.clone(),
                })
            })
            .collect();
        SuccessorGenerator {
            forced_transitions,
            transitions,
            registry: &model.table_registry,
        }
    }
}

impl<'a, T: Numeric, U: Numeric + ParseNumericExpression>
    SuccessorGenerator<'a, TransitionWithCustomCost<T, U>>
where
    <U as str::FromStr>::Err: fmt::Debug,
{
    pub fn with_expressions(
        model: &'a didp_parser::Model<T>,
        backward: bool,
        cost_expressions: &FxHashMap<String, String>,
    ) -> Result<SuccessorGenerator<'a, TransitionWithCustomCost<T, U>>, Box<dyn Error>> {
        let original_forced_transitions = if backward {
            &model.backward_forced_transitions
        } else {
            &model.forward_forced_transitions
        };
        let mut forced_transitions = Vec::with_capacity(original_forced_transitions.len());
        let mut parameters = FxHashMap::default();
        for t in original_forced_transitions {
            for (name, value) in t.parameter_names.iter().zip(t.parameter_values.iter()) {
                parameters.insert(name.clone(), *value);
            }
            let custom_cost = if let Some(expression) = cost_expressions.get(&t.name) {
                U::parse_expression(
                    expression.clone(),
                    &model.state_metadata,
                    &model.table_registry,
                    &parameters,
                )?
            } else {
                return Err(
                    ConfigErr::new(format!("expression for `{}` is undefined", t.name)).into(),
                );
            };
            forced_transitions.push(Rc::new(TransitionWithCustomCost {
                transition: t.clone(),
                custom_cost,
            }));
            parameters.clear();
        }
        let original_transitions = if backward {
            &model.backward_transitions
        } else {
            &model.forward_transitions
        };
        let mut transitions = Vec::with_capacity(original_transitions.len());
        for t in original_transitions {
            for (name, value) in t.parameter_names.iter().zip(t.parameter_values.iter()) {
                parameters.insert(name.clone(), *value);
            }
            let custom_cost = if let Some(expression) = cost_expressions.get(&t.name) {
                U::parse_expression(
                    expression.clone(),
                    &model.state_metadata,
                    &model.table_registry,
                    &parameters,
                )?
            } else {
                return Err(
                    ConfigErr::new(format!("expression for `{}` is undefined", t.name)).into(),
                );
            };
            transitions.push(Rc::new(TransitionWithCustomCost {
                transition: t.clone(),
                custom_cost,
            }));
            parameters.clear();
        }
        Ok(SuccessorGenerator {
            forced_transitions,
            transitions,
            registry: &model.table_registry,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use didp_parser::expression::*;
    use didp_parser::variable::Integer;
    use didp_parser::GroundedCondition;
    use didp_parser::Transition;
    use std::rc::Rc;

    fn generate_model() -> didp_parser::Model<Integer> {
        didp_parser::Model {
            forward_transitions: vec![Transition {
                name: String::from("forward"),
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(1)),
                ),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                name: String::from("forward_forced"),
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(2)),
                ),
                ..Default::default()
            }],
            backward_transitions: vec![Transition {
                name: String::from("backward"),
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(3)),
                ),
                ..Default::default()
            }],
            backward_forced_transitions: vec![Transition {
                name: String::from("backward_forced"),
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::Constant(4)),
                ),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn is_applicable() {
        let transition = TransitionWithCustomCost::<Integer, Integer> {
            transition: Transition {
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
            custom_cost: NumericExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(NumericExpression::Cost),
                Box::new(NumericExpression::Constant(2)),
            ),
        };

        let state = didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };
        let registry = didp_parser::TableRegistry::default();
        assert!(transition.is_applicable(&state, &registry));
        let state = didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                integer_variables: vec![3],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!transition.is_applicable(&state, &registry));
    }

    #[test]
    fn new() {
        let model = generate_model();

        let generator =
            SuccessorGenerator::<TransitionWithCustomCost<Integer, Integer>>::new(&model, false);
        assert_eq!(
            generator,
            SuccessorGenerator {
                transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("forward"),
                        cost: NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::Cost),
                            Box::new(NumericExpression::Constant(1)),
                        ),
                        ..Default::default()
                    },
                    custom_cost: NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(1)),
                    )
                })],
                forced_transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("forward_forced"),
                        cost: NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::Cost),
                            Box::new(NumericExpression::Constant(2)),
                        ),
                        ..Default::default()
                    },
                    custom_cost: NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(2)),
                    ),
                })],
                registry: &model.table_registry,
            }
        );

        let generator =
            SuccessorGenerator::<TransitionWithCustomCost<Integer, Integer>>::new(&model, true);
        assert_eq!(
            generator,
            SuccessorGenerator {
                transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("backward"),
                        cost: NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::Cost),
                            Box::new(NumericExpression::Constant(3)),
                        ),
                        ..Default::default()
                    },
                    custom_cost: NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(3)),
                    )
                })],
                forced_transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("backward_forced"),
                        cost: NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::Cost),
                            Box::new(NumericExpression::Constant(4)),
                        ),
                        ..Default::default()
                    },
                    custom_cost: NumericExpression::NumericOperation(
                        NumericOperator::Add,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(4)),
                    ),
                })],
                registry: &model.table_registry,
            }
        );
    }

    #[test]
    fn with_expressions_ok() {
        let model = generate_model();

        let mut cost_expressions = FxHashMap::default();
        cost_expressions.insert(String::from("forward"), String::from("(* cost 1)"));
        cost_expressions.insert(String::from("forward_forced"), String::from("(* cost 2)"));
        cost_expressions.insert(String::from("backward"), String::from("(* cost 3)"));
        cost_expressions.insert(String::from("backward_forced"), String::from("(* cost 4)"));

        let generator =
            SuccessorGenerator::<TransitionWithCustomCost<Integer, Integer>>::with_expressions(
                &model,
                false,
                &cost_expressions,
            );
        assert!(generator.is_ok());
        assert_eq!(
            generator.unwrap(),
            SuccessorGenerator {
                transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("forward"),
                        cost: NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::Cost),
                            Box::new(NumericExpression::Constant(1)),
                        ),
                        ..Default::default()
                    },
                    custom_cost: NumericExpression::NumericOperation(
                        NumericOperator::Multiply,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(1)),
                    )
                })],
                forced_transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("forward_forced"),
                        cost: NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::Cost),
                            Box::new(NumericExpression::Constant(2)),
                        ),
                        ..Default::default()
                    },
                    custom_cost: NumericExpression::NumericOperation(
                        NumericOperator::Multiply,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(2)),
                    ),
                })],
                registry: &model.table_registry,
            }
        );

        let generator =
            SuccessorGenerator::<TransitionWithCustomCost<Integer, Integer>>::with_expressions(
                &model,
                true,
                &cost_expressions,
            );
        assert!(generator.is_ok());
        assert_eq!(
            generator.unwrap(),
            SuccessorGenerator {
                transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("backward"),
                        cost: NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::Cost),
                            Box::new(NumericExpression::Constant(3)),
                        ),
                        ..Default::default()
                    },
                    custom_cost: NumericExpression::NumericOperation(
                        NumericOperator::Multiply,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(3)),
                    )
                })],
                forced_transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("backward_forced"),
                        cost: NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::Cost),
                            Box::new(NumericExpression::Constant(4)),
                        ),
                        ..Default::default()
                    },
                    custom_cost: NumericExpression::NumericOperation(
                        NumericOperator::Multiply,
                        Box::new(NumericExpression::Cost),
                        Box::new(NumericExpression::Constant(4)),
                    ),
                })],
                registry: &model.table_registry,
            }
        );
    }

    #[test]
    fn with_expressions_err() {
        let model = generate_model();

        let mut cost_expressions = FxHashMap::default();
        cost_expressions.insert(String::from("forward"), String::from("(* cost 1)"));
        cost_expressions.insert(String::from("backward"), String::from("(* cost 3)"));
        cost_expressions.insert(String::from("backward_forced"), String::from("(* cost 4)"));

        let generator =
            SuccessorGenerator::<TransitionWithCustomCost<Integer, Integer>>::with_expressions(
                &model,
                false,
                &cost_expressions,
            );
        assert!(generator.is_err());

        cost_expressions.insert(String::from("forward_forced"), String::from("(^ cost 2)"));
        let generator =
            SuccessorGenerator::<TransitionWithCustomCost<Integer, Integer>>::with_expressions(
                &model,
                false,
                &cost_expressions,
            );
        assert!(generator.is_err());
    }
}
