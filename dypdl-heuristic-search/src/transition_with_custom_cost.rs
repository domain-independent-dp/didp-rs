use crate::successor_generator::{MaybeApplicable, SuccessorGenerator};
use dypdl::CostExpression;
use std::rc::Rc;

/// Transition with a customized cost expression.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct TransitionWithCustomCost {
    /// Transition.
    pub transition: dypdl::Transition,
    /// Customized cost expression.
    pub custom_cost: CostExpression,
}

impl MaybeApplicable for TransitionWithCustomCost {
    #[inline]
    fn is_applicable<S: dypdl::DPState>(&self, state: &S, registry: &dypdl::TableRegistry) -> bool {
        self.transition.is_applicable(state, registry)
    }
}

impl<'a> SuccessorGenerator<'a, TransitionWithCustomCost> {
    pub fn new(
        model: &'a dypdl::Model,
        backward: bool,
    ) -> SuccessorGenerator<'a, TransitionWithCustomCost> {
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

    /// Returns a successor generator returning applicable transitions with customized cost expressions.
    pub fn with_custom_costs(
        model: &'a dypdl::Model,
        custom_costs: &[CostExpression],
        forced_custom_costs: &[CostExpression],
        backward: bool,
    ) -> SuccessorGenerator<'a, TransitionWithCustomCost> {
        let forced_transitions = if backward {
            &model.backward_forced_transitions
        } else {
            &model.forward_forced_transitions
        };
        let forced_transitions = forced_transitions
            .iter()
            .zip(forced_custom_costs)
            .map(|(t, c)| {
                Rc::new(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: c.simplify(&model.table_registry),
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
            .zip(custom_costs)
            .map(|(t, c)| {
                Rc::new(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: c.simplify(&model.table_registry),
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

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::GroundedCondition;
    use dypdl::Transition;
    use std::rc::Rc;

    fn generate_model() -> dypdl::Model {
        dypdl::Model {
            forward_transitions: vec![Transition {
                name: String::from("forward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(1)),
                )),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                name: String::from("forward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(2)),
                )),
                ..Default::default()
            }],
            backward_transitions: vec![Transition {
                name: String::from("backward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(3)),
                )),
                ..Default::default()
            }],
            backward_forced_transitions: vec![Transition {
                name: String::from("backward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(4)),
                )),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn is_applicable() {
        let transition = TransitionWithCustomCost {
            transition: Transition {
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
            custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(2)),
            )),
        };

        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };
        let registry = dypdl::TableRegistry::default();
        assert!(transition.is_applicable(&state, &registry));
        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
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

        let generator = SuccessorGenerator::<TransitionWithCustomCost>::new(&model, false);
        assert_eq!(
            generator,
            SuccessorGenerator {
                transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("forward"),
                        cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Cost),
                            Box::new(IntegerExpression::Constant(1)),
                        )),
                        ..Default::default()
                    },
                    custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(1)),
                    ))
                })],
                forced_transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("forward_forced"),
                        cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Cost),
                            Box::new(IntegerExpression::Constant(2)),
                        )),
                        ..Default::default()
                    },
                    custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(2)),
                    )),
                })],
                registry: &model.table_registry,
            }
        );
    }

    #[test]
    fn new_backward() {
        let model = generate_model();

        let generator = SuccessorGenerator::<TransitionWithCustomCost>::new(&model, true);
        assert_eq!(
            generator,
            SuccessorGenerator {
                transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("backward"),
                        cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Cost),
                            Box::new(IntegerExpression::Constant(3)),
                        )),
                        ..Default::default()
                    },
                    custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(3)),
                    ))
                })],
                forced_transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("backward_forced"),
                        cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Cost),
                            Box::new(IntegerExpression::Constant(4)),
                        )),
                        ..Default::default()
                    },
                    custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                        BinaryOperator::Add,
                        Box::new(IntegerExpression::Cost),
                        Box::new(IntegerExpression::Constant(4)),
                    )),
                })],
                registry: &model.table_registry,
            }
        );
    }

    #[test]
    fn with_custom_costs() {
        let model = generate_model();

        let custom_costs = vec![CostExpression::Continuous(
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(1.0)),
            ),
        )];
        let forced_custom_costs = [CostExpression::Continuous(
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(2.0)),
                Box::new(ContinuousExpression::Constant(2.0)),
            ),
        )];
        let generator = SuccessorGenerator::<TransitionWithCustomCost>::with_custom_costs(
            &model,
            &custom_costs,
            &forced_custom_costs,
            false,
        );
        assert_eq!(
            generator,
            SuccessorGenerator {
                transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("forward"),
                        cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Cost),
                            Box::new(IntegerExpression::Constant(1)),
                        )),
                        ..Default::default()
                    },
                    custom_cost: CostExpression::Continuous(ContinuousExpression::Constant(2.0))
                })],
                forced_transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("forward_forced"),
                        cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Cost),
                            Box::new(IntegerExpression::Constant(2)),
                        )),
                        ..Default::default()
                    },
                    custom_cost: CostExpression::Continuous(ContinuousExpression::Constant(4.0)),
                })],
                registry: &model.table_registry,
            }
        );
    }

    #[test]
    fn with_custom_costs_backward() {
        let model = generate_model();

        let custom_costs = vec![CostExpression::Continuous(
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(3.0)),
                Box::new(ContinuousExpression::Constant(3.0)),
            ),
        )];
        let forced_custom_costs = [CostExpression::Continuous(
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(4.0)),
                Box::new(ContinuousExpression::Constant(4.0)),
            ),
        )];

        let generator = SuccessorGenerator::<TransitionWithCustomCost>::with_custom_costs(
            &model,
            &custom_costs,
            &forced_custom_costs,
            true,
        );
        assert_eq!(
            generator,
            SuccessorGenerator {
                transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("backward"),
                        cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Cost),
                            Box::new(IntegerExpression::Constant(3)),
                        )),
                        ..Default::default()
                    },
                    custom_cost: CostExpression::Continuous(ContinuousExpression::Constant(6.0))
                })],
                forced_transitions: vec![Rc::new(TransitionWithCustomCost {
                    transition: Transition {
                        name: String::from("backward_forced"),
                        cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Cost),
                            Box::new(IntegerExpression::Constant(4)),
                        )),
                        ..Default::default()
                    },
                    custom_cost: CostExpression::Continuous(ContinuousExpression::Constant(8.0)),
                })],
                registry: &model.table_registry,
            }
        );
    }
}
