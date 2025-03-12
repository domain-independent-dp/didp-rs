use crate::expression::*;
use crate::state::StateInterface;
use crate::state_functions::{StateFunctionCache, StateFunctions};
use crate::table_registry;

/// Condition with element parameters.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct GroundedCondition {
    /// Pairs of an index of a set variable and a parameter.
    /// The condition is evaluated only when all parameters are included in the set variables.
    /// Otherwise, the condition is evaluated to true.
    pub elements_in_set_variable: Vec<(usize, usize)>,
    /// Pairs of an index of a vector variable, a parameter, and the capacity.
    /// The condition is evaluated only when all parameters are included in the vector variables.
    /// Otherwise, the condition is evaluated to true.
    pub elements_in_vector_variable: Vec<(usize, usize, usize)>,
    /// Condition.
    pub condition: Condition,
}

impl From<Condition> for GroundedCondition {
    /// Creates a grounded condition from a condition.
    fn from(condition: Condition) -> Self {
        let mut elements_in_set_variable = vec![];
        let mut elements_in_vector_variable = vec![];
        let condition = Self::check_or(
            condition,
            &mut elements_in_set_variable,
            &mut elements_in_vector_variable,
        )
        .unwrap_or(Condition::Constant(false));

        Self {
            condition,
            elements_in_set_variable,
            elements_in_vector_variable,
        }
    }
}

impl From<GroundedCondition> for Condition {
    /// Creates a condition from a grounded condition.
    fn from(grounded_condition: GroundedCondition) -> Self {
        let mut condition = match grounded_condition.condition {
            Condition::Constant(true) => return Condition::Constant(true),
            Condition::Constant(false) => None,
            condition => Some(condition),
        };

        for (i, e, capacity) in grounded_condition
            .elements_in_vector_variable
            .into_iter()
            .rev()
        {
            condition = if let Some(condition) = condition {
                Some(Condition::Or(
                    Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                        SetCondition::IsIn(
                            ElementExpression::Constant(e),
                            SetExpression::FromVector(
                                capacity,
                                Box::new(VectorExpression::Reference(
                                    ReferenceExpression::Variable(i),
                                )),
                            ),
                        ),
                    ))))),
                    Box::new(condition),
                ))
            } else {
                Some(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsIn(
                        ElementExpression::Constant(e),
                        SetExpression::FromVector(
                            capacity,
                            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                                i,
                            ))),
                        ),
                    ),
                )))))
            };
        }

        for (i, e) in grounded_condition
            .elements_in_set_variable
            .into_iter()
            .rev()
        {
            condition = if let Some(condition) = condition {
                Some(Condition::Or(
                    Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                        SetCondition::IsIn(
                            ElementExpression::Constant(e),
                            SetExpression::Reference(ReferenceExpression::Variable(i)),
                        ),
                    ))))),
                    Box::new(condition),
                ))
            } else {
                Some(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsIn(
                        ElementExpression::Constant(e),
                        SetExpression::Reference(ReferenceExpression::Variable(i)),
                    ),
                )))))
            }
        }

        condition.unwrap_or(Condition::Constant(false))
    }
}

impl GroundedCondition {
    /// Returns true if the condition is satisfied and false if the condition is not satisfied.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn is_satisfied<U: StateInterface>(
        &self,
        state: &U,
        function_cache: &mut StateFunctionCache,
        state_functions: &StateFunctions,
        registry: &table_registry::TableRegistry,
    ) -> bool {
        for (i, v) in &self.elements_in_set_variable {
            if !state.get_set_variable(*i).contains(*v) {
                return true;
            }
        }
        for (i, v, _) in &self.elements_in_vector_variable {
            if !state.get_vector_variable(*i).contains(v) {
                return true;
            }
        }
        self.condition
            .eval(state, function_cache, state_functions, registry)
    }

    fn check_or(
        condition: Condition,
        elements_in_set_variable: &mut Vec<(usize, usize)>,
        elements_vector_variable: &mut Vec<(usize, usize, usize)>,
    ) -> Option<Condition> {
        match condition {
            Condition::Or(a, b) => {
                let a = Self::check_or(*a, elements_in_set_variable, elements_vector_variable);
                let b = Self::check_or(*b, elements_in_set_variable, elements_vector_variable);

                match (a, b) {
                    (Some(a), Some(b)) => Some(Condition::Or(Box::new(a), Box::new(b))),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            }
            condition => {
                if let Some((i, e, capacity)) = Self::check_parameter(&condition) {
                    if let Some(capacity) = capacity {
                        elements_vector_variable.push((i, e, capacity));
                    } else {
                        elements_in_set_variable.push((i, e));
                    }

                    None
                } else {
                    Some(condition)
                }
            }
        }
    }

    fn check_parameter(condition: &Condition) -> Option<(usize, usize, Option<usize>)> {
        if let Condition::Not(condition) = condition {
            if let Condition::Set(condition) = condition.as_ref() {
                match condition.as_ref() {
                    SetCondition::IsIn(
                        ElementExpression::Constant(e),
                        SetExpression::Reference(ReferenceExpression::Variable(i)),
                    ) => return Some((*i, *e, None)),
                    SetCondition::IsIn(
                        ElementExpression::Constant(e),
                        SetExpression::FromVector(capacity, v),
                    ) => {
                        if let VectorExpression::Reference(ReferenceExpression::Variable(i)) =
                            v.as_ref()
                        {
                            return Some((*i, *e, Some(*capacity)));
                        }
                    }
                    _ => {}
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state;
    use crate::table;
    use crate::table_data;
    use crate::variable_type;
    use rustc_hash::FxHashMap;

    fn generate_registry() -> table_registry::TableRegistry {
        let tables_1d = vec![table::Table1D::new(vec![true, false])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("b1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![false, true],
            vec![true, false],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("b2"), 0);

        table_registry::TableRegistry {
            bool_tables: table_data::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn from_condition_to_grounded() {
        let condition = GroundedCondition::from(Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::Variable(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        ))));
        let expected = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            ..Default::default()
        };
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_condition_to_grounded_with_set_parameters_a() {
        let condition = GroundedCondition::from(Condition::Or(
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                ),
            ))))),
            Box::new(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        ));
        let expected = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_set_variable: vec![(1, 2)],
            ..Default::default()
        };
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_condition_to_grounded_with_set_parameters_b() {
        let condition = GroundedCondition::from(Condition::Or(
            Box::new(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                ),
            ))))),
        ));
        let expected = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_set_variable: vec![(1, 2)],
            ..Default::default()
        };
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_condition_to_grounded_with_set_parameters_ab() {
        let condition = GroundedCondition::from(Condition::Or(
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(1),
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                ),
            ))))),
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                ),
            ))))),
        ));
        let expected = GroundedCondition {
            condition: Condition::Constant(false),
            elements_in_set_variable: vec![(0, 1), (1, 2)],
            ..Default::default()
        };
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_condition_to_grounded_with_vector_parameters_a() {
        let condition = GroundedCondition::from(Condition::Or(
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::FromVector(
                        3,
                        Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                            1,
                        ))),
                    ),
                ),
            ))))),
            Box::new(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        ));
        let expected = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_vector_variable: vec![(1, 2, 3)],
            ..Default::default()
        };
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_condition_to_grounded_with_vector_parameters_b() {
        let condition = GroundedCondition::from(Condition::Or(
            Box::new(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::FromVector(
                        3,
                        Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                            1,
                        ))),
                    ),
                ),
            ))))),
        ));
        let expected = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_vector_variable: vec![(1, 2, 3)],
            ..Default::default()
        };
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_condition_to_grounded_with_vector_parameters_ab() {
        let condition = GroundedCondition::from(Condition::Or(
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(1),
                    SetExpression::FromVector(
                        2,
                        Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                            0,
                        ))),
                    ),
                ),
            ))))),
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::FromVector(
                        3,
                        Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                            1,
                        ))),
                    ),
                ),
            ))))),
        ));
        let expected = GroundedCondition {
            condition: Condition::Constant(false),
            elements_in_vector_variable: vec![(0, 1, 2), (1, 2, 3)],
            ..Default::default()
        };
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_condition_to_grounded_with_set_vector_parameters() {
        let condition = GroundedCondition::from(Condition::Or(
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(1),
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                ),
            ))))),
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::FromVector(
                        3,
                        Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                            1,
                        ))),
                    ),
                ),
            ))))),
        ));
        let expected = GroundedCondition {
            condition: Condition::Constant(false),
            elements_in_set_variable: vec![(0, 1)],
            elements_in_vector_variable: vec![(1, 2, 3)],
        };
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_condition_to_grounded_with_vector_set_parameters() {
        let condition = GroundedCondition::from(Condition::Or(
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(1),
                    SetExpression::FromVector(
                        2,
                        Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                            0,
                        ))),
                    ),
                ),
            ))))),
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(2),
                    SetExpression::Reference(ReferenceExpression::Variable(1)),
                ),
            ))))),
        ));
        let expected = GroundedCondition {
            condition: Condition::Constant(false),
            elements_in_set_variable: vec![(1, 2)],
            elements_in_vector_variable: vec![(0, 1, 2)],
        };
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_grounded_to_condition_true() {
        let condition = Condition::from(GroundedCondition {
            condition: Condition::Constant(true),
            elements_in_set_variable: vec![(0, 1), (3, 4)],
            elements_in_vector_variable: vec![(1, 2, 3), (4, 5, 6)],
        });
        let expected = Condition::Constant(true);
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_grounded_to_condition_false_single() {
        let condition = Condition::from(GroundedCondition {
            condition: Condition::Constant(false),
            elements_in_set_variable: vec![(0, 1)],
            ..Default::default()
        });
        let expected = Condition::Not(Box::new(Condition::Set(Box::new(SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        )))));
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_grounded_to_condition_false_multi() {
        let condition = Condition::from(GroundedCondition {
            condition: Condition::Constant(false),
            elements_in_set_variable: vec![(0, 1), (3, 4)],
            elements_in_vector_variable: vec![(1, 2, 3), (4, 5, 6)],
        });
        let expected = Condition::Or(
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(1),
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                ),
            ))))),
            Box::new(Condition::Or(
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsIn(
                        ElementExpression::Constant(4),
                        SetExpression::Reference(ReferenceExpression::Variable(3)),
                    ),
                ))))),
                Box::new(Condition::Or(
                    Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                        SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::FromVector(
                                3,
                                Box::new(VectorExpression::Reference(
                                    ReferenceExpression::Variable(1),
                                )),
                            ),
                        ),
                    ))))),
                    Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                        SetCondition::IsIn(
                            ElementExpression::Constant(5),
                            SetExpression::FromVector(
                                6,
                                Box::new(VectorExpression::Reference(
                                    ReferenceExpression::Variable(4),
                                )),
                            ),
                        ),
                    ))))),
                )),
            )),
        );
        assert_eq!(condition, expected);
    }

    #[test]
    fn from_grounded_to_condition() {
        let condition = Condition::from(GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_set_variable: vec![(0, 1), (3, 4)],
            elements_in_vector_variable: vec![(1, 2, 3), (4, 5, 6)],
        });
        let expected = Condition::Or(
            Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                SetCondition::IsIn(
                    ElementExpression::Constant(1),
                    SetExpression::Reference(ReferenceExpression::Variable(0)),
                ),
            ))))),
            Box::new(Condition::Or(
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsIn(
                        ElementExpression::Constant(4),
                        SetExpression::Reference(ReferenceExpression::Variable(3)),
                    ),
                ))))),
                Box::new(Condition::Or(
                    Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                        SetCondition::IsIn(
                            ElementExpression::Constant(2),
                            SetExpression::FromVector(
                                3,
                                Box::new(VectorExpression::Reference(
                                    ReferenceExpression::Variable(1),
                                )),
                            ),
                        ),
                    ))))),
                    Box::new(Condition::Or(
                        Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                            SetCondition::IsIn(
                                ElementExpression::Constant(5),
                                SetExpression::FromVector(
                                    6,
                                    Box::new(VectorExpression::Reference(
                                        ReferenceExpression::Variable(4),
                                    )),
                                ),
                            ),
                        ))))),
                        Box::new(Condition::Set(Box::new(SetCondition::IsIn(
                            ElementExpression::Variable(0),
                            SetExpression::Reference(ReferenceExpression::Variable(0)),
                        )))),
                    )),
                )),
            )),
        );
        assert_eq!(condition, expected);
    }

    #[test]
    fn is_satisfied_condition() {
        let registry = generate_registry();
        let mut s0 = variable_type::Set::with_capacity(2);
        s0.insert(0);
        let state = state::State {
            signature_variables: state::SignatureVariables {
                set_variables: vec![s0],
                vector_variables: vec![vec![1]],
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);

        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            ..Default::default()
        };
        assert!(condition.is_satisfied(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_satisfied_set_parameter() {
        let registry = generate_registry();
        let mut s0 = variable_type::Set::with_capacity(2);
        s0.insert(0);
        let state = state::State {
            signature_variables: state::SignatureVariables {
                set_variables: vec![s0],
                vector_variables: vec![vec![1]],
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);

        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![(0, 0, 2)],
        };
        assert!(condition.is_satisfied(&state, &mut function_cache, &state_functions, &registry));
    }

    #[test]
    fn is_satisfied_vector_parameter() {
        let registry = generate_registry();
        let mut s0 = variable_type::Set::with_capacity(2);
        s0.insert(0);
        let state = state::State {
            signature_variables: state::SignatureVariables {
                set_variables: vec![s0],
                vector_variables: vec![vec![1]],
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);

        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(1),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![(0, 1, 2)],
        };
        assert!(!condition.is_satisfied(&state, &mut function_cache, &state_functions, &registry));
    }
}
