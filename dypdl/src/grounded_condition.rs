use crate::expression::*;
use crate::state::StateInterface;
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
        if let Condition::Or(a, b) = condition {
            match (
                Self::check_parameter(a.as_ref()),
                Self::check_parameter(b.as_ref()),
            ) {
                (Some((i, e, Some(capacity))), None) => GroundedCondition {
                    condition: *b,
                    elements_in_vector_variable: vec![(i, e, capacity)],
                    ..Default::default()
                },
                (Some((i, e, None)), None) => GroundedCondition {
                    condition: *b,
                    elements_in_set_variable: vec![(i, e)],
                    ..Default::default()
                },
                (None, Some((e, i, Some(capacity)))) => GroundedCondition {
                    condition: *a,
                    elements_in_vector_variable: vec![(i, e, capacity)],
                    ..Default::default()
                },
                (None, Some((e, i, None))) => GroundedCondition {
                    condition: *a,
                    elements_in_set_variable: vec![(i, e)],
                    ..Default::default()
                },
                (Some((e, i, capacity1)), Some((e2, i2, capacity2))) => {
                    let mut elements_in_set_variable = vec![];
                    let mut elements_in_vector_variable = vec![];

                    if let Some(capacity) = capacity1 {
                        elements_in_vector_variable.push((i, e, capacity));
                    } else {
                        elements_in_set_variable.push((i, e));
                    }
                    if let Some(capacity) = capacity2 {
                        elements_in_vector_variable.push((i2, e2, capacity));
                    } else {
                        elements_in_set_variable.push((i2, e2));
                    }

                    GroundedCondition {
                        condition: Condition::Constant(false),
                        elements_in_set_variable,
                        elements_in_vector_variable,
                    }
                }
                _ => GroundedCondition {
                    condition: Condition::Or(a, b),
                    ..Default::default()
                },
            }
        } else {
            GroundedCondition {
                condition,
                ..Default::default()
            }
        }
    }
}

impl From<GroundedCondition> for Condition {
    /// Creates a condition from a grounded condition.
    fn from(grounded_condition: GroundedCondition) -> Self {
        let mut condition = grounded_condition.condition;

        for (i, e) in grounded_condition.elements_in_set_variable {
            condition = Condition::Or(
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsIn(
                        ElementExpression::Constant(e),
                        SetExpression::Reference(ReferenceExpression::Variable(i)),
                    ),
                ))))),
                Box::new(condition),
            );
        }

        for (i, e, capacity) in grounded_condition.elements_in_vector_variable {
            condition = Condition::Or(
                Box::new(Condition::Not(Box::new(Condition::Set(Box::new(
                    SetCondition::IsIn(
                        ElementExpression::Constant(e),
                        SetExpression::FromVector(
                            capacity,
                            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                                i,
                            ))),
                        ),
                    ),
                ))))),
                Box::new(condition),
            );
        }

        condition
    }
}

impl GroundedCondition {
    /// Returns true if the condition is satisfied, false if the condition is not satisfied, and None if an parameter is not included in the corresponding set or vector variable.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn is_satisfied<U: StateInterface>(
        &self,
        state: &U,
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
        self.condition.eval(state, registry)
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
    fn from_test() {
        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            ..Default::default()
        };
        assert_eq!(
            condition,
            GroundedCondition::from(Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))))
        );
    }

    #[test]
    fn is_satisfied_test() {
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

        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            ..Default::default()
        };
        assert!(condition.is_satisfied(&state, &registry));

        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![(0, 0, 2)],
        };
        assert!(condition.is_satisfied(&state, &registry));

        let condition = GroundedCondition {
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(1),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![(0, 1, 2)],
        };
        assert!(!condition.is_satisfied(&state, &registry));
    }
}
