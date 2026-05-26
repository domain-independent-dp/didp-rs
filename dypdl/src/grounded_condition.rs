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
    /// Condition.
    pub condition: Condition,
}

impl From<Condition> for GroundedCondition {
    /// Creates a grounded condition from a condition.
    fn from(condition: Condition) -> Self {
        let mut elements_in_set_variable = vec![];
        let condition = Self::check_or(condition, &mut elements_in_set_variable)
            .unwrap_or(Condition::Constant(false));

        Self {
            condition,
            elements_in_set_variable,
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
        self.condition
            .eval(state, function_cache, state_functions, registry)
    }

    fn check_or(
        condition: Condition,
        elements_in_set_variable: &mut Vec<(usize, usize)>,
    ) -> Option<Condition> {
        match condition {
            Condition::Or(a, b) => {
                let a = Self::check_or(*a, elements_in_set_variable);
                let b = Self::check_or(*b, elements_in_set_variable);

                match (a, b) {
                    (Some(a), Some(b)) => Some(Condition::Or(Box::new(a), Box::new(b))),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            }
            condition => {
                if let Some((i, e)) = Self::check_parameter(&condition) {
                    elements_in_set_variable.push((i, e));
                    None
                } else {
                    Some(condition)
                }
            }
        }
    }

    fn check_parameter(condition: &Condition) -> Option<(usize, usize)> {
        if let Condition::Not(condition) = condition {
            if let Condition::Set(condition) = condition.as_ref() {
                if let SetCondition::IsIn(
                    ElementExpression::Constant(e),
                    SetExpression::Reference(ReferenceExpression::Variable(i)),
                ) = condition.as_ref()
                {
                    return Some((*i, *e));
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
