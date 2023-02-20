use super::element_expression::ElementExpression;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use crate::state::StateInterface;
use crate::table_registry::TableRegistry;

/// Condition related to sets.
#[derive(Debug, PartialEq, Clone)]
pub enum SetCondition {
    /// Constant.
    Constant(bool),
    /// If a set is equal to another set.
    IsEqual(SetExpression, SetExpression),
    /// If a set is not equal to another set.
    IsNotEqual(SetExpression, SetExpression),
    /// If an element is included in a set.
    IsIn(ElementExpression, SetExpression),
    /// If a set is a subset of another set.
    IsSubset(SetExpression, SetExpression),
    /// If a set is empty.
    IsEmpty(SetExpression),
}

impl SetCondition {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<T: StateInterface>(&self, state: &T, registry: &TableRegistry) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::IsIn(element, SetExpression::Reference(set)) => {
                let f = |i| state.get_set_variable(i);
                let set = set.eval(state, registry, &f, &registry.set_tables);
                set.contains(element.eval(state, registry))
            }
            Self::IsIn(e, s) => s.eval(state, registry).contains(e.eval(state, registry)),
            Self::IsEqual(SetExpression::Reference(x), SetExpression::Reference(y)) => {
                let f = |i| state.get_set_variable(i);
                let x = x.eval(state, registry, &f, &registry.set_tables);
                let y = y.eval(state, registry, &f, &registry.set_tables);
                x == y
            }
            Self::IsEqual(x, SetExpression::Reference(y)) => {
                let f = |i| state.get_set_variable(i);
                let y = y.eval(state, registry, &f, &registry.set_tables);
                x.eval(state, registry) == *y
            }
            Self::IsEqual(SetExpression::Reference(x), y) => {
                let f = |i| state.get_set_variable(i);
                let x = x.eval(state, registry, &f, &registry.set_tables);
                *x == y.eval(state, registry)
            }
            Self::IsEqual(x, y) => x.eval(state, registry) == y.eval(state, registry),
            Self::IsNotEqual(SetExpression::Reference(x), SetExpression::Reference(y)) => {
                let f = |i| state.get_set_variable(i);
                let x = x.eval(state, registry, &f, &registry.set_tables);
                let y = y.eval(state, registry, &f, &registry.set_tables);
                x != y
            }
            Self::IsNotEqual(x, SetExpression::Reference(y)) => {
                let f = |i| state.get_set_variable(i);
                let y = y.eval(state, registry, &f, &registry.set_tables);
                x.eval(state, registry) != *y
            }
            Self::IsNotEqual(SetExpression::Reference(x), y) => {
                let f = |i| state.get_set_variable(i);
                let x = x.eval(state, registry, &f, &registry.set_tables);
                *x != y.eval(state, registry)
            }
            Self::IsNotEqual(x, y) => x.eval(state, registry) != y.eval(state, registry),
            Self::IsSubset(SetExpression::Reference(x), SetExpression::Reference(y)) => {
                let f = |i| state.get_set_variable(i);
                let x = x.eval(state, registry, &f, &registry.set_tables);
                let y = y.eval(state, registry, &f, &registry.set_tables);
                x.is_subset(y)
            }
            Self::IsSubset(x, SetExpression::Reference(y)) => {
                let f = |i| state.get_set_variable(i);
                let y = y.eval(state, registry, &f, &registry.set_tables);
                x.eval(state, registry).is_subset(y)
            }
            Self::IsSubset(SetExpression::Reference(x), y) => {
                let f = |i| state.get_set_variable(i);
                let x = x.eval(state, registry, &f, &registry.set_tables);
                x.is_subset(&y.eval(state, registry))
            }
            Self::IsSubset(x, y) => x.eval(state, registry).is_subset(&y.eval(state, registry)),
            Self::IsEmpty(SetExpression::Reference(set)) => {
                let f = |i| state.get_set_variable(i);
                let set = set.eval(state, registry, &f, &registry.set_tables);
                set.count_ones(..) == 0
            }
            Self::IsEmpty(s) => s.eval(state, registry).count_ones(..) == 0,
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> SetCondition {
        match self {
            Self::IsIn(element, set) => {
                match (element.simplify(registry), set.simplify(registry)) {
                    (
                        ElementExpression::Constant(element),
                        SetExpression::Reference(ReferenceExpression::Constant(set)),
                    ) => Self::Constant(set.contains(element)),
                    (element, set) => Self::IsIn(element, set),
                }
            }
            Self::IsEqual(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(x == y),
                (
                    SetExpression::Reference(ReferenceExpression::Variable(x)),
                    SetExpression::Reference(ReferenceExpression::Variable(y)),
                ) if x == y => Self::Constant(true),
                (x, y) => Self::IsEqual(x, y),
            },
            Self::IsNotEqual(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(x != y),
                (
                    SetExpression::Reference(ReferenceExpression::Variable(x)),
                    SetExpression::Reference(ReferenceExpression::Variable(y)),
                ) if x == y => Self::Constant(false),
                (x, y) => Self::IsNotEqual(x, y),
            },
            Self::IsSubset(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (
                    SetExpression::Reference(ReferenceExpression::Constant(x)),
                    SetExpression::Reference(ReferenceExpression::Constant(y)),
                ) => Self::Constant(x.is_subset(&y)),
                (
                    SetExpression::Reference(ReferenceExpression::Variable(x)),
                    SetExpression::Reference(ReferenceExpression::Variable(y)),
                ) if x == y => Self::Constant(true),
                (x, y) => Self::IsSubset(x, y),
            },
            Self::IsEmpty(x) => match x.simplify(registry) {
                SetExpression::Reference(ReferenceExpression::Constant(x)) => {
                    Self::Constant(x.count_ones(..) == 0)
                }
                x => Self::IsEmpty(x),
            },
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::set_expression::SetOperator;
    use super::*;
    use crate::state::*;
    use crate::variable_type::*;

    fn generate_state() -> State {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        let mut set3 = Set::with_capacity(3);
        set3.insert(0);
        set3.insert(1);
        let set4 = Set::with_capacity(3);
        State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2, set3, set4],
                vector_variables: vec![vec![0, 2], vec![], vec![], vec![]],
                element_variables: vec![1, 2, 3, 4],
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let state = generate_state();
        let registry = TableRegistry::default();

        let expression = SetCondition::Constant(true);
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::Constant(false);
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn is_in_eval() {
        let state = generate_state();
        let registry = TableRegistry::default();

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(2),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(2),
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(0),
            ))),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn is_equal_eval() {
        let state = generate_state();
        let registry = TableRegistry::default();

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn is_not_equal_eval() {
        let state = generate_state();
        let registry = TableRegistry::default();

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn is_subset_eval() {
        let state = generate_state();
        let registry = TableRegistry::default();

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(2)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(3)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(1)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(2)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(3)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(2)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )))),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsSubset(
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(3)),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::Reference(ReferenceExpression::Variable(1)),
            )))),
        );
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn is_empty_eval() {
        let state = generate_state();
        let registry = TableRegistry::default();

        let expression =
            SetCondition::IsEmpty(SetExpression::Reference(ReferenceExpression::Variable(0)));
        assert!(!expression.eval(&state, &registry));

        let expression =
            SetCondition::IsEmpty(SetExpression::Reference(ReferenceExpression::Variable(3)));
        assert!(expression.eval(&state, &registry));

        let expression = SetCondition::IsEmpty(SetExpression::Complement(Box::new(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        )));
        assert!(!expression.eval(&state, &registry));

        let expression = SetCondition::IsEmpty(SetExpression::Complement(Box::new(
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
            ),
        )));
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn constant_simplify() {
        let registry = TableRegistry::default();
        let expression = SetCondition::Constant(true);
        assert_eq!(expression.simplify(&registry), expression);

        let expression = SetCondition::Constant(false);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_in_simplify() {
        let registry = TableRegistry::default();

        let mut set = Set::with_capacity(3);
        set.insert(0);
        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let mut set = Set::with_capacity(3);
        set.insert(0);
        let expression = SetCondition::IsIn(
            ElementExpression::Constant(1),
            SetExpression::Reference(ReferenceExpression::Constant(set)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let expression = SetCondition::IsIn(
            ElementExpression::Constant(0),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_equal_simplify() {
        let registry = TableRegistry::default();

        let mut x = Set::with_capacity(3);
        x.insert(0);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let mut x = Set::with_capacity(3);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let expression = SetCondition::IsEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_not_equal_simplify() {
        let registry = TableRegistry::default();

        let mut x = Set::with_capacity(3);
        x.insert(0);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let mut x = Set::with_capacity(3);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let expression = SetCondition::IsNotEqual(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_subset_simplify() {
        let registry = TableRegistry::default();

        let mut x = Set::with_capacity(3);
        x.insert(0);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let mut x = Set::with_capacity(3);
        x.insert(2);
        let mut y = Set::with_capacity(3);
        y.insert(0);
        y.insert(1);
        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Constant(x)),
            SetExpression::Reference(ReferenceExpression::Constant(y)),
        );
        assert_eq!(
            expression.simplify(&registry),
            SetCondition::Constant(false)
        );

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let expression = SetCondition::IsSubset(
            SetExpression::Reference(ReferenceExpression::Variable(0)),
            SetExpression::Reference(ReferenceExpression::Variable(1)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn is_empty_simplify() {
        let registry = TableRegistry::default();

        let expression = SetCondition::IsEmpty(SetExpression::Reference(
            ReferenceExpression::Constant(Set::with_capacity(3)),
        ));
        assert_eq!(expression.simplify(&registry), SetCondition::Constant(true));

        let expression =
            SetCondition::IsEmpty(SetExpression::Reference(ReferenceExpression::Variable(0)));
        assert_eq!(expression.simplify(&registry), expression);
    }
}
