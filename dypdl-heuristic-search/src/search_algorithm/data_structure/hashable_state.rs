use super::state_registry::StateInRegistry;
use dypdl::variable_type::{Continuous, Element, Integer, OrderedContinuous, Set, Vector};
use dypdl::ResourceVariables;
use ordered_float::OrderedFloat;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;

/// Signature variables that can be hashed.
///
/// However, using continuous variables is not recommended as it may cause a numerical issue.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Default)]
pub struct HashableSignatureVariables {
    /// Set variables.
    pub set_variables: Vec<Set>,
    /// Vector variables.
    pub vector_variables: Vec<Vector>,
    /// Element variables.
    pub element_variables: Vec<Element>,
    /// Integer numeric variables.
    pub integer_variables: Vec<Integer>,
    /// Continuous numeric variables.
    pub continuous_variables: Vec<OrderedContinuous>,
}

impl From<dypdl::SignatureVariables> for HashableSignatureVariables {
    fn from(variables: dypdl::SignatureVariables) -> HashableSignatureVariables {
        HashableSignatureVariables {
            set_variables: variables.set_variables,
            vector_variables: variables.vector_variables,
            element_variables: variables.element_variables,
            integer_variables: variables.integer_variables,
            continuous_variables: variables
                .continuous_variables
                .into_iter()
                .map(OrderedFloat)
                .collect(),
        }
    }
}

/// Signature variables that can be hashed.
///
/// However, using continuous variables is not recommended as it may cause a numerical issue.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Default)]
pub struct HashableResourceVariables {
    /// Element variables.
    pub element_variables: Vec<Element>,
    /// Integer numeric variables.
    pub integer_variables: Vec<Integer>,
    /// Continuous variables.
    pub continuous_variables: Vec<OrderedContinuous>,
}

impl From<dypdl::ResourceVariables> for HashableResourceVariables {
    fn from(variables: dypdl::ResourceVariables) -> HashableResourceVariables {
        HashableResourceVariables {
            element_variables: variables.element_variables,
            integer_variables: variables.integer_variables,
            continuous_variables: variables
                .continuous_variables
                .into_iter()
                .map(OrderedFloat)
                .collect(),
        }
    }
}

/// State that can be hashed.
///
/// However, using continuous variables is not recommended as it may cause a numerical issue.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Default)]
pub struct HashableState {
    signature_variables: HashableSignatureVariables,
    resource_variables: HashableResourceVariables,
}

impl From<dypdl::State> for HashableState {
    fn from(state: dypdl::State) -> HashableState {
        HashableState {
            signature_variables: HashableSignatureVariables::from(state.signature_variables),
            resource_variables: HashableResourceVariables::from(state.resource_variables),
        }
    }
}

impl dypdl::StateInterface for HashableState {
    #[inline]
    fn get_number_of_set_variables(&self) -> usize {
        self.signature_variables.set_variables.len()
    }

    #[inline]
    fn get_set_variable(&self, i: usize) -> &Set {
        &self.signature_variables.set_variables[i]
    }

    #[inline]
    fn get_number_of_vector_variables(&self) -> usize {
        self.signature_variables.vector_variables.len()
    }

    #[inline]
    fn get_vector_variable(&self, i: usize) -> &Vector {
        &self.signature_variables.vector_variables[i]
    }

    #[inline]
    fn get_number_of_element_variables(&self) -> usize {
        self.signature_variables.element_variables.len()
    }

    #[inline]
    fn get_element_variable(&self, i: usize) -> Element {
        self.signature_variables.element_variables[i]
    }

    #[inline]
    fn get_number_of_integer_variables(&self) -> usize {
        self.signature_variables.integer_variables.len()
    }

    #[inline]
    fn get_integer_variable(&self, i: usize) -> Integer {
        self.signature_variables.integer_variables[i]
    }

    #[inline]
    fn get_number_of_continuous_variables(&self) -> usize {
        self.signature_variables.continuous_variables.len()
    }

    #[inline]
    fn get_continuous_variable(&self, i: usize) -> Continuous {
        self.signature_variables.continuous_variables[i].into_inner()
    }

    #[inline]
    fn get_number_of_element_resource_variables(&self) -> usize {
        self.resource_variables.element_variables.len()
    }

    #[inline]
    fn get_element_resource_variable(&self, i: usize) -> Element {
        self.resource_variables.element_variables[i]
    }

    #[inline]
    fn get_number_of_integer_resource_variables(&self) -> usize {
        self.resource_variables.integer_variables.len()
    }

    #[inline]
    fn get_integer_resource_variable(&self, i: usize) -> Integer {
        self.resource_variables.integer_variables[i]
    }

    #[inline]
    fn get_number_of_continuous_resource_variables(&self) -> usize {
        self.resource_variables.continuous_variables.len()
    }

    #[inline]
    fn get_continuous_resource_variable(&self, i: usize) -> Continuous {
        self.resource_variables.continuous_variables[i].into_inner()
    }
}

/// State that can be hashed by signature variables.
///
/// However, using continuous variables in signature variables is not recommended
/// because it may cause a numerical issue.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct StateWithHashableSignatureVariables {
    /// Signature variables.
    pub signature_variables: HashableSignatureVariables,
    /// Resource variables.
    pub resource_variables: ResourceVariables,
}

impl From<dypdl::State> for StateWithHashableSignatureVariables {
    fn from(state: dypdl::State) -> StateWithHashableSignatureVariables {
        StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables::from(state.signature_variables),
            resource_variables: state.resource_variables,
        }
    }
}

impl<K> From<StateInRegistry<K>> for StateWithHashableSignatureVariables
where
    K: Hash + Eq + Clone + Debug + Deref<Target = HashableSignatureVariables>,
{
    fn from(state: StateInRegistry<K>) -> StateWithHashableSignatureVariables {
        StateWithHashableSignatureVariables {
            signature_variables: state.signature_variables.deref().clone(),
            resource_variables: state.resource_variables,
        }
    }
}

impl dypdl::StateInterface for StateWithHashableSignatureVariables {
    #[inline]
    fn get_number_of_set_variables(&self) -> usize {
        self.signature_variables.set_variables.len()
    }

    #[inline]
    fn get_set_variable(&self, i: usize) -> &Set {
        &self.signature_variables.set_variables[i]
    }

    #[inline]
    fn get_number_of_vector_variables(&self) -> usize {
        self.signature_variables.vector_variables.len()
    }

    #[inline]
    fn get_vector_variable(&self, i: usize) -> &Vector {
        &self.signature_variables.vector_variables[i]
    }

    #[inline]
    fn get_number_of_element_variables(&self) -> usize {
        self.signature_variables.element_variables.len()
    }

    #[inline]
    fn get_element_variable(&self, i: usize) -> Element {
        self.signature_variables.element_variables[i]
    }

    #[inline]
    fn get_number_of_integer_variables(&self) -> usize {
        self.signature_variables.integer_variables.len()
    }

    #[inline]
    fn get_integer_variable(&self, i: usize) -> Integer {
        self.signature_variables.integer_variables[i]
    }

    #[inline]
    fn get_number_of_continuous_variables(&self) -> usize {
        self.signature_variables.continuous_variables.len()
    }

    #[inline]
    fn get_continuous_variable(&self, i: usize) -> Continuous {
        self.signature_variables.continuous_variables[i].into_inner()
    }

    #[inline]
    fn get_number_of_element_resource_variables(&self) -> usize {
        self.resource_variables.element_variables.len()
    }

    #[inline]
    fn get_element_resource_variable(&self, i: usize) -> Element {
        self.resource_variables.element_variables[i]
    }

    #[inline]
    fn get_number_of_integer_resource_variables(&self) -> usize {
        self.resource_variables.integer_variables.len()
    }

    #[inline]
    fn get_integer_resource_variable(&self, i: usize) -> Integer {
        self.resource_variables.integer_variables[i]
    }

    #[inline]
    fn get_number_of_continuous_resource_variables(&self) -> usize {
        self.resource_variables.continuous_variables.len()
    }

    #[inline]
    fn get_continuous_resource_variable(&self, i: usize) -> Continuous {
        self.resource_variables.continuous_variables[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::variable_type::Set;
    use dypdl::StateInterface;
    use dypdl::{expression::*, TableRegistry};
    use std::rc::Rc;

    #[test]
    fn hashable_signature_variables() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let signature_variables = dypdl::SignatureVariables {
            set_variables: vec![set1, set2],
            vector_variables: vec![vec![0, 2], vec![1, 2]],
            element_variables: vec![1, 2],
            integer_variables: vec![1, 2, 3],
            continuous_variables: vec![1.0, 2.0, 3.0],
        };
        let hashable_signature_variables =
            HashableSignatureVariables::from(signature_variables.clone());
        assert_eq!(
            hashable_signature_variables.set_variables,
            signature_variables.set_variables
        );
        assert_eq!(
            hashable_signature_variables.vector_variables,
            signature_variables.vector_variables
        );
        assert_eq!(
            hashable_signature_variables.element_variables,
            signature_variables.element_variables
        );
        assert_eq!(
            hashable_signature_variables.integer_variables,
            signature_variables.integer_variables
        );
        assert_eq!(
            hashable_signature_variables.continuous_variables,
            vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)]
        );
    }

    #[test]
    fn hashable_resource_variables() {
        let resource_variables = dypdl::ResourceVariables {
            element_variables: vec![],
            integer_variables: vec![4, 5, 6],
            continuous_variables: vec![4.0, 5.0, 6.0],
        };
        let hashable_resource_variables =
            HashableResourceVariables::from(resource_variables.clone());
        assert_eq!(
            hashable_resource_variables.element_variables,
            resource_variables.element_variables
        );
        assert_eq!(
            hashable_resource_variables.integer_variables,
            resource_variables.integer_variables
        );
        assert_eq!(
            hashable_resource_variables.continuous_variables,
            vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)]
        );
    }

    #[test]
    fn hashable_state_from() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let signature_variables = dypdl::SignatureVariables {
            set_variables: vec![set1, set2],
            vector_variables: vec![vec![0, 2], vec![1, 2]],
            element_variables: vec![1, 2],
            integer_variables: vec![1, 2, 3],
            continuous_variables: vec![1.0, 2.0, 3.0],
        };
        let resource_variables = dypdl::ResourceVariables {
            element_variables: vec![],
            integer_variables: vec![4, 5, 6],
            continuous_variables: vec![4.0, 5.0, 6.0],
        };
        let state = HashableState::from(dypdl::State {
            signature_variables: signature_variables.clone(),
            resource_variables: resource_variables.clone(),
        });
        assert_eq!(
            state.signature_variables.set_variables,
            signature_variables.set_variables
        );
        assert_eq!(
            state.signature_variables.vector_variables,
            signature_variables.vector_variables
        );
        assert_eq!(
            state.signature_variables.element_variables,
            signature_variables.element_variables
        );
        assert_eq!(
            state.signature_variables.integer_variables,
            signature_variables.integer_variables
        );
        assert_eq!(
            state.signature_variables.continuous_variables,
            vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)]
        );
        assert_eq!(
            state.resource_variables.element_variables,
            resource_variables.element_variables
        );
        assert_eq!(
            state.resource_variables.integer_variables,
            resource_variables.integer_variables
        );
        assert_eq!(
            state.resource_variables.continuous_variables,
            vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)]
        );
    }

    #[test]
    fn state_get_number_of_set_variables() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![Set::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_set_variables(), 1);
    }

    #[test]
    fn state_get_set_variable() {
        let mut set = Set::with_capacity(2);
        set.insert(1);
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![Set::with_capacity(2), set.clone()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_set_variable(0), &Set::with_capacity(2));
        assert_eq!(state.get_set_variable(1), &set);
    }

    #[test]
    #[should_panic]
    fn state_get_set_variable_panic() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![Set::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_set_variable(1);
    }

    #[test]
    fn state_get_number_of_vector_variables() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                vector_variables: vec![Vector::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_vector_variables(), 1);
    }

    #[test]
    fn state_get_vector_variable() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                vector_variables: vec![Vector::default(), vec![1]],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_vector_variable(0), &Vector::default());
        assert_eq!(state.get_vector_variable(1), &vec![1]);
    }

    #[test]
    #[should_panic]
    fn state_get_vector_variable_panic() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                vector_variables: vec![Vector::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_vector_variable(1);
    }

    #[test]
    fn state_get_number_of_element_variables() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_element_variables(), 1);
    }

    #[test]
    fn state_get_element_variable() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                element_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_element_variable(0), 0);
        assert_eq!(state.get_element_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_element_variable_panic() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_element_variable(1);
    }

    #[test]
    fn state_get_number_of_integer_variables() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_integer_variables(), 1);
    }

    #[test]
    fn state_get_integer_variable() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                integer_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_integer_variable(0), 0);
        assert_eq!(state.get_integer_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_integer_variable_panic() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_integer_variable(1);
    }

    #[test]
    fn state_get_number_of_continuous_variables() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                continuous_variables: vec![OrderedContinuous::from(0.0)],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_continuous_variables(), 1);
    }

    #[test]
    fn state_get_continuous_variable() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                continuous_variables: vec![
                    OrderedContinuous::from(0.0),
                    OrderedContinuous::from(1.0),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_continuous_variable(0), 0.0);
        assert_eq!(state.get_continuous_variable(1), 1.0);
    }

    #[test]
    #[should_panic]
    fn state_get_continuous_variable_panic() {
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                continuous_variables: vec![OrderedContinuous::from(0.0)],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_continuous_variable(1);
    }

    #[test]
    fn state_get_number_of_element_resource_variables() {
        let state = HashableState {
            resource_variables: HashableResourceVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_element_resource_variables(), 1);
    }

    #[test]
    fn state_get_element_resource_variable() {
        let state = HashableState {
            resource_variables: HashableResourceVariables {
                element_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_element_resource_variable(0), 0);
        assert_eq!(state.get_element_resource_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_element_resource_variable_panic() {
        let state = HashableState {
            resource_variables: HashableResourceVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_element_resource_variable(1);
    }

    #[test]
    fn state_get_number_of_integer_resource_variables() {
        let state = HashableState {
            resource_variables: HashableResourceVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_integer_resource_variables(), 1);
    }

    #[test]
    fn state_get_integer_resource_variable() {
        let state = HashableState {
            resource_variables: HashableResourceVariables {
                integer_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_integer_resource_variable(0), 0);
        assert_eq!(state.get_integer_resource_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_integer_resource_variable_panic() {
        let state = HashableState {
            resource_variables: HashableResourceVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_integer_resource_variable(1);
    }

    #[test]
    fn state_get_number_of_continuous_resource_variables() {
        let state = HashableState {
            resource_variables: HashableResourceVariables {
                continuous_variables: vec![OrderedContinuous::from(0.0)],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_continuous_resource_variables(), 1);
    }

    #[test]
    fn state_get_continuous_resource_variable() {
        let state = HashableState {
            resource_variables: HashableResourceVariables {
                continuous_variables: vec![
                    OrderedContinuous::from(0.0),
                    OrderedContinuous::from(1.0),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_continuous_resource_variable(0), 0.0);
        assert_eq!(state.get_continuous_resource_variable(1), 1.0);
    }

    #[test]
    #[should_panic]
    fn state_get_continuous_resource_variable_panic() {
        let state = HashableState {
            resource_variables: HashableResourceVariables {
                continuous_variables: vec![OrderedContinuous::from(0.0)],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_continuous_resource_variable(1);
    }

    #[test]
    fn apply_effect() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            },
            resource_variables: HashableResourceVariables {
                element_variables: vec![],
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
            },
        };
        let registry = TableRegistry::default();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let vector_effect1 = VectorExpression::Push(
            ElementExpression::Constant(1),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let vector_effect2 = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                1,
            ))),
        );
        let element_effect1 = ElementExpression::Constant(2);
        let element_effect2 = ElementExpression::Constant(1);
        let integer_effect1 = IntegerExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Constant(1)),
        );
        let integer_effect2 = IntegerExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(IntegerExpression::Variable(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        let continuous_effect1 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        );
        let continuous_effect2 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(ContinuousExpression::Variable(1)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        let integer_resource_effect1 = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::ResourceVariable(0)),
            Box::new(IntegerExpression::Constant(1)),
        );
        let integer_resource_effect2 = IntegerExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(IntegerExpression::ResourceVariable(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        let continuous_resource_effect1 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::ResourceVariable(0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        );
        let continuous_resource_effect2 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(ContinuousExpression::ResourceVariable(1)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        let effect = dypdl::Effect {
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
            continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
            element_resource_effects: vec![],
            integer_resource_effects: vec![
                (0, integer_resource_effect1),
                (1, integer_resource_effect2),
            ],
            continuous_resource_effects: vec![
                (0, continuous_resource_effect1),
                (1, continuous_resource_effect2),
            ],
        };

        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(1);
        let expected = HashableState {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![OrderedFloat(0.0), OrderedFloat(4.0), OrderedFloat(3.0)],
            },
            resource_variables: HashableResourceVariables {
                element_variables: vec![],
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![OrderedFloat(5.0), OrderedFloat(2.5), OrderedFloat(6.0)],
            },
        };
        let successor: HashableState = state.apply_effect(&effect, &registry);
        assert_eq!(successor, expected);
    }

    #[test]
    fn state_with_hashable_signature_variable_from() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let signature_variables = dypdl::SignatureVariables {
            set_variables: vec![set1, set2],
            vector_variables: vec![vec![0, 2], vec![1, 2]],
            element_variables: vec![1, 2],
            integer_variables: vec![1, 2, 3],
            continuous_variables: vec![1.0, 2.0, 3.0],
        };
        let resource_variables = dypdl::ResourceVariables {
            element_variables: vec![],
            integer_variables: vec![4, 5, 6],
            continuous_variables: vec![4.0, 5.0, 6.0],
        };
        let state = StateWithHashableSignatureVariables::from(dypdl::State {
            signature_variables: signature_variables.clone(),
            resource_variables: resource_variables.clone(),
        });
        assert_eq!(
            state.signature_variables.set_variables,
            signature_variables.set_variables
        );
        assert_eq!(
            state.signature_variables.vector_variables,
            signature_variables.vector_variables
        );
        assert_eq!(
            state.signature_variables.element_variables,
            signature_variables.element_variables
        );
        assert_eq!(
            state.signature_variables.integer_variables,
            signature_variables.integer_variables
        );
        assert_eq!(
            state.signature_variables.continuous_variables,
            vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)]
        );
        assert_eq!(
            state.resource_variables.element_variables,
            resource_variables.element_variables
        );
        assert_eq!(
            state.resource_variables.integer_variables,
            resource_variables.integer_variables
        );
        assert_eq!(
            state.resource_variables.continuous_variables,
            resource_variables.continuous_variables
        );
    }

    #[test]
    fn state_with_hashable_signature_variable_from_state_in_registry() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let signature_variables = HashableSignatureVariables {
            set_variables: vec![set1, set2],
            vector_variables: vec![vec![0, 2], vec![1, 2]],
            element_variables: vec![1, 2],
            integer_variables: vec![1, 2, 3],
            continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
        };
        let resource_variables = dypdl::ResourceVariables {
            element_variables: vec![],
            integer_variables: vec![4, 5, 6],
            continuous_variables: vec![4.0, 5.0, 6.0],
        };
        let state = StateWithHashableSignatureVariables::from(StateInRegistry {
            signature_variables: Rc::new(signature_variables.clone()),
            resource_variables: resource_variables.clone(),
        });
        assert_eq!(
            state.signature_variables.set_variables,
            signature_variables.set_variables
        );
        assert_eq!(
            state.signature_variables.vector_variables,
            signature_variables.vector_variables
        );
        assert_eq!(
            state.signature_variables.element_variables,
            signature_variables.element_variables
        );
        assert_eq!(
            state.signature_variables.integer_variables,
            signature_variables.integer_variables
        );
        assert_eq!(
            state.signature_variables.continuous_variables,
            vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)]
        );
        assert_eq!(
            state.resource_variables.element_variables,
            resource_variables.element_variables
        );
        assert_eq!(
            state.resource_variables.integer_variables,
            resource_variables.integer_variables
        );
        assert_eq!(
            state.resource_variables.continuous_variables,
            resource_variables.continuous_variables
        );
    }

    #[test]
    fn state_with_hashable_signature_variables_get_number_of_set_variables() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![Set::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_set_variables(), 1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_set_variable() {
        let mut set = Set::with_capacity(2);
        set.insert(1);
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![Set::with_capacity(2), set.clone()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_set_variable(0), &Set::with_capacity(2));
        assert_eq!(state.get_set_variable(1), &set);
    }

    #[test]
    #[should_panic]
    fn state_with_hashable_signature_variables_get_set_variable_panic() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![Set::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_set_variable(1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_number_of_vector_variables() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                vector_variables: vec![Vector::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_vector_variables(), 1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_vector_variable() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                vector_variables: vec![Vector::default(), vec![1]],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_vector_variable(0), &Vector::default());
        assert_eq!(state.get_vector_variable(1), &vec![1]);
    }

    #[test]
    #[should_panic]
    fn state_with_hashable_signature_variables_get_vector_variable_panic() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                vector_variables: vec![Vector::default()],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_vector_variable(1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_number_of_element_variables() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_element_variables(), 1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_element_variable() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                element_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_element_variable(0), 0);
        assert_eq!(state.get_element_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_with_hashable_signature_variables_get_element_variable_panic() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_element_variable(1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_number_of_integer_variables() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_integer_variables(), 1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_integer_variable() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                integer_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_integer_variable(0), 0);
        assert_eq!(state.get_integer_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_with_hashable_signature_variables_get_integer_variable_panic() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_integer_variable(1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_number_of_continuous_variables() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                continuous_variables: vec![OrderedContinuous::from(0.0)],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_continuous_variables(), 1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_continuous_variable() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                continuous_variables: vec![
                    OrderedContinuous::from(0.0),
                    OrderedContinuous::from(1.0),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_continuous_variable(0), 0.0);
        assert_eq!(state.get_continuous_variable(1), 1.0);
    }

    #[test]
    #[should_panic]
    fn state_with_hashable_signature_variables_get_continuous_variable_panic() {
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                continuous_variables: vec![OrderedContinuous::from(0.0)],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_continuous_variable(1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_number_of_element_resource_variables() {
        let state = StateWithHashableSignatureVariables {
            resource_variables: ResourceVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_element_resource_variables(), 1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_element_resource_variable() {
        let state = StateWithHashableSignatureVariables {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_element_resource_variable(0), 0);
        assert_eq!(state.get_element_resource_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_with_hashable_signature_variables_get_element_resource_variable_panic() {
        let state = StateWithHashableSignatureVariables {
            resource_variables: ResourceVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_element_resource_variable(1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_number_of_integer_resource_variables() {
        let state = StateWithHashableSignatureVariables {
            resource_variables: ResourceVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_integer_resource_variables(), 1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_integer_resource_variable() {
        let state = StateWithHashableSignatureVariables {
            resource_variables: ResourceVariables {
                integer_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_integer_resource_variable(0), 0);
        assert_eq!(state.get_integer_resource_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_with_hashable_signature_variables_get_integer_resource_variable_panic() {
        let state = StateWithHashableSignatureVariables {
            resource_variables: ResourceVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_integer_resource_variable(1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_number_of_continuous_resource_variables() {
        let state = StateWithHashableSignatureVariables {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_continuous_resource_variables(), 1);
    }

    #[test]
    fn state_with_hashable_signature_variables_get_continuous_resource_variable() {
        let state = StateWithHashableSignatureVariables {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0, 1.0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_continuous_resource_variable(0), 0.0);
        assert_eq!(state.get_continuous_resource_variable(1), 1.0);
    }

    #[test]
    #[should_panic]
    fn state_with_hashable_signature_variables_get_continuous_resource_variable_panic() {
        let state = StateWithHashableSignatureVariables {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_continuous_resource_variable(1);
    }

    #[test]
    fn state_with_hashable_signature_variables_apply_effect() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let state = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![],
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        };
        let registry = TableRegistry::default();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let vector_effect1 = VectorExpression::Push(
            ElementExpression::Constant(1),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let vector_effect2 = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                1,
            ))),
        );
        let element_effect1 = ElementExpression::Constant(2);
        let element_effect2 = ElementExpression::Constant(1);
        let integer_effect1 = IntegerExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Constant(1)),
        );
        let integer_effect2 = IntegerExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(IntegerExpression::Variable(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        let continuous_effect1 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        );
        let continuous_effect2 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(ContinuousExpression::Variable(1)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        let integer_resource_effect1 = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::ResourceVariable(0)),
            Box::new(IntegerExpression::Constant(1)),
        );
        let integer_resource_effect2 = IntegerExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(IntegerExpression::ResourceVariable(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        let continuous_resource_effect1 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::ResourceVariable(0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        );
        let continuous_resource_effect2 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(ContinuousExpression::ResourceVariable(1)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        let effect = dypdl::Effect {
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
            continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
            element_resource_effects: vec![],
            integer_resource_effects: vec![
                (0, integer_resource_effect1),
                (1, integer_resource_effect2),
            ],
            continuous_resource_effects: vec![
                (0, continuous_resource_effect1),
                (1, continuous_resource_effect2),
            ],
        };

        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(1);
        let expected = StateWithHashableSignatureVariables {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![OrderedFloat(0.0), OrderedFloat(4.0), OrderedFloat(3.0)],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![],
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
        };
        let successor: StateWithHashableSignatureVariables = state.apply_effect(&effect, &registry);
        assert_eq!(successor, expected);
    }
}
