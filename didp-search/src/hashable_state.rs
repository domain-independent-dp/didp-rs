use didp_parser::variable::{Continuous, Element, Integer, OrderedContinuous, Set, Vector};
use ordered_float::OrderedFloat;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Default)]
pub struct HashableSignatureVariables {
    pub set_variables: Vec<Set>,
    pub vector_variables: Vec<Vector>,
    pub element_variables: Vec<Element>,
    pub integer_variables: Vec<Integer>,
    pub continuous_variables: Vec<OrderedContinuous>,
}

impl HashableSignatureVariables {
    pub fn new(variables: &didp_parser::SignatureVariables) -> HashableSignatureVariables {
        HashableSignatureVariables {
            set_variables: variables.set_variables.clone(),
            vector_variables: variables.vector_variables.clone(),
            element_variables: variables.element_variables.clone(),
            integer_variables: variables.integer_variables.clone(),
            continuous_variables: variables
                .continuous_variables
                .iter()
                .map(|v| OrderedFloat(*v))
                .collect(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Default)]
pub struct HashableResourceVariables {
    pub integer_variables: Vec<Integer>,
    pub continuous_variables: Vec<OrderedContinuous>,
}

impl HashableResourceVariables {
    pub fn new(variables: &didp_parser::ResourceVariables) -> HashableResourceVariables {
        HashableResourceVariables {
            integer_variables: variables.integer_variables.clone(),
            continuous_variables: variables
                .continuous_variables
                .iter()
                .map(|v| OrderedFloat(*v))
                .collect(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Default)]
pub struct HashableState {
    signature_variables: HashableSignatureVariables,
    resource_variables: HashableResourceVariables,
}

impl HashableState {
    pub fn new(state: &didp_parser::State) -> HashableState {
        HashableState {
            signature_variables: HashableSignatureVariables::new(&state.signature_variables),
            resource_variables: HashableResourceVariables::new(&state.resource_variables),
        }
    }
}

impl didp_parser::DPState for HashableState {
    #[inline]
    fn get_set_variable(&self, i: usize) -> &Set {
        &self.signature_variables.set_variables[i]
    }

    #[inline]
    fn get_vector_variable(&self, i: usize) -> &Vector {
        &self.signature_variables.vector_variables[i]
    }

    #[inline]
    fn get_element_variable(&self, i: usize) -> Element {
        self.signature_variables.element_variables[i]
    }

    #[inline]
    fn get_integer_variable(&self, i: usize) -> Integer {
        self.signature_variables.integer_variables[i]
    }

    #[inline]
    fn get_continuous_variable(&self, i: usize) -> Continuous {
        self.signature_variables.continuous_variables[i].into_inner()
    }

    #[inline]
    fn get_integer_resource_variable(&self, i: usize) -> Integer {
        self.resource_variables.integer_variables[i]
    }

    #[inline]
    fn get_continuous_resource_variable(&self, i: usize) -> Continuous {
        self.resource_variables.continuous_variables[i].into_inner()
    }

    fn apply_effect(
        &self,
        effect: &didp_parser::Effect,
        registry: &didp_parser::TableRegistry,
    ) -> Self {
        let len = self.signature_variables.set_variables.len();
        let mut set_variables = Vec::with_capacity(len);
        let mut i = 0;
        for e in &effect.set_effects {
            while i < e.0 {
                set_variables.push(self.signature_variables.set_variables[i].clone());
                i += 1;
            }
            set_variables.push(e.1.eval(self, registry));
            i += 1;
        }
        while i < len {
            set_variables.push(self.signature_variables.set_variables[i].clone());
            i += 1;
        }

        let len = self.signature_variables.vector_variables.len();
        let mut vector_variables = Vec::with_capacity(len);
        for e in &effect.vector_effects {
            while i < e.0 {
                vector_variables.push(self.signature_variables.vector_variables[i].clone());
                i += 1;
            }
            vector_variables.push(e.1.eval(self, registry));
            i += 1;
        }
        while i < len {
            vector_variables.push(self.signature_variables.vector_variables[i].clone());
            i += 1;
        }

        let mut element_variables = self.signature_variables.element_variables.clone();
        for e in &effect.element_effects {
            element_variables[e.0] = e.1.eval(self, registry);
        }

        let mut integer_variables = self.signature_variables.integer_variables.clone();
        for e in &effect.integer_effects {
            integer_variables[e.0] = e.1.eval(self, registry);
        }

        let mut continuous_variables = self.signature_variables.continuous_variables.clone();
        for e in &effect.continuous_effects {
            continuous_variables[e.0] = OrderedFloat(e.1.eval(self, registry));
        }

        let mut integer_resource_variables = self.resource_variables.integer_variables.clone();
        for e in &effect.integer_resource_effects {
            integer_resource_variables[e.0] = e.1.eval(self, registry);
        }

        let mut continuous_resource_variables =
            self.resource_variables.continuous_variables.clone();
        for e in &effect.continuous_resource_effects {
            continuous_resource_variables[e.0] = OrderedFloat(e.1.eval(self, registry));
        }

        HashableState {
            signature_variables: HashableSignatureVariables {
                set_variables,
                vector_variables,
                element_variables,
                integer_variables,
                continuous_variables,
            },
            resource_variables: HashableResourceVariables {
                integer_variables: integer_resource_variables,
                continuous_variables: continuous_resource_variables,
            },
        }
    }

    fn apply_effect_in_place(
        &mut self,
        effect: &didp_parser::Effect,
        registry: &didp_parser::TableRegistry,
    ) {
        for e in &effect.set_effects {
            self.signature_variables.set_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.vector_effects {
            self.signature_variables.vector_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.element_effects {
            self.signature_variables.element_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.integer_effects {
            self.signature_variables.integer_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.continuous_effects {
            self.signature_variables.continuous_variables[e.0] =
                OrderedFloat(e.1.eval(self, registry));
        }
        for e in &effect.integer_resource_effects {
            self.resource_variables.integer_variables[e.0] = e.1.eval(self, registry);
        }
        for e in &effect.continuous_resource_effects {
            self.resource_variables.continuous_variables[e.0] =
                OrderedFloat(e.1.eval(self, registry));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use didp_parser::expression::*;
    use didp_parser::variable::Set;
    use didp_parser::DPState;
    use rustc_hash::FxHashMap;

    fn generate_registry() -> didp_parser::TableRegistry {
        let tables_1d = vec![didp_parser::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![didp_parser::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        didp_parser::TableRegistry {
            integer_tables: didp_parser::TableData {
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
    fn hashable_signature_variables() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let signature_variables = didp_parser::SignatureVariables {
            set_variables: vec![set1, set2],
            vector_variables: vec![vec![0, 2], vec![1, 2]],
            element_variables: vec![1, 2],
            integer_variables: vec![1, 2, 3],
            continuous_variables: vec![1.0, 2.0, 3.0],
        };
        let hashable_signature_variables = HashableSignatureVariables::new(&signature_variables);
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
    fn hashable_resouce_variables() {
        let resource_variables = didp_parser::ResourceVariables {
            integer_variables: vec![4, 5, 6],
            continuous_variables: vec![4.0, 5.0, 6.0],
        };
        let hashable_resouce_variables = HashableResourceVariables::new(&resource_variables);
        assert_eq!(
            hashable_resouce_variables.integer_variables,
            resource_variables.integer_variables
        );
        assert_eq!(
            hashable_resouce_variables.continuous_variables,
            vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)]
        );
    }

    #[test]
    fn state_getter() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let state = HashableState {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![set1.clone(), set2.clone()],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            },
            resource_variables: HashableResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
            },
        };
        assert_eq!(state.get_set_variable(0), &set1);
        assert_eq!(state.get_set_variable(1), &set2);
        assert_eq!(state.get_vector_variable(0), &vec![0, 2]);
        assert_eq!(state.get_vector_variable(1), &vec![1, 2]);
        assert_eq!(state.get_element_variable(0), 1);
        assert_eq!(state.get_element_variable(1), 2);
        assert_eq!(state.get_integer_variable(0), 1);
        assert_eq!(state.get_integer_variable(1), 2);
        assert_eq!(state.get_integer_variable(2), 3);
        assert_eq!(state.get_continuous_variable(0), 1.0);
        assert_eq!(state.get_continuous_variable(1), 2.0);
        assert_eq!(state.get_continuous_variable(2), 3.0);
        assert_eq!(state.get_integer_resource_variable(0), 4);
        assert_eq!(state.get_integer_resource_variable(1), 5);
        assert_eq!(state.get_integer_resource_variable(2), 6);
        assert_eq!(state.get_continuous_resource_variable(0), 4.0);
        assert_eq!(state.get_continuous_resource_variable(1), 5.0);
        assert_eq!(state.get_continuous_resource_variable(2), 6.0);
        let state = HashableState::new(&didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                set_variables: vec![set1.clone(), set2.clone()],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: didp_parser::ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        });
        assert_eq!(state.get_set_variable(0), &set1);
        assert_eq!(state.get_set_variable(1), &set2);
        assert_eq!(state.get_vector_variable(0), &vec![0, 2]);
        assert_eq!(state.get_vector_variable(1), &vec![1, 2]);
        assert_eq!(state.get_element_variable(0), 1);
        assert_eq!(state.get_element_variable(1), 2);
        assert_eq!(state.get_integer_variable(0), 1);
        assert_eq!(state.get_integer_variable(1), 2);
        assert_eq!(state.get_integer_variable(2), 3);
        assert_eq!(state.get_continuous_variable(0), 1.0);
        assert_eq!(state.get_continuous_variable(1), 2.0);
        assert_eq!(state.get_continuous_variable(2), 3.0);
        assert_eq!(state.get_integer_resource_variable(0), 4);
        assert_eq!(state.get_integer_resource_variable(1), 5);
        assert_eq!(state.get_integer_resource_variable(2), 6);
        assert_eq!(state.get_continuous_resource_variable(0), 4.0);
        assert_eq!(state.get_continuous_resource_variable(1), 5.0);
        assert_eq!(state.get_continuous_resource_variable(2), 6.0);
    }

    #[test]
    fn appy_effect() {
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
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
            },
        };
        let registry = generate_registry();
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
        let integer_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::IntegerVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::ContinuousVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::ContinuousVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let integer_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::IntegerResourceVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::IntegerResourceVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::ContinuousResourceVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::ContinuousResourceVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let effect = didp_parser::Effect {
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
            continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
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
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![OrderedFloat(5.0), OrderedFloat(2.5), OrderedFloat(6.0)],
            },
        };
        let successor = state.apply_effect(&effect, &registry);
        assert_eq!(successor, expected);
    }

    #[test]
    fn appy_effect_in_place() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let mut state = HashableState {
            signature_variables: HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            },
            resource_variables: HashableResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![OrderedFloat(4.0), OrderedFloat(5.0), OrderedFloat(6.0)],
            },
        };
        let registry = generate_registry();
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
        let integer_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::IntegerVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::IntegerVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(NumericExpression::ContinuousVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(NumericExpression::ContinuousVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let integer_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::IntegerResourceVariable(0)),
            Box::new(NumericExpression::Constant(1)),
        );
        let integer_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::IntegerResourceVariable(1)),
            Box::new(NumericExpression::Constant(2)),
        );
        let continuous_resource_effect1 = NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(NumericExpression::ContinuousResourceVariable(0)),
            Box::new(NumericExpression::Constant(1.0)),
        );
        let continuous_resource_effect2 = NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(NumericExpression::ContinuousResourceVariable(1)),
            Box::new(NumericExpression::Constant(2.0)),
        );
        let effect = didp_parser::Effect {
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
            continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
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
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![OrderedFloat(5.0), OrderedFloat(2.5), OrderedFloat(6.0)],
            },
        };
        state.apply_effect_in_place(&effect, &registry);
        assert_eq!(state, expected);
    }
}
