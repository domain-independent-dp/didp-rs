use crate::hashable_state;
use didp_parser::variable::{Continuous, Element, Integer, Numeric, Set, Vector};
use didp_parser::ReduceFunction;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections;
use std::fmt;
use std::mem;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct StateInRegistry {
    pub signature_variables: Rc<hashable_state::HashableSignatureVariables>,
    pub resource_variables: didp_parser::ResourceVariables,
}

impl StateInRegistry {
    pub fn new(state: &didp_parser::State) -> StateInRegistry {
        StateInRegistry {
            signature_variables: Rc::new(hashable_state::HashableSignatureVariables::new(
                &state.signature_variables,
            )),
            resource_variables: state.resource_variables.clone(),
        }
    }
}

impl didp_parser::DPState for StateInRegistry {
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
    fn get_element_resource_variable(&self, i: usize) -> Element {
        self.resource_variables.element_variables[i]
    }

    #[inline]
    fn get_integer_resource_variable(&self, i: usize) -> Integer {
        self.resource_variables.integer_variables[i]
    }

    #[inline]
    fn get_continuous_resource_variable(&self, i: usize) -> Continuous {
        self.resource_variables.continuous_variables[i]
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
            continuous_variables[e.0] = ordered_float::OrderedFloat(e.1.eval(self, registry));
        }

        let mut element_resource_variables = self.resource_variables.element_variables.clone();
        for e in &effect.element_resource_effects {
            element_resource_variables[e.0] = e.1.eval(self, registry);
        }

        let mut integer_resource_variables = self.resource_variables.integer_variables.clone();
        for e in &effect.integer_resource_effects {
            integer_resource_variables[e.0] = e.1.eval(self, registry);
        }

        let mut continuous_resource_variables =
            self.resource_variables.continuous_variables.clone();
        for e in &effect.continuous_resource_effects {
            continuous_resource_variables[e.0] = e.1.eval(self, registry);
        }

        StateInRegistry {
            signature_variables: {
                Rc::new(hashable_state::HashableSignatureVariables {
                    set_variables,
                    vector_variables,
                    element_variables,
                    integer_variables,
                    continuous_variables,
                })
            },
            resource_variables: didp_parser::ResourceVariables {
                element_variables: element_resource_variables,
                integer_variables: integer_resource_variables,
                continuous_variables: continuous_resource_variables,
            },
        }
    }

    #[inline]
    fn apply_effect_in_place(
        &mut self,
        effect: &didp_parser::Effect,
        registry: &didp_parser::TableRegistry,
    ) {
        *self = self.apply_effect(effect, registry);
    }
}

pub trait StateInformation<T: Numeric>: Clone + fmt::Debug {
    fn state(&self) -> &StateInRegistry;

    fn cost(&self) -> T;
}

pub struct StateRegistry<'a, T: Numeric, I: StateInformation<T>> {
    registry: FxHashMap<Rc<hashable_state::HashableSignatureVariables>, Vec<I>>,
    metadata: &'a didp_parser::StateMetadata,
    reduce_function: &'a ReduceFunction,
    phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Numeric, I: StateInformation<T>> StateRegistry<'a, T, I> {
    #[inline]
    pub fn new(model: &'a didp_parser::Model<T>) -> StateRegistry<T, I> {
        StateRegistry {
            registry: FxHashMap::default(),
            metadata: &model.state_metadata,
            reduce_function: &model.reduce_function,
            phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn reserve(&mut self, capacity: usize) {
        self.registry.reserve(capacity);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.registry.clear();
    }

    pub fn get(&self, state: &StateInRegistry, cost: T) -> Option<&I> {
        if let Some(v) = self.registry.get(&state.signature_variables) {
            for other in v {
                let result = self.metadata.dominance(state, other.state());
                match result {
                    Some(Ordering::Equal) | Some(Ordering::Less)
                        if (*self.reduce_function == ReduceFunction::Max
                            && cost <= other.cost())
                            || (*self.reduce_function == ReduceFunction::Min
                                && cost >= other.cost()) =>
                    {
                        return Some(other)
                    }
                    _ => {}
                }
            }
        }
        None
    }

    pub fn insert<F>(
        &mut self,
        mut state: StateInRegistry,
        cost: T,
        constructor: F,
    ) -> Option<(I, Option<I>)>
    where
        F: FnOnce(StateInRegistry, T, Option<&I>) -> Option<I>,
    {
        let entry = self.registry.entry(state.signature_variables.clone());
        let v = match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                state.signature_variables = entry.key().clone();
                let v = entry.into_mut();
                for other in v.iter_mut() {
                    let result = self.metadata.dominance(&state, other.state());
                    match result {
                        Some(Ordering::Equal) | Some(Ordering::Less)
                            if (*self.reduce_function == ReduceFunction::Max
                                && cost <= other.cost())
                                || (*self.reduce_function == ReduceFunction::Min
                                    && cost >= other.cost()) =>
                        {
                            // dominated
                            return None;
                        }
                        Some(Ordering::Equal) | Some(Ordering::Greater)
                            if (*self.reduce_function == ReduceFunction::Max
                                && cost >= other.cost())
                                || (*self.reduce_function == ReduceFunction::Min
                                    && cost <= other.cost()) =>
                        {
                            // dominating
                            if let Some(information) = match result.unwrap() {
                                // if the same state is saved, reuse some information
                                Ordering::Equal => constructor(state, cost, Some(other)),
                                _ => constructor(state, cost, None),
                            } {
                                let mut tmp = information.clone();
                                mem::swap(other, &mut tmp);
                                return Some((information, Some(tmp)));
                            } else {
                                return None;
                            }
                        }
                        _ => {}
                    }
                }
                v
            }
            collections::hash_map::Entry::Vacant(entry) => entry.insert(Vec::with_capacity(1)),
        };
        if let Some(information) = constructor(state, cost, None) {
            v.push(information.clone());
            Some((information, None))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashable_state::HashableSignatureVariables;
    use didp_parser::expression::*;
    use didp_parser::variable::Set;
    use didp_parser::DPState;
    use didp_parser::ResourceVariables;
    use ordered_float::OrderedFloat;
    use rustc_hash::FxHashMap;
    use std::cell::RefCell;

    #[derive(Debug, PartialEq, Clone)]
    struct MockInformation {
        state: StateInRegistry,
        value: RefCell<Option<i32>>,
    }

    impl StateInformation<Integer> for Rc<MockInformation> {
        fn state(&self) -> &StateInRegistry {
            &self.state
        }

        fn cost(&self) -> Integer {
            1
        }
    }

    fn generate_model() -> didp_parser::Model<Integer> {
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert("n1".to_string(), 0);
        name_to_integer_variable.insert("n2".to_string(), 1);
        name_to_integer_variable.insert("n3".to_string(), 2);

        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert("r1".to_string(), 0);
        name_to_integer_resource_variable.insert("r2".to_string(), 1);
        name_to_integer_resource_variable.insert("r3".to_string(), 2);

        let state_metadata = didp_parser::StateMetadata {
            integer_variable_names: vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
            name_to_integer_variable,
            integer_resource_variable_names: vec![
                "r1".to_string(),
                "r2".to_string(),
                "r3".to_string(),
            ],
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true],
            ..Default::default()
        };
        didp_parser::Model {
            state_metadata,
            reduce_function: didp_parser::ReduceFunction::Min,
            ..Default::default()
        }
    }

    fn generate_signature_variables(
        integer_variables: Vec<Integer>,
    ) -> Rc<hashable_state::HashableSignatureVariables> {
        Rc::new(hashable_state::HashableSignatureVariables {
            integer_variables,
            ..Default::default()
        })
    }

    fn generate_resource_variables(
        integer_variables: Vec<Integer>,
    ) -> didp_parser::ResourceVariables {
        didp_parser::ResourceVariables {
            integer_variables,
            ..Default::default()
        }
    }

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
    fn state_getter() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1.clone(), set2.clone()],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2],
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
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
        assert_eq!(state.get_element_resource_variable(0), 0);
        assert_eq!(state.get_element_resource_variable(1), 1);
        assert_eq!(state.get_element_resource_variable(2), 2);
        assert_eq!(state.get_integer_resource_variable(0), 4);
        assert_eq!(state.get_integer_resource_variable(1), 5);
        assert_eq!(state.get_integer_resource_variable(2), 6);
        assert_eq!(state.get_continuous_resource_variable(0), 4.0);
        assert_eq!(state.get_continuous_resource_variable(1), 5.0);
        assert_eq!(state.get_continuous_resource_variable(2), 6.0);
        let state = StateInRegistry::new(&didp_parser::State {
            signature_variables: didp_parser::SignatureVariables {
                set_variables: vec![set1.clone(), set2.clone()],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: didp_parser::ResourceVariables {
                element_variables: vec![0, 1, 2],
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
        assert_eq!(state.get_element_resource_variable(0), 0);
        assert_eq!(state.get_element_resource_variable(1), 1);
        assert_eq!(state.get_element_resource_variable(2), 2);
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
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1],
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
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
        let element_resource_effect1 = ElementExpression::Constant(1);
        let element_resource_effect2 = ElementExpression::Constant(0);
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
            element_resource_effects: vec![
                (0, element_resource_effect1),
                (1, element_resource_effect2),
            ],
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
        let expected = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![OrderedFloat(0.0), OrderedFloat(4.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
                element_variables: vec![1, 0],
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
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
        let mut state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1],
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
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
        let element_resource_effect1 = ElementExpression::Constant(1);
        let element_resource_effect2 = ElementExpression::Constant(0);
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
            element_resource_effects: vec![
                (0, element_resource_effect1),
                (1, element_resource_effect2),
            ],
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
        let expected = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![OrderedFloat(0.0), OrderedFloat(4.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
                element_variables: vec![1, 0],
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
        };
        state.apply_effect_in_place(&effect, &registry);
        assert_eq!(state, expected);
    }

    #[test]
    fn get_new_information() {
        let model = generate_model();
        let mut registry = StateRegistry::new(&model);
        let constructor = |state: StateInRegistry, _: Integer, _: Option<&Rc<MockInformation>>| {
            Some(Rc::new(MockInformation {
                state,
                value: RefCell::new(None),
            }))
        };

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = registry.insert(state, 1, &constructor);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = registry.insert(state, 1, &constructor);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = registry.insert(state, 1, &constructor);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information = registry.insert(state, 0, &constructor);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);
    }

    #[test]
    fn information_dominated() {
        let model = generate_model();
        let mut registry = StateRegistry::new(&model);

        let constructor = |state: StateInRegistry, _: Integer, _: Option<&Rc<MockInformation>>| {
            Some(Rc::new(MockInformation {
                state,
                value: RefCell::new(None),
            }))
        };

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let previous = registry.insert(state, 1, &constructor);
        assert!(previous.is_some());
        let (previous, _) = previous.unwrap();

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(&previous));
        let information = registry.insert(state, 1, &constructor);
        assert!(information.is_none());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(&previous));
        let information = registry.insert(state, 1, &constructor);
        assert!(information.is_none());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 2), Some(&previous));
        let information = registry.insert(state, 2, &constructor);
        assert!(information.is_none());
    }

    #[test]
    fn get_dominating_information() {
        let model = generate_model();
        let mut registry = StateRegistry::new(&model);
        let constructor =
            |state: StateInRegistry, _: Integer, other: Option<&Rc<MockInformation>>| {
                if let Some(other) = other {
                    other.value.borrow().map(|value| {
                        Rc::new(MockInformation {
                            state,
                            value: RefCell::new(Some(value)),
                        })
                    })
                } else {
                    Some(Rc::new(MockInformation {
                        state,
                        value: RefCell::new(None),
                    }))
                }
            };

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information1 = registry.insert(state, 1, &constructor);
        assert!(information1.is_some());
        let (information1, _) = information1.unwrap();
        *information1.value.borrow_mut() = Some(10);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information2 = registry.insert(state, 0, &constructor);
        assert!(information2.is_some());
        let (information2, dominated) = information2.unwrap();
        assert_eq!(
            information2.state.signature_variables,
            information1.state.signature_variables
        );
        assert_eq!(
            information2.state.resource_variables,
            information1.state.resource_variables
        );
        assert_eq!(*information2.value.borrow(), Some(10));
        assert_eq!(dominated, Some(information1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information3 = registry.insert(state, 0, &constructor);
        assert!(information3.is_some());
        let (information3, dominated) = information3.unwrap();
        assert_eq!(
            information3.state.signature_variables,
            information2.state.signature_variables
        );
        assert_ne!(
            information3.state.resource_variables,
            information2.state.resource_variables,
        );
        assert_eq!(*information3.value.borrow(), None);
        assert_eq!(dominated, Some(information2));
    }

    #[test]
    fn clear() {
        let model = generate_model();
        let mut registry = StateRegistry::new(&model);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let constructor = |state: StateInRegistry, _: Integer, _: Option<&Rc<MockInformation>>| {
            Some(Rc::new(MockInformation {
                state,
                value: RefCell::new(None),
            }))
        };
        let information = registry.insert(state, 1, &constructor);
        assert!(information.is_some());
        let (information, _) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information.state, state);

        registry.clear();

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let information = registry.insert(state, 1, &constructor);
        assert!(information.is_some());
        let (information, _) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information.state, state);
    }
}
