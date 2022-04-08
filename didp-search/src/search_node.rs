use crate::hashable_state;
use crate::priority_queue;
use didp_parser::variable::{Continuous, Element, Integer, Numeric, Set, Vector};
use didp_parser::ReduceFunction;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct StateForSearchNode {
    pub signature_variables: Rc<hashable_state::HashableSignatureVariables>,
    pub resource_variables: didp_parser::ResourceVariables,
}

impl StateForSearchNode {
    pub fn new(state: &didp_parser::State) -> StateForSearchNode {
        StateForSearchNode {
            signature_variables: Rc::new(hashable_state::HashableSignatureVariables::new(
                &state.signature_variables,
            )),
            resource_variables: state.resource_variables.clone(),
        }
    }
}

impl didp_parser::DPState for StateForSearchNode {
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

        let mut integer_resource_variables = self.resource_variables.integer_variables.clone();
        for e in &effect.integer_resource_effects {
            integer_resource_variables[e.0] = e.1.eval(self, registry);
        }

        let mut continuous_resource_variables =
            self.resource_variables.continuous_variables.clone();
        for e in &effect.continuous_resource_effects {
            continuous_resource_variables[e.0] = e.1.eval(self, registry);
        }

        StateForSearchNode {
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
        *self = self.apply_effect(effect, registry);
    }
}

#[derive(Debug)]
pub struct SearchNode<T: Numeric> {
    pub state: StateForSearchNode,
    pub operator: Option<Rc<didp_parser::Transition<T>>>,
    pub parent: Option<Rc<SearchNode<T>>>,
    pub g: T,
    pub h: RefCell<Option<T>>,
    pub f: RefCell<Option<T>>,
    pub closed: RefCell<bool>,
}

impl<T: Numeric + PartialOrd> PartialEq for SearchNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

impl<T: Numeric + Ord> Eq for SearchNode<T> {}

impl<T: Numeric + Ord> Ord for SearchNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.f.cmp(&other.f)
    }
}

impl<T: Numeric + Ord> PartialOrd for SearchNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric> SearchNode<T> {
    pub fn trace_transitions(
        &self,
        base_cost: T,
        model: &didp_parser::Model<T>,
    ) -> (T, Vec<Rc<didp_parser::Transition<T>>>) {
        let mut result = Vec::new();
        let mut cost = base_cost;
        if let (Some(mut node), Some(operator)) = (self.parent.as_ref(), self.operator.as_ref()) {
            cost = operator.eval_cost(cost, &node.state, &model.table_registry);
            result.push(operator.clone());
            while let (Some(parent), Some(operator)) =
                (node.parent.as_ref(), node.operator.as_ref())
            {
                cost = operator.eval_cost(cost, &parent.state, &model.table_registry);
                result.push(operator.clone());
                node = parent;
            }
            result.reverse();
        }
        (cost, result)
    }
}

pub type OpenList<T> = priority_queue::PriorityQueue<Rc<SearchNode<T>>>;

pub struct SearchNodeRegistry<'a, T: Numeric> {
    registry: FxHashMap<Rc<hashable_state::HashableSignatureVariables>, Vec<Rc<SearchNode<T>>>>,
    metadata: &'a didp_parser::StateMetadata,
    reduce_function: &'a ReduceFunction,
}

impl<'a, T: Numeric + Ord> SearchNodeRegistry<'a, T> {
    pub fn new(model: &'a didp_parser::Model<T>) -> SearchNodeRegistry<T> {
        SearchNodeRegistry {
            registry: FxHashMap::default(),
            metadata: &model.state_metadata,
            reduce_function: &model.reduce_function,
        }
    }

    pub fn reserve(&mut self, capacity: usize) {
        self.registry.reserve(capacity);
    }

    pub fn clear(&mut self) {
        self.registry.clear();
    }

    pub fn get_node(
        &mut self,
        mut state: StateForSearchNode,
        g: T,
        operator: Option<Rc<didp_parser::Transition<T>>>,
        parent: Option<Rc<SearchNode<T>>>,
    ) -> Option<Rc<SearchNode<T>>> {
        let entry = self.registry.entry(state.signature_variables.clone());
        let v = match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                state.signature_variables = entry.key().clone();
                let v = entry.into_mut();
                for other in v.iter_mut() {
                    let result = self.metadata.dominance(&state, &other.state);
                    match result {
                        Some(Ordering::Equal) | Some(Ordering::Less)
                            if (*self.reduce_function == ReduceFunction::Max && g <= other.g)
                                || (*self.reduce_function == ReduceFunction::Min
                                    && g >= other.g) =>
                        {
                            // dominated
                            return None;
                        }
                        Some(Ordering::Equal) | Some(Ordering::Greater)
                            if (*self.reduce_function == ReduceFunction::Max && g >= other.g)
                                || (*self.reduce_function == ReduceFunction::Min
                                    && g <= other.g) =>
                        {
                            // dominating
                            if !*other.closed.borrow() {
                                *other.closed.borrow_mut() = true;
                            }
                            let h = match result.unwrap() {
                                Ordering::Equal => {
                                    if let Some(h) = *other.h.borrow() {
                                        // cached value
                                        RefCell::new(Some(h))
                                    } else {
                                        // dead end
                                        return None;
                                    }
                                }
                                _ => RefCell::new(None),
                            };
                            let node = Rc::new(SearchNode {
                                state,
                                operator,
                                parent,
                                g,
                                h,
                                f: RefCell::new(None),
                                closed: RefCell::new(false),
                            });
                            *other = node.clone();
                            return Some(node);
                        }
                        _ => {}
                    }
                }
                v
            }
            collections::hash_map::Entry::Vacant(entry) => entry.insert(Vec::with_capacity(1)),
        };
        let node = Rc::new(SearchNode {
            state,
            operator,
            g,
            h: RefCell::new(None),
            f: RefCell::new(None),
            parent,
            closed: RefCell::new(false),
        });
        v.push(node.clone());
        Some(node)
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

    fn generate_node(
        signature_variables: Rc<hashable_state::HashableSignatureVariables>,
        integer_resource_variables: Vec<Integer>,
        parent: Option<Rc<SearchNode<Integer>>>,
        operator: Option<Rc<didp_parser::Transition<Integer>>>,
        g: Integer,
        h: Integer,
        f: Integer,
    ) -> SearchNode<Integer> {
        SearchNode {
            state: StateForSearchNode {
                signature_variables,
                resource_variables: didp_parser::ResourceVariables {
                    integer_variables: integer_resource_variables,
                    ..Default::default()
                },
            },
            operator,
            parent,
            g,
            h: RefCell::new(Some(h)),
            f: RefCell::new(Some(f)),
            closed: RefCell::new(false),
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
        let state = StateForSearchNode {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1.clone(), set2.clone()],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
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
        assert_eq!(state.get_integer_resource_variable(0), 4);
        assert_eq!(state.get_integer_resource_variable(1), 5);
        assert_eq!(state.get_integer_resource_variable(2), 6);
        assert_eq!(state.get_continuous_resource_variable(0), 4.0);
        assert_eq!(state.get_continuous_resource_variable(1), 5.0);
        assert_eq!(state.get_continuous_resource_variable(2), 6.0);
        let state = StateForSearchNode::new(&didp_parser::State {
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
        let state = StateForSearchNode {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
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
        let expected = StateForSearchNode {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![OrderedFloat(0.0), OrderedFloat(4.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
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
        let mut state = StateForSearchNode {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
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
        let expected = StateForSearchNode {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![OrderedFloat(0.0), OrderedFloat(4.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
        };
        state.apply_effect_in_place(&effect, &registry);
        assert_eq!(state, expected);
    }

    #[test]
    fn search_node_eq() {
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let node1 = generate_node(signature_variables, vec![0, 0, 0], None, None, 1, 1, 2);
        let signature_variables = generate_signature_variables(vec![1, 2, 3]);
        let node2 = generate_node(signature_variables, vec![0, 0, 0], None, None, 1, 1, 2);
        assert_eq!(node1, node2);
    }

    #[test]
    fn search_node_neq() {
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let node1 = generate_node(signature_variables, vec![0, 0, 0], None, None, 1, 1, 2);
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let node2 = generate_node(signature_variables, vec![0, 0, 0], None, None, 1, 2, 3);
        assert!(node1 < node2);
    }

    #[test]
    fn trace_transitions() {
        let model = generate_model();
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let node1 = Rc::new(generate_node(
            signature_variables,
            vec![0, 0, 0],
            None,
            None,
            0,
            0,
            0,
        ));
        assert_eq!(node1.trace_transitions(0, &model), (0, Vec::new()));
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let op1 = Rc::new(didp_parser::Transition {
            name: String::from("op1"),
            cost: didp_parser::expression::NumericExpression::NumericOperation(
                didp_parser::expression::NumericOperator::Add,
                Box::new(didp_parser::expression::NumericExpression::Cost),
                Box::new(didp_parser::expression::NumericExpression::Constant(1)),
            ),
            ..Default::default()
        });
        let node2 = Rc::new(generate_node(
            signature_variables,
            vec![0, 0, 0],
            None,
            Some(op1.clone()),
            0,
            0,
            0,
        ));
        assert_eq!(node2.trace_transitions(0, &model), (0, Vec::new()));
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let node2 = Rc::new(generate_node(
            signature_variables,
            vec![0, 0, 0],
            Some(node1.clone()),
            None,
            0,
            0,
            0,
        ));
        assert_eq!(node2.trace_transitions(0, &model), (0, Vec::new()));
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let node2 = Rc::new(generate_node(
            signature_variables,
            vec![0, 0, 0],
            Some(node1),
            Some(op1.clone()),
            0,
            0,
            0,
        ));
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let op2 = Rc::new(didp_parser::Transition {
            name: String::from("op2"),
            cost: didp_parser::expression::NumericExpression::NumericOperation(
                didp_parser::expression::NumericOperator::Add,
                Box::new(didp_parser::expression::NumericExpression::Cost),
                Box::new(didp_parser::expression::NumericExpression::Constant(1)),
            ),
            ..Default::default()
        });
        let node3 = Rc::new(generate_node(
            signature_variables,
            vec![0, 0, 0],
            Some(node2),
            Some(op2.clone()),
            0,
            0,
            0,
        ));
        assert_eq!(node3.trace_transitions(0, &model), (2, vec![op1, op2]));
    }

    #[test]
    fn get_new_node() {
        let model = generate_model();
        let mut registry = SearchNodeRegistry::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert!(!*node.closed.borrow());

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        let node = registry.get_node(state, 0, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert!(!*node.closed.borrow());
    }

    #[test]
    fn node_dominated() {
        let model = generate_model();
        let mut registry = SearchNodeRegistry::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        registry.get_node(state, 2, None, None);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 2, None, None);
        assert!(node.is_none());

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        let node = registry.get_node(state, 2, None, None);
        assert!(node.is_none());

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 3, None, None);
        assert!(node.is_none());
    }

    #[test]
    fn node_dead_end() {
        let model = generate_model();
        let mut registry = SearchNodeRegistry::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 2, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert!(node.h.borrow().is_none());

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_none());
    }

    #[test]
    fn get_dominating_node() {
        let model = generate_model();
        let mut registry = SearchNodeRegistry::<Integer>::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node1 = registry.get_node(state, 2, None, None);
        assert!(node1.is_some());
        let node1 = node1.unwrap();
        *node1.h.borrow_mut() = Some(3);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let op = Rc::new(didp_parser::Transition::default());
        let node2 = registry.get_node(state, 1, Some(op), Some(node1.clone()));
        assert!(node2.is_some());
        let node2 = node2.unwrap();
        assert_eq!(
            node2.state.signature_variables,
            node1.state.signature_variables
        );
        assert_eq!(
            node2.state.resource_variables,
            node1.state.resource_variables
        );
        assert!(node2.g < node1.g);
        assert_eq!(*node2.h.borrow(), *node1.h.borrow());
        assert!(node2.f.borrow().is_none());
        assert!(*node1.closed.borrow());
        assert!(!*node2.closed.borrow());
        assert_ne!(node2.parent, node1.parent);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        let node3 = registry.get_node(state, 1, None, None);
        assert!(node3.is_some());
        let node3 = node3.unwrap();
        assert_eq!(
            node3.state.signature_variables,
            node2.state.signature_variables
        );
        assert_ne!(
            node3.state.resource_variables,
            node2.state.resource_variables,
        );
        assert_eq!(node3.g, node2.g);
        assert!(node3.h.borrow().is_none());
        assert!(node3.f.borrow().is_none());
        assert!(*node2.closed.borrow());
        assert!(!*node3.closed.borrow());
        assert!(node3.parent.is_none());
    }

    #[test]
    fn clear() {
        let model = generate_model();
        let mut registry = SearchNodeRegistry::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);

        registry.clear();

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);
    }
}
