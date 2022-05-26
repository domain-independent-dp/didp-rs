use crate::hashable_state;
use crate::search_node::StateForSearchNode;
use crate::solver::ConfigErr;
use crate::successor_generator::{MaybeApplicable, SuccessorGenerator};
use didp_parser::expression_parser::ParseNumericExpression;
use didp_parser::variable::Numeric;
use didp_parser::ReduceFunction;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

pub struct BFSNode<T: Numeric> {
    pub state: StateForSearchNode,
    pub operator: Option<Rc<didp_parser::Transition<T>>>,
    pub parent: Option<Rc<BFSNode<T>>>,
    pub g: T,
    pub h: RefCell<Option<T>>,
    pub f: RefCell<Option<T>>,
    pub closed: RefCell<bool>,
}

impl<T: Numeric + PartialOrd> PartialEq for BFSNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

impl<T: Numeric + Ord> Eq for BFSNode<T> {}

impl<T: Numeric + Ord> Ord for BFSNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.f.cmp(&other.f)
    }
}

impl<T: Numeric + Ord> PartialOrd for BFSNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric> BFSNode<T> {
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

pub struct BFSNodeRegistry<'a, T: Numeric, U: Numeric> {
    registry: FxHashMap<Rc<hashable_state::HashableSignatureVariables>, Vec<Rc<BFSNode<T, U>>>>,
    metadata: &'a didp_parser::StateMetadata,
    reduce_function: &'a ReduceFunction,
}

impl<'a, T: Numeric, U: Numeric + Ord> BFSNodeRegistry<'a, T, U> {
    pub fn new(model: &'a didp_parser::Model<T>) -> BFSNodeRegistry<T, U> {
        BFSNodeRegistry {
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
        cost: T,
        g: U,
        operator: Option<Rc<TransitionWithG<T, U>>>,
        parent: Option<Rc<BFSNode<T, U>>>,
    ) -> Option<Rc<BFSNode<T, U>>> {
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
                            if (*self.reduce_function == ReduceFunction::Max
                                && cost <= other.cost)
                                || (*self.reduce_function == ReduceFunction::Min
                                    && cost >= other.cost) =>
                        {
                            // dominated
                            return None;
                        }
                        Some(Ordering::Equal) | Some(Ordering::Greater)
                            if (*self.reduce_function == ReduceFunction::Max
                                && cost >= other.cost)
                                || (*self.reduce_function == ReduceFunction::Min
                                    && cost <= other.cost) =>
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
                            let node = Rc::new(BFSNode {
                                state,
                                operator,
                                parent,
                                cost,
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
        let node = Rc::new(BFSNode {
            state,
            operator,
            cost,
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
    use didp_parser::variable::*;
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
        parent: Option<Rc<BFSNode<Integer, Integer>>>,
        operator: Option<Rc<TransitionWithG<Integer, Integer>>>,
        g: Integer,
        h: Integer,
        f: Integer,
    ) -> BFSNode<Integer, Integer> {
        BFSNode {
            state: StateForSearchNode {
                signature_variables,
                resource_variables: didp_parser::ResourceVariables {
                    integer_variables: integer_resource_variables,
                    ..Default::default()
                },
            },
            operator,
            parent,
            cost: g,
            g,
            h: RefCell::new(Some(h)),
            f: RefCell::new(Some(f)),
            closed: RefCell::new(false),
        }
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
        let op1 = Rc::new(TransitionWithG {
            transition: didp_parser::Transition {
                name: String::from("op1"),
                cost: didp_parser::expression::NumericExpression::NumericOperation(
                    didp_parser::expression::NumericOperator::Add,
                    Box::new(didp_parser::expression::NumericExpression::Cost),
                    Box::new(didp_parser::expression::NumericExpression::Constant(1)),
                ),
                ..Default::default()
            },
            g: didp_parser::expression::NumericExpression::NumericOperation(
                didp_parser::expression::NumericOperator::Add,
                Box::new(didp_parser::expression::NumericExpression::Cost),
                Box::new(didp_parser::expression::NumericExpression::Constant(1)),
            ),
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
        let op2 = Rc::new(TransitionWithG {
            transition: didp_parser::Transition {
                name: String::from("op2"),
                cost: didp_parser::expression::NumericExpression::NumericOperation(
                    didp_parser::expression::NumericOperator::Add,
                    Box::new(didp_parser::expression::NumericExpression::Cost),
                    Box::new(didp_parser::expression::NumericExpression::Constant(1)),
                ),
                ..Default::default()
            },
            g: didp_parser::expression::NumericExpression::NumericOperation(
                didp_parser::expression::NumericOperator::Add,
                Box::new(didp_parser::expression::NumericExpression::Cost),
                Box::new(didp_parser::expression::NumericExpression::Constant(1)),
            ),
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
        let mut registry = BFSNodeRegistry::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 1, 1, None, None);
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
        let node = registry.get_node(state, 1, 1, None, None);
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
        let node = registry.get_node(state, 1, 1, None, None);
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
        let node = registry.get_node(state, 0, 0, None, None);
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
        let mut registry = BFSNodeRegistry::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        registry.get_node(state, 2, 2, None, None);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 2, 2, None, None);
        assert!(node.is_none());

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        let node = registry.get_node(state, 2, 2, None, None);
        assert!(node.is_none());

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 3, 3, None, None);
        assert!(node.is_none());
    }

    #[test]
    fn node_dead_end() {
        let model = generate_model();
        let mut registry = BFSNodeRegistry::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 2, 2, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert!(node.h.borrow().is_none());

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 1, 1, None, None);
        assert!(node.is_none());
    }

    #[test]
    fn get_dominating_node() {
        let model = generate_model();
        let mut registry = BFSNodeRegistry::<Integer, Integer>::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node1 = registry.get_node(state, 2, 2, None, None);
        assert!(node1.is_some());
        let node1 = node1.unwrap();
        *node1.h.borrow_mut() = Some(3);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let op = Rc::new(TransitionWithG::default());
        let node2 = registry.get_node(state, 1, 1, Some(op), Some(node1.clone()));
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
        assert!(node2.cost < node1.cost);
        assert_eq!(*node2.h.borrow(), *node1.h.borrow());
        assert!(node2.f.borrow().is_none());
        assert!(*node1.closed.borrow());
        assert!(!*node2.closed.borrow());
        assert_ne!(node2.parent, node1.parent);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        let node3 = registry.get_node(state, 1, 1, None, None);
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
        assert_eq!(node3.cost, node2.cost);
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
        let mut registry = BFSNodeRegistry::new(&model);

        let state = StateForSearchNode {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let node = registry.get_node(state, 1, 1, None, None);
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
        let node = registry.get_node(state, 1, 1, None, None);
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
