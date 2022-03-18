use crate::priority_queue;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::rc::Rc;

use didp_parser::variable;

#[derive(Debug, Eq)]
pub struct SearchNode<T: variable::Numeric> {
    pub state: didp_parser::State<T>,
    pub h: RefCell<Option<T>>,
    pub f: RefCell<Option<T>>,
    pub parent: Option<Rc<SearchNode<T>>>,
    pub closed: RefCell<bool>,
}

impl<T: variable::Numeric> Ord for SearchNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.f.cmp(&other.f)
    }
}

impl<T: variable::Numeric> PartialOrd for SearchNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: variable::Numeric> PartialEq for SearchNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

pub type OpenList<T> = priority_queue::PriorityQueue<Rc<SearchNode<T>>>;

#[derive(Default)]
pub struct SearchNodeRegistry<T: variable::Numeric>(
    collections::HashMap<Rc<didp_parser::SignatureVariables<T>>, Vec<Rc<SearchNode<T>>>>,
);

impl<T: variable::Numeric> SearchNodeRegistry<T> {
    pub fn with_capcaity(capacity: usize) -> SearchNodeRegistry<T> {
        SearchNodeRegistry(collections::HashMap::with_capacity(capacity))
    }

    pub fn get_node(
        &mut self,
        mut state: didp_parser::State<T>,
        parent: Option<Rc<SearchNode<T>>>,
        metadata: &didp_parser::StateMetadata,
    ) -> Option<Rc<SearchNode<T>>> {
        let entry = self.0.entry(state.signature_variables.clone());
        let v = match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                state.signature_variables = entry.key().clone();
                let v = entry.into_mut();
                for (i, other) in v.iter().enumerate() {
                    let result = metadata
                        .dominance(&state.resource_variables, &other.state.resource_variables);
                    match result {
                        Some(Ordering::Equal) | Some(Ordering::Less)
                            if (metadata.maximize && state.cost <= other.state.cost)
                                || (!metadata.maximize && state.cost >= other.state.cost) =>
                        {
                            // dominated
                            return None;
                        }
                        Some(Ordering::Equal) | Some(Ordering::Greater)
                            if (metadata.maximize && state.cost >= other.state.cost)
                                || (!metadata.maximize && state.cost <= other.state.cost) =>
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
                                h,
                                f: RefCell::new(None),
                                parent,
                                closed: RefCell::new(false),
                            });
                            v[i] = node.clone();
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
    use didp_parser::variable;

    fn generate_state_metadata() -> didp_parser::StateMetadata {
        let mut name_to_numeric_variable = collections::HashMap::new();
        name_to_numeric_variable.insert("n1".to_string(), 0);
        name_to_numeric_variable.insert("n2".to_string(), 1);
        name_to_numeric_variable.insert("n3".to_string(), 2);

        let mut name_to_resource_variable = collections::HashMap::new();
        name_to_resource_variable.insert("r1".to_string(), 0);
        name_to_resource_variable.insert("r2".to_string(), 1);
        name_to_resource_variable.insert("r3".to_string(), 2);

        didp_parser::StateMetadata {
            maximize: false,
            numeric_variable_names: vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
            name_to_numeric_variable,
            resource_variable_names: vec!["r1".to_string(), "r2".to_string(), "r3".to_string()],
            name_to_resource_variable,
            less_is_better: vec![false, false, true],
            ..Default::default()
        }
    }

    fn generate_signature_variables(
        numeric_variables: Vec<variable::IntegerVariable>,
    ) -> Rc<didp_parser::SignatureVariables<variable::IntegerVariable>> {
        Rc::new(didp_parser::SignatureVariables {
            numeric_variables,
            ..Default::default()
        })
    }

    fn generate_node(
        signature_variables: Rc<didp_parser::SignatureVariables<variable::IntegerVariable>>,
        resource_variables: Vec<variable::IntegerVariable>,
        cost: variable::IntegerVariable,
        h: variable::IntegerVariable,
        f: variable::IntegerVariable,
    ) -> SearchNode<variable::IntegerVariable> {
        SearchNode {
            state: didp_parser::State {
                signature_variables,
                resource_variables,
                stage: 0,
                cost,
            },
            h: RefCell::new(Some(h)),
            f: RefCell::new(Some(f)),
            parent: None,
            closed: RefCell::new(false),
        }
    }

    #[test]
    fn search_node_eq() {
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let node1 = generate_node(signature_variables, vec![0, 0, 0], 1, 1, 2);
        let signature_variables = generate_signature_variables(vec![1, 2, 3]);
        let node2 = generate_node(signature_variables, vec![0, 0, 0], 1, 1, 2);
        assert_eq!(node1, node2);
    }

    #[test]
    fn search_node_neq() {
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let node1 = generate_node(signature_variables, vec![0, 0, 0], 1, 1, 2);
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let node2 = generate_node(signature_variables, vec![0, 0, 0], 1, 2, 3);
        assert!(node1 < node2);
    }

    #[test]
    fn get_new_node() {
        let metadata = generate_state_metadata();
        let mut registry = SearchNodeRegistry::default();

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 1,
        };
        let node = registry.get_node(state, None, &metadata);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 1,
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 1,
        };
        let node = registry.get_node(state, None, &metadata);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 1,
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: vec![3, 1, 3],
            stage: 0,
            cost: 1,
        };
        let node = registry.get_node(state, None, &metadata);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: vec![3, 1, 3],
            stage: 0,
            cost: 1,
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert!(!*node.closed.borrow());

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: vec![0, 1, 3],
            stage: 0,
            cost: 0,
        };
        let node = registry.get_node(state, None, &metadata);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: vec![0, 1, 3],
            stage: 0,
            cost: 0,
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert!(!*node.closed.borrow());
    }

    #[test]
    fn node_dominated() {
        let metadata = generate_state_metadata();
        let mut registry = SearchNodeRegistry::default();

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 2,
        };
        registry.get_node(state, None, &metadata);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 2,
        };
        let node = registry.get_node(state, None, &metadata);
        assert!(node.is_none());

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![0, 2, 3],
            stage: 0,
            cost: 2,
        };
        let node = registry.get_node(state, None, &metadata);
        assert!(node.is_none());

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 3,
        };
        let node = registry.get_node(state, None, &metadata);
        assert!(node.is_none());
    }

    #[test]
    fn node_dead_end() {
        let metadata = generate_state_metadata();
        let mut registry = SearchNodeRegistry::default();

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 2,
        };
        let node = registry.get_node(state, None, &metadata);
        assert!(node.is_some());
        let node = node.unwrap();
        assert!(node.h.borrow().is_none());

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 1,
        };
        let node = registry.get_node(state, None, &metadata);
        assert!(node.is_none());
    }

    #[test]
    fn get_dominating_node() {
        let metadata = generate_state_metadata();
        let mut registry = SearchNodeRegistry::default();

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![1, 2, 3],
            stage: 0,
            cost: 2,
        };
        let node1 = registry.get_node(state, None, &metadata);
        assert!(node1.is_some());
        let node1 = node1.unwrap();
        *node1.h.borrow_mut() = Some(3);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![1, 2, 3],
            stage: 1,
            cost: 1,
        };
        let node2 = registry.get_node(state, Some(node1.clone()), &metadata);
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
        assert!(node2.state.cost < node1.state.cost);
        assert_eq!(*node2.h.borrow(), *node1.h.borrow());
        assert!(node2.f.borrow().is_none());
        assert!(*node1.closed.borrow());
        assert!(!*node2.closed.borrow());
        assert_ne!(node2.parent, node1.parent);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: vec![2, 3, 3],
            stage: 0,
            cost: 1,
        };
        let node3 = registry.get_node(state, None, &metadata);
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
        assert_eq!(node3.state.cost, node2.state.cost);
        assert!(node3.h.borrow().is_none());
        assert!(node3.f.borrow().is_none());
        assert!(*node2.closed.borrow());
        assert!(!*node3.closed.borrow());
        assert!(node3.parent.is_none());
    }
}
