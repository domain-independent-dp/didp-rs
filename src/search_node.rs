use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::fmt::Display;
use std::rc::Rc;

use crate::state;

#[derive(Debug, Eq)]
pub struct SearchNode<T: Ord + Copy> {
    pub state: state::State,
    pub g: T,
    pub h: RefCell<Option<T>>,
    pub f: RefCell<Option<T>>,
    pub parent: Option<Rc<SearchNode<T>>>,
    pub closed: RefCell<bool>,
}

impl<T: Ord + Copy> Ord for SearchNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.f.cmp(&other.f).reverse()
    }
}

impl<T: Ord + Copy> PartialOrd for SearchNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord + Copy> PartialEq for SearchNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

pub struct SearchNodeRegistry<T: Ord + Copy>(
    collections::HashMap<Rc<state::SignatureVariables>, Vec<Rc<SearchNode<T>>>>,
);

impl<T: Ord + Copy> SearchNodeRegistry<T> {
    pub fn new() -> SearchNodeRegistry<T> {
        SearchNodeRegistry(collections::HashMap::new())
    }

    pub fn with_capcaity(capacity: usize) -> SearchNodeRegistry<T> {
        SearchNodeRegistry(collections::HashMap::with_capacity(capacity))
    }

    pub fn get_node(
        &mut self,
        mut state: state::State,
        g: T,
        parent: Option<Rc<SearchNode<T>>>,
    ) -> Option<Rc<SearchNode<T>>> {
        let entry = self.0.entry(state.signature_variables.clone());
        let v = match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                state.signature_variables = entry.key().clone();
                let v = entry.into_mut();
                for (i, other) in v.iter().enumerate() {
                    let result = other
                        .state
                        .resource_variables
                        .partial_cmp(&state.resource_variables);
                    match result {
                        Some(Ordering::Equal) | Some(Ordering::Greater) if g >= other.g => {
                            // dominated
                            return None;
                        }
                        Some(Ordering::Equal) | Some(Ordering::Less) if g <= other.g => {
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
                                g,
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
    use crate::variable;
    use std::ops;

    fn generate_signature_variables(
        integer_variables: Vec<variable::IntegerVariable>,
    ) -> Rc<state::SignatureVariables> {
        Rc::new(state::SignatureVariables {
            set_variables: Vec::new(),
            permutation_variables: Vec::new(),
            element_variables: Vec::new(),
            integer_variables,
            continuous_variables: Vec::new(),
        })
    }

    fn generate_resource_variables(
        integer_variables: Vec<variable::IntegerVariable>,
    ) -> state::ResourceVariables {
        state::ResourceVariables {
            integer_variables,
            continuous_variables: Vec::new(),
        }
    }

    fn generate_node<T: Ord + Copy + ops::Add>(
        signature_variables: Rc<state::SignatureVariables>,
        resource_variables: state::ResourceVariables,
        g: T,
        h: T,
        f: T,
    ) -> SearchNode<T> {
        SearchNode {
            state: state::State {
                signature_variables,
                resource_variables,
            },
            g,
            h: RefCell::new(Some(h)),
            f: RefCell::new(Some(f)),
            parent: None,
            closed: RefCell::new(false),
        }
    }

    #[test]
    fn search_node_eq() {
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let resource_variables = generate_resource_variables(vec![]);
        let node1 = generate_node(signature_variables, resource_variables, 1, 1, 2);
        let signature_variables = generate_signature_variables(vec![1, 2, 3]);
        let resource_variables = generate_resource_variables(vec![]);
        let node2 = generate_node(signature_variables, resource_variables, 1, 1, 2);
        assert_eq!(node1, node2);
    }

    #[test]
    fn search_node_neq() {
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let resource_variables = generate_resource_variables(vec![]);
        let node1 = generate_node(signature_variables, resource_variables, 1, 1, 2);
        let signature_variables = generate_signature_variables(vec![0, 1, 2]);
        let resource_variables = generate_resource_variables(vec![]);
        let node2 = generate_node(signature_variables, resource_variables, 1, 2, 3);
        assert!(node1 > node2);
    }

    #[test]
    fn get_new_node() {
        let mut registry = SearchNodeRegistry::new();

        let state = state::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2]),
        };
        let node = registry.get_node(state, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = state::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2]),
        };
        assert_eq!(node.state, state);
        assert_eq!(node.g, 1);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);

        let state = state::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2]),
        };
        let node = registry.get_node(state, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = state::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2]),
        };
        assert_eq!(node.state, state);
        assert_eq!(node.g, 1);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);

        let state = state::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1]),
        };
        let node = registry.get_node(state, 1, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = state::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1]),
        };
        assert_eq!(node.state, state);
        assert_eq!(node.g, 1);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert!(!*node.closed.borrow());

        let state = state::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1]),
        };
        let node = registry.get_node(state, 0, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = state::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1]),
        };
        assert_eq!(node.state, state);
        assert_eq!(node.g, 0);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert!(!*node.closed.borrow());
    }

    #[test]
    fn node_dominated() {
        let mut registry = SearchNodeRegistry::new();

        let state = state::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2]),
        };
        registry.get_node(state, 2, None);

        let state = state::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2]),
        };
        let node = registry.get_node(state, 2, None);
        assert!(node.is_none());

        let state = state::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2]),
        };
        let node = registry.get_node(state, 2, None);
        assert!(node.is_none());

        let state = state::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2]),
        };
        let node = registry.get_node(state, 3, None);
        assert!(node.is_none());
    }

    #[test]
    fn get_dominating_node() {
        let mut registry = SearchNodeRegistry::new();

        let state = state::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2]),
        };
        let node1 = registry.get_node(state, 2, None);
        assert!(node1.is_some());
        let node1 = node1.unwrap();
        *node1.h.borrow_mut() = Some(3);

        let state = state::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2]),
        };
        let node2 = registry.get_node(state, 1, Some(node1.clone()));
        assert!(node2.is_some());
        let node2 = node2.unwrap();
        assert_eq!(node2.state, node1.state);
        assert!(node2.g < node1.g);
        assert_eq!(*node2.h.borrow(), *node1.h.borrow());
        assert!(node2.f.borrow().is_none());
        assert!(*node1.closed.borrow());
        assert!(!*node2.closed.borrow());
        assert_ne!(node2.parent, node1.parent);

        let state = state::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3]),
        };
        let node3 = registry.get_node(state, 1, None);
        assert!(node3.is_some());
        let node3 = node3.unwrap();
        assert_eq!(
            node3.state.signature_variables,
            node2.state.signature_variables
        );
        assert_ne!(
            node3.state.resource_variables,
            node2.state.resource_variables
        );
        assert_eq!(node3.g, node2.g);
        assert!(node3.h.borrow().is_none());
        assert!(node3.f.borrow().is_none());
        assert!(*node2.closed.borrow());
        assert!(!*node3.closed.borrow());
        assert!(node3.parent.is_none());
    }
}
