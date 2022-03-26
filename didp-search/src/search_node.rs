use crate::priority_queue;
use didp_parser::variable;
use didp_parser::ReduceFunction;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::rc::Rc;

#[derive(Debug)]
pub struct SearchNode<T> {
    pub state: didp_parser::State,
    pub operator: Option<usize>,
    pub parent: Option<Rc<SearchNode<T>>>,
    pub g: T,
    pub h: RefCell<Option<T>>,
    pub f: RefCell<Option<T>>,
    pub closed: RefCell<bool>,
}

impl<T: Ord> PartialEq for SearchNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

impl<T: Ord> Eq for SearchNode<T> {}

impl<T: Ord> Ord for SearchNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.f.cmp(&other.f)
    }
}

impl<T: Ord> PartialOrd for SearchNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub type OpenList<T> = priority_queue::PriorityQueue<Rc<SearchNode<T>>>;

pub struct SearchNodeRegistry<'a, T: PartialOrd> {
    registry: FxHashMap<Rc<didp_parser::SignatureVariables>, Vec<Rc<SearchNode<T>>>>,
    metadata: &'a didp_parser::StateMetadata,
    reduce_function: &'a ReduceFunction,
}

impl<'a, T: variable::Numeric + Ord> SearchNodeRegistry<'a, T> {
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

    pub fn get_node(
        &mut self,
        mut state: didp_parser::State,
        g: T,
        operator: Option<usize>,
        parent: Option<Rc<SearchNode<T>>>,
    ) -> Option<Rc<SearchNode<T>>> {
        let entry = self.registry.entry(state.signature_variables.clone());
        let v = match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                state.signature_variables = entry.key().clone();
                let v = entry.into_mut();
                for (i, other) in v.iter().enumerate() {
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
    use didp_parser::variable;

    fn generate_model() -> didp_parser::Model<variable::Integer> {
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
        integer_variables: Vec<variable::Integer>,
    ) -> Rc<didp_parser::SignatureVariables> {
        Rc::new(didp_parser::SignatureVariables {
            integer_variables,
            ..Default::default()
        })
    }

    fn generate_resource_variables(
        integer_variables: Vec<variable::Integer>,
    ) -> didp_parser::ResourceVariables {
        didp_parser::ResourceVariables {
            integer_variables,
            ..Default::default()
        }
    }

    fn generate_node(
        signature_variables: Rc<didp_parser::SignatureVariables>,
        integer_resource_variables: Vec<variable::Integer>,
        g: variable::Integer,
        h: variable::Integer,
        f: variable::Integer,
    ) -> SearchNode<variable::Integer> {
        SearchNode {
            state: didp_parser::State {
                signature_variables,
                resource_variables: didp_parser::ResourceVariables {
                    integer_variables: integer_resource_variables,
                    ..Default::default()
                },
                stage: 0,
            },
            operator: None,
            parent: None,
            g,
            h: RefCell::new(Some(h)),
            f: RefCell::new(Some(f)),
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
        let model = generate_model();
        let mut registry = SearchNodeRegistry::new(&model);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert_eq!(*node.closed.borrow(), false);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
            stage: 0,
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
            stage: 0,
        };
        assert_eq!(node.state, state);
        assert!(node.h.borrow().is_none());
        assert!(node.f.borrow().is_none());
        assert!(node.parent.is_none());
        assert!(!*node.closed.borrow());

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
            stage: 0,
        };
        let node = registry.get_node(state, 0, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
            stage: 0,
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

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        registry.get_node(state, 2, None, None);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        let node = registry.get_node(state, 2, None, None);
        assert!(node.is_none());

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
            stage: 0,
        };
        let node = registry.get_node(state, 2, None, None);
        assert!(node.is_none());

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        let node = registry.get_node(state, 3, None, None);
        assert!(node.is_none());
    }

    #[test]
    fn node_dead_end() {
        let model = generate_model();
        let mut registry = SearchNodeRegistry::new(&model);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        let node = registry.get_node(state, 2, None, None);
        assert!(node.is_some());
        let node = node.unwrap();
        assert!(node.h.borrow().is_none());

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        let node = registry.get_node(state, 1, None, None);
        assert!(node.is_none());
    }

    #[test]
    fn get_dominating_node() {
        let model = generate_model();
        let mut registry = SearchNodeRegistry::<variable::Integer>::new(&model);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 0,
        };
        let node1 = registry.get_node(state, 2, None, None);
        assert!(node1.is_some());
        let node1 = node1.unwrap();
        *node1.h.borrow_mut() = Some(3);

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
            stage: 1,
        };
        let node2 = registry.get_node(state, 1, Some(0), Some(node1.clone()));
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

        let state = didp_parser::State {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
            stage: 0,
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
}
