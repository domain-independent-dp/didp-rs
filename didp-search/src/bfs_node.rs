use crate::search_node::DPSearchNode;
use crate::state_registry::{StateInRegistry, StateInformation};
use didp_parser::variable::Numeric;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

#[derive(Debug, Default)]
pub struct BFSNode<T: Numeric> {
    pub g: T,
    pub h: T,
    pub f: T,
    pub state: StateInRegistry,
    pub operator: Option<Rc<didp_parser::Transition<T>>>,
    pub parent: Option<Rc<BFSNode<T>>>,
    pub closed: RefCell<bool>,
}

impl<T: Numeric + PartialOrd> PartialEq for BFSNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.h == other.h
    }
}

impl<T: Numeric + Ord> Eq for BFSNode<T> {}

impl<T: Numeric + Ord> Ord for BFSNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => self.h.cmp(&other.h),
            result => result,
        }
    }
}

impl<T: Numeric + Ord> PartialOrd for BFSNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric> StateInformation<T> for Rc<BFSNode<T>> {
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    fn cost(&self) -> T {
        self.g
    }

    fn close(&self) -> bool {
        if *self.closed.borrow() {
            true
        } else {
            *self.closed.borrow_mut() = true;
            false
        }
    }
}

impl<T: Numeric> DPSearchNode<T> for Rc<BFSNode<T>> {
    fn parent(&self) -> Option<Self> {
        self.parent.as_ref().cloned()
    }

    fn operator(&self) -> Option<Rc<didp_parser::Transition<T>>> {
        self.operator.as_ref().cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashable_state::HashableSignatureVariables;

    #[test]
    fn search_node_getter() {
        let node = Rc::new(BFSNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        });
        assert_eq!(node.state(), &node.state);
        assert_eq!(node.cost(), 1);
        assert!(node.parent().is_none());
        assert!(node.operator().is_none());
    }

    #[test]
    fn search_node_close() {
        let node = Rc::new(BFSNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        });
        assert!(!node.close());
        assert!(*node.closed.borrow());
        assert!(node.close());
    }

    #[test]
    fn search_node_cmp() {
        let node1 = BFSNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        };
        let node2 = BFSNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        };
        assert_eq!(node1, node2);
        let node2 = BFSNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 2,
            h: 2,
            f: 4,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        };
        assert!(node1 < node2);
        let node2 = BFSNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 0,
            h: 3,
            f: 3,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        };
        assert!(node1 < node2)
    }
}
