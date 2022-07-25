use crate::search_node::DPSearchNode;
use crate::state_registry::{StateInRegistry, StateInformation};
use dypdl::variable_type::Numeric;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

/// Node for best-first search.
///
/// Nodes are totally ordered by their f-values, and tie is broken by the h-values.
#[derive(Debug, Default)]
pub struct BFSNode<T: Numeric> {
    /// g-value.
    pub g: T,
    /// h-value.
    pub h: RefCell<Option<T>>,
    /// f-value.
    pub f: RefCell<Option<T>>,
    /// State.
    pub state: StateInRegistry,
    /// Transition applied to reach this node.
    pub operator: Option<Rc<dypdl::Transition>>,
    /// Parent node.
    pub parent: Option<Rc<BFSNode<T>>>,
    /// If already expanded.
    pub closed: RefCell<bool>,
}

impl<T: Numeric + PartialOrd> PartialEq for BFSNode<T> {
    #[inline]
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
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric> StateInformation<T> for Rc<BFSNode<T>> {
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.g
    }
}

impl<T: Numeric> DPSearchNode<T> for Rc<BFSNode<T>> {
    #[inline]
    fn parent(&self) -> Option<Self> {
        self.parent.as_ref().cloned()
    }

    #[inline]
    fn operator(&self) -> Option<Rc<dypdl::Transition>> {
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
            h: RefCell::new(Some(2)),
            f: RefCell::new(Some(3)),
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
            h: RefCell::new(Some(2)),
            f: RefCell::new(Some(3)),
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
            h: RefCell::new(Some(2)),
            f: RefCell::new(Some(3)),
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
            h: RefCell::new(Some(2)),
            f: RefCell::new(Some(4)),
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
            h: RefCell::new(Some(3)),
            f: RefCell::new(Some(3)),
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        };
        assert!(node1 < node2)
    }
}