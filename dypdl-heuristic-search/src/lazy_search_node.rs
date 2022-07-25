use crate::search_node::DPSearchNode;
use crate::state_registry::{StateInRegistry, StateInformation};
use dypdl::variable_type::Numeric;
use std::cmp::Ordering;
use std::rc::Rc;

/// Search node for lazy search, where a state is generated when it is expanded.
///
/// Nodes are totally ordered by their costs.
#[derive(Debug, Default)]
pub struct LazySearchNode<T: Numeric> {
    /// State.
    pub state: StateInRegistry,
    /// Transition to reach this node.
    pub operator: Option<Rc<dypdl::Transition>>,
    /// Parent node.
    pub parent: Option<Rc<LazySearchNode<T>>>,
    /// Cost to reach this node.
    pub cost: T,
}

impl<T: Numeric + PartialOrd> PartialEq for LazySearchNode<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl<T: Numeric + Ord> Eq for LazySearchNode<T> {}

impl<T: Numeric + Ord> Ord for LazySearchNode<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl<T: Numeric + Ord> PartialOrd for LazySearchNode<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric> StateInformation<T> for Rc<LazySearchNode<T>> {
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.cost
    }
}

impl<T: Numeric> DPSearchNode<T> for Rc<LazySearchNode<T>> {
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
        let node = Rc::new(LazySearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 0,
            parent: None,
            operator: None,
        });
        assert_eq!(node.state(), &node.state);
        assert_eq!(node.cost(), 0);
        assert!(node.parent().is_none());
        assert!(node.operator().is_none());
    }

    #[test]
    fn search_node_cmp() {
        let node1 = LazySearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 0,
            parent: None,
            operator: None,
        };
        let node2 = LazySearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 0,
            parent: None,
            operator: None,
        };
        assert_eq!(node1, node2);
        let node2 = LazySearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 2,
            parent: None,
            operator: None,
        };
        assert!(node1 < node2)
    }
}
