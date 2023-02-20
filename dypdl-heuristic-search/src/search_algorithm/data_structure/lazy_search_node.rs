use super::hashable_state::HashableSignatureVariables;
use super::state_registry::{StateInRegistry, StateInformation};
use super::transition_chain::TransitionChain;
use dypdl::variable_type::Numeric;
use std::cmp::Ordering;
use std::rc::Rc;

/// Search node for lazy search, where a state is generated when it is expanded.
///
/// Nodes are totally ordered by their costs.
#[derive(Debug, Default)]
pub struct LazySearchNode<T: Numeric> {
    /// State.
    pub state: StateInRegistry<Rc<HashableSignatureVariables>>,
    /// Parent node.
    pub cost: T,
    /// Transitions to reach this node.
    pub transitions: Option<Rc<TransitionChain>>,
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

impl<T: Numeric> StateInformation<T, Rc<HashableSignatureVariables>> for LazySearchNode<T> {
    #[inline]
    fn state(&self) -> &StateInRegistry<Rc<HashableSignatureVariables>> {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.cost
    }
}

#[cfg(test)]
mod tests {
    use super::super::hashable_state::HashableSignatureVariables;
    use super::*;

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
            transitions: None,
        });
        assert_eq!(node.state(), &node.state);
        assert_eq!(node.cost(), 0);
        assert_eq!(node.transitions, None);
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
            transitions: None,
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
            transitions: None,
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
            transitions: None,
        };
        assert!(node1 < node2)
    }
}
