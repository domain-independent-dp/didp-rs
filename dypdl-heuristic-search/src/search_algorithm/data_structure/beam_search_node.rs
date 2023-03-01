use super::beam::{InBeam, InformationInBeam};
use super::prioritized_node::PrioritizedNode;
use super::state_registry::{StateInRegistry, StateInformation};
use super::transition_chain::TransitionChainInterface;
use super::transition_with_custom_cost::{
    CustomCostNodeInterface, TransitionWithCustomCost, TransitionWithCustomCostChain,
};
use dypdl::variable_type::Numeric;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::rc::Rc;

/// Search node for beam search.
///
/// Nodes totally ordered by their f-values.
/// If the f-values are the same, the nodes are reversely ordered by their g-values.
#[derive(Debug, Default)]
pub struct BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
    /// g-value.
    pub g: U,
    /// f-value.
    pub f: U,
    /// State.
    pub state: StateInRegistry,
    /// Accumulated cost along the path so far.
    pub cost: T,
    /// If included in a beam.
    pub in_beam: RefCell<bool>,
    /// Transitions to reach this node.
    pub transitions: Option<Rc<TransitionWithCustomCostChain>>,
}

impl<T, U> CustomCostNodeInterface<T, U> for BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
    #[inline]
    fn new(
        g: U,
        f: U,
        state: StateInRegistry,
        cost: T,
        parent: Option<&Self>,
        transition: Option<Rc<TransitionWithCustomCost>>,
    ) -> BeamSearchNode<T, U> {
        let transitions = transition.map(|transition| {
            Rc::new(TransitionWithCustomCostChain::new(
                parent.and_then(|parent| parent.transitions.clone()),
                transition,
            ))
        });

        BeamSearchNode {
            g,
            f,
            state,
            cost,
            in_beam: RefCell::new(true),
            transitions,
        }
    }

    fn transitions(&self) -> Vec<TransitionWithCustomCost> {
        self.transitions
            .as_ref()
            .map_or_else(Vec::new, |transitions| transitions.transitions())
    }
}

impl<T, U> PartialEq for BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
    /// Nodes are compared by their f- and g-values.
    /// This does not mean that the nodes are the same.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.g == other.g
    }
}

impl<T, U> Eq for BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
}

impl<T, U> Ord for BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric + Ord,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => other.g.cmp(&self.g),
            result => result,
        }
    }
}

impl<T, U> PartialOrd for BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric + Ord,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, U> StateInformation<T> for BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.cost
    }
}

impl<T, U> PrioritizedNode<U> for BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
    #[inline]
    fn g(&self) -> U {
        self.g
    }

    #[inline]
    fn f(&self) -> U {
        self.f
    }
}

impl<T, U> InBeam for BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric,
{
    #[inline]
    fn in_beam(&self) -> bool {
        *self.in_beam.borrow()
    }

    #[inline]
    fn remove_from_beam(&self) {
        *self.in_beam.borrow_mut() = false;
    }
}

impl<T, U> InformationInBeam<T, U> for BeamSearchNode<T, U>
where
    T: Numeric,
    U: Numeric + Ord,
{
}

#[cfg(test)]
mod tests {
    use dypdl::Transition;

    use super::super::hashable_state::HashableSignatureVariables;
    use super::*;

    #[test]
    fn search_node_getter() {
        let node = Rc::new(BeamSearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 1,
            g: 1,
            f: 3,
            in_beam: RefCell::new(false),
            transitions: None,
        });
        assert_eq!(node.state(), &node.state);
        assert_eq!(node.cost(), 1);
        assert_eq!(
            CustomCostNodeInterface::<_, _>::transitions(&*node),
            Vec::new()
        );
        assert_eq!(node.g(), 1);
        assert_eq!(node.f(), 3);
        assert!(!node.in_beam());
    }

    #[test]
    fn remove_from_beam() {
        let node = BeamSearchNode::<i32, i32> {
            in_beam: RefCell::new(true),
            ..Default::default()
        };
        assert!(node.in_beam());
        node.remove_from_beam();
        assert!(!node.in_beam());
    }

    #[test]
    fn beam_search_node_new() {
        let g = 1;
        let f = 3;
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 2;
        let t1 = Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("t1"),
                ..Default::default()
            },
            ..Default::default()
        });
        let t2 = Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("t2"),
                ..Default::default()
            },
            ..Default::default()
        });
        let chain1 = Rc::new(TransitionWithCustomCostChain::new(None, t1.clone()));
        let parent = Rc::new(BeamSearchNode {
            transitions: Some(chain1.clone()),
            ..Default::default()
        });
        let node = BeamSearchNode::new(g, f, state.clone(), cost, Some(&parent), Some(t2.clone()));
        assert_eq!(node.state(), &state);
        assert_eq!(node.g(), 1);
        assert_eq!(node.f(), 3);
        assert_eq!(node.cost(), 2);
        assert_eq!(
            node.transitions,
            Some(Rc::new(TransitionWithCustomCostChain::new(
                Some(chain1),
                t2.clone()
            )))
        );
        assert_eq!(
            CustomCostNodeInterface::<_, _>::transitions(&node),
            vec![(*t1).clone(), (*t2).clone()]
        );
    }

    #[test]
    fn search_node_cmp() {
        let node1 = BeamSearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 1,
            g: 1,
            f: 3,
            in_beam: RefCell::new(false),
            transitions: None,
        };
        let node2 = BeamSearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 1,
            g: 1,
            f: 3,
            in_beam: RefCell::new(false),
            transitions: None,
        };
        assert_eq!(node1, node2);
        let node2 = BeamSearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 2,
            g: 2,
            f: 4,
            in_beam: RefCell::new(false),
            transitions: None,
        };
        assert!(node1 < node2);
    }
}
