use crate::search_node::DPSearchNode;
use crate::state_registry::{StateInRegistry, StateInformation};
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use didp_parser::variable::Numeric;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::rc::Rc;

#[derive(Debug, Default)]
pub struct BeamSearchNode<T: Numeric, U: Numeric> {
    pub g: U,
    pub f: U,
    pub state: StateInRegistry,
    pub cost: T,
    pub operator: Option<Rc<TransitionWithCustomCost<T, U>>>,
    pub parent: Option<Rc<BeamSearchNode<T, U>>>,
    pub closed: RefCell<bool>,
}

impl<T: Numeric, U: Numeric + PartialOrd> PartialEq for BeamSearchNode<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

impl<T: Numeric, U: Numeric + Ord> Eq for BeamSearchNode<T, U> {}

impl<T: Numeric, U: Numeric + Ord> Ord for BeamSearchNode<T, U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.f.cmp(&other.f)
    }
}

impl<T: Numeric, U: Numeric + Ord> PartialOrd for BeamSearchNode<T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric, U: Numeric> StateInformation<T> for Rc<BeamSearchNode<T, U>> {
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    fn cost(&self) -> T {
        self.cost
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

impl<T: Numeric, U: Numeric> DPSearchNode<T> for Rc<BeamSearchNode<T, U>> {
    fn parent(&self) -> Option<Self> {
        self.parent.as_ref().cloned()
    }

    fn operator(&self) -> Option<Rc<didp_parser::Transition<T>>> {
        self.operator
            .as_ref()
            .map(|operator| Rc::new(operator.transition.clone()))
    }
}

pub struct Beam<T: Numeric, U: Numeric + Ord> {
    beam_size: usize,
    queue: collections::BinaryHeap<Rc<BeamSearchNode<T, U>>>,
}

impl<T: Numeric, U: Numeric + Ord> Beam<T, U> {
    pub fn new(beam_size: usize) -> Beam<T, U> {
        Beam {
            beam_size,
            queue: collections::BinaryHeap::with_capacity(beam_size),
        }
    }

    pub fn is_empty(&mut self) -> bool {
        let mut peek = self.queue.peek();
        while peek.map_or(false, |peek| *peek.closed.borrow()) {
            self.queue.pop();
            peek = self.queue.peek();
        }
        self.queue.is_empty()
    }

    pub fn drain(&mut self) -> NodesInBeam<'_, T, U> {
        NodesInBeam(self.queue.drain())
    }

    pub fn is_eligible(&mut self, f: U) -> bool {
        let mut peek = self.queue.peek();
        while peek.map_or(false, |peek| *peek.closed.borrow()) {
            self.queue.pop();
            peek = self.queue.peek();
        }
        self.queue.len() < self.beam_size || peek.map_or(true, |node| f < node.f)
    }

    pub fn push(&mut self, node: Rc<BeamSearchNode<T, U>>) {
        self.queue.push(node)
    }
}

pub struct NodesInBeam<'a, T: Numeric, U: Numeric>(
    collections::binary_heap::Drain<'a, Rc<BeamSearchNode<T, U>>>,
);

impl<'a, T: Numeric, U: Numeric> Iterator for NodesInBeam<'a, T, U> {
    type Item = Rc<BeamSearchNode<T, U>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some(node) if *node.closed.borrow() => self.next(),
            node => node,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashable_state::HashableSignatureVariables;

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
            closed: RefCell::new(false),
            parent: None,
            operator: None,
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
            closed: RefCell::new(false),
            parent: None,
            operator: None,
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
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        };
        assert!(node1 < node2);
    }

    #[test]
    fn beam_is_empty() {
        let mut beam = Beam::new(2);
        assert!(beam.is_empty());
        let node = Rc::new(BeamSearchNode {
            cost: 0,
            f: 1,
            ..Default::default()
        });
        beam.push(node);
        assert!(!beam.is_empty());
    }

    #[test]
    fn beam_is_eligible() {
        let mut beam = Beam::new(2);
        assert!(beam.is_eligible(1));
        let node = Rc::new(BeamSearchNode {
            cost: 0,
            f: 1,
            ..Default::default()
        });
        beam.push(node);
        assert!(beam.is_eligible(1));
        let node = Rc::new(BeamSearchNode {
            cost: 0,
            f: 1,
            ..Default::default()
        });
        beam.push(node);
        assert!(!beam.is_eligible(1));
        assert!(beam.is_eligible(0));

        let mut beam = Beam::new(2);
        let node = Rc::new(BeamSearchNode {
            cost: 0,
            f: 1,
            ..Default::default()
        });
        beam.push(node);
        let node = Rc::new(BeamSearchNode {
            cost: 0,
            f: 2,
            closed: RefCell::new(true),
            ..Default::default()
        });
        beam.push(node);
        assert!(beam.is_eligible(2));
    }

    #[test]
    fn beam_drain() {
        let mut beam = Beam::new(2);
        assert!(beam.is_eligible(1));
        let node = Rc::new(BeamSearchNode {
            cost: 0,
            f: 1,
            ..Default::default()
        });
        beam.push(node);
        let node = Rc::new(BeamSearchNode {
            cost: 1,
            f: 2,
            closed: RefCell::new(true),
            ..Default::default()
        });
        beam.push(node);
        let node = Rc::new(BeamSearchNode {
            cost: 0,
            f: 1,
            ..Default::default()
        });
        beam.push(node);
        let mut iter = beam.drain();
        assert_eq!(
            iter.next(),
            Some(Rc::new(BeamSearchNode {
                cost: 0,
                f: 1,
                ..Default::default()
            }))
        );
        assert_eq!(
            iter.next(),
            Some(Rc::new(BeamSearchNode {
                cost: 0,
                f: 1,
                ..Default::default()
            }))
        );
        assert_eq!(iter.next(), None);
    }
}
