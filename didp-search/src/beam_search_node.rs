use crate::search_node::DPSearchNode;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
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
    pub in_beam: RefCell<bool>,
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

#[derive(Debug, Default)]
pub struct BeamSearchNodeArgs<T: Numeric, U: Numeric> {
    pub g: U,
    pub f: U,
    pub operator: Option<Rc<TransitionWithCustomCost<T, U>>>,
    pub parent: Option<Rc<BeamSearchNode<T, U>>>,
}

pub struct Beam<T: Numeric, U: Numeric + Ord> {
    pub size: usize,
    beam_size: usize,
    queue: collections::BinaryHeap<Rc<BeamSearchNode<T, U>>>,
}

impl<T: Numeric, U: Numeric + Ord> Beam<T, U> {
    pub fn new(beam_size: usize) -> Beam<T, U> {
        Beam {
            size: 0,
            beam_size,
            queue: collections::BinaryHeap::with_capacity(beam_size),
        }
    }

    pub fn is_empty(&mut self) -> bool {
        self.queue.is_empty()
    }

    pub fn drain(&mut self) -> NodesInBeam<'_, T, U> {
        self.size = 0;
        NodesInBeam(self.queue.drain())
    }

    pub fn insert<'a>(
        &mut self,
        registry: &mut StateRegistry<'a, T, Rc<BeamSearchNode<T, U>>>,
        state: StateInRegistry,
        cost: T,
        args: BeamSearchNodeArgs<T, U>,
    ) {
        if self.size < self.beam_size || self.queue.peek().map_or(true, |node| args.f < node.f) {
            let constructor =
                |state: StateInRegistry, cost: T, _: Option<&Rc<BeamSearchNode<T, U>>>| {
                    Some(Rc::new(BeamSearchNode {
                        g: args.g,
                        f: args.f,
                        state,
                        cost,
                        operator: args.operator,
                        parent: args.parent,
                        in_beam: RefCell::new(true),
                    }))
                };
            if let Some((node, dominated)) = registry.insert(state, cost, constructor) {
                if let Some(dominated) = dominated {
                    if *dominated.in_beam.borrow() {
                        *dominated.in_beam.borrow_mut() = false;
                        self.size -= 1;
                    }
                }
                let mut peek = self.queue.peek();
                while peek.map_or(false, |node| !*node.in_beam.borrow()) {
                    self.queue.pop();
                    peek = self.queue.peek();
                }
                if self.size == self.beam_size {
                    if let Some(node) = self.queue.pop() {
                        *node.in_beam.borrow_mut() = false;
                        self.size -= 1;
                    }
                }
                let mut peek = self.queue.peek();
                while peek.map_or(false, |node| !*node.in_beam.borrow()) {
                    self.queue.pop();
                    peek = self.queue.peek();
                }
                self.queue.push(node);
                self.size += 1;
            }
        }
    }
}

pub struct NodesInBeam<'a, T: Numeric, U: Numeric>(
    collections::binary_heap::Drain<'a, Rc<BeamSearchNode<T, U>>>,
);

impl<'a, T: Numeric, U: Numeric> Iterator for NodesInBeam<'a, T, U> {
    type Item = Rc<BeamSearchNode<T, U>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some(node) if !*node.in_beam.borrow() => self.next(),
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
            in_beam: RefCell::new(false),
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
            in_beam: RefCell::new(false),
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
            in_beam: RefCell::new(false),
            parent: None,
            operator: None,
        };
        assert!(node1 < node2);
    }

    #[test]
    fn beam_is_empty() {
        let model = didp_parser::Model::default();
        let mut registry = StateRegistry::new(&model);
        let mut beam = Beam::new(2);
        assert!(beam.is_empty());
        let state = StateInRegistry::default();
        let cost = 0;
        let args = BeamSearchNodeArgs {
            g: 0,
            f: 1,
            ..Default::default()
        };
        beam.insert(&mut registry, state, cost, args);
        assert!(!beam.is_empty());
    }

    #[test]
    fn beam_drain() {
        let model = didp_parser::Model::default();
        let mut registry = StateRegistry::new(&model);
        let mut beam = Beam::new(1);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 1;
        let args = BeamSearchNodeArgs {
            g: 0,
            f: 2,
            ..Default::default()
        };
        beam.insert(&mut registry, state, cost, args);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 0;
        let args = BeamSearchNodeArgs {
            g: 0,
            f: 1,
            ..Default::default()
        };
        beam.insert(&mut registry, state, cost, args);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![2, 3, 4],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 0;
        let args = BeamSearchNodeArgs {
            g: 0,
            f: 2,
            ..Default::default()
        };
        beam.insert(&mut registry, state, cost, args);

        let mut iter = beam.drain();
        assert_eq!(
            iter.next(),
            Some(Rc::new(BeamSearchNode {
                state: StateInRegistry {
                    signature_variables: Rc::new(HashableSignatureVariables {
                        integer_variables: vec![2, 3, 4],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                cost: 0,
                g: 0,
                f: 1,
                operator: None,
                parent: None,
                in_beam: RefCell::new(true)
            }))
        );
        assert_eq!(iter.next(), None);
    }
}
