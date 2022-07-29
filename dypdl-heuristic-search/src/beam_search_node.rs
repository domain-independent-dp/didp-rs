use crate::search_node::DPSearchNode;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type::Numeric;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::rc::Rc;

/// Search node for beam search.
///
/// Nodes totaly ordered by their f-values.
#[derive(Debug, Default)]
pub struct BeamSearchNode<T: Numeric, U: Numeric> {
    /// g-value.
    pub g: U,
    /// f-value.
    pub f: U,
    /// State.
    pub state: StateInRegistry,
    /// Accumulated cost along the path so far.
    pub cost: T,
    /// Transition applied to reach this node.
    pub operator: Option<Rc<TransitionWithCustomCost>>,
    /// Parent node.
    pub parent: Option<Rc<BeamSearchNode<T, U>>>,
    /// If included in a beam.
    pub in_beam: RefCell<bool>,
}

impl<T: Numeric, U: Numeric + PartialOrd> PartialEq for BeamSearchNode<T, U> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

impl<T: Numeric, U: Numeric + Ord> Eq for BeamSearchNode<T, U> {}

impl<T: Numeric, U: Numeric + Ord> Ord for BeamSearchNode<T, U> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.f.cmp(&other.f)
    }
}

impl<T: Numeric, U: Numeric + Ord> PartialOrd for BeamSearchNode<T, U> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric, U: Numeric> StateInformation<T> for Rc<BeamSearchNode<T, U>> {
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.cost
    }
}

impl<T: Numeric, U: Numeric> DPSearchNode<T> for Rc<BeamSearchNode<T, U>> {
    #[inline]
    fn parent(&self) -> Option<Self> {
        self.parent.as_ref().cloned()
    }

    fn operator(&self) -> Option<Rc<dypdl::Transition>> {
        self.operator
            .as_ref()
            .map(|operator| Rc::new(operator.transition.clone()))
    }
}

/// Parameters to create a beam search node.
#[derive(Debug, Default)]
pub struct BeamSearchNodeArgs<T: Numeric, U: Numeric> {
    pub g: U,
    pub f: U,
    pub operator: Option<Rc<TransitionWithCustomCost>>,
    pub parent: Option<Rc<BeamSearchNode<T, U>>>,
}

/// Common structure of beam.
#[derive(Debug, Clone)]
pub struct BeamBase<T: Numeric, U: Numeric + Ord> {
    /// Priority queue to store nodes.
    pub queue: collections::BinaryHeap<Rc<BeamSearchNode<T, U>>>,
    /// Vector to store nodes.
    pub pool: Vec<Rc<BeamSearchNode<T, U>>>,
}

impl<T: Numeric, U: Numeric + Ord> BeamBase<T, U> {
    /// Returns true if no state in beam and false otherwise.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty() && self.pool.is_empty()
    }

    /// Removes nodes from the beam, returning all removed nodes as an iterator.
    pub fn drain(&mut self) -> NodesInBeam<'_, T, U> {
        NodesInBeam {
            queue_iter: self.queue.drain(),
            pool_iter: self.pool.drain(..),
            pool_mode: false,
        }
    }
}

/// An draining iterator for `Beam<T, U>`
pub struct NodesInBeam<'a, T: Numeric, U: Numeric> {
    queue_iter: collections::binary_heap::Drain<'a, Rc<BeamSearchNode<T, U>>>,
    pool_iter: std::vec::Drain<'a, Rc<BeamSearchNode<T, U>>>,
    pool_mode: bool,
}

impl<'a, T: Numeric, U: Numeric> Iterator for NodesInBeam<'a, T, U> {
    type Item = Rc<BeamSearchNode<T, U>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pool_mode {
            match self.pool_iter.next() {
                Some(node) if !*node.in_beam.borrow() => self.next(),
                node => {
                    println!("node");
                    node
                }
            }
        } else {
            match self.queue_iter.next() {
                Some(node) if !*node.in_beam.borrow() => self.next(),
                None => {
                    self.pool_mode = true;
                    self.next()
                }
                node => {
                    println!("node");
                    node
                }
            }
        }
    }
}

pub trait Beam<T: Numeric, U: Numeric + Ord> {
    /// Returns true if no state in beam and false otherwise.
    fn is_empty(&self) -> bool;

    /// Returns the capacity of the beam.
    fn capacity(&self) -> usize;

    // Remove a node from the beam.
    fn pop(&mut self) -> Option<Rc<BeamSearchNode<T, U>>>;

    /// Removes nodes from the beam, returning all removed nodes as an iterator.
    fn drain(&mut self) -> NodesInBeam<'_, T, U>;

    /// Crate a node from a state and insert it into the beam.
    fn insert(
        &mut self,
        registry: &mut StateRegistry<'_, T, Rc<BeamSearchNode<T, U>>>,
        state: StateInRegistry,
        cost: T,
        args: BeamSearchNodeArgs<T, U>,
    );
}

/// Beam for beam search.
#[derive(Debug, Clone)]
pub struct NormalBeam<T: Numeric, U: Numeric + Ord> {
    /// Capacity of the beam, or the beam size.
    pub capacity: usize,
    size: usize,
    beam: BeamBase<T, U>,
}

impl<T: Numeric, U: Numeric + Ord> NormalBeam<T, U> {
    pub fn new(capacity: usize) -> NormalBeam<T, U> {
        NormalBeam {
            capacity,
            size: 0,
            beam: BeamBase {
                queue: collections::BinaryHeap::with_capacity(capacity),
                pool: vec![],
            },
        }
    }

    fn clean_garbage(&mut self) {
        let mut peek = self.beam.queue.peek();
        while peek.map_or(false, |node| !*node.in_beam.borrow()) {
            self.beam.queue.pop();
            peek = self.beam.queue.peek();
        }
    }
}

impl<T: Numeric, U: Numeric + Ord> Beam<T, U> for NormalBeam<T, U> {
    fn is_empty(&self) -> bool {
        self.beam.is_empty()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn drain(&mut self) -> NodesInBeam<'_, T, U> {
        println!("capacity: {}", self.capacity);
        println!("size: {}", self.size);
        self.beam.drain()
    }

    fn pop(&mut self) -> Option<Rc<BeamSearchNode<T, U>>> {
        self.beam.queue.pop().map(|node| {
            *node.in_beam.borrow_mut() = false;
            self.size -= 1;
            self.clean_garbage();
            node
        })
    }

    fn insert(
        &mut self,
        registry: &mut StateRegistry<'_, T, Rc<BeamSearchNode<T, U>>>,
        state: StateInRegistry,
        cost: T,
        args: BeamSearchNodeArgs<T, U>,
    ) {
        if self.size < self.capacity || self.beam.queue.peek().map_or(true, |node| args.f < node.f)
        {
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
                        self.clean_garbage();
                    }
                }
                if self.size == self.capacity {
                    self.pop();
                }
                self.beam.queue.push(node);
                self.size += 1;
            }
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
    fn normal_beam_capacity() {
        let model = dypdl::Model::default();
        let mut registry = StateRegistry::new(&model);
        let mut beam = NormalBeam::new(2);
        assert_eq!(beam.capacity(), 2);
        let state = StateInRegistry::default();
        let cost = 0;
        let args = BeamSearchNodeArgs {
            g: 0,
            f: 1,
            ..Default::default()
        };
        beam.insert(&mut registry, state, cost, args);
        assert_eq!(beam.capacity(), 2);
    }

    #[test]
    fn normal_beam_is_empty() {
        let model = dypdl::Model::default();
        let mut registry = StateRegistry::new(&model);
        let mut beam = NormalBeam::new(2);
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
    fn normal_beam_pop() {
        let model = dypdl::Model::default();
        let mut registry = StateRegistry::new(&model);
        let mut beam = NormalBeam::new(1);

        let peek = beam.pop();
        assert_eq!(peek, None);

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

        let peek = beam.pop();
        assert_eq!(
            peek,
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
                in_beam: RefCell::new(false)
            }))
        );
        let peek = beam.pop();
        assert_eq!(peek, None);
    }

    #[test]
    fn normal_beam_drain() {
        let model = dypdl::Model::default();
        let mut registry = StateRegistry::new(&model);
        let mut beam = NormalBeam::new(1);

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
