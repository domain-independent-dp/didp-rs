use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type::Numeric;
use dypdl::Transition;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::rc::Rc;

/// Trait to check if it is included in the beam.
pub trait InBeam {
    /// Returns if it is included in the queue.
    fn in_queue(&self) -> bool;

    /// Returns if it is included in the pool.
    fn in_pool(&self) -> bool;
}

/// Trait to get g and f-values.
pub trait PrioritizedNode<T> {
    /// Returns the g-value.
    fn g(&self) -> T;

    /// Returns the f-value.
    fn f(&self) -> T;
}

/// Trait to get transitions.
pub trait GetTransitions {
    fn transitions(&self) -> Vec<Transition>;
}

/// Search node for beam search.
///
/// Nodes totally ordered by their f-values.
#[derive(Debug, Default)]
pub struct CustomCostBeamSearchNode<T: Numeric, U: Numeric> {
    /// g-value.
    pub g: U,
    /// f-value.
    pub f: U,
    /// State.
    pub state: StateInRegistry,
    /// Accumulated cost along the path so far.
    pub cost: T,
    /// Transitions applied to reach this node.
    pub transitions: Vec<Rc<TransitionWithCustomCost>>,
    /// If included in a beam.
    pub in_beam: RefCell<bool>,
}

impl<T: Numeric, U: Numeric> InBeam for Rc<CustomCostBeamSearchNode<T, U>> {
    fn in_queue(&self) -> bool {
        *self.in_beam.borrow()
    }

    fn in_pool(&self) -> bool {
        false
    }
}

impl<T: Numeric, U: Numeric> PrioritizedNode<U> for Rc<CustomCostBeamSearchNode<T, U>> {
    fn g(&self) -> U {
        self.g
    }

    fn f(&self) -> U {
        self.f
    }
}

impl<T: Numeric, U: Numeric + PartialOrd> PartialEq for CustomCostBeamSearchNode<T, U> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.g == other.g
    }
}

impl<T: Numeric, U: Numeric + Ord> Eq for CustomCostBeamSearchNode<T, U> {}

impl<T: Numeric, U: Numeric + Ord> Ord for CustomCostBeamSearchNode<T, U> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => other.g.cmp(&self.g),
            result => result,
        }
    }
}

impl<T: Numeric, U: Numeric + Ord> PartialOrd for CustomCostBeamSearchNode<T, U> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric, U: Numeric> StateInformation<T> for Rc<CustomCostBeamSearchNode<T, U>> {
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.cost
    }
}

impl<T: Numeric, U: Numeric> GetTransitions for Rc<CustomCostBeamSearchNode<T, U>> {
    fn transitions(&self) -> Vec<Transition> {
        self.transitions
            .iter()
            .map(|t| t.transition.clone())
            .collect()
    }
}

/// Parameters to create a beam search node.
#[derive(Debug, Default)]
pub struct BeamSearchNodeArgs<T: Numeric, U: Ord> {
    pub g: T,
    pub f: T,
    pub operator: Option<Rc<TransitionWithCustomCost>>,
    pub parent: Option<U>,
}

/// Common structure of beam.
#[derive(Debug, Clone)]
pub struct BeamBase<T: Ord> {
    /// Priority queue to store nodes.
    pub queue: collections::BinaryHeap<T>,
    /// Vector to store nodes.
    pub pool: Vec<T>,
}

impl<T: InBeam + Ord> BeamBase<T> {
    /// Removes nodes from the beam, returning all removed nodes as an iterator.
    pub fn drain(&mut self) -> BeamDrain<'_, T> {
        BeamDrain {
            queue_iter: self.queue.drain(),
            pool_iter: self.pool.drain(..),
            pool_mode: false,
        }
    }
}

/// An draining iterator for `BeamBeam<T>`
pub struct BeamDrain<'a, T: InBeam + Ord> {
    queue_iter: collections::binary_heap::Drain<'a, T>,
    pool_iter: std::vec::Drain<'a, T>,
    pool_mode: bool,
}

impl<'a, T: InBeam + Ord> Iterator for BeamDrain<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pool_mode {
            match self.pool_iter.next() {
                Some(node) if !node.in_pool() => self.next(),
                node => node,
            }
        } else {
            match self.queue_iter.next() {
                Some(node) if !node.in_queue() => self.next(),
                None => {
                    self.pool_mode = true;
                    self.next()
                }
                node => node,
            }
        }
    }
}

pub trait Beam<T: Numeric, U: Numeric + Ord, V: InBeam + Ord + StateInformation<T>> {
    /// Returns true if no state in beam and false otherwise.
    fn is_empty(&self) -> bool;

    /// Returns the capacity of the beam.
    fn capacity(&self) -> usize;

    /// Removes nodes from the beam, returning all removed nodes as an iterator.
    fn drain(&mut self) -> BeamDrain<'_, V>;

    /// Returns true if the state is not pruned due to the capacity.
    fn insert(
        &mut self,
        registry: &mut StateRegistry<'_, T, V>,
        state: StateInRegistry,
        cost: T,
        args: BeamSearchNodeArgs<U, V>,
    ) -> bool;
}

/// Beam for beam search.
#[derive(Debug, Clone)]
pub struct NormalBeam<T: Numeric, U: Numeric + Ord> {
    /// Capacity of the beam, or the beam size.
    pub capacity: usize,
    size: usize,
    beam: BeamBase<Rc<CustomCostBeamSearchNode<T, U>>>,
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

    fn pop(&mut self) -> Option<Rc<CustomCostBeamSearchNode<T, U>>> {
        self.beam.queue.pop().map(|node| {
            *node.in_beam.borrow_mut() = false;
            self.size -= 1;
            self.clean_garbage();
            node
        })
    }

    fn clean_garbage(&mut self) {
        let mut peek = self.beam.queue.peek();
        while peek.map_or(false, |node| !*node.in_beam.borrow()) {
            self.beam.queue.pop();
            peek = self.beam.queue.peek();
        }
    }
}

impl<T: Numeric, U: Numeric + Ord> Beam<T, U, Rc<CustomCostBeamSearchNode<T, U>>>
    for NormalBeam<T, U>
{
    fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn drain(&mut self) -> BeamDrain<'_, Rc<CustomCostBeamSearchNode<T, U>>> {
        self.size = 0;
        self.beam.drain()
    }

    fn insert(
        &mut self,
        registry: &mut StateRegistry<'_, T, Rc<CustomCostBeamSearchNode<T, U>>>,
        state: StateInRegistry,
        cost: T,
        args: BeamSearchNodeArgs<U, Rc<CustomCostBeamSearchNode<T, U>>>,
    ) -> bool {
        if self.size < self.capacity
            || self.beam.queue.peek().map_or(true, |node| {
                (args.f < node.f) || (args.f == node.f && args.g > node.g)
            })
        {
            let constructor =
                |state: StateInRegistry,
                 cost: T,
                 _: Option<&Rc<CustomCostBeamSearchNode<T, U>>>| {
                    let transitions = args.parent.map_or_else(Vec::new, |parent| {
                        Vec::from_iter(
                            parent
                                .transitions
                                .iter()
                                .cloned()
                                .chain(args.operator.into_iter()),
                        )
                    });

                    Some(Rc::new(CustomCostBeamSearchNode {
                        g: args.g,
                        f: args.f,
                        state,
                        cost,
                        transitions,
                        in_beam: RefCell::new(true),
                    }))
                };
            let mut pruned_by_capacity = false;
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
                    pruned_by_capacity = true;
                }
                if self.size < self.capacity {
                    self.beam.queue.push(node);
                    self.size += 1;
                }
            }
            !pruned_by_capacity
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashable_state::HashableSignatureVariables;

    #[test]
    fn search_node_getter() {
        let node = Rc::new(CustomCostBeamSearchNode {
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
            transitions: Vec::new(),
        });
        assert_eq!(node.state(), &node.state);
        assert_eq!(node.cost(), 1);
        assert_eq!(node.transitions(), Vec::new());
    }

    #[test]
    fn search_node_cmp() {
        let node1 = CustomCostBeamSearchNode {
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
            transitions: Vec::new(),
        };
        let node2 = CustomCostBeamSearchNode {
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
            transitions: Vec::new(),
        };
        assert_eq!(node1, node2);
        let node2 = CustomCostBeamSearchNode {
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
            transitions: Vec::new(),
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
        assert!(beam.insert(&mut registry, state, cost, args));
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
        assert!(beam.insert(&mut registry, state, cost, args));
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
        assert!(beam.insert(&mut registry, state, cost, args));

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
        assert!(beam.insert(&mut registry, state, cost, args));

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
        assert!(!beam.insert(&mut registry, state, cost, args));

        let peek = beam.pop();
        assert_eq!(
            peek,
            Some(Rc::new(CustomCostBeamSearchNode {
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
                transitions: Vec::new(),
                in_beam: RefCell::new(false)
            }))
        );
        let peek = beam.pop();
        assert_eq!(peek, None);
    }

    #[test]
    fn normal_beam_pop_with_parent() {
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
        assert!(beam.insert(&mut registry, state, cost, args));

        let parent = Rc::new(CustomCostBeamSearchNode {
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
            transitions: vec![Rc::new(TransitionWithCustomCost {
                transition: Transition {
                    name: String::from("tr1"),
                    ..Default::default()
                },
                ..Default::default()
            })],
        });

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
            parent: Some(parent),
            operator: Some(Rc::new(TransitionWithCustomCost {
                transition: Transition {
                    name: String::from("tr1"),
                    ..Default::default()
                },
                ..Default::default()
            })),
        };
        assert!(beam.insert(&mut registry, state, cost, args));

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
        assert!(!beam.insert(&mut registry, state, cost, args));

        let peek = beam.pop();
        assert_eq!(
            peek,
            Some(Rc::new(CustomCostBeamSearchNode {
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
                transitions: vec![
                    Rc::new(TransitionWithCustomCost {
                        transition: Transition {
                            name: String::from("tr1"),
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
                    Rc::new(TransitionWithCustomCost {
                        transition: Transition {
                            name: String::from("tr2"),
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
                ],
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
        assert!(beam.insert(&mut registry, state, cost, args));

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
        assert!(beam.insert(&mut registry, state, cost, args));

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
        assert!(!beam.insert(&mut registry, state, cost, args));

        let mut iter = beam.drain();
        assert_eq!(
            iter.next(),
            Some(Rc::new(CustomCostBeamSearchNode {
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
                transitions: Vec::new(),
                in_beam: RefCell::new(true)
            }))
        );
        assert_eq!(iter.next(), None);
    }
}
