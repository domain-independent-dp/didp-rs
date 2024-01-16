//! A module for beam.

use super::hashable_state::HashableSignatureVariables;
use super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use core::ops::Deref;
use dypdl::variable_type::Numeric;
use dypdl::StateInterface;
use std::cmp::Reverse;
use std::collections;
use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;

enum BeamDrainInner<'a, V> {
    QueueIter(collections::binary_heap::Drain<'a, Reverse<V>>),
    VecIter(std::vec::Drain<'a, V>),
}

/// An draining iterator for `Beam`
pub struct BeamDrain<'a, T, I, V = Rc<I>, K = Rc<HashableSignatureVariables>>
where
    T: Numeric,
    I: StateInformation<T, K>,
    K: Hash + Eq + Clone + Debug,
    V: Ord + Deref<Target = I>,
{
    iter: BeamDrainInner<'a, V>,
    phantom: std::marker::PhantomData<(T, I, K)>,
}

impl<'a, T, I, V, K> Iterator for BeamDrain<'a, T, I, V, K>
where
    T: Numeric,
    I: StateInformation<T, K>,
    K: Hash + Eq + Clone + Debug,
    V: Ord + Deref<Target = I>,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.iter {
            BeamDrainInner::QueueIter(iter) => match iter.next() {
                Some(node) if node.0.is_closed() => self.next(),
                node => node.map(|node| node.0),
            },
            BeamDrainInner::VecIter(iter) => iter.next(),
        }
    }
}

/// Beam for beam search.
///
/// It only keeps the best `capacity` nodes that maximizes the f-value and h-value.
/// Nodes must be ordered by f-values and h-values, implementing Ord trait.
/// If you want to keep the best nodes that minimizes the f- and h-value, use negative values.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{
///     FNode, StateInRegistry, StateRegistry,
/// };
/// use dypdl_heuristic_search::search_algorithm::data_structure::{
///     Beam, GetTransitions, StateInformation,
/// };
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let signature = model.add_integer_variable("signature", 1).unwrap();
/// let resource = model.add_integer_resource_variable("resource", false, 1).unwrap();
///
/// let mut increment = Transition::new("increment");
/// increment.set_cost(IntegerExpression::Cost + 2);
/// increment.add_effect(signature, signature + 1).unwrap();
/// let increment = Rc::new(increment);
///
/// let mut decrement = Transition::new("decrement");
/// decrement.set_cost(IntegerExpression::Cost + 3);
/// decrement.add_effect(signature, signature - 1).unwrap();
/// let decrement = Rc::new(decrement);
///
/// let mut produce = Transition::new("produce");
/// produce.set_cost(IntegerExpression::Cost + 1);
/// produce.add_effect(signature, signature + 1).unwrap();
/// produce.add_effect(resource, resource + 1).unwrap();
/// let produce = Rc::new(produce);
///
/// let model = Rc::new(model);
/// let mut registry = StateRegistry::new(model.clone());
///
/// let h_evaluator = |_: &StateInRegistry| Some(0);
/// let f_evaluator = |g, h, _: &StateInRegistry| g + h;
/// let node = FNode::generate_root_node(
///     model.target.clone(), 0, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
///
/// let mut beam = Beam::new(1);
/// assert!(beam.is_empty());
/// assert_eq!(beam.capacity(), 1);
///
/// let successor = node.generate_successor_node(
///     increment, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let (generated, pruned) = beam.insert(&mut registry, successor);
/// assert!(generated);
/// assert!(!pruned);
///
/// let successor = node.generate_successor_node(
///     decrement, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let (generated, pruned) = beam.insert(&mut registry, successor);
/// assert!(!generated);
/// assert!(pruned);
///
/// let successor = node.generate_successor_node(
///     produce.clone(), &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let state = successor.state().clone();
/// let (generated, pruned) = beam.insert(&mut registry, successor);
/// assert!(!generated);
/// assert!(!pruned);
///
/// let mut iter = beam.drain();
/// let node: Rc<FNode<_>> = iter.next().unwrap();
/// assert_eq!(node.state(), &state);
/// assert_eq!(node.cost(&model), 1);
/// assert_eq!(node.bound(&model), Some(1));
/// assert_eq!(node.transitions(), vec![(*produce).clone()]);
/// assert_eq!(iter.next(), None);
/// ```
#[derive(Debug, Clone)]
pub struct Beam<T, I, V = Rc<I>, K = Rc<HashableSignatureVariables>>
where
    T: Numeric,
    I: StateInformation<T, K>,
    V: Deref<Target = I> + From<I> + Clone + Ord,
    K: Hash + Eq + Clone + Debug,
{
    /// Capacity of the beam, or the beam size.
    pub capacity: usize,
    size: usize,
    queue: collections::BinaryHeap<Reverse<V>>,
    tmp_vec_for_drain: Vec<V>,
    phantom: std::marker::PhantomData<(T, I, K)>,
}

impl<T, I, V, K> Beam<T, I, V, K>
where
    T: Numeric,
    I: StateInformation<T, K> + Ord,
    V: Deref<Target = I> + From<I> + Clone + Ord,
    K: Hash + Eq + Clone + Debug,
{
    /// Creates a new beam with a given capacity.
    #[inline]
    pub fn new(capacity: usize) -> Beam<T, I, V, K> {
        Beam {
            capacity,
            size: 0,
            queue: collections::BinaryHeap::with_capacity(capacity),
            tmp_vec_for_drain: Vec::default(),
            phantom: std::marker::PhantomData,
        }
    }

    /// Removes a node having the lowest priority from the beam.
    pub fn pop(&mut self) -> Option<V> {
        self.queue.pop().map(|node| {
            node.0.close();
            self.size -= 1;
            self.clean_garbage();
            node.0
        })
    }

    fn clean_garbage(&mut self) {
        let mut peek = self.queue.peek();

        while peek.map_or(false, |node| node.0.is_closed()) {
            self.queue.pop();
            peek = self.queue.peek();
        }
    }

    /// Returns true if no state in beam and false otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the capacity of the beam.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Removes nodes from the beam, returning all removed nodes as an iterator.
    /// This method does not close the removed nodes.
    #[inline]
    pub fn drain(&mut self) -> BeamDrain<'_, T, I, V, K> {
        self.size = 0;
        BeamDrain {
            iter: BeamDrainInner::QueueIter(self.queue.drain()),
            phantom: std::marker::PhantomData,
        }
    }

    /// Removes nodes from the beam and closes all of them, returning all removed nodes as an iterator.
    #[inline]
    pub fn close_and_drain(&mut self) -> BeamDrain<'_, T, I, V, K> {
        self.tmp_vec_for_drain.reserve(self.size);
        self.size = 0;

        self.tmp_vec_for_drain
            .extend(self.queue.drain().filter_map(|node| {
                if !node.0.is_closed() {
                    node.0.close();
                    Some(node.0)
                } else {
                    None
                }
            }));

        BeamDrain {
            iter: BeamDrainInner::VecIter(self.tmp_vec_for_drain.drain(..)),
            phantom: std::marker::PhantomData,
        }
    }

    /// Insert a node if it is not dominated and its h- and f-value are sufficiently large to be top `capacity` nodes.
    ///
    /// The first returned value represents if a new search node (not an update version of an existing node) is generated.
    /// The second returned value represents if the pruning due to the beam size happened.
    pub fn insert<R>(
        &mut self,
        registry: &mut StateRegistry<T, I, V, K, R>,
        node: I,
    ) -> (bool, bool)
    where
        R: Deref<Target = dypdl::Model>,
        StateInRegistry<K>: StateInterface,
    {
        if self.size < self.capacity || self.queue.peek().map_or(true, |peek| node > *peek.0) {
            let mut generated = false;
            let mut beam_pruning = false;

            if let Some((node, dominated)) = registry.insert(node) {
                if let Some(dominated) = dominated {
                    if !dominated.is_closed() {
                        dominated.close();
                        self.size -= 1;
                        self.clean_garbage();
                    }
                } else {
                    generated = true;
                }

                if self.size == self.capacity {
                    self.pop();
                    beam_pruning = true;
                }

                if self.size < self.capacity {
                    self.queue.push(Reverse(node));
                    self.size += 1;
                }
            }

            (generated, beam_pruning)
        } else {
            (false, true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::hashable_state::HashableSignatureVariables;
    use super::*;
    use dypdl::Model;
    use std::cell::Cell;
    use std::cmp::Ordering;

    #[derive(Debug)]
    struct MockInBeam {
        state: StateInRegistry,
        cost: i32,
        h: i32,
        f: i32,
        closed: Cell<bool>,
    }

    impl StateInformation<i32> for MockInBeam {
        fn state(&self) -> &StateInRegistry {
            &self.state
        }

        fn state_mut(&mut self) -> &mut StateInRegistry {
            &mut self.state
        }

        fn cost(&self, _: &Model) -> i32 {
            self.cost
        }

        fn bound(&self, _: &Model) -> Option<i32> {
            Some(self.f)
        }

        fn is_closed(&self) -> bool {
            self.closed.get()
        }

        fn close(&self) {
            self.closed.set(true);
        }
    }

    impl PartialEq for MockInBeam {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.f == other.f && self.h == other.h
        }
    }

    impl Eq for MockInBeam {}

    impl Ord for MockInBeam {
        fn cmp(&self, other: &Self) -> Ordering {
            match self.f.cmp(&other.f) {
                Ordering::Equal => self.h.cmp(&other.h),
                result => result,
            }
        }
    }

    impl PartialOrd for MockInBeam {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    #[test]
    fn normal_beam_capacity() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::new(model);
        let mut beam = Beam::<_, _>::new(2);
        assert_eq!(beam.capacity(), 2);
        let state = StateInRegistry::default();
        let cost = 0;
        let h = -1;
        let f = -1;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(generated);
        assert!(!beam_pruning);
        assert_eq!(beam.capacity(), 2);
    }

    #[test]
    fn normal_beam_is_empty() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::new(model);
        let mut beam = Beam::<_, _, Rc<_>>::new(2);
        assert!(beam.is_empty());
        let state = StateInRegistry::default();
        let cost = 0;
        let h = -1;
        let f = -1;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(generated);
        assert!(!beam_pruning);
        assert!(!beam.is_empty());
    }

    #[test]
    fn normal_beam_pop() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::<i32, MockInBeam>::new(model.clone());
        let mut beam = Beam::new(1);

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
        let h = -1;
        let f = -2;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(generated);
        assert!(!beam_pruning);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 0;
        let h = -1;
        let f = -1;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(!generated);
        assert!(!beam_pruning);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![2, 3, 4],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 0;
        let h = -2;
        let f = -2;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(!generated);
        assert!(beam_pruning);

        let peek = beam.pop();
        assert!(peek.is_some());
        let node = peek.unwrap();
        assert_eq!(
            node.state,
            StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        assert_eq!(node.cost(&model), 0);
        assert!(node.is_closed());
        let peek = beam.pop();
        assert_eq!(peek, None);
    }

    #[test]
    fn normal_beam_drain() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::<i32, MockInBeam>::new(model.clone());
        let mut beam = Beam::new(1);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 1;
        let h = -1;
        let f = -2;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(generated);
        assert!(!beam_pruning);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 0;
        let h = -1;
        let f = -1;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(!generated);
        assert!(!beam_pruning);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![2, 3, 4],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 0;
        let h = -2;
        let f = -2;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(!generated);
        assert!(beam_pruning);

        let mut iter = beam.drain();
        let peek = iter.next();
        assert!(peek.is_some());
        let node = peek.unwrap();
        assert_eq!(
            node.state,
            StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        assert_eq!(node.cost(&model), 0);
        assert!(!node.is_closed());
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn normal_beam_close_and_drain() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::<i32, MockInBeam>::new(model.clone());
        let mut beam = Beam::new(1);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 1;
        let h = -1;
        let f = -2;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(generated);
        assert!(!beam_pruning);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 0;
        let h = -1;
        let f = -1;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(!generated);
        assert!(!beam_pruning);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![2, 3, 4],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 0;
        let h = -2;
        let f = -2;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, node);
        assert!(!generated);
        assert!(beam_pruning);

        let mut iter = beam.close_and_drain();
        let peek = iter.next();
        assert!(peek.is_some());
        let node = peek.unwrap();
        assert_eq!(
            node.state,
            StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        assert_eq!(node.cost(&model), 0);
        assert!(node.is_closed());
        assert_eq!(iter.next(), None);
    }
}
