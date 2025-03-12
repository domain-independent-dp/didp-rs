//! A module for beam.

use super::state_registry::{StateInformation, StateRegistry};
use core::ops::Deref;
use dypdl::variable_type::Numeric;
use smallvec::SmallVec;
use std::cmp::Reverse;
use std::collections;
use std::fmt::Debug;
use std::rc::Rc;

enum BeamDrainInner<'a, V> {
    QueueIter(collections::binary_heap::Drain<'a, Reverse<V>>),
    VecIter(std::vec::Drain<'a, V>),
}

/// An draining iterator for `Beam`
pub struct BeamDrain<'a, T, I>
where
    T: Numeric,
    I: StateInformation<T>,
{
    iter: BeamDrainInner<'a, Rc<I>>,
    phantom: std::marker::PhantomData<T>,
}

impl<T, I> Iterator for BeamDrain<'_, T, I>
where
    T: Numeric,
    I: StateInformation<T>,
{
    type Item = Rc<I>;

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
/// It only keeps the best `capacity` nodes according to `Ord`.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{
///     FNode, StateInRegistry, StateRegistry,
/// };
/// use dypdl_heuristic_search::search_algorithm::data_structure::{
///     Beam, GetTransitions, StateInformation, ParentAndChildStateFunctionCache,
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
/// let mut function_cache = ParentAndChildStateFunctionCache::new(&model.state_functions);
/// let h_evaluator = |_: &StateInRegistry, _: &mut _| Some(0);
/// let f_evaluator = |g, h, _: &StateInRegistry| g + h;
/// let node = FNode::generate_root_node(
///     model.target.clone(),
///     &mut function_cache.parent,
///     0,
///     &model,
///     &h_evaluator,
///     &f_evaluator,
///     None,
/// ).unwrap();
///
/// let mut beam = Beam::new(1);
/// assert!(beam.is_empty());
/// assert_eq!(beam.capacity(), 1);
///
/// let successor = node.generate_successor_node(
///     increment, &mut function_cache, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let status = beam.insert(&mut registry, successor);
/// assert!(status.is_inserted);
/// assert!(status.is_newly_registered);
/// assert!(!status.is_pruned);
/// assert!(status.dominated.is_empty());
/// assert_eq!(status.removed, None);
///
/// let successor = node.generate_successor_node(
///     decrement, &mut function_cache, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let status = beam.insert(&mut registry, successor);
/// assert!(!status.is_inserted);
/// assert!(!status.is_newly_registered);
/// assert!(status.is_pruned);
/// assert!(status.dominated.is_empty());
/// assert_eq!(status.removed, None);
///
/// let successor = node.generate_successor_node(
///     produce.clone(), &mut function_cache, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let state = successor.state().clone();
/// let status = beam.insert(&mut registry, successor);
/// assert!(status.is_inserted);
/// assert!(!status.is_newly_registered);
/// assert!(!status.is_pruned);
/// assert_eq!(status.dominated.len(), 1);
/// assert_eq!(status.removed, None);
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
pub struct Beam<T, I>
where
    T: Numeric,
    I: StateInformation<T>,
{
    /// Capacity of the beam, or the beam size.
    pub capacity: usize,
    size: usize,
    queue: collections::BinaryHeap<Reverse<Rc<I>>>,
    tmp_vec_for_drain: Vec<Rc<I>>,
    phantom: std::marker::PhantomData<T>,
}

/// Result of insertion to beam.
#[derive(Debug, Clone)]
pub struct BeamInsertionStatus<I> {
    /// The given node is inserted into the beam.
    pub is_inserted: bool,
    /// The given node is newly registered to the state registry.
    pub is_newly_registered: bool,
    /// The given node is not inserted into the beam due to the beam width.
    pub is_pruned: bool,
    /// A node dominated by the given node.
    pub dominated: SmallVec<[Rc<I>; 1]>,
    /// A node removed from the beam.
    pub removed: Option<Rc<I>>,
}

impl<T, I> Beam<T, I>
where
    T: Numeric,
    I: StateInformation<T> + Ord,
{
    /// Creates a new beam with a given capacity.
    #[inline]
    pub fn new(capacity: usize) -> Beam<T, I> {
        Beam {
            capacity,
            size: 0,
            queue: collections::BinaryHeap::with_capacity(capacity),
            tmp_vec_for_drain: Vec::default(),
            phantom: std::marker::PhantomData,
        }
    }

    /// Removes a node having the lowest priority from the beam.
    pub fn pop(&mut self) -> Option<Rc<I>> {
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
    #[inline]
    pub fn drain(&mut self) -> BeamDrain<'_, T, I> {
        self.size = 0;
        BeamDrain {
            iter: BeamDrainInner::QueueIter(self.queue.drain()),
            phantom: std::marker::PhantomData,
        }
    }

    /// Removes nodes from the beam and closes all of them, returning all removed nodes as an iterator.
    #[inline]
    pub fn close_and_drain(&mut self) -> BeamDrain<'_, T, I> {
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

    /// Insert a node if it is not dominated and its priority is sufficiently large to be top `capacity` nodes.
    pub fn insert<R>(
        &mut self,
        registry: &mut StateRegistry<T, I, R>,
        node: I,
    ) -> BeamInsertionStatus<I>
    where
        R: Deref<Target = dypdl::Model>,
    {
        let mut result = BeamInsertionStatus {
            is_inserted: false,
            is_newly_registered: false,
            is_pruned: false,
            dominated: smallvec::smallvec![],
            removed: None,
        };

        if self.size < self.capacity || self.queue.peek().map_or(true, |peek| node > *peek.0) {
            let insertion_result = registry.insert(node);

            for d in insertion_result.dominated.iter() {
                if !d.is_closed() {
                    d.close();
                    self.size -= 1;
                    self.clean_garbage();
                }
            }

            result.dominated = insertion_result.dominated;

            if let Some(node) = insertion_result.information {
                if result.dominated.is_empty() {
                    result.is_newly_registered = true;
                }

                if self.size == self.capacity {
                    result.removed = self.pop();
                }

                if self.size < self.capacity {
                    self.queue.push(Reverse(node));
                    self.size += 1;
                    result.is_inserted = true;
                } else {
                    result.is_pruned = true;
                }
            }
        } else {
            result.is_pruned = true;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::hashable_state::HashableSignatureVariables;
    use super::super::state_registry::StateInRegistry;
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
    fn beam_capacity() {
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
        let status = beam.insert(&mut registry, node);
        assert!(status.is_inserted);
        assert!(status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(status.removed, None);
        assert_eq!(beam.capacity(), 2);
    }

    #[test]
    fn beam_is_empty() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::new(model);
        let mut beam = Beam::<_, _>::new(2);
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
        let status = beam.insert(&mut registry, node);
        assert!(status.is_inserted);
        assert!(status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(status.removed, None);
        assert!(!beam.is_empty());
    }

    #[test]
    fn beam_pop() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::<i32, MockInBeam>::new(model.clone());
        let mut beam = Beam::new(1);

        let peek = beam.pop();
        assert_eq!(peek, None);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![3, 1, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 2;
        let h = -1;
        let f = -3;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let status = beam.insert(&mut registry, node);
        assert!(status.is_inserted);
        assert!(status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(status.removed, None);

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
        let status = beam.insert(&mut registry, node);

        let expected_state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![3, 1, 3],
                ..Default::default()
            }),
            ..Default::default()
        };

        assert!(status.is_inserted);
        assert!(status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert!(status.removed.is_some());
        let removed = status.removed.unwrap();
        assert_eq!(removed.cost, 2);
        assert_eq!(removed.h, -1);
        assert_eq!(removed.f, -3);
        assert_eq!(removed.state, expected_state);
        assert_eq!(removed.closed, Cell::new(true));

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
        let status = beam.insert(&mut registry, node);

        let expected_state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };

        assert!(status.is_inserted);
        assert!(!status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated.len(), 1);
        assert_eq!(status.dominated[0].cost, 1);
        assert_eq!(status.dominated[0].h, -1);
        assert_eq!(status.dominated[0].f, -2);
        assert_eq!(status.dominated[0].state, expected_state);
        assert_eq!(status.dominated[0].closed, Cell::new(true));
        assert_eq!(status.removed, None);

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

        let status = beam.insert(&mut registry, node);
        assert!(!status.is_inserted);
        assert!(!status.is_newly_registered);
        assert!(status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(status.removed, None);

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
    fn beam_drain() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::<i32, MockInBeam>::new(model.clone());
        let mut beam = Beam::new(1);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![3, 1, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 2;
        let h = -1;
        let f = -3;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let status = beam.insert(&mut registry, node);
        assert!(status.is_inserted);
        assert!(status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(status.removed, None);

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
        let status = beam.insert(&mut registry, node);

        let expected_state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![3, 1, 3],
                ..Default::default()
            }),
            ..Default::default()
        };

        assert!(status.is_inserted);
        assert!(status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert!(status.removed.is_some());
        let removed = status.removed.unwrap();
        assert_eq!(removed.cost, 2);
        assert_eq!(removed.h, -1);
        assert_eq!(removed.f, -3);
        assert_eq!(removed.state, expected_state);
        assert_eq!(removed.closed, Cell::new(true));

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
        let status = beam.insert(&mut registry, node);

        let expected_state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };

        assert!(status.is_inserted);
        assert!(!status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated.len(), 1);
        assert_eq!(status.dominated[0].cost, 1);
        assert_eq!(status.dominated[0].h, -1);
        assert_eq!(status.dominated[0].f, -2);
        assert_eq!(status.dominated[0].state, expected_state);
        assert_eq!(status.dominated[0].closed, Cell::new(true));
        assert_eq!(status.removed, None);

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
        let status = beam.insert(&mut registry, node);
        assert!(!status.is_inserted);
        assert!(!status.is_newly_registered);
        assert!(status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(status.removed, None);

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
    fn beam_close_and_drain() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::<i32, MockInBeam>::new(model.clone());
        let mut beam = Beam::new(1);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![3, 1, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 2;
        let h = -1;
        let f = -3;
        let node = MockInBeam {
            state,
            cost,
            h,
            f,
            closed: Cell::new(false),
        };
        let status = beam.insert(&mut registry, node);
        assert!(status.is_inserted);
        assert!(status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(status.removed, None);

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
        let status = beam.insert(&mut registry, node);

        let expected_state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![3, 1, 3],
                ..Default::default()
            }),
            ..Default::default()
        };

        assert!(status.is_inserted);
        assert!(status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert!(status.removed.is_some());
        let removed = status.removed.unwrap();
        assert_eq!(removed.cost, 2);
        assert_eq!(removed.h, -1);
        assert_eq!(removed.f, -3);
        assert_eq!(removed.state, expected_state);
        assert_eq!(removed.closed, Cell::new(true));

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
        let status = beam.insert(&mut registry, node);

        let expected_state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };

        assert!(status.is_inserted);
        assert!(!status.is_newly_registered);
        assert!(!status.is_pruned);
        assert_eq!(status.dominated.len(), 1);
        assert_eq!(status.dominated[0].cost, 1);
        assert_eq!(status.dominated[0].h, -1);
        assert_eq!(status.dominated[0].f, -2);
        assert_eq!(status.dominated[0].state, expected_state);
        assert_eq!(status.dominated[0].closed, Cell::new(true));
        assert_eq!(status.removed, None);

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
        let status = beam.insert(&mut registry, node);
        assert!(!status.is_inserted);
        assert!(!status.is_newly_registered);
        assert!(status.is_pruned);
        assert_eq!(status.dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(status.removed, None);

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
