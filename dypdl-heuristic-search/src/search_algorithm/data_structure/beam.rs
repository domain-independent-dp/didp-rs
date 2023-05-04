//! A module for beam.

use super::hashable_state::HashableSignatureVariables;
use super::prioritized_node::PrioritizedNode;
use super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use core::ops::Deref;
use dypdl::variable_type::Numeric;
use std::collections;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;

/// State information in beam.
pub trait InformationInBeam<T, U, K = Rc<HashableSignatureVariables>>:
    Ord + StateInformation<T, K> + PrioritizedNode<U> + InBeam
where
    T: Numeric,
    U: Numeric,
    K: Hash + Eq + Clone + Debug,
{
}

/// Trait to check if it is included in the beam.
pub trait InBeam {
    /// Returns if it is included in the beam.
    fn in_beam(&self) -> bool;

    /// Remove it from the beam.
    fn remove_from_beam(&self);
}

/// An draining iterator for `Beam`
pub struct BeamDrain<'a, I: InBeam, V: Ord + Deref<Target = I>> {
    queue_iter: collections::binary_heap::Drain<'a, V>,
}

impl<'a, I: InBeam, V: Ord + Deref<Target = I>> Iterator for BeamDrain<'a, I, V> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        match self.queue_iter.next() {
            Some(node) if !node.in_beam() => self.next(),
            node => node,
        }
    }
}

/// Trait representing beam.
///
/// It keeps the best `capacity` nodes according to some criteria based on the f- and g-values.
pub trait BeamInterface<
    T,
    U,
    I,
    V = Rc<I>,
    K = Rc<HashableSignatureVariables>,
    R = Rc<dypdl::Model>,
> where
    T: Numeric,
    U: Numeric,
    I: InformationInBeam<T, U, K>,
    V: Deref<Target = I> + From<I> + Clone + Ord,
    K: Hash + Eq + Clone + Debug,
    R: Deref<Target = dypdl::Model>,
{
    /// Returns true if no state in beam and false otherwise.
    fn is_empty(&self) -> bool;

    /// Returns the capacity of the beam.
    fn capacity(&self) -> usize;

    /// Removes nodes from the beam, returning all removed nodes as an iterator.
    fn drain(&mut self) -> BeamDrain<'_, I, V>;

    /// Insert a node if it is not dominated and its g- and f-value satisfy criteria.
    ///
    /// The first returned value represents if a new search node (not an update version of an existing node) is generated.
    /// The second returned value represents if the pruning due to the beam size happened.
    fn insert<F>(
        &mut self,
        registry: &mut StateRegistry<T, I, V, K, R>,
        state: StateInRegistry<K>,
        cost: T,
        g: U,
        f: U,
        constructor: F,
    ) -> (bool, bool)
    where
        F: FnOnce(StateInRegistry<K>, T, Option<&I>) -> Option<I>;
}

/// Beam for beam search.
///
/// It only keeps the best `capacity` nodes that minimizes the f-value.
/// If the f-values are the same, it prefers a node with a larger g-value.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::BeamSearchNode;
/// use dypdl_heuristic_search::search_algorithm::data_structure::state_registry::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::beam::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::CustomCostNodeInterface;
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let signature = model.add_integer_variable("signature", 1).unwrap();
/// let resource = model.add_integer_resource_variable("resource", false, 1).unwrap();
/// let parent = StateInRegistry::<Rc<_>>::from(model.target.clone());
/// let parent_cost = 0;
///
/// let mut increment = Transition::new("increment");
/// increment.set_cost(IntegerExpression::Cost + 2);
/// increment.add_effect(signature, signature + 1).unwrap();
///
/// let mut decrement = Transition::new("decrement");
/// decrement.set_cost(IntegerExpression::Cost + 3);
/// decrement.add_effect(signature, signature - 1).unwrap();
///
/// let mut produce = Transition::new("produce");
/// produce.set_cost(IntegerExpression::Cost + 1);
/// produce.add_effect(signature, signature + 1).unwrap();
/// produce.add_effect(resource, resource + 1).unwrap();
///
/// let model = Rc::new(model);
/// let mut registry = StateRegistry::<_, BeamSearchNode<_, _>>::new(model.clone());
/// let constructor = |state, cost, _: Option<&_>| {
///     Some(BeamSearchNode::new(cost, cost, state, cost, None, None))
/// };
///
/// let mut beam = Beam::new(1);
/// assert!(BeamInterface::<_, _, _>::is_empty(&beam));
/// assert_eq!(BeamInterface::<_, _, _>::capacity(&beam), 1);
///
/// let state: StateInRegistry = increment.apply(&parent, &model.table_registry);
/// let cost = increment.eval_cost(parent_cost, &parent, &model.table_registry);
/// let (generated, pruned) = beam.insert(&mut registry, state, cost, cost, cost, constructor);
/// assert!(generated);
/// assert!(!pruned);
///
/// let state: StateInRegistry = decrement.apply(&parent, &model.table_registry);
/// let cost = increment.eval_cost(parent_cost, &parent, &model.table_registry);
/// let (generated, pruned) = beam.insert(&mut registry, state, cost, cost, cost, constructor);
/// assert!(!generated);
/// assert!(pruned);
///
/// let state: StateInRegistry = produce.apply(&parent, &model.table_registry);
/// let cost = produce.eval_cost(parent_cost, &parent, &model.table_registry);
/// let (generated, pruned) = beam.insert(&mut registry, state.clone(), cost, cost, cost, constructor);
/// assert!(!generated);
/// assert!(!pruned);
///
/// let expected = Rc::new(BeamSearchNode { g: cost, f: cost, state, cost, ..Default::default() });
/// let mut iter = BeamInterface::<_, _, _>::drain(&mut beam);
/// let node = iter.next().unwrap();
/// assert_eq!(node.g, expected.g);
/// assert_eq!(node.f, expected.f);
/// assert_eq!(node.state, expected.state);
/// assert_eq!(node.cost, expected.cost);
/// assert_eq!(iter.next(), None);
/// ```
#[derive(Debug, Clone)]
pub struct Beam<T, U, I, V = Rc<I>, K = Rc<HashableSignatureVariables>>
where
    T: Numeric,
    U: Numeric + Ord,
    I: InformationInBeam<T, U, K>,
    V: Deref<Target = I> + From<I> + Clone + Ord,
    K: Hash + Eq + Clone + Debug,
{
    /// Capacity of the beam, or the beam size.
    pub capacity: usize,
    size: usize,
    queue: collections::BinaryHeap<V>,
    phantom: PhantomData<(T, U, K)>,
}

impl<T, U, I, V, K> Beam<T, U, I, V, K>
where
    T: Numeric,
    U: Numeric + Ord,
    I: InformationInBeam<T, U, K>,
    V: Deref<Target = I> + From<I> + Clone + Ord,
    K: Hash + Eq + Clone + Debug,
{
    /// Creates a new beam with a given capacity.
    #[inline]
    pub fn new(capacity: usize) -> Beam<T, U, I, V, K> {
        Beam {
            capacity,
            size: 0,
            queue: collections::BinaryHeap::with_capacity(capacity),
            phantom: PhantomData::default(),
        }
    }

    /// Removes a node having the lowest priority from the beam.
    pub fn pop(&mut self) -> Option<V> {
        self.queue.pop().map(|node| {
            node.remove_from_beam();
            self.size -= 1;
            self.clean_garbage();
            node
        })
    }

    fn clean_garbage(&mut self) {
        let mut peek = self.queue.peek();

        while peek.map_or(false, |node| !node.in_beam()) {
            self.queue.pop();
            peek = self.queue.peek();
        }
    }
}

impl<T, U, I, V, K, R> BeamInterface<T, U, I, V, K, R> for Beam<T, U, I, V, K>
where
    T: Numeric,
    U: Numeric + Ord,
    I: InformationInBeam<T, U, K>,
    V: Deref<Target = I> + From<I> + Clone + Ord,
    K: Hash + Eq + Clone + Debug,
    R: Deref<Target = dypdl::Model>,
    StateInRegistry<K>: dypdl::StateInterface,
{
    #[inline]
    fn is_empty(&self) -> bool {
        self.size == 0
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    fn drain(&mut self) -> BeamDrain<'_, I, V> {
        self.size = 0;
        BeamDrain {
            queue_iter: self.queue.drain(),
        }
    }

    fn insert<F>(
        &mut self,
        registry: &mut StateRegistry<T, I, V, K, R>,
        state: StateInRegistry<K>,
        cost: T,
        g: U,
        f: U,
        constructor: F,
    ) -> (bool, bool)
    where
        F: FnOnce(StateInRegistry<K>, T, Option<&I>) -> Option<I>,
    {
        if self.size < self.capacity
            || self.queue.peek().map_or(true, |node| {
                (f < node.f()) || (f == node.f() && g > node.g())
            })
        {
            let mut generated = false;
            let mut beam_pruning = false;

            if let Some((node, dominated)) = registry.insert(state, cost, constructor) {
                if let Some(dominated) = dominated {
                    if dominated.in_beam() {
                        dominated.remove_from_beam();
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
                    self.queue.push(node);
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
    use std::cell::RefCell;
    use std::cmp::Ordering;

    #[derive(Debug)]
    struct MockInBeam {
        state: StateInRegistry,
        cost: i32,
        g: i32,
        f: i32,
        in_beam: RefCell<bool>,
    }

    impl StateInformation<i32> for MockInBeam {
        fn state(&self) -> &StateInRegistry {
            &self.state
        }

        fn cost(&self) -> i32 {
            self.cost
        }
    }

    impl PartialEq for MockInBeam {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.f == other.f && self.g == other.g
        }
    }

    impl Eq for MockInBeam {}

    impl Ord for MockInBeam {
        fn cmp(&self, other: &Self) -> Ordering {
            match self.f.cmp(&other.f) {
                Ordering::Equal => other.g.cmp(&self.g),
                result => result,
            }
        }
    }

    impl PartialOrd for MockInBeam {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl PrioritizedNode<i32> for MockInBeam {
        fn g(&self) -> i32 {
            self.g
        }

        fn f(&self) -> i32 {
            self.f
        }
    }

    impl InBeam for MockInBeam {
        fn in_beam(&self) -> bool {
            *self.in_beam.borrow()
        }

        fn remove_from_beam(&self) {
            *self.in_beam.borrow_mut() = false;
        }
    }

    impl InformationInBeam<i32, i32> for MockInBeam {}

    #[test]
    fn normal_beam_capacity() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::new(model);
        let mut beam = Beam::<i32, i32, MockInBeam>::new(2);
        assert_eq!(BeamInterface::<i32, i32, MockInBeam>::capacity(&beam), 2);
        let state = StateInRegistry::default();
        let cost = 0;
        let g = 0;
        let f = 1;
        let constructor = |state, cost, _: Option<&_>| {
            Some(MockInBeam {
                state,
                cost,
                g,
                f,
                in_beam: RefCell::new(true),
            })
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, state, cost, g, f, constructor);
        assert!(generated);
        assert!(!beam_pruning);
        assert_eq!(BeamInterface::<i32, i32, MockInBeam>::capacity(&beam), 2);
    }

    #[test]
    fn normal_beam_is_empty() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::new(model);
        let mut beam = Beam::<i32, i32, MockInBeam>::new(2);
        assert!(BeamInterface::<i32, i32, MockInBeam>::is_empty(&beam));
        let state = StateInRegistry::default();
        let cost = 0;
        let g = 0;
        let f = 1;
        let constructor = |state, cost, _: Option<&_>| {
            Some(MockInBeam {
                state,
                cost,
                g,
                f,
                in_beam: RefCell::new(true),
            })
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, state, cost, g, f, constructor);
        assert!(generated);
        assert!(!beam_pruning);
        assert!(!BeamInterface::<i32, i32, MockInBeam>::is_empty(&beam));
    }

    #[test]
    fn normal_beam_pop() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::<i32, MockInBeam>::new(model);
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
        let g = 0;
        let f = 2;
        let constructor = |state, cost, _: Option<&_>| {
            Some(MockInBeam {
                state,
                cost,
                g,
                f,
                in_beam: RefCell::new(true),
            })
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, state, cost, g, f, constructor);
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
        let g = 0;
        let f = 1;
        let constructor = |state, cost, _: Option<&_>| {
            Some(MockInBeam {
                state,
                cost,
                g,
                f,
                in_beam: RefCell::new(true),
            })
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, state, cost, g, f, constructor);
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
        let g = 0;
        let f = 2;
        let constructor = |state, cost, _: Option<&_>| {
            Some(MockInBeam {
                state,
                cost,
                g,
                f,
                in_beam: RefCell::new(true),
            })
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, state, cost, g, f, constructor);
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
        assert_eq!(node.cost(), 0);
        assert_eq!(node.g(), 0);
        assert_eq!(node.f(), 1);
        assert!(!node.in_beam());
        let peek = beam.pop();
        assert_eq!(peek, None);
    }

    #[test]
    fn normal_beam_drain() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::<i32, MockInBeam>::new(model);
        let mut beam = Beam::new(1);

        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![1, 2, 3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let cost = 1;
        let g = 0;
        let f = 2;
        let constructor = |state, cost, _: Option<&_>| {
            Some(MockInBeam {
                state,
                cost,
                g,
                f,
                in_beam: RefCell::new(true),
            })
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, state, cost, g, f, constructor);
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
        let g = 0;
        let f = 1;
        let constructor = |state, cost, _: Option<&_>| {
            Some(MockInBeam {
                state,
                cost,
                g,
                f,
                in_beam: RefCell::new(true),
            })
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, state, cost, g, f, constructor);
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
        let g = 0;
        let f = 2;
        let constructor = |state, cost, _: Option<&_>| {
            Some(MockInBeam {
                state,
                cost,
                g,
                f,
                in_beam: RefCell::new(true),
            })
        };
        let (generated, beam_pruning) = beam.insert(&mut registry, state, cost, g, f, constructor);
        assert!(!generated);
        assert!(beam_pruning);

        let mut iter = BeamInterface::<i32, i32, MockInBeam>::drain(&mut beam);
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
        assert_eq!(node.cost(), 0);
        assert_eq!(node.g(), 0);
        assert_eq!(node.f(), 1);
        assert!(node.in_beam());
        assert_eq!(iter.next(), None);
    }
}
