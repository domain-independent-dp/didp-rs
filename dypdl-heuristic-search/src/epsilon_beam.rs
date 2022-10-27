use super::beam::*;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::transition_with_custom_cost::TransitionWithCustomCost;
use dypdl::variable_type::Numeric;
use dypdl::Transition;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::mem;
use std::rc::Rc;

/// Node for epsilon beam search.
#[derive(Debug, Default)]
pub struct EpsilonBeamSearchNode<T: Numeric, U: Numeric> {
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
    /// If included in a queue.
    pub in_queue: RefCell<bool>,
}

impl<T: Numeric, U: Numeric> InBeam for Rc<EpsilonBeamSearchNode<T, U>> {
    fn in_queue(&self) -> bool {
        *self.in_queue.borrow()
    }

    fn in_pool(&self) -> bool {
        *self.in_beam.borrow()
    }
}

impl<T: Numeric, U: Numeric> PrioritizedNode<U> for Rc<EpsilonBeamSearchNode<T, U>> {
    fn g(&self) -> U {
        self.g
    }

    fn f(&self) -> U {
        self.f
    }
}

impl<T: Numeric, U: Numeric + PartialOrd> PartialEq for EpsilonBeamSearchNode<T, U> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.g == other.g
    }
}

impl<T: Numeric, U: Numeric + Ord> Eq for EpsilonBeamSearchNode<T, U> {}

impl<T: Numeric, U: Numeric + Ord> Ord for EpsilonBeamSearchNode<T, U> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => other.g.cmp(&self.g),
            result => result,
        }
    }
}

impl<T: Numeric, U: Numeric + Ord> PartialOrd for EpsilonBeamSearchNode<T, U> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric, U: Numeric> StateInformation<T> for Rc<EpsilonBeamSearchNode<T, U>> {
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.cost
    }
}

impl<T: Numeric, U: Numeric> GetTransitions for Rc<EpsilonBeamSearchNode<T, U>> {
    fn transitions(&self) -> Vec<Transition> {
        self.transitions
            .iter()
            .map(|t| t.transition.clone())
            .collect()
    }
}

/// Beam for epsilon beam search.
#[derive(Debug, Clone)]
pub struct EpsilonBeam<T: Numeric, U: Numeric + Ord> {
    /// Capacity of the beam, or the beam size.
    pub capacity: usize,
    size: usize,
    beam: BeamBase<Rc<EpsilonBeamSearchNode<T, U>>>,
    all_pool: Vec<Rc<EpsilonBeamSearchNode<T, U>>>,
    epsilon: f64,
    rng: rand_xoshiro::Xoshiro256StarStar,
}

impl<T: Numeric, U: Numeric + Ord> EpsilonBeam<T, U> {
    pub fn new(capacity: usize, epsilon: f64, seed: u64) -> EpsilonBeam<T, U> {
        EpsilonBeam {
            capacity,
            size: 0,
            beam: BeamBase {
                queue: collections::BinaryHeap::with_capacity(capacity),
                pool: Vec::with_capacity(capacity),
            },
            all_pool: Vec::with_capacity(capacity),
            epsilon,
            rng: rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed),
        }
    }

    fn prepare_pool(&mut self) {
        for _ in 0..self.size {
            if self.rng.gen::<f64>() <= self.epsilon {
                while let Some(mut node) = self.all_pool.pop() {
                    let index = self.rng.gen::<usize>() / (self.all_pool.len() + 1);
                    if index < self.all_pool.len() {
                        mem::swap(&mut self.all_pool[index], &mut node);
                    }
                    if *node.in_beam.borrow() {
                        if *node.in_queue.borrow() {
                            *node.in_queue.borrow_mut() = false;
                            self.clean_queue_garbage();
                        } else if let Some(peek) = self.beam.queue.pop() {
                            *peek.in_queue.borrow_mut() = false;
                            self.clean_queue_garbage();
                        }
                        self.beam.pool.push(node);
                        break;
                    }
                }
            }
        }
        self.all_pool.clear();
    }

    fn clean_queue_garbage(&mut self) {
        let mut peek = self.beam.queue.peek();
        while peek.map_or(false, |node| !*node.in_queue.borrow()) {
            self.beam.queue.pop();
            peek = self.beam.queue.peek();
        }
    }

    fn clean_pool_garbage(&mut self) {
        let mut peek = self.all_pool.last();
        while peek.map_or(false, |node| !*node.in_beam.borrow()) {
            self.all_pool.pop();
            peek = self.all_pool.last();
        }
    }
}

impl<T: Numeric, U: Numeric + Ord> Beam<T, U, Rc<EpsilonBeamSearchNode<T, U>>>
    for EpsilonBeam<T, U>
{
    fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn drain(&mut self) -> BeamDrain<'_, Rc<EpsilonBeamSearchNode<T, U>>> {
        self.prepare_pool();
        self.size = 0;
        self.beam.drain()
    }

    fn insert(
        &mut self,
        registry: &mut StateRegistry<'_, T, Rc<EpsilonBeamSearchNode<T, U>>>,
        state: StateInRegistry,
        cost: T,
        args: BeamSearchNodeArgs<U, Rc<EpsilonBeamSearchNode<T, U>>>,
    ) -> bool {
        let constructor =
            |state: StateInRegistry, cost: T, _: Option<&Rc<EpsilonBeamSearchNode<T, U>>>| {
                let transitions = args.parent.map_or_else(Vec::new, |parent| {
                    Vec::from_iter(
                        parent
                            .transitions
                            .iter()
                            .cloned()
                            .chain(args.operator.into_iter()),
                    )
                });

                Some(Rc::new(EpsilonBeamSearchNode {
                    g: args.g,
                    f: args.f,
                    state,
                    cost,
                    transitions,
                    in_beam: RefCell::new(true),
                    in_queue: RefCell::new(false),
                }))
            };
        if let Some((node, dominated)) = registry.insert(state, cost, constructor) {
            if let Some(dominated) = dominated {
                if *dominated.in_beam.borrow() {
                    *dominated.in_beam.borrow_mut() = false;
                    self.clean_pool_garbage();
                    if *dominated.in_queue.borrow() {
                        self.size -= 1;
                        *dominated.in_queue.borrow_mut() = false;
                        self.clean_queue_garbage();
                    }
                }
            }
            self.all_pool.push(node.clone());
            if self.size < self.capacity
                || self.beam.queue.peek().map_or(true, |peek| {
                    (args.f < peek.f) || (args.f == peek.f && args.g > peek.g)
                })
            {
                let mut pruned_by_capacity = false;
                if self.size == self.capacity {
                    if let Some(peek) = self.beam.queue.pop() {
                        pruned_by_capacity = true;
                        self.size -= 1;
                        *peek.in_queue.borrow_mut() = false;
                        self.clean_queue_garbage();
                    }
                }
                if self.size < self.capacity {
                    *node.in_queue.borrow_mut() = true;
                    self.beam.queue.push(node);
                    self.size += 1;
                }
                !pruned_by_capacity
            } else {
                false
            }
        } else {
            true
        }
    }
}
