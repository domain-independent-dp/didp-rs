use super::beam_search_node::*;
use crate::state_registry::{StateInRegistry, StateRegistry};
use dypdl::variable_type::Numeric;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
use std::collections;
use std::mem;
use std::rc::Rc;

/// Beam for beam epsilon beam search.
#[derive(Debug, Clone)]
pub struct EpsilonBeam<T: Numeric, U: Numeric + Ord> {
    /// Capacity of the beam, or the beam size.
    pub capacity: usize,
    size: usize,
    beam: BeamBase<T, U>,
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
            epsilon,
            rng: rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed),
        }
    }

    fn clean_queue_garbage(&mut self) {
        let mut peek = self.beam.queue.peek();
        while peek.map_or(false, |node| !*node.in_beam.borrow()) {
            self.beam.queue.pop();
            peek = self.beam.queue.peek();
        }
    }

    fn clean_pool_garbage(&mut self) {
        let mut peek = self.beam.pool.last();
        while peek.map_or(false, |node| !*node.in_beam.borrow()) {
            self.beam.pool.pop();
            peek = self.beam.pool.last();
        }
    }
}

impl<T: Numeric, U: Numeric + Ord> Beam<T, U> for EpsilonBeam<T, U> {
    fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn drain(&mut self) -> NodesInBeam<'_, T, U> {
        self.size = 0;
        self.beam.drain()
    }

    fn pop(&mut self) -> Option<Rc<BeamSearchNode<T, U>>> {
        if let Some(node) = self.beam.queue.pop() {
            *node.in_beam.borrow_mut() = false;
            self.size -= 1;
            self.clean_queue_garbage();
            Some(node)
        } else if let Some(node) = self.beam.pool.pop() {
            self.clean_pool_garbage();
            let mut node = node;
            loop {
                let index = self.rng.gen::<usize>() % (self.beam.pool.len() + 1);
                if index == self.beam.pool.len() {
                    break;
                }
                mem::swap(&mut self.beam.pool[index], &mut node);
                if *(self.beam.pool[index].in_beam.borrow()) {
                    break;
                }
                node = self.beam.pool.pop().unwrap();
                self.clean_pool_garbage();
            }
            *node.in_beam.borrow_mut() = false;
            self.size -= 1;
            Some(node)
        } else {
            None
        }
    }

    fn insert(
        &mut self,
        registry: &mut StateRegistry<'_, T, Rc<BeamSearchNode<T, U>>>,
        state: StateInRegistry,
        cost: T,
        args: BeamSearchNodeArgs<T, U>,
    ) {
        let must_keep = self.rng.gen::<f64>() <= self.epsilon;
        if must_keep
            || self.size < self.capacity
            || self.beam.queue.peek().map_or(true, |node| args.f < node.f)
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
                        self.clean_queue_garbage();
                        self.clean_pool_garbage();
                    }
                }
                if self.size == self.capacity {
                    self.pop();
                }
                if self.size < self.capacity {
                    if must_keep {
                        self.beam.pool.push(node);
                    } else {
                        self.beam.queue.push(node);
                    }
                    self.size += 1;
                }
            }
        }
    }
}
