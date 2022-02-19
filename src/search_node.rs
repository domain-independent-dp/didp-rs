use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections;
use std::rc::Rc;

use crate::state;

#[derive(Debug, Eq)]
pub struct SearchNode<T: Ord + Copy> {
    pub state: state::State,
    pub g: T,
    pub h: RefCell<Option<T>>,
    pub f: RefCell<Option<T>>,
    pub parent: Rc<SearchNode<T>>,
    pub closed: RefCell<bool>,
}

impl<T: Ord + Copy> Ord for SearchNode<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.f.cmp(&other.f).reverse()
    }
}

impl<T: Ord + Copy> PartialOrd for SearchNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord + Copy> PartialEq for SearchNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

pub struct StateRegistry<T: Ord + Copy>(
    collections::HashMap<Rc<state::SignatureVariables>, Vec<Rc<SearchNode<T>>>>,
);

impl<T: Ord + Copy> StateRegistry<T> {
    pub fn new() -> StateRegistry<T> {
        StateRegistry(collections::HashMap::new())
    }

    pub fn with_capcaity(capacity: usize) -> StateRegistry<T> {
        StateRegistry(collections::HashMap::with_capacity(capacity))
    }

    pub fn get_node(
        &mut self,
        mut state: state::State,
        g: T,
        parent: Rc<SearchNode<T>>,
    ) -> Option<Rc<SearchNode<T>>> {
        let entry = self.0.entry(state.signature_variables.clone());
        let v = match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                state.signature_variables = entry.key().clone();
                let v = entry.into_mut();
                for (i, other) in v.iter().enumerate() {
                    let result = other
                        .state
                        .resource_variables
                        .partial_cmp(&state.resource_variables);
                    match result {
                        Some(Ordering::Equal) | Some(Ordering::Greater) if g >= other.g => {
                            // dominated
                            return None;
                        }
                        Some(Ordering::Equal) | Some(Ordering::Less) if g <= other.g => {
                            // dominating
                            if !*other.closed.borrow() {
                                *other.closed.borrow_mut() = true;
                            }
                            let h = match result.unwrap() {
                                Ordering::Equal => {
                                    if let Some(h) = *other.h.borrow() {
                                        // cached value
                                        RefCell::new(Some(h))
                                    } else {
                                        // dead end
                                        return None;
                                    }
                                }
                                _ => RefCell::new(None),
                            };
                            let node = Rc::new(SearchNode {
                                state,
                                g,
                                h,
                                f: RefCell::new(None),
                                parent,
                                closed: RefCell::new(false),
                            });
                            v[i] = node.clone();
                            return Some(node);
                        }
                        _ => {}
                    }
                }
                v
            }
            collections::hash_map::Entry::Vacant(entry) => entry.insert(Vec::with_capacity(1)),
        };
        let node = Rc::new(SearchNode {
            state,
            g,
            h: RefCell::new(None),
            f: RefCell::new(None),
            parent,
            closed: RefCell::new(false),
        });
        v.push(node.clone());
        Some(node)
    }
}
