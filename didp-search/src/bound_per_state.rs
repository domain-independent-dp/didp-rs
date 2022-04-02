use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections;
use std::rc::Rc;

pub struct BoundPerState<'a, T> {
    registry: FxHashMap<Rc<didp_parser::SignatureVariables>, Vec<(didp_parser::State, T)>>,
    metadata: &'a didp_parser::StateMetadata,
}

impl<'a, T> BoundPerState<'a, T> {
    pub fn new<U: variable::Numeric>(model: &'a didp_parser::Model<U>) -> BoundPerState<'a, T> {
        BoundPerState {
            registry: FxHashMap::default(),
            metadata: &model.state_metadata,
        }
    }

    pub fn reserve(&mut self, capacity: usize) {
        self.registry.reserve(capacity);
    }

    pub fn clear(&mut self) {
        self.registry.clear();
    }

    pub fn get(&self, state: &didp_parser::State) -> Option<&T> {
        if let Some(v) = self.registry.get(&state.signature_variables) {
            for (other, value) in v.iter() {
                let result = self.metadata.dominance(state, other);
                match result {
                    Some(Ordering::Equal) | Some(Ordering::Less) => {
                        // dominated
                        return Some(value);
                    }
                    _ => {}
                }
            }
        }
        None
    }

    pub fn insert(&mut self, mut state: didp_parser::State, value: T) {
        let entry = self.registry.entry(state.signature_variables.clone());
        match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                state.signature_variables = entry.key().clone();
                let v = entry.into_mut();
                for other in v.iter_mut() {
                    let result = self.metadata.dominance(&state, &other.0);
                    match result {
                        Some(Ordering::Equal) | Some(Ordering::Less) => {
                            // dominated
                            return;
                        }
                        Some(Ordering::Greater) => {
                            // dominating
                            *other = (state, value);
                            return;
                        }
                        _ => {}
                    }
                }
                v.push((state, value));
            }
            collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(vec![(state, value)]);
            }
        }
    }

    pub fn remove(&mut self, state: &didp_parser::State) {
        if let Some(v) = self.registry.get_mut(&state.signature_variables) {
            let metadata = self.metadata;
            v.retain(|x| metadata.dominance(state, &x.0).is_some())
        }
    }
}
