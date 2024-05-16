//! A module for state registries for duplicate detection.

use crate::search_algorithm::{
    self,
    data_structure::{HashableSignatureVariables, StateInformation},
    InsertionResult, StateInRegistry,
};
use dashmap::{DashMap, TryReserveError};
use dypdl::{prelude::*, variable_type::Numeric, ReduceFunction};
use rustc_hash::FxHasher;
use std::cmp::Ordering;
use std::hash::BuildHasherDefault;
use std::sync::Arc;

/// Registry storing generated states considering dominance relationship.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{
///     search_algorithm::{data_structure::StateInformation, StateInRegistry},
///     parallel_search_algorithm::{ConcurrentStateRegistry, SendableFNode},
/// };
/// use std::sync::Arc;
///
/// let mut model = Model::default();
/// let signature = model.add_integer_variable("signature", 1).unwrap();
/// let resource = model.add_integer_resource_variable("resource", false, 1).unwrap();
///
/// let mut increment = Transition::new("increment");
/// increment.set_cost(IntegerExpression::Cost + 1);
/// increment.add_effect(signature, signature + 1).unwrap();
///
/// let mut increase_cost = Transition::new("increase_cost");
/// increase_cost.set_cost(IntegerExpression::Cost + 1);
///
/// let mut consume = Transition::new("consume");
/// consume.set_cost(IntegerExpression::Cost);
/// consume.add_effect(resource, resource - 1).unwrap();
///
/// let mut produce = Transition::new("produce");
/// produce.set_cost(IntegerExpression::Cost);
/// produce.add_effect(resource, resource + 1).unwrap();
///
/// let model = Arc::new(model);
/// let registry = ConcurrentStateRegistry::new(model.clone());
///
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let h_evaluator = |_: &_, _: &mut _| Some(0);
/// let f_evaluator = |g, h, _: &_| g + h;
///
/// let node = SendableFNode::<_, Transition>::generate_root_node(
///     model.target.clone(),
///     &mut function_cache,
///     0,
///     &model,
///     &h_evaluator,
///     &f_evaluator,
///     None,
/// ).unwrap();
/// assert_eq!(registry.get(node.state(), node.cost(&model)), None);
/// let result = registry.insert(node.clone());
/// let information = result.information.unwrap();
/// assert_eq!(information.state(), node.state());
/// assert_eq!(information.cost(&model), node.cost(&model));
/// assert_eq!(information.bound(&model), node.bound(&model));
/// assert!(result.dominated.is_empty());
/// let got = registry.get(node.state(), node.cost(&model)).unwrap();
/// assert_eq!(node.state(), got.state());
/// assert_eq!(node.cost(&model), got.cost(&model));
/// assert_eq!(node.bound(&model), got.bound(&model));
///
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let irrelevant: StateInRegistry<_> = increment.apply(
///     node.state(), &mut function_cache, &model.state_functions, &model.table_registry,
/// );
/// let cost = increment.eval_cost(
///     node.cost(&model),  
///     node.state(),
///     &mut function_cache,
///     &model.state_functions,
///     &model.table_registry,
/// );
/// assert_eq!(registry.get(&irrelevant, cost), None);
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let irrelevant = SendableFNode::<_, Transition>::generate_root_node(
///     irrelevant, &mut function_cache, cost, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let result = registry.insert(irrelevant.clone());
/// let information = result.information.unwrap();
/// assert_eq!(information.state(), irrelevant.state());
/// assert_eq!(information.cost(&model), irrelevant.cost(&model));
/// assert_eq!(information.bound(&model), irrelevant.bound(&model));
/// assert!(result.dominated.is_empty());
///
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let dominated: StateInRegistry<_> = increase_cost.apply(
///     node.state(), &mut function_cache, &model.state_functions, &model.table_registry,
/// );
/// let cost = consume.eval_cost(
///     node.cost(&model),
///     node.state(),
///     &mut function_cache,
///     &model.state_functions,
///     &model.table_registry,
/// );
/// let dominating = registry.get(&dominated, cost).unwrap();
/// assert_eq!(dominating.state(), node.state());
/// assert_eq!(dominating.cost(&model), node.cost(&model));
/// assert_eq!(dominating.bound(&model), node.bound(&model));
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let dominated = SendableFNode::<_, Transition>::generate_root_node(
///     dominated, &mut function_cache, cost, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let result = registry.insert(dominated);
/// assert!(result.information.is_none());
///
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let dominating: StateInRegistry<_> = produce.apply(
///     node.state(), &mut function_cache, &model.state_functions, &model.table_registry,
/// );
/// let cost = produce.eval_cost(
///     node.cost(&model),
///     node.state(),
///     &mut function_cache,
///     &model.state_functions,
///     &model.table_registry,
/// );
/// assert_eq!(registry.get(&dominating, cost), None);
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let dominating = SendableFNode::<_, Transition>::generate_root_node(
///     dominating, &mut function_cache, cost, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let result = registry.insert(dominating.clone());
/// let information = result.information.unwrap();
/// assert_eq!(information.state(), dominating.state());
/// assert_eq!(information.cost(&model), dominating.cost(&model));
/// assert_eq!(information.bound(&model), dominating.bound(&model));
/// assert_eq!(result.dominated.len(), 1);
/// assert_eq!(result.dominated[0].state(), node.state());
/// assert_eq!(result.dominated[0].cost(&model), node.cost(&model));
/// assert_eq!(result.dominated[0].bound(&model), node.bound(&model));
/// let got = registry.get(node.state(), node.cost(&model)).unwrap();
/// assert_eq!(dominating.state(), got.state());
/// assert_eq!(dominating.cost(&model), got.cost(&model));
/// assert_eq!(dominating.bound(&model), got.bound(&model));
/// ```
pub struct ConcurrentStateRegistry<T, I>
where
    T: Numeric,
    I: StateInformation<T, Arc<HashableSignatureVariables>>,
{
    registry: DashMap<Arc<HashableSignatureVariables>, Vec<Arc<I>>, BuildHasherDefault<FxHasher>>,
    model: Arc<Model>,
    phantom: std::marker::PhantomData<T>,
}

impl<T, I> ConcurrentStateRegistry<T, I>
where
    T: Numeric,
    I: StateInformation<T, Arc<HashableSignatureVariables>>,
{
    /// Creates a new state registry.
    #[inline]
    pub fn new(model: Arc<Model>) -> ConcurrentStateRegistry<T, I> {
        ConcurrentStateRegistry {
            registry: DashMap::default(),
            model,
            phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn with_capacity_and_shard_amount(
        model: Arc<Model>,
        capacity: usize,
        shard_amount: usize,
    ) -> ConcurrentStateRegistry<T, I> {
        ConcurrentStateRegistry {
            registry: DashMap::with_capacity_and_hasher_and_shard_amount(
                capacity,
                BuildHasherDefault::<FxHasher>::default(),
                shard_amount,
            ),
            model,
            phantom: std::marker::PhantomData,
        }
    }

    /// Tries to reserve capacity.
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.registry.try_reserve(additional)
    }

    /// Returns a pointer to the model.
    #[inline]
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Deletes all entries.
    #[inline]
    pub fn clear(&mut self) {
        self.registry.clear();
    }

    /// Gets the information of a state that dominates the given state.
    pub fn get(
        &self,
        state: &StateInRegistry<Arc<HashableSignatureVariables>>,
        cost: T,
    ) -> Option<Arc<I>> {
        if let Some(v) = self.registry.get(&state.signature_variables) {
            for other in v.value() {
                let result = self.model.state_metadata.dominance(state, other.state());
                match result {
                    Some(Ordering::Equal) | Some(Ordering::Less)
                        if (self.model.reduce_function == ReduceFunction::Max
                            && cost <= other.cost(&self.model))
                            || (self.model.reduce_function == ReduceFunction::Min
                                && cost >= other.cost(&self.model)) =>
                    {
                        return Some(other.clone())
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Checks if the state is dominated by existing states.
    pub fn contains_state(
        &self,
        state: &StateInRegistry<Arc<HashableSignatureVariables>>,
        cost: T,
    ) -> bool {
        if let Some(v) = self.registry.get(&state.signature_variables) {
            for other in v.value() {
                let result = self.model.state_metadata.dominance(state, other.state());
                match result {
                    Some(Ordering::Equal) | Some(Ordering::Less)
                        if (self.model.reduce_function == ReduceFunction::Max
                            && cost <= other.cost(&self.model))
                            || (self.model.reduce_function == ReduceFunction::Min
                                && cost >= other.cost(&self.model)) =>
                    {
                        return true
                    }
                    _ => {}
                }
            }
        }
        false
    }

    /// Checks if the information is dominated by existing states.
    #[inline]
    pub fn contains(&self, information: &I) -> bool {
        let state = information.state();
        let cost = information.cost(&self.model);
        self.contains_state(state, cost)
    }

    /// Inserts a state and its information in the registry if it is not dominated by existing states
    /// given the state, its cost, and a constructor for the information.
    ///
    /// The constructor takes a state and its cost as arguments and returns the information.
    /// In addition, if exactly the same state is already saved in the registry,
    /// its information is passed as the last argument.
    /// It might be used to avoid recomputing values depending only on the state.
    /// The constructor is called only when the state is not dominated.
    pub fn insert_with<F>(
        &self,
        mut state: StateInRegistry<Arc<HashableSignatureVariables>>,
        cost: T,
        constructor: F,
    ) -> InsertionResult<Arc<I>>
    where
        F: FnOnce(StateInRegistry<Arc<HashableSignatureVariables>>, T, Option<&I>) -> Option<I>,
    {
        // Checks if the state is dominated using only a read lock.
        if self.contains_state(&state, cost) {
            return InsertionResult {
                information: None,
                dominated: smallvec::smallvec![],
            };
        }

        let entry = self.registry.entry(state.signature_variables.clone());
        match entry {
            dashmap::mapref::entry::Entry::Occupied(entry) => {
                // use signature variables already stored
                state.signature_variables = entry.key().clone();

                let mut entry = entry.into_ref();
                let v = entry.value_mut();
                let (dominating, dominated, same_state_index) =
                    search_algorithm::data_structure::remove_dominated(
                        v,
                        &self.model,
                        &state,
                        cost,
                    );

                if dominating {
                    return InsertionResult {
                        information: None,
                        dominated: smallvec::smallvec![],
                    };
                }

                let same_state_information = same_state_index.map(|i| dominated[i].as_ref());

                if let Some(information) =
                    constructor(state, cost, same_state_information).map(Arc::from)
                {
                    v.push(information.clone());

                    InsertionResult {
                        information: Some(information),
                        dominated,
                    }
                } else {
                    InsertionResult {
                        information: None,
                        dominated,
                    }
                }
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                if let Some(information) = constructor(state, cost, None).map(Arc::from) {
                    entry.insert(vec![information.clone()]);

                    InsertionResult {
                        information: Some(information),
                        dominated: smallvec::smallvec![],
                    }
                } else {
                    InsertionResult {
                        information: None,
                        dominated: smallvec::smallvec![],
                    }
                }
            }
        }
    }

    /// Inserts state information.
    pub fn insert(&self, mut information: I) -> InsertionResult<Arc<I>> {
        // Checks if the information is dominated using only a read lock.
        if self.contains(&information) {
            return InsertionResult {
                information: None,
                dominated: smallvec::smallvec![],
            };
        }

        let entry = self
            .registry
            .entry(information.state().signature_variables.clone());
        let (mut entry, removed) = match entry {
            dashmap::mapref::entry::Entry::Occupied(entry) => {
                // use signature variables already stored
                information.state_mut().signature_variables = entry.key().clone();

                let mut entry = entry.into_ref();
                let v = entry.value_mut();
                let (dominating, dominated, _) = search_algorithm::data_structure::remove_dominated(
                    v,
                    &self.model,
                    information.state(),
                    information.cost(&self.model),
                );

                if dominating {
                    return InsertionResult {
                        information: None,
                        dominated: smallvec::smallvec![],
                    };
                }

                (entry, dominated)
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                (entry.insert(Vec::with_capacity(1)), smallvec::smallvec![])
            }
        };

        let information = Arc::new(information);
        entry.push(information.clone());

        InsertionResult {
            information: Some(information),
            dominated: removed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::Integer;
    use rustc_hash::FxHashMap;
    use smallvec::SmallVec;
    use std::cell::Cell;

    #[derive(Debug, Clone)]
    struct MockInformation {
        state: StateInRegistry<Arc<HashableSignatureVariables>>,
        cost: i32,
        value: Cell<Option<i32>>,
    }

    impl StateInformation<Integer, Arc<HashableSignatureVariables>> for MockInformation {
        fn state(&self) -> &StateInRegistry<Arc<HashableSignatureVariables>> {
            &self.state
        }

        fn state_mut(&mut self) -> &mut StateInRegistry<Arc<HashableSignatureVariables>> {
            &mut self.state
        }

        fn cost(&self, _: &Model) -> Integer {
            self.cost
        }

        fn bound(&self, _: &Model) -> Option<Integer> {
            None
        }

        fn is_closed(&self) -> bool {
            false
        }

        fn close(&self) {}
    }

    impl PartialEq for MockInformation {
        fn eq(&self, other: &Self) -> bool {
            self.cost == other.cost
        }
    }

    impl Eq for MockInformation {}

    impl PartialOrd for MockInformation {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for MockInformation {
        fn cmp(&self, other: &Self) -> Ordering {
            self.cost.cmp(&other.cost)
        }
    }

    fn generate_model() -> dypdl::Model {
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert("n1".to_string(), 0);
        name_to_integer_variable.insert("n2".to_string(), 1);
        name_to_integer_variable.insert("n3".to_string(), 2);

        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert("r1".to_string(), 0);
        name_to_integer_resource_variable.insert("r2".to_string(), 1);
        name_to_integer_resource_variable.insert("r3".to_string(), 2);

        let state_metadata = dypdl::StateMetadata {
            integer_variable_names: vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
            name_to_integer_variable,
            integer_resource_variable_names: vec![
                "r1".to_string(),
                "r2".to_string(),
                "r3".to_string(),
            ],
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true],
            ..Default::default()
        };
        dypdl::Model {
            state_metadata,
            reduce_function: dypdl::ReduceFunction::Min,
            ..Default::default()
        }
    }

    fn generate_signature_variables(
        integer_variables: Vec<Integer>,
    ) -> Arc<HashableSignatureVariables> {
        Arc::new(HashableSignatureVariables {
            integer_variables,
            ..Default::default()
        })
    }

    fn generate_resource_variables(integer_variables: Vec<Integer>) -> dypdl::ResourceVariables {
        dypdl::ResourceVariables {
            integer_variables,
            ..Default::default()
        }
    }

    #[test]
    fn insert_get_new_information() {
        let model = Arc::new(generate_model());
        let registry = ConcurrentStateRegistry::<i32, MockInformation>::new(model);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = MockInformation {
            state: state.clone(),
            cost: 1,
            value: Cell::new(None),
        };
        assert!(!registry.contains(&information));
        let result = registry.insert(information);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = MockInformation {
            state: state.clone(),
            cost: 1,
            value: Cell::new(None),
        };
        assert!(!registry.contains(&information));
        let result = registry.insert(information);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert!(registry.contains(&information));
        assert_eq!(registry.get(&state, 1), Some(information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = MockInformation {
            state: state.clone(),
            cost: 1,
            value: Cell::new(None),
        };
        assert!(!registry.contains(&information));
        let result = registry.insert(information);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert!(registry.contains(&information));
        assert_eq!(registry.get(&state, 1), Some(information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information = MockInformation {
            state: state.clone(),
            cost: 0,
            value: Cell::new(None),
        };
        assert!(!registry.contains(&information));
        let result = registry.insert(information);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert!(registry.contains(&information));
        assert_eq!(registry.get(&state, 0), Some(information));
    }

    #[test]
    fn insert_information_dominated() {
        let model = Arc::new(generate_model());
        let registry = ConcurrentStateRegistry::<i32, MockInformation>::new(model);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let previous = MockInformation {
            state: state.clone(),
            cost: 1,
            value: Cell::new(None),
        };
        let result = registry.insert(previous);
        assert!(result.information.is_some());
        let previous = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(previous.state, state);
        assert_eq!(previous.value.get(), None);
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(previous.clone()));
        let information = MockInformation {
            state: state.clone(),
            cost: 1,
            value: Cell::new(None),
        };
        assert!(registry.contains(&information));
        let result = registry.insert(information.clone());
        assert!(result.information.is_none());
        assert_eq!(registry.get(&state, 1), Some(previous.clone()));
        assert!(registry.contains(&information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(previous.clone()));
        let information = MockInformation {
            state: state.clone(),
            cost: 1,
            value: Cell::new(None),
        };
        assert!(registry.contains(&information));
        let result = registry.insert(information.clone());
        assert!(result.information.is_none());
        assert_eq!(registry.get(&state, 1), Some(previous.clone()));
        assert!(registry.contains(&information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 2), Some(previous.clone()));
        let information = MockInformation {
            state: state.clone(),
            cost: 2,
            value: Cell::new(None),
        };
        assert!(registry.contains(&information));
        let result = registry.insert(information.clone());
        assert!(result.information.is_none());
        assert_eq!(registry.get(&state, 2), Some(previous));
        assert!(registry.contains(&information));
    }

    #[test]
    fn insert_get_dominating_information() {
        let model = Arc::new(generate_model());
        let registry = ConcurrentStateRegistry::<i32, MockInformation>::new(model);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = MockInformation {
            state: state.clone(),
            cost: 1,
            value: Cell::new(None),
        };
        assert!(!registry.contains(&information));
        let result = registry.insert(information.clone());
        assert!(result.information.is_some());
        let information1 = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(information1.state, state);
        assert_eq!(information1.value.get(), None);
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(
            registry.get(&information1.state, 1),
            Some(information1.clone())
        );
        assert!(registry.contains(&information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information = MockInformation {
            state,
            cost: 0,
            value: Cell::new(None),
        };
        assert!(!registry.contains(&information));
        let result = registry.insert(information.clone());
        assert!(result.information.is_some());
        let information2 = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(
            information2.state.signature_variables,
            information1.state.signature_variables
        );
        assert_eq!(
            information2.state.resource_variables,
            information1.state.resource_variables
        );
        assert_eq!(information2.value.get(), None);
        assert_eq!(dominated, SmallVec::<[_; 1]>::from([information1.clone()]));
        assert_eq!(
            registry.get(&information1.state, 0),
            Some(information2.clone())
        );
        assert_eq!(
            registry.get(&information2.state, 0),
            Some(information2.clone())
        );
        assert!(registry.contains(&information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 1, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information = MockInformation {
            state,
            cost: 0,
            value: Cell::new(None),
        };
        assert!(!registry.contains(&information));
        let result = registry.insert(information.clone());
        assert!(result.information.is_some());
        let information3 = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(
            information3.state.signature_variables,
            information2.state.signature_variables
        );
        assert_ne!(
            information3.state.resource_variables,
            information2.state.resource_variables,
        );
        assert_eq!(information3.value.get(), None);
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(
            registry.get(&information1.state, 1),
            Some(information2.clone())
        );
        assert_eq!(
            registry.get(&information2.state, 0),
            Some(information2.clone())
        );
        assert_eq!(
            registry.get(&information3.state, 0),
            Some(information3.clone())
        );
        assert!(registry.contains(&information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information = MockInformation {
            state,
            cost: 0,
            value: Cell::new(None),
        };
        assert!(!registry.contains(&information));
        let result = registry.insert(information.clone());
        assert!(result.information.is_some());
        let information4 = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(
            information4.state.signature_variables,
            information2.state.signature_variables
        );
        assert_eq!(
            information4.state.signature_variables,
            information3.state.signature_variables
        );
        assert_ne!(
            information4.state.resource_variables,
            information2.state.resource_variables,
        );
        assert_ne!(
            information4.state.resource_variables,
            information3.state.resource_variables,
        );
        assert_eq!(information4.value.get(), None);
        assert_eq!(
            dominated,
            SmallVec::<[_; 1]>::from(vec![information2.clone(), information3.clone()])
        );
        assert_eq!(
            registry.get(&information1.state, 1),
            Some(information4.clone())
        );
        assert_eq!(
            registry.get(&information2.state, 0),
            Some(information4.clone())
        );
        assert_eq!(
            registry.get(&information3.state, 0),
            Some(information4.clone())
        );
        assert_eq!(registry.get(&information4.state, 0), Some(information4));
        assert!(registry.contains(&information));
    }

    #[test]
    fn insert_with_get_new_information() {
        let model = Arc::new(generate_model());
        let registry = ConcurrentStateRegistry::<i32, MockInformation>::new(model);
        let constructor = |state: _, cost: Integer, _: Option<&MockInformation>| {
            Some(MockInformation {
                state,
                cost,
                value: Cell::new(None),
            })
        };

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        assert!(!registry.contains_state(&state, 1));
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(information));
        assert!(registry.contains_state(&state, 1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        assert!(!registry.contains_state(&state, 1));
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(information));
        assert!(registry.contains_state(&state, 1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        assert!(!registry.contains_state(&state, 0));
        let result = registry.insert_with(state, 0, constructor);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 0), Some(information));
        assert!(registry.contains_state(&state, 0));
    }

    #[test]
    fn insert_with_information_dominated() {
        let model = Arc::new(generate_model());
        let registry = ConcurrentStateRegistry::<i32, MockInformation>::new(model);

        let constructor = |state: _, cost: Integer, _: Option<&MockInformation>| {
            Some(MockInformation {
                state,
                cost,
                value: Cell::new(None),
            })
        };

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_some());
        let previous = result.information.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(previous.state, state);
        assert_eq!(previous.value.get(), None);
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(previous.clone()));
        assert!(registry.contains_state(&state, 1));
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_none());
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(previous.clone()));
        assert!(registry.contains_state(&state, 1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(previous.clone()));
        assert!(registry.contains_state(&state, 1));
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_none());
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(previous.clone()));
        assert!(registry.contains_state(&state, 1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 2), Some(previous.clone()));
        assert!(registry.contains_state(&state, 2));
        let result = registry.insert_with(state, 2, constructor);
        assert!(result.information.is_none());
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 2), Some(previous));
        assert!(registry.contains_state(&state, 2));
    }

    #[test]
    fn insert_with_get_dominating_information() {
        let model = Arc::new(generate_model());
        let registry = ConcurrentStateRegistry::<i32, MockInformation>::new(model);
        let constructor = |state: _, cost: Integer, other: Option<&MockInformation>| {
            if let Some(other) = other {
                other.value.get().map(|value| MockInformation {
                    state,
                    cost,
                    value: Cell::new(Some(value)),
                })
            } else {
                Some(MockInformation {
                    state,
                    cost,
                    value: Cell::new(None),
                })
            }
        };

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        assert!(!registry.contains_state(&state, 1));
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_some());
        let information1 = result.information.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information1.state, state);
        assert_eq!(information1.value.get(), None);
        information1.value.set(Some(10));
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(
            registry.get(&information1.state, 1),
            Some(information1.clone())
        );
        assert!(registry.contains_state(&state, 1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        assert!(!registry.contains_state(&state, 0));
        let result = registry.insert_with(state, 0, constructor);
        assert!(result.information.is_some());
        let information2 = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(
            information2.state.signature_variables,
            information1.state.signature_variables
        );
        assert_eq!(
            information2.state.resource_variables,
            information1.state.resource_variables
        );
        assert_eq!(information2.value.get(), Some(10));
        assert_eq!(dominated, SmallVec::<[_; 1]>::from([information1.clone()]));
        assert_eq!(
            registry.get(&information1.state, 0),
            Some(information2.clone())
        );
        assert_eq!(
            registry.get(&information2.state, 0),
            Some(information2.clone())
        );
        assert!(registry.contains_state(&information2.state, 0));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 1, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        assert!(!registry.contains_state(&state, 0));
        let result = registry.insert_with(state, 0, constructor);
        assert!(result.information.is_some());
        let information3 = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(
            information3.state.signature_variables,
            information2.state.signature_variables
        );
        assert_ne!(
            information3.state.resource_variables,
            information2.state.resource_variables,
        );
        assert_eq!(information3.value.get(), None);
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(
            registry.get(&information1.state, 1),
            Some(information3.clone())
        );
        assert_eq!(
            registry.get(&information2.state, 0),
            Some(information3.clone())
        );
        assert!(registry.contains_state(&information3.state, 0));
        assert_eq!(
            registry.get(&information3.state, 0),
            Some(information3.clone())
        );

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        assert!(!registry.contains_state(&state, 0));
        let result = registry.insert_with(state, 0, constructor);
        assert!(result.information.is_some());
        let information4 = result.information.unwrap();
        let dominated = result.dominated;
        assert_eq!(
            information4.state.signature_variables,
            information2.state.signature_variables
        );
        assert_eq!(
            information4.state.signature_variables,
            information3.state.signature_variables
        );
        assert_ne!(
            information4.state.resource_variables,
            information2.state.resource_variables,
        );
        assert_ne!(
            information4.state.resource_variables,
            information3.state.resource_variables,
        );
        assert_eq!(information4.value.get(), None);
        assert_eq!(
            dominated,
            SmallVec::<[_; 1]>::from(vec![information2.clone(), information3.clone()])
        );
        assert_eq!(
            registry.get(&information1.state, 1),
            Some(information4.clone())
        );
        assert_eq!(
            registry.get(&information2.state, 0),
            Some(information4.clone())
        );
        assert_eq!(
            registry.get(&information3.state, 0),
            Some(information4.clone())
        );
        assert!(registry.contains_state(&information4.state, 0));
        assert_eq!(registry.get(&information4.state, 0), Some(information4));
    }

    #[test]
    fn get_model() {
        let model = Arc::new(generate_model());
        let registry = ConcurrentStateRegistry::<i32, MockInformation>::new(model.clone());
        assert_eq!(registry.model(), &model);
    }

    #[test]
    fn reserve() {
        let model = Arc::new(generate_model());
        let mut registry = ConcurrentStateRegistry::<i32, MockInformation>::new(model);
        let result = registry.try_reserve(10);
        assert!(result.is_ok());
        assert!(registry.registry.capacity() >= 10);
    }

    #[test]
    fn clear() {
        let model = Arc::new(generate_model());
        let mut registry = ConcurrentStateRegistry::<i32, MockInformation>::new(model);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let constructor = |state: _, cost: Integer, _: Option<&MockInformation>| {
            Some(MockInformation {
                state,
                cost,
                value: Cell::new(None),
            })
        };
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information.state, state);
        assert_eq!(
            registry.get(&information.state, 1),
            Some(information.clone())
        );

        registry.clear();

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&information.state, 1), None);
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_some());
        let information = result.information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information.state, state);
        assert_eq!(registry.get(&information.state, 1), Some(information));
    }
}
