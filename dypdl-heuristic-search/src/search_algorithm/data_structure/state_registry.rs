//! A module for state registries for duplicate detection.

use super::hashable_state::{HashableSignatureVariables, StateWithHashableSignatureVariables};
use core::ops::Deref;
use dypdl::{prelude::*, variable_type::Numeric, ReduceFunction};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections;
use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;

/// State stored in a state registry using signature variables as the key.
///
/// Using continuous variables in signature variables is not recommended
/// because it may cause a numerical issue.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct StateInRegistry<K = Rc<HashableSignatureVariables>>
where
    K: Hash + Eq + Clone + Debug,
{
    /// Sharable pointer to signature variables.
    pub signature_variables: K,
    /// Resource variables.
    pub resource_variables: dypdl::ResourceVariables,
}

impl<K> From<dypdl::State> for StateInRegistry<K>
where
    K: Hash
        + Eq
        + Clone
        + Debug
        + Deref<Target = HashableSignatureVariables>
        + From<HashableSignatureVariables>,
{
    fn from(state: dypdl::State) -> StateInRegistry<K> {
        StateInRegistry {
            signature_variables: K::from(HashableSignatureVariables::from(
                state.signature_variables,
            )),
            resource_variables: state.resource_variables,
        }
    }
}

impl<K> From<StateWithHashableSignatureVariables> for StateInRegistry<K>
where
    K: Hash + Eq + Clone + Debug + From<HashableSignatureVariables>,
{
    fn from(state: StateWithHashableSignatureVariables) -> StateInRegistry<K> {
        StateInRegistry {
            signature_variables: K::from(state.signature_variables),
            resource_variables: state.resource_variables,
        }
    }
}

impl<K> dypdl::StateInterface for StateInRegistry<K>
where
    K: Hash + Eq + Clone + Debug + Deref<Target = HashableSignatureVariables>,
    StateInRegistry<K>: From<dypdl::State>,
{
    #[inline]
    fn get_number_of_set_variables(&self) -> usize {
        self.signature_variables.set_variables.len()
    }

    #[inline]
    fn get_set_variable(&self, i: usize) -> &Set {
        &self.signature_variables.set_variables[i]
    }

    #[inline]
    fn get_number_of_vector_variables(&self) -> usize {
        self.signature_variables.vector_variables.len()
    }

    #[inline]
    fn get_vector_variable(&self, i: usize) -> &Vector {
        &self.signature_variables.vector_variables[i]
    }

    #[inline]
    fn get_number_of_element_variables(&self) -> usize {
        self.signature_variables.element_variables.len()
    }

    #[inline]
    fn get_element_variable(&self, i: usize) -> Element {
        self.signature_variables.element_variables[i]
    }

    #[inline]
    fn get_number_of_integer_variables(&self) -> usize {
        self.signature_variables.integer_variables.len()
    }

    #[inline]
    fn get_integer_variable(&self, i: usize) -> Integer {
        self.signature_variables.integer_variables[i]
    }

    #[inline]
    fn get_number_of_continuous_variables(&self) -> usize {
        self.signature_variables.continuous_variables.len()
    }

    #[inline]
    fn get_continuous_variable(&self, i: usize) -> Continuous {
        self.signature_variables.continuous_variables[i].into_inner()
    }

    #[inline]
    fn get_number_of_element_resource_variables(&self) -> usize {
        self.resource_variables.element_variables.len()
    }

    #[inline]
    fn get_element_resource_variable(&self, i: usize) -> Element {
        self.resource_variables.element_variables[i]
    }

    #[inline]
    fn get_number_of_integer_resource_variables(&self) -> usize {
        self.resource_variables.integer_variables.len()
    }

    #[inline]
    fn get_integer_resource_variable(&self, i: usize) -> Integer {
        self.resource_variables.integer_variables[i]
    }

    #[inline]
    fn get_number_of_continuous_resource_variables(&self) -> usize {
        self.resource_variables.continuous_variables.len()
    }

    #[inline]
    fn get_continuous_resource_variable(&self, i: usize) -> Continuous {
        self.resource_variables.continuous_variables[i]
    }
}

/// Information stored in a state registry.
pub trait StateInformation<T, K = Rc<HashableSignatureVariables>>
where
    T: Numeric,
    K: Hash + Eq + Clone + Debug,
{
    /// Returns the state.
    fn state(&self) -> &StateInRegistry<K>;

    /// Returns a mutable reference to the state.
    fn state_mut(&mut self) -> &mut StateInRegistry<K>;

    /// Returns the cost of the state.
    fn cost(&self, model: &Model) -> T;

    /// Returns the dual bound of a solution through this node, which might be different from the f-value.
    fn bound(&self, model: &Model) -> Option<T>;

    /// Checks if closed.
    fn is_closed(&self) -> bool;

    /// Closes.
    fn close(&self);
}

/// Remove dominated state information from a vector of non-dominated information given a model, a state, and the cost of the state.
///
/// Returns a tuple of a boolean indicating if information in the vector dominates the given state,
/// a vector of dominated information removed from the input vector,
/// and the index of the information (in the returned vector) that has the same state as the given state.
pub fn remove_dominated<T, K, I, D>(
    non_dominated: &mut Vec<D>,
    model: &Model,
    state: &StateInRegistry<K>,
    cost: T,
) -> (bool, SmallVec<[D; 1]>, Option<usize>)
where
    T: Numeric,
    K: Hash
        + Eq
        + Clone
        + Debug
        + Deref<Target = HashableSignatureVariables>
        + From<HashableSignatureVariables>,
    I: StateInformation<T, K>,
    D: Deref<Target = I>,
{
    let mut same_state_index = None;
    let mut dominated_indices = SmallVec::<[usize; 1]>::new();

    for (i, other) in non_dominated.iter().enumerate() {
        match model.state_metadata.dominance(state, other.state()) {
            Some(Ordering::Equal) | Some(Ordering::Less)
                if (model.reduce_function == ReduceFunction::Max && cost <= other.cost(model))
                    || (model.reduce_function == ReduceFunction::Min
                        && cost >= other.cost(model)) =>
            {
                return (true, smallvec::smallvec![], None);
            }
            Some(Ordering::Equal) => {
                if same_state_index.is_none() {
                    same_state_index = Some(dominated_indices.len());
                }

                dominated_indices.push(i);
            }
            Some(Ordering::Greater)
                if (model.reduce_function == ReduceFunction::Max && cost >= other.cost(model))
                    || (model.reduce_function == ReduceFunction::Min
                        && cost <= other.cost(model)) =>
            {
                dominated_indices.push(i);
            }
            _ => {}
        }
    }

    if dominated_indices.is_empty() {
        return (false, smallvec::smallvec![], None);
    }

    // Remove dominated information starting from the end of the vector using `swap_remove`.
    // Convert `same_state_index` to the index in the vector of the removed information.
    let same_state_index = same_state_index.map(|i| dominated_indices.len() - 1 - i);
    let mut dominated = SmallVec::with_capacity(dominated_indices.len());
    dominated_indices.into_iter().rev().for_each(|i| {
        dominated.push(non_dominated.swap_remove(i));
    });

    (false, dominated, same_state_index)
}

/// Registry storing generated states considering dominance relationship.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{
///     data_structure::StateInformation, FNode, StateInRegistry, StateRegistry,
/// };
/// use std::rc::Rc;
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
/// let model = Rc::new(model);
/// let mut registry = StateRegistry::<_, FNode<_>>::new(model.clone());
/// registry.reserve(2);
///
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let h_evaluator = |_: &StateInRegistry, _: &mut _| Some(0);
/// let f_evaluator = |g, h, _: &StateInRegistry| g + h;
///
/// let node = FNode::generate_root_node(
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
/// let irrelevant: StateInRegistry = increment.apply(
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
/// let irrelevant = FNode::generate_root_node(
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
/// let dominated: StateInRegistry = increase_cost.apply(
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
/// let dominated = FNode::generate_root_node(
///     dominated, &mut function_cache, cost, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// let result = registry.insert(dominated);
/// assert_eq!(result.information, None);
/// assert!(result.dominated.is_empty());
///
/// let mut function_cache = StateFunctionCache::new(&model.state_functions);
/// let dominating: StateInRegistry = produce.apply(
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
/// let dominating = FNode::generate_root_node(
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
///
/// registry.clear();
/// assert_eq!(registry.get(node.state(), node.cost(&model)), None);
/// ```
pub struct StateRegistry<T, I, R = Rc<dypdl::Model>>
where
    T: Numeric,
    I: StateInformation<T>,
    R: Deref<Target = dypdl::Model>,
{
    registry: FxHashMap<Rc<HashableSignatureVariables>, Vec<Rc<I>>>,
    model: R,
    phantom: std::marker::PhantomData<T>,
}

/// Result of insertion to a state registry.
#[derive(Debug, Clone)]
pub struct InsertionResult<D> {
    /// The inserted information. `None` if the given information is dominated.
    pub information: Option<D>,
    /// Vector of dominated information by the given information.
    pub dominated: SmallVec<[D; 1]>,
}

impl<T, I, R> StateRegistry<T, I, R>
where
    T: Numeric,
    I: StateInformation<T>,
    R: Deref<Target = dypdl::Model>,
{
    /// Creates a new state registry.
    #[inline]
    pub fn new(model: R) -> StateRegistry<T, I, R> {
        StateRegistry {
            registry: FxHashMap::default(),
            model,
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns a pointer to the model.
    #[inline]
    pub fn model(&self) -> &R {
        &self.model
    }

    /// Reserves the capacity.
    #[inline]
    pub fn reserve(&mut self, capacity: usize) {
        self.registry.reserve(capacity);
    }

    /// Deletes all entries.
    #[inline]
    pub fn clear(&mut self) {
        self.registry.clear();
    }

    /// Clears the registry and returns all information.
    #[inline]
    pub fn drain(
        &mut self,
    ) -> impl Iterator<Item = (Rc<HashableSignatureVariables>, Vec<Rc<I>>)> + '_ {
        self.registry.drain()
    }

    /// Gets the first information of a state that dominates the given state.
    pub fn get(&self, state: &StateInRegistry, cost: T) -> Option<&Rc<I>> {
        if let Some(v) = self.registry.get(&state.signature_variables) {
            for other in v {
                let result = self.model.state_metadata.dominance(state, other.state());
                match result {
                    Some(Ordering::Equal) | Some(Ordering::Less)
                        if (self.model.reduce_function == ReduceFunction::Max
                            && cost <= other.cost(&self.model))
                            || (self.model.reduce_function == ReduceFunction::Min
                                && cost >= other.cost(&self.model)) =>
                    {
                        return Some(other)
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Removes states from the registry with given signature variables.
    pub fn remove(&mut self, state: &Rc<HashableSignatureVariables>) -> Option<Vec<Rc<I>>> {
        self.registry.remove(state)
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
        &mut self,
        mut state: StateInRegistry,
        cost: T,
        constructor: F,
    ) -> InsertionResult<Rc<I>>
    where
        F: FnOnce(StateInRegistry, T, Option<&I>) -> Option<I>,
    {
        let entry = self.registry.entry(state.signature_variables.clone());
        match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                state.signature_variables = entry.key().clone();

                let v = entry.into_mut();
                let (dominating, dominated, same_state_index) =
                    remove_dominated(v, &self.model, &state, cost);

                if dominating {
                    return InsertionResult {
                        information: None,
                        dominated: smallvec::smallvec![],
                    };
                }

                let same_state_information = same_state_index.map(|i| dominated[i].as_ref());

                if let Some(information) =
                    constructor(state, cost, same_state_information).map(Rc::new)
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
            collections::hash_map::Entry::Vacant(entry) => {
                if let Some(information) = constructor(state, cost, None).map(Rc::new) {
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
    pub fn insert(&mut self, mut information: I) -> InsertionResult<Rc<I>> {
        let entry = self
            .registry
            .entry(information.state().signature_variables.clone());
        let (v, removed) = match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                information.state_mut().signature_variables = entry.key().clone();

                let v = entry.into_mut();
                let (dominating, dominated, _) = remove_dominated(
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

                (v, dominated)
            }
            collections::hash_map::Entry::Vacant(entry) => {
                (entry.insert(Vec::with_capacity(1)), smallvec::smallvec![])
            }
        };

        let information = Rc::new(information);
        v.push(information.clone());

        InsertionResult {
            information: Some(information),
            dominated: removed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::hashable_state::HashableSignatureVariables;
    use super::*;
    use dypdl::expression::*;
    use dypdl::variable_type::OrderedContinuous;
    use dypdl::variable_type::Set;
    use dypdl::ResourceVariables;
    use dypdl::StateInterface;
    use ordered_float::OrderedFloat;
    use rustc_hash::FxHashMap;
    use std::cell::Cell;

    #[derive(Debug, Clone)]
    struct MockInformation {
        state: StateInRegistry,
        cost: i32,
        value: Cell<Option<i32>>,
    }

    impl StateInformation<Integer> for MockInformation {
        fn state(&self) -> &StateInRegistry {
            &self.state
        }

        fn state_mut(&mut self) -> &mut StateInRegistry {
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
    ) -> Rc<HashableSignatureVariables> {
        Rc::new(HashableSignatureVariables {
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

    fn generate_registry() -> dypdl::TableRegistry {
        let tables_1d = vec![dypdl::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![dypdl::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
        ])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        dypdl::TableRegistry {
            integer_tables: dypdl::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn state_in_registry_from_state() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let signature_variables = dypdl::SignatureVariables {
            set_variables: vec![set1, set2],
            vector_variables: vec![vec![0, 2], vec![1, 2]],
            element_variables: vec![1, 2],
            integer_variables: vec![1, 2, 3],
            continuous_variables: vec![1.0, 2.0, 3.0],
        };
        let resource_variables = dypdl::ResourceVariables {
            element_variables: vec![],
            integer_variables: vec![4, 5, 6],
            continuous_variables: vec![4.0, 5.0, 6.0],
        };
        let state = StateInRegistry::<Rc<_>>::from(dypdl::State {
            signature_variables: signature_variables.clone(),
            resource_variables: resource_variables.clone(),
        });
        assert_eq!(
            state.signature_variables.set_variables,
            signature_variables.set_variables
        );
        assert_eq!(
            state.signature_variables.vector_variables,
            signature_variables.vector_variables
        );
        assert_eq!(
            state.signature_variables.element_variables,
            signature_variables.element_variables
        );
        assert_eq!(
            state.signature_variables.integer_variables,
            signature_variables.integer_variables
        );
        assert_eq!(
            state.signature_variables.continuous_variables,
            vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)]
        );
        assert_eq!(
            state.resource_variables.element_variables,
            resource_variables.element_variables
        );
        assert_eq!(
            state.resource_variables.integer_variables,
            resource_variables.integer_variables
        );
        assert_eq!(
            state.resource_variables.continuous_variables,
            resource_variables.continuous_variables
        );
    }

    #[test]
    fn state_in_registry_from_state_with_hashable_signature_variables() {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let signature_variables = HashableSignatureVariables {
            set_variables: vec![set1, set2],
            vector_variables: vec![vec![0, 2], vec![1, 2]],
            element_variables: vec![1, 2],
            integer_variables: vec![1, 2, 3],
            continuous_variables: vec![
                OrderedContinuous::from(1.0),
                OrderedContinuous::from(2.0),
                OrderedContinuous::from(3.0),
            ],
        };
        let resource_variables = dypdl::ResourceVariables {
            element_variables: vec![],
            integer_variables: vec![4, 5, 6],
            continuous_variables: vec![4.0, 5.0, 6.0],
        };
        let state = StateInRegistry::<Rc<_>>::from(StateWithHashableSignatureVariables {
            signature_variables: signature_variables.clone(),
            resource_variables: resource_variables.clone(),
        });
        assert_eq!(
            state.signature_variables.set_variables,
            signature_variables.set_variables
        );
        assert_eq!(
            state.signature_variables.vector_variables,
            signature_variables.vector_variables
        );
        assert_eq!(
            state.signature_variables.element_variables,
            signature_variables.element_variables
        );
        assert_eq!(
            state.signature_variables.integer_variables,
            signature_variables.integer_variables
        );
        assert_eq!(
            state.signature_variables.continuous_variables,
            signature_variables.continuous_variables
        );
        assert_eq!(
            state.resource_variables.element_variables,
            resource_variables.element_variables
        );
        assert_eq!(
            state.resource_variables.integer_variables,
            resource_variables.integer_variables
        );
        assert_eq!(
            state.resource_variables.continuous_variables,
            resource_variables.continuous_variables
        );
    }

    #[test]
    fn state_get_number_of_set_variables() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![Set::default()],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_number_of_set_variables(), 1);
    }

    #[test]
    fn state_get_set_variable() {
        let mut set = Set::with_capacity(2);
        set.insert(1);
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![Set::with_capacity(2), set.clone()],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_set_variable(0), &Set::with_capacity(2));
        assert_eq!(state.get_set_variable(1), &set);
    }

    #[test]
    #[should_panic]
    fn state_get_set_variable_panic() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![Set::default()],
                ..Default::default()
            }),
            ..Default::default()
        };
        state.get_set_variable(1);
    }

    #[test]
    fn state_get_number_of_vector_variables() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                vector_variables: vec![Vector::default()],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_number_of_vector_variables(), 1);
    }

    #[test]
    fn state_get_vector_variable() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                vector_variables: vec![Vector::default(), vec![1]],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_vector_variable(0), &Vector::default());
        assert_eq!(state.get_vector_variable(1), &vec![1]);
    }

    #[test]
    #[should_panic]
    fn state_get_vector_variable_panic() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                vector_variables: vec![Vector::default()],
                ..Default::default()
            }),
            ..Default::default()
        };
        state.get_vector_variable(1);
    }

    #[test]
    fn state_get_number_of_element_variables() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_number_of_element_variables(), 1);
    }

    #[test]
    fn state_get_element_variable() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                element_variables: vec![0, 1],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_element_variable(0), 0);
        assert_eq!(state.get_element_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_element_variable_panic() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                element_variables: vec![0],
                ..Default::default()
            }),
            ..Default::default()
        };
        state.get_element_variable(1);
    }

    #[test]
    fn state_get_number_of_integer_variables() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_number_of_integer_variables(), 1);
    }

    #[test]
    fn state_get_integer_variable() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![0, 1],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_integer_variable(0), 0);
        assert_eq!(state.get_integer_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_integer_variable_panic() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            }),
            ..Default::default()
        };
        state.get_integer_variable(1);
    }

    #[test]
    fn state_get_number_of_continuous_variables() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                continuous_variables: vec![OrderedContinuous::from(0.0)],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_number_of_continuous_variables(), 1);
    }

    #[test]
    fn state_get_continuous_variable() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                continuous_variables: vec![
                    OrderedContinuous::from(0.0),
                    OrderedContinuous::from(1.0),
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(state.get_continuous_variable(0), 0.0);
        assert_eq!(state.get_continuous_variable(1), 1.0);
    }

    #[test]
    #[should_panic]
    fn state_get_continuous_variable_panic() {
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                continuous_variables: vec![OrderedContinuous::from(0.0)],
                ..Default::default()
            }),
            ..Default::default()
        };
        state.get_continuous_variable(1);
    }

    #[test]
    fn state_get_number_of_element_resource_variables() {
        let state = StateInRegistry::<Rc<_>> {
            resource_variables: ResourceVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_element_resource_variables(), 1);
    }

    #[test]
    fn state_get_element_resource_variable() {
        let state = StateInRegistry::<Rc<_>> {
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_element_resource_variable(0), 0);
        assert_eq!(state.get_element_resource_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_element_resource_variable_panic() {
        let state = StateInRegistry::<Rc<_>> {
            resource_variables: ResourceVariables {
                element_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_element_resource_variable(1);
    }

    #[test]
    fn state_get_number_of_integer_resource_variables() {
        let state = StateInRegistry::<Rc<_>> {
            resource_variables: ResourceVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_integer_resource_variables(), 1);
    }

    #[test]
    fn state_get_integer_resource_variable() {
        let state = StateInRegistry::<Rc<_>> {
            resource_variables: ResourceVariables {
                integer_variables: vec![0, 1],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_integer_resource_variable(0), 0);
        assert_eq!(state.get_integer_resource_variable(1), 1);
    }

    #[test]
    #[should_panic]
    fn state_get_integer_resource_variable_panic() {
        let state = StateInRegistry::<Rc<_>> {
            resource_variables: ResourceVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_integer_resource_variable(1);
    }

    #[test]
    fn state_get_number_of_continuous_resource_variables() {
        let state = StateInRegistry::<Rc<_>> {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_number_of_continuous_resource_variables(), 1);
    }

    #[test]
    fn state_get_continuous_resource_variable() {
        let state = StateInRegistry::<Rc<_>> {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0, 1.0],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(state.get_continuous_resource_variable(0), 0.0);
        assert_eq!(state.get_continuous_resource_variable(1), 1.0);
    }

    #[test]
    #[should_panic]
    fn state_get_continuous_resource_variable_panic() {
        let state = StateInRegistry::<Rc<_>> {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        state.get_continuous_resource_variable(1);
    }

    #[test]
    fn apply_effect() {
        let state_functions = StateFunctions::default();
        let mut function_cache = StateFunctionCache::new(&state_functions);

        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        let state = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2], vec![1, 2]],
                element_variables: vec![1, 2],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1],
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        };
        let registry = generate_registry();
        let set_effect1 = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            ElementExpression::Constant(1),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(0))),
        );
        let set_effect2 = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            ElementExpression::Constant(0),
            Box::new(SetExpression::Reference(ReferenceExpression::Variable(1))),
        );
        let vector_effect1 = VectorExpression::Push(
            ElementExpression::Constant(1),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                0,
            ))),
        );
        let vector_effect2 = VectorExpression::Push(
            ElementExpression::Constant(0),
            Box::new(VectorExpression::Reference(ReferenceExpression::Variable(
                1,
            ))),
        );
        let element_effect1 = ElementExpression::Constant(2);
        let element_effect2 = ElementExpression::Constant(1);
        let integer_effect1 = IntegerExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(IntegerExpression::Variable(0)),
            Box::new(IntegerExpression::Constant(1)),
        );
        let integer_effect2 = IntegerExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(IntegerExpression::Variable(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        let continuous_effect1 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        );
        let continuous_effect2 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(ContinuousExpression::Variable(1)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        let element_resource_effect1 = ElementExpression::Constant(1);
        let element_resource_effect2 = ElementExpression::Constant(0);
        let integer_resource_effect1 = IntegerExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(IntegerExpression::ResourceVariable(0)),
            Box::new(IntegerExpression::Constant(1)),
        );
        let integer_resource_effect2 = IntegerExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(IntegerExpression::ResourceVariable(1)),
            Box::new(IntegerExpression::Constant(2)),
        );
        let continuous_resource_effect1 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::ResourceVariable(0)),
            Box::new(ContinuousExpression::Constant(1.0)),
        );
        let continuous_resource_effect2 = ContinuousExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(ContinuousExpression::ResourceVariable(1)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        let effect = dypdl::Effect {
            set_effects: vec![(0, set_effect1), (1, set_effect2)],
            vector_effects: vec![(0, vector_effect1), (1, vector_effect2)],
            element_effects: vec![(0, element_effect1), (1, element_effect2)],
            integer_effects: vec![(0, integer_effect1), (1, integer_effect2)],
            continuous_effects: vec![(0, continuous_effect1), (1, continuous_effect2)],
            element_resource_effects: vec![
                (0, element_resource_effect1),
                (1, element_resource_effect2),
            ],
            integer_resource_effects: vec![
                (0, integer_resource_effect1),
                (1, integer_resource_effect2),
            ],
            continuous_resource_effects: vec![
                (0, continuous_resource_effect1),
                (1, continuous_resource_effect2),
            ],
        };

        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(1);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(1);
        let expected = StateInRegistry {
            signature_variables: Rc::new(HashableSignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2, 1], vec![1, 2, 0]],
                element_variables: vec![2, 1],
                integer_variables: vec![0, 4, 3],
                continuous_variables: vec![OrderedFloat(0.0), OrderedFloat(4.0), OrderedFloat(3.0)],
            }),
            resource_variables: ResourceVariables {
                element_variables: vec![1, 0],
                integer_variables: vec![5, 2, 6],
                continuous_variables: vec![5.0, 2.5, 6.0],
            },
        };
        let successor: StateInRegistry = state.apply_effect(
            &effect,
            &mut function_cache,
            &state_functions,
            &registry,
        );
        assert_eq!(successor, expected);
    }

    #[test]
    fn remove_dominated_non_dominating_non_dominated() {
        let model = generate_model();

        let state1 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let state2 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        let mut non_dominated = vec![
            Rc::new(MockInformation {
                state: state1,
                cost: 1,
                value: Cell::new(None),
            }),
            Rc::new(MockInformation {
                state: state2,
                cost: 3,
                value: Cell::new(None),
            }),
        ];

        let state3 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 2, 3]),
        };
        let cost = 2;

        let (dominating, dominated, same_state_index) =
            remove_dominated(&mut non_dominated, &model, &state3, cost);
        assert!(!dominating);
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(same_state_index, None);
    }

    #[test]
    fn remove_dominated_dominated() {
        let model = generate_model();

        let state1 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let state2 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        let state3 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![3, 3, 3]),
        };
        let mut non_dominated = vec![
            Rc::new(MockInformation {
                state: state1.clone(),
                cost: 1,
                value: Cell::new(None),
            }),
            Rc::new(MockInformation {
                state: state2,
                cost: 3,
                value: Cell::new(None),
            }),
            Rc::new(MockInformation {
                state: state3,
                cost: 1,
                value: Cell::new(None),
            }),
        ];

        let state4 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        let cost = 0;

        let (dominating, dominated, same_state_index) =
            remove_dominated(&mut non_dominated, &model, &state4, cost);
        assert!(!dominating);
        assert_eq!(dominated.len(), 2);
        assert!(same_state_index.is_some());
        let same_state_index = same_state_index.unwrap();
        assert_eq!(dominated[same_state_index].state, state4);
        assert_eq!(dominated[same_state_index].cost, 3);
        assert_eq!(dominated[1 - same_state_index].state, state1);
        assert_eq!(dominated[1 - same_state_index].cost, 1);
    }

    #[test]
    fn remove_dominated_dominating() {
        let model = generate_model();

        let state1 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let state2 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        let mut non_dominated = vec![
            Rc::new(MockInformation {
                state: state1,
                cost: 1,
                value: Cell::new(None),
            }),
            Rc::new(MockInformation {
                state: state2,
                cost: 3,
                value: Cell::new(None),
            }),
        ];

        let state3 = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 2, 3]),
        };
        let cost = 4;

        let (dominating, dominated, same_state_index) =
            remove_dominated(&mut non_dominated, &model, &state3, cost);
        assert!(dominating);
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(same_state_index, None);
    }

    #[test]
    fn insert_get_new_information() {
        let model = Rc::new(generate_model());
        let mut registry = StateRegistry::<i32, MockInformation>::new(model);

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
        let result = registry.insert(information);
        let information1 = result.information;
        assert!(information1.is_some());
        let information1 = information1.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information1.state, state);
        assert_eq!(information1.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information1));

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
        let result = registry.insert(information);
        let information2 = result.information;
        assert!(information2.is_some());
        let information2 = information2.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information2.state, state);
        assert_eq!(information2.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information2));

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
        let result = registry.insert(information);
        let information3 = result.information;
        assert!(information3.is_some());
        let information3 = information3.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information3.state, state);
        assert_eq!(information3.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information3));

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
        let result = registry.insert(information);
        let information4 = result.information;
        assert!(information4.is_some());
        let information4 = information4.unwrap();
        let dominated = result.dominated;
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information4.state, state);
        assert_eq!(information4.value.get(), None);
        assert_eq!(registry.get(&state, 0), Some(&information4));

        let information_vec = registry.remove(&generate_signature_variables(vec![0, 1, 2]));
        assert_eq!(information_vec, Some(vec![information1]));

        let information_vec = registry.remove(&generate_signature_variables(vec![1, 2, 3]));
        assert_eq!(
            information_vec,
            Some(vec![information2, information3, information4])
        );

        let information_vec = registry.remove(&generate_signature_variables(vec![2, 3, 4]));
        assert_eq!(information_vec, None);
    }

    #[test]
    fn insert_information_dominated() {
        let model = Rc::new(generate_model());
        let mut registry = StateRegistry::<i32, MockInformation>::new(model);

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
        let previous = result.information;
        assert!(previous.is_some());
        let previous = previous.unwrap();
        let dominated = result.dominated;
        assert_eq!(previous.state, state);
        assert_eq!(previous.value.get(), None);
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(&previous));
        let information = MockInformation {
            state: state.clone(),
            cost: 1,
            value: Cell::new(None),
        };
        let result = registry.insert(information);
        assert!(result.information.is_none());
        assert!(result.dominated.is_empty());
        assert_eq!(registry.get(&state, 1), Some(&previous));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(&previous));
        let information = MockInformation {
            state: state.clone(),
            cost: 1,
            value: Cell::new(None),
        };
        let result = registry.insert(information);
        assert!(result.information.is_none());
        assert!(result.dominated.is_empty());
        assert_eq!(registry.get(&state, 1), Some(&previous));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 2), Some(&previous));
        let information = MockInformation {
            state: state.clone(),
            cost: 2,
            value: Cell::new(None),
        };
        let result = registry.insert(information);
        assert!(result.information.is_none());
        assert!(result.dominated.is_empty());
        assert_eq!(registry.get(&state, 2), Some(&previous));

        let mut iter = registry.drain();
        assert_eq!(
            iter.next(),
            Some((previous.state.signature_variables.clone(), vec![previous]))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn insert_get_dominating_information() {
        let model = Rc::new(generate_model());
        let mut registry = StateRegistry::<i32, MockInformation>::new(model);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 2), None);
        let information = MockInformation {
            state: state.clone(),
            cost: 2,
            value: Cell::new(None),
        };
        let result = registry.insert(information);
        let information1 = result.information;
        assert!(information1.is_some());
        let information1 = information1.unwrap();
        let dominated = result.dominated;
        assert_eq!(information1.state, state);
        assert_eq!(information1.value.get(), None);
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(registry.get(&information1.state, 2), Some(&information1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = MockInformation {
            state,
            cost: 1,
            value: Cell::new(None),
        };
        let result = registry.insert(information);
        let information2 = result.information;
        assert!(information2.is_some());
        let information2 = information2.unwrap();
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
        assert_eq!(registry.get(&information1.state, 2), Some(&information2));
        assert_eq!(registry.get(&information2.state, 1), Some(&information2));

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
        let result = registry.insert(information);
        let information3 = result.information;
        assert!(information3.is_some());
        let information3 = information3.unwrap();
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
        assert_eq!(registry.get(&information2.state, 1), Some(&information2));
        assert_eq!(registry.get(&information3.state, 0), Some(&information3));

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
        let result = registry.insert(information);
        let information4 = result.information;
        assert!(information4.is_some());
        let information4 = information4.unwrap();
        let mut dominated = result.dominated;
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
        dominated.sort();
        assert_eq!(
            dominated,
            SmallVec::<[_; 1]>::from(vec![information3.clone(), information2.clone()])
        );
        assert_eq!(registry.get(&information1.state, 2), Some(&information4));
        assert_eq!(registry.get(&information2.state, 1), Some(&information4));
        assert_eq!(registry.get(&information3.state, 0), Some(&information4));
        assert_eq!(registry.get(&information4.state, 0), Some(&information4));

        let mut iter = registry.drain();
        assert_eq!(
            iter.next(),
            Some((
                information4.state.signature_variables.clone(),
                vec![information4]
            ))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn insert_with_get_new_information() {
        let model = Rc::new(generate_model());
        let mut registry = StateRegistry::<i32, MockInformation>::new(model);
        let constructor = |state: StateInRegistry, cost: Integer, _: Option<&MockInformation>| {
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
        let information1 = result.information;
        assert!(information1.is_some());
        let information1 = information1.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information1.state, state);
        assert_eq!(information1.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let result = registry.insert_with(state, 1, constructor);
        let information2 = result.information;
        assert!(information2.is_some());
        let information2 = information2.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information2.state, state);
        assert_eq!(information2.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information2));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let result = registry.insert_with(state, 1, constructor);
        let information3 = result.information;
        assert!(information3.is_some());
        let information3 = information3.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information3.state, state);
        assert_eq!(information3.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information3));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let result = registry.insert_with(state, 0, constructor);
        let information4 = result.information;
        assert!(information4.is_some());
        let information4 = information4.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(information4.state, state);
        assert_eq!(information4.value.get(), None);
        assert_eq!(registry.get(&state, 0), Some(&information4));

        let information_vec = registry.remove(&generate_signature_variables(vec![0, 1, 2]));
        assert_eq!(information_vec, Some(vec![information1]));

        let information_vec = registry.remove(&generate_signature_variables(vec![1, 2, 3]));
        assert_eq!(
            information_vec,
            Some(vec![information2, information3, information4])
        );

        let information_vec = registry.remove(&generate_signature_variables(vec![2, 3, 4]));
        assert_eq!(information_vec, None);
    }

    #[test]
    fn insert_with_information_dominated() {
        let model = Rc::new(generate_model());
        let mut registry = StateRegistry::<i32, MockInformation>::new(model);

        let constructor = |state: StateInRegistry, cost: Integer, _: Option<&MockInformation>| {
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
        let previous = result.information;
        assert!(previous.is_some());
        let previous = previous.unwrap();
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
        assert_eq!(registry.get(&state, 1), Some(&previous));
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_none());
        assert!(result.dominated.is_empty());
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(&previous));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(&previous));
        let result = registry.insert_with(state, 1, constructor);
        assert!(result.information.is_none());
        assert!(result.dominated.is_empty());
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(&previous));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 2), Some(&previous));
        let result = registry.insert_with(state, 2, constructor);
        assert!(result.information.is_none());
        assert!(result.dominated.is_empty());
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 2), Some(&previous));

        let mut iter = registry.drain();
        assert_eq!(
            iter.next(),
            Some((previous.state.signature_variables.clone(), vec![previous]))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn insert_with_get_dominating_information() {
        let model = Rc::new(generate_model());
        let mut registry = StateRegistry::<i32, MockInformation>::new(model);
        let constructor =
            |state: StateInRegistry, cost: Integer, other: Option<&MockInformation>| {
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
        assert_eq!(registry.get(&state, 2), None);
        let result = registry.insert_with(state, 2, constructor);
        let information1 = result.information;
        assert!(information1.is_some());
        let information1 = information1.unwrap();
        let dominated = result.dominated;
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information1.state, state);
        assert_eq!(information1.value.get(), None);
        information1.value.set(Some(10));
        assert_eq!(dominated, SmallVec::<[_; 1]>::new());
        assert_eq!(registry.get(&information1.state, 2), Some(&information1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let result = registry.insert_with(state, 1, constructor);
        let information2 = result.information;
        assert!(information2.is_some());
        let information2 = information2.unwrap();
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
        assert_eq!(registry.get(&information1.state, 2), Some(&information2));
        assert_eq!(registry.get(&information2.state, 1), Some(&information2));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 1, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let result = registry.insert_with(state, 0, constructor);
        let information3 = result.information;
        assert!(information3.is_some());
        let information3 = information3.unwrap();
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
        assert_eq!(registry.get(&information1.state, 2), Some(&information2));
        assert_eq!(registry.get(&information2.state, 1), Some(&information2));
        assert_eq!(registry.get(&information3.state, 0), Some(&information3));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let result = registry.insert_with(state, 0, constructor);
        let information4 = result.information;
        assert!(information4.is_some());
        let information4 = information4.unwrap();
        let mut dominated = result.dominated;
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
        assert_eq!(information4.value.get(), None);
        dominated.sort();
        assert_eq!(
            dominated,
            SmallVec::<[_; 1]>::from(vec![information3.clone(), information2.clone()])
        );
        assert_eq!(registry.get(&information1.state, 2), Some(&information4));
        assert_eq!(registry.get(&information2.state, 1), Some(&information4));
        assert_eq!(registry.get(&information3.state, 0), Some(&information3));
        assert_eq!(registry.get(&information4.state, 0), Some(&information4));

        let mut iter = registry.drain();
        assert_eq!(
            iter.next(),
            Some((
                information4.state.signature_variables.clone(),
                vec![information4]
            ))
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn get_model() {
        let model = Rc::new(generate_model());
        let registry = StateRegistry::<i32, MockInformation>::new(model.clone());
        assert_eq!(registry.model(), &model);
    }

    #[test]
    fn reserve() {
        let model = Rc::new(generate_model());
        let mut registry = StateRegistry::<i32, MockInformation>::new(model);
        registry.reserve(10);
        assert!(registry.registry.capacity() >= 10);
    }

    #[test]
    fn clear() {
        let model = Rc::new(generate_model());
        let mut registry = StateRegistry::<i32, MockInformation>::new(model);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let constructor = |state: StateInRegistry, cost: Integer, _: Option<&MockInformation>| {
            Some(MockInformation {
                state,
                cost,
                value: Cell::new(None),
            })
        };
        let result = registry.insert_with(state, 1, constructor);
        let information = result.information;
        assert!(information.is_some());
        let information = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information.state, state);
        assert_eq!(registry.get(&information.state, 1), Some(&information));

        registry.clear();

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&information.state, 1), None);
        let result = registry.insert_with(state, 1, constructor);
        let information = result.information;
        assert!(information.is_some());
        let information = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information.state, state);
        assert_eq!(registry.get(&information.state, 1), Some(&information));
    }

    #[test]
    fn remove() {
        let model = Rc::new(generate_model());
        let mut registry = StateRegistry::<i32, MockInformation>::new(model);

        let information_vec = registry.remove(&generate_signature_variables(vec![0, 1, 2]));
        assert_eq!(information_vec, None);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        let information = MockInformation {
            state,
            cost: 0,
            value: Cell::new(None),
        };
        let information = registry.insert(information);
        assert!(information.information.is_some());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 4]),
        };
        let information = MockInformation {
            state,
            cost: 1,
            value: Cell::new(None),
        };
        let information = registry.insert(information);
        assert!(information.information.is_none());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 2]),
        };
        let information = MockInformation {
            state,
            cost: 0,
            value: Cell::new(None),
        };
        let information = registry.insert(information);
        assert!(information.information.is_some());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 4]),
        };
        let information = MockInformation {
            state,
            cost: 1,
            value: Cell::new(None),
        };
        let information = registry.insert(information);
        assert!(information.information.is_some());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 4]),
        };
        let information = MockInformation {
            state,
            cost: 1,
            value: Cell::new(None),
        };
        let information = registry.insert(information);
        assert!(information.information.is_some());

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 2]),
        };
        let expected1 = Rc::new(MockInformation {
            state,
            cost: 0,
            value: Cell::new(None),
        });
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 4]),
        };
        let expected2 = Rc::new(MockInformation {
            state,
            cost: 1,
            value: Cell::new(None),
        });

        let information_vec = registry.remove(&generate_signature_variables(vec![0, 1, 2]));
        assert_eq!(information_vec, Some(vec![expected1, expected2]));
    }
}
