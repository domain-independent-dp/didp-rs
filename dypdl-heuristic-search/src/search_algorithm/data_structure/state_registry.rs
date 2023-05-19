//! A module for state registries for duplicate detection.

use super::hashable_state::{HashableSignatureVariables, StateWithHashableSignatureVariables};
use core::ops::Deref;
use dypdl::variable_type::{Continuous, Element, Integer, Numeric, Set, Vector};
use dypdl::{Model, ReduceFunction};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
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

/// Registry storing generated states considering dominance relationship.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::{FNode, StateInRegistry, StateRegistry};
/// use dypdl_heuristic_search::search_algorithm::data_structure::{
///     StateInformation,
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
/// let h_evaluator = |_: &StateInRegistry| Some(0);
/// let f_evaluator = |g, h, _: &StateInRegistry| g + h;
///
/// let node = FNode::generate_root_node(
///     model.target.clone(), 0, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// assert_eq!(registry.get(node.state(), node.cost(&model)), None);
/// let result = registry.insert(node.clone());
/// assert_eq!(result, Some((Rc::new(node), None)));
/// let (node, _) = result.unwrap();
/// assert_eq!(registry.get(node.state(), node.cost(&model)), Some(&node));
///
/// let irrelevant: StateInRegistry = increment.apply(node.state(), &model.table_registry);
/// let cost = increment.eval_cost(node.cost(&model), node.state(), &model.table_registry);
/// let new_node = FNode::generate_root_node(
///     irrelevant.clone(), cost, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// assert_eq!(registry.get(&irrelevant, cost), None);
/// assert_eq!(registry.insert(new_node.clone()), Some((Rc::new(new_node), None)));
///
/// let dominated: StateInRegistry = increase_cost.apply(node.state(), &model.table_registry);
/// let cost = consume.eval_cost(node.cost(&model), node.state(), &model.table_registry);
/// let new_node = FNode::generate_root_node(
///     dominated.clone(), cost, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// assert_eq!(registry.get(&dominated, cost), Some(&node));
/// assert_eq!(registry.insert(new_node), None);
///
/// let dominating: StateInRegistry = produce.apply(node.state(), &model.table_registry);
/// let cost = produce.eval_cost(node.cost(&model), node.state(), &model.table_registry);
/// let new_node = FNode::generate_root_node(
///     dominating.clone(), cost, &model, &h_evaluator, &f_evaluator, None,
/// ).unwrap();
/// assert_eq!(registry.get(&dominating, cost), None);
/// assert_eq!(
///     registry.insert(new_node.clone()),
///     Some((Rc::new(new_node.clone()), Some(node.clone()))),
/// );
/// assert_eq!(registry.get(node.state(), node.cost(&model)), Some(&node));
///
/// registry.clear();
/// assert_eq!(registry.get(node.state(), node.cost(&model)), None);
/// ```
pub struct StateRegistry<T, I, V = Rc<I>, K = Rc<HashableSignatureVariables>, R = Rc<dypdl::Model>>
where
    T: Numeric,
    I: StateInformation<T, K>,
    V: Deref<Target = I> + From<I> + Clone,
    K: Hash + Eq + Clone + Debug,
    R: Deref<Target = dypdl::Model>,
{
    registry: FxHashMap<K, Vec<V>>,
    model: R,
    phantom: std::marker::PhantomData<T>,
}

impl<T, I, V, K, R> StateRegistry<T, I, V, K, R>
where
    T: Numeric,
    I: StateInformation<T, K>,
    V: Deref<Target = I> + From<I> + Clone,
    K: Hash + Eq + Clone + Debug,
    R: Deref<Target = dypdl::Model>,
    StateInRegistry<K>: dypdl::StateInterface,
{
    /// Creates a new state registry.
    #[inline]
    pub fn new(model: R) -> StateRegistry<T, I, V, K, R> {
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

    /// Gets the information of a state that dominates the given state.
    pub fn get(&self, state: &StateInRegistry<K>, cost: T) -> Option<&V> {
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

    /// Inserts a state and its information in the registry if it is not dominated by existing states
    /// given the state, its cost, and a constructor for the information.
    ///
    /// The constructor takes a state and its cost as arguments and returns the information.
    /// In addition, if exactly the same state is already saved in the registry,
    /// its information is passed as the last argument.
    /// It might be used to avoid recomputing values depending only on the state.
    /// It is called only when the state is not dominated.
    ///
    /// If the given state is not dominated, returns the created information and the information of a dominated state if it exists.
    pub fn insert_with<F>(
        &mut self,
        mut state: StateInRegistry<K>,
        cost: T,
        constructor: F,
    ) -> Option<(V, Option<V>)>
    where
        F: FnOnce(StateInRegistry<K>, T, Option<&I>) -> Option<I>,
    {
        let entry = self.registry.entry(state.signature_variables.clone());
        let v = match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                state.signature_variables = entry.key().clone();
                let v = entry.into_mut();
                for other in v.iter_mut() {
                    let result = self.model.state_metadata.dominance(&state, other.state());
                    match result {
                        Some(Ordering::Equal) | Some(Ordering::Less)
                            if (self.model.reduce_function == ReduceFunction::Max
                                && cost <= other.cost(&self.model))
                                || (self.model.reduce_function == ReduceFunction::Min
                                    && cost >= other.cost(&self.model)) =>
                        {
                            // dominated
                            return None;
                        }
                        Some(Ordering::Equal) | Some(Ordering::Greater)
                            if (self.model.reduce_function == ReduceFunction::Max
                                && cost >= other.cost(&self.model))
                                || (self.model.reduce_function == ReduceFunction::Min
                                    && cost <= other.cost(&self.model)) =>
                        {
                            // dominating
                            if let Some(information) = match result.unwrap() {
                                // if the same state is saved, reuse some information
                                Ordering::Equal => {
                                    constructor(state, cost, Some(&*other)).map(V::from)
                                }
                                _ => constructor(state, cost, None).map(V::from),
                            } {
                                let mut tmp = information.clone();
                                mem::swap(other, &mut tmp);
                                return Some((information, Some(tmp)));
                            } else {
                                return None;
                            }
                        }
                        _ => {}
                    }
                }
                v
            }
            collections::hash_map::Entry::Vacant(entry) => entry.insert(Vec::with_capacity(1)),
        };
        if let Some(information) = constructor(state, cost, None).map(V::from) {
            v.push(information.clone());
            Some((information, None))
        } else {
            None
        }
    }

    /// Inserts state information.
    ///
    /// If the given state is not dominated, returns a pointer to the information and the information of a dominated state if it exists.
    pub fn insert(&mut self, mut information: I) -> Option<(V, Option<V>)> {
        let entry = self
            .registry
            .entry(information.state().signature_variables.clone());
        let v = match entry {
            collections::hash_map::Entry::Occupied(entry) => {
                // use signature variables already stored
                information.state_mut().signature_variables = entry.key().clone();
                let v = entry.into_mut();
                for other in v.iter_mut() {
                    let result = self
                        .model
                        .state_metadata
                        .dominance(information.state(), other.state());
                    match result {
                        Some(Ordering::Equal) | Some(Ordering::Less)
                            if (self.model.reduce_function == ReduceFunction::Max
                                && information.cost(&self.model) <= other.cost(&self.model))
                                || (self.model.reduce_function == ReduceFunction::Min
                                    && information.cost(&self.model)
                                        >= other.cost(&self.model)) =>
                        {
                            // dominated
                            return None;
                        }
                        Some(Ordering::Equal) | Some(Ordering::Greater)
                            if (self.model.reduce_function == ReduceFunction::Max
                                && information.cost(&self.model) >= other.cost(&self.model))
                                || (self.model.reduce_function == ReduceFunction::Min
                                    && information.cost(&self.model)
                                        <= other.cost(&self.model)) =>
                        {
                            // dominating
                            let information = V::from(information);
                            let mut tmp = information.clone();
                            mem::swap(other, &mut tmp);
                            return Some((information, Some(tmp)));
                        }
                        _ => {}
                    }
                }
                v
            }
            collections::hash_map::Entry::Vacant(entry) => entry.insert(Vec::with_capacity(1)),
        };
        let information = V::from(information);
        v.push(information.clone());
        Some((information, None))
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
            self.cost.partial_cmp(&other.cost)
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
        let successor: StateInRegistry = state.apply_effect(&effect, &registry);
        assert_eq!(successor, expected);
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
        let information = registry.insert(information);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information));

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
        let information = registry.insert(information);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information));

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
        let information = registry.insert(information);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information));

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
        let information = registry.insert(information);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 0), Some(&information));
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
        let previous = registry.insert(previous);
        assert!(previous.is_some());
        let (previous, dominated) = previous.unwrap();
        assert_eq!(previous.state, state);
        assert_eq!(previous.value.get(), None);
        assert_eq!(dominated, None);

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
        let information = registry.insert(information);
        assert!(information.is_none());
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
        let information = registry.insert(information);
        assert!(information.is_none());
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
        let information = registry.insert(information);
        assert!(information.is_none());
        assert_eq!(registry.get(&state, 2), Some(&previous));
    }

    #[test]
    fn insert_get_dominating_information() {
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
        let information1 = registry.insert(information);
        assert!(information1.is_some());
        let (information1, dominated) = information1.unwrap();
        assert_eq!(information1.state, state);
        assert_eq!(information1.value.get(), None);
        assert_eq!(dominated, None);
        assert_eq!(registry.get(&information1.state, 1), Some(&information1));

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
        let information2 = registry.insert(information);
        assert!(information2.is_some());
        let (information2, dominated) = information2.unwrap();
        assert_eq!(
            information2.state.signature_variables,
            information1.state.signature_variables
        );
        assert_eq!(
            information2.state.resource_variables,
            information1.state.resource_variables
        );
        assert_eq!(information2.value.get(), None);
        assert_eq!(dominated.as_ref(), Some(&information1));
        assert_eq!(registry.get(&information1.state, 0), Some(&information2));
        assert_eq!(registry.get(&information2.state, 0), Some(&information2));

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
        let information3 = registry.insert(information);
        assert!(information3.is_some());
        let (information3, dominated) = information3.unwrap();
        assert_eq!(
            information3.state.signature_variables,
            information2.state.signature_variables
        );
        assert_ne!(
            information3.state.resource_variables,
            information2.state.resource_variables,
        );
        assert_eq!(information3.value.get(), None);
        assert_eq!(dominated.as_ref(), Some(&information2));
        assert_eq!(registry.get(&information1.state, 1), Some(&information3));
        assert_eq!(registry.get(&information2.state, 0), Some(&information3));
        assert_eq!(registry.get(&information3.state, 0), Some(&information3));
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
        let information = registry.insert_with(state, 1, constructor);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = registry.insert_with(state, 1, constructor);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(registry.get(&state, 1), None);
        let information = registry.insert_with(state, 1, constructor);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![3, 1, 3]),
        };
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 1), Some(&information));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information = registry.insert_with(state, 0, constructor);
        assert!(information.is_some());
        let (information, dominated) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![1, 2, 3]),
            resource_variables: generate_resource_variables(vec![0, 1, 3]),
        };
        assert_eq!(dominated, None);
        assert_eq!(information.state, state);
        assert_eq!(information.value.get(), None);
        assert_eq!(registry.get(&state, 0), Some(&information));
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
        let previous = registry.insert_with(state, 1, constructor);
        assert!(previous.is_some());
        let (previous, dominated) = previous.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(previous.state, state);
        assert_eq!(previous.value.get(), None);
        assert_eq!(dominated, None);

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 1), Some(&previous));
        let information = registry.insert_with(state, 1, constructor);
        assert!(information.is_none());
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
        let information = registry.insert_with(state, 1, constructor);
        assert!(information.is_none());
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
        let information = registry.insert_with(state, 2, constructor);
        assert!(information.is_none());
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![0, 2, 3]),
        };
        assert_eq!(registry.get(&state, 2), Some(&previous));
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
        assert_eq!(registry.get(&state, 1), None);
        let information1 = registry.insert_with(state, 1, constructor);
        assert!(information1.is_some());
        let (information1, dominated) = information1.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information1.state, state);
        assert_eq!(information1.value.get(), None);
        information1.value.set(Some(10));
        assert_eq!(dominated, None);
        assert_eq!(registry.get(&information1.state, 1), Some(&information1));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information2 = registry.insert_with(state, 0, constructor);
        assert!(information2.is_some());
        let (information2, dominated) = information2.unwrap();
        assert_eq!(
            information2.state.signature_variables,
            information1.state.signature_variables
        );
        assert_eq!(
            information2.state.resource_variables,
            information1.state.resource_variables
        );
        assert_eq!(information2.value.get(), Some(10));
        assert_eq!(dominated.as_ref(), Some(&information1));
        assert_eq!(registry.get(&information1.state, 0), Some(&information2));
        assert_eq!(registry.get(&information2.state, 0), Some(&information2));

        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![2, 3, 3]),
        };
        assert_eq!(registry.get(&state, 0), None);
        let information3 = registry.insert_with(state, 0, constructor);
        assert!(information3.is_some());
        let (information3, dominated) = information3.unwrap();
        assert_eq!(
            information3.state.signature_variables,
            information2.state.signature_variables
        );
        assert_ne!(
            information3.state.resource_variables,
            information2.state.resource_variables,
        );
        assert_eq!(information3.value.get(), None);
        assert_eq!(dominated.as_ref(), Some(&information2));
        assert_eq!(registry.get(&information1.state, 1), Some(&information3));
        assert_eq!(registry.get(&information2.state, 0), Some(&information3));
        assert_eq!(registry.get(&information3.state, 0), Some(&information3));
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
        let information = registry.insert_with(state, 1, constructor);
        assert!(information.is_some());
        let (information, _) = information.unwrap();
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
        let information = registry.insert_with(state, 1, constructor);
        assert!(information.is_some());
        let (information, _) = information.unwrap();
        let state = StateInRegistry {
            signature_variables: generate_signature_variables(vec![0, 1, 2]),
            resource_variables: generate_resource_variables(vec![1, 2, 3]),
        };
        assert_eq!(information.state, state);
        assert_eq!(registry.get(&information.state, 1), Some(&information));
    }
}
