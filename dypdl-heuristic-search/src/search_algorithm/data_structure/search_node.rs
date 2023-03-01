use super::hashable_state::HashableSignatureVariables;
use super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::transition_chain::{TransitionChain, TransitionChainInterface};
use dypdl::variable_type::Numeric;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::rc::Rc;

/// Search node.
///
/// Nodes are totally ordered by their costs.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::{
///     TransitionChainInterface, SearchNode,
/// };
/// use dypdl_heuristic_search::search_algorithm::data_structure::state_registry::{
///     StateInformation, StateInRegistry, StateRegistry,
/// };
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_variable("variable", 0).unwrap();
///
/// let model = Rc::new(model);
/// let mut registry = StateRegistry::<Integer, SearchNode<_>>::new(model.clone());
///
/// let node = SearchNode::generate_initial_node(&mut registry).unwrap();
/// assert_eq!(node.state(), &StateInRegistry::from(model.target.clone()));
/// assert_eq!(node.cost(), 0);
/// assert!(!*node.closed.borrow_mut());
/// assert_eq!(node.transitions, None);
///
/// let mut transition = Transition::new("transition");
/// transition.set_cost(IntegerExpression::Cost + 1);
/// transition.add_effect(variable, variable + 1).unwrap();
/// let expected_state: StateInRegistry = transition.apply(&model.target, &model.table_registry);
///
/// let (successor, generated) = node.generate_successor(Rc::new(transition.clone()), &mut registry, None).unwrap();
/// assert!(generated);
/// assert_eq!(successor.state(), &expected_state);
/// assert_eq!(successor.cost(), 1);
/// assert!(!*successor.closed.borrow_mut());
/// assert_eq!(successor.transitions.as_ref().unwrap().transitions(), vec![transition]);
/// ```
#[derive(Debug, Default)]
pub struct SearchNode<T: Numeric> {
    /// State.
    pub state: StateInRegistry<Rc<HashableSignatureVariables>>,
    /// Cost to reach this node.
    pub cost: T,
    /// Dominated or not.
    pub closed: RefCell<bool>,
    /// Transitions to reach this node.
    pub transitions: Option<Rc<TransitionChain>>,
}

impl<T: Numeric + PartialOrd> PartialEq for SearchNode<T> {
    /// Nodes are compared by their costs.
    /// This does not mean that the nodes are the same.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl<T: Numeric + Ord> Eq for SearchNode<T> {}

impl<T: Numeric + Ord> Ord for SearchNode<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl<T: Numeric + Ord> PartialOrd for SearchNode<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric> StateInformation<T> for SearchNode<T> {
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.cost
    }
}

impl<T: Numeric + Ord> SearchNode<T> {
    /// Returns the initial node.
    pub fn generate_initial_node(registry: &mut StateRegistry<T, Self>) -> Option<Rc<Self>> {
        let cost = T::zero();
        let initial_state = StateInRegistry::from(registry.model().target.clone());
        let constructor = |state: StateInRegistry, cost: T, _: Option<&SearchNode<T>>| {
            Some(SearchNode {
                state,
                cost,
                ..Default::default()
            })
        };
        let (initial_node, _) = registry.insert(initial_state, cost, constructor)?;
        Some(initial_node)
    }

    /// Returns a successor node if it it is not dominated by an existing node and its cost does not exceed the given primal bound.
    /// The last value returned indicates if a new search node is generated without dominating another open node.
    ///
    /// # Panics
    ///
    /// if an expression used in the transition is invalid.
    pub fn generate_successor(
        &self,
        transition: Rc<dypdl::Transition>,
        registry: &mut StateRegistry<T, Self>,
        primal_bound: Option<T>,
    ) -> Option<(Rc<SearchNode<T>>, bool)> {
        let (state, cost) = registry.model().generate_successor_state(
            self.state(),
            self.cost(),
            transition.as_ref(),
            primal_bound,
        )?;
        let constructor = |state: StateInRegistry, cost: T, _: Option<&SearchNode<T>>| {
            Some(SearchNode {
                state,
                cost,
                closed: RefCell::new(false),
                transitions: Some(Rc::new(TransitionChain::new(
                    self.transitions.clone(),
                    transition,
                ))),
            })
        };

        if let Some((successor, dominated)) = registry.insert(state, cost, constructor) {
            let mut generated = true;

            if let Some(dominated) = dominated {
                if !*dominated.closed.borrow() {
                    *dominated.closed.borrow_mut() = true;
                    generated = false;
                }
            }

            Some((successor, generated))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::hashable_state::HashableSignatureVariables;
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;
    use dypdl::Effect;
    use dypdl::GroundedCondition;
    use rustc_hash::FxHashMap;

    #[test]
    fn search_node_getter() {
        let node = Rc::new(SearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 0,
            closed: RefCell::new(false),
            transitions: None,
        });
        assert_eq!(node.state(), &node.state);
        assert_eq!(node.cost(), 0);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn search_node_cmp() {
        let node1 = SearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 0,
            closed: RefCell::new(false),
            transitions: None,
        };
        let node2 = SearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 0,
            closed: RefCell::new(false),
            transitions: None,
        };
        assert_eq!(node1, node2);
        let node2 = SearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 2,
            closed: RefCell::new(false),
            transitions: None,
        };
        assert!(node1 < node2)
    }

    #[test]
    fn generate_initial_node() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry = StateRegistry::<_, SearchNode<Integer>>::new(model.clone());
        let result = SearchNode::generate_initial_node(&mut registry);
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.state(), &StateInRegistry::from(model.target.clone()));
        assert_eq!(node.cost(), 0);
        assert_eq!(node.transitions, None);
    }

    #[test]
    fn generate_initial_node_dominated() {
        let model = Rc::new(dypdl::Model::default());
        let mut registry: StateRegistry<_, SearchNode<_>> = StateRegistry::new(model);
        let result = SearchNode::<i32>::generate_initial_node(&mut registry);
        assert!(result.is_some());
        let result = SearchNode::<i32>::generate_initial_node(&mut registry);
        assert_eq!(result, None);
    }

    #[test]
    fn generate_successor_non_dominance() {
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert(String::from("v1"), 0);
        name_to_integer_resource_variable.insert(String::from("v2"), 1);

        let model = Rc::new(Model {
            state_metadata: StateMetadata {
                integer_resource_variable_names: vec![String::from("v1"), String::from("v2")],
                name_to_integer_resource_variable,
                integer_less_is_better: vec![true, false],
                ..Default::default()
            },
            target: State {
                resource_variables: ResourceVariables {
                    integer_variables: vec![0, 0],
                    ..Default::default()
                },
                ..Default::default()
            },
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        });
        let mut registry = StateRegistry::<_, SearchNode<Integer>>::new(model);

        let node = SearchNode::generate_initial_node(&mut registry);
        assert!(node.is_some());
        let node = node.unwrap();

        let transition = Rc::new(Transition {
            name: String::from("increase"),
            effect: Effect {
                integer_resource_effects: vec![
                    (
                        0,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::ResourceVariable(0)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::ResourceVariable(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(1)),
            )),
            ..Default::default()
        });
        let result = node.generate_successor(transition.clone(), &mut registry, None);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(
            successor.state,
            StateInRegistry {
                resource_variables: ResourceVariables {
                    integer_variables: vec![1, 1],
                    ..Default::default()
                },
                ..Default::default()
            }
        );
        assert_eq!(successor.cost, 1);
        assert_eq!(
            successor.transitions,
            Some(Rc::new(TransitionChain::new(None, transition)))
        );
        assert!(!*successor.closed.borrow());
        assert!(generated);
        assert!(!*node.closed.borrow());
    }

    #[test]
    fn generate_successor_dominate() {
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert(String::from("v1"), 0);
        name_to_integer_resource_variable.insert(String::from("v2"), 1);

        let model = Rc::new(Model {
            state_metadata: StateMetadata {
                integer_resource_variable_names: vec![String::from("v1"), String::from("v2")],
                name_to_integer_resource_variable,
                integer_less_is_better: vec![false, false],
                ..Default::default()
            },
            target: State {
                resource_variables: ResourceVariables {
                    integer_variables: vec![0, 0],
                    ..Default::default()
                },
                ..Default::default()
            },
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        });
        let mut registry = StateRegistry::<_, SearchNode<Integer>>::new(model);

        let node = SearchNode::generate_initial_node(&mut registry);
        assert!(node.is_some());
        let node = node.unwrap();

        let transition = Rc::new(Transition {
            name: String::from("increase"),
            effect: Effect {
                integer_resource_effects: vec![
                    (
                        0,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::ResourceVariable(0)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::ResourceVariable(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::Cost),
            ..Default::default()
        });
        let result = node.generate_successor(transition.clone(), &mut registry, None);
        assert!(result.is_some());
        let (successor, generated) = result.unwrap();
        assert_eq!(
            successor.state,
            StateInRegistry {
                resource_variables: ResourceVariables {
                    integer_variables: vec![1, 1],
                    ..Default::default()
                },
                ..Default::default()
            }
        );
        assert_eq!(successor.cost, 0);
        assert_eq!(
            successor.transitions,
            Some(Rc::new(TransitionChain::new(None, transition)))
        );
        assert!(!generated);
        assert!(*node.closed.borrow());
    }

    #[test]
    fn generate_successor_pruned_by_constraint() {
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert(String::from("v1"), 0);
        name_to_integer_resource_variable.insert(String::from("v2"), 1);

        let model = Rc::new(Model {
            state_metadata: StateMetadata {
                integer_resource_variable_names: vec![String::from("v1"), String::from("v2")],
                name_to_integer_resource_variable,
                integer_less_is_better: vec![true, false],
                ..Default::default()
            },
            state_constraints: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Le,
                    Box::new(IntegerExpression::ResourceVariable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            target: State {
                resource_variables: ResourceVariables {
                    integer_variables: vec![0, 0],
                    ..Default::default()
                },
                ..Default::default()
            },
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        });
        let mut registry = StateRegistry::<_, SearchNode<Integer>>::new(model);

        let node = SearchNode::generate_initial_node(&mut registry);
        assert!(node.is_some());
        let node = node.unwrap();

        let transition = Rc::new(Transition {
            name: String::from("increase"),
            effect: Effect {
                integer_resource_effects: vec![
                    (
                        0,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::ResourceVariable(0)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::ResourceVariable(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(1)),
            )),
            ..Default::default()
        });
        let result = node.generate_successor(transition, &mut registry, None);
        assert_eq!(result, None);
        assert!(!*node.closed.borrow());
    }

    #[test]
    fn generate_successor_dominated() {
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert(String::from("v1"), 0);
        name_to_integer_resource_variable.insert(String::from("v2"), 1);

        let model = Rc::new(Model {
            state_metadata: StateMetadata {
                integer_resource_variable_names: vec![String::from("v1"), String::from("v2")],
                name_to_integer_resource_variable,
                integer_less_is_better: vec![true, true],
                ..Default::default()
            },
            target: State {
                resource_variables: ResourceVariables {
                    integer_variables: vec![0, 0],
                    ..Default::default()
                },
                ..Default::default()
            },
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        });
        let mut registry = StateRegistry::<_, SearchNode<Integer>>::new(model);

        let node = SearchNode::generate_initial_node(&mut registry);
        assert!(node.is_some());
        let node = node.unwrap();

        let transition = Rc::new(Transition {
            name: String::from("increase"),
            effect: Effect {
                integer_resource_effects: vec![
                    (
                        0,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::ResourceVariable(0)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                    (
                        1,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::ResourceVariable(1)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    ),
                ],
                ..Default::default()
            },
            cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(1)),
            )),
            ..Default::default()
        });
        let result = node.generate_successor(transition, &mut registry, None);
        assert_eq!(result, None);
        assert!(!*node.closed.borrow());
    }
}
