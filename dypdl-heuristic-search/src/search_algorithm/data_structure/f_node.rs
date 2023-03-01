use super::bfs_node::BfsNodeInterface;
use super::hashable_state::HashableSignatureVariables;
use super::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use super::transition_chain::{TransitionChain, TransitionChainInterface};
use super::util::exceed_bound;
use dypdl::variable_type::Numeric;
use dypdl::ReduceFunction;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

/// Node ordered by the f-value.
///
/// Nodes are totally ordered by their f-values, and tie is broken by the h-values.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::search_algorithm::data_structure::{
///     FNode, BfsNodeInterface,
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
/// let mut registry = StateRegistry::new(model.clone());
///
/// let h_evaluator = |_: &_, _: &_| Some(0);
/// let f_evaluator = |g, h, _: &_, _: &_| g + h;
///
/// let (node, h, f) = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator).unwrap();
/// assert_eq!(h, 0);
/// assert_eq!(f, 0);
/// assert_eq!(node.state(), &StateInRegistry::from(model.target.clone()));
/// assert_eq!(node.cost(), 0);
/// assert_eq!(node.get_bound(&model), 0);
/// assert!(!node.closed());
/// assert_eq!(node.transitions(), vec![]);
///
/// node.close();
/// assert!(node.closed());
///
/// let mut transition = Transition::new("transition");
/// transition.set_cost(IntegerExpression::Cost + 1);
/// transition.add_effect(variable, variable + 1).unwrap();
/// let expected_state: StateInRegistry = transition.apply(&model.target, &model.table_registry);
///
/// let (successor, h, f, generated) = node.generate_successor(
///     Rc::new(transition.clone()), &mut registry, h_evaluator, f_evaluator, None
/// ).unwrap();
/// assert_eq!(h, 0);
/// assert_eq!(f, 1);
/// assert!(generated);
/// assert_eq!(successor.state(), &expected_state);
/// assert_eq!(successor.cost(), 1);
/// assert_eq!(successor.get_bound(&model), 1);
/// assert!(!successor.closed());
/// assert_eq!(successor.transitions(), vec![transition]);
/// ```
#[derive(Debug, Default)]
pub struct FNode<T: Numeric> {
    g: T,
    h: T,
    f: T,
    state: StateInRegistry<Rc<HashableSignatureVariables>>,
    closed: RefCell<bool>,
    transitions: Option<Rc<TransitionChain>>,
}

impl<T: Numeric + PartialOrd> PartialEq for FNode<T> {
    /// Nodes are compared by their f- and h-values.
    /// This does not mean that the nodes are the same.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f && self.h == other.h
    }
}

impl<T: Numeric + Ord> Eq for FNode<T> {}

impl<T: Numeric + Ord> Ord for FNode<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.cmp(&other.f) {
            Ordering::Equal => self.h.cmp(&other.h),
            result => result,
        }
    }
}

impl<T: Numeric + Ord> PartialOrd for FNode<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric> StateInformation<T, Rc<HashableSignatureVariables>> for FNode<T> {
    #[inline]
    fn state(&self) -> &StateInRegistry<Rc<HashableSignatureVariables>> {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.g
    }
}

impl<T: Numeric + Ord> BfsNodeInterface<T> for FNode<T> {
    fn generate_initial_node<H, F>(
        registry: &mut StateRegistry<T, Self>,
        h_evaluator: H,
        f_evaluator: F,
    ) -> Option<(Rc<Self>, T, T)>
    where
        H: Fn(&StateInRegistry<Rc<HashableSignatureVariables>>, &dypdl::Model) -> Option<T>,
        F: Fn(T, T, &StateInRegistry<Rc<HashableSignatureVariables>>, &dypdl::Model) -> T,
    {
        let initial_state = StateInRegistry::from(registry.model().target.clone());
        let g = T::zero();
        let h = h_evaluator(&initial_state, registry.model())?;
        let f = f_evaluator(g, h, &initial_state, registry.model());

        let (h_priority, f_priority) = if registry.model().reduce_function == ReduceFunction::Min {
            (-h, -f)
        } else {
            (h, f)
        };
        let constructor = |state: StateInRegistry, g: T, _: Option<&FNode<T>>| {
            Some(FNode {
                state,
                g,
                h: h_priority,
                f: f_priority,
                ..Default::default()
            })
        };
        let (node, _) = registry.insert(initial_state, g, constructor)?;
        Some((node, h, f))
    }

    fn generate_successor<H, F>(
        &self,
        transition: Rc<dypdl::Transition>,
        registry: &mut StateRegistry<T, Self>,
        h_evaluator: H,
        f_evaluator: F,
        primal_bound: Option<T>,
    ) -> Option<(Rc<FNode<T>>, T, T, bool)>
    where
        H: Fn(&StateInRegistry, &dypdl::Model) -> Option<T>,
        F: Fn(T, T, &StateInRegistry, &dypdl::Model) -> T,
    {
        let (state, g) = registry.model().generate_successor_state(
            self.state(),
            self.cost(),
            transition.as_ref(),
            None,
        )?;

        let h = h_evaluator(&state, registry.model())?;
        let f = f_evaluator(g, h, &state, registry.model());

        if exceed_bound(registry.model(), f, primal_bound) {
            return None;
        }

        let (h_priority, f_priority) = if registry.model().reduce_function == ReduceFunction::Min {
            (-h, -f)
        } else {
            (h, f)
        };

        let constructor = |state: StateInRegistry, g: T, _: Option<&FNode<T>>| {
            Some(FNode {
                state,
                g,
                h: h_priority,
                f: f_priority,
                closed: RefCell::new(false),
                transitions: Some(Rc::new(TransitionChain::new(
                    self.transitions.clone(),
                    transition,
                ))),
            })
        };

        let (successor, dominated) = registry.insert(state, g, constructor)?;

        let mut generated = true;

        if let Some(dominated) = dominated {
            if !*dominated.closed.borrow() {
                *dominated.closed.borrow_mut() = true;
                generated = false;
            }
        }

        Some((successor, h, f, generated))
    }

    fn closed(&self) -> bool {
        *self.closed.borrow()
    }

    fn close(&self) {
        *self.closed.borrow_mut() = true;
    }

    fn get_bound(&self, model: &dypdl::Model) -> T {
        if model.reduce_function == ReduceFunction::Min {
            -self.f
        } else {
            self.f
        }
    }

    fn transitions(&self) -> Vec<dypdl::Transition> {
        self.transitions
            .as_ref()
            .map_or_else(Vec::new, |transitions| transitions.transitions())
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
    fn node_state() {
        let node = FNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            transitions: None,
        };
        assert_eq!(node.state(), &node.state);
    }

    #[test]
    fn node_cost() {
        let node = FNode {
            state: StateInRegistry::default(),
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            transitions: None,
        };
        assert_eq!(node.cost(), 1);
    }

    #[test]
    fn node_eq() {
        let node1 = FNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            transitions: None,
        };
        let node2 = FNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            transitions: None,
        };
        assert_eq!(node1, node2);
        assert!(node1 >= node2);
        assert!(node1 <= node2);
        assert!(node2 >= node1);
        assert!(node2 <= node1);
    }

    #[test]
    fn node_lt() {
        let node1 = FNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            transitions: None,
        };
        let node2 = FNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 2,
            h: 2,
            f: 4,
            closed: RefCell::new(false),
            transitions: None,
        };
        assert!(node1 < node2);
        assert!(node1 <= node2);
        assert!(node2 > node1);
        assert!(node2 >= node1);
    }

    #[test]
    fn node_lt_tie_breaking() {
        let node1 = FNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 1,
            h: 2,
            f: 3,
            closed: RefCell::new(false),
            transitions: None,
        };
        let node2 = FNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            g: 0,
            h: 3,
            f: 3,
            closed: RefCell::new(false),
            transitions: None,
        };
        assert!(node1 < node2);
        assert!(node1 <= node2);
        assert!(node2 > node1);
        assert!(node2 >= node1);
    }

    #[test]
    fn generate_initial_node_minimization() {
        let model = Rc::new(Model {
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        });
        let mut registry = StateRegistry::new(model.clone());
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 1;
        let result = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        let (node, h, f) = result.unwrap();
        assert_eq!(node.state, StateInRegistry::from(model.target.clone()));
        assert_eq!(node.g, 0);
        assert_eq!(node.h, -1);
        assert_eq!(node.f, -1);
        assert!(!*node.closed.borrow());
        assert_eq!(node.transitions, None);
        assert_eq!(h, 1);
        assert_eq!(f, 1);
    }

    #[test]
    fn generate_initial_node_maximization() {
        let model = Rc::new(Model {
            reduce_function: ReduceFunction::Max,
            ..Default::default()
        });
        let mut registry = StateRegistry::new(model.clone());
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 1;
        let result = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        let (node, h, f) = result.unwrap();
        assert_eq!(node.state, StateInRegistry::from(model.target.clone()));
        assert_eq!(node.g, 0);
        assert_eq!(node.h, 1);
        assert_eq!(node.f, 1);
        assert!(!*node.closed.borrow());
        assert_eq!(node.transitions, None);
        assert_eq!(h, 1);
        assert_eq!(f, 1);
    }

    #[test]
    fn generate_initial_node_pruned_by_h() {
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("v1"), 0);
        let model = Rc::new(Model::default());
        let mut registry = StateRegistry::new(model);
        let h_evaluator = |_: &StateInRegistry, _: &Model| None;
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 1;
        let result = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert_eq!(result, None)
    }

    #[test]
    fn generate_initial_node_pruned_by_duplicate() {
        let model = Rc::new(Model::default());
        let mut registry = StateRegistry::new(model);
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 1;
        let result = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert!(result.is_some());
        let result = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert_eq!(result, None);
    }

    #[test]
    fn generate_successor_minimize() {
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
        let mut registry = StateRegistry::new(model);
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 2;

        let node = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert!(node.is_some());
        let (node, _, _) = node.unwrap();

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
        let result = node.generate_successor(
            transition.clone(),
            &mut registry,
            h_evaluator,
            f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, h, f, generated) = result.unwrap();
        assert_eq!(
            successor.state,
            StateInRegistry {
                resource_variables: ResourceVariables {
                    integer_variables: vec![1, 1],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        assert_eq!(
            successor.transitions,
            Some(Rc::new(TransitionChain::new(
                node.transitions.clone(),
                transition
            )))
        );
        assert_eq!(successor.h, -1);
        assert_eq!(successor.f, -2);
        assert_eq!(h, 1);
        assert_eq!(f, 2);
        assert!(generated);
        assert!(!*node.closed.borrow());
    }

    #[test]
    fn generate_successor_maximize() {
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
            reduce_function: ReduceFunction::Max,
            ..Default::default()
        });
        let mut registry = StateRegistry::new(model);
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 2;

        let node = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert!(node.is_some());
        let (node, _, _) = node.unwrap();

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
        let result = node.generate_successor(
            transition.clone(),
            &mut registry,
            h_evaluator,
            f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, h, f, generated) = result.unwrap();
        assert_eq!(
            successor.state,
            StateInRegistry {
                resource_variables: ResourceVariables {
                    integer_variables: vec![1, 1],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        assert_eq!(
            successor.transitions,
            Some(Rc::new(TransitionChain::new(
                node.transitions.clone(),
                transition
            )))
        );
        assert_eq!(successor.h, 1);
        assert_eq!(successor.f, 2);
        assert_eq!(h, 1);
        assert_eq!(f, 2);
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
        let mut registry = StateRegistry::new(model);
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 2;

        let node = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert!(node.is_some());
        let (node, _, _) = node.unwrap();

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
        let result = node.generate_successor(
            transition.clone(),
            &mut registry,
            h_evaluator,
            f_evaluator,
            None,
        );
        assert!(result.is_some());
        let (successor, h, f, generated) = result.unwrap();
        assert_eq!(
            successor.state,
            StateInRegistry {
                resource_variables: ResourceVariables {
                    integer_variables: vec![1, 1],
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        assert_eq!(
            successor.transitions,
            Some(Rc::new(TransitionChain::new(
                node.transitions.clone(),
                transition
            )))
        );
        assert_eq!(successor.h, -1);
        assert_eq!(successor.f, -2);
        assert_eq!(h, 1);
        assert_eq!(f, 2);
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
                integer_less_is_better: vec![false, false],
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
        let mut registry = StateRegistry::new(model);
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 1;

        let node = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert!(node.is_some());
        let (node, _, _) = node.unwrap();

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
        let result =
            node.generate_successor(transition, &mut registry, h_evaluator, f_evaluator, None);
        assert_eq!(result, None);
    }

    #[test]
    fn generate_successor_pruned_by_h() {
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
        let mut registry = StateRegistry::new(model);
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 1;

        let node = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert!(node.is_some());
        let (node, _, _) = node.unwrap();

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
        let h_evaluator = |_: &StateInRegistry, _: &Model| None;
        let result =
            node.generate_successor(transition, &mut registry, h_evaluator, f_evaluator, None);
        assert_eq!(result, None);
    }

    #[test]
    fn generate_successor_pruned_by_bound() {
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
        let mut registry = StateRegistry::new(model);
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 1;

        let node = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert!(node.is_some());
        let (node, _, _) = node.unwrap();

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
        let result =
            node.generate_successor(transition, &mut registry, h_evaluator, f_evaluator, Some(1));
        assert_eq!(result, None);
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
        let mut registry = StateRegistry::new(model);
        let h_evaluator = |_: &StateInRegistry, _: &Model| Some(1);
        let f_evaluator = |_, _, _: &StateInRegistry, _: &Model| 1;

        let node = FNode::generate_initial_node(&mut registry, h_evaluator, f_evaluator);
        assert!(node.is_some());
        let (node, _, _) = node.unwrap();

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
        let result =
            node.generate_successor(transition, &mut registry, h_evaluator, f_evaluator, None);
        assert_eq!(result, None);
    }

    #[test]
    fn node_close() {
        let node = FNode::<i32> {
            closed: RefCell::new(false),
            ..Default::default()
        };
        assert!(!node.closed());
        node.close();
        assert!(node.closed());
    }

    #[test]
    fn node_no_transition() {
        let node = FNode::<i32> {
            transitions: None,
            ..Default::default()
        };
        assert_eq!(node.transitions(), vec![]);
    }

    #[test]
    fn node_transitions() {
        let parent = Rc::new(TransitionChain::new(
            None,
            Rc::new(Transition {
                name: String::from("t1"),
                ..Default::default()
            }),
        ));

        let node = FNode::<i32> {
            transitions: Some(Rc::new(TransitionChain::new(
                Some(parent),
                Rc::new(Transition {
                    name: String::from("t2"),
                    ..Default::default()
                }),
            ))),
            ..Default::default()
        };
        assert_eq!(
            node.transitions(),
            vec![
                Transition {
                    name: String::from("t1"),
                    ..Default::default()
                },
                Transition {
                    name: String::from("t2"),
                    ..Default::default()
                },
            ]
        );
    }

    #[test]
    fn get_bound_minimization() {
        let model = Rc::new(Model {
            reduce_function: ReduceFunction::Min,
            ..Default::default()
        });
        let node = FNode {
            f: -1,
            ..Default::default()
        };
        assert_eq!(node.get_bound(&model), 1);
    }

    #[test]
    fn get_bound_maximization() {
        let model = Rc::new(Model {
            reduce_function: ReduceFunction::Max,
            ..Default::default()
        });
        let node = FNode {
            f: 1,
            ..Default::default()
        };
        assert_eq!(node.get_bound(&model), 1);
    }
}
