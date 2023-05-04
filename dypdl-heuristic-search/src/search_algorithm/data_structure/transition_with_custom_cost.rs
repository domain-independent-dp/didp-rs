use super::hashable_state::HashableSignatureVariables;
use super::prioritized_node::PrioritizedNode;
use super::state_registry::{StateInRegistry, StateInformation};
use super::successor_generator::SuccessorGenerator;
use super::transition_chain::TransitionChainInterface;
use core::ops::Deref;
use dypdl::variable_type::Numeric;
use dypdl::{CostExpression, StateInterface, Transition, TransitionInterface};
use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;

/// Transition with a customized cost expression.
#[derive(Debug, PartialEq, Clone, Default)]
pub struct TransitionWithCustomCost {
    /// Transition.
    pub transition: Transition,
    /// Customized cost expression.
    pub custom_cost: CostExpression,
    /// If forced.
    pub forced: bool,
    /// ID.
    pub id: usize,
}

impl TransitionInterface for TransitionWithCustomCost {
    #[inline]
    fn is_applicable<S: dypdl::StateInterface>(
        &self,
        state: &S,
        registry: &dypdl::TableRegistry,
    ) -> bool {
        self.transition.is_applicable(state, registry)
    }

    #[inline]
    fn apply<S: dypdl::StateInterface, T: From<dypdl::State>>(
        &self,
        state: &S,
        registry: &dypdl::TableRegistry,
    ) -> T {
        self.transition.apply(state, registry)
    }

    #[inline]
    fn eval_cost<U: Numeric, T: dypdl::StateInterface>(
        &self,
        cost: U,
        state: &T,
        registry: &dypdl::TableRegistry,
    ) -> U {
        self.transition.eval_cost(cost, state, registry)
    }
}

impl From<TransitionWithCustomCost> for Transition {
    fn from(transition: TransitionWithCustomCost) -> Self {
        transition.transition
    }
}

impl<U, R> SuccessorGenerator<TransitionWithCustomCost, U, R>
where
    U: Deref<Target = TransitionWithCustomCost> + Clone + From<TransitionWithCustomCost>,
    R: Deref<Target = dypdl::Model>,
{
    /// Returns a successor generator given a model and the direction.
    pub fn from_model_without_custom_cost(
        model: R,
        backward: bool,
    ) -> SuccessorGenerator<TransitionWithCustomCost, U, R> {
        let forced_transitions = if backward {
            &model.backward_forced_transitions
        } else {
            &model.forward_forced_transitions
        };
        let forced_transitions = forced_transitions
            .iter()
            .enumerate()
            .map(|(id, t)| {
                U::from(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: t.cost.clone(),
                    forced: true,
                    id,
                })
            })
            .collect();

        let transitions = if backward {
            &model.backward_transitions
        } else {
            &model.forward_transitions
        };
        let transitions = transitions
            .iter()
            .enumerate()
            .map(|(id, t)| {
                U::from(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: t.cost.clone(),
                    forced: false,
                    id,
                })
            })
            .collect();

        SuccessorGenerator::new(forced_transitions, transitions, backward, model)
    }

    /// Returns a successor generator returning applicable transitions with customized cost expressions.
    pub fn from_model_with_custom_costs(
        model: R,
        custom_costs: &[CostExpression],
        forced_custom_costs: &[CostExpression],
        backward: bool,
    ) -> SuccessorGenerator<TransitionWithCustomCost, U, R> {
        let forced_transitions = if backward {
            &model.backward_forced_transitions
        } else {
            &model.forward_forced_transitions
        };
        let forced_transitions = forced_transitions
            .iter()
            .enumerate()
            .zip(forced_custom_costs)
            .map(|((id, t), c)| {
                U::from(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: c.simplify(&model.table_registry),
                    forced: true,
                    id,
                })
            })
            .collect();

        let transitions = if backward {
            &model.backward_transitions
        } else {
            &model.forward_transitions
        };
        let transitions = transitions
            .iter()
            .enumerate()
            .zip(custom_costs)
            .map(|((id, t), c)| {
                U::from(TransitionWithCustomCost {
                    transition: t.clone(),
                    custom_cost: c.simplify(&model.table_registry),
                    forced: false,
                    id,
                })
            })
            .collect();
        SuccessorGenerator::new(forced_transitions, transitions, backward, model)
    }
}

/// Parent information to generate a successor using a transition with a custom cost.
#[derive(Clone)]
pub struct CustomCostParent<'a, T: Numeric, U: Numeric, S: StateInterface = StateInRegistry> {
    /// State.
    pub state: &'a S,
    /// Cost.
    pub cost: T,
    /// g-value.
    pub g: U,
}

impl TransitionWithCustomCost {
    /// Returns the successor state, its cost, g-value, and f-value.
    ///
    /// The successor is not generated if the `h_evaluator` returns `None`.
    /// The `f_evaluator` takes the g- and h-values as input in addition to the state and the model.
    ///
    /// # Examples
    ///
    /// ```
    /// use dypdl::prelude::*;
    /// use dypdl_heuristic_search::search_algorithm::data_structure::{
    ///     CustomCostParent, TransitionWithCustomCost
    /// };
    ///
    /// let mut model = Model::default();
    /// let variable = model.add_integer_variable("variable", 1).unwrap();
    /// let state = model.target.clone();
    ///
    /// let mut transition = Transition::new("transition");
    /// transition.set_cost(IntegerExpression::Cost + 1);
    /// transition.add_effect(variable, variable + 1);
    /// let transition = TransitionWithCustomCost {
    ///     transition,
    ///     custom_cost: CostExpression::from(ContinuousExpression::Cost + 1.5),
    ///     forced: false,
    ///     id: 0,
    /// };
    ///
    /// let parent = CustomCostParent { state: &state, cost: 0, g: 0.0 };
    /// let h_evaluator = |_: &_| Some(0.0);
    /// let f_evaluator = |g, h, _: &_| g + h;
    /// let expected_state = transition.apply(parent.state, &model.table_registry);
    /// let result = transition.generate_successor_state(
    ///     &parent, &model, None, h_evaluator, f_evaluator, false
    /// );
    /// assert_eq!(result, Some((expected_state, 1, 1.5, 1.5)));
    /// ```
    pub fn generate_successor_state<T, U, S, H, F>(
        &self,
        parent: &CustomCostParent<'_, T, U, S>,
        model: &dypdl::Model,
        bound: Option<U>,
        h_evaluator: H,
        f_evaluator: F,
        maximize: bool,
    ) -> Option<(S, T, U, U)>
    where
        T: Numeric,
        U: Numeric,
        S: StateInterface + From<dypdl::State>,
        H: Fn(&S) -> Option<U>,
        F: Fn(U, U, &S) -> U,
    {
        let state = self.apply(parent.state, &model.table_registry);

        if model.check_constraints(&state) {
            let g = self
                .custom_cost
                .eval_cost(parent.g, parent.state, &model.table_registry);
            let h = h_evaluator(&state)?;
            let f = f_evaluator(g, h, &state);

            if bound.map_or(false, |bound| {
                (maximize && f <= bound) || (!maximize && f >= bound)
            }) {
                return None;
            }

            let cost = self.eval_cost(parent.cost, parent.state, &model.table_registry);

            Some((state, cost, g, f))
        } else {
            None
        }
    }
}

/// Chain of transitions with custom costs implemented by a linked list of `Rc`.
#[derive(PartialEq, Debug)]
pub struct TransitionWithCustomCostChain {
    parent: Option<Rc<Self>>,
    last: Rc<TransitionWithCustomCost>,
}

impl TransitionChainInterface<TransitionWithCustomCost> for TransitionWithCustomCostChain {
    fn new(parent: Option<Rc<Self>>, last: Rc<TransitionWithCustomCost>) -> Self {
        Self { parent, last }
    }

    fn last(&self) -> &TransitionWithCustomCost {
        &self.last
    }

    fn parent(&self) -> Option<&Rc<Self>> {
        self.parent.as_ref()
    }
}

/// A trait to get an iterator of pointers to TransitionsWithCustomCost.
pub trait CustomCostNodeInterface<
    T: Numeric,
    U: Numeric,
    K: Hash + Eq + Clone + Debug = Rc<HashableSignatureVariables>,
    R: Deref<Target = TransitionWithCustomCost> = Rc<TransitionWithCustomCost>,
>: StateInformation<T, K> + PrioritizedNode<U>
{
    /// Create a new node given a g-value, an f-value, a state, a cost, a parent, and a transition.
    fn new(
        g: U,
        f: U,
        state: StateInRegistry<K>,
        cost: T,
        parent: Option<&Self>,
        transition: Option<R>,
    ) -> Self;

    /// Get transitions.
    fn transitions(&self) -> Vec<TransitionWithCustomCost>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::GroundedCondition;
    use dypdl::SignatureVariables;
    use std::rc::Rc;

    struct MockCustomCostNode {
        state: StateInRegistry,
        cost: i32,
        g: i32,
        f: i32,
    }

    impl StateInformation<i32> for MockCustomCostNode {
        fn cost(&self) -> i32 {
            self.cost
        }

        fn state(&self) -> &StateInRegistry {
            &self.state
        }
    }

    impl PrioritizedNode<i32> for MockCustomCostNode {
        fn f(&self) -> i32 {
            self.f
        }

        fn g(&self) -> i32 {
            self.g
        }
    }

    impl CustomCostNodeInterface<i32, i32> for MockCustomCostNode {
        fn new(
            g: i32,
            f: i32,
            state: StateInRegistry<Rc<HashableSignatureVariables>>,
            cost: i32,
            _: Option<&Self>,
            _: Option<Rc<TransitionWithCustomCost>>,
        ) -> Self {
            MockCustomCostNode { state, cost, g, f }
        }

        fn transitions(&self) -> Vec<TransitionWithCustomCost> {
            Vec::default()
        }
    }

    fn generate_model() -> dypdl::Model {
        dypdl::Model {
            forward_transitions: vec![Transition {
                name: String::from("forward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(1)),
                )),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                name: String::from("forward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(2)),
                )),
                ..Default::default()
            }],
            backward_transitions: vec![Transition {
                name: String::from("backward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(3)),
                )),
                ..Default::default()
            }],
            backward_forced_transitions: vec![Transition {
                name: String::from("backward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(4)),
                )),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn transition_with_custom_cost_to_transition() {
        let transition = TransitionWithCustomCost {
            transition: Transition {
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Le,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(2)),
                    ),
                    ..Default::default()
                }],
                cost: CostExpression::Integer(IntegerExpression::Constant(3)),
                ..Default::default()
            },
            custom_cost: CostExpression::Integer(IntegerExpression::Constant(4)),
            id: 0,
            forced: false,
        };
        let expected = Transition {
            preconditions: vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Le,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(2)),
                ),
                ..Default::default()
            }],
            cost: CostExpression::Integer(IntegerExpression::Constant(3)),
            ..Default::default()
        };
        assert_eq!(Transition::from(transition), expected);
    }

    #[test]
    fn is_applicable() {
        let transition = Rc::new(TransitionWithCustomCost {
            transition: Transition {
                preconditions: vec![GroundedCondition {
                    condition: Condition::ComparisonI(
                        ComparisonOperator::Le,
                        Box::new(IntegerExpression::Variable(0)),
                        Box::new(IntegerExpression::Constant(2)),
                    ),
                    ..Default::default()
                }],
                ..Default::default()
            },
            custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(2)),
            )),
            id: 0,
            forced: false,
        });

        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
                integer_variables: vec![2],
                ..Default::default()
            },
            ..Default::default()
        };
        let registry = dypdl::TableRegistry::default();
        assert!(transition.is_applicable(&state, &registry));
        let state = dypdl::State {
            signature_variables: dypdl::SignatureVariables {
                integer_variables: vec![3],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!transition.is_applicable(&state, &registry));
    }

    #[test]
    fn new() {
        let model = Rc::new(generate_model());

        let generator = SuccessorGenerator::from_model_without_custom_cost(model.clone(), false);

        let transitions = vec![Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("forward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(1)),
                )),
                ..Default::default()
            },
            custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(1)),
            )),
            forced: false,
            id: 0,
        })];
        let forced_transitions = vec![Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("forward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(2)),
                )),
                ..Default::default()
            },
            custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(2)),
            )),
            forced: true,
            id: 0,
        })];
        let expected = SuccessorGenerator::new(forced_transitions, transitions, false, model);
        assert_eq!(generator, expected);
    }

    #[test]
    fn new_backward() {
        let model = Rc::new(generate_model());

        let generator = SuccessorGenerator::from_model_without_custom_cost(model.clone(), true);
        let transitions = vec![Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("backward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(3)),
                )),
                ..Default::default()
            },
            custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(3)),
            )),
            forced: false,
            id: 0,
        })];
        let forced_transitions = vec![Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("backward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(4)),
                )),
                ..Default::default()
            },
            custom_cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(4)),
            )),
            forced: true,
            id: 0,
        })];
        let expected = SuccessorGenerator::new(forced_transitions, transitions, true, model);
        assert_eq!(generator, expected);
    }

    #[test]
    fn with_custom_costs() {
        let model = Rc::new(generate_model());

        let custom_costs = vec![CostExpression::Continuous(
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(1.0)),
            ),
        )];
        let forced_custom_costs = [CostExpression::Continuous(
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(2.0)),
                Box::new(ContinuousExpression::Constant(2.0)),
            ),
        )];
        let generator = SuccessorGenerator::from_model_with_custom_costs(
            model.clone(),
            &custom_costs,
            &forced_custom_costs,
            false,
        );
        let transitions = vec![Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("forward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(1)),
                )),
                ..Default::default()
            },
            custom_cost: CostExpression::Continuous(ContinuousExpression::Constant(2.0)),
            forced: false,
            id: 0,
        })];
        let forced_transitions = vec![Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("forward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(2)),
                )),
                ..Default::default()
            },
            custom_cost: CostExpression::Continuous(ContinuousExpression::Constant(4.0)),
            forced: true,
            id: 0,
        })];
        let expected = SuccessorGenerator::new(forced_transitions, transitions, false, model);
        assert_eq!(generator, expected);
    }

    #[test]
    fn with_custom_costs_backward() {
        let model = Rc::new(generate_model());

        let custom_costs = vec![CostExpression::Continuous(
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(3.0)),
                Box::new(ContinuousExpression::Constant(3.0)),
            ),
        )];
        let forced_custom_costs = [CostExpression::Continuous(
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(4.0)),
                Box::new(ContinuousExpression::Constant(4.0)),
            ),
        )];

        let generator = SuccessorGenerator::from_model_with_custom_costs(
            model.clone(),
            &custom_costs,
            &forced_custom_costs,
            true,
        );
        let transitions = vec![Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("backward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(3)),
                )),
                ..Default::default()
            },
            custom_cost: CostExpression::Continuous(ContinuousExpression::Constant(6.0)),
            forced: false,
            id: 0,
        })];
        let forced_transitions = vec![Rc::new(TransitionWithCustomCost {
            transition: Transition {
                name: String::from("backward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(4)),
                )),
                ..Default::default()
            },
            custom_cost: CostExpression::Continuous(ContinuousExpression::Constant(8.0)),
            forced: true,
            id: 0,
        })];
        let expected = SuccessorGenerator::new(forced_transitions, transitions, true, model);
        assert_eq!(generator, expected);
    }

    #[test]
    fn generate_successor_state_some() {
        let model = dypdl::Model::default();
        let state = dypdl::State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };
        let transition = TransitionWithCustomCost {
            transition: dypdl::Transition {
                name: String::from("op1"),
                cost: CostExpression::Integer(IntegerExpression::Constant(1)),
                effect: dypdl::Effect {
                    integer_effects: vec![(0, IntegerExpression::Constant(2))],
                    ..Default::default()
                },
                ..Default::default()
            },
            custom_cost: CostExpression::Integer(IntegerExpression::Constant(3)),
            forced: false,
            id: 0,
        };
        let h_evaluator = |_: &dypdl::State| Some(1);
        let f_evaluator = |_, _, _: &dypdl::State| 4;
        let parent = CustomCostParent {
            state: &state,
            cost: 0,
            g: 0,
        };
        let result = transition.generate_successor_state(
            &parent,
            &model,
            None,
            h_evaluator,
            f_evaluator,
            false,
        );
        assert!(result.is_some());
        let (state, cost, g, f) = result.unwrap();
        assert_eq!(
            state,
            dypdl::State {
                signature_variables: SignatureVariables {
                    integer_variables: vec![2],
                    ..Default::default()
                },
                ..Default::default()
            }
        );
        assert_eq!(cost, 1);
        assert_eq!(g, 3);
        assert_eq!(f, 4);
    }

    #[test]
    fn generate_successor_state_pruned_by_constraint() {
        let model = dypdl::Model {
            state_constraints: vec![GroundedCondition {
                condition: Condition::Constant(false),
                ..Default::default()
            }],
            ..Default::default()
        };
        let state = dypdl::State::default();
        let transition = TransitionWithCustomCost::default();
        let h_evaluator = |_: &dypdl::State| Some(1);
        let f_evaluator = |_, _, _: &dypdl::State| 4;
        let parent = CustomCostParent {
            state: &state,
            cost: 0,
            g: 0,
        };
        let result = transition.generate_successor_state(
            &parent,
            &model,
            None,
            h_evaluator,
            f_evaluator,
            false,
        );
        assert!(result.is_none());
    }

    #[test]
    fn generate_successor_state_pruned_by_bound() {
        let model = dypdl::Model::default();
        let state = dypdl::State::default();
        let transition = TransitionWithCustomCost::default();
        let h_evaluator = |_: &dypdl::State| Some(1);
        let f_evaluator = |_, _, _: &dypdl::State| 4;
        let parent = CustomCostParent {
            state: &state,
            cost: 0,
            g: 0,
        };
        let result = transition.generate_successor_state(
            &parent,
            &model,
            Some(4),
            h_evaluator,
            f_evaluator,
            false,
        );
        assert!(result.is_none());
    }

    #[test]
    fn chain_new_no_parent() {
        let op1 = Rc::new(TransitionWithCustomCost {
            transition: dypdl::Transition {
                name: String::from("op1"),
                ..Default::default()
            },
            ..Default::default()
        });
        let chain = TransitionWithCustomCostChain::new(None, op1.clone());
        assert_eq!(chain.parent(), None);
        assert_eq!(chain.last(), &*op1);
    }

    #[test]
    fn chain_new_with_parent() {
        let op1 = Rc::new(TransitionWithCustomCost {
            transition: dypdl::Transition {
                name: String::from("op1"),
                ..Default::default()
            },
            ..Default::default()
        });
        let chain1 = Rc::new(TransitionWithCustomCostChain::new(None, op1));
        let op2 = Rc::new(TransitionWithCustomCost {
            transition: dypdl::Transition {
                name: String::from("op2"),
                ..Default::default()
            },
            ..Default::default()
        });
        let chain2 = TransitionWithCustomCostChain::new(Some(chain1.clone()), op2.clone());
        assert_eq!(chain2.parent(), Some(&chain1));
        assert_eq!(chain2.last(), &*op2);
    }
}
