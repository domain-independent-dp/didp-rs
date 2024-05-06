use crate::search_algorithm::data_structure::HashableSignatureVariables;
use crate::search_algorithm::BfsNode;
use crate::search_algorithm::SuccessorGenerator;
use crate::ConcurrentStateRegistry;
use dypdl::variable_type::Numeric;
use dypdl::{Model, TransitionInterface};
use std::fmt::Display;
use std::sync::Arc;

/// Iterator returning a node and its applicable transition.
pub struct SendableSuccessorIterator<'a, T, N, E, V>
where
    T: Numeric + Display,
    N: BfsNode<T, V, Arc<HashableSignatureVariables>>,
    E: Fn(&N, Arc<V>, Option<T>) -> Option<N>,
    V: TransitionInterface + Clone,
{
    node: Arc<N>,
    generator: &'a SuccessorGenerator<V, Arc<V>, Arc<Model>>,
    evaluator: E,
    registry: &'a ConcurrentStateRegistry<T, N>,
    primal_bound: Option<T>,
    iter: std::slice::Iter<'a, Arc<V>>,
    forced: bool,
    end: bool,
}

impl<'a, T, N, E, V> SendableSuccessorIterator<'a, T, N, E, V>
where
    T: Numeric + Display,
    N: BfsNode<T, V, Arc<HashableSignatureVariables>>,
    E: Fn(&N, Arc<V>, Option<T>) -> Option<N>,
    V: TransitionInterface + Clone,
{
    /// Creates a new iterator.
    pub fn new(
        node: Arc<N>,
        generator: &'a SuccessorGenerator<V, Arc<V>, Arc<Model>>,
        evaluator: E,
        registry: &'a ConcurrentStateRegistry<T, N>,
        primal_bound: Option<T>,
    ) -> Self {
        Self {
            node,
            generator,
            evaluator,
            registry,
            primal_bound,
            iter: generator.forced_transitions.iter(),
            forced: true,
            end: false,
        }
    }
}

impl<'a, T, N, E, V> Iterator for SendableSuccessorIterator<'a, T, N, E, V>
where
    T: Numeric + Display,
    N: BfsNode<T, V, Arc<HashableSignatureVariables>>,
    E: Fn(&N, Arc<V>, Option<T>) -> Option<N>,
    V: TransitionInterface + Clone,
{
    type Item = Arc<N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.end {
            return None;
        }

        match self.iter.next() {
            Some(op) => {
                if op.is_applicable(self.node.state(), &self.generator.model.table_registry) {
                    if self.forced {
                        self.end = true;
                    }

                    if let Some(successor) =
                        (self.evaluator)(&self.node, op.clone(), self.primal_bound)
                    {
                        let result = self.registry.insert(successor);

                        for d in result.dominated {
                            if !d.is_closed() {
                                d.close();
                            }
                        }

                        if let Some(node) = result.information {
                            Some(node)
                        } else {
                            self.next()
                        }
                    } else {
                        self.next()
                    }
                } else {
                    self.next()
                }
            }
            None => {
                if self.forced {
                    self.forced = false;
                    self.iter = self.generator.transitions.iter();
                    self.next()
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::search_node::SendableFNode;
    use super::*;
    use crate::search_algorithm::data_structure::StateInformation;
    use dypdl::prelude::*;

    #[test]
    fn generate_forced_transition() {
        let mut model = Model::default();
        let var = model.add_integer_variable("var", 1);
        assert!(var.is_ok());
        let var = var.unwrap();

        let mut transition = Transition::new("forced inapplicable");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Lt, var, 1));
        let result = transition.add_effect(var, var + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let result = model.add_forward_forced_transition(transition);
        assert!(result.is_ok());

        let mut transition = Transition::new("forced applicable 1");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Ge, var, 1));
        let result = transition.add_effect(var, var + 2);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 2);
        let result = model.generate_successor_state(&model.target, 0, &transition, None);
        assert!(result.is_some());
        let (expected_state, expected_cost) = result.unwrap();
        let result = model.add_forward_forced_transition(transition);
        assert!(result.is_ok());

        let mut transition = Transition::new("forced applicable 2");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Ge, var, 1));
        let result = transition.add_effect(var, var + 3);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 3);
        let result = model.add_forward_forced_transition(transition);
        assert!(result.is_ok());

        let mut transition = Transition::new("inapplicable");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Lt, var, 1));
        let result = transition.add_effect(var, var + 4);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 4);
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());

        let mut transition = Transition::new("applicable");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Ge, var, 1));
        let result = transition.add_effect(var, var + 5);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 5);
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());

        let model = Arc::new(model);
        let generator = SuccessorGenerator::<Transition, Arc<Transition>, Arc<_>>::from_model(
            model.clone(),
            false,
        );

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: Integer, h: Integer, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            model.target.clone(),
            0,
            &model,
            h_evaluator,
            f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = Arc::new(node.unwrap());
        let registry = ConcurrentStateRegistry::new(model.clone());
        let evaluator = |node: &SendableFNode<_>,
                         transition: Arc<Transition>,
                         primal_bound: Option<Integer>| {
            SendableFNode::generate_successor_node(
                node,
                transition,
                &model,
                h_evaluator,
                f_evaluator,
                primal_bound,
            )
        };

        let mut iter = SendableSuccessorIterator::new(node, &generator, evaluator, &registry, None);
        let node = iter.next();
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.cost(&model), expected_cost);
        assert_eq!(node.bound(&model), Some(expected_cost));
        assert_eq!(node.state(), &expected_state);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn generate_transitions() {
        let mut model = Model::default();
        let var = model.add_integer_variable("var", 1);
        assert!(var.is_ok());
        let var = var.unwrap();

        let mut transition = Transition::new("forced inapplicable");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Lt, var, 1));
        let result = transition.add_effect(var, var + 1);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 1);
        let result = model.add_forward_forced_transition(transition);
        assert!(result.is_ok());

        let mut transition = Transition::new("inapplicable");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Lt, var, 1));
        let result = transition.add_effect(var, var + 2);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 2);
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());

        let mut transition = Transition::new("applicable 1");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Ge, var, 1));
        let result = transition.add_effect(var, var + 3);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 3);
        let result = model.generate_successor_state(&model.target, 0, &transition, None);
        assert!(result.is_some());
        let (expected_state1, expected_cost1) = result.unwrap();
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());

        let mut transition = Transition::new("applicable 1");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Le, var, 1));
        let result = transition.add_effect(var, var + 4);
        assert!(result.is_ok());
        transition.set_cost(IntegerExpression::Cost + 4);
        let result = model.generate_successor_state(&model.target, 0, &transition, None);
        assert!(result.is_some());
        let (expected_state2, expected_cost2) = result.unwrap();
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());

        let model = Arc::new(model);
        let generator = SuccessorGenerator::<Transition, Arc<Transition>, Arc<_>>::from_model(
            model.clone(),
            false,
        );

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: Integer, h: Integer, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            model.target.clone(),
            0,
            &model,
            h_evaluator,
            f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = Arc::new(node.unwrap());
        let registry = ConcurrentStateRegistry::new(model.clone());
        let evaluator = |node: &SendableFNode<_>,
                         transition: Arc<Transition>,
                         primal_bound: Option<Integer>| {
            SendableFNode::generate_successor_node(
                node,
                transition,
                &model,
                h_evaluator,
                f_evaluator,
                primal_bound,
            )
        };

        let mut iter = SendableSuccessorIterator::new(node, &generator, evaluator, &registry, None);
        let node = iter.next();
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.cost(&model), expected_cost1);
        assert_eq!(node.bound(&model), Some(expected_cost1));
        assert_eq!(node.state(), &expected_state1);
        let node = iter.next();
        assert!(node.is_some());
        let node = node.unwrap();
        assert_eq!(node.cost(&model), expected_cost2);
        assert_eq!(node.bound(&model), Some(expected_cost2));
        assert_eq!(node.state(), &expected_state2);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn generate_no_transition() {
        let mut model = Model::default();
        let var = model.add_integer_variable("var", 1);
        assert!(var.is_ok());
        let var = var.unwrap();

        let mut transition = Transition::new("forced inapplicable");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Lt, var, 1));
        let result = model.add_forward_forced_transition(transition);
        assert!(result.is_ok());

        let mut transition = Transition::new("inapplicable");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Lt, var, 1));
        let result = model.add_forward_transition(transition);
        assert!(result.is_ok());

        let model = Arc::new(model);
        let generator = SuccessorGenerator::<Transition, Arc<Transition>, Arc<_>>::from_model(
            model.clone(),
            false,
        );

        let h_evaluator = |_: &_| Some(0);
        let f_evaluator = |g: Integer, h: Integer, _: &_| g + h;
        let node = SendableFNode::<_>::generate_root_node(
            model.target.clone(),
            0,
            &model,
            h_evaluator,
            f_evaluator,
            None,
        );
        assert!(node.is_some());
        let node = Arc::new(node.unwrap());

        let registry = ConcurrentStateRegistry::new(model.clone());
        let evaluator = |node: &SendableFNode<_>,
                         transition: Arc<Transition>,
                         primal_bound: Option<Integer>| {
            SendableFNode::generate_successor_node(
                node,
                transition,
                &model,
                h_evaluator,
                f_evaluator,
                primal_bound,
            )
        };
        let mut iter = SendableSuccessorIterator::new(node, &generator, evaluator, &registry, None);
        assert_eq!(iter.next(), None);
    }
}
