use super::super::successor_generator::SuccessorGenerator;
use std::ops::Deref;

use dypdl::{variable_type::Numeric, Transition, TransitionInterface};

#[derive(Debug, PartialEq, Clone, Default)]
pub struct TransitionWithId<T = Transition>
where
    T: TransitionInterface,
{
    /// Transition.
    pub transition: T,
    /// If forced.
    pub forced: bool,
    /// ID.
    pub id: usize,
}

impl<T: TransitionInterface> TransitionInterface for TransitionWithId<T> {
    #[inline]
    fn is_applicable<S: dypdl::StateInterface>(
        &self,
        state: &S,
        registry: &dypdl::TableRegistry,
    ) -> bool {
        self.transition.is_applicable(state, registry)
    }

    #[inline]
    fn apply<S: dypdl::StateInterface, U: From<dypdl::State>>(
        &self,
        state: &S,
        registry: &dypdl::TableRegistry,
    ) -> U {
        self.transition.apply(state, registry)
    }

    #[inline]
    fn eval_cost<U: Numeric, S: dypdl::StateInterface>(
        &self,
        cost: U,
        state: &S,
        registry: &dypdl::TableRegistry,
    ) -> U {
        self.transition.eval_cost(cost, state, registry)
    }
}

impl<T> From<TransitionWithId<T>> for Transition
where
    T: TransitionInterface,
    Transition: From<T>,
{
    fn from(transition: TransitionWithId<T>) -> Self {
        transition.transition.into()
    }
}

impl<U, R> SuccessorGenerator<TransitionWithId, U, R>
where
    U: Deref<Target = TransitionWithId> + Clone + From<TransitionWithId>,
    R: Deref<Target = dypdl::Model>,
{
    pub fn from_model(model: R, backward: bool) -> Self {
        let forced_transitions = if backward {
            &model.backward_forced_transitions
        } else {
            &model.forward_forced_transitions
        };
        let forced_transitions = forced_transitions
            .iter()
            .enumerate()
            .map(|(id, t)| {
                U::from(TransitionWithId {
                    transition: t.clone(),
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
                U::from(TransitionWithId {
                    transition: t.clone(),
                    forced: false,
                    id,
                })
            })
            .collect();

        SuccessorGenerator::new(forced_transitions, transitions, backward, model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;
    use std::rc::Rc;

    #[test]
    fn transition_with_custom_cost_to_transition() {
        let mut transition = Transition::new("transition");
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition_with_custom_cost = TransitionWithId {
            transition: transition.clone(),
            forced: false,
            id: 0,
        };
        assert_eq!(Transition::from(transition_with_custom_cost), transition);
    }

    #[test]
    fn is_applicable() {
        let mut model = Model::default();
        let var = model.add_integer_variable("v", 0);
        assert!(var.is_ok());
        let var = var.unwrap();

        let mut transition = Transition::new("transition");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Le, var, 1));
        let transition = TransitionWithId {
            transition,
            forced: false,
            id: 0,
        };
        let state = model.target;
        assert!(transition.is_applicable(&state, &model.table_registry));
    }

    #[test]
    fn is_not_applicable() {
        let mut model = Model::default();
        let var = model.add_integer_variable("v", 0);
        assert!(var.is_ok());
        let var = var.unwrap();

        let mut transition = Transition::new("transition");
        transition.add_precondition(Condition::comparison_i(ComparisonOperator::Le, var, 0));
        let transition = TransitionWithId {
            transition,
            forced: false,
            id: 0,
        };
        assert!(transition.is_applicable(&model.target, &model.table_registry));
    }

    #[test]
    fn apply() {
        let mut model = Model::default();
        let var1 = model.add_integer_variable("var1", 0);
        assert!(var1.is_ok());
        let var1 = var1.unwrap();
        let var2 = model.add_integer_variable("var2", 0);
        assert!(var2.is_ok());

        let mut transition = Transition::new("transition");
        let result = transition.add_effect(var1, var1 + 1);
        assert!(result.is_ok());
        let transition = TransitionWithId {
            transition,
            id: 0,
            forced: false,
        };

        let state: State = transition.apply(&model.target, &model.table_registry);
        assert_eq!(state.get_integer_variable(0), 1);
        assert_eq!(state.get_integer_variable(1), 0);
    }

    #[test]
    fn eval_cost() {
        let model = Model::default();

        let mut transition = Transition::new("transition");
        transition.set_cost(IntegerExpression::Cost + 1);
        let transition = TransitionWithId {
            transition,
            id: 0,
            forced: false,
        };
        let cost = transition.eval_cost(0, &model.target, &model.table_registry);
        assert_eq!(cost, 1);
    }

    #[test]
    fn from_model_forward() {
        let mut model = Model::default();
        let mut transition1 = Transition::new("transition1");
        transition1.set_cost(IntegerExpression::Cost + 1);
        let result = model.add_forward_transition(transition1.clone());
        assert!(result.is_ok());
        let mut transition2 = Transition::new("transition2");
        transition2.set_cost(IntegerExpression::Cost + 2);
        let result = model.add_forward_transition(transition2.clone());
        assert!(result.is_ok());
        let mut transition3 = Transition::new("transition3");
        transition3.set_cost(IntegerExpression::Cost + 3);
        let result = model.add_forward_forced_transition(transition3.clone());
        assert!(result.is_ok());
        let mut transition4 = Transition::new("transition4");
        transition4.set_cost(IntegerExpression::Cost + 4);
        let result = model.add_forward_forced_transition(transition4.clone());
        assert!(result.is_ok());
        let mut transition5 = Transition::new("transition5");
        transition5.set_cost(IntegerExpression::Cost + 5);
        let result = model.add_backward_transition(transition5.clone());
        assert!(result.is_ok());
        let mut transition6 = Transition::new("transition6");
        transition6.set_cost(IntegerExpression::Cost + 6);
        let result = model.add_backward_forced_transition(transition6.clone());
        assert!(result.is_ok());
        let model = Rc::new(model);

        let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), false);

        assert_eq!(generator.model, model);
        assert_eq!(
            generator.transitions,
            vec![
                Rc::new(TransitionWithId {
                    transition: transition1,
                    id: 0,
                    forced: false,
                }),
                Rc::new(TransitionWithId {
                    transition: transition2,
                    id: 1,
                    forced: false,
                }),
            ]
        );
        assert_eq!(
            generator.forced_transitions,
            vec![
                Rc::new(TransitionWithId {
                    transition: transition3,
                    id: 0,
                    forced: true,
                }),
                Rc::new(TransitionWithId {
                    transition: transition4,
                    id: 1,
                    forced: true,
                }),
            ]
        );
    }

    #[test]
    fn from_model_backward() {
        let mut model = Model::default();
        let mut transition1 = Transition::new("transition1");
        transition1.set_cost(IntegerExpression::Cost + 1);
        let result = model.add_backward_transition(transition1.clone());
        assert!(result.is_ok());
        let mut transition2 = Transition::new("transition2");
        transition2.set_cost(IntegerExpression::Cost + 2);
        let result = model.add_backward_transition(transition2.clone());
        assert!(result.is_ok());
        let mut transition3 = Transition::new("transition3");
        transition3.set_cost(IntegerExpression::Cost + 3);
        let result = model.add_backward_forced_transition(transition3.clone());
        assert!(result.is_ok());
        let mut transition4 = Transition::new("transition4");
        transition4.set_cost(IntegerExpression::Cost + 4);
        let result = model.add_backward_forced_transition(transition4.clone());
        assert!(result.is_ok());
        let mut transition5 = Transition::new("transition5");
        transition5.set_cost(IntegerExpression::Cost + 5);
        let result = model.add_forward_transition(transition5.clone());
        assert!(result.is_ok());
        let mut transition6 = Transition::new("transition6");
        transition6.set_cost(IntegerExpression::Cost + 6);
        let result = model.add_forward_forced_transition(transition6.clone());
        assert!(result.is_ok());
        let model = Rc::new(model);

        let generator = SuccessorGenerator::<TransitionWithId>::from_model(model.clone(), true);

        assert_eq!(generator.model, model);
        assert_eq!(
            generator.transitions,
            vec![
                Rc::new(TransitionWithId {
                    transition: transition1,
                    id: 0,
                    forced: false,
                }),
                Rc::new(TransitionWithId {
                    transition: transition2,
                    id: 1,
                    forced: false,
                }),
            ]
        );
        assert_eq!(
            generator.forced_transitions,
            vec![
                Rc::new(TransitionWithId {
                    transition: transition3,
                    id: 0,
                    forced: true,
                }),
                Rc::new(TransitionWithId {
                    transition: transition4,
                    id: 1,
                    forced: true,
                }),
            ]
        );
    }
}
