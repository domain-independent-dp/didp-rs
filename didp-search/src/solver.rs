use didp_parser::variable;
use std::error::Error;
use std::fmt;
use std::rc::Rc;

pub type Solution<T> = Option<(T, Vec<Rc<didp_parser::Transition<T>>>)>;

#[derive(Debug, Clone)]
pub struct ConfigErr(String);

impl ConfigErr {
    pub fn new(message: String) -> ConfigErr {
        ConfigErr(format!("Error in config: {}", message))
    }
}

impl fmt::Display for ConfigErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ConfigErr {}

pub trait Solver<T: variable::Numeric> {
    fn set_primal_bound(&mut self, _: Option<T>) {}

    fn solve(&mut self, model: &didp_parser::Model<T>) -> Result<Solution<T>, Box<dyn Error>>;
}

pub fn compute_solution_cost<T: variable::Numeric, U: didp_parser::DPState>(
    base_cost: T,
    transitions: &[Rc<didp_parser::Transition<T>>],
    state: &U,
    model: &didp_parser::Model<T>,
) -> T {
    let mut state_sequence = Vec::with_capacity(transitions.len());
    state_sequence.push(state.clone());
    for t in transitions[..transitions.len() - 1].iter() {
        state_sequence.push(t.apply(
            state_sequence.last().as_ref().unwrap(),
            &model.table_registry,
        ));
    }
    state_sequence.reverse();
    let mut transitions: Vec<Rc<didp_parser::Transition<T>>> = transitions.to_vec();
    transitions.reverse();
    let mut cost = base_cost;
    for (state, t) in state_sequence.into_iter().zip(transitions) {
        cost = t.eval_cost(cost, &state, &model.table_registry);
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use didp_parser::expression::*;

    #[test]
    fn test_compute_solution_cost() {
        let model = didp_parser::Model::<variable::Integer> {
            target: didp_parser::State {
                signature_variables: didp_parser::SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let transitions = vec![
            Rc::new(didp_parser::Transition::<variable::Integer> {
                effect: didp_parser::Effect {
                    integer_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::IntegerVariable(0)),
                            Box::new(NumericExpression::Constant(1)),
                        ),
                    )],
                    ..Default::default()
                },
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::IntegerVariable(0)),
                ),
                ..Default::default()
            }),
            Rc::new(didp_parser::Transition::<variable::Integer> {
                effect: didp_parser::Effect {
                    integer_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::IntegerVariable(0)),
                            Box::new(NumericExpression::Constant(2)),
                        ),
                    )],
                    ..Default::default()
                },
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::IntegerVariable(0)),
                ),
                ..Default::default()
            }),
            Rc::new(didp_parser::Transition::<variable::Integer> {
                effect: didp_parser::Effect {
                    integer_effects: vec![(
                        0,
                        NumericExpression::NumericOperation(
                            NumericOperator::Add,
                            Box::new(NumericExpression::IntegerVariable(0)),
                            Box::new(NumericExpression::Constant(3)),
                        ),
                    )],
                    ..Default::default()
                },
                cost: NumericExpression::NumericOperation(
                    NumericOperator::Add,
                    Box::new(NumericExpression::Cost),
                    Box::new(NumericExpression::IntegerVariable(0)),
                ),
                ..Default::default()
            }),
        ];
        assert_eq!(
            compute_solution_cost(0, &transitions, &model.target, &model),
            4
        );
    }
}
