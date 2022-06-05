use didp_parser::variable;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;
use std::time;

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

pub struct TimeKeeper {
    time_limit: time::Duration,
    start: time::Instant,
}

impl TimeKeeper {
    pub fn new(time_limit: u64) -> TimeKeeper {
        TimeKeeper {
            time_limit: time::Duration::new(time_limit, 0),
            start: time::Instant::now(),
        }
    }

    pub fn remaining_time_limit(&self) -> time::Duration {
        self.time_limit - (time::Instant::now() - self.start)
    }

    pub fn check_time_limit(&self) -> bool {
        if time::Instant::now() - self.start > self.time_limit {
            println!("Reached time limit.");
            true
        } else {
            false
        }
    }
}

impl Error for ConfigErr {}
#[derive(Debug, Default, Clone, Copy)]
pub struct SolverParameters<T> {
    pub primal_bound: Option<T>,
    pub time_limit: Option<u64>,
}

impl<T: variable::Numeric> SolverParameters<T> {
    pub fn parse_from_map(
        map: &linked_hash_map::LinkedHashMap<yaml_rust::Yaml, yaml_rust::Yaml>,
    ) -> Result<SolverParameters<T>, ConfigErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let primal_bound = match map.get(&yaml_rust::Yaml::from_str("primal_bound")) {
            Some(yaml_rust::Yaml::Integer(value)) => {
                Some(T::from_integer(*value as variable::Integer))
            }
            Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().map_err(|e| {
                ConfigErr::new(format!("could not parse {} as a number: {:?}", value, e))
            })?),
            None => None,
            value => {
                return Err(ConfigErr::new(format!(
                    "expected Integer or Real, but found `{:?}`",
                    value
                )))
            }
        };
        let time_limit = match map.get(&yaml_rust::Yaml::from_str("time_limit")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as u64),
            None => None,
            value => {
                return Err(ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                )))
            }
        };
        Ok(SolverParameters {
            primal_bound,
            time_limit,
        })
    }
}

#[derive(Debug, Default)]
pub struct Solution<T: variable::Numeric> {
    pub cost: Option<T>,
    pub best_bound: Option<T>,
    pub is_optimal: bool,
    pub is_infeasible: bool,
    pub transitions: Vec<Rc<didp_parser::Transition<T>>>,
}

pub trait Solver<T: variable::Numeric> {
    fn solve(&mut self, model: &didp_parser::Model<T>) -> Result<Solution<T>, Box<dyn Error>>;

    fn set_primal_bound(&mut self, _: T) {}

    fn set_time_limit(&mut self, _: u64);
}

pub fn compute_solution_cost<T: variable::Numeric, U: didp_parser::DPState>(
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
    let mut cost = T::zero();
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
            compute_solution_cost(&transitions, &model.target, &model),
            4
        );
    }
}
