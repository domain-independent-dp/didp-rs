use dypdl::variable_type;
use std::error::Error;
use std::rc::Rc;
use std::time;

pub struct TimeKeeper {
    time_limit: Option<time::Duration>,
    start: time::Instant,
}

impl Default for TimeKeeper {
    fn default() -> Self {
        TimeKeeper {
            time_limit: None,
            start: time::Instant::now(),
        }
    }
}

impl TimeKeeper {
    pub fn with_time_limit(time_limit: f64) -> TimeKeeper {
        TimeKeeper {
            time_limit: Some(time::Duration::from_secs_f64(time_limit)),
            start: time::Instant::now(),
        }
    }

    pub fn elapsed_time(&self) -> f64 {
        (time::Instant::now() - self.start).as_secs_f64()
    }

    pub fn remaining_time_limit(&self) -> Option<time::Duration> {
        self.time_limit
            .map(|time_limit| time_limit - (time::Instant::now() - self.start))
    }

    pub fn check_time_limit(&self) -> bool {
        self.time_limit.map_or(false, |time_limit| {
            if time::Instant::now() - self.start > time_limit {
                println!("Reached time limit.");
                true
            } else {
                false
            }
        })
    }
}

/// Common parameters for heuristic search solvers.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct SolverParameters<T> {
    /// Primal bound.
    pub primal_bound: Option<T>,
    /// Time limit.
    pub time_limit: Option<f64>,
    /// Suppress log output or not.
    pub quiet: bool,
}

impl<T> Default for SolverParameters<T> {
    fn default() -> Self {
        SolverParameters {
            primal_bound: None,
            time_limit: None,
            quiet: false,
        }
    }
}

/// Information about a solution.
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Solution<T: variable_type::Numeric> {
    /// Solution cost.
    pub cost: Option<T>,
    /// Best dual bound.
    pub best_bound: Option<T>,
    /// Solved to optimialty or not.
    pub is_optimal: bool,
    /// Infeasible model or not.
    pub is_infeasible: bool,
    /// Transitions corresponding to the solution.
    pub transitions: Vec<dypdl::Transition>,
    /// Number of expanded nodes.
    pub expanded: usize,
    /// Number of generated nodes.
    pub generated: usize,
    /// Elapsed time in seconds.
    pub time: f64,
}

pub type Callback<T> = dyn FnMut(&Solution<T>) + Send;

/// A trait defining the interface of a solver.
pub trait Solver<T: variable_type::Numeric> {
    /// Tries to find a solution.
    fn solve(&mut self, model: &dypdl::Model) -> Result<Solution<T>, Box<dyn Error>>;

    /// Sets the primal bound.
    fn set_primal_bound(&mut self, _: T) {}

    /// Returns the primal bound.
    fn get_primal_bound(&self) -> Option<T>;

    /// Sets the time limit.
    fn set_time_limit(&mut self, _: f64);

    /// Returns the time limit.
    fn get_time_limit(&self) -> Option<f64>;

    /// Sets if quiet.
    fn set_quiet(&mut self, _: bool);

    /// Get if quiet.
    fn get_quiet(&self) -> bool;
}

/// Returns the cost of applying transitions from a state.
pub fn compute_solution_cost<T: variable_type::Numeric, U: dypdl::DPState>(
    transitions: &[Rc<dypdl::Transition>],
    state: &U,
    model: &dypdl::Model,
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
    let mut transitions: Vec<Rc<dypdl::Transition>> = transitions.to_vec();
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
    use dypdl::expression::*;
    use dypdl::CostExpression;

    #[test]
    fn test_compute_solution_cost() {
        let model = dypdl::Model {
            target: dypdl::State {
                signature_variables: dypdl::SignatureVariables {
                    integer_variables: vec![0],
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let transitions = vec![
            Rc::new(dypdl::Transition {
                effect: dypdl::Effect {
                    integer_effects: vec![(
                        0,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(1)),
                        ),
                    )],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Variable(0)),
                )),
                ..Default::default()
            }),
            Rc::new(dypdl::Transition {
                effect: dypdl::Effect {
                    integer_effects: vec![(
                        0,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(2)),
                        ),
                    )],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Variable(0)),
                )),
                ..Default::default()
            }),
            Rc::new(dypdl::Transition {
                effect: dypdl::Effect {
                    integer_effects: vec![(
                        0,
                        IntegerExpression::BinaryOperation(
                            BinaryOperator::Add,
                            Box::new(IntegerExpression::Variable(0)),
                            Box::new(IntegerExpression::Constant(3)),
                        ),
                    )],
                    ..Default::default()
                },
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Variable(0)),
                )),
                ..Default::default()
            }),
        ];
        assert_eq!(
            compute_solution_cost::<variable_type::Integer, _>(&transitions, &model.target, &model),
            4
        );
    }
}
