use super::function_expression;
use super::set_expression;
use std::boxed::Box;
use std::cmp;

use crate::problem;
use crate::state;
use crate::variable;

#[derive(Debug)]
pub enum NumericExpression<'a, T: variable::Numeric> {
    Constant(T),
    Variable(usize),
    ResourceVariable(usize),
    Cost,
    NumericOperation(
        NumericOperator,
        Box<NumericExpression<'a, T>>,
        Box<NumericExpression<'a, T>>,
    ),
    Cardinality(set_expression::SetExpression),
    Function(function_expression::FunctionExpression<'a, T>),
}

#[derive(Debug)]
pub enum NumericOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Max,
    Min,
}

impl<'a, T: variable::Numeric> NumericExpression<'a, T> {
    pub fn eval(&self, state: &state::State<T>, problem: &problem::Problem) -> T {
        match self {
            NumericExpression::Constant(x) => *x,
            NumericExpression::Variable(i) => state.signature_variables.numeric_variables[*i],
            NumericExpression::ResourceVariable(i) => state.resource_variables[*i],
            NumericExpression::Cost => state.cost,
            NumericExpression::NumericOperation(op, a, b) => {
                let a = a.eval(state, problem);
                let b = b.eval(state, problem);
                match op {
                    NumericOperator::Add => a + b,
                    NumericOperator::Subtract => a - b,
                    NumericOperator::Multiply => a * b,
                    NumericOperator::Divide => a / b,
                    NumericOperator::Max => cmp::max(a, b),
                    NumericOperator::Min => cmp::min(a, b),
                }
            }
            NumericExpression::Cardinality(set) => {
                T::from(set.eval(state, problem).count_ones(..)).unwrap()
            }
            NumericExpression::Function(f) => f.eval(state, &problem),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state;
    use std::rc::Rc;

    fn generate_problem() -> problem::Problem {
        problem::Problem {
            set_variable_to_max_size: vec![3],
            permutation_variable_to_max_length: vec![3],
            element_to_set: vec![0],
        }
    }

    fn generate_state() -> state::State<variable::IntegerVariable> {
        let mut set1 = variable::SetVariable::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = variable::SetVariable::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![set1, set2],
                permutation_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                numeric_variables: vec![1, 2, 3],
            }),
            resource_variables: vec![4, 5, 6],
            stage: 0,
            cost: 0,
        }
    }

    #[test]
    fn number_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = NumericExpression::Constant(2);
        assert_eq!(expression.eval(&state, &problem), 2);
    }

    #[test]
    fn numeric_variable_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = NumericExpression::Variable(0);
        assert_eq!(expression.eval(&state, &problem), 1);
        let expression = NumericExpression::Variable(1);
        assert_eq!(expression.eval(&state, &problem), 2);
        let expression = NumericExpression::Variable(2);
        assert_eq!(expression.eval(&state, &problem), 3);
    }

    #[test]
    fn resource_variable_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression = NumericExpression::ResourceVariable(0);
        assert_eq!(expression.eval(&state, &problem), 4);
        let expression = NumericExpression::ResourceVariable(1);
        assert_eq!(expression.eval(&state, &problem), 5);
        let expression = NumericExpression::ResourceVariable(2);
        assert_eq!(expression.eval(&state, &problem), 6);
    }

    #[test]
    fn cost_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> = NumericExpression::Cost {};
        assert_eq!(expression.eval(&state, &problem), 0);
    }

    #[test]
    fn add_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &problem), 5);
    }

    #[test]
    fn subtract_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Subtract,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &problem), 1);
    }

    #[test]
    fn multiply_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &problem), 6);
    }

    #[test]
    fn divide_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Divide,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &problem), 1);
    }

    #[test]
    fn max_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Max,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &problem), 3);
    }

    #[test]
    fn min_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Min,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&state, &problem), 2);
    }

    #[test]
    fn cardinality_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let expression =
            NumericExpression::Cardinality(set_expression::SetExpression::SetVariable(0));
        assert_eq!(expression.eval(&state, &problem), 2);
        let expression =
            NumericExpression::Cardinality(set_expression::SetExpression::SetVariable(1));
        assert_eq!(expression.eval(&state, &problem), 2);
    }
}
