use super::function_expression;
use super::set_expression;
use std::boxed::Box;
use std::cmp;

use crate::problem;
use crate::search_node;
use crate::variable;

#[derive(Debug)]
pub enum NumericExpression<'a, T: variable::Numeric> {
    Constant(T),
    Variable(usize),
    ResourceVariable(usize),
    G,
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
    pub fn eval(&self, node: &search_node::SearchNode<T>, problem: &problem::Problem<T>) -> T {
        match self {
            NumericExpression::Constant(x) => *x,
            NumericExpression::Variable(i) => node.state.signature_variables.numeric_variables[*i],
            NumericExpression::ResourceVariable(i) => {
                node.state.resource_variables.numeric_variables[*i]
            }
            NumericExpression::G => node.g,
            NumericExpression::NumericOperation(op, a, b) => {
                let a = a.eval(node, problem);
                let b = b.eval(node, problem);
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
                T::from(set.eval(&node.state, problem).count_ones(..)).unwrap()
            }
            NumericExpression::Function(f) => f.eval(&node.state, &problem),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_problem() -> problem::Problem<variable::IntegerVariable> {
        problem::Problem {
            set_variable_to_max_size: vec![3],
            permutation_variable_to_max_length: vec![3],
            element_to_set: vec![0],
            functions_1d: HashMap::new(),
            functions_2d: HashMap::new(),
            functions_3d: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    fn generate_node() -> search_node::SearchNode<variable::IntegerVariable> {
        let mut set1 = variable::SetVariable::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = variable::SetVariable::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        search_node::SearchNode {
            state: state::State {
                signature_variables: Rc::new(state::SignatureVariables {
                    set_variables: vec![set1, set2],
                    permutation_variables: vec![vec![0, 2]],
                    element_variables: vec![1],
                    numeric_variables: vec![1, 2, 3],
                }),
                resource_variables: state::ResourceVariables {
                    numeric_variables: vec![4, 5, 6],
                },
            },
            g: 0,
            h: RefCell::new(None),
            f: RefCell::new(None),
            parent: None,
            closed: RefCell::new(false),
        }
    }

    #[test]
    fn number_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = NumericExpression::Constant(2);
        assert_eq!(expression.eval(&node, &problem), 2);
    }

    #[test]
    fn numeric_variable_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = NumericExpression::Variable(0);
        assert_eq!(expression.eval(&node, &problem), 1);
        let expression = NumericExpression::Variable(1);
        assert_eq!(expression.eval(&node, &problem), 2);
        let expression = NumericExpression::Variable(2);
        assert_eq!(expression.eval(&node, &problem), 3);
    }

    #[test]
    fn resource_variable_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = NumericExpression::ResourceVariable(0);
        assert_eq!(expression.eval(&node, &problem), 4);
        let expression = NumericExpression::ResourceVariable(1);
        assert_eq!(expression.eval(&node, &problem), 5);
        let expression = NumericExpression::ResourceVariable(2);
        assert_eq!(expression.eval(&node, &problem), 6);
    }

    #[test]
    fn g_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression: NumericExpression<variable::IntegerVariable> = NumericExpression::G {};
        assert_eq!(expression.eval(&node, &problem), 0);
    }

    #[test]
    fn add_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&node, &problem), 5);
    }

    #[test]
    fn subtract_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Subtract,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&node, &problem), 1);
    }

    #[test]
    fn multiply_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&node, &problem), 6);
    }

    #[test]
    fn divide_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Divide,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&node, &problem), 1);
    }

    #[test]
    fn max_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Max,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&node, &problem), 3);
    }

    #[test]
    fn min_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression: NumericExpression<variable::IntegerVariable> =
            NumericExpression::NumericOperation(
                NumericOperator::Min,
                Box::new(NumericExpression::Constant(3)),
                Box::new(NumericExpression::Constant(2)),
            );
        assert_eq!(expression.eval(&node, &problem), 2);
    }

    #[test]
    fn cardinality_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression =
            NumericExpression::Cardinality(set_expression::SetExpression::SetVariable(0));
        assert_eq!(expression.eval(&node, &problem), 2);
        let expression =
            NumericExpression::Cardinality(set_expression::SetExpression::SetVariable(1));
        assert_eq!(expression.eval(&node, &problem), 2);
    }
}
