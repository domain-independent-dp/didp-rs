use super::numeric_expression::NumericExpression;
use super::set_condition;
use crate::problem;
use crate::search_node;
use crate::variable;

pub enum Condition<'a, T: variable::Numeric> {
    Not(Box<Condition<'a, T>>),
    And(Box<Condition<'a, T>>, Box<Condition<'a, T>>),
    Or(Box<Condition<'a, T>>, Box<Condition<'a, T>>),
    Comparison(
        ComparisonOperator,
        NumericExpression<'a, T>,
        NumericExpression<'a, T>,
    ),
    Set(set_condition::SetCondition),
}

pub enum ComparisonOperator {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}

impl<'a, T: variable::Numeric> Condition<'a, T> {
    pub fn eval(&self, node: &search_node::SearchNode<T>, problem: &problem::Problem<T>) -> bool {
        match self {
            Condition::Not(c) => !c.eval(node, problem),
            Condition::And(x, y) => x.eval(node, problem) && y.eval(node, problem),
            Condition::Or(x, y) => x.eval(node, problem) || y.eval(node, problem),
            Condition::Comparison(op, x, y) => {
                Self::eval_comparison(op, x.eval(node, problem), y.eval(node, problem))
            }
            Condition::Set(c) => c.eval(&node.state, problem),
        }
    }

    fn eval_comparison(op: &ComparisonOperator, x: T, y: T) -> bool {
        match op {
            ComparisonOperator::Eq => x == y,
            ComparisonOperator::Ne => x != y,
            ComparisonOperator::Ge => x >= y,
            ComparisonOperator::Gt => x > y,
            ComparisonOperator::Le => x <= y,
            ComparisonOperator::Lt => x < y,
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
    fn eval_eq() {
        let problem = generate_problem();
        let node = generate_node();

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&node, &problem));
    }

    #[test]
    fn eval_neq() {
        let problem = generate_problem();
        let node = generate_node();

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&node, &problem));
    }

    #[test]
    fn eval_ge() {
        let problem = generate_problem();
        let node = generate_node();

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&node, &problem));
    }

    #[test]
    fn eval_gt() {
        let problem = generate_problem();
        let node = generate_node();

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&node, &problem));
    }

    #[test]
    fn eval_le() {
        let problem = generate_problem();
        let node = generate_node();

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&node, &problem));
    }

    #[test]
    fn eval_lt() {
        let problem = generate_problem();
        let node = generate_node();

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&node, &problem));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&node, &problem));
    }

    #[test]
    fn eval_not() {
        let problem = generate_problem();
        let node = generate_node();

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        )));
        assert!(!expression.eval(&node, &problem));

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        )));
        assert!(expression.eval(&node, &problem));
    }

    #[test]
    fn eval_and() {
        let problem = generate_problem();
        let node = generate_node();

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(expression.eval(&node, &problem));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(!expression.eval(&node, &problem));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(!expression.eval(&node, &problem));
    }

    #[test]
    fn eval_or() {
        let problem = generate_problem();
        let node = generate_node();

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(expression.eval(&node, &problem));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(expression.eval(&node, &problem));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(!expression.eval(&node, &problem));
    }
}
