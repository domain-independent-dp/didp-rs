use std::boxed::Box;
use std::cmp;
use std::collections;
use std::iter;

use crate::search_node;
use crate::state;
use crate::variable;

pub enum NumericExpression<T: variable::Numeric> {
    Number(T),
    G,
    Variable(usize),
    ResourceVariable(usize),
    NumericOperation(
        NumericOperator,
        Box<NumericExpression<T>>,
        Box<NumericExpression<T>>,
    ),
    Cardinality(PropositionalExpression),
    Function(NumericFunction<T>, Vec<PropositionalExpression>),
}

impl<T: variable::Numeric> NumericExpression<T> {
    pub fn eval(&self, node: &search_node::SearchNode<T>) -> T {
        match self {
            NumericExpression::Number(x) => *x,
            NumericExpression::G => node.g,
            NumericExpression::Variable(i) => node.state.signature_variables.numeric_variables[*i],
            NumericExpression::ResourceVariable(i) => {
                node.state.resource_variables.numeric_variables[*i]
            }
            NumericExpression::NumericOperation(op, a, b) => {
                let a = a.eval(node);
                let b = b.eval(node);
                match op {
                    NumericOperator::Add => a + b,
                    NumericOperator::Subtract => a - b,
                    NumericOperator::Multiply => a * b,
                    NumericOperator::Divide => a / b,
                    NumericOperator::Max => cmp::max(a, b),
                    NumericOperator::Min => cmp::min(a, b),
                }
            }
            NumericExpression::Cardinality(expression) => match expression.eval(&node.state) {
                Argument::Element(_) => T::one(),
                Argument::Set(s) => T::from(s.count_ones(..)).unwrap(),
            },
            NumericExpression::Function(f, args) => f.eval(&args, &node.state),
        }
    }
}

pub enum NumericOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Max,
    Min,
}

pub struct NumericFunction<T: variable::Numeric>(
    collections::HashMap<Vec<variable::ElementVariable>, T>,
);

impl<T: variable::Numeric> NumericFunction<T> {
    pub fn eval(&self, args: &[PropositionalExpression], state: &state::State<T>) -> T {
        let mut result = vec![vec![]];
        for v in args {
            match v.eval(state) {
                Argument::Set(s) => {
                    result = result
                        .into_iter()
                        .flat_map(|x| {
                            iter::repeat(x).zip(s.ones()).map(|(mut x, e)| {
                                x.push(e);
                                x
                            })
                        })
                        .collect()
                }
                Argument::Element(e) => {
                    for r in &mut result {
                        r.push(e);
                    }
                }
            }
        }
        result.into_iter().map(|x| self.0[&x]).sum()
    }
}

pub enum PropositionalExpression {
    ElementVariable(usize),
    SetVariable(usize),
    PropositonalOperation(
        PropositionalOperator,
        Box<PropositionalExpression>,
        Box<PropositionalExpression>,
    ),
}

impl PropositionalExpression {
    pub fn eval<T: variable::Numeric>(&self, state: &state::State<T>) -> Argument {
        match self {
            PropositionalExpression::ElementVariable(i) => {
                Argument::Element(state.signature_variables.element_variables[*i])
            }
            PropositionalExpression::SetVariable(i) => {
                Argument::Set(state.signature_variables.set_variables[*i].clone())
            }
            PropositionalExpression::PropositonalOperation(op, a, b) => {
                let a = Self::eval_argument(a, state);
                let b = Self::eval_argument(b, state);
                match op {
                    PropositionalOperator::Union => Argument::Set(a.union(&b).collect()),
                    PropositionalOperator::Difference => Argument::Set(a.difference(&b).collect()),
                    PropositionalOperator::Intersection => {
                        Argument::Set(a.intersection(&b).collect())
                    }
                }
            }
        }
    }

    fn eval_argument<T: variable::Numeric>(
        argument: &PropositionalExpression,
        state: &state::State<T>,
    ) -> variable::SetVariable {
        match argument.eval(state) {
            Argument::Set(x) => x,
            Argument::Element(x) => variable::SetVariable::new(),
        }
    }
}

enum Argument {
    Set(variable::SetVariable),
    Element(variable::ElementVariable),
}

pub enum PropositionalOperator {
    Union,
    Difference,
    Intersection,
}
