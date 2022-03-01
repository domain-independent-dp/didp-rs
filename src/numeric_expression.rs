use std::boxed::Box;
use std::cmp;
use std::collections;
use std::iter;

use crate::search_node;
use crate::state;
use crate::variable;

pub enum NumericExpression<'a, T: variable::Numeric> {
    Number(T),
    Variable(usize),
    ResourceVariable(usize),
    G,
    NumericOperation(
        NumericOperator,
        Box<NumericExpression<'a, T>>,
        Box<NumericExpression<'a, T>>,
    ),
    Cardinality(SetExpression),
    Function1D(&'a NumericFunction1D<T>, ElementExpression),
    Function1DSum(&'a NumericFunction1D<T>, SetExpression),
    Function2D(
        &'a NumericFunction2D<T>,
        ElementExpression,
        ElementExpression,
    ),
    Function2DSum(&'a NumericFunction2D<T>, SetExpression, SetExpression),
    Function2DSumPartial1(&'a NumericFunction2D<T>, SetExpression, ElementExpression),
    Function2DSumPartial2(&'a NumericFunction2D<T>, ElementExpression, SetExpression),
    Function3D(
        &'a NumericFunction3D<T>,
        ElementExpression,
        ElementExpression,
        ElementExpression,
    ),
    Function3DSum(
        &'a NumericFunction3D<T>,
        SetExpression,
        SetExpression,
        SetExpression,
    ),
    Function(&'a NumericFunction<T>, Vec<ElementExpression>),
    FunctionSum(&'a NumericFunction<T>, Vec<ArgumentExpression>),
}

impl<'a, T: variable::Numeric> NumericExpression<'a, T> {
    pub fn eval(&self, node: &search_node::SearchNode<T>) -> T {
        match self {
            NumericExpression::Number(x) => *x,
            NumericExpression::Variable(i) => node.state.signature_variables.numeric_variables[*i],
            NumericExpression::ResourceVariable(i) => {
                node.state.resource_variables.numeric_variables[*i]
            }
            NumericExpression::G => node.g,
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
            NumericExpression::Cardinality(set) => {
                T::from(set.eval(&node.state).count_ones(..)).unwrap()
            }
            NumericExpression::Function1D(f, x) => f.eval(x, &node.state),
            NumericExpression::Function1DSum(f, x) => f.sum(x, &node.state),
            NumericExpression::Function2D(f, x, y) => f.eval(x, y, &node.state),
            NumericExpression::Function2DSum(f, x, y) => f.sum(x, y, &node.state),
            NumericExpression::Function2DSumPartial1(f, x, y) => f.sum_partial1(x, y, &node.state),
            NumericExpression::Function2DSumPartial2(f, x, y) => f.sum_partial2(x, y, &node.state),
            NumericExpression::Function3D(f, x, y, z) => f.eval(x, y, z, &node.state),
            NumericExpression::Function3DSum(f, x, y, z) => f.sum(x, y, z, &node.state),
            NumericExpression::Function(f, args) => f.eval(args, &node.state),
            NumericExpression::FunctionSum(f, args) => f.sum(args, &node.state),
        }
    }
}

pub enum ElementExpression {
    Number(variable::ElementVariable),
    ElementVariable(usize),
}

impl ElementExpression {
    pub fn eval<T: variable::Numeric>(&self, state: &state::State<T>) -> variable::ElementVariable {
        match self {
            ElementExpression::Number(x) => *x,
            ElementExpression::ElementVariable(i) => {
                state.signature_variables.element_variables[*i]
            }
        }
    }
}

pub enum SetExpression {
    SetVariable(usize),
    SetOperation(SetOperator, Box<SetExpression>, Box<SetExpression>),
    SetElementOperation(SetElementOperator, Box<SetExpression>, ElementExpression),
}

impl SetExpression {
    pub fn eval<T: variable::Numeric>(&self, state: &state::State<T>) -> variable::SetVariable {
        match self {
            SetExpression::SetVariable(i) => state.signature_variables.set_variables[*i].clone(),
            SetExpression::SetOperation(op, a, b) => {
                let mut a = a.eval(&state);
                let b = b.eval(&state);
                match op {
                    SetOperator::Union => {
                        a.union_with(&b);
                        a
                    }
                    SetOperator::Difference => {
                        a.difference_with(&b);
                        a
                    }
                    SetOperator::Intersect => {
                        a.intersect_with(&b);
                        a
                    }
                }
            }
            SetExpression::SetElementOperation(op, s, e) => {
                let mut s = s.eval(&state);
                let e = e.eval(&state);
                match op {
                    SetElementOperator::Add => {
                        s.set(e, true);
                        s
                    }
                    SetElementOperator::Remove => {
                        s.set(e, false);
                        s
                    }
                }
            }
        }
    }
}

pub enum ArgumentExpression {
    Set(SetExpression),
    Element(ElementExpression),
}

pub enum NumericOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Max,
    Min,
}

pub enum SetOperator {
    Union,
    Difference,
    Intersect,
}

pub enum SetElementOperator {
    Add,
    Remove,
}

pub struct NumericFunction1D<T: variable::Numeric>(Vec<T>);

impl<T: variable::Numeric> NumericFunction1D<T> {
    pub fn eval(&self, x: &ElementExpression, state: &state::State<T>) -> T {
        self.0[x.eval(state)]
    }

    pub fn sum(&self, x: &SetExpression, state: &state::State<T>) -> T {
        x.eval(state).ones().map(|x| self.0[x]).sum()
    }
}

pub struct NumericFunction2D<T: variable::Numeric>(Vec<Vec<T>>);

impl<T: variable::Numeric> NumericFunction2D<T> {
    pub fn eval(&self, x: &ElementExpression, y: &ElementExpression, state: &state::State<T>) -> T {
        self.0[x.eval(state)][y.eval(state)]
    }

    pub fn sum(&self, x: &SetExpression, y: &SetExpression, state: &state::State<T>) -> T {
        let x = x.eval(state);
        let y = y.eval(state);
        x.ones().map(|x| y.ones().map(|y| self.0[x][y]).sum()).sum()
    }

    pub fn sum_partial1(
        &self,
        x: &SetExpression,
        y: &ElementExpression,
        state: &state::State<T>,
    ) -> T {
        x.eval(state)
            .ones()
            .zip(iter::repeat(y.eval(state)))
            .map(|(x, y)| self.0[x][y])
            .sum()
    }

    pub fn sum_partial2(
        &self,
        x: &ElementExpression,
        y: &SetExpression,
        state: &state::State<T>,
    ) -> T {
        y.eval(state)
            .ones()
            .zip(iter::repeat(x.eval(state)))
            .map(|(y, x)| self.0[x][y])
            .sum()
    }
}

pub struct NumericFunction3D<T: variable::Numeric>(Vec<Vec<Vec<T>>>);

impl<T: variable::Numeric> NumericFunction3D<T> {
    pub fn eval(
        &self,
        x: &ElementExpression,
        y: &ElementExpression,
        z: &ElementExpression,
        state: &state::State<T>,
    ) -> T {
        self.0[x.eval(state)][y.eval(state)][z.eval(state)]
    }

    pub fn sum(
        &self,
        x: &SetExpression,
        y: &SetExpression,
        z: &SetExpression,
        state: &state::State<T>,
    ) -> T {
        let x = x.eval(state);
        let y = y.eval(state);
        let z = z.eval(state);
        x.ones()
            .map(|x| {
                y.ones()
                    .map(|y| z.ones().map(|z| self.0[x][y][z]).sum())
                    .sum()
            })
            .sum()
    }
}

pub struct NumericFunction<T: variable::Numeric>(
    collections::HashMap<Vec<variable::ElementVariable>, T>,
);

impl<T: variable::Numeric> NumericFunction<T> {
    pub fn eval(&self, args: &[ElementExpression], state: &state::State<T>) -> T {
        let args: Vec<variable::ElementVariable> = args.iter().map(|x| x.eval(state)).collect();
        self.0[&args]
    }

    pub fn sum(&self, args: &[ArgumentExpression], state: &state::State<T>) -> T {
        let mut result = vec![vec![]];
        for v in args {
            match v {
                ArgumentExpression::Set(s) => {
                    let s = s.eval(state);
                    result = result
                        .into_iter()
                        .map(|mut v| {
                            s.ones().for_each(|x| v.push(x));
                            v
                        })
                        .collect();
                }
                ArgumentExpression::Element(e) => {
                    for r in &mut result {
                        r.push(e.eval(state));
                    }
                }
            }
        }
        result.into_iter().map(|x| self.0[&x]).sum()
    }
}
