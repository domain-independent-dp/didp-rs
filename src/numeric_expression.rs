use std::boxed::Box;
use std::cmp;
use std::collections;
use std::iter;

use crate::problem;
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
    Function2DSumX(&'a NumericFunction2D<T>, SetExpression, ElementExpression),
    Function2DSumY(&'a NumericFunction2D<T>, ElementExpression, SetExpression),
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
    pub fn eval(&self, node: &search_node::SearchNode<T>, problem: &problem::Problem) -> T {
        match self {
            NumericExpression::Number(x) => *x,
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
            NumericExpression::Function1D(f, x) => f.eval(x, &node.state),
            NumericExpression::Function1DSum(f, x) => f.sum(x, &node.state, problem),
            NumericExpression::Function2D(f, x, y) => f.eval(x, y, &node.state),
            NumericExpression::Function2DSum(f, x, y) => f.sum(x, y, &node.state, problem),
            NumericExpression::Function2DSumX(f, x, y) => f.sum_x(x, y, &node.state, problem),
            NumericExpression::Function2DSumY(f, x, y) => f.sum_y(x, y, &node.state, problem),
            NumericExpression::Function3D(f, x, y, z) => f.eval(x, y, z, &node.state),
            NumericExpression::Function3DSum(f, x, y, z) => f.sum(x, y, z, &node.state, problem),
            NumericExpression::Function(f, args) => f.eval(args, &node.state),
            NumericExpression::FunctionSum(f, args) => f.sum(args, &node.state, problem),
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
    PermutationVariable(usize),
    SetOperation(SetOperator, Box<SetExpression>, Box<SetExpression>),
    SetElementOperation(SetElementOperator, Box<SetExpression>, ElementExpression),
}

impl SetExpression {
    pub fn eval<T: variable::Numeric>(
        &self,
        state: &state::State<T>,
        problem: &problem::Problem,
    ) -> variable::SetVariable {
        match self {
            SetExpression::SetVariable(i) => state.signature_variables.set_variables[*i].clone(),
            SetExpression::PermutationVariable(i) => {
                let mut set = variable::SetVariable::with_capacity(
                    problem.permutation_variable_to_max_length[*i],
                );
                for v in &state.signature_variables.permutation_variables[*i] {
                    set.insert(*v);
                }
                set
            }
            SetExpression::SetOperation(op, a, b) => {
                let mut a = a.eval(&state, problem);
                let b = b.eval(&state, problem);
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
                let mut s = s.eval(&state, problem);
                let e = e.eval(&state);
                match op {
                    SetElementOperator::Add => {
                        s.insert(e);
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

    pub fn sum(&self, x: &SetExpression, state: &state::State<T>, problem: &problem::Problem) -> T {
        x.eval(state, problem).ones().map(|x| self.0[x]).sum()
    }
}

pub struct NumericFunction2D<T: variable::Numeric>(Vec<Vec<T>>);

impl<T: variable::Numeric> NumericFunction2D<T> {
    pub fn eval(&self, x: &ElementExpression, y: &ElementExpression, state: &state::State<T>) -> T {
        self.0[x.eval(state)][y.eval(state)]
    }

    pub fn sum(
        &self,
        x: &SetExpression,
        y: &SetExpression,
        state: &state::State<T>,
        problem: &problem::Problem,
    ) -> T {
        let x = x.eval(state, problem);
        let y = y.eval(state, problem);
        x.ones().map(|x| y.ones().map(|y| self.0[x][y]).sum()).sum()
    }

    pub fn sum_x(
        &self,
        x: &SetExpression,
        y: &ElementExpression,
        state: &state::State<T>,
        problem: &problem::Problem,
    ) -> T {
        x.eval(state, problem)
            .ones()
            .zip(iter::repeat(y.eval(state)))
            .map(|(x, y)| self.0[x][y])
            .sum()
    }

    pub fn sum_y(
        &self,
        x: &ElementExpression,
        y: &SetExpression,
        state: &state::State<T>,
        problem: &problem::Problem,
    ) -> T {
        y.eval(state, problem)
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
        problem: &problem::Problem,
    ) -> T {
        let x = x.eval(state, problem);
        let y = y.eval(state, problem);
        let z = z.eval(state, problem);
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

    pub fn sum(
        &self,
        args: &[ArgumentExpression],
        state: &state::State<T>,
        problem: &problem::Problem,
    ) -> T {
        let mut result = vec![vec![]];
        for v in args {
            match v {
                ArgumentExpression::Set(s) => {
                    let s = s.eval(state, problem);
                    result = result
                        .into_iter()
                        .flat_map(|r| {
                            iter::repeat(r)
                                .zip(s.ones())
                                .map(|(mut r, e)| {
                                    r.push(e);
                                    r
                                })
                                .collect::<Vec<Vec<variable::ElementVariable>>>()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn generate_problem() -> problem::Problem {
        problem::Problem {
            set_variable_to_max_size: vec![3],
            permutation_variable_to_max_length: vec![3],
            element_to_set: vec![0],
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
        let expression = NumericExpression::Number(2);
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
                Box::new(NumericExpression::Number(3)),
                Box::new(NumericExpression::Number(2)),
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
                Box::new(NumericExpression::Number(3)),
                Box::new(NumericExpression::Number(2)),
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
                Box::new(NumericExpression::Number(3)),
                Box::new(NumericExpression::Number(2)),
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
                Box::new(NumericExpression::Number(3)),
                Box::new(NumericExpression::Number(2)),
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
                Box::new(NumericExpression::Number(3)),
                Box::new(NumericExpression::Number(2)),
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
                Box::new(NumericExpression::Number(3)),
                Box::new(NumericExpression::Number(2)),
            );
        assert_eq!(expression.eval(&node, &problem), 2);
    }

    #[test]
    fn element_number_eval() {
        let node = generate_node();
        let expression = ElementExpression::Number(2);
        assert_eq!(expression.eval(&node.state), 2);
    }

    #[test]
    fn element_variable_eval() {
        let node = generate_node();
        let expression = ElementExpression::ElementVariable(0);
        assert_eq!(expression.eval(&node.state), 1);
    }

    #[test]
    fn set_variable_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = SetExpression::SetVariable(0);
        assert_eq!(
            expression.eval(&node.state, &problem),
            node.state.signature_variables.set_variables[0]
        );
        let expression = SetExpression::SetVariable(1);
        assert_eq!(
            expression.eval(&node.state, &problem),
            node.state.signature_variables.set_variables[1]
        );
    }

    #[test]
    fn permutation_variable_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = SetExpression::PermutationVariable(0);
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        set.insert(2);
        assert_eq!(expression.eval(&node.state, &problem), set);
    }

    #[test]
    fn union_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&node.state, &problem), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Union,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&node.state, &problem),
            node.state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn difference_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(2);
        assert_eq!(expression.eval(&node.state, &problem), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Difference,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&node.state, &problem),
            variable::SetVariable::with_capacity(3)
        );
    }

    #[test]
    fn intersect_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = SetExpression::SetOperation(
            SetOperator::Intersect,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(1)),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&node.state, &problem), set);
        let expression = SetExpression::SetOperation(
            SetOperator::Intersect,
            Box::new(SetExpression::SetVariable(0)),
            Box::new(SetExpression::SetVariable(0)),
        );
        assert_eq!(
            expression.eval(&node.state, &problem),
            node.state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_add_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Number(1),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        set.insert(1);
        set.insert(2);
        assert_eq!(expression.eval(&node.state, &problem), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Add,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Number(0),
        );
        assert_eq!(
            expression.eval(&node.state, &problem),
            node.state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn set_remove_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Number(2),
        );
        let mut set = variable::SetVariable::with_capacity(3);
        set.insert(0);
        assert_eq!(expression.eval(&node.state, &problem), set);
        let expression = SetExpression::SetElementOperation(
            SetElementOperator::Remove,
            Box::new(SetExpression::SetVariable(0)),
            ElementExpression::Number(1),
        );
        assert_eq!(
            expression.eval(&node.state, &problem),
            node.state.signature_variables.set_variables[0]
        );
    }

    #[test]
    fn cardinality_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let expression = NumericExpression::Cardinality(SetExpression::SetVariable(0));
        assert_eq!(expression.eval(&node, &problem), 2);
        let expression = NumericExpression::Cardinality(SetExpression::SetVariable(1));
        assert_eq!(expression.eval(&node, &problem), 2);
    }

    #[test]
    fn function_1d_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let f = NumericFunction1D(vec![10, 20, 30]);
        let expression = NumericExpression::Function1D(&f, ElementExpression::Number(0));
        assert_eq!(expression.eval(&node, &problem), 10);
        let expression = NumericExpression::Function1D(&f, ElementExpression::Number(1));
        assert_eq!(expression.eval(&node, &problem), 20);
        let expression = NumericExpression::Function1D(&f, ElementExpression::Number(2));
        assert_eq!(expression.eval(&node, &problem), 30);
    }

    #[test]
    fn function_1d_sum_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let f = NumericFunction1D(vec![10, 20, 30]);
        let expression = NumericExpression::Function1DSum(&f, SetExpression::SetVariable(0));
        assert_eq!(expression.eval(&node, &problem), 40);
        let expression = NumericExpression::Function1DSum(&f, SetExpression::SetVariable(1));
        assert_eq!(expression.eval(&node, &problem), 30);
    }

    #[test]
    fn function_2d_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let f = NumericFunction2D(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let expression = NumericExpression::Function2D(
            &f,
            ElementExpression::Number(0),
            ElementExpression::Number(1),
        );
        assert_eq!(expression.eval(&node, &problem), 20);
    }

    #[test]
    fn function_2d_sum_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let f = NumericFunction2D(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let expression = NumericExpression::Function2DSum(
            &f,
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&node, &problem), 180);
    }

    #[test]
    fn function_2d_sum_x_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let f = NumericFunction2D(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let expression = NumericExpression::Function2DSumX(
            &f,
            SetExpression::SetVariable(0),
            ElementExpression::Number(0),
        );
        assert_eq!(expression.eval(&node, &problem), 80);
    }

    #[test]
    fn function_2d_sum_y_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let f = NumericFunction2D(vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]]);
        let expression = NumericExpression::Function2DSumY(
            &f,
            ElementExpression::Number(0),
            SetExpression::SetVariable(0),
        );
        assert_eq!(expression.eval(&node, &problem), 40);
    }

    #[test]
    fn function_3d_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let f = NumericFunction3D(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = NumericExpression::Function3D(
            &f,
            ElementExpression::Number(0),
            ElementExpression::Number(1),
            ElementExpression::Number(2),
        );
        assert_eq!(expression.eval(&node, &problem), 60);
    }

    #[test]
    fn function_3d_sum_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let f = NumericFunction3D(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = NumericExpression::Function3DSum(
            &f,
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(1),
            SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&node, &problem), 240);
    }

    #[test]
    fn function_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let mut map =
            collections::HashMap::<Vec<variable::ElementVariable>, variable::IntegerVariable>::new(
            );
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let f = NumericFunction(map);
        let expression = NumericExpression::Function(
            &f,
            vec![
                ElementExpression::Number(0),
                ElementExpression::Number(1),
                ElementExpression::Number(0),
                ElementExpression::Number(0),
            ],
        );
        assert_eq!(expression.eval(&node, &problem), 100);
        let expression = NumericExpression::Function(
            &f,
            vec![
                ElementExpression::Number(0),
                ElementExpression::Number(1),
                ElementExpression::Number(0),
                ElementExpression::Number(1),
            ],
        );
        assert_eq!(expression.eval(&node, &problem), 200);
        let expression = NumericExpression::Function(
            &f,
            vec![
                ElementExpression::Number(0),
                ElementExpression::Number(1),
                ElementExpression::Number(2),
                ElementExpression::Number(0),
            ],
        );
        assert_eq!(expression.eval(&node, &problem), 300);
        let expression = NumericExpression::Function(
            &f,
            vec![
                ElementExpression::Number(0),
                ElementExpression::Number(1),
                ElementExpression::Number(2),
                ElementExpression::Number(1),
            ],
        );
        assert_eq!(expression.eval(&node, &problem), 400);
    }

    #[test]
    fn function_sum_eval() {
        let problem = generate_problem();
        let node = generate_node();
        let mut map =
            collections::HashMap::<Vec<variable::ElementVariable>, variable::IntegerVariable>::new(
            );
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let f = NumericFunction(map);
        let expression = NumericExpression::FunctionSum(
            &f,
            vec![
                ArgumentExpression::Element(ElementExpression::Number(0)),
                ArgumentExpression::Element(ElementExpression::Number(1)),
                ArgumentExpression::Set(SetExpression::SetVariable(0)),
                ArgumentExpression::Set(SetExpression::SetVariable(1)),
            ],
        );
        assert_eq!(expression.eval(&node, &problem), 1000);
    }
}
