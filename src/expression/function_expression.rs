use super::set_expression;
use crate::numeric_function;
use crate::problem;
use crate::state;
use crate::variable;
use std::iter;

#[derive(Debug)]
pub enum FunctionExpression<'a, T: variable::Numeric> {
    Function1D(
        &'a numeric_function::NumericFunction1D<T>,
        set_expression::ElementExpression,
    ),
    Function1DSum(
        &'a numeric_function::NumericFunction1D<T>,
        set_expression::SetExpression,
    ),
    Function2D(
        &'a numeric_function::NumericFunction2D<T>,
        set_expression::ElementExpression,
        set_expression::ElementExpression,
    ),
    Function2DSum(
        &'a numeric_function::NumericFunction2D<T>,
        set_expression::SetExpression,
        set_expression::SetExpression,
    ),
    Function2DSumX(
        &'a numeric_function::NumericFunction2D<T>,
        set_expression::SetExpression,
        set_expression::ElementExpression,
    ),
    Function2DSumY(
        &'a numeric_function::NumericFunction2D<T>,
        set_expression::ElementExpression,
        set_expression::SetExpression,
    ),
    Function3D(
        &'a numeric_function::NumericFunction3D<T>,
        set_expression::ElementExpression,
        set_expression::ElementExpression,
        set_expression::ElementExpression,
    ),
    Function3DSum(
        &'a numeric_function::NumericFunction3D<T>,
        set_expression::SetExpression,
        set_expression::SetExpression,
        set_expression::SetExpression,
    ),
    Function3DSumX(
        &'a numeric_function::NumericFunction3D<T>,
        set_expression::SetExpression,
        set_expression::ElementExpression,
        set_expression::ElementExpression,
    ),
    Function3DSumY(
        &'a numeric_function::NumericFunction3D<T>,
        set_expression::ElementExpression,
        set_expression::SetExpression,
        set_expression::ElementExpression,
    ),
    Function3DSumZ(
        &'a numeric_function::NumericFunction3D<T>,
        set_expression::ElementExpression,
        set_expression::ElementExpression,
        set_expression::SetExpression,
    ),
    Function3DSumXY(
        &'a numeric_function::NumericFunction3D<T>,
        set_expression::SetExpression,
        set_expression::SetExpression,
        set_expression::ElementExpression,
    ),
    Function3DSumXZ(
        &'a numeric_function::NumericFunction3D<T>,
        set_expression::SetExpression,
        set_expression::ElementExpression,
        set_expression::SetExpression,
    ),
    Function3DSumYZ(
        &'a numeric_function::NumericFunction3D<T>,
        set_expression::ElementExpression,
        set_expression::SetExpression,
        set_expression::SetExpression,
    ),
    Function(
        &'a numeric_function::NumericFunction<T>,
        Vec<set_expression::ElementExpression>,
    ),
    FunctionSum(
        &'a numeric_function::NumericFunction<T>,
        Vec<set_expression::ArgumentExpression>,
    ),
}

impl<'a, T: variable::Numeric> FunctionExpression<'a, T> {
    pub fn eval(&self, state: &state::State<T>, problem: &problem::Problem<T>) -> T {
        match self {
            Self::Function1D(f, x) => f.eval(x.eval(&state)),
            Self::Function1DSum(f, x) => f.sum(&x.eval(&state, problem)),
            Self::Function2D(f, x, y) => f.eval(x.eval(&state), y.eval(&state)),
            Self::Function2DSum(f, x, y) => {
                f.sum(&x.eval(&state, problem), &y.eval(&state, problem))
            }
            Self::Function2DSumX(f, x, y) => f.sum_x(&x.eval(&state, problem), y.eval(&state)),
            Self::Function2DSumY(f, x, y) => f.sum_y(x.eval(&state), &y.eval(&state, problem)),
            Self::Function3D(f, x, y, z) => f.eval(x.eval(&state), y.eval(&state), z.eval(&state)),
            Self::Function3DSum(f, x, y, z) => f.sum(
                &x.eval(&state, problem),
                &y.eval(&state, problem),
                &z.eval(&state, problem),
            ),
            Self::Function3DSumX(f, x, y, z) => {
                f.sum_x(&x.eval(&state, problem), y.eval(&state), z.eval(&state))
            }
            Self::Function3DSumY(f, x, y, z) => {
                f.sum_y(x.eval(&state), &y.eval(&state, problem), z.eval(&state))
            }
            Self::Function3DSumZ(f, x, y, z) => {
                f.sum_z(x.eval(&state), y.eval(&state), &z.eval(&state, problem))
            }
            Self::Function3DSumXY(f, x, y, z) => f.sum_xy(
                &x.eval(&state, problem),
                &y.eval(&state, problem),
                z.eval(&state),
            ),
            Self::Function3DSumXZ(f, x, y, z) => f.sum_xz(
                &x.eval(&state, problem),
                y.eval(&state),
                &z.eval(&state, problem),
            ),
            Self::Function3DSumYZ(f, x, y, z) => f.sum_yz(
                x.eval(&state),
                &y.eval(&state, problem),
                &z.eval(&state, problem),
            ),
            Self::Function(f, args) => eval_numeric_function(f, args, &state),
            Self::FunctionSum(f, args) => sum_numeric_function(f, args, &state, problem),
        }
    }
}

fn eval_numeric_function<T: variable::Numeric>(
    f: &numeric_function::NumericFunction<T>,
    args: &[set_expression::ElementExpression],
    state: &state::State<T>,
) -> T {
    let args: Vec<variable::ElementVariable> = args.iter().map(|x| x.eval(state)).collect();
    f.eval(&args)
}

fn sum_numeric_function<T: variable::Numeric>(
    f: &numeric_function::NumericFunction<T>,
    args: &[set_expression::ArgumentExpression],
    state: &state::State<T>,
    problem: &problem::Problem<T>,
) -> T {
    let mut result = vec![vec![]];
    for v in args {
        match v {
            set_expression::ArgumentExpression::Set(s) => {
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
            set_expression::ArgumentExpression::Element(e) => {
                for r in &mut result {
                    r.push(e.eval(state));
                }
            }
        }
    }
    result.into_iter().map(|x| f.eval(&x)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
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
            resource_variables: state::ResourceVariables {
                numeric_variables: vec![4, 5, 6],
            },
        }
    }

    #[test]
    fn function_1d_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction1D::new(vec![10, 20, 30]);
        let expression =
            FunctionExpression::Function1D(&f, set_expression::ElementExpression::Constant(0));
        assert_eq!(expression.eval(&state, &problem), 10);
        let expression =
            FunctionExpression::Function1D(&f, set_expression::ElementExpression::Constant(1));
        assert_eq!(expression.eval(&state, &problem), 20);
        let expression =
            FunctionExpression::Function1D(&f, set_expression::ElementExpression::Constant(2));
        assert_eq!(expression.eval(&state, &problem), 30);
    }

    #[test]
    fn function_1d_sum_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction1D::new(vec![10, 20, 30]);
        let expression =
            FunctionExpression::Function1DSum(&f, set_expression::SetExpression::SetVariable(0));
        assert_eq!(expression.eval(&state, &problem), 40);
        let expression =
            FunctionExpression::Function1DSum(&f, set_expression::SetExpression::SetVariable(1));
        assert_eq!(expression.eval(&state, &problem), 30);
    }

    #[test]
    fn function_2d_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
        ]);
        let expression = FunctionExpression::Function2D(
            &f,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(1),
        );
        assert_eq!(expression.eval(&state, &problem), 20);
    }

    #[test]
    fn function_2d_sum_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
        ]);
        let expression = FunctionExpression::Function2DSum(
            &f,
            set_expression::SetExpression::SetVariable(0),
            set_expression::SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &problem), 180);
    }

    #[test]
    fn function_2d_sum_x_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
        ]);
        let expression = FunctionExpression::Function2DSumX(
            &f,
            set_expression::SetExpression::SetVariable(0),
            set_expression::ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &problem), 80);
    }

    #[test]
    fn function_2d_sum_y_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
        ]);
        let expression = FunctionExpression::Function2DSumY(
            &f,
            set_expression::ElementExpression::Constant(0),
            set_expression::SetExpression::SetVariable(0),
        );
        assert_eq!(expression.eval(&state, &problem), 40);
    }

    #[test]
    fn function_3d_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = FunctionExpression::Function3D(
            &f,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(1),
            set_expression::ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &problem), 60);
    }

    #[test]
    fn function_3d_sum_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = FunctionExpression::Function3DSum(
            &f,
            set_expression::SetExpression::SetVariable(0),
            set_expression::SetExpression::SetVariable(1),
            set_expression::SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &problem), 240);
    }

    #[test]
    fn function_3d_sum_x_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = FunctionExpression::Function3DSumX(
            &f,
            set_expression::SetExpression::SetVariable(0),
            set_expression::ElementExpression::Constant(1),
            set_expression::ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &problem), 120);
    }

    #[test]
    fn function_3d_sum_y_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = FunctionExpression::Function3DSumY(
            &f,
            set_expression::ElementExpression::Constant(1),
            set_expression::SetExpression::SetVariable(0),
            set_expression::ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &problem), 120);
    }

    #[test]
    fn function_3d_sum_z_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = FunctionExpression::Function3DSumZ(
            &f,
            set_expression::ElementExpression::Constant(1),
            set_expression::ElementExpression::Constant(2),
            set_expression::SetExpression::SetVariable(0),
        );
        assert_eq!(expression.eval(&state, &problem), 160);
    }

    #[test]
    fn function_3d_sum_xy_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = FunctionExpression::Function3DSumXY(
            &f,
            set_expression::SetExpression::SetVariable(0),
            set_expression::SetExpression::SetVariable(1),
            set_expression::ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &problem), 180);
    }

    #[test]
    fn function_3d_sum_xz_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = FunctionExpression::Function3DSumXZ(
            &f,
            set_expression::SetExpression::SetVariable(0),
            set_expression::ElementExpression::Constant(2),
            set_expression::SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &problem), 300);
    }

    #[test]
    fn function_3d_sum_yz_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let f = numeric_function::NumericFunction3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ]);
        let expression = FunctionExpression::Function3DSumYZ(
            &f,
            set_expression::ElementExpression::Constant(2),
            set_expression::SetExpression::SetVariable(0),
            set_expression::SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &problem), 180);
    }

    #[test]
    fn function_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let mut map = HashMap::<Vec<variable::ElementVariable>, variable::IntegerVariable>::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let f = numeric_function::NumericFunction::new(map);
        let expression = FunctionExpression::Function(
            &f,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
            ],
        );
        assert_eq!(expression.eval(&state, &problem), 100);
        let expression = FunctionExpression::Function(
            &f,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
            ],
        );
        assert_eq!(expression.eval(&state, &problem), 200);
        let expression = FunctionExpression::Function(
            &f,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
                set_expression::ElementExpression::Constant(2),
                set_expression::ElementExpression::Constant(0),
            ],
        );
        assert_eq!(expression.eval(&state, &problem), 300);
        let expression = FunctionExpression::Function(
            &f,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
                set_expression::ElementExpression::Constant(2),
                set_expression::ElementExpression::Constant(1),
            ],
        );
        assert_eq!(expression.eval(&state, &problem), 400);
    }

    #[test]
    fn function_sum_eval() {
        let problem = generate_problem();
        let state = generate_state();
        let mut map = HashMap::<Vec<variable::ElementVariable>, variable::IntegerVariable>::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let f = numeric_function::NumericFunction::new(map);
        let expression = FunctionExpression::FunctionSum(
            &f,
            vec![
                set_expression::ArgumentExpression::Element(
                    set_expression::ElementExpression::Constant(0),
                ),
                set_expression::ArgumentExpression::Element(
                    set_expression::ElementExpression::Constant(1),
                ),
                set_expression::ArgumentExpression::Set(
                    set_expression::SetExpression::SetVariable(0),
                ),
                set_expression::ArgumentExpression::Set(
                    set_expression::SetExpression::SetVariable(1),
                ),
            ],
        );
        assert_eq!(expression.eval(&state, &problem), 1000);
    }
}
