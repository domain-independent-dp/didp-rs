use super::set_expression;
use crate::numeric_function;
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
    pub fn eval(&self, state: &state::State<T>, metadata: &state::StateMetadata) -> T {
        match self {
            Self::Function1D(f, x) => f.eval(x.eval(&state)),
            Self::Function1DSum(f, x) => f.sum(&x.eval(&state, metadata)),
            Self::Function2D(f, x, y) => f.eval(x.eval(&state), y.eval(&state)),
            Self::Function2DSum(f, x, y) => {
                f.sum(&x.eval(&state, metadata), &y.eval(&state, metadata))
            }
            Self::Function2DSumX(f, x, y) => f.sum_x(&x.eval(&state, metadata), y.eval(&state)),
            Self::Function2DSumY(f, x, y) => f.sum_y(x.eval(&state), &y.eval(&state, metadata)),
            Self::Function3D(f, x, y, z) => f.eval(x.eval(&state), y.eval(&state), z.eval(&state)),
            Self::Function3DSum(f, x, y, z) => f.sum(
                &x.eval(&state, metadata),
                &y.eval(&state, metadata),
                &z.eval(&state, metadata),
            ),
            Self::Function3DSumX(f, x, y, z) => {
                f.sum_x(&x.eval(&state, metadata), y.eval(&state), z.eval(&state))
            }
            Self::Function3DSumY(f, x, y, z) => {
                f.sum_y(x.eval(&state), &y.eval(&state, metadata), z.eval(&state))
            }
            Self::Function3DSumZ(f, x, y, z) => {
                f.sum_z(x.eval(&state), y.eval(&state), &z.eval(&state, metadata))
            }
            Self::Function3DSumXY(f, x, y, z) => f.sum_xy(
                &x.eval(&state, metadata),
                &y.eval(&state, metadata),
                z.eval(&state),
            ),
            Self::Function3DSumXZ(f, x, y, z) => f.sum_xz(
                &x.eval(&state, metadata),
                y.eval(&state),
                &z.eval(&state, metadata),
            ),
            Self::Function3DSumYZ(f, x, y, z) => f.sum_yz(
                x.eval(&state),
                &y.eval(&state, metadata),
                &z.eval(&state, metadata),
            ),
            Self::Function(f, args) => eval_numeric_function(f, args, &state),
            Self::FunctionSum(f, args) => sum_numeric_function(f, args, &state, metadata),
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
    metadata: &state::StateMetadata,
) -> T {
    let mut result = vec![vec![]];
    for v in args {
        match v {
            set_expression::ArgumentExpression::Set(s) => {
                let s = s.eval(state, metadata);
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

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = HashMap::new();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            "s0".to_string(),
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
        ];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        name_to_set_variable.insert("s2".to_string(), 2);
        name_to_set_variable.insert("s3".to_string(), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let permutation_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_permutation_variable = HashMap::new();
        name_to_permutation_variable.insert("p0".to_string(), 0);
        name_to_permutation_variable.insert("p1".to_string(), 1);
        name_to_permutation_variable.insert("p2".to_string(), 2);
        name_to_permutation_variable.insert("p3".to_string(), 3);
        let permutation_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
        ];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        name_to_element_variable.insert("e2".to_string(), 2);
        name_to_element_variable.insert("e3".to_string(), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let numeric_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_numeric_variable = HashMap::new();
        name_to_numeric_variable.insert("n0".to_string(), 0);
        name_to_numeric_variable.insert("n1".to_string(), 1);
        name_to_numeric_variable.insert("n2".to_string(), 2);
        name_to_numeric_variable.insert("n3".to_string(), 3);

        let resource_variable_names = vec![
            "r0".to_string(),
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ];
        let mut name_to_resource_variable = HashMap::new();
        name_to_resource_variable.insert("r0".to_string(), 0);
        name_to_resource_variable.insert("r1".to_string(), 1);
        name_to_resource_variable.insert("r2".to_string(), 2);
        name_to_resource_variable.insert("r3".to_string(), 3);

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            permutation_variable_names,
            name_to_permutation_variable,
            permutation_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            numeric_variable_names,
            name_to_numeric_variable,
            resource_variable_names,
            name_to_resource_variable,
            less_is_better: vec![false, false, true, false],
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
    fn function_1d_eval() {
        let metadata = generate_metadata();
        let state = generate_state();
        let f = numeric_function::NumericFunction1D::new(vec![10, 20, 30]);
        let expression =
            FunctionExpression::Function1D(&f, set_expression::ElementExpression::Constant(0));
        assert_eq!(expression.eval(&state, &metadata), 10);
        let expression =
            FunctionExpression::Function1D(&f, set_expression::ElementExpression::Constant(1));
        assert_eq!(expression.eval(&state, &metadata), 20);
        let expression =
            FunctionExpression::Function1D(&f, set_expression::ElementExpression::Constant(2));
        assert_eq!(expression.eval(&state, &metadata), 30);
    }

    #[test]
    fn function_1d_sum_eval() {
        let metadata = generate_metadata();
        let state = generate_state();
        let f = numeric_function::NumericFunction1D::new(vec![10, 20, 30]);
        let expression =
            FunctionExpression::Function1DSum(&f, set_expression::SetExpression::SetVariable(0));
        assert_eq!(expression.eval(&state, &metadata), 40);
        let expression =
            FunctionExpression::Function1DSum(&f, set_expression::SetExpression::SetVariable(1));
        assert_eq!(expression.eval(&state, &metadata), 30);
    }

    #[test]
    fn function_2d_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 20);
    }

    #[test]
    fn function_2d_sum_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 180);
    }

    #[test]
    fn function_2d_sum_x_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 80);
    }

    #[test]
    fn function_2d_sum_y_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 40);
    }

    #[test]
    fn function_3d_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 60);
    }

    #[test]
    fn function_3d_sum_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 240);
    }

    #[test]
    fn function_3d_sum_x_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 120);
    }

    #[test]
    fn function_3d_sum_y_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 120);
    }

    #[test]
    fn function_3d_sum_z_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 160);
    }

    #[test]
    fn function_3d_sum_xy_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 180);
    }

    #[test]
    fn function_3d_sum_xz_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 300);
    }

    #[test]
    fn function_3d_sum_yz_eval() {
        let metadata = generate_metadata();
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
        assert_eq!(expression.eval(&state, &metadata), 180);
    }

    #[test]
    fn function_eval() {
        let metadata = generate_metadata();
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
        let f = numeric_function::NumericFunction::new(map, 0);
        let expression = FunctionExpression::Function(
            &f,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
            ],
        );
        assert_eq!(expression.eval(&state, &metadata), 100);
        let expression = FunctionExpression::Function(
            &f,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
            ],
        );
        assert_eq!(expression.eval(&state, &metadata), 200);
        let expression = FunctionExpression::Function(
            &f,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
                set_expression::ElementExpression::Constant(2),
                set_expression::ElementExpression::Constant(0),
            ],
        );
        assert_eq!(expression.eval(&state, &metadata), 300);
        let expression = FunctionExpression::Function(
            &f,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
                set_expression::ElementExpression::Constant(2),
                set_expression::ElementExpression::Constant(1),
            ],
        );
        assert_eq!(expression.eval(&state, &metadata), 400);
    }

    #[test]
    fn function_sum_eval() {
        let metadata = generate_metadata();
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
        let f = numeric_function::NumericFunction::new(map, 0);
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
        assert_eq!(expression.eval(&state, &metadata), 1000);
    }
}
