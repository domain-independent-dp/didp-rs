use super::set_expression::{ArgumentExpression, ElementExpression, SetExpression};
use crate::state;
use crate::table;
use crate::table_registry::TableData;
use crate::variable;
use std::iter;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum NumericTableExpression<T: variable::Numeric> {
    Constant(T),
    Table1D(usize, ElementExpression),
    Table1DSum(usize, SetExpression),
    Table2D(usize, ElementExpression, ElementExpression),
    Table2DSum(usize, SetExpression, SetExpression),
    Table2DSumX(usize, SetExpression, ElementExpression),
    Table2DSumY(usize, ElementExpression, SetExpression),
    Table3D(
        usize,
        ElementExpression,
        ElementExpression,
        ElementExpression,
    ),
    Table3DSum(usize, SetExpression, SetExpression, SetExpression),
    Table3DSumX(usize, SetExpression, ElementExpression, ElementExpression),
    Table3DSumY(usize, ElementExpression, SetExpression, ElementExpression),
    Table3DSumZ(usize, ElementExpression, ElementExpression, SetExpression),
    Table3DSumXY(usize, SetExpression, SetExpression, ElementExpression),
    Table3DSumXZ(usize, SetExpression, ElementExpression, SetExpression),
    Table3DSumYZ(usize, ElementExpression, SetExpression, SetExpression),
    Table(usize, Vec<ElementExpression>),
    TableSum(usize, Vec<ArgumentExpression>),
}

impl<T: variable::Numeric> NumericTableExpression<T> {
    pub fn eval(
        &self,
        state: &state::State,
        metadata: &state::StateMetadata,
        tables: &TableData<T>,
    ) -> T {
        match self {
            Self::Constant(value) => *value,
            Self::Table1D(i, x) => tables.tables_1d[*i].eval(x.eval(state)),
            Self::Table1DSum(i, SetExpression::SetVariable(x)) => {
                tables.tables_1d[*i].sum(&state.signature_variables.set_variables[*x])
            }
            Self::Table1DSum(i, SetExpression::VectorVariable(x)) => {
                let x = &state.signature_variables.vector_variables[*x];
                x.iter().map(|x| tables.tables_1d[*i].eval(*x)).sum()
            }
            Self::Table1DSum(i, x) => tables.tables_1d[*i].sum(&x.eval(state, metadata)),
            Self::Table2D(i, x, y) => tables.tables_2d[*i].eval(x.eval(state), y.eval(state)),
            Self::Table2DSum(i, SetExpression::SetVariable(x), SetExpression::SetVariable(y)) => {
                let x = &state.signature_variables.set_variables[*x];
                let y = &state.signature_variables.set_variables[*y];
                tables.tables_2d[*i].sum(x, y)
            }
            Self::Table2DSum(
                i,
                SetExpression::VectorVariable(x),
                SetExpression::SetVariable(y),
            ) => {
                let x = &state.signature_variables.vector_variables[*x];
                let y = &state.signature_variables.set_variables[*y];
                x.iter().map(|x| tables.tables_2d[*i].sum_y(*x, y)).sum()
            }
            Self::Table2DSum(
                i,
                SetExpression::SetVariable(x),
                SetExpression::VectorVariable(y),
            ) => {
                let x = &state.signature_variables.set_variables[*x];
                let y = &state.signature_variables.vector_variables[*y];
                y.iter().map(|y| tables.tables_2d[*i].sum_x(x, *y)).sum()
            }
            Self::Table2DSum(
                i,
                SetExpression::VectorVariable(x),
                SetExpression::VectorVariable(y),
            ) => {
                let x = &state.signature_variables.vector_variables[*x];
                let y = &state.signature_variables.vector_variables[*y];
                y.iter()
                    .map(|y| x.iter().map(|x| tables.tables_2d[*i].eval(*x, *y)).sum())
                    .sum()
            }
            Self::Table2DSum(i, x, SetExpression::SetVariable(y)) => {
                let y = &state.signature_variables.set_variables[*y];
                tables.tables_2d[*i].sum(&x.eval(state, metadata), y)
            }
            Self::Table2DSum(i, SetExpression::SetVariable(x), y) => {
                let x = &state.signature_variables.set_variables[*x];
                tables.tables_2d[*i].sum(x, &y.eval(state, metadata))
            }
            Self::Table2DSum(i, x, SetExpression::VectorVariable(y)) => {
                let x = x.eval(state, metadata);
                let y = &state.signature_variables.vector_variables[*y];
                y.iter().map(|y| tables.tables_2d[*i].sum_x(&x, *y)).sum()
            }
            Self::Table2DSum(i, SetExpression::VectorVariable(x), y) => {
                let x = &state.signature_variables.vector_variables[*x];
                let y = y.eval(state, metadata);
                x.iter().map(|x| tables.tables_2d[*i].sum_y(*x, &y)).sum()
            }
            Self::Table2DSum(i, x, y) => {
                tables.tables_2d[*i].sum(&x.eval(&state, metadata), &y.eval(&state, metadata))
            }
            Self::Table2DSumX(i, SetExpression::SetVariable(x), y) => {
                let x = &state.signature_variables.set_variables[*x];
                tables.tables_2d[*i].sum_x(x, y.eval(state))
            }
            Self::Table2DSumX(i, SetExpression::VectorVariable(x), y) => {
                let x = &state.signature_variables.vector_variables[*x];
                let y = y.eval(state);
                x.iter().map(|x| tables.tables_2d[*i].eval(*x, y)).sum()
            }
            Self::Table2DSumX(i, x, y) => {
                tables.tables_2d[*i].sum_x(&x.eval(&state, metadata), y.eval(&state))
            }
            Self::Table2DSumY(i, x, SetExpression::SetVariable(y)) => {
                let y = &state.signature_variables.set_variables[*y];
                tables.tables_2d[*i].sum_y(x.eval(&state), y)
            }
            Self::Table2DSumY(i, x, SetExpression::VectorVariable(y)) => {
                let y = &state.signature_variables.vector_variables[*y];
                let x = x.eval(state);
                y.iter().map(|y| tables.tables_2d[*i].eval(x, *y)).sum()
            }
            Self::Table2DSumY(i, x, y) => {
                tables.tables_2d[*i].sum_y(x.eval(state), &y.eval(state, metadata))
            }
            Self::Table3D(i, x, y, z) => {
                tables.tables_3d[*i].eval(x.eval(state), y.eval(state), z.eval(state))
            }
            Self::Table3DSum(
                i,
                SetExpression::SetVariable(x),
                SetExpression::SetVariable(y),
                SetExpression::SetVariable(z),
            ) => tables.tables_3d[*i].sum(
                &state.signature_variables.set_variables[*x],
                &state.signature_variables.set_variables[*y],
                &state.signature_variables.set_variables[*z],
            ),
            Self::Table3DSum(i, x, y, z) => tables.tables_3d[*i].sum(
                &x.eval(&state, metadata),
                &y.eval(&state, metadata),
                &z.eval(&state, metadata),
            ),
            Self::Table3DSumX(i, SetExpression::SetVariable(x), y, z) => tables.tables_3d[*i]
                .sum_x(
                    &state.signature_variables.set_variables[*x],
                    y.eval(state),
                    z.eval(state),
                ),
            Self::Table3DSumX(i, SetExpression::VectorVariable(x), y, z) => {
                let x = &state.signature_variables.vector_variables[*x];
                let y = y.eval(state);
                let z = z.eval(state);
                x.iter().map(|x| tables.tables_3d[*i].eval(*x, y, z)).sum()
            }
            Self::Table3DSumX(i, x, y, z) => {
                tables.tables_3d[*i].sum_x(&x.eval(state, metadata), y.eval(state), z.eval(state))
            }
            Self::Table3DSumY(i, x, SetExpression::SetVariable(y), z) => tables.tables_3d[*i]
                .sum_y(
                    x.eval(state),
                    &state.signature_variables.set_variables[*y],
                    z.eval(state),
                ),
            Self::Table3DSumY(i, x, SetExpression::VectorVariable(y), z) => {
                let x = x.eval(state);
                let y = &state.signature_variables.vector_variables[*y];
                let z = z.eval(state);
                y.iter().map(|y| tables.tables_3d[*i].eval(x, *y, z)).sum()
            }
            Self::Table3DSumY(i, x, y, z) => {
                tables.tables_3d[*i].sum_y(x.eval(state), &y.eval(state, metadata), z.eval(state))
            }
            Self::Table3DSumZ(i, x, y, SetExpression::SetVariable(z)) => tables.tables_3d[*i]
                .sum_z(
                    x.eval(state),
                    y.eval(state),
                    &state.signature_variables.set_variables[*z],
                ),
            Self::Table3DSumZ(i, x, y, SetExpression::VectorVariable(z)) => {
                let x = x.eval(state);
                let y = y.eval(state);
                let z = &state.signature_variables.vector_variables[*z];
                z.iter().map(|z| tables.tables_3d[*i].eval(x, y, *z)).sum()
            }
            Self::Table3DSumZ(i, x, y, z) => {
                tables.tables_3d[*i].sum_z(x.eval(state), y.eval(state), &z.eval(state, metadata))
            }
            Self::Table3DSumXY(
                i,
                SetExpression::SetVariable(x),
                SetExpression::SetVariable(y),
                z,
            ) => tables.tables_3d[*i].sum_xy(
                &state.signature_variables.set_variables[*x],
                &state.signature_variables.set_variables[*y],
                z.eval(state),
            ),
            Self::Table3DSumXY(i, x, y, z) => tables.tables_3d[*i].sum_xy(
                &x.eval(state, metadata),
                &y.eval(state, metadata),
                z.eval(state),
            ),
            Self::Table3DSumXZ(
                i,
                SetExpression::SetVariable(x),
                y,
                SetExpression::SetVariable(z),
            ) => tables.tables_3d[*i].sum_xz(
                &state.signature_variables.set_variables[*x],
                y.eval(state),
                &state.signature_variables.set_variables[*z],
            ),
            Self::Table3DSumXZ(i, x, y, z) => tables.tables_3d[*i].sum_xz(
                &x.eval(state, metadata),
                y.eval(state),
                &z.eval(state, metadata),
            ),
            Self::Table3DSumYZ(
                i,
                x,
                SetExpression::SetVariable(y),
                SetExpression::SetVariable(z),
            ) => tables.tables_3d[*i].sum_yz(
                x.eval(state),
                &state.signature_variables.set_variables[*y],
                &state.signature_variables.set_variables[*z],
            ),
            Self::Table3DSumYZ(i, x, y, z) => tables.tables_3d[*i].sum_yz(
                x.eval(state),
                &y.eval(state, metadata),
                &z.eval(state, metadata),
            ),
            Self::Table(i, args) => {
                let args: Vec<variable::Element> = args.iter().map(|x| x.eval(state)).collect();
                tables.tables[*i].eval(&args)
            }
            Self::TableSum(i, args) => Self::sum_table(&tables.tables[*i], args, state, metadata),
        }
    }

    pub fn simplify(&self, tables: &TableData<T>) -> NumericTableExpression<T> {
        match self {
            Self::Table1D(i, ElementExpression::Constant(x)) => {
                Self::Constant(tables.tables_1d[*i].eval(*x))
            }
            Self::Table2D(i, ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                Self::Constant(tables.tables_2d[*i].eval(*x, *y))
            }
            Self::Table3D(
                i,
                ElementExpression::Constant(x),
                ElementExpression::Constant(y),
                ElementExpression::Constant(z),
            ) => Self::Constant(tables.tables_3d[*i].eval(*x, *y, *z)),
            Self::Table(i, args) => {
                let mut simplified_args = Vec::with_capacity(args.len());
                for arg in args {
                    match arg {
                        ElementExpression::Constant(arg) => {
                            simplified_args.push(*arg);
                        }
                        _ => return self.clone(),
                    }
                }
                Self::Constant(tables.tables[*i].eval(&simplified_args))
            }
            Self::TableSum(i, args) => {
                let mut simplified_args = Vec::with_capacity(args.len());
                for arg in args {
                    match arg {
                        ArgumentExpression::Element(ElementExpression::Constant(arg)) => {
                            simplified_args.push(*arg);
                        }
                        _ => return self.clone(),
                    }
                }
                Self::Constant(tables.tables[*i].eval(&simplified_args))
            }
            _ => self.clone(),
        }
    }

    fn sum_table(
        f: &table::Table<T>,
        args: &[ArgumentExpression],
        state: &state::State,
        metadata: &state::StateMetadata,
    ) -> T {
        let mut result = vec![vec![]];
        for v in args {
            match v {
                ArgumentExpression::Set(s) => {
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
                                .collect::<Vec<Vec<variable::Element>>>()
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
        result.into_iter().map(|x| f.eval(&x)).sum()
    }
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

        let vector_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_vector_variable = HashMap::new();
        name_to_vector_variable.insert("p0".to_string(), 0);
        name_to_vector_variable.insert("p1".to_string(), 1);
        name_to_vector_variable.insert("p2".to_string(), 2);
        name_to_vector_variable.insert("p3".to_string(), 3);
        let vector_variable_to_object = vec![0, 0, 0, 0];

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

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            vector_variable_names,
            name_to_vector_variable,
            vector_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            ..Default::default()
        }
    }

    fn generate_tables() -> TableData<variable::Integer> {
        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![table::Table1D::new(vec![10, 20, 30])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![10, 20, 30],
            vec![40, 50, 60],
            vec![70, 80, 90],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
            vec![vec![10, 20, 30], vec![40, 50, 60], vec![70, 80, 90]],
        ])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 1, 0, 0];
        map.insert(key, 100);
        let key = vec![0, 1, 0, 1];
        map.insert(key, 200);
        let key = vec![0, 1, 2, 0];
        map.insert(key, 300);
        let key = vec![0, 1, 2, 1];
        map.insert(key, 400);
        let tables = vec![table::Table::new(map, 0)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("f4"), 0);

        TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        }
    }

    fn generate_state() -> state::State {
        let mut set1 = variable::Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = variable::Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![
                    set1,
                    set2,
                    variable::Set::with_capacity(3),
                    variable::Set::with_capacity(3),
                ],
                vector_variables: vec![vec![0, 2], vec![0, 1], vec![], vec![]],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();
        let expression = NumericTableExpression::Constant(10);
        assert_eq!(expression.eval(&state, &metadata, &tables), 10);
    }

    #[test]
    fn table_1d_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();
        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(expression.eval(&state, &metadata, &tables), 10);
        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(expression.eval(&state, &metadata, &tables), 20);
        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(2));
        assert_eq!(expression.eval(&state, &metadata, &tables), 30);
    }

    #[test]
    fn table_1d_sum_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();
        let expression = NumericTableExpression::Table1DSum(0, SetExpression::SetVariable(0));
        assert_eq!(expression.eval(&state, &metadata, &tables), 40);
        let expression = NumericTableExpression::Table1DSum(0, SetExpression::SetVariable(1));
        assert_eq!(expression.eval(&state, &metadata, &tables), 30);
        let expression = NumericTableExpression::Table1DSum(0, SetExpression::VectorVariable(0));
        assert_eq!(expression.eval(&state, &metadata, &tables), 40);
        let expression = NumericTableExpression::Table1DSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::SetVariable(0))),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 20);
    }

    #[test]
    fn table_2d_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();
        let expression = NumericTableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 20);
    }

    #[test]
    fn table_2d_sum_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::VectorVariable(0),
            SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::SetVariable(0),
            SetExpression::VectorVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::VectorVariable(0),
            SetExpression::VectorVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::SetVariable(0),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(1),
            )))),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::VectorVariable(0),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(1),
            )))),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::VectorVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(1),
            )))),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);
    }

    #[test]
    fn table_2d_sum_x_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DSumX(
            0,
            SetExpression::SetVariable(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 80);

        let expression = NumericTableExpression::Table2DSumX(
            0,
            SetExpression::VectorVariable(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 80);

        let expression = NumericTableExpression::Table2DSumX(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 80);
    }

    #[test]
    fn table_2d_sum_y_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();

        let expression = NumericTableExpression::Table2DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 40);

        let expression = NumericTableExpression::Table2DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::VectorVariable(0),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 40);

        let expression = NumericTableExpression::Table2DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 40);
    }

    #[test]
    fn table_3d_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();
        let expression = NumericTableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 60);
    }

    #[test]
    fn table_3d_sum_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();
        let expression = NumericTableExpression::Table3DSum(
            0,
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(1),
            SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 240);

        let expression = NumericTableExpression::Table3DSum(
            0,
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(1),
            SetExpression::VectorVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 240);
    }

    #[test]
    fn table_3d_sum_x_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumX(
            0,
            SetExpression::SetVariable(0),
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 120);

        let expression = NumericTableExpression::Table3DSumX(
            0,
            SetExpression::VectorVariable(0),
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 120);

        let expression = NumericTableExpression::Table3DSumX(
            0,
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 120);
    }

    #[test]
    fn table_3d_sum_y_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumY(
            0,
            ElementExpression::Constant(1),
            SetExpression::SetVariable(0),
            ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 120);

        let expression = NumericTableExpression::Table3DSumY(
            0,
            ElementExpression::Constant(1),
            SetExpression::VectorVariable(0),
            ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 120);

        let expression = NumericTableExpression::Table3DSumY(
            0,
            ElementExpression::Constant(1),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
            ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 120);
    }

    #[test]
    fn table_3d_sum_z_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumZ(
            0,
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
            SetExpression::SetVariable(0),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 160);

        let expression = NumericTableExpression::Table3DSumZ(
            0,
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
            SetExpression::VectorVariable(0),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 160);

        let expression = NumericTableExpression::Table3DSumZ(
            0,
            ElementExpression::Constant(1),
            ElementExpression::Constant(2),
            SetExpression::Complement(Box::new(SetExpression::Complement(Box::new(
                SetExpression::SetVariable(0),
            )))),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 160);
    }

    #[test]
    fn table_3d_sum_xy_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumXY(
            0,
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table3DSumXY(
            0,
            SetExpression::SetVariable(0),
            SetExpression::VectorVariable(1),
            ElementExpression::Constant(2),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);
    }

    #[test]
    fn table_3d_sum_xz_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumXZ(
            0,
            SetExpression::SetVariable(0),
            ElementExpression::Constant(2),
            SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 300);

        let expression = NumericTableExpression::Table3DSumXZ(
            0,
            SetExpression::SetVariable(0),
            ElementExpression::Constant(2),
            SetExpression::VectorVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 300);
    }

    #[test]
    fn table_3d_sum_yz_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();

        let expression = NumericTableExpression::Table3DSumYZ(
            0,
            ElementExpression::Constant(2),
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);

        let expression = NumericTableExpression::Table3DSumYZ(
            0,
            ElementExpression::Constant(2),
            SetExpression::SetVariable(0),
            SetExpression::VectorVariable(1),
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 180);
    }

    #[test]
    fn table_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();
        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 100);
        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 200);
        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(2),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 300);
        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(2),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 400);
    }

    #[test]
    fn table_sum_eval() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let state = generate_state();
        let expression = NumericTableExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Set(SetExpression::SetVariable(0)),
                ArgumentExpression::Set(SetExpression::SetVariable(1)),
            ],
        );
        assert_eq!(expression.eval(&state, &metadata, &tables), 1000);
    }

    #[test]
    fn constant_simplify() {
        let tables = generate_tables();
        let expression = NumericTableExpression::Constant(0);
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Constant(0)
        ));
    }

    #[test]
    fn table_1d_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table1D(0, ElementExpression::Constant(0));
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Constant(10)
        ));

        let expression = NumericTableExpression::Table1D(0, ElementExpression::Variable(0));
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table1D(0, ElementExpression::Variable(0))
        ));
    }

    #[test]
    fn table_1d_sum_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table1DSum(0, SetExpression::SetVariable(0));
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table1DSum(0, SetExpression::SetVariable(0),)
        ));
    }

    #[test]
    fn table_2d_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Constant(10)
        ));

        let expression = NumericTableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table2D(
                0,
                ElementExpression::Constant(0),
                ElementExpression::Variable(0)
            )
        ));
    }

    #[test]
    fn table_2d_sum_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table2DSum(
            0,
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(1),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table2DSum(
                0,
                SetExpression::SetVariable(0),
                SetExpression::SetVariable(1),
            )
        ));
    }

    #[test]
    fn table_2d_sum_x_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table2DSumX(
            0,
            SetExpression::SetVariable(0),
            ElementExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table2DSumX(
                0,
                SetExpression::SetVariable(0),
                ElementExpression::Constant(0),
            )
        ));
    }

    #[test]
    fn table_2d_sum_y_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table2DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table2DSumY(
                0,
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
            )
        ));
    }

    #[test]
    fn table_3d_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Constant(10)
        ));

        let expression = NumericTableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table3D(
                0,
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Variable(0)
            )
        ));
    }

    #[test]
    fn table_3d_sum_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table3DSum(
            0,
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(1),
            SetExpression::SetVariable(2),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table3DSum(
                0,
                SetExpression::SetVariable(0),
                SetExpression::SetVariable(1),
                SetExpression::SetVariable(2),
            )
        ));
    }

    #[test]
    fn table_3d_sum_x_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table3DSumX(
            0,
            SetExpression::SetVariable(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table3DSumX(
                0,
                SetExpression::SetVariable(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )
        ));
    }

    #[test]
    fn table_3d_sum_y_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table3DSumY(
            0,
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
            ElementExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table3DSumY(
                0,
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
                ElementExpression::Constant(0),
            )
        ));
    }

    #[test]
    fn table_3d_sum_z_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table3DSumZ(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table3DSumZ(
                0,
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
            )
        ));
    }

    #[test]
    fn table_3d_sum_xy_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table3DSumXY(
            0,
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(0),
            ElementExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table3DSumXY(
                0,
                SetExpression::SetVariable(0),
                SetExpression::SetVariable(0),
                ElementExpression::Constant(0),
            )
        ));
    }

    #[test]
    fn table_3d_sum_xz_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table3DSumXZ(
            0,
            SetExpression::SetVariable(0),
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table3DSumXZ(
                0,
                SetExpression::SetVariable(0),
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
            )
        ));
    }

    #[test]
    fn table_3d_sum_yz_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table3DSumYZ(
            0,
            ElementExpression::Constant(0),
            SetExpression::SetVariable(0),
            SetExpression::SetVariable(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Table3DSumYZ(
                0,
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
                SetExpression::SetVariable(0),
            )
        ));
    }

    #[test]
    fn table_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Constant(100)
        ));

        let expression = NumericTableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Constant(0),
                ElementExpression::Variable(0),
            ],
        );
        let simplified = expression.simplify(&tables);
        assert!(matches!(simplified, NumericTableExpression::Table(0, _)));
        if let NumericTableExpression::Table(_, args) = simplified {
            assert_eq!(args.len(), 4);
            assert!(matches!(args[0], ElementExpression::Constant(0)));
            assert!(matches!(args[1], ElementExpression::Constant(1)));
            assert!(matches!(args[2], ElementExpression::Constant(0)));
            assert!(matches!(args[3], ElementExpression::Variable(0)));
        }
    }
    #[test]
    fn table_sum_simplify() {
        let tables = generate_tables();

        let expression = NumericTableExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
            ],
        );
        assert!(matches!(
            expression.simplify(&tables),
            NumericTableExpression::Constant(100)
        ));

        let expression = NumericTableExpression::TableSum(
            0,
            vec![
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Variable(0)),
            ],
        );
        let simplified = expression.simplify(&tables);
        assert!(matches!(simplified, NumericTableExpression::TableSum(0, _)));
        if let NumericTableExpression::TableSum(_, args) = simplified {
            assert_eq!(args.len(), 4);
            assert!(matches!(
                args[0],
                ArgumentExpression::Element(ElementExpression::Constant(0))
            ));
            assert!(matches!(
                args[1],
                ArgumentExpression::Element(ElementExpression::Constant(1))
            ));
            assert!(matches!(
                args[2],
                ArgumentExpression::Element(ElementExpression::Constant(0))
            ));
            assert!(matches!(
                args[3],
                ArgumentExpression::Element(ElementExpression::Variable(0))
            ));
        }
    }
}
