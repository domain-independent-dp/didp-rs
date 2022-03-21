use super::set_expression;
use crate::state;
use crate::table_registry;
use crate::variable;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BoolTableExpression {
    Constant(bool),
    Table1D(usize, set_expression::ElementExpression),
    Table2D(
        usize,
        set_expression::ElementExpression,
        set_expression::ElementExpression,
    ),
    Table3D(
        usize,
        set_expression::ElementExpression,
        set_expression::ElementExpression,
        set_expression::ElementExpression,
    ),
    Table(usize, Vec<set_expression::ElementExpression>),
}

impl BoolTableExpression {
    pub fn eval(&self, state: &state::State, tables: &table_registry::TableData<bool>) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::Table1D(i, x) => tables.tables_1d[*i].eval(x.eval(state)),
            Self::Table2D(i, x, y) => tables.tables_2d[*i].eval(x.eval(state), y.eval(state)),
            Self::Table3D(i, x, y, z) => {
                tables.tables_3d[*i].eval(x.eval(state), y.eval(state), z.eval(state))
            }
            Self::Table(i, args) => {
                let args: Vec<variable::Element> = args.iter().map(|x| x.eval(state)).collect();
                tables.tables[*i].eval(&args)
            }
        }
    }

    pub fn simplify(&self, tables: &table_registry::TableData<bool>) -> BoolTableExpression {
        match self {
            Self::Table1D(i, set_expression::ElementExpression::Constant(x)) => {
                Self::Constant(tables.tables_1d[*i].eval(*x))
            }
            Self::Table2D(
                i,
                set_expression::ElementExpression::Constant(x),
                set_expression::ElementExpression::Constant(y),
            ) => Self::Constant(tables.tables_2d[*i].eval(*x, *y)),
            Self::Table3D(
                i,
                set_expression::ElementExpression::Constant(x),
                set_expression::ElementExpression::Constant(y),
                set_expression::ElementExpression::Constant(z),
            ) => Self::Constant(tables.tables_3d[*i].eval(*x, *y, *z)),
            Self::Table(i, args) => {
                let mut simplified_args = Vec::with_capacity(args.len());
                for arg in args {
                    match arg {
                        set_expression::ElementExpression::Constant(arg) => {
                            simplified_args.push(*arg)
                        }
                        _ => return self.clone(),
                    }
                }
                Self::Constant(tables.tables[*i].eval(&simplified_args))
            }
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table;
    use crate::table_registry;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_tables() -> table_registry::TableData<bool> {
        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("f0"), true);

        let tables_1d = vec![table::Table1D::new(vec![true, false])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![vec![true, false]])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![vec![vec![true, false]]])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 0, 0, 0];
        map.insert(key, true);
        let key = vec![0, 0, 0, 1];
        map.insert(key, false);
        let tables = vec![table::Table::new(map, false)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("f4"), 0);

        table_registry::TableData {
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
                set_variables: vec![set1, set2],
                permutation_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn constant_eval() {
        let tables = generate_tables();
        let state = generate_state();

        let expression = BoolTableExpression::Constant(true);
        assert!(expression.eval(&state, &tables));

        let expression = BoolTableExpression::Constant(false);
        assert!(!expression.eval(&state, &tables));
    }

    #[test]
    fn table_1d_eval() {
        let tables = generate_tables();
        let state = generate_state();
        let expression =
            BoolTableExpression::Table1D(0, set_expression::ElementExpression::Constant(0));
        assert!(expression.eval(&state, &tables));
        let expression =
            BoolTableExpression::Table1D(0, set_expression::ElementExpression::Constant(1));
        assert!(!expression.eval(&state, &tables));
    }

    #[test]
    fn table_2d_eval() {
        let tables = generate_tables();
        let state = generate_state();
        let expression = BoolTableExpression::Table2D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
        );
        assert!(expression.eval(&state, &tables));
        let expression = BoolTableExpression::Table2D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &tables));
    }

    #[test]
    fn table_3d_eval() {
        let tables = generate_tables();
        let state = generate_state();
        let expression = BoolTableExpression::Table3D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
        );
        assert!(expression.eval(&state, &tables));
        let expression = BoolTableExpression::Table3D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &tables));
    }

    #[test]
    fn table_eval() {
        let tables = generate_tables();
        let state = generate_state();
        let expression = BoolTableExpression::Table(
            0,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
            ],
        );
        assert!(expression.eval(&state, &tables));
        let expression = BoolTableExpression::Table(
            0,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
            ],
        );
        assert!(!expression.eval(&state, &tables));
        let expression = BoolTableExpression::Table(
            0,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(2),
            ],
        );
        assert!(!expression.eval(&state, &tables));
    }

    #[test]
    fn constant_simplify() {
        let tables = generate_tables();
        let expression = BoolTableExpression::Constant(true);
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(true)
        ));
    }

    #[test]
    fn table_1d_simplify() {
        let tables = generate_tables();

        let expression =
            BoolTableExpression::Table1D(0, set_expression::ElementExpression::Constant(0));
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(true)
        ));

        let expression =
            BoolTableExpression::Table1D(0, set_expression::ElementExpression::Constant(1));
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(false)
        ));

        let expression =
            BoolTableExpression::Table1D(0, set_expression::ElementExpression::Variable(0));
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Table1D(0, set_expression::ElementExpression::Variable(0))
        ));
    }

    #[test]
    fn table_2d_simplify() {
        let tables = generate_tables();

        let expression = BoolTableExpression::Table2D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(true)
        ));

        let expression = BoolTableExpression::Table2D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(false)
        ));

        let expression = BoolTableExpression::Table2D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Table2D(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Variable(0),
            )
        ));
    }

    #[test]
    fn table_3d_simplify() {
        let tables = generate_tables();

        let expression = BoolTableExpression::Table3D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(true)
        ));

        let expression = BoolTableExpression::Table3D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(false)
        ));

        let expression = BoolTableExpression::Table3D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Table3D(
                0,
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Variable(0),
            )
        ));
    }

    #[test]
    fn table_simplify() {
        let tables = generate_tables();

        let expression = BoolTableExpression::Table(
            0,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
            ],
        );
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(true)
        ));

        let expression = BoolTableExpression::Table(
            0,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
            ],
        );
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(false)
        ));

        let expression = BoolTableExpression::Table(
            0,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(2),
            ],
        );
        assert!(matches!(
            expression.simplify(&tables),
            BoolTableExpression::Constant(false)
        ));

        let expression = BoolTableExpression::Table(
            0,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Variable(0),
            ],
        );
        let simplified = expression.simplify(&tables);
        assert!(matches!(simplified, BoolTableExpression::Table(0, _)));
        if let BoolTableExpression::Table(_, args) = simplified {
            assert_eq!(args.len(), 4);
            assert!(matches!(
                args[0],
                set_expression::ElementExpression::Constant(0)
            ));
            assert!(matches!(
                args[1],
                set_expression::ElementExpression::Constant(0)
            ));
            assert!(matches!(
                args[2],
                set_expression::ElementExpression::Constant(0)
            ));
            assert!(matches!(
                args[3],
                set_expression::ElementExpression::Variable(0)
            ));
        }
    }
}
