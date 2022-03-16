use super::set_expression;
use crate::state;
use crate::table_registry;
use crate::variable;

#[derive(Debug)]
pub enum BoolTableExpression {
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
    pub fn eval<T: variable::Numeric>(
        &self,
        state: &state::State<T>,
        registry: &table_registry::TableRegistry<T>,
    ) -> bool {
        let tables = &registry.bool_tables;
        match self {
            BoolTableExpression::Table1D(i, x) => tables.tables_1d[*i].eval(x.eval(state)),
            BoolTableExpression::Table2D(i, x, y) => {
                tables.tables_2d[*i].eval(x.eval(state), y.eval(state))
            }
            BoolTableExpression::Table3D(i, x, y, z) => {
                tables.tables_3d[*i].eval(x.eval(state), y.eval(state), z.eval(state))
            }
            BoolTableExpression::Table(i, args) => {
                let args: Vec<variable::ElementVariable> =
                    args.iter().map(|x| x.eval(state)).collect();
                tables.tables[*i].eval(&args)
            }
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

    fn generate_registry() -> table_registry::TableRegistry<variable::IntegerVariable> {
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

        table_registry::TableRegistry {
            numeric_tables: table_registry::TableData {
                tables_1d: Vec::new(),
                name_to_table_1d: HashMap::new(),
                tables_2d: Vec::new(),
                name_to_table_2d: HashMap::new(),
                tables_3d: Vec::new(),
                name_to_table_3d: HashMap::new(),
                tables: Vec::new(),
                name_to_table: HashMap::new(),
            },
            bool_tables: table_registry::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                tables_3d,
                name_to_table_3d,
                tables,
                name_to_table,
            },
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
    fn table_1d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression =
            BoolTableExpression::Table1D(0, set_expression::ElementExpression::Constant(0));
        assert!(expression.eval(&state, &registry));
        let expression =
            BoolTableExpression::Table1D(0, set_expression::ElementExpression::Constant(1));
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn table_2d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = BoolTableExpression::Table2D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));
        let expression = BoolTableExpression::Table2D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn table_3d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = BoolTableExpression::Table3D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));
        let expression = BoolTableExpression::Table3D(
            0,
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(0),
            set_expression::ElementExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn table_eval() {
        let registry = generate_registry();
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
        assert!(expression.eval(&state, &registry));
        let expression = BoolTableExpression::Table(
            0,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(1),
            ],
        );
        assert!(!expression.eval(&state, &registry));
        let expression = BoolTableExpression::Table(
            0,
            vec![
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(0),
                set_expression::ElementExpression::Constant(2),
            ],
        );
        assert!(!expression.eval(&state, &registry));
    }
}
