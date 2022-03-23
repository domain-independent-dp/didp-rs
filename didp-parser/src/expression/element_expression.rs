use crate::state;
use crate::table_data;
use crate::variable;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ElementExpression {
    Stage,
    Constant(variable::Element),
    Variable(usize),
    Table(Box<TableExpression<variable::Element>>),
}

impl ElementExpression {
    pub fn eval(
        &self,
        state: &state::State,
        tables: &table_data::TableData<variable::Element>,
    ) -> variable::Element {
        match self {
            Self::Stage => state.stage,
            Self::Constant(x) => *x,
            Self::Variable(i) => state.signature_variables.element_variables[*i],
            Self::Table(table) => table.eval(state, tables, tables),
        }
    }

    pub fn simplify(&self, tables: &table_data::TableData<variable::Element>) -> ElementExpression {
        match self {
            Self::Table(table) => Self::Table(Box::new(table.simplify(tables, tables))),
            _ => self.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TableExpression<T: Clone> {
    Constant(T),
    Table1D(usize, ElementExpression),
    Table2D(usize, ElementExpression, ElementExpression),
    Table3D(
        usize,
        ElementExpression,
        ElementExpression,
        ElementExpression,
    ),
    Table(usize, Vec<ElementExpression>),
}

impl<T: Clone> TableExpression<T> {
    pub fn eval(
        &self,
        state: &state::State,
        element_tables: &table_data::TableData<variable::Element>,
        tables: &table_data::TableData<T>,
    ) -> T {
        match self {
            Self::Constant(value) => value.clone(),
            Self::Table1D(i, x) => tables.tables_1d[*i]
                .get(x.eval(state, element_tables))
                .clone(),
            Self::Table2D(i, x, y) => tables.tables_2d[*i]
                .get(x.eval(state, element_tables), y.eval(state, element_tables))
                .clone(),
            Self::Table3D(i, x, y, z) => tables.tables_3d[*i]
                .get(
                    x.eval(state, element_tables),
                    y.eval(state, element_tables),
                    z.eval(state, element_tables),
                )
                .clone(),
            Self::Table(i, args) => {
                let args: Vec<variable::Element> =
                    args.iter().map(|x| x.eval(state, element_tables)).collect();
                tables.tables[*i].get(&args).clone()
            }
        }
    }

    pub fn simplify(
        &self,
        element_tables: &table_data::TableData<variable::Element>,
        tables: &table_data::TableData<T>,
    ) -> TableExpression<T> {
        match self {
            Self::Table1D(i, x) => match x.simplify(element_tables) {
                ElementExpression::Constant(x) => {
                    Self::Constant(tables.tables_1d[*i].get(x).clone())
                }
                x => Self::Table1D(*i, x),
            },
            Self::Table2D(i, x, y) => {
                match (x.simplify(element_tables), y.simplify(element_tables)) {
                    (ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                        Self::Constant(tables.tables_2d[*i].get(x, y).clone())
                    }
                    (x, y) => Self::Table2D(*i, x, y),
                }
            }
            Self::Table3D(i, x, y, z) => match (
                x.simplify(element_tables),
                y.simplify(element_tables),
                z.simplify(element_tables),
            ) {
                (
                    ElementExpression::Constant(x),
                    ElementExpression::Constant(y),
                    ElementExpression::Constant(z),
                ) => Self::Constant(tables.tables_3d[*i].get(x, y, z).clone()),
                (x, y, z) => Self::Table3D(*i, x, y, z),
            },
            Self::Table(i, args) => {
                let mut simplified_args = Vec::with_capacity(args.len());
                for arg in args {
                    match arg.simplify(element_tables) {
                        ElementExpression::Constant(arg) => simplified_args.push(arg),
                        _ => return self.clone(),
                    }
                }
                Self::Constant(tables.tables[*i].get(&simplified_args).clone())
            }
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table;
    use crate::variable;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_tables() -> table_data::TableData<variable::Element> {
        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("f0"), 1);

        let tables_1d = vec![table::Table1D::new(vec![1, 0])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![vec![1, 0]])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![vec![vec![1, 0]]])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 0, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 0, 0, 1];
        map.insert(key, 0);
        let tables = vec![table::Table::new(map, 0)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("f4"), 0);

        table_data::TableData {
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
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn element_number_eval() {
        let state = generate_state();
        let tables = generate_tables();
        let expression = ElementExpression::Constant(2);
        assert_eq!(expression.eval(&state, &tables), 2);
    }

    #[test]
    fn element_variable_eval() {
        let state = generate_state();
        let tables = generate_tables();
        let expression = ElementExpression::Variable(0);
        assert_eq!(expression.eval(&state, &tables), 1);
    }

    #[test]
    fn constant_eval() {
        let tables = generate_tables();
        let state = generate_state();

        let expression = TableExpression::Constant(1);
        assert_eq!(expression.eval(&state, &tables, &tables), 1);
    }

    #[test]
    fn table_1d_eval() {
        let tables = generate_tables();
        let state = generate_state();
        let expression = TableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(expression.eval(&state, &tables, &tables), 1);
        let expression = TableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(expression.eval(&state, &tables, &tables), 0);
    }

    #[test]
    fn table_2d_eval() {
        let tables = generate_tables();
        let state = generate_state();
        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &tables, &tables), 1);
        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(expression.eval(&state, &tables, &tables), 0);
    }

    #[test]
    fn table_3d_eval() {
        let tables = generate_tables();
        let state = generate_state();
        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &tables, &tables), 1);
        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(expression.eval(&state, &tables, &tables), 0);
    }

    #[test]
    fn table_eval() {
        let tables = generate_tables();
        let state = generate_state();
        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(expression.eval(&state, &tables, &tables), 1);
        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(expression.eval(&state, &tables, &tables), 0);
        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(2),
            ],
        );
        assert_eq!(expression.eval(&state, &tables, &tables), 0);
    }

    #[test]
    fn constant_simplify() {
        let tables = generate_tables();
        let expression = TableExpression::Constant(1);
        assert_eq!(expression.simplify(&tables, &tables), expression);
    }

    #[test]
    fn table_1d_simplify() {
        let tables = generate_tables();

        let expression = TableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            expression.simplify(&tables, &tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(
            expression.simplify(&tables, &tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table1D(0, ElementExpression::Variable(0));
        assert_eq!(expression.simplify(&tables, &tables), expression);
    }

    #[test]
    fn table_2d_simplify() {
        let tables = generate_tables();

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&tables, &tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&tables, &tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&tables, &tables), expression);
    }

    #[test]
    fn table_3d_simplify() {
        let tables = generate_tables();

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&tables, &tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&tables, &tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(expression.simplify(&tables, &tables), expression);
    }

    #[test]
    fn table_simplify() {
        let tables = generate_tables();

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            ],
        );
        assert_eq!(
            expression.simplify(&tables, &tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
            ],
        );
        assert_eq!(
            expression.simplify(&tables, &tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(2),
            ],
        );
        assert_eq!(
            expression.simplify(&tables, &tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table(
            0,
            vec![
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Variable(0),
            ],
        );
        assert_eq!(expression.simplify(&tables, &tables), expression);
    }
}
