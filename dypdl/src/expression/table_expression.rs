use super::element_expression::ElementExpression;
use crate::state::StateInterface;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable_type::Element;

/// Expression referring to a constant in a table.
#[derive(Debug, PartialEq, Clone)]
pub enum TableExpression<T: Clone> {
    /// Constant.
    Constant(T),
    /// Constant in a 1D table.
    Table1D(usize, ElementExpression),
    /// Constant in a 2D table.
    Table2D(usize, ElementExpression, ElementExpression),
    /// Constant in a 3D table.
    Table3D(
        usize,
        ElementExpression,
        ElementExpression,
        ElementExpression,
    ),
    /// Constant in a table.
    Table(usize, Vec<ElementExpression>),
}

impl<T: Clone> TableExpression<T> {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    pub fn eval<'a, U: StateInterface>(
        &'a self,
        state: &U,
        registry: &'a TableRegistry,
        tables: &'a TableData<T>,
    ) -> &'a T {
        match self {
            Self::Constant(value) => value,
            Self::Table1D(i, x) => tables.tables_1d[*i].get(x.eval(state, registry)),
            Self::Table2D(i, x, y) => {
                tables.tables_2d[*i].get(x.eval(state, registry), y.eval(state, registry))
            }
            Self::Table3D(i, x, y, z) => tables.tables_3d[*i].get(
                x.eval(state, registry),
                y.eval(state, registry),
                z.eval(state, registry),
            ),
            Self::Table(i, args) => {
                let args: Vec<Element> = args.iter().map(|x| x.eval(state, registry)).collect();
                tables.tables[*i].get(&args)
            }
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry, tables: &TableData<T>) -> TableExpression<T> {
        match self {
            Self::Table1D(i, x) => match x.simplify(registry) {
                ElementExpression::Constant(x) => {
                    Self::Constant(tables.tables_1d[*i].get(x).clone())
                }
                x => Self::Table1D(*i, x),
            },
            Self::Table2D(i, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (ElementExpression::Constant(x), ElementExpression::Constant(y)) => {
                    Self::Constant(tables.tables_2d[*i].get(x, y).clone())
                }
                (x, y) => Self::Table2D(*i, x, y),
            },
            Self::Table3D(i, x, y, z) => match (
                x.simplify(registry),
                y.simplify(registry),
                z.simplify(registry),
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
                    match arg.simplify(registry) {
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
    use crate::state::*;
    use crate::table::*;
    use crate::table_data::TableData;
    use crate::variable_type::*;
    use rustc_hash::FxHashMap;

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 1);

        let tables_1d = vec![Table1D::new(vec![1, 0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![Table2D::new(vec![vec![1, 0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![Table3D::new(vec![vec![vec![1, 0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 0, 0, 1];
        map.insert(key, 0);
        let tables = vec![Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        let element_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("t1"), 0);
        let vector_tables = TableData {
            tables_1d: vec![Table1D::new(vec![vec![0, 1]])],
            name_to_table_1d,
            ..Default::default()
        };

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = Set::with_capacity(3);
        let tables_1d = vec![Table1D::new(vec![set, default.clone(), default])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("s1"), 0);
        let set_tables = TableData {
            tables_1d,
            name_to_table_1d,
            ..Default::default()
        };

        TableRegistry {
            element_tables,
            set_tables,
            vector_tables,
            ..Default::default()
        }
    }

    fn generate_state() -> State {
        let mut set1 = Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                ..Default::default()
            },
            resource_variables: ResourceVariables {
                element_variables: vec![2],
                ..Default::default()
            },
        }
    }

    #[test]
    fn table_constant_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = TableExpression::Constant(1);
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
    }

    #[test]
    fn table_1d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_2d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_3d_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
        );
        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_eval() {
        let registry = generate_registry();
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
        assert_eq!(
            *expression.eval(&state, &registry, &registry.element_tables),
            1
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
            *expression.eval(&state, &registry, &registry.element_tables),
            0
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
            *expression.eval(&state, &registry, &registry.element_tables),
            0
        );
    }

    #[test]
    fn table_constant_simplify() {
        let registry = generate_registry();
        let expression = TableExpression::Constant(1);
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_1d_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table1D(0, ElementExpression::Constant(0));
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table1D(0, ElementExpression::Constant(1));
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table1D(0, ElementExpression::Variable(0));
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_2d_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table2D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_3d_simplify() {
        let registry = generate_registry();

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(1)
        );

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Constant(1),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            TableExpression::Constant(0)
        );

        let expression = TableExpression::Table3D(
            0,
            ElementExpression::Constant(0),
            ElementExpression::Constant(0),
            ElementExpression::Variable(0),
        );
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }

    #[test]
    fn table_simplify() {
        let registry = generate_registry();

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
            expression.simplify(&registry, &registry.element_tables),
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
            expression.simplify(&registry, &registry.element_tables),
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
            expression.simplify(&registry, &registry.element_tables),
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
        assert_eq!(
            expression.simplify(&registry, &registry.element_tables),
            expression
        );
    }
}
