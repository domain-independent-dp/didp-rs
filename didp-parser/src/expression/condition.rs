use super::bool_table_expression;
use super::numeric_expression::NumericExpression;
use super::set_condition;
use crate::state;
use crate::table_registry;
use crate::variable;

#[derive(Debug, Clone)]
pub enum Condition<T: variable::Numeric> {
    Constant(bool),
    Not(Box<Condition<T>>),
    And(Box<Condition<T>>, Box<Condition<T>>),
    Or(Box<Condition<T>>, Box<Condition<T>>),
    Comparison(
        ComparisonOperator,
        NumericExpression<T>,
        NumericExpression<T>,
    ),
    Set(set_condition::SetCondition),
    Table(bool_table_expression::BoolTableExpression),
}

#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}

impl<T: variable::Numeric> Condition<T> {
    pub fn eval(
        &self,
        state: &state::State<T>,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry<T>,
    ) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::Not(c) => !c.eval(state, metadata, registry),
            Self::And(x, y) => {
                x.eval(state, metadata, registry) && y.eval(state, metadata, registry)
            }
            Self::Or(x, y) => {
                x.eval(state, metadata, registry) || y.eval(state, metadata, registry)
            }
            Self::Comparison(op, x, y) => Self::eval_comparison(
                op,
                x.eval(state, metadata, registry),
                y.eval(state, metadata, registry),
            ),
            Self::Set(c) => c.eval(state, metadata),
            Self::Table(c) => c.eval(state, registry),
        }
    }

    pub fn simplify(&self, registry: &table_registry::TableRegistry<T>) -> Condition<T> {
        match self {
            Self::Not(c) => match c.simplify(registry) {
                Self::Constant(value) => Self::Constant(!value),
                c => Self::Not(Box::new(c)),
            },
            Self::And(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (x, Self::Constant(true)) => x,
                (Self::Constant(true), y) => y,
                (Self::Constant(false), _) | (_, Self::Constant(false)) => Self::Constant(false),
                (x, y) => Self::And(Box::new(x), Box::new(y)),
            },
            Self::Or(x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (x, Self::Constant(false)) => x,
                (Self::Constant(false), y) => y,
                (Self::Constant(true), _) | (_, Self::Constant(true)) => Self::Constant(true),
                (x, y) => Self::And(Box::new(x), Box::new(y)),
            },
            Self::Comparison(op, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (NumericExpression::Constant(x), NumericExpression::Constant(y)) => {
                    Self::Constant(Self::eval_comparison(op, x, y))
                }
                (NumericExpression::Variable(x), NumericExpression::Variable(y))
                | (
                    NumericExpression::ResourceVariable(x),
                    NumericExpression::ResourceVariable(y),
                ) if x == y => match op {
                    ComparisonOperator::Eq | ComparisonOperator::Ge | ComparisonOperator::Le => {
                        Self::Constant(true)
                    }
                    ComparisonOperator::Ne | ComparisonOperator::Gt | ComparisonOperator::Lt => {
                        Self::Constant(false)
                    }
                },
                (x, y) => Self::Comparison(op.clone(), x, y),
            },
            Self::Set(condition) => match condition.simplify() {
                set_condition::SetCondition::Constant(value) => Self::Constant(value),
                condition => Self::Set(condition),
            },
            Self::Table(condition) => match condition.simplify(registry) {
                bool_table_expression::BoolTableExpression::Constant(value) => {
                    Self::Constant(value)
                }
                condition => Self::Table(condition),
            },
            _ => self.clone(),
        }
    }

    fn eval_comparison(op: &ComparisonOperator, x: T, y: T) -> bool {
        match op {
            ComparisonOperator::Eq => x == y,
            ComparisonOperator::Ne => x != y,
            ComparisonOperator::Ge => x >= y,
            ComparisonOperator::Gt => x > y,
            ComparisonOperator::Le => x <= y,
            ComparisonOperator::Lt => x < y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::set_expression;
    use super::*;
    use crate::table;
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
    fn constant_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Constant(true);
        assert!(expression.eval(&state, &metadata, &registry));
        let expression = Condition::Constant(false);
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn eq_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn ne_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn ge_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn gt_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn le_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn lt_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn not_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        )));
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        )));
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn and_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(expression.eval(&state, &metadata, &registry));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(!expression.eval(&state, &metadata, &registry));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn or_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(expression.eval(&state, &metadata, &registry));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(expression.eval(&state, &metadata, &registry));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn table_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = Condition::Table(bool_table_expression::BoolTableExpression::Table1D(
            0,
            set_expression::ElementExpression::Constant(0),
        ));
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn set_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();
        let expression = Condition::Set(set_condition::SetCondition::Eq(
            set_expression::ElementExpression::Variable(0),
            set_expression::ElementExpression::Constant(1),
        ));
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn constant_simplify() {
        let registry = generate_registry();

        let expression = Condition::Constant(true);
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Constant(false);
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));
    }

    #[test]
    fn eq_simplify() {
        let registry = generate_registry();

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Variable(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Eq,
                NumericExpression::Constant(0),
                NumericExpression::Variable(0)
            )
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Variable(0),
            NumericExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Eq,
                NumericExpression::Variable(0),
                NumericExpression::Variable(1)
            )
        ));
    }

    #[test]
    fn ne_simplify() {
        let registry = generate_registry();

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Variable(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Ne,
                NumericExpression::Constant(0),
                NumericExpression::Variable(0)
            )
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Ne,
            NumericExpression::Variable(0),
            NumericExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Ne,
                NumericExpression::Variable(0),
                NumericExpression::Variable(1)
            )
        ));
    }

    #[test]
    fn ge_simplify() {
        let registry = generate_registry();

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Variable(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Ge,
                NumericExpression::Constant(0),
                NumericExpression::Variable(0)
            )
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Ge,
            NumericExpression::Variable(0),
            NumericExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Ge,
                NumericExpression::Variable(0),
                NumericExpression::Variable(1)
            )
        ));
    }

    #[test]
    fn gt_simplify() {
        let registry = generate_registry();

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Variable(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Gt,
                NumericExpression::Constant(0),
                NumericExpression::Variable(0)
            )
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Gt,
            NumericExpression::Variable(0),
            NumericExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Gt,
                NumericExpression::Variable(0),
                NumericExpression::Variable(1)
            )
        ));
    }

    #[test]
    fn le_simplify() {
        let registry = generate_registry();

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Variable(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Le,
                NumericExpression::Constant(0),
                NumericExpression::Variable(0)
            )
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Le,
            NumericExpression::Variable(0),
            NumericExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Le,
                NumericExpression::Variable(0),
                NumericExpression::Variable(1)
            )
        ));
    }

    #[test]
    fn lt_simplify() {
        let registry = generate_registry();

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Variable(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Variable(0),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Lt,
                NumericExpression::Constant(0),
                NumericExpression::Variable(0)
            )
        ));

        let expression = Condition::Comparison(
            ComparisonOperator::Lt,
            NumericExpression::Variable(0),
            NumericExpression::Variable(1),
        );
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Lt,
                NumericExpression::Variable(0),
                NumericExpression::Variable(1)
            )
        ));
    }

    #[test]
    fn not_simplify() {
        let registry = generate_registry();

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        )));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        )));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Not(Box::new(Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Variable(0),
        )));
        let simplified = expression.simplify(&registry);
        assert!(matches!(simplified, Condition::Not(_)));
        if let Condition::Not(simplified) = simplified {
            assert!(matches!(
                *simplified,
                Condition::Comparison(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::Variable(0),
                )
            ));
        }
    }

    #[test]
    fn and_simplify() {
        let registry = generate_registry();

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Variable(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Eq,
                NumericExpression::Constant(0),
                NumericExpression::Variable(1),
            )
        ));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Variable(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Variable(0),
        );
        let expression = Condition::And(Box::new(x), Box::new(y));
        let simplified = expression.simplify(&registry);
        assert!(matches!(simplified, Condition::And(_, _)));
        if let Condition::And(x, y) = simplified {
            assert!(matches!(
                *x,
                Condition::Comparison(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::Variable(1),
                )
            ));
            assert!(matches!(
                *y,
                Condition::Comparison(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::Variable(0),
                )
            ));
        }
    }

    #[test]
    fn or_simplify() {
        let registry = generate_registry();

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(false)
        ));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Variable(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Comparison(
                ComparisonOperator::Eq,
                NumericExpression::Constant(0),
                NumericExpression::Variable(1),
            )
        ));

        let x = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Variable(1),
        );
        let y = Condition::Comparison(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Variable(0),
        );
        let expression = Condition::Or(Box::new(x), Box::new(y));
        let simplified = expression.simplify(&registry);
        assert!(matches!(simplified, Condition::And(_, _)));
        if let Condition::Or(x, y) = simplified {
            assert!(matches!(
                *x,
                Condition::Comparison(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::Variable(1),
                )
            ));
            assert!(matches!(
                *y,
                Condition::Comparison(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::Variable(0),
                )
            ));
        }
    }

    #[test]
    fn table_simplify() {
        let registry = generate_registry();

        let expression = Condition::Table(bool_table_expression::BoolTableExpression::Table1D(
            0,
            set_expression::ElementExpression::Constant(0),
        ));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Constant(true)
        ));

        let expression = Condition::Table(bool_table_expression::BoolTableExpression::Table1D(
            0,
            set_expression::ElementExpression::Variable(0),
        ));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Table(bool_table_expression::BoolTableExpression::Table1D(
                0,
                set_expression::ElementExpression::Variable(0),
            ))
        ));
    }

    #[test]
    fn set_simplify() {
        let registry = generate_registry();
        let expression = Condition::Set(set_condition::SetCondition::Eq(
            set_expression::ElementExpression::Variable(0),
            set_expression::ElementExpression::Constant(1),
        ));
        assert!(matches!(
            expression.simplify(&registry),
            Condition::Set(set_condition::SetCondition::Eq(
                set_expression::ElementExpression::Variable(0),
                set_expression::ElementExpression::Constant(1),
            ))
        ));
    }
}
