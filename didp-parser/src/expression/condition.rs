use super::bool_table_expression;
use super::numeric_expression::NumericExpression;
use super::set_condition;
use crate::state;
use crate::table_registry;
use crate::variable;
use num_traits::FromPrimitive;

#[derive(Debug, PartialEq, Clone)]
pub enum Condition {
    Constant(bool),
    Not(Box<Condition>),
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    Comparison(Box<Comparison>),
    Set(set_condition::SetCondition),
    Table(bool_table_expression::BoolTableExpression),
}

impl Default for Condition {
    fn default() -> Condition {
        Self::Constant(false)
    }
}

impl Condition {
    pub fn eval(
        &self,
        state: &state::State,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
    ) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::Not(condition) => !condition.eval(state, metadata, registry),
            Self::And(x, y) => {
                x.eval(state, metadata, registry) && y.eval(state, metadata, registry)
            }
            Self::Or(x, y) => {
                x.eval(state, metadata, registry) || y.eval(state, metadata, registry)
            }
            Self::Comparison(condition) => condition.eval(state, metadata, registry),
            Self::Set(condition) => condition.eval(state, metadata),
            Self::Table(condition) => condition.eval(state, &registry.bool_tables),
        }
    }

    pub fn simplify(&self, registry: &table_registry::TableRegistry) -> Condition {
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
            Self::Comparison(condition) => match condition.simplify(registry) {
                Comparison::Constant(value) => Self::Constant(value),
                condition => Self::Comparison(Box::new(condition)),
            },
            Self::Set(condition) => match condition.simplify() {
                set_condition::SetCondition::Constant(value) => Self::Constant(value),
                condition => Self::Set(condition),
            },
            Self::Table(condition) => match condition.simplify(&registry.bool_tables) {
                bool_table_expression::BoolTableExpression::Constant(value) => {
                    Self::Constant(value)
                }
                condition => Self::Table(condition),
            },
            _ => self.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Comparison {
    Constant(bool),
    ComparisonII(
        ComparisonOperator,
        NumericExpression<variable::Integer>,
        NumericExpression<variable::Integer>,
    ),
    ComparisonCC(
        ComparisonOperator,
        NumericExpression<variable::Continuous>,
        NumericExpression<variable::Continuous>,
    ),
    ComparisonIC(
        ComparisonOperator,
        NumericExpression<variable::Integer>,
        NumericExpression<variable::Continuous>,
    ),
    ComparisonCI(
        ComparisonOperator,
        NumericExpression<variable::Continuous>,
        NumericExpression<variable::Integer>,
    ),
}

impl Default for Comparison {
    fn default() -> Comparison {
        Self::Constant(false)
    }
}

impl Comparison {
    pub fn eval(
        &self,
        state: &state::State,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
    ) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::ComparisonII(op, x, y) => Self::eval_comparison(
                op,
                x.eval(state, metadata, registry),
                y.eval(state, metadata, registry),
            ),
            Self::ComparisonIC(op, x, y) => Self::eval_comparison(
                op,
                variable::Continuous::from_i32(x.eval(state, metadata, registry)).unwrap(),
                y.eval(state, metadata, registry),
            ),
            Self::ComparisonCI(op, x, y) => Self::eval_comparison(
                op,
                x.eval(state, metadata, registry),
                variable::Continuous::from_i32(y.eval(state, metadata, registry)).unwrap(),
            ),
            Self::ComparisonCC(op, x, y) => Self::eval_comparison(
                op,
                x.eval(state, metadata, registry),
                y.eval(state, metadata, registry),
            ),
        }
    }

    pub fn simplify(&self, registry: &table_registry::TableRegistry) -> Comparison {
        match self {
            Self::Constant(value) => Self::Constant(*value),
            Self::ComparisonII(op, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (NumericExpression::Constant(x), NumericExpression::Constant(y)) => {
                    Self::Constant(Self::eval_comparison(op, x, y))
                }
                (NumericExpression::IntegerVariable(x), NumericExpression::IntegerVariable(y))
                | (
                    NumericExpression::IntegerResourceVariable(x),
                    NumericExpression::IntegerResourceVariable(y),
                ) if x == y => match op {
                    ComparisonOperator::Eq | ComparisonOperator::Ge | ComparisonOperator::Le => {
                        Self::Constant(true)
                    }
                    ComparisonOperator::Ne | ComparisonOperator::Gt | ComparisonOperator::Lt => {
                        Self::Constant(false)
                    }
                },
                (x, y) => Self::ComparisonII(op.clone(), x, y),
            },
            Self::ComparisonIC(op, x, y) => {
                Self::ComparisonIC(op.clone(), x.simplify(registry), y.simplify(registry))
            }
            Self::ComparisonCI(op, x, y) => {
                Self::ComparisonCI(op.clone(), x.simplify(registry), y.simplify(registry))
            }
            Self::ComparisonCC(op, x, y) => match (x.simplify(registry), y.simplify(registry)) {
                (NumericExpression::Constant(x), NumericExpression::Constant(y)) => {
                    Self::Constant(Self::eval_comparison(op, x, y))
                }
                (NumericExpression::IntegerVariable(x), NumericExpression::IntegerVariable(y))
                | (
                    NumericExpression::ContinuousVariable(x),
                    NumericExpression::ContinuousVariable(y),
                )
                | (
                    NumericExpression::IntegerResourceVariable(x),
                    NumericExpression::IntegerResourceVariable(y),
                )
                | (
                    NumericExpression::ContinuousResourceVariable(x),
                    NumericExpression::ContinuousResourceVariable(y),
                ) if x == y => match op {
                    ComparisonOperator::Eq | ComparisonOperator::Ge | ComparisonOperator::Le => {
                        Self::Constant(true)
                    }
                    ComparisonOperator::Ne | ComparisonOperator::Gt | ComparisonOperator::Lt => {
                        Self::Constant(false)
                    }
                },
                (x, y) => Self::ComparisonCC(op.clone(), x, y),
            },
        }
    }

    fn eval_comparison<T: variable::Numeric>(op: &ComparisonOperator, x: T, y: T) -> bool {
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ComparisonOperator {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}

#[cfg(test)]
mod tests {
    use super::super::set_expression;
    use super::*;
    use crate::table;
    use ordered_float::OrderedFloat;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = HashMap::new();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec!["s0".to_string(), "s1".to_string()];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        let set_variable_to_object = vec![0, 0];

        let permutation_variable_names = vec!["p0".to_string()];
        let mut name_to_permutation_variable = HashMap::new();
        name_to_permutation_variable.insert("p0".to_string(), 0);
        let permutation_variable_to_object = vec![0];

        let element_variable_names = vec!["e0".to_string()];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert("e0".to_string(), 0);
        let element_variable_to_object = vec![0];

        let integer_variable_names = vec!["i0".to_string(), "i1".to_string(), "i2".to_string()];
        let mut name_to_integer_variable = HashMap::new();
        name_to_integer_variable.insert("i0".to_string(), 0);
        name_to_integer_variable.insert("i1".to_string(), 1);
        name_to_integer_variable.insert("i2".to_string(), 2);

        let continuous_variable_names = vec!["c0".to_string(), "c1".to_string(), "c2".to_string()];
        let mut name_to_continuous_variable = HashMap::new();
        name_to_continuous_variable.insert("c0".to_string(), 0);
        name_to_continuous_variable.insert("c1".to_string(), 1);
        name_to_continuous_variable.insert("c2".to_string(), 2);

        let integer_resource_variable_names =
            vec!["ir0".to_string(), "ir1".to_string(), "ir2".to_string()];
        let mut name_to_integer_resource_variable = HashMap::new();
        name_to_integer_resource_variable.insert("ir0".to_string(), 0);
        name_to_integer_resource_variable.insert("ir1".to_string(), 1);
        name_to_integer_resource_variable.insert("ir2".to_string(), 2);

        let continuous_resource_variable_names =
            vec!["cr0".to_string(), "cr1".to_string(), "cr2".to_string()];
        let mut name_to_continuous_resource_variable = HashMap::new();
        name_to_continuous_resource_variable.insert("cr0".to_string(), 0);
        name_to_continuous_resource_variable.insert("cr1".to_string(), 1);
        name_to_continuous_resource_variable.insert("cr2".to_string(), 2);

        state::StateMetadata {
            maximize: false,
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
            integer_variable_names,
            name_to_integer_variable,
            continuous_variable_names,
            name_to_continuous_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true, false],
            continuous_resource_variable_names,
            name_to_continuous_resource_variable,
            continuous_less_is_better: vec![false, false, true, false],
        }
    }

    fn generate_registry() -> table_registry::TableRegistry {
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

        table_registry::TableRegistry {
            bool_tables: table_registry::TableData {
                name_to_constant,
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                tables_3d,
                name_to_table_3d,
                tables,
                name_to_table,
            },
            ..Default::default()
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
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)],
            }),
            resource_variables: state::ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
            stage: 0,
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

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
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

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
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

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
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

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
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

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
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

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Comparison::ComparisonII(
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

        let expression = Condition::Not(Box::new(Condition::Constant(true)));
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::Not(Box::new(Condition::Constant(false)));
        assert!(expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn and_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(false)),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert!(!expression.eval(&state, &metadata, &registry));

        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
        assert!(!expression.eval(&state, &metadata, &registry));
    }

    #[test]
    fn or_eval() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(false)),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert!(expression.eval(&state, &metadata, &registry));

        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
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
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));

        let expression = Condition::Constant(false);
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));
    }

    #[test]
    fn eq_simplify() {
        let registry = generate_registry();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Eq,
                NumericExpression::Constant(0),
                NumericExpression::IntegerVariable(0)
            )
        );

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Eq,
                NumericExpression::IntegerVariable(0),
                NumericExpression::IntegerVariable(1)
            )
        );
    }

    #[test]
    fn ne_simplify() {
        let registry = generate_registry();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Ne,
                NumericExpression::Constant(0),
                NumericExpression::IntegerVariable(0)
            )
        );

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Ne,
                NumericExpression::IntegerVariable(0),
                NumericExpression::IntegerVariable(1)
            )
        );
    }

    #[test]
    fn ge_simplify() {
        let registry = generate_registry();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Ge,
                NumericExpression::Constant(0),
                NumericExpression::IntegerVariable(0)
            )
        );

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Ge,
                NumericExpression::IntegerVariable(0),
                NumericExpression::IntegerVariable(1)
            )
        );
    }

    #[test]
    fn gt_simplify() {
        let registry = generate_registry();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Gt,
                NumericExpression::Constant(0),
                NumericExpression::IntegerVariable(0)
            )
        );

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Gt,
                NumericExpression::IntegerVariable(0),
                NumericExpression::IntegerVariable(1)
            )
        );
    }

    #[test]
    fn le_simplify() {
        let registry = generate_registry();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Le,
                NumericExpression::Constant(0),
                NumericExpression::IntegerVariable(0)
            )
        );

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Le,
                NumericExpression::IntegerVariable(0),
                NumericExpression::IntegerVariable(1)
            )
        );
    }

    #[test]
    fn lt_simplify() {
        let registry = generate_registry();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Lt,
                NumericExpression::Constant(0),
                NumericExpression::IntegerVariable(0)
            )
        );

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(
            expression.simplify(&registry),
            Comparison::ComparisonII(
                ComparisonOperator::Lt,
                NumericExpression::IntegerVariable(0),
                NumericExpression::IntegerVariable(1)
            )
        );
    }

    #[test]
    fn not_simplify() {
        let registry = generate_registry();

        let expression = Condition::Not(Box::new(Condition::Comparison(Box::new(
            Comparison::ComparisonII(
                ComparisonOperator::Eq,
                NumericExpression::Constant(0),
                NumericExpression::Constant(0),
            ),
        ))));
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));

        let expression = Condition::Not(Box::new(Condition::Comparison(Box::new(
            Comparison::ComparisonII(
                ComparisonOperator::Eq,
                NumericExpression::Constant(0),
                NumericExpression::Constant(1),
            ),
        ))));
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));

        let expression = Condition::Not(Box::new(Condition::Comparison(Box::new(
            Comparison::ComparisonII(
                ComparisonOperator::Eq,
                NumericExpression::Constant(0),
                NumericExpression::IntegerVariable(0),
            ),
        ))));
        assert_eq!(
            expression.simplify(&registry),
            Condition::Not(Box::new(Condition::Comparison(Box::new(
                Comparison::ComparisonII(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::IntegerVariable(0),
                )
            ))))
        );
    }

    #[test]
    fn and_simplify() {
        let registry = generate_registry();

        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));

        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));

        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));

        let x = Condition::Comparison(Box::new(Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(1),
        )));
        let expression = Condition::And(Box::new(x), Box::new(Condition::Constant(true)));
        assert_eq!(
            expression.simplify(&registry),
            Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Eq,
                NumericExpression::Constant(0),
                NumericExpression::IntegerVariable(1),
            )))
        );

        let x = Condition::Comparison(Box::new(Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(1),
        )));
        let y = Condition::Comparison(Box::new(Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        )));
        let expression = Condition::And(Box::new(x), Box::new(y));
        assert_eq!(
            expression.simplify(&registry),
            Condition::And(
                Box::new(Condition::Comparison(Box::new(Comparison::ComparisonII(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::IntegerVariable(1),
                )))),
                Box::new(Condition::Comparison(Box::new(Comparison::ComparisonII(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::IntegerVariable(0),
                ))))
            )
        );
    }

    #[test]
    fn or_simplify() {
        let registry = generate_registry();

        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));

        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(false)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));

        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));

        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
        assert_eq!(expression.simplify(&registry), Condition::Constant(false));

        let x = Condition::Comparison(Box::new(Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(1),
        )));
        let expression = Condition::Or(Box::new(x), Box::new(Condition::Constant(false)));
        assert_eq!(
            expression.simplify(&registry),
            Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Eq,
                NumericExpression::Constant(0),
                NumericExpression::IntegerVariable(1),
            )))
        );

        let x = Condition::Comparison(Box::new(Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(1),
        )));
        let y = Condition::Comparison(Box::new(Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        )));
        let expression = Condition::Or(Box::new(x), Box::new(y));
        assert_eq!(
            expression.simplify(&registry),
            Condition::And(
                Box::new(Condition::Comparison(Box::new(Comparison::ComparisonII(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::IntegerVariable(1),
                )))),
                Box::new(Condition::Comparison(Box::new(Comparison::ComparisonII(
                    ComparisonOperator::Eq,
                    NumericExpression::Constant(0),
                    NumericExpression::IntegerVariable(0),
                ))))
            )
        );
    }

    #[test]
    fn table_simplify() {
        let registry = generate_registry();

        let expression = Condition::Table(bool_table_expression::BoolTableExpression::Table1D(
            0,
            set_expression::ElementExpression::Constant(0),
        ));
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));

        let expression = Condition::Table(bool_table_expression::BoolTableExpression::Table1D(
            0,
            set_expression::ElementExpression::Variable(0),
        ));
        assert_eq!(
            expression.simplify(&registry),
            Condition::Table(bool_table_expression::BoolTableExpression::Table1D(
                0,
                set_expression::ElementExpression::Variable(0),
            ))
        );
    }

    #[test]
    fn set_simplify() {
        let registry = generate_registry();
        let expression = Condition::Set(set_condition::SetCondition::Eq(
            set_expression::ElementExpression::Variable(0),
            set_expression::ElementExpression::Constant(1),
        ));
        assert_eq!(
            expression.simplify(&registry),
            Condition::Set(set_condition::SetCondition::Eq(
                set_expression::ElementExpression::Variable(0),
                set_expression::ElementExpression::Constant(1),
            ))
        );
    }
}