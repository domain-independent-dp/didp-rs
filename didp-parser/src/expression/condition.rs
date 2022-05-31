use super::element_expression::TableExpression;
use super::numeric_expression::NumericExpression;
use super::set_condition;
use crate::state::DPState;
use crate::table_registry::TableRegistry;
use crate::variable;
use num_traits::FromPrimitive;

#[derive(Debug, PartialEq, Clone)]
pub enum Condition {
    Constant(bool),
    Not(Box<Condition>),
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    Comparison(Box<Comparison>),
    Set(Box<set_condition::SetCondition>),
    Table(TableExpression<bool>),
}

impl Default for Condition {
    fn default() -> Condition {
        Self::Constant(false)
    }
}

impl Condition {
    pub fn eval<T: DPState>(&self, state: &T, registry: &TableRegistry) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::Not(condition) => !condition.eval(state, registry),
            Self::And(x, y) => x.eval(state, registry) && y.eval(state, registry),
            Self::Or(x, y) => x.eval(state, registry) || y.eval(state, registry),
            Self::Comparison(condition) => condition.eval(state, registry),
            Self::Set(set) => set.eval(state, registry),
            Self::Table(table) => *table.eval(state, registry, &registry.bool_tables),
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> Condition {
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
                (x, y) => Self::Or(Box::new(x), Box::new(y)),
            },
            Self::Comparison(condition) => match condition.simplify(registry) {
                Comparison::Constant(value) => Self::Constant(value),
                condition => Self::Comparison(Box::new(condition)),
            },
            Self::Set(condition) => match condition.simplify(registry) {
                set_condition::SetCondition::Constant(value) => Self::Constant(value),
                condition => Self::Set(Box::new(condition)),
            },
            Self::Table(condition) => match condition.simplify(registry, &registry.bool_tables) {
                TableExpression::Constant(value) => Self::Constant(value),
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
    pub fn eval<T: DPState>(&self, state: &T, registry: &TableRegistry) -> bool {
        match self {
            Self::Constant(value) => *value,
            Self::ComparisonII(op, x, y) => {
                Self::eval_comparison(op, x.eval(state, registry), y.eval(state, registry))
            }
            Self::ComparisonIC(op, x, y) => Self::eval_comparison(
                op,
                variable::Continuous::from_i32(x.eval(state, registry)).unwrap(),
                y.eval(state, registry),
            ),
            Self::ComparisonCI(op, x, y) => Self::eval_comparison(
                op,
                x.eval(state, registry),
                variable::Continuous::from_i32(y.eval(state, registry)).unwrap(),
            ),
            Self::ComparisonCC(op, x, y) => {
                Self::eval_comparison(op, x.eval(state, registry), y.eval(state, registry))
            }
        }
    }

    pub fn simplify(&self, registry: &TableRegistry) -> Comparison {
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
    use super::super::element_expression;
    use super::*;
    use crate::state::*;
    use crate::table;
    use crate::table_data;
    use rustc_hash::FxHashMap;

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), true);

        let tables_1d = vec![table::Table1D::new(vec![true, false])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![vec![true, false]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![vec![vec![true, false]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, true);
        let key = vec![0, 0, 0, 1];
        map.insert(key, false);
        let tables = vec![table::Table::new(map, false)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        TableRegistry {
            bool_tables: table_data::TableData {
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

    fn generate_state() -> State {
        let mut set1 = variable::Set::with_capacity(3);
        set1.insert(0);
        set1.insert(2);
        let mut set2 = variable::Set::with_capacity(3);
        set2.insert(0);
        set2.insert(1);
        State {
            signature_variables: SignatureVariables {
                set_variables: vec![set1, set2],
                vector_variables: vec![vec![0, 2]],
                element_variables: vec![1],
                integer_variables: vec![1, 2, 3],
                continuous_variables: vec![1.0, 2.0, 3.0],
            },
            resource_variables: ResourceVariables {
                integer_variables: vec![4, 5, 6],
                continuous_variables: vec![4.0, 5.0, 6.0],
            },
        }
    }

    #[test]
    fn constant_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Constant(true);
        assert!(expression.eval(&state, &registry));
        let expression = Condition::Constant(false);
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn eq_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn ne_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
        );
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn ge_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1.0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(1.0),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn gt_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1.0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(1.0),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn le_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Le,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Le,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Le,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Le,
            NumericExpression::Constant(1.0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Le,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Le,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Le,
            NumericExpression::Constant(1.0),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn lt_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1.0),
            NumericExpression::Constant(0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(1.0),
            NumericExpression::Constant(0.0),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn not_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Not(Box::new(Condition::Constant(true)));
        assert!(!expression.eval(&state, &registry));

        let expression = Condition::Not(Box::new(Condition::Constant(false)));
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn and_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Condition::And(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(false)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert!(!expression.eval(&state, &registry));

        let expression = Condition::And(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn or_eval() {
        let registry = generate_registry();
        let state = generate_state();

        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(true)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Condition::Or(
            Box::new(Condition::Constant(true)),
            Box::new(Condition::Constant(false)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(true)),
        );
        assert!(expression.eval(&state, &registry));

        let expression = Condition::Or(
            Box::new(Condition::Constant(false)),
            Box::new(Condition::Constant(false)),
        );
        assert!(!expression.eval(&state, &registry));
    }

    #[test]
    fn table_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = Condition::Table(TableExpression::Table1D(
            0,
            element_expression::ElementExpression::Constant(0),
        ));
        assert!(expression.eval(&state, &registry));
    }

    #[test]
    fn set_eval() {
        let registry = generate_registry();
        let state = generate_state();
        let expression = Condition::Set(Box::new(set_condition::SetCondition::Eq(
            element_expression::ElementExpression::Variable(0),
            element_expression::ElementExpression::Constant(1),
        )));
        assert!(expression.eval(&state, &registry));
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

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
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
            NumericExpression::IntegerResourceVariable(0),
            NumericExpression::IntegerResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Eq,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Eq,
            NumericExpression::ContinuousResourceVariable(0),
            NumericExpression::ContinuousResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Eq,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Eq,
            NumericExpression::Constant(0.0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Eq,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);
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

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
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
            NumericExpression::IntegerResourceVariable(0),
            NumericExpression::IntegerResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ne,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ne,
            NumericExpression::ContinuousResourceVariable(0),
            NumericExpression::ContinuousResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ne,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ne,
            NumericExpression::Constant(0.0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ne,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn gt_simplify() {
        let registry = generate_registry();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
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
            NumericExpression::IntegerResourceVariable(0),
            NumericExpression::IntegerResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Gt,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Gt,
            NumericExpression::ContinuousResourceVariable(0),
            NumericExpression::ContinuousResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Gt,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Gt,
            NumericExpression::Constant(0.0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Gt,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);
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

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
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
            NumericExpression::IntegerResourceVariable(0),
            NumericExpression::IntegerResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ge,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ge,
            NumericExpression::ContinuousResourceVariable(0),
            NumericExpression::ContinuousResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Ge,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ge,
            NumericExpression::Constant(0.0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Ge,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn lt_simplify() {
        let registry = generate_registry();

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::IntegerResourceVariable(0),
            NumericExpression::IntegerResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Lt,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Lt,
            NumericExpression::ContinuousResourceVariable(0),
            NumericExpression::ContinuousResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(false));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Lt,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Lt,
            NumericExpression::Constant(0.0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Lt,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);
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

        let expression = Comparison::ComparisonIC(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCI(
            ComparisonOperator::Le,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Le,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(0.0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::Constant(1),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Le,
            NumericExpression::Constant(0.0),
            NumericExpression::Constant(1.0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::IntegerResourceVariable(0),
            NumericExpression::IntegerResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Le,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Le,
            NumericExpression::ContinuousResourceVariable(0),
            NumericExpression::ContinuousResourceVariable(0),
        );
        assert_eq!(expression.simplify(&registry), Comparison::Constant(true));

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::Constant(0),
            NumericExpression::IntegerVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonII(
            ComparisonOperator::Le,
            NumericExpression::IntegerVariable(0),
            NumericExpression::IntegerVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Le,
            NumericExpression::Constant(0.0),
            NumericExpression::ContinuousVariable(0),
        );
        assert_eq!(expression.simplify(&registry), expression);

        let expression = Comparison::ComparisonCC(
            ComparisonOperator::Le,
            NumericExpression::ContinuousVariable(0),
            NumericExpression::ContinuousVariable(1),
        );
        assert_eq!(expression.simplify(&registry), expression);
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
            Condition::Or(
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

        let expression = Condition::Table(TableExpression::Table1D(
            0,
            element_expression::ElementExpression::Constant(0),
        ));
        assert_eq!(expression.simplify(&registry), Condition::Constant(true));

        let expression = Condition::Table(TableExpression::Table1D(
            0,
            element_expression::ElementExpression::Variable(0),
        ));
        assert_eq!(
            expression.simplify(&registry),
            Condition::Table(TableExpression::Table1D(
                0,
                element_expression::ElementExpression::Variable(0),
            ))
        );
    }

    #[test]
    fn set_simplify() {
        let registry = generate_registry();
        let expression = Condition::Set(Box::new(set_condition::SetCondition::Eq(
            element_expression::ElementExpression::Variable(0),
            element_expression::ElementExpression::Constant(1),
        )));
        assert_eq!(
            expression.simplify(&registry),
            Condition::Set(Box::new(set_condition::SetCondition::Eq(
                element_expression::ElementExpression::Variable(0),
                element_expression::ElementExpression::Constant(1),
            )))
        );
    }
}
