//! A module for expressions.

mod argument_expression;
mod condition;
mod continuous_expression;
mod continuous_vector_expression;
mod element_expression;
mod integer_expression;
mod integer_vector_expression;
mod numeric_operator;
mod numeric_table_expression;
mod reference_expression;
mod set_condition;
mod set_expression;
mod set_reduce_expression;
mod table_expression;
mod table_vector_expression;
mod util;
mod vector_expression;

pub use condition::{ComparisonOperator, Condition, IfThenElse};
pub use continuous_expression::ContinuousExpression;
pub use continuous_vector_expression::ContinuousVectorExpression;
pub use element_expression::ElementExpression;
pub use integer_expression::IntegerExpression;
pub use integer_vector_expression::IntegerVectorExpression;
pub use numeric_operator::{
    BinaryOperator, CastOperator, ContinuousBinaryOperation, ContinuousBinaryOperator,
    ContinuousUnaryOperator, MaxMin, ReduceOperator, UnaryOperator,
};
pub use set_expression::{SetElementOperation, SetElementOperator, SetExpression, SetOperator};
pub use set_reduce_expression::{SetReduceExpression, SetReduceOperator};
pub use table_expression::TableExpression;
pub use table_vector_expression::{TableVectorExpression, VectorOrElementExpression};
pub use vector_expression::VectorExpression;

pub use argument_expression::ArgumentExpression;
pub use numeric_table_expression::NumericTableExpression;
pub use reference_expression::ReferenceExpression;
pub use set_condition::SetCondition;
