use crate::evaluator;
use didp_parser::expression;
use didp_parser::expression_parser;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::str;

pub struct ExpressionEvaluator<T: variable::Numeric>(expression::NumericExpression<T>);

impl<T: variable::Numeric> ExpressionEvaluator<T> {
    pub fn new(
        expression: String,
        model: &didp_parser::Model<T>,
    ) -> Result<ExpressionEvaluator<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let parameters = FxHashMap::default();
        let expression = expression_parser::parse_numeric(
            expression,
            &model.state_metadata,
            &model.table_registry,
            &parameters,
        )?;
        Ok(ExpressionEvaluator(expression))
    }
}

impl<T: variable::Numeric> Default for ExpressionEvaluator<T> {
    fn default() -> Self {
        ExpressionEvaluator(expression::NumericExpression::Constant(T::zero()))
    }
}

impl<T: variable::Numeric> evaluator::Evaluator<T> for ExpressionEvaluator<T> {
    fn eval<U: didp_parser::DPState>(&self, state: &U, model: &didp_parser::Model<T>) -> Option<T> {
        Some(self.0.eval(state, &model.table_registry))
    }
}
