use crate::evaluator;
use crate::util;
use didp_parser::expression;
use didp_parser::expression_parser;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::str;

pub struct ExpressionEvaluator<T: variable::Numeric>(expression::NumericExpression<T>);

impl<T: variable::Numeric> ExpressionEvaluator<T> {
    pub fn load_from_yaml(
        value: &yaml_rust::Yaml,
        model: &didp_parser::Model<T>,
    ) -> Result<ExpressionEvaluator<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let parameters = FxHashMap::default();
        let expression = match value {
            yaml_rust::Yaml::String(string) => expression_parser::parse_numeric(
                string.clone(),
                &model.state_metadata,
                &model.table_registry,
                &parameters,
            )?,
            value => {
                return Err(util::ConfigErr::new(format!(
                    "expected String, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        Ok(ExpressionEvaluator(expression))
    }
}

impl<T: variable::Numeric> Default for ExpressionEvaluator<T> {
    fn default() -> Self {
        ExpressionEvaluator(expression::NumericExpression::Constant(T::zero()))
    }
}

impl<T: variable::Numeric> evaluator::Evaluator<T> for ExpressionEvaluator<T> {
    fn eval(&self, state: &didp_parser::State, model: &didp_parser::Model<T>) -> T {
        self.0.eval(state, &model.table_registry)
    }
}
