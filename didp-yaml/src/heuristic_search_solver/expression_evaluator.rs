use crate::dypdl_parser::expression_parser;
use dypdl::CostExpression;
use dypdl_heuristic_search::ExpressionEvaluator;
use rustc_hash::FxHashMap;
use std::error::Error;

pub fn load_from_string(
    expression: String,
    model: &dypdl::Model,
    cost_type: &dypdl::CostType,
) -> Result<ExpressionEvaluator, Box<dyn Error>> {
    let parameters = FxHashMap::default();
    match cost_type {
        dypdl::CostType::Integer => Ok(ExpressionEvaluator::new(CostExpression::Integer(
            expression_parser::parse_integer(
                expression,
                &model.state_metadata,
                &model.table_registry,
                &parameters,
            )?
            .simplify(&model.table_registry),
        ))),
        dypdl::CostType::Continuous => Ok(ExpressionEvaluator::new(CostExpression::Continuous(
            expression_parser::parse_continuous(
                expression,
                &model.state_metadata,
                &model.table_registry,
                &parameters,
            )?
            .simplify(&model.table_registry),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::prelude::*;

    #[test]
    fn load_from_string_ok() {
        let model = Model::default();
        let cost_type = CostType::Integer;
        let expression = String::from("1");
        let evaluator = load_from_string(expression, &model, &cost_type);
        assert!(evaluator.is_ok());
        assert_eq!(
            evaluator.unwrap(),
            ExpressionEvaluator::new(CostExpression::Integer(IntegerExpression::Constant(1)))
        );
        let cost_type = CostType::Continuous;
        let expression = String::from("1.0");
        let evaluator = load_from_string(expression, &model, &cost_type);
        assert!(evaluator.is_ok());
        assert_eq!(
            evaluator.unwrap(),
            ExpressionEvaluator::new(CostExpression::Continuous(ContinuousExpression::Constant(
                1.0
            )))
        );
    }

    #[test]
    fn laod_from_string_err() {
        let model = Model::default();
        let cost_type = CostType::Integer;
        let expression = String::from("foo");
        let evaluator = load_from_string(expression, &model, &cost_type);
        assert!(evaluator.is_err());
        let cost_type = CostType::Continuous;
        let expression = String::from("foo");
        let evaluator = load_from_string(expression, &model, &cost_type);
        assert!(evaluator.is_err());
    }
}
