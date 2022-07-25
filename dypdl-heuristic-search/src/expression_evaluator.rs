use crate::evaluator::Evaluator;
use dypdl::expression;
use dypdl::variable_type;
use dypdl::CostExpression;

/// Evaluator using an expression.
#[derive(Debug, PartialEq, Clone)]
pub struct ExpressionEvaluator(CostExpression);

impl Default for ExpressionEvaluator {
    fn default() -> Self {
        Self(CostExpression::Integer(
            expression::IntegerExpression::Constant(0),
        ))
    }
}

impl ExpressionEvaluator {
    pub fn new(expression: CostExpression) -> ExpressionEvaluator {
        ExpressionEvaluator(expression)
    }
}

impl Evaluator for ExpressionEvaluator {
    fn eval<T: variable_type::Numeric, S: dypdl::DPState>(
        &self,
        state: &S,
        model: &dypdl::Model,
    ) -> Option<T> {
        Some(self.0.eval(state, &model.table_registry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_new() {
        let evaluator = ExpressionEvaluator::default();
        assert_eq!(
            evaluator,
            ExpressionEvaluator::new(CostExpression::Integer(
                expression::IntegerExpression::Constant(0)
            ))
        );
    }

    #[test]
    fn eval() {
        let model = dypdl::Model::default();
        let state = dypdl::State::default();
        let evaluator = ExpressionEvaluator::new(CostExpression::Integer(
            expression::IntegerExpression::Constant(10),
        ));
        assert_eq!(evaluator.eval(&state, &model), Some(10));
    }
}
