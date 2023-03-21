use dypdl_heuristic_search::FEvaluatorType;
use pyo3::prelude::*;

/// An enum representing an operator to compute the f-value combining an h-value and a g-value.
///
/// :attr:`~FOperator.Plus` (0): f = g + h
///
/// :attr:`~FOperator.Max` (1): f = max(g, h)
///
/// :attr:`~FOperator.Min` (2): f = min(g, h)
///
/// :attr:`~FOperator.Product` (3): f = g * h
///
/// :attr:`~FOperator.Overwrite` (4): f = h
#[pyclass(name = "FOperator")]
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum FOperator {
    /// f = g + h
    Plus = 0,
    /// f = max(g, h)
    Max = 1,
    /// f = min(g, h)
    Min = 2,
    /// f = g * h
    Product = 3,
    /// f = h
    Overwrite = 4,
}

impl From<FOperator> for FEvaluatorType {
    fn from(f_operator: FOperator) -> Self {
        match f_operator {
            FOperator::Plus => FEvaluatorType::Plus,
            FOperator::Max => FEvaluatorType::Max,
            FOperator::Min => FEvaluatorType::Min,
            FOperator::Product => FEvaluatorType::Product,
            FOperator::Overwrite => FEvaluatorType::Overwrite,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_evaluator_type_from_f_operator() {
        assert_eq!(FEvaluatorType::from(FOperator::Plus), FEvaluatorType::Plus);
        assert_eq!(FEvaluatorType::from(FOperator::Max), FEvaluatorType::Max);
        assert_eq!(FEvaluatorType::from(FOperator::Min), FEvaluatorType::Min);
        assert_eq!(
            FEvaluatorType::from(FOperator::Product),
            FEvaluatorType::Product
        );
        assert_eq!(
            FEvaluatorType::from(FOperator::Overwrite),
            FEvaluatorType::Overwrite
        );
    }
}
