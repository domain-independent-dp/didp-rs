use dypdl::variable_type;
use std::cmp;

/// How to combine the g-value and the h-value to compute the f-value.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FEvaluatorType {
    /// f = g + h.
    Plus,
    /// f = max(g, h).
    Max,
    /// f = min(g, h).
    Min,
    /// f = g * h.
    Product,
    /// f = h.
    Overwrite,
}

impl Default for FEvaluatorType {
    fn default() -> Self {
        FEvaluatorType::Plus
    }
}

impl FEvaluatorType {
    /// Compute the f-value given the g-value and the h-value.
    pub fn eval<T: variable_type::Numeric + Ord>(&self, g: T, h: T) -> T {
        match self {
            FEvaluatorType::Plus => g + h,
            FEvaluatorType::Max => cmp::max(g, h),
            FEvaluatorType::Min => cmp::min(g, h),
            FEvaluatorType::Product => g * h,
            FEvaluatorType::Overwrite => h,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_evaluator_type_plus() {
        let f_evaluator_type = FEvaluatorType::Plus;
        assert_eq!(f_evaluator_type.eval(2, 3), 5);
    }

    #[test]
    fn f_evaluator_type_max() {
        let f_evaluator_type = FEvaluatorType::Max;
        assert_eq!(f_evaluator_type.eval(2, 3), 3);
    }

    #[test]
    fn f_evaluator_type_min() {
        let f_evaluator_type = FEvaluatorType::Min;
        assert_eq!(f_evaluator_type.eval(2, 3), 2);
    }

    #[test]
    fn f_evaluator_type_product() {
        let f_evaluator_type = FEvaluatorType::Product;
        assert_eq!(f_evaluator_type.eval(2, 3), 6);
    }

    #[test]
    fn f_evaluator_type_overwrite() {
        let f_evaluator_type = FEvaluatorType::Overwrite;
        assert_eq!(f_evaluator_type.eval(2, 3), 3);
    }
}
