#[derive(Debug, PartialEq, Eq, Clone)]
pub enum NumericOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Max,
    Min,
}

impl NumericOperator {
    pub fn eval<T: num_traits::Num + PartialOrd>(&self, a: T, b: T) -> T {
        match self {
            NumericOperator::Add => a + b,
            NumericOperator::Subtract => a - b,
            NumericOperator::Multiply => a * b,
            NumericOperator::Divide => a / b,
            NumericOperator::Max => {
                if a > b {
                    a
                } else {
                    b
                }
            }
            NumericOperator::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let op = NumericOperator::Add;
        assert_eq!(op.eval(6, 3), 9);
    }

    #[test]
    fn subtract() {
        let op = NumericOperator::Subtract;
        assert_eq!(op.eval(6, 3), 3);
    }

    #[test]
    fn multiply() {
        let op = NumericOperator::Multiply;
        assert_eq!(op.eval(6, 3), 18);
    }

    #[test]
    fn divide() {
        let op = NumericOperator::Divide;
        assert_eq!(op.eval(6, 3), 2);
    }

    #[test]
    fn max() {
        let op = NumericOperator::Max;
        assert_eq!(op.eval(6, 3), 6);
    }

    #[test]
    fn min() {
        let op = NumericOperator::Min;
        assert_eq!(op.eval(6, 3), 3);
    }
}
