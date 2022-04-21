use crate::variable::Numeric;

#[derive(Debug, PartialEq, Clone)]
pub enum MathFunction {
    Sqrt,
    Abs,
    Floor,
    Ceiling,
}

impl MathFunction {
    pub fn eval<T: Numeric + num_traits::Signed>(&self, x: T) -> T {
        match self {
            Self::Sqrt => T::from(x.to_continuous().sqrt()),
            Self::Abs => x.abs(),
            Self::Floor => T::from(x.to_continuous().floor()),
            Self::Ceiling => T::from(x.to_continuous().ceil()),
        }
    }
}

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
    fn sqrt() {
        let op = MathFunction::Sqrt;
        assert_eq!(op.eval(4.0), 2.0);
    }

    #[test]
    fn abs() {
        let op = MathFunction::Abs;
        assert_eq!(op.eval(-4.0), 4.0);
    }

    #[test]
    fn floor() {
        let op = MathFunction::Floor;
        assert_eq!(op.eval(4.5), 4.0);
    }

    #[test]
    fn ceiling() {
        let op = MathFunction::Ceiling;
        assert_eq!(op.eval(4.5), 5.0);
    }

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
