use crate::variable_type::Continuous;
use num_traits::{Num, Signed};
use std::iter::{Product, Sum};

/// Unary arithmetic operator.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum UnaryOperator {
    /// Negative.
    Neg,
    /// Absolute value.
    Abs,
}

impl UnaryOperator {
    /// Returns the evaluation result.
    pub fn eval<T: Num + Signed>(&self, x: T) -> T {
        match self {
            Self::Abs => x.abs(),
            Self::Neg => -x,
        }
    }

    /// Returns the evaluation result for a vector.
    pub fn eval_vector<T: Num + Signed + Copy>(&self, mut x: Vec<T>) -> Vec<T> {
        x.iter_mut().for_each(|x| *x = self.eval(*x));
        x
    }
}

/// Unary arithmetic operator specific to continuous values.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ContinuousUnaryOperator {
    /// Square root.
    Sqrt,
}

impl ContinuousUnaryOperator {
    /// Returns the evaluation result.
    pub fn eval(&self, x: Continuous) -> Continuous {
        match self {
            Self::Sqrt => x.sqrt(),
        }
    }

    /// Returns the evaluation result for a vector.
    pub fn eval_vector(&self, mut x: Vec<Continuous>) -> Vec<Continuous> {
        x.iter_mut().for_each(|x| *x = self.eval(*x));
        x
    }
}

/// Operator to convert a continuous value to an integer value.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum CastOperator {
    // Floor.
    Floor,
    // Ceiling.
    Ceil,
    // Rounding.
    Round,
    // Truncate.
    Trunc,
}

impl CastOperator {
    /// Returns the evaluation result.
    pub fn eval(&self, x: Continuous) -> Continuous {
        match self {
            Self::Floor => x.floor(),
            Self::Ceil => x.ceil(),
            Self::Round => x.round(),
            Self::Trunc => x.trunc(),
        }
    }

    /// Returns the evaluation result for a vector.
    pub fn eval_vector(&self, mut x: Vec<Continuous>) -> Vec<Continuous> {
        x.iter_mut().for_each(|x| *x = self.eval(*x));
        x
    }
}

/// Binary arithmetic operator.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BinaryOperator {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    // Division.
    Div,
    /// Remainder.
    Rem,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}

/// A trait for max/min binary operation.
pub trait MaxMin<Rhs = Self> {
    type Output;
    /// Returns an expression representing the maximum.
    fn max(self, rhs: Rhs) -> Self::Output;
    /// Returns an expression representing the minimum.
    fn min(self, rhs: Rhs) -> Self::Output;
}

impl BinaryOperator {
    /// Returns the result of the evaluation.
    pub fn eval<T: Num + PartialOrd>(&self, a: T, b: T) -> T {
        match self {
            BinaryOperator::Add => a + b,
            BinaryOperator::Sub => a - b,
            BinaryOperator::Mul => a * b,
            BinaryOperator::Div => a / b,
            BinaryOperator::Rem => a % b,
            BinaryOperator::Max => {
                if a > b {
                    a
                } else {
                    b
                }
            }
            BinaryOperator::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
        }
    }

    /// Returns the result of the evaluation with `x` and each element in `y`.
    pub fn eval_operation_x<T: Num + PartialOrd + Copy>(&self, x: T, mut y: Vec<T>) -> Vec<T> {
        y.iter_mut().for_each(|y| *y = self.eval(x, *y));
        y
    }

    /// Returns the result of the evaluation with each element in `x` and `y`.
    pub fn eval_operation_y<T: Num + PartialOrd + Copy>(&self, mut x: Vec<T>, y: T) -> Vec<T> {
        x.iter_mut().for_each(|x| *x = self.eval(*x, y));
        x
    }

    /// Returns the result of the evaluation with each element in `x` and each element in `y`.
    /// If `y` is longer than `x`, the result is truncated to the length of `x`.
    pub fn eval_vector_operation_in_x<T: Num + PartialOrd + Copy>(
        &self,
        mut x: Vec<T>,
        y: &[T],
    ) -> Vec<T> {
        x.truncate(y.len());
        x.iter_mut()
            .zip(y)
            .for_each(|(x, y)| *x = self.eval(*x, *y));
        x
    }

    /// Returns the result of the evaluation with each element in `x` and each element in `y`.
    /// If `x` is longer than `y`, the result is truncated to the length of `y`.
    pub fn eval_vector_operation_in_y<T: Num + PartialOrd + Copy>(
        &self,
        x: &[T],
        mut y: Vec<T>,
    ) -> Vec<T> {
        y.truncate(x.len());
        y.iter_mut()
            .zip(x)
            .for_each(|(y, x)| *y = self.eval(*x, *y));
        y
    }
}

/// Binary arithmetic operator specific to continuous values.

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ContinuousBinaryOperator {
    /// Power.
    Pow,
    /// Logarithm.
    Log,
}

/// A trait for binary arithmetic operator specific to continuous values.
pub trait ContinuousBinaryOperation<Rhs = Self> {
    type Output;
    /// Returns an expression representing the power.
    fn pow(self, rhs: Rhs) -> Self::Output;
    /// Returns an expression representing the logarithm.
    fn log(self, rhs: Rhs) -> Self::Output;
}

impl ContinuousBinaryOperator {
    /// Returns the evaluation result.
    pub fn eval(&self, x: Continuous, y: Continuous) -> Continuous {
        match self {
            Self::Pow => x.powf(y),
            Self::Log => x.log(y),
        }
    }

    /// Returns the evaluation result for a vector.
    pub fn eval_operation_x(&self, x: Continuous, mut y: Vec<Continuous>) -> Vec<Continuous> {
        y.iter_mut().for_each(|y| *y = self.eval(x, *y));
        y
    }

    /// Returns the evaluation result for a vector.
    pub fn eval_operation_y(&self, mut x: Vec<Continuous>, y: Continuous) -> Vec<Continuous> {
        x.iter_mut().for_each(|x| *x = self.eval(*x, y));
        x
    }

    /// Returns the evaluation result for an elementwise computation.
    pub fn eval_vector_operation_in_x(
        &self,
        mut x: Vec<Continuous>,
        y: &[Continuous],
    ) -> Vec<Continuous> {
        x.truncate(y.len());
        x.iter_mut()
            .zip(y)
            .for_each(|(x, y)| *x = self.eval(*x, *y));
        x
    }

    /// Returns the evaluation result for an elementwise computation.
    pub fn eval_vector_operation_in_y(
        &self,
        x: &[Continuous],
        mut y: Vec<Continuous>,
    ) -> Vec<Continuous> {
        y.truncate(x.len());
        y.iter_mut()
            .zip(x)
            .for_each(|(y, x)| *y = self.eval(*x, *y));
        y
    }
}

/// Operator to reduce a vector to a single value.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ReduceOperator {
    /// Summation.
    Sum,
    /// Product.
    Product,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}

impl ReduceOperator {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty vector.
    pub fn eval<T: Num + PartialOrd + Copy + Sum + Product>(&self, vector: &[T]) -> T {
        self.eval_iter(vector.iter().copied()).unwrap()
    }

    /// Returns the evaluation result.
    pub fn eval_iter<T: Num + PartialOrd + Copy + Sum + Product, I: Iterator<Item = T>>(
        &self,
        iter: I,
    ) -> Option<T> {
        match self {
            Self::Sum => Some(iter.sum()),
            Self::Product => Some(iter.product()),
            Self::Max => iter.reduce(|x, y| if y > x { y } else { x }),
            Self::Min => iter.reduce(|x, y| if y < x { y } else { x }),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg() {
        let op = UnaryOperator::Neg;
        assert_eq!(op.eval(1), -1);
    }

    #[test]
    fn neg_vector() {
        let op = UnaryOperator::Neg;
        assert_eq!(op.eval_vector(vec![1, -2]), vec![-1, 2]);
    }

    #[test]
    fn abs() {
        let op = UnaryOperator::Abs;
        assert_eq!(op.eval(-1), 1);
    }

    #[test]
    fn abs_vector() {
        let op = UnaryOperator::Abs;
        assert_eq!(op.eval_vector(vec![1, -2]), vec![1, 2]);
    }

    #[test]
    fn sqrt() {
        let op = ContinuousUnaryOperator::Sqrt;
        assert_eq!(op.eval(4.0), 2.0);
    }

    #[test]
    fn sqrt_vector() {
        let op = ContinuousUnaryOperator::Sqrt;
        assert_eq!(op.eval_vector(vec![4.0, 9.0]), vec![2.0, 3.0]);
    }

    #[test]
    fn floor() {
        let op = CastOperator::Floor;
        assert_eq!(op.eval(4.5), 4.0);
    }

    #[test]
    fn floor_vector() {
        let op = CastOperator::Floor;
        assert_eq!(op.eval_vector(vec![4.5, 4.3]), vec![4.0, 4.0]);
    }

    #[test]
    fn ceil() {
        let op = CastOperator::Ceil;
        assert_eq!(op.eval(4.5), 5.0);
    }

    #[test]
    fn ceil_vector() {
        let op = CastOperator::Ceil;
        assert_eq!(op.eval_vector(vec![4.5, 4.3]), vec![5.0, 5.0]);
    }

    #[test]
    fn round() {
        let op = CastOperator::Round;
        assert_eq!(op.eval(4.2), 4.0);
        assert_eq!(op.eval(4.6), 5.0);
    }

    #[test]
    fn round_vector() {
        let op = CastOperator::Round;
        assert_eq!(op.eval_vector(vec![4.2, 4.6]), vec![4.0, 5.0]);
    }

    #[test]
    fn trunc() {
        let op = CastOperator::Trunc;
        assert_eq!(op.eval(4.2), 4.0);
        assert_eq!(op.eval(-4.6), -4.0);
    }

    #[test]
    fn trunc_vector() {
        let op = CastOperator::Trunc;
        assert_eq!(op.eval_vector(vec![4.2, -4.6]), vec![4.0, -4.0]);
    }

    #[test]
    fn add() {
        let op = BinaryOperator::Add;
        assert_eq!(op.eval(6, 3), 9);
    }

    #[test]
    fn add_x() {
        let op = BinaryOperator::Add;
        assert_eq!(op.eval_operation_x(6, vec![3, 4]), vec![9, 10]);
    }

    #[test]
    fn add_y() {
        let op = BinaryOperator::Add;
        assert_eq!(op.eval_operation_y(vec![3, 4], 6), vec![9, 10]);
    }

    #[test]
    fn add_vector_in_x() {
        let op = BinaryOperator::Add;
        assert_eq!(
            op.eval_vector_operation_in_x(vec![3, 4], &[1, 2]),
            vec![4, 6]
        );
    }

    #[test]
    fn add_vector_in_y() {
        let op = BinaryOperator::Add;
        assert_eq!(
            op.eval_vector_operation_in_y(&[3, 4], vec![1, 2]),
            vec![4, 6]
        );
    }

    #[test]
    fn sub() {
        let op = BinaryOperator::Sub;
        assert_eq!(op.eval(6, 3), 3);
    }

    #[test]
    fn sub_x() {
        let op = BinaryOperator::Sub;
        assert_eq!(op.eval_operation_x(6, vec![3, 4]), vec![3, 2]);
    }

    #[test]
    fn sub_y() {
        let op = BinaryOperator::Sub;
        assert_eq!(op.eval_operation_y(vec![3, 4], 6), vec![-3, -2]);
    }

    #[test]
    fn sub_vector_in_x() {
        let op = BinaryOperator::Sub;
        assert_eq!(
            op.eval_vector_operation_in_x(vec![3, 4], &[1, 2]),
            vec![2, 2]
        );
    }

    #[test]
    fn sub_vector_in_y() {
        let op = BinaryOperator::Sub;
        assert_eq!(
            op.eval_vector_operation_in_y(&[3, 4], vec![1, 2]),
            vec![2, 2]
        );
    }

    #[test]
    fn mul() {
        let op = BinaryOperator::Mul;
        assert_eq!(op.eval(6, 3), 18);
    }

    #[test]
    fn mul_x() {
        let op = BinaryOperator::Mul;
        assert_eq!(op.eval_operation_x(6, vec![3, 4]), vec![18, 24]);
    }

    #[test]
    fn mul_y() {
        let op = BinaryOperator::Mul;
        assert_eq!(op.eval_operation_y(vec![3, 4], 6), vec![18, 24]);
    }

    #[test]
    fn mul_vector_in_x() {
        let op = BinaryOperator::Mul;
        assert_eq!(
            op.eval_vector_operation_in_x(vec![3, 4], &[1, 2]),
            vec![3, 8]
        );
    }

    #[test]
    fn mul_vector_in_y() {
        let op = BinaryOperator::Mul;
        assert_eq!(
            op.eval_vector_operation_in_y(&[3, 4], vec![1, 2]),
            vec![3, 8]
        );
    }

    #[test]
    fn div() {
        let op = BinaryOperator::Div;
        assert_eq!(op.eval(6, 3), 2);
    }

    #[test]
    fn div_x() {
        let op = BinaryOperator::Div;
        assert_eq!(op.eval_operation_x(6, vec![3, 2]), vec![2, 3]);
    }

    #[test]
    fn div_y() {
        let op = BinaryOperator::Div;
        assert_eq!(op.eval_operation_y(vec![4, 4], 2), vec![2, 2]);
    }

    #[test]
    fn div_vector_in_x() {
        let op = BinaryOperator::Div;
        assert_eq!(
            op.eval_vector_operation_in_x(vec![3, 4], &[1, 2]),
            vec![3, 2]
        );
    }

    #[test]
    fn div_vector_in_y() {
        let op = BinaryOperator::Div;
        assert_eq!(
            op.eval_vector_operation_in_y(&[3, 4], vec![1, 2]),
            vec![3, 2]
        );
    }

    #[test]
    fn rem() {
        let op = BinaryOperator::Rem;
        assert_eq!(op.eval(14, 3), 2);
    }

    #[test]
    fn rem_x() {
        let op = BinaryOperator::Rem;
        assert_eq!(op.eval_operation_x(6, vec![4, 2]), vec![2, 0]);
    }

    #[test]
    fn rem_y() {
        let op = BinaryOperator::Rem;
        assert_eq!(op.eval_operation_y(vec![4, 4], 3), vec![1, 1]);
    }

    #[test]
    fn rem_vector_in_x() {
        let op = BinaryOperator::Rem;
        assert_eq!(
            op.eval_vector_operation_in_x(vec![3, 3], &[1, 2]),
            vec![0, 1]
        );
    }

    #[test]
    fn rem_vector_in_y() {
        let op = BinaryOperator::Rem;
        assert_eq!(
            op.eval_vector_operation_in_y(&[3, 4], vec![1, 3]),
            vec![0, 1]
        );
    }

    #[test]
    fn max() {
        let op = BinaryOperator::Max;
        assert_eq!(op.eval(6, 3), 6);
    }

    #[test]
    fn max_x() {
        let op = BinaryOperator::Max;
        assert_eq!(op.eval_operation_x(6, vec![4, 2]), vec![6, 6]);
    }

    #[test]
    fn max_y() {
        let op = BinaryOperator::Max;
        assert_eq!(op.eval_operation_y(vec![4, 4], 3), vec![4, 4]);
    }

    #[test]
    fn max_vector_in_x() {
        let op = BinaryOperator::Max;
        assert_eq!(
            op.eval_vector_operation_in_x(vec![3, 3], &[1, 2]),
            vec![3, 3]
        );
    }

    #[test]
    fn max_vector_in_y() {
        let op = BinaryOperator::Max;
        assert_eq!(
            op.eval_vector_operation_in_y(&[3, 4], vec![1, 3]),
            vec![3, 4]
        );
    }

    #[test]
    fn min() {
        let op = BinaryOperator::Min;
        assert_eq!(op.eval(6, 3), 3);
    }

    #[test]
    fn min_x() {
        let op = BinaryOperator::Min;
        assert_eq!(op.eval_operation_x(6, vec![4, 2]), vec![4, 2]);
    }

    #[test]
    fn min_y() {
        let op = BinaryOperator::Min;
        assert_eq!(op.eval_operation_y(vec![4, 4], 3), vec![3, 3]);
    }

    #[test]
    fn min_vector_in_x() {
        let op = BinaryOperator::Min;
        assert_eq!(
            op.eval_vector_operation_in_x(vec![3, 3], &[1, 2]),
            vec![1, 2]
        );
    }

    #[test]
    fn min_vector_in_y() {
        let op = BinaryOperator::Min;
        assert_eq!(
            op.eval_vector_operation_in_y(&[3, 4], vec![1, 3]),
            vec![1, 3]
        );
    }

    #[test]
    fn pow() {
        let op = ContinuousBinaryOperator::Pow;
        assert_eq!(op.eval(2.0, 2.0), 4.0);
    }

    #[test]
    fn pow_x() {
        let op = ContinuousBinaryOperator::Pow;
        assert_eq!(op.eval_operation_x(2.0, vec![2.0, 3.0]), vec![4.0, 8.0]);
    }

    #[test]
    fn pow_y() {
        let op = ContinuousBinaryOperator::Pow;
        assert_eq!(op.eval_operation_y(vec![2.0, 3.0], 2.0), vec![4.0, 9.0]);
    }

    #[test]
    fn pow_vector_in_x() {
        let op = ContinuousBinaryOperator::Pow;
        assert_eq!(
            op.eval_vector_operation_in_x(vec![2.0, 3.0], &[2.0, 3.0]),
            vec![4.0, 27.0]
        );
    }

    #[test]
    fn pow_vector_in_y() {
        let op = ContinuousBinaryOperator::Pow;
        assert_eq!(
            op.eval_vector_operation_in_y(&[2.0, 3.0], vec![2.0, 3.0]),
            vec![4.0, 27.0]
        );
    }

    #[test]
    fn log() {
        let op = ContinuousBinaryOperator::Log;
        assert_eq!(op.eval(4.0, 2.0), 2.0);
    }

    #[test]
    fn log_x() {
        let op = ContinuousBinaryOperator::Log;
        assert_eq!(op.eval_operation_x(16.0, vec![2.0, 4.0]), vec![4.0, 2.0]);
    }

    #[test]
    fn log_y() {
        let op = ContinuousBinaryOperator::Log;
        assert_eq!(op.eval_operation_y(vec![2.0, 4.0], 2.0), vec![1.0, 2.0]);
    }

    #[test]
    fn log_vector_in_x() {
        let op = ContinuousBinaryOperator::Log;
        assert_eq!(
            op.eval_vector_operation_in_x(vec![2.0, 4.0], &[2.0, 2.0]),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn log_vector_in_y() {
        let op = ContinuousBinaryOperator::Log;
        assert_eq!(
            op.eval_vector_operation_in_y(&[2.0, 4.0], vec![2.0, 2.0]),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn reduce_sum() {
        let op = ReduceOperator::Sum;
        assert_eq!(op.eval(&[1, 2, 4]), 7);
    }

    #[test]
    fn reduce_product() {
        let op = ReduceOperator::Product;
        assert_eq!(op.eval(&[1, 2, 4]), 8);
    }

    #[test]
    fn reduce_max() {
        let op = ReduceOperator::Max;
        assert_eq!(op.eval(&[1, 2, 4]), 4);
    }

    #[test]
    fn reduce_min() {
        let op = ReduceOperator::Min;
        assert_eq!(op.eval(&[1, 2, 4]), 1);
    }

    #[test]
    fn reduce_iter_sum() {
        let op = ReduceOperator::Sum;
        assert_eq!(op.eval_iter(vec![1, 2, 4].into_iter()), Some(7));
    }

    #[test]
    fn reduce_iter_product() {
        let op = ReduceOperator::Product;
        assert_eq!(op.eval_iter(vec![1, 2, 4].into_iter()), Some(8));
    }

    #[test]
    fn reduce_iter_max() {
        let op = ReduceOperator::Max;
        assert_eq!(op.eval_iter(vec![1, 2, 4].into_iter()), Some(4));
    }

    #[test]
    fn reduce_iter_min() {
        let op = ReduceOperator::Min;
        assert_eq!(op.eval_iter(vec![1, 2, 4].into_iter()), Some(1));
    }
}
