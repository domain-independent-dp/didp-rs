use super::argument_expression::ArgumentExpression;
use super::condition::{Condition, IfThenElse};
use super::continuous_vector_expression::ContinuousVectorExpression;
use super::element_expression::ElementExpression;
use super::integer_expression::IntegerExpression;
use super::numeric_operator::{
    BinaryOperator, CastOperator, ContinuousBinaryOperation, ContinuousBinaryOperator,
    ContinuousUnaryOperator, MaxMin, ReduceOperator, UnaryOperator,
};
use super::numeric_table_expression::NumericTableExpression;
use super::reference_expression::ReferenceExpression;
use super::set_expression::SetExpression;
use super::vector_expression::VectorExpression;
use crate::state::{
    ContinuousResourceVariable, ContinuousVariable, IntegerResourceVariable, IntegerVariable,
    SetVariable, StateInterface,
};
use crate::table_data::{Table1DHandle, Table2DHandle, Table3DHandle, TableHandle};
use crate::table_registry::TableRegistry;
use crate::variable_type::{Continuous, Integer};
use std::boxed::Box;
use std::ops;

/// Continuous numeric expression.
#[derive(Debug, PartialEq, Clone)]
pub enum ContinuousExpression {
    /// Constant.
    Constant(Continuous),
    /// Variable index.
    Variable(usize),
    /// Resource variable index.
    ResourceVariable(usize),
    /// The cost of the transitioned state.
    Cost,
    /// Unary arithmetic operation.
    UnaryOperation(UnaryOperator, Box<ContinuousExpression>),
    /// Unary arithmetic operation specific to continuous values.
    ContinuousUnaryOperation(ContinuousUnaryOperator, Box<ContinuousExpression>),
    /// Rounding operation.
    Round(CastOperator, Box<ContinuousExpression>),
    /// Binary arithmetic operation.
    BinaryOperation(
        BinaryOperator,
        Box<ContinuousExpression>,
        Box<ContinuousExpression>,
    ),
    /// Binary arithmetic operation specific to continuous values.
    ContinuousBinaryOperation(
        ContinuousBinaryOperator,
        Box<ContinuousExpression>,
        Box<ContinuousExpression>,
    ),
    /// Cardinality of a set expression.
    Cardinality(SetExpression),
    /// Length of a vector expression.
    Length(VectorExpression),
    /// A constant in a continuous table.
    Table(Box<NumericTableExpression<Continuous>>),
    /// If-then-else expression, which returns the first one if the condition holds and the second one otherwise.
    If(
        Box<Condition>,
        Box<ContinuousExpression>,
        Box<ContinuousExpression>,
    ),
    /// Conversion from an integer expression.
    FromInteger(Box<IntegerExpression>),
    /// The last element of a continuosu vector expression.
    Last(Box<ContinuousVectorExpression>),
    /// An item of a continuosu vector expression.
    At(Box<ContinuousVectorExpression>, ElementExpression),
    /// Reduce operation on a continuous vector expression.
    Reduce(ReduceOperator, Box<ContinuousVectorExpression>),
}

impl Default for ContinuousExpression {
    #[inline]
    fn default() -> Self {
        Self::Constant(0.0)
    }
}

impl From<Integer> for ContinuousExpression {
    #[inline]
    fn from(v: Integer) -> Self {
        Self::Constant(v as Continuous)
    }
}

impl From<Continuous> for ContinuousExpression {
    #[inline]
    fn from(v: Continuous) -> Self {
        Self::Constant(v)
    }
}

impl From<ContinuousVariable> for ContinuousExpression {
    #[inline]
    fn from(v: ContinuousVariable) -> Self {
        Self::Variable(v.id())
    }
}

impl From<ContinuousResourceVariable> for ContinuousExpression {
    #[inline]
    fn from(v: ContinuousResourceVariable) -> Self {
        Self::ResourceVariable(v.id())
    }
}

impl From<IntegerExpression> for ContinuousExpression {
    #[inline]
    fn from(v: IntegerExpression) -> Self {
        Self::FromInteger(Box::new(v))
    }
}

impl From<IntegerVariable> for ContinuousExpression {
    #[inline]
    fn from(v: IntegerVariable) -> Self {
        Self::FromInteger(Box::new(IntegerExpression::from(v)))
    }
}

impl From<IntegerResourceVariable> for ContinuousExpression {
    #[inline]
    fn from(v: IntegerResourceVariable) -> Self {
        Self::FromInteger(Box::new(IntegerExpression::from(v)))
    }
}

impl ContinuousExpression {
    /// Returns the absolute value.
    #[inline]
    pub fn abs(self) -> ContinuousExpression {
        Self::UnaryOperation(UnaryOperator::Abs, Box::new(self))
    }

    /// Returns the square root.
    #[inline]
    pub fn sqrt(self) -> ContinuousExpression {
        Self::ContinuousUnaryOperation(ContinuousUnaryOperator::Sqrt, Box::new(self))
    }

    /// Returns the floor.
    #[inline]
    pub fn floor(self) -> ContinuousExpression {
        Self::Round(CastOperator::Floor, Box::new(self))
    }

    /// Returns the ceiling.
    #[inline]
    pub fn ceil(self) -> ContinuousExpression {
        Self::Round(CastOperator::Ceil, Box::new(self))
    }

    /// Returns the rounded value.
    #[inline]
    pub fn round(self) -> ContinuousExpression {
        Self::Round(CastOperator::Round, Box::new(self))
    }

    /// Returns the truncated value.
    #[inline]
    pub fn trunc(self) -> ContinuousExpression {
        Self::Round(CastOperator::Trunc, Box::new(self))
    }
}

impl ops::Neg for ContinuousExpression {
    type Output = ContinuousExpression;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::UnaryOperation(UnaryOperator::Neg, Box::new(self))
    }
}

impl ops::Add for ContinuousExpression {
    type Output = ContinuousExpression;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        ContinuousExpression::BinaryOperation(BinaryOperator::Add, Box::new(self), Box::new(rhs))
    }
}

impl ops::Sub for ContinuousExpression {
    type Output = ContinuousExpression;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        ContinuousExpression::BinaryOperation(BinaryOperator::Sub, Box::new(self), Box::new(rhs))
    }
}

impl ops::Mul for ContinuousExpression {
    type Output = ContinuousExpression;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        ContinuousExpression::BinaryOperation(BinaryOperator::Mul, Box::new(self), Box::new(rhs))
    }
}

impl ops::Div for ContinuousExpression {
    type Output = ContinuousExpression;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        ContinuousExpression::BinaryOperation(BinaryOperator::Div, Box::new(self), Box::new(rhs))
    }
}

impl ops::Rem for ContinuousExpression {
    type Output = ContinuousExpression;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        ContinuousExpression::BinaryOperation(BinaryOperator::Rem, Box::new(self), Box::new(rhs))
    }
}

impl MaxMin for ContinuousExpression {
    type Output = ContinuousExpression;

    #[inline]
    fn max(self, rhs: Self) -> Self::Output {
        ContinuousExpression::BinaryOperation(BinaryOperator::Max, Box::new(self), Box::new(rhs))
    }

    #[inline]
    fn min(self, rhs: Self) -> Self::Output {
        ContinuousExpression::BinaryOperation(BinaryOperator::Min, Box::new(self), Box::new(rhs))
    }
}

impl ContinuousBinaryOperation for ContinuousExpression {
    type Output = ContinuousExpression;

    #[inline]
    fn pow(self, rhs: Self) -> Self::Output {
        ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(self),
            Box::new(rhs),
        )
    }

    #[inline]
    fn log(self, rhs: Self) -> Self::Output {
        ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Log,
            Box::new(self),
            Box::new(rhs),
        )
    }
}

impl SetExpression {
    /// Returns the cardinalty.
    #[inline]
    pub fn len_continuous(self) -> ContinuousExpression {
        ContinuousExpression::Cardinality(self)
    }
}

impl SetVariable {
    /// Returns the cardinalty.
    #[inline]
    pub fn len_continuous(self) -> ContinuousExpression {
        ContinuousExpression::Cardinality(SetExpression::from(self))
    }
}

impl Table1DHandle<Continuous> {
    /// Returns a constant in a 1D continuous table.
    #[inline]
    pub fn element<T>(&self, x: T) -> ContinuousExpression
    where
        ElementExpression: From<T>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table1D(
            self.id(),
            ElementExpression::from(x),
        )))
    }

    /// Returns the sum of constants over a set expression in a 1D continuous table.
    #[inline]
    pub fn sum<T>(&self, x: T) -> ContinuousExpression
    where
        SetExpression: From<T>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
            ReduceOperator::Sum,
            self.id(),
            SetExpression::from(x),
        )))
    }

    /// Returns the product of constants over a set expression in a 1D continuous table.
    #[inline]
    pub fn product<T>(&self, x: T) -> ContinuousExpression
    where
        SetExpression: From<T>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
            ReduceOperator::Product,
            self.id(),
            SetExpression::from(x),
        )))
    }

    /// Returns the maximum of constants over a set expression in a 1D continuous table.
    #[inline]
    pub fn max<T>(&self, x: T) -> ContinuousExpression
    where
        SetExpression: From<T>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
            ReduceOperator::Max,
            self.id(),
            SetExpression::from(x),
        )))
    }

    /// Returns the minimum of constants over a set expression in a 1D continuous table.
    #[inline]
    pub fn min<T>(&self, x: T) -> ContinuousExpression
    where
        SetExpression: From<T>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
            ReduceOperator::Min,
            self.id(),
            SetExpression::from(x),
        )))
    }
}

impl Table2DHandle<Continuous> {
    /// Returns a constant in a 2D continuous table.
    #[inline]
    pub fn element<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        ElementExpression: From<T>,
        ElementExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the sum of constants over a set expression in a 2D continuous table.
    #[inline]
    pub fn sum_x<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        SetExpression: From<T>,
        ElementExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
            ReduceOperator::Sum,
            self.id(),
            SetExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the sum of constants over a set expression in a 2D continuous table.
    #[inline]
    pub fn sum_y<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        ElementExpression: From<T>,
        SetExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
            ReduceOperator::Sum,
            self.id(),
            ElementExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the sum of constants over two set expressions in a 2D continuous table.
    #[inline]
    pub fn sum<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        SetExpression: From<T>,
        SetExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
            ReduceOperator::Sum,
            self.id(),
            SetExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the product of constants over a set expression in a 2D continuous table.
    #[inline]
    pub fn product_x<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        SetExpression: From<T>,
        ElementExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
            ReduceOperator::Product,
            self.id(),
            SetExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the product of constants over a set expression in a 2D continuous table.
    #[inline]
    pub fn product_y<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        ElementExpression: From<T>,
        SetExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
            ReduceOperator::Product,
            self.id(),
            ElementExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the product of constants over two set expressions in a 2D continuous table.
    #[inline]
    pub fn product<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        SetExpression: From<T>,
        SetExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
            ReduceOperator::Product,
            self.id(),
            SetExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the maximum of constants over a set expression in a 2D continuous table.
    #[inline]
    pub fn max_x<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        SetExpression: From<T>,
        ElementExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
            ReduceOperator::Max,
            self.id(),
            SetExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the maximum of constants over a set expression in a 2D continuous table.
    #[inline]
    pub fn max_y<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        ElementExpression: From<T>,
        SetExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
            ReduceOperator::Max,
            self.id(),
            ElementExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the maximum of constants over two set expressions in a 2D continuous table.
    #[inline]
    pub fn max<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        SetExpression: From<T>,
        SetExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
            ReduceOperator::Max,
            self.id(),
            SetExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the minimum of constants over a set expression in a 2D continuous table.
    #[inline]
    pub fn min_x<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        SetExpression: From<T>,
        ElementExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
            ReduceOperator::Min,
            self.id(),
            SetExpression::from(x),
            ElementExpression::from(y),
        )))
    }

    /// Returns the minimum of constants over a set expression in a 2D continuous table.
    #[inline]
    pub fn min_y<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        ElementExpression: From<T>,
        SetExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
            ReduceOperator::Min,
            self.id(),
            ElementExpression::from(x),
            SetExpression::from(y),
        )))
    }

    /// Returns the minimum of constants over two set expressions in a 2D continuous table.
    #[inline]
    pub fn min<T, U>(&self, x: T, y: U) -> ContinuousExpression
    where
        SetExpression: From<T>,
        SetExpression: From<U>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
            ReduceOperator::Min,
            self.id(),
            SetExpression::from(x),
            SetExpression::from(y),
        )))
    }
}

impl Table3DHandle<Continuous> {
    /// Returns a constant in a 3D continuous table.
    #[inline]
    pub fn element<T, U, V>(&self, x: T, y: U, z: V) -> ContinuousExpression
    where
        ElementExpression: From<T>,
        ElementExpression: From<U>,
        ElementExpression: From<V>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table3D(
            self.id(),
            ElementExpression::from(x),
            ElementExpression::from(y),
            ElementExpression::from(z),
        )))
    }

    /// Returns the sum of constants over set expressions in a 3D continuous table.
    #[inline]
    pub fn sum<T, U, V>(&self, x: T, y: U, z: V) -> ContinuousExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
            ReduceOperator::Sum,
            self.id(),
            ArgumentExpression::from(x),
            ArgumentExpression::from(y),
            ArgumentExpression::from(z),
        )))
    }

    /// Returns the product of constants over set expressions in a 3D continuous table.
    #[inline]
    pub fn product<T, U, V>(&self, x: T, y: U, z: V) -> ContinuousExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
            ReduceOperator::Product,
            self.id(),
            ArgumentExpression::from(x),
            ArgumentExpression::from(y),
            ArgumentExpression::from(z),
        )))
    }

    /// Returns the maximum of constants over set expressions in a 3D continuous table.
    #[inline]
    pub fn max<T, U, V>(&self, x: T, y: U, z: V) -> ContinuousExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
            ReduceOperator::Max,
            self.id(),
            ArgumentExpression::from(x),
            ArgumentExpression::from(y),
            ArgumentExpression::from(z),
        )))
    }

    /// Returns the minimum of constants over set expressions in a 3D continuous table.
    #[inline]
    pub fn min<T, U, V>(&self, x: T, y: U, z: V) -> ContinuousExpression
    where
        ArgumentExpression: From<T>,
        ArgumentExpression: From<U>,
        ArgumentExpression: From<V>,
    {
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
            ReduceOperator::Min,
            self.id(),
            ArgumentExpression::from(x),
            ArgumentExpression::from(y),
            ArgumentExpression::from(z),
        )))
    }
}

impl TableHandle<Continuous> {
    /// Returns a constant in a continuous table.
    #[inline]
    pub fn element<T>(&self, index: Vec<T>) -> ContinuousExpression
    where
        ElementExpression: From<T>,
    {
        let indices = index.into_iter().map(ElementExpression::from).collect();
        ContinuousExpression::Table(Box::new(NumericTableExpression::Table(self.id(), indices)))
    }

    /// Returns the sum of constants over set expressions in a continuous table.
    #[inline]
    pub fn sum<T>(&self, index: Vec<T>) -> ContinuousExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = index.into_iter().map(ArgumentExpression::from).collect();
        ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
            ReduceOperator::Sum,
            self.id(),
            indices,
        )))
    }

    /// Returns the product of constants over set expressions in a continuous table.
    #[inline]
    pub fn product<T>(&self, index: Vec<T>) -> ContinuousExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = index.into_iter().map(ArgumentExpression::from).collect();
        ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
            ReduceOperator::Product,
            self.id(),
            indices,
        )))
    }

    /// Returns the maximum of constants over set expressions in a continuous table.
    #[inline]
    pub fn max<T>(&self, index: Vec<T>) -> ContinuousExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = index.into_iter().map(ArgumentExpression::from).collect();
        ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
            ReduceOperator::Max,
            self.id(),
            indices,
        )))
    }

    /// Returns the minimum of constants over set expressions in a continuous table.
    #[inline]
    pub fn min<T>(&self, index: Vec<T>) -> ContinuousExpression
    where
        ArgumentExpression: From<T>,
    {
        let indices = index.into_iter().map(ArgumentExpression::from).collect();
        ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
            ReduceOperator::Min,
            self.id(),
            indices,
        )))
    }
}

impl IfThenElse<ContinuousExpression> for Condition {
    #[inline]
    fn if_then_else<U, V>(self, lhs: U, rhs: V) -> ContinuousExpression
    where
        ContinuousExpression: From<U> + From<V>,
    {
        ContinuousExpression::If(
            Box::new(self),
            Box::new(ContinuousExpression::from(lhs)),
            Box::new(ContinuousExpression::from(rhs)),
        )
    }
}

macro_rules! impl_unary_ops {
    ($T:ty) => {
        impl $T {
            /// Returns the absolute value.
            #[inline]
            pub fn abs(self) -> ContinuousExpression {
                ContinuousExpression::from(self).abs()
            }

            /// Returns the square root.
            #[inline]
            pub fn sqrt(self) -> ContinuousExpression {
                ContinuousExpression::from(self).sqrt()
            }

            /// Returns the floor.
            #[inline]
            pub fn floor(self) -> ContinuousExpression {
                ContinuousExpression::from(self).floor()
            }

            /// Returns the ceiling.
            #[inline]
            pub fn ceil(self) -> ContinuousExpression {
                ContinuousExpression::from(self).ceil()
            }

            /// Returns the rounded value.
            #[inline]
            pub fn round(self) -> ContinuousExpression {
                ContinuousExpression::from(self).round()
            }

            /// Returns the truncated value.
            #[inline]
            pub fn trunc(self) -> ContinuousExpression {
                ContinuousExpression::from(self).trunc()
            }
        }

        impl ops::Neg for $T {
            type Output = ContinuousExpression;

            #[inline]
            fn neg(self) -> Self::Output {
                -ContinuousExpression::from(self)
            }
        }
    };
}

macro_rules! impl_binary_ops {
    ($T:ty,$U:ty) => {
        impl ops::Add<$U> for $T {
            type Output = ContinuousExpression;

            #[inline]
            fn add(self, rhs: $U) -> ContinuousExpression {
                ContinuousExpression::from(self) + ContinuousExpression::from(rhs)
            }
        }

        impl ops::Sub<$U> for $T {
            type Output = ContinuousExpression;

            #[inline]
            fn sub(self, rhs: $U) -> ContinuousExpression {
                ContinuousExpression::from(self) - ContinuousExpression::from(rhs)
            }
        }

        impl ops::Mul<$U> for $T {
            type Output = ContinuousExpression;

            #[inline]
            fn mul(self, rhs: $U) -> ContinuousExpression {
                ContinuousExpression::from(self) * ContinuousExpression::from(rhs)
            }
        }

        impl ops::Div<$U> for $T {
            type Output = ContinuousExpression;

            #[inline]
            fn div(self, rhs: $U) -> ContinuousExpression {
                ContinuousExpression::from(self) / ContinuousExpression::from(rhs)
            }
        }

        impl ops::Rem<$U> for $T {
            type Output = ContinuousExpression;

            #[inline]
            fn rem(self, rhs: $U) -> ContinuousExpression {
                ContinuousExpression::from(self) % ContinuousExpression::from(rhs)
            }
        }

        impl MaxMin<$U> for $T {
            type Output = ContinuousExpression;

            #[inline]
            fn max(self, rhs: $U) -> ContinuousExpression {
                ContinuousExpression::from(self).max(ContinuousExpression::from(rhs))
            }

            #[inline]
            fn min(self, rhs: $U) -> ContinuousExpression {
                ContinuousExpression::from(self).min(ContinuousExpression::from(rhs))
            }
        }

        impl ContinuousBinaryOperation<$U> for $T {
            type Output = ContinuousExpression;

            #[inline]
            fn pow(self, rhs: $U) -> ContinuousExpression {
                ContinuousExpression::from(self).pow(ContinuousExpression::from(rhs))
            }

            #[inline]
            fn log(self, rhs: $U) -> ContinuousExpression {
                ContinuousExpression::from(self).log(ContinuousExpression::from(rhs))
            }
        }
    };
}

impl_binary_ops!(ContinuousExpression, Continuous);
impl_binary_ops!(ContinuousExpression, ContinuousVariable);
impl_binary_ops!(ContinuousExpression, ContinuousResourceVariable);
impl_binary_ops!(ContinuousExpression, IntegerExpression);
impl_binary_ops!(ContinuousExpression, Integer);
impl_binary_ops!(ContinuousExpression, IntegerVariable);
impl_binary_ops!(ContinuousExpression, IntegerResourceVariable);
impl_binary_ops!(Continuous, ContinuousExpression);
impl_binary_ops!(Continuous, ContinuousVariable);
impl_binary_ops!(Continuous, ContinuousResourceVariable);
impl_binary_ops!(Continuous, IntegerExpression);
impl_binary_ops!(Continuous, IntegerVariable);
impl_binary_ops!(Continuous, IntegerResourceVariable);
impl_unary_ops!(ContinuousVariable);
impl_binary_ops!(ContinuousVariable, ContinuousExpression);
impl_binary_ops!(ContinuousVariable, Continuous);
impl_binary_ops!(ContinuousVariable, ContinuousVariable);
impl_binary_ops!(ContinuousVariable, ContinuousResourceVariable);
impl_binary_ops!(ContinuousVariable, IntegerExpression);
impl_binary_ops!(ContinuousVariable, IntegerVariable);
impl_binary_ops!(ContinuousVariable, IntegerResourceVariable);
impl_binary_ops!(ContinuousVariable, Integer);
impl_unary_ops!(ContinuousResourceVariable);
impl_binary_ops!(ContinuousResourceVariable, ContinuousExpression);
impl_binary_ops!(ContinuousResourceVariable, Continuous);
impl_binary_ops!(ContinuousResourceVariable, ContinuousVariable);
impl_binary_ops!(ContinuousResourceVariable, ContinuousResourceVariable);
impl_binary_ops!(ContinuousResourceVariable, IntegerExpression);
impl_binary_ops!(ContinuousResourceVariable, IntegerVariable);
impl_binary_ops!(ContinuousResourceVariable, IntegerResourceVariable);
impl_binary_ops!(ContinuousResourceVariable, Integer);
impl_binary_ops!(IntegerExpression, ContinuousExpression);
impl_binary_ops!(IntegerExpression, Continuous);
impl_binary_ops!(IntegerExpression, ContinuousVariable);
impl_binary_ops!(IntegerExpression, ContinuousResourceVariable);
impl_binary_ops!(IntegerVariable, ContinuousExpression);
impl_binary_ops!(IntegerVariable, Continuous);
impl_binary_ops!(IntegerVariable, ContinuousVariable);
impl_binary_ops!(IntegerVariable, ContinuousResourceVariable);
impl_binary_ops!(IntegerResourceVariable, ContinuousExpression);
impl_binary_ops!(IntegerResourceVariable, Continuous);
impl_binary_ops!(IntegerResourceVariable, ContinuousVariable);
impl_binary_ops!(IntegerResourceVariable, ContinuousResourceVariable);
impl_binary_ops!(Integer, ContinuousExpression);
impl_binary_ops!(Integer, ContinuousVariable);
impl_binary_ops!(Integer, ContinuousResourceVariable);

impl ContinuousExpression {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if the cost of the transition state is used or a min/max reduce operation is performed on an empty set or vector.
    #[inline]
    pub fn eval<U: StateInterface>(&self, state: &U, registry: &TableRegistry) -> Continuous {
        self.eval_inner(None, state, registry)
    }

    /// Returns the evaluation result of a cost expression.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    #[inline]
    pub fn eval_cost<U: StateInterface>(
        &self,
        cost: Continuous,
        state: &U,
        registry: &TableRegistry,
    ) -> Continuous {
        self.eval_inner(Some(cost), state, registry)
    }

    fn eval_inner<U: StateInterface>(
        &self,
        cost: Option<Continuous>,
        state: &U,
        registry: &TableRegistry,
    ) -> Continuous {
        match self {
            Self::Constant(x) => *x,
            Self::Variable(i) => state.get_continuous_variable(*i),
            Self::ResourceVariable(i) => state.get_continuous_resource_variable(*i),
            Self::Cost => cost.unwrap(),
            Self::UnaryOperation(op, x) => op.eval(x.eval_inner(cost, state, registry)),
            Self::ContinuousUnaryOperation(op, x) => op.eval(x.eval_inner(cost, state, registry)),
            Self::Round(op, x) => op.eval(x.eval_inner(cost, state, registry)),
            Self::BinaryOperation(op, a, b) => {
                let a = a.eval_inner(cost, state, registry);
                let b = b.eval_inner(cost, state, registry);
                op.eval(a, b)
            }
            Self::ContinuousBinaryOperation(op, a, b) => {
                let a = a.eval_inner(cost, state, registry);
                let b = b.eval_inner(cost, state, registry);
                op.eval(a, b)
            }
            Self::Cardinality(SetExpression::Reference(expression)) => {
                let f = |i| state.get_set_variable(i);
                let set = expression.eval(state, registry, &f, &registry.set_tables);
                set.count_ones(..) as Continuous
            }
            Self::Cardinality(set) => set.eval(state, registry).count_ones(..) as Continuous,
            Self::Table(t) => t.eval(state, registry, &registry.continuous_tables),
            Self::If(condition, x, y) => {
                if condition.eval(state, registry) {
                    x.eval_inner(cost, state, registry)
                } else {
                    y.eval_inner(cost, state, registry)
                }
            }
            Self::FromInteger(x) => Continuous::from(cost.map_or_else(
                || x.eval(state, registry),
                |cost| x.eval_cost(cost as Integer, state, registry),
            )),
            Self::Length(VectorExpression::Reference(expression)) => {
                let f = |i| state.get_vector_variable(i);
                let vector = expression.eval(state, registry, &f, &registry.vector_tables);
                vector.len() as Continuous
            }
            Self::Length(vector) => vector.eval(state, registry).len() as Continuous,
            Self::Last(vector) => match vector.as_ref() {
                ContinuousVectorExpression::Constant(vector) => *vector.last().unwrap(),
                vector => *vector.eval_inner(cost, state, registry).last().unwrap(),
            },
            Self::At(vector, i) => match vector.as_ref() {
                ContinuousVectorExpression::Constant(vector) => vector[i.eval(state, registry)],
                vector => vector.eval_inner(cost, state, registry)[i.eval(state, registry)],
            },
            Self::Reduce(op, vector) => match vector.as_ref() {
                ContinuousVectorExpression::Constant(vector) => op.eval(vector),
                vector => op.eval(&vector.eval_inner(cost, state, registry)),
            },
        }
    }

    /// Returns a simplified version by precomputation.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty set or vector.
    pub fn simplify(&self, registry: &TableRegistry) -> ContinuousExpression {
        match self {
            Self::UnaryOperation(op, x) => match x.simplify(registry) {
                Self::Constant(x) => Self::Constant(op.eval(x)),
                x => Self::UnaryOperation(op.clone(), Box::new(x)),
            },
            Self::ContinuousUnaryOperation(op, x) => match x.simplify(registry) {
                Self::Constant(x) => Self::Constant(op.eval(x)),
                x => Self::ContinuousUnaryOperation(op.clone(), Box::new(x)),
            },
            Self::Round(op, x) => match x.simplify(registry) {
                Self::Constant(x) => Self::Constant(op.eval(x)),
                x => Self::Round(op.clone(), Box::new(x)),
            },
            Self::BinaryOperation(op, a, b) => match (a.simplify(registry), b.simplify(registry)) {
                (Self::Constant(a), Self::Constant(b)) => Self::Constant(op.eval(a, b)),
                (a, b) => Self::BinaryOperation(op.clone(), Box::new(a), Box::new(b)),
            },
            Self::ContinuousBinaryOperation(op, a, b) => {
                match (a.simplify(registry), b.simplify(registry)) {
                    (Self::Constant(a), Self::Constant(b)) => Self::Constant(op.eval(a, b)),
                    (a, b) => Self::ContinuousBinaryOperation(op.clone(), Box::new(a), Box::new(b)),
                }
            }
            Self::Cardinality(expression) => match expression.simplify(registry) {
                SetExpression::Reference(ReferenceExpression::Constant(set)) => {
                    Self::Constant(set.count_ones(..) as Continuous)
                }
                expression => Self::Cardinality(expression),
            },
            Self::Length(expression) => match expression.simplify(registry) {
                VectorExpression::Reference(ReferenceExpression::Constant(vector)) => {
                    Self::Constant(vector.len() as Continuous)
                }
                expression => Self::Length(expression),
            },
            Self::Table(expression) => {
                match expression.simplify(registry, &registry.continuous_tables) {
                    NumericTableExpression::Constant(value) => Self::Constant(value),
                    expression => Self::Table(Box::new(expression)),
                }
            }
            Self::If(condition, x, y) => match condition.simplify(registry) {
                Condition::Constant(true) => x.simplify(registry),
                Condition::Constant(false) => y.simplify(registry),
                condition => Self::If(
                    Box::new(condition),
                    Box::new(x.simplify(registry)),
                    Box::new(y.simplify(registry)),
                ),
            },
            Self::FromInteger(x) => match x.simplify(registry) {
                IntegerExpression::Constant(x) => Self::Constant(x as Continuous),
                x => Self::FromInteger(Box::new(x)),
            },
            Self::Last(vector) => match vector.simplify(registry) {
                ContinuousVectorExpression::Constant(vector) => {
                    Self::Constant(*vector.last().unwrap())
                }
                vector => Self::Last(Box::new(vector)),
            },
            Self::At(vector, i) => match (vector.simplify(registry), i.simplify(registry)) {
                (ContinuousVectorExpression::Constant(vector), ElementExpression::Constant(i)) => {
                    Self::Constant(vector[i])
                }
                (vector, i) => Self::At(Box::new(vector), i),
            },
            Self::Reduce(op, vector) => match vector.simplify(registry) {
                ContinuousVectorExpression::Constant(vector) => Self::Constant(op.eval(&vector)),
                vector => Self::Reduce(op.clone(), Box::new(vector)),
            },
            _ => self.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::table_expression::TableExpression;
    use super::super::table_vector_expression::TableVectorExpression;
    use super::*;
    use crate::state::*;
    use crate::table_data::TableInterface;
    use crate::variable_type::*;
    use rustc_hash::FxHashMap;

    #[test]
    fn default() {
        assert_eq!(
            ContinuousExpression::default(),
            ContinuousExpression::Constant(0.0)
        );
    }

    #[test]
    fn from() {
        assert_eq!(
            ContinuousExpression::from(1),
            ContinuousExpression::Constant(1.0)
        );

        assert_eq!(
            ContinuousExpression::from(1.0),
            ContinuousExpression::Constant(1.0)
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_continuous_variable(String::from("cv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            ContinuousExpression::from(v),
            ContinuousExpression::Variable(v.id())
        );

        let v = metadata.add_continuous_resource_variable(String::from("crv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            ContinuousExpression::from(v),
            ContinuousExpression::ResourceVariable(v.id())
        );

        assert_eq!(
            ContinuousExpression::from(IntegerExpression::Constant(1)),
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Constant(1)))
        );

        let v = metadata.add_integer_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            ContinuousExpression::from(v),
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Variable(v.id())))
        );

        let v = metadata.add_integer_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            ContinuousExpression::from(v),
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::ResourceVariable(
                v.id()
            )))
        );
    }

    #[test]
    fn abs() {
        let expression = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression.abs(),
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(1.0))
            )
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_continuous_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.abs(),
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Variable(0))
            )
        );

        let v = metadata.add_continuous_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.abs(),
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::ResourceVariable(0))
            )
        );
    }

    #[test]
    fn sqrt() {
        let expression = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression.sqrt(),
            ContinuousExpression::ContinuousUnaryOperation(
                ContinuousUnaryOperator::Sqrt,
                Box::new(ContinuousExpression::Constant(1.0))
            )
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_continuous_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.sqrt(),
            ContinuousExpression::ContinuousUnaryOperation(
                ContinuousUnaryOperator::Sqrt,
                Box::new(ContinuousExpression::Variable(0))
            )
        );

        let v = metadata.add_continuous_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.sqrt(),
            ContinuousExpression::ContinuousUnaryOperation(
                ContinuousUnaryOperator::Sqrt,
                Box::new(ContinuousExpression::ResourceVariable(0))
            )
        );
    }

    #[test]
    fn floor() {
        let expression = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression.floor(),
            ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::Constant(1.0))
            )
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_continuous_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.floor(),
            ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::Variable(0))
            )
        );

        let v = metadata.add_continuous_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.floor(),
            ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::ResourceVariable(0))
            )
        );
    }

    #[test]
    fn ceil() {
        let expression = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression.ceil(),
            ContinuousExpression::Round(
                CastOperator::Ceil,
                Box::new(ContinuousExpression::Constant(1.0))
            )
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_continuous_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.ceil(),
            ContinuousExpression::Round(
                CastOperator::Ceil,
                Box::new(ContinuousExpression::Variable(0))
            )
        );

        let v = metadata.add_continuous_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.ceil(),
            ContinuousExpression::Round(
                CastOperator::Ceil,
                Box::new(ContinuousExpression::ResourceVariable(0))
            )
        );
    }

    #[test]
    fn round() {
        let expression = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression.round(),
            ContinuousExpression::Round(
                CastOperator::Round,
                Box::new(ContinuousExpression::Constant(1.0))
            )
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_continuous_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.round(),
            ContinuousExpression::Round(
                CastOperator::Round,
                Box::new(ContinuousExpression::Variable(0))
            )
        );

        let v = metadata.add_continuous_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.round(),
            ContinuousExpression::Round(
                CastOperator::Round,
                Box::new(ContinuousExpression::ResourceVariable(0))
            )
        );
    }

    #[test]
    fn trunc() {
        let expression = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression.trunc(),
            ContinuousExpression::Round(
                CastOperator::Trunc,
                Box::new(ContinuousExpression::Constant(1.0))
            )
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_continuous_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.trunc(),
            ContinuousExpression::Round(
                CastOperator::Trunc,
                Box::new(ContinuousExpression::Variable(0))
            )
        );

        let v = metadata.add_continuous_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.trunc(),
            ContinuousExpression::Round(
                CastOperator::Trunc,
                Box::new(ContinuousExpression::ResourceVariable(0))
            )
        );
    }

    #[test]
    fn neg() {
        let expression = ContinuousExpression::Constant(1.0);
        assert_eq!(
            -expression,
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(ContinuousExpression::Constant(1.0))
            )
        );

        let mut metadata = StateMetadata::default();

        let v = metadata.add_continuous_variable(String::from("iv"));
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            -v,
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(ContinuousExpression::Variable(0))
            )
        );

        let v = metadata.add_continuous_resource_variable(String::from("irv"), true);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            -v,
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Neg,
                Box::new(ContinuousExpression::ResourceVariable(0))
            )
        );
    }

    #[test]
    fn add() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 + 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 + cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 + crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 + iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 + irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1.0 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1.0 + cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1.0 + crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1.0 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            1.0 + iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            1.0 + irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            cv1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 + 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 + cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            cv1 + crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            cv1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            cv1 + iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            cv1 + irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            crv1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 + 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 + cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            crv1 + crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            crv1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            crv1 + iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            crv1 + irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 + 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 + cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 + crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            iv1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 + 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 + cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            iv1 + crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            irv1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 + 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 + cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            irv1 + crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1 + expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1 + cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1 + crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn sub() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 - 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 - cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 - crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 - iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 - irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1.0 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1.0 - cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1.0 - crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1.0 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            1.0 - iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            1.0 - irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            cv1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 - 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 - cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            cv1 - crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            cv1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            cv1 - iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            cv1 - irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            crv1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 - 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 - cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            crv1 - crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            crv1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            crv1 - iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            crv1 - irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 - 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 - cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 - crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            iv1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 - 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 - cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            iv1 - crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            irv1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 - 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 - cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            irv1 - crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1 - expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1 - cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1 - crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn mul() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 * 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 * cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 * crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 * iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 * irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1.0 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1.0 * cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1.0 * crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1.0 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            1.0 * iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            1.0 * irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            cv1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 * 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 * cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            cv1 * crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            cv1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            cv1 * iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            cv1 * irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            crv1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 * 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 * cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            crv1 * crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            crv1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            crv1 * iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            crv1 * irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 * 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 * cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 * crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            iv1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 * 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 * cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            iv1 * crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            irv1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 * 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 * cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            irv1 * crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1 * expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1 * cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1 * crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn div() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 / 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 / cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 / crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 / iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 / irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1.0 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1.0 / cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1.0 / crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1.0 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            1.0 / iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            1.0 / irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            cv1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 / 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 / cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            cv1 / crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            cv1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            cv1 / iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            cv1 / irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            crv1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 / 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 / cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            crv1 / crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            crv1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            crv1 / iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            crv1 / irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 / 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 / cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 / crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            iv1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 / 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 / cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            iv1 / crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            irv1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 / 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 / cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            irv1 / crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1 / expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1 / cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1 / crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn rem() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 % 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 % cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 % crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            expression1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 % iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            expression1 % irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1.0 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1.0 % cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1.0 % crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            1.0 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            1.0 % iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            1.0 % irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            cv1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 % 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            cv1 % cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            cv1 % crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            cv1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            cv1 % iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            cv1 % irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            crv1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 % 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            crv1 % cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            crv1 % crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            crv1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            crv1 % iv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            crv1 % irv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            expression1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 % 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 % cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            expression1 % crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            iv1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 % 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            iv1 % cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            iv1 % crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            irv1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 % 2.0,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            irv1 % cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            irv1 % crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            1 % expression2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            1 % cv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            1 % crv2,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn max() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::max(expression1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::max(expression1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::max(expression1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::max(expression1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::max(expression1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::max(expression1, iv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::max(expression1, irv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::max(1.0, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(1.0, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::max(1.0, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::max(1.0, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            MaxMin::max(1.0, iv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            MaxMin::max(1.0, irv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::max(cv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(cv1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(cv1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::max(cv1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::max(cv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            MaxMin::max(cv1, iv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            MaxMin::max(cv1, irv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::max(crv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(crv1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(crv1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::max(crv1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::max(crv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            MaxMin::max(crv1, iv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            MaxMin::max(crv1, irv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::max(expression1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            MaxMin::max(expression1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            MaxMin::max(expression1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            MaxMin::max(expression1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::max(iv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(iv1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(iv1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::max(iv1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::max(irv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(irv1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(irv1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::max(irv1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::max(1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::max(1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::max(1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn min() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::min(expression1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::min(expression1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::min(expression1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::min(expression1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::min(expression1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::min(expression1, iv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            MaxMin::min(expression1, irv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::min(1.0, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(1.0, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::min(1.0, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::min(1.0, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            MaxMin::min(1.0, iv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            MaxMin::min(1.0, irv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::min(cv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(cv1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(cv1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::min(cv1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::min(cv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            MaxMin::min(cv1, iv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            MaxMin::min(cv1, irv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::min(crv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(crv1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(crv1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::min(crv1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            MaxMin::min(crv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            MaxMin::min(crv1, iv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            MaxMin::min(crv1, irv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::min(expression1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            MaxMin::min(expression1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            MaxMin::min(expression1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            MaxMin::min(expression1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::min(iv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(iv1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(iv1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::min(iv1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::min(irv1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(irv1, 2.0),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(irv1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::min(irv1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            MaxMin::min(1, expression2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            MaxMin::min(1, cv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            MaxMin::min(1, crv2),
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn pow() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, iv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, irv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(1.0, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(1.0, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(1.0, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            ContinuousBinaryOperation::pow(1.0, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(1.0, iv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(1.0, irv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(cv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(cv1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(cv1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(cv1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            ContinuousBinaryOperation::pow(cv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(cv1, iv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(cv1, irv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(crv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(crv1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(crv1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(crv1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            ContinuousBinaryOperation::pow(crv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(crv1, iv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(crv1, irv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            ContinuousBinaryOperation::pow(expression1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(iv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(iv1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(iv1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(iv1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(irv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(irv1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(irv1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(irv1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::pow(1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::pow(1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn log() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, iv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, irv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::log(1.0, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(1.0, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(1.0, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            ContinuousBinaryOperation::log(1.0, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(1.0, iv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(1.0, irv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::log(cv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(cv1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(cv1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(cv1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            ContinuousBinaryOperation::log(cv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(cv1, iv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(cv1, irv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::log(crv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(crv1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(crv1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(crv1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            ContinuousBinaryOperation::log(crv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(crv1, iv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(crv1, irv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            ContinuousBinaryOperation::log(expression1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::log(iv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(iv1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(iv1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(iv1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::log(irv1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(irv1, 2.0),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(irv1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(irv1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            ContinuousBinaryOperation::log(1, expression2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(1, cv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        assert_eq!(
            ContinuousBinaryOperation::log(1, crv2),
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn set_len() {
        let expression = SetExpression::Reference(ReferenceExpression::Constant(Set::default()));
        assert_eq!(
            expression.clone().len_continuous(),
            ContinuousExpression::Cardinality(expression)
        );

        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 10);
        assert!(ob.is_ok());
        let ob = ob.unwrap();
        let v = metadata.add_set_variable(String::from("sv"), ob);
        assert!(v.is_ok());
        let v = v.unwrap();
        assert_eq!(
            v.len_continuous(),
            ContinuousExpression::Cardinality(SetExpression::Reference(
                ReferenceExpression::Variable(v.id())
            ))
        );
    }

    #[test]
    fn table_element() {
        let mut registry = TableRegistry::default();

        let t = registry.add_table_1d(String::from("t1"), vec![0.0, 1.0]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table1DHandle::<Continuous>::element(&t, 0),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table1D(
                t.id(),
                ElementExpression::Constant(0)
            )))
        );
        assert_eq!(
            t.sum(SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
                ReduceOperator::Sum,
                t.id(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.product(SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
                ReduceOperator::Product,
                t.id(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.max(SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
                ReduceOperator::Max,
                t.id(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.min(SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table1DReduce(
                ReduceOperator::Min,
                t.id(),
                SetExpression::default(),
            )))
        );

        let t = registry.add_table_2d(String::from("t2"), vec![vec![0.0, 1.0]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table2DHandle::<Continuous>::element(&t, 0, 0),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.sum_x(SetExpression::default(), 0),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
                ReduceOperator::Sum,
                t.id(),
                SetExpression::default(),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.sum_y(0, SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
                ReduceOperator::Sum,
                t.id(),
                ElementExpression::Constant(0),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.sum(SetExpression::default(), SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
                ReduceOperator::Sum,
                t.id(),
                SetExpression::default(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.product_x(SetExpression::default(), 0),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
                ReduceOperator::Product,
                t.id(),
                SetExpression::default(),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.product_y(0, SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
                ReduceOperator::Product,
                t.id(),
                ElementExpression::Constant(0),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.product(SetExpression::default(), SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
                ReduceOperator::Product,
                t.id(),
                SetExpression::default(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.max_x(SetExpression::default(), 0),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
                ReduceOperator::Max,
                t.id(),
                SetExpression::default(),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.max_y(0, SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
                ReduceOperator::Max,
                t.id(),
                ElementExpression::Constant(0),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.max(SetExpression::default(), SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
                ReduceOperator::Max,
                t.id(),
                SetExpression::default(),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.min_x(SetExpression::default(), 0),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceX(
                ReduceOperator::Min,
                t.id(),
                SetExpression::default(),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            t.min_y(0, SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduceY(
                ReduceOperator::Min,
                t.id(),
                ElementExpression::Constant(0),
                SetExpression::default(),
            )))
        );
        assert_eq!(
            t.min(SetExpression::default(), SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table2DReduce(
                ReduceOperator::Min,
                t.id(),
                SetExpression::default(),
                SetExpression::default(),
            )))
        );

        let t = registry.add_table_3d(String::from("t3"), vec![vec![vec![0.0, 1.0]]]);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            Table3DHandle::<Continuous>::element(&t, 0, 0, 0),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table3D(
                t.id(),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )))
        );
        assert_eq!(
            Table3DHandle::<Continuous>::sum(&t, 0, 0, SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
                ReduceOperator::Sum,
                t.id(),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::default()),
            )))
        );
        assert_eq!(
            Table3DHandle::<Continuous>::product(&t, 0, 0, SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
                ReduceOperator::Product,
                t.id(),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::default()),
            )))
        );
        assert_eq!(
            Table3DHandle::<Continuous>::max(&t, 0, 0, SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
                ReduceOperator::Max,
                t.id(),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::default()),
            )))
        );
        assert_eq!(
            Table3DHandle::<Continuous>::min(&t, 0, 0, SetExpression::default()),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table3DReduce(
                ReduceOperator::Min,
                t.id(),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Element(ElementExpression::Constant(0)),
                ArgumentExpression::Set(SetExpression::default()),
            )))
        );

        let mut map = FxHashMap::default();
        map.insert(vec![0, 0, 0, 0], 1.0);
        let t = registry.add_table(String::from("t"), map, 0.0);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(
            TableHandle::<Continuous>::element(&t, vec![0, 0, 0, 0]),
            ContinuousExpression::Table(Box::new(NumericTableExpression::Table(
                t.id(),
                vec![
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(0),
                ],
            )))
        );
        assert_eq!(
            TableHandle::<Continuous>::sum(
                &t,
                vec![
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(Set::default())
                ]
            ),
            ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Sum,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::default()),
                ]
            )))
        );
        assert_eq!(
            TableHandle::<Continuous>::product(
                &t,
                vec![
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(Set::default())
                ]
            ),
            ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Product,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::default()),
                ]
            )))
        );
        assert_eq!(
            TableHandle::<Continuous>::max(
                &t,
                vec![
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(Set::default())
                ]
            ),
            ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Max,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::default()),
                ]
            )))
        );
        assert_eq!(
            TableHandle::<Continuous>::min(
                &t,
                vec![
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(0),
                    ArgumentExpression::from(Set::default())
                ]
            ),
            ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Min,
                t.id(),
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Set(SetExpression::default()),
                ]
            )))
        );
    }

    #[test]
    fn if_then_else() {
        let mut metadata = StateMetadata::default();
        let iv1 = metadata.add_integer_variable(String::from("iv1"));
        assert!(iv1.is_ok());
        let iv1 = iv1.unwrap();
        let iv2 = metadata.add_integer_variable(String::from("iv2"));
        assert!(iv2.is_ok());
        let iv2 = iv2.unwrap();
        let irv1 = metadata.add_integer_resource_variable(String::from("irv1"), true);
        assert!(irv1.is_ok());
        let irv1 = irv1.unwrap();
        let irv2 = metadata.add_integer_resource_variable(String::from("irv2"), false);
        assert!(irv2.is_ok());
        let irv2 = irv2.unwrap();
        let cv1 = metadata.add_continuous_variable(String::from("cv1"));
        assert!(cv1.is_ok());
        let cv1 = cv1.unwrap();
        let cv2 = metadata.add_continuous_variable(String::from("cv2"));
        assert!(cv2.is_ok());
        let cv2 = cv2.unwrap();
        let crv1 = metadata.add_continuous_resource_variable(String::from("crv1"), true);
        assert!(crv1.is_ok());
        let crv1 = crv1.unwrap();
        let crv2 = metadata.add_continuous_resource_variable(String::from("crv2"), false);
        assert!(crv2.is_ok());
        let crv2 = crv2.unwrap();

        let condition = Condition::Constant(true);
        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, 2.0),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, cv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, crv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = ContinuousExpression::Constant(1.0);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, iv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = ContinuousExpression::Constant(1.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, irv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, 1.0, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, 1.0, cv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, 1.0, crv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, 1.0, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, 1.0, iv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, 1.0, irv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, cv1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, cv1, 2.0),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, cv1, cv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, cv1, crv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, cv1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, cv1, iv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, cv1, irv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Variable(cv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, crv1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, crv1, 2.0),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, crv1, cv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, crv1, crv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = IntegerExpression::Constant(2);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, crv1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(2)
                )))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, crv1, iv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv2.id())
                )))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, crv1, irv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::ResourceVariable(crv1.id())),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv2.id())
                )))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = IntegerExpression::Constant(1);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, 2.0),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, cv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression1 = IntegerExpression::Constant(1);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, expression1, crv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Constant(1)
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, iv1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, iv1, 2.0),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, iv1, cv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, iv1, crv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::Variable(iv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, irv1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, irv1, 2.0),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, irv1, cv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, irv1, crv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::FromInteger(Box::new(
                    IntegerExpression::ResourceVariable(irv1.id())
                ))),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        let expression2 = ContinuousExpression::Constant(2.0);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, 1, expression2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, 1, cv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::Variable(cv2.id()))
            )
        );

        let condition = Condition::Constant(true);
        assert_eq!(
            IfThenElse::<ContinuousExpression>::if_then_else(condition, 1, crv2),
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(1.0)),
                Box::new(ContinuousExpression::ResourceVariable(crv2.id()))
            )
        );
    }

    #[test]
    fn constant_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Constant(1.0);
        assert_eq!(expression.eval(&state, &registry), 1.0);
    }

    #[test]
    fn variable_eval() {
        let state = State {
            signature_variables: SignatureVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Variable(0);
        assert_eq!(expression.eval(&state, &registry), 0.0);
    }

    #[test]
    fn resource_variable_eval() {
        let state = State {
            resource_variables: ResourceVariables {
                continuous_variables: vec![0.0],
                ..Default::default()
            },
            ..Default::default()
        };
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::ResourceVariable(0);
        assert_eq!(expression.eval(&state, &registry), 0.0);
    }

    #[test]
    fn eval_cost() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Cost;
        assert_eq!(expression.eval_cost(10.0, &state, &registry), 10.0);
    }

    #[test]
    #[should_panic]
    fn eval_cost_panic() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Cost;
        expression.eval(&state, &registry);
    }

    #[test]
    fn unary_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousExpression::Constant(-1.0)),
        );
        assert_eq!(expression.eval(&state, &registry), 1.0);
    }

    #[test]
    fn continuous_unary_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousExpression::Constant(4.0)),
        );
        assert_eq!(expression.eval(&state, &registry), 2.0);
    }

    #[test]
    fn round_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Round(
            CastOperator::Floor,
            Box::new(ContinuousExpression::Constant(2.5)),
        );
        assert_eq!(expression.eval(&state, &registry), 2.0);
    }

    #[test]
    fn binary_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::Constant(1.0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(expression.eval(&state, &registry), 3.0);
    }

    #[test]
    fn continuous_binary_operation_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousExpression::Constant(2.0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(expression.eval(&state, &registry), 4.0);
    }

    #[test]
    fn cardinality_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let mut set = Set::with_capacity(5);
        set.insert(1);
        set.insert(4);
        let expression = ContinuousExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        ));
        assert_eq!(expression.eval(&state, &registry), 2.0);
    }

    #[test]
    fn length_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 4]),
        ));
        assert_eq!(expression.eval(&state, &registry), 2.0);
    }

    #[test]
    fn table_eval() {
        let state = State::default();
        let registry = TableRegistry::default();
        let expression =
            ContinuousExpression::Table(Box::new(NumericTableExpression::Constant(0.0)));
        assert_eq!(expression.eval(&state, &registry), 0.0);
    }

    #[test]
    fn if_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression = ContinuousExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousExpression::Constant(1.0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(expression.eval(&state, &registry), 1.0);

        let expression = ContinuousExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(ContinuousExpression::Constant(1.0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(expression.eval(&state, &registry), 2.0);
    }

    #[test]
    fn from_integer_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression =
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Constant(1)));
        assert_eq!(expression.eval(&state, &registry), 1.0);
    }

    #[test]
    fn last_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression =
            ContinuousExpression::Last(Box::new(ContinuousVectorExpression::Constant(vec![
                1.0, 2.0, 3.0,
            ])));
        assert_eq!(expression.eval(&state, &registry), 3.0);
    }

    #[test]
    fn at_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression = ContinuousExpression::At(
            Box::new(ContinuousVectorExpression::Constant(vec![1.0, 2.0, 3.0])),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.eval(&state, &registry), 1.0);
    }

    #[test]
    fn reduce_eval() {
        let state = State::default();
        let registry = TableRegistry::default();

        let expression = ContinuousExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(ContinuousVectorExpression::Constant(vec![1.0, 2.0, 3.0])),
        );
        assert_eq!(expression.eval(&state, &registry), 6.0);
    }

    #[test]
    fn constant_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Constant(1.0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn variable_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Variable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn resource_variable_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::ResourceVariable(0);
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn cost_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Cost;
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn unary_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousExpression::Constant(-1.0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(1.0)
        );
        let expression = ContinuousExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_unary_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousExpression::Constant(4.0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(2.0)
        );
        let expression = ContinuousExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn round_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Round(
            CastOperator::Floor,
            Box::new(ContinuousExpression::Constant(1.5)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(1.0)
        );
        let expression = ContinuousExpression::Round(
            CastOperator::Floor,
            Box::new(ContinuousExpression::Variable(0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn binary_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::Constant(1.0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(3.0)
        );
        let expression = ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn continuous_binary_operation_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousExpression::Constant(2.0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(4.0)
        );
        let expression = ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(ContinuousExpression::Variable(0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn cardinality_simplify() {
        let registry = TableRegistry::default();
        let mut set = Set::with_capacity(5);
        set.insert(1);
        set.insert(4);
        let expression = ContinuousExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Constant(set),
        ));
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(2.0)
        );
        let expression = ContinuousExpression::Cardinality(SetExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn length_simplify() {
        let registry = TableRegistry::default();
        let expression = ContinuousExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Constant(vec![1, 4]),
        ));
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(2.0)
        );
        let expression = ContinuousExpression::Length(VectorExpression::Reference(
            ReferenceExpression::Variable(0),
        ));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn table_simplify() {
        let registry = TableRegistry::default();
        let expression =
            ContinuousExpression::Table(Box::new(NumericTableExpression::Constant(0.0)));
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(0.0)
        );
        let expression = ContinuousExpression::Table(Box::new(NumericTableExpression::Table1D(
            0,
            ElementExpression::Variable(0),
        )));
        assert_eq!(expression.simplify(&registry), expression,);
    }

    #[test]
    fn if_simplify() {
        let registry = TableRegistry::default();

        let expression = ContinuousExpression::If(
            Box::new(Condition::Constant(true)),
            Box::new(ContinuousExpression::Constant(1.0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(1.0)
        );

        let expression = ContinuousExpression::If(
            Box::new(Condition::Constant(false)),
            Box::new(ContinuousExpression::Constant(1.0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(2.0)
        );

        let expression = ContinuousExpression::If(
            Box::new(Condition::Table(Box::new(TableExpression::Table1D(
                0,
                ElementExpression::Variable(0),
            )))),
            Box::new(ContinuousExpression::Constant(1.0)),
            Box::new(ContinuousExpression::Constant(2.0)),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn from_integer_simplify() {
        let registry = TableRegistry::default();

        let expression =
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Constant(1)));
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(1.0)
        );

        let expression =
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Variable(0)));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn last_simplify() {
        let registry = TableRegistry::default();

        let expression =
            ContinuousExpression::Last(Box::new(ContinuousVectorExpression::Constant(vec![
                1.0, 2.0, 3.0,
            ])));
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(3.0)
        );

        let expression = ContinuousExpression::Last(Box::new(ContinuousVectorExpression::Table(
            Box::new(TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
            )),
        )));
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn at_simplify() {
        let registry = TableRegistry::default();

        let expression = ContinuousExpression::At(
            Box::new(ContinuousVectorExpression::Constant(vec![1.0, 2.0, 3.0])),
            ElementExpression::Constant(0),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(1.0)
        );

        let expression = ContinuousExpression::At(
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ),
            ))),
            ElementExpression::Constant(0),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }

    #[test]
    fn reduce_simplify() {
        let registry = TableRegistry::default();

        let expression = ContinuousExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(ContinuousVectorExpression::Constant(vec![1.0, 2.0, 3.0])),
        );
        assert_eq!(
            expression.simplify(&registry),
            ContinuousExpression::Constant(6.0)
        );

        let expression = ContinuousExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(ContinuousVectorExpression::Table(Box::new(
                TableVectorExpression::Table1D(
                    0,
                    VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ),
            ))),
        );
        assert_eq!(expression.simplify(&registry), expression);
    }
}
