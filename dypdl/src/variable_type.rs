//! A module defining types of values of state variables.

use std::default;
use std::fmt;
use std::iter::{Product, Sum};
use std::str;

/// Set value.
pub type Set = fixedbitset::FixedBitSet;
/// Vector value.
pub type Vector = Vec<usize>;
/// Element value.
pub type Element = usize;
/// Integer numeric value.
pub type Integer = i32;
/// Continuous numeric value.
pub type Continuous = f64;
/// Continuous numeric value with a total order.
pub type OrderedContinuous = ordered_float::OrderedFloat<Continuous>;

/// Numeric value.
pub trait Numeric:
    num_traits::Num
    + ToNumeric
    + FromNumeric
    + num_traits::FromPrimitive
    + num_traits::Signed
    + Copy
    + Sum
    + Product
    + PartialOrd
    + str::FromStr
    + fmt::Debug
    + default::Default
{
}

impl Numeric for Integer {}
impl Numeric for Continuous {}
impl Numeric for OrderedContinuous {}

/// Trait for converting to numeric values.
pub trait ToNumeric {
    /// Convert to an integer value.
    fn to_integer(self) -> Integer;
    /// Convert to a continuous value.
    fn to_continuous(self) -> Continuous;
}

/// Trait for converting from numeric values.
pub trait FromNumeric {
    /// Convert from an integer value.
    fn from_integer(n: Integer) -> Self;
    /// Convert from a continuos value.
    fn from_continuous(n: Continuous) -> Self;
    /// Convert from usize.
    fn from_usize(n: usize) -> Self;
    /// Convert from value that can be converted to a numeric value.
    fn from<T: ToNumeric>(n: T) -> Self;
}

impl ToNumeric for Integer {
    #[inline]
    fn to_integer(self) -> Integer {
        self
    }

    #[inline]
    fn to_continuous(self) -> Continuous {
        self as Continuous
    }
}

impl FromNumeric for Integer {
    #[inline]
    fn from_integer(n: Integer) -> Integer {
        n
    }

    #[inline]
    fn from_continuous(n: Continuous) -> Integer {
        n as Integer
    }

    #[inline]
    fn from_usize(n: usize) -> Integer {
        n as Integer
    }

    #[inline]
    fn from<T: ToNumeric>(n: T) -> Integer {
        n.to_integer()
    }
}

impl ToNumeric for Continuous {
    #[inline]
    fn to_integer(self) -> Integer {
        self as Integer
    }

    #[inline]
    fn to_continuous(self) -> Continuous {
        self
    }
}

impl FromNumeric for Continuous {
    #[inline]
    fn from_integer(n: Integer) -> Continuous {
        n as Continuous
    }

    #[inline]
    fn from_continuous(n: Continuous) -> Continuous {
        n
    }

    #[inline]
    fn from_usize(n: usize) -> Continuous {
        n as Continuous
    }

    #[inline]
    fn from<T: ToNumeric>(n: T) -> Continuous {
        n.to_continuous()
    }
}

impl ToNumeric for OrderedContinuous {
    #[inline]
    fn to_integer(self) -> Integer {
        self.to_continuous() as Integer
    }

    #[inline]
    fn to_continuous(self) -> Continuous {
        self.into_inner()
    }
}

impl FromNumeric for OrderedContinuous {
    #[inline]
    fn from_integer(n: Integer) -> OrderedContinuous {
        ordered_float::OrderedFloat(n as Continuous)
    }

    #[inline]
    fn from_continuous(n: Continuous) -> OrderedContinuous {
        ordered_float::OrderedFloat(n)
    }

    #[inline]
    fn from_usize(n: usize) -> OrderedContinuous {
        ordered_float::OrderedFloat(n as Continuous)
    }

    #[inline]
    fn from<T: ToNumeric>(n: T) -> OrderedContinuous {
        ordered_float::OrderedFloat(n.to_continuous())
    }
}
