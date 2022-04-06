use std::default;
use std::fmt;
use std::iter::{Product, Sum};
use std::str;

pub type Set = fixedbitset::FixedBitSet;
pub type Vector = Vec<usize>;
pub type Element = usize;
pub type Integer = i32;
pub type Continuous = f64;
pub type OrderedContinuous = ordered_float::OrderedFloat<Continuous>;

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

pub trait ToNumeric {
    fn to_integer(self) -> Integer;
    fn to_continuous(self) -> Continuous;
}

pub trait FromNumeric {
    fn from_integer(n: Integer) -> Self;
    fn from_continuous(n: Continuous) -> Self;
    fn from_usize(n: usize) -> Self;
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
