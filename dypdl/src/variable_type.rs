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

/// Trait for string representation of variables, since the ToString trait for Set outputs binary
/// bits, which is hard to read, ToVariableString overrides it with readable string representations.
pub trait ToVariableString {
    fn to_variable_string(&self) -> String;
}

impl ToVariableString for Set {
    fn to_variable_string(&self) -> String {
        let debug_string = format!("{:?}", self.ones().collect::<Vec<usize>>()).replace(',', "");
        let len = debug_string.len();
        format!("{{{} : {}}}", &debug_string[1..(len - 1)], self.len())
    }
}

macro_rules! create_default_ToVariableString {
    ($t:ty) => {
        impl ToVariableString for $t {
            fn to_variable_string(&self) -> String {
                self.to_string()
            }
        }
    };
}

create_default_ToVariableString!(Element);
create_default_ToVariableString!(Integer);
create_default_ToVariableString!(Continuous);
create_default_ToVariableString!(OrderedContinuous);
create_default_ToVariableString!(bool);

/// Numeric value.
pub trait Numeric:
    num_traits::Num
    + ToNumeric
    + FromNumeric
    + num_traits::FromPrimitive
    + num_traits::Signed
    + num_traits::Bounded
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

#[cfg(test)]
mod tests {
    use crate::{
        variable_type::{OrderedContinuous, ToVariableString},
        Continuous, Element, Integer, Set,
    };

    #[test]
    fn integer_to_variable_string() {
        assert_eq!((10 as Integer).to_variable_string(), "10".to_owned());
    }

    #[test]
    fn float_to_variable_string() {
        assert_eq!((3.3 as Continuous).to_variable_string(), "3.3".to_owned());
    }

    #[test]
    fn bool_to_variable_string() {
        assert_eq!((false).to_variable_string(), "false".to_owned());
    }

    #[test]
    fn element_to_variable_string() {
        assert_eq!((10 as Element).to_variable_string(), "10".to_owned());
    }

    #[test]
    fn ordered_continuous_to_variable_string() {
        assert_eq!(
            OrderedContinuous::from(3.3).to_variable_string(),
            "3.3".to_owned()
        );
    }

    #[test]
    fn set_to_variable_string() {
        let mut set = Set::with_capacity(10);
        set.insert(1);
        set.insert(3);

        assert_eq!(set.to_variable_string(), "{1 3 : 10}".to_owned());
    }
}
