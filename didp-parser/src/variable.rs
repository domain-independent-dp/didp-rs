use std::default;
use std::fmt;
use std::hash::Hash;
use std::iter::Sum;
use std::str;

pub type SetVariable = fixedbitset::FixedBitSet;
pub type PermutationVariable = Vec<usize>;
pub type ElementVariable = usize;
pub type IntegerVariable = i32;
pub type ContinuousVariable = ordered_float::OrderedFloat<f64>;

pub trait Numeric:
    num_traits::Num
    + num_traits::cast::NumCast
    + Ord
    + Hash
    + Copy
    + Sum
    + str::FromStr
    + fmt::Debug
    + default::Default
{
}

impl Numeric for IntegerVariable {}
impl Numeric for ContinuousVariable {}
