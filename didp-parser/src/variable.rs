use std::default;
use std::fmt;
use std::hash::Hash;
use std::iter::Sum;
use std::str;

pub type Set = fixedbitset::FixedBitSet;
pub type Permutation = Vec<usize>;
pub type Element = usize;
pub type Integer = i32;
pub type Continuous = ordered_float::OrderedFloat<f64>;

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

impl Numeric for Integer {}
impl Numeric for Continuous {}
