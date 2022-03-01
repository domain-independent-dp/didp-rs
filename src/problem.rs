use std::collections;

use crate::numeric_function;
use crate::variable;

pub struct Problem<T: variable::Numeric> {
    pub set_variable_to_max_size: Vec<usize>,
    pub permutation_variable_to_max_length: Vec<usize>,
    pub element_to_set: Vec<usize>,
    pub functions_1d: collections::HashMap<String, numeric_function::NumericFunction1D<T>>,
    pub functions_2d: collections::HashMap<String, numeric_function::NumericFunction2D<T>>,
    pub functions_3d: collections::HashMap<String, numeric_function::NumericFunction3D<T>>,
    pub functions: collections::HashMap<String, numeric_function::NumericFunction<T>>,
}
