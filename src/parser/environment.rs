use std::collections;

use crate::numeric_function;
use crate::variable;

pub struct Environment<T: variable::Numeric> {
    pub set_variables: collections::HashMap<String, usize>,
    pub permutation_variables: collections::HashMap<String, usize>,
    pub element_variables: collections::HashMap<String, usize>,
    pub numeric_variables: collections::HashMap<String, usize>,
    pub resource_variables: collections::HashMap<String, usize>,
    pub functions_1d: collections::HashMap<String, numeric_function::NumericFunction1D<T>>,
    pub functions_2d: collections::HashMap<String, numeric_function::NumericFunction2D<T>>,
    pub functions_3d: collections::HashMap<String, numeric_function::NumericFunction3D<T>>,
    pub functions: collections::HashMap<String, numeric_function::NumericFunction<T>>,
}
