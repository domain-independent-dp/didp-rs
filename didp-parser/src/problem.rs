use crate::expression;

use crate::numeric_function;
use crate::operator;
use crate::state;
use crate::variable;
use std::collections;

pub struct Problem {
    pub set_variable_to_max_size: Vec<usize>,
    pub permutation_variable_to_max_length: Vec<usize>,
    pub element_to_set: Vec<usize>,
}

pub struct Problem_<'a, T: variable::Numeric> {
    pub domain_name: String,
    pub problem_name: String,
    pub variable_metadata: state::StateMetadata,
    pub function_registry: FunctionRegistry<T>,
    pub initial_state: state::State<T>,
    pub goal_families: Vec<Vec<expression::Condition<'a, T>>>,
    pub constraint_families: Vec<Vec<expression::Condition<'a, T>>>,
    pub operator_families: Vec<Vec<operator::Operator<'a, T>>>,
}

pub struct FunctionRegistry<T: variable::Numeric> {
    pub functions_1d: collections::HashMap<String, numeric_function::NumericFunction1D<T>>,
    pub functions_2d: collections::HashMap<String, numeric_function::NumericFunction2D<T>>,
    pub functions_3d: collections::HashMap<String, numeric_function::NumericFunction3D<T>>,
    pub functions: collections::HashMap<String, numeric_function::NumericFunction<T>>,
}
