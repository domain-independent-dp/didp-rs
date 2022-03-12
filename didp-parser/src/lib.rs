use std::collections;
use std::error::Error;
use yaml_rust::Yaml;

pub mod expression;
pub mod expression_parser;
pub mod numeric_function;
pub mod operator;
pub mod problem;
pub mod state;
pub mod variable;
pub mod yaml_util;

pub struct Problem<'a, T: variable::Numeric> {
    pub domain_name: String,
    pub problem_name: String,
    pub state_metadata: state::StateMetadata,
    pub function_registry: FunctionRegistry<T>,
    pub initial_state: state::State<T>,
    pub goal_families: Vec<Vec<expression::Condition<'a, T>>>,
    pub constraint_families: Vec<Vec<expression::Condition<'a, T>>>,
    pub operator_families: Vec<Vec<operator::Operator<'a, T>>>,
}

impl<'a, T: variable::Numeric> Problem<'a, T> {
    pub fn new(domain: &str, problem: &str) -> Result<Problem<'a, T>, Box<dyn Error>> {
        let domain = Yaml::from_str(domain);
        let domain = yaml_util::get_hash(&domain)?;
        let domain_name = yaml_util::get_string_value(domain, "domain")?;

        let problem = Yaml::from_str(problem);
        let problem = yaml_util::get_hash(&problem)?;
        let problem_name = yaml_util::get_string_value(problem, "problem")?;

        let state_metadata = state::StateMetadata::new(&domain, &problem)?;

        Err(yaml_util::YamlContentErr::new("not implemented".to_string()).into())
    }
}

pub struct FunctionRegistry<T: variable::Numeric> {
    pub functions_1d: collections::HashMap<String, numeric_function::NumericFunction1D<T>>,
    pub functions_2d: collections::HashMap<String, numeric_function::NumericFunction2D<T>>,
    pub functions_3d: collections::HashMap<String, numeric_function::NumericFunction3D<T>>,
    pub functions: collections::HashMap<String, numeric_function::NumericFunction<T>>,
}
