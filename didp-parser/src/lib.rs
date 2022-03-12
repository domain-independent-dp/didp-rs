use std::collections;
use yaml_rust::Yaml;

pub mod errors;
pub mod expression;
pub mod expression_parser;
pub mod numeric_function;
pub mod operator;
pub mod problem;
pub mod state;
pub mod variable;

pub use errors::ProblemErr;

pub struct Problem<'a, T: variable::Numeric> {
    pub domain_name: String,
    pub problem_name: String,
    pub state_metadata: state::StateMetadata,
    pub object_numbers: Vec<usize>,
    pub function_registry: FunctionRegistry<T>,
    pub initial_state: state::State<T>,
    pub goal_families: Vec<Vec<expression::Condition<'a, T>>>,
    pub constraint_families: Vec<Vec<expression::Condition<'a, T>>>,
    pub operator_families: Vec<Vec<operator::Operator<'a, T>>>,
}

impl<'a, T: variable::Numeric> Problem<'a, T> {
    pub fn new(domain: &str, problem: &str) -> Result<Problem<'a, T>, ProblemErr> {
        let domain = Yaml::from_str(domain);
        let domain = match domain {
            Yaml::Hash(map) => map,
            _ => {
                return Err(ProblemErr::Reason(format!(
                    "expected Yaml::Hash for the domain yaml, but was {:?}",
                    domain
                )))
            }
        };
        let domain_name = match domain.get(&Yaml::String("domain".to_string())) {
            Some(Yaml::String(name)) => name,
            Some(_) => {
                return Err(ProblemErr::Reason(
                    "key `domain` found in the domain yaml but not String".to_string(),
                ))
            }
            None => {
                return Err(ProblemErr::Reason(
                    "key `domain` not found in the domain yaml".to_string(),
                ))
            }
        };

        let problem = Yaml::from_str(problem);
        let problem = match problem {
            Yaml::Hash(map) => map,
            _ => {
                return Err(ProblemErr::Reason(format!(
                    "expected Yaml::Hash for the domain texts, but was {:?}",
                    problem
                )))
            }
        };

        Err(ProblemErr::Reason("aaa".to_string()))
    }
}

pub struct FunctionRegistry<T: variable::Numeric> {
    pub functions_1d: collections::HashMap<String, numeric_function::NumericFunction1D<T>>,
    pub functions_2d: collections::HashMap<String, numeric_function::NumericFunction2D<T>>,
    pub functions_3d: collections::HashMap<String, numeric_function::NumericFunction3D<T>>,
    pub functions: collections::HashMap<String, numeric_function::NumericFunction<T>>,
}
