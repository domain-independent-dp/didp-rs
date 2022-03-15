use std::error::Error;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

pub mod expression;
pub mod expression_parser;
pub mod function_registry;
pub mod numeric_function;
pub mod operator;
pub mod problem;
pub mod state;
pub mod variable;
pub mod yaml_util;

pub struct Problem<'a, T: variable::Numeric> {
    pub minimize: bool,
    pub domain_name: String,
    pub problem_name: String,
    pub state_metadata: state::StateMetadata,
    pub initial_state: state::State<T>,
    pub function_registry: function_registry::FunctionRegistry<T>,
    pub goal_families: Vec<Vec<expression::Condition<'a, T>>>,
    pub constraint_families: Vec<Vec<expression::Condition<'a, T>>>,
    pub operator_families: Vec<Vec<operator::Operator<'a, T>>>,
}

impl<'a, T: variable::Numeric> Problem<'a, T> {
    pub fn load_from_yaml(domain: &Yaml, problem: &Yaml) -> Result<Problem<'a, T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let state_metadata = state::StateMetadata::load_from_yaml(&domain, &problem)?;

        let domain = yaml_util::get_map(&domain)?;
        let minimize = match yaml_util::get_string_by_key(&domain, "metric") {
            Ok(value) => match &value[..] {
                "minimize" => true,
                "maximize" => false,
                _ => {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "expected `minimize` or `maximize`, but is {}",
                        value
                    ))
                    .into())
                }
            },
            Err(_) => true,
        };
        let domain_name = yaml_util::get_string_by_key(&domain, "domain")?;
        let problem = yaml_util::get_map(&problem)?;
        let domain_name2 = yaml_util::get_string_by_key(&problem, "domain")?;
        if domain_name != domain_name2 {
            return Err(yaml_util::YamlContentErr::new(format!(
                "domain mismatch: expected `{}`, but is `{}`",
                domain_name, domain_name2
            ))
            .into());
        }
        let problem_name = yaml_util::get_string_by_key(&problem, "problem")?;

        let initial_state = yaml_util::get_yaml_by_key(&problem, "initial_state")?;
        let initial_state =
            state::State::<T>::load_initial_state_from_yaml(initial_state, &state_metadata)?;

        let function_domain = yaml_util::get_yaml_by_key(&domain, "functions")?;
        let function_problem = yaml_util::get_yaml_by_key(&problem, "functions")?;
        let function_registry = function_registry::FunctionRegistry::<T>::load_from_yaml(
            &function_domain,
            &function_problem,
            &state_metadata,
        );

        Err(yaml_util::YamlContentErr::new("not implemented".to_string()).into())
    }
}
