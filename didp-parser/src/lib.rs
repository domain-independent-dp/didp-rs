use std::error::Error;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

pub mod expression;
pub mod expression_parser;
pub mod grounded_condition;
pub mod operator;
pub mod state;
pub mod table;
pub mod table_registry;
pub mod variable;
pub mod yaml_util;

#[derive(Debug)]
pub struct ProblemErr(String);

impl ProblemErr {
    pub fn new(message: String) -> ProblemErr {
        ProblemErr(format!("Error in problem definiton: {}", message))
    }
}

impl fmt::Display for ProblemErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ProblemErr {}

pub enum NumericType {
    Integer,
    Continuous,
}

impl NumericType {
    pub fn load_from_yaml(value: &Yaml) -> Result<NumericType, Box<dyn Error>> {
        let map = yaml_util::get_map(value)?;
        let numeric_type = yaml_util::get_string_by_key(&map, "numeric_type")?;
        match &numeric_type[..] {
            "integer" => Ok(Self::Integer),
            "continuous" => Ok(Self::Continuous),
            _ => Err(ProblemErr::new(format!("no such numeric type `{}`", numeric_type)).into()),
        }
    }
}

pub struct Problem<T: variable::Numeric> {
    pub minimize: bool,
    pub domain_name: String,
    pub problem_name: String,
    pub state_metadata: state::StateMetadata,
    pub initial_state: state::State<T>,
    pub table_registry: table_registry::TableRegistry<T>,
    pub constraints: Vec<grounded_condition::GroundedCondition<T>>,
    pub goals: Vec<grounded_condition::GroundedCondition<T>>,
    pub operators: Vec<operator::Operator<T>>,
}

impl<T: variable::Numeric> Problem<T> {
    pub fn load_from_yaml(domain: &Yaml, problem: &Yaml) -> Result<Problem<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
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

        let objects = yaml_util::get_yaml_by_key(&domain, "objects")?;
        let variables = yaml_util::get_yaml_by_key(&domain, "variables")?;
        let object_numbers = yaml_util::get_yaml_by_key(&domain, "object_numbers")?;
        let state_metadata =
            state::StateMetadata::load_from_yaml(objects, variables, object_numbers)?;

        let initial_state = yaml_util::get_yaml_by_key(&problem, "initial_state")?;
        let initial_state = state::State::<T>::load_from_yaml(initial_state, &state_metadata)?;

        let tables = yaml_util::get_yaml_by_key(&domain, "tables")?;
        let table_values = yaml_util::get_yaml_by_key(&problem, "table_values")?;
        let table_registry = table_registry::TableRegistry::<T>::load_from_yaml(
            &tables,
            &table_values,
            &state_metadata,
        )?;

        let mut constraints = Vec::new();
        for constraint in yaml_util::get_array_by_key(&domain, "constraints")? {
            let conditions = grounded_condition::load_grounded_conditions_from_yaml(
                &constraint,
                &state_metadata,
                &table_registry,
            )?;
            let conditions = Self::filiter_grounded_conditions(conditions)?;
            constraints.extend(conditions);
        }

        let mut goals = Vec::new();
        for goal in yaml_util::get_array_by_key(&problem, "goals")? {
            let conditions = grounded_condition::load_grounded_conditions_from_yaml(
                &goal,
                &state_metadata,
                &table_registry,
            )?;
            let conditions = Self::filiter_grounded_conditions(conditions)?;
            goals.extend(conditions);
        }

        let mut operators = Vec::new();
        for operator in yaml_util::get_array_by_key(&problem, "operators")? {
            operators.extend(operator::load_operators_from_yaml(
                &operator,
                &state_metadata,
                &table_registry,
            )?);
        }

        Ok(Problem {
            minimize,
            domain_name,
            problem_name,
            state_metadata,
            initial_state,
            table_registry,
            constraints,
            goals,
            operators,
        })
    }

    fn filiter_grounded_conditions(
        conditions: Vec<grounded_condition::GroundedCondition<T>>,
    ) -> Result<Vec<grounded_condition::GroundedCondition<T>>, ProblemErr> {
        let mut result = Vec::with_capacity(conditions.len());
        for condition in conditions {
            match condition.condition {
                expression::Condition::Constant(true) => continue,
                expression::Condition::Constant(false) => {
                    return Err(ProblemErr::new(String::from(
                        "problem has a condition never satisfied",
                    )))
                }
                _ => result.push(condition),
            }
        }
        Ok(result)
    }
}
