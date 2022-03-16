use crate::expression;
use crate::expression_parser;
use crate::function_registry;
use crate::state;
use crate::variable;
use crate::yaml_util;
use std::error::Error;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

pub struct GroundedCondition<T: variable::Numeric> {
    pub elements_in_set_variable: Vec<(usize, usize)>,
    pub elements_in_permutation_variable: Vec<(usize, usize)>,
    pub condition: expression::Condition<T>,
}

pub fn load_grounded_conditions_from_yaml<T: variable::Numeric>(
    value: &Yaml,
    metadata: &state::StateMetadata,
    registry: &function_registry::FunctionRegistry<T>,
) -> Result<Vec<GroundedCondition<T>>, Box<dyn Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = yaml_util::get_map(value)?;

    let parameters = yaml_util::get_yaml_by_key(map, "parameters")?;
    let (parameters_set, elements_in_set_variable_set, elements_in_permutation_variable_set) =
        metadata.get_grounded_parameter_set_from_yaml(parameters)?;

    let condition = yaml_util::get_string_by_key(map, "condition")?;

    let mut conditions = Vec::with_capacity(parameters_set.len());
    for ((parameters, elements_in_set_variable), elements_in_permutation_variable) in parameters_set
        .into_iter()
        .zip(elements_in_set_variable_set.into_iter())
        .zip(elements_in_permutation_variable_set.into_iter())
    {
        let condition =
            expression_parser::parse_condition(condition.clone(), metadata, registry, &parameters)?;
        conditions.push(GroundedCondition {
            condition,
            elements_in_set_variable,
            elements_in_permutation_variable,
        });
    }

    Ok(conditions)
}
