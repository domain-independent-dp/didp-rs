use crate::expression;
use crate::expression_parser;
use crate::state;
use crate::table_registry;
use crate::variable;
use crate::yaml_util;
use std::collections;
use std::error::Error;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct GroundedCondition<T: variable::Numeric> {
    pub elements_in_set_variable: Vec<(usize, usize)>,
    pub elements_in_permutation_variable: Vec<(usize, usize)>,
    pub condition: expression::Condition<T>,
}

pub fn load_grounded_conditions_from_yaml<T: variable::Numeric>(
    value: &Yaml,
    metadata: &state::StateMetadata,
    registry: &table_registry::TableRegistry<T>,
) -> Result<Vec<GroundedCondition<T>>, Box<dyn Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = yaml_util::get_map(value)?;
    let condition = yaml_util::get_string_by_key(map, "condition")?;

    match map.get(&Yaml::from_str("parameters")) {
        Some(parameters) => {
            let (
                parameters_array,
                elements_in_set_variable_array,
                elements_in_permutation_variable_array,
            ) = metadata.ground_parameters_from_yaml(parameters)?;

            let mut conditions = Vec::with_capacity(parameters_array.len());
            for ((parameters, elements_in_set_variable), elements_in_permutation_variable) in
                parameters_array
                    .into_iter()
                    .zip(elements_in_set_variable_array.into_iter())
                    .zip(elements_in_permutation_variable_array.into_iter())
            {
                let condition = expression_parser::parse_condition(
                    condition.clone(),
                    metadata,
                    registry,
                    &parameters,
                )?;
                conditions.push(GroundedCondition {
                    condition: condition.simplify(registry),
                    elements_in_set_variable,
                    elements_in_permutation_variable,
                });
            }
            Ok(conditions)
        }
        None => {
            let parameters = collections::HashMap::new();
            let condition =
                expression_parser::parse_condition(condition, metadata, registry, &parameters)?;
            Ok(vec![GroundedCondition {
                condition: condition.simplify(registry),
                elements_in_set_variable: vec![],
                elements_in_permutation_variable: vec![],
            }])
        }
    }
}
