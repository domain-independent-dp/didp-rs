use super::expression_parser;
use super::state_parser::ground_parameters_from_yaml;
use crate::util;
use dypdl::prelude::*;
use dypdl::GroundedCondition;
use dypdl::{StateMetadata, TableRegistry};
use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use std::error::Error;
use yaml_rust::Yaml;

/// Returns a grounded condition loaded from YAML
///
/// # Errors
///
/// If the format is invalid.
pub fn load_grounded_conditions_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<Vec<GroundedCondition>, Box<dyn Error>> {
    lazy_static! {
        static ref FORALL_KEY: Yaml = Yaml::from_str("forall");
    }
    match value {
        Yaml::String(condition) => {
            let condition = expression_parser::parse_condition(
                condition.clone(),
                metadata,
                functions,
                registry,
                parameters,
            )?;
            Ok(vec![GroundedCondition::from(condition.simplify(registry))])
        }
        Yaml::Hash(map) => {
            let condition = util::get_string_by_key(map, "condition")?;
            match map.get(&FORALL_KEY) {
                Some(forall) => {
                    let (parameters_array, elements_in_set_variable_array) =
                        ground_parameters_from_yaml(metadata, forall)?;
                    let mut conditions = Vec::with_capacity(parameters_array.len());
                    for (forall, elements_in_set_variable) in parameters_array
                        .into_iter()
                        .zip(elements_in_set_variable_array.into_iter())
                    {
                        let mut parameters = parameters.clone();
                        parameters.extend(forall);
                        let condition = expression_parser::parse_condition(
                            condition.clone(),
                            metadata,
                            functions,
                            registry,
                            &parameters,
                        )?;
                        conditions.push(GroundedCondition {
                            condition: condition.simplify(registry),
                            elements_in_set_variable,
                        });
                    }
                    Ok(conditions)
                }
                None => {
                    let condition = expression_parser::parse_condition(
                        condition, metadata, functions, registry, parameters,
                    )?;
                    Ok(vec![GroundedCondition::from(condition.simplify(registry))])
                }
            }
        }
        _ => Err(
            util::YamlContentErr::new(format!("expected String or Hash, found `{value:?}`",))
                .into(),
        ),
    }
}
