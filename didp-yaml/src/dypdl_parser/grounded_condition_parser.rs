use super::expression_parser;
use super::expression_parser::ModelData;
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
    local_variable_data: &mut LocalVariableData,
) -> Result<Vec<GroundedCondition>, Box<dyn Error>> {
    lazy_static! {
        static ref FORALL_KEY: Yaml = Yaml::from_str("forall");
    }
    match value {
        Yaml::String(condition) => {
            let mut model_data = ModelData {
                metadata,
                functions,
                registry,
                parameters,
                local_variable_data,
            };
            let condition = expression_parser::parse_condition(condition.clone(), &mut model_data)?;
            Ok(vec![GroundedCondition::from(condition.simplify(registry))])
        }
        Yaml::Hash(map) => {
            let condition = util::get_string_by_key(map, "condition")?;
            match map.get(&FORALL_KEY) {
                Some(forall) => {
                    let (
                        parameters_array,
                        elements_in_set_variable_array,
                        elements_in_set_resource_variable_array,
                    ) = ground_parameters_from_yaml(metadata, forall)?;
                    let mut conditions = Vec::with_capacity(parameters_array.len());
                    for (forall, (elements_in_set_variable, elements_in_set_resource_variable)) in
                        parameters_array.into_iter().zip(
                            elements_in_set_variable_array
                                .into_iter()
                                .zip(elements_in_set_resource_variable_array),
                        )
                    {
                        let mut parameters = parameters.clone();
                        parameters.extend(forall);
                        let mut model_data = ModelData {
                            metadata,
                            functions,
                            registry,
                            parameters: &parameters,
                            local_variable_data: &mut *local_variable_data,
                        };
                        let condition =
                            expression_parser::parse_condition(condition.clone(), &mut model_data)?;
                        conditions.push(GroundedCondition {
                            condition: condition.simplify(registry),
                            elements_in_set_variable,
                            elements_in_set_resource_variable,
                        });
                    }
                    Ok(conditions)
                }
                None => {
                    let mut model_data = ModelData {
                        metadata,
                        functions,
                        registry,
                        parameters,
                        local_variable_data,
                    };
                    let condition = expression_parser::parse_condition(condition, &mut model_data)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;

    #[test]
    fn load_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob);
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d(String::from("b1"), vec![true, false]);
        assert!(result.is_ok());
        let result = registry.add_table_2d(
            String::from("b2"),
            vec![vec![false, true], vec![true, false]],
        );
        assert!(result.is_ok());

        let condition = r"
condition: (and (is_in e0 s0) true)
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];
        let parameters = FxHashMap::default();

        let conditions = load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut LocalVariableData::default(),
        );
        assert!(conditions.is_ok());
        let expected = vec![GroundedCondition {
            elements_in_set_variable: Vec::new(),
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            ..Default::default()
        }];
        assert_eq!(conditions.unwrap(), expected);

        let condition = r"(is_in e0 s0)";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];
        let parameters = FxHashMap::default();
        let conditions = load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut LocalVariableData::default(),
        );
        assert!(conditions.is_ok());
        assert_eq!(conditions.unwrap(), expected);

        let condition = r"(is_in a s0)";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];
        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("a"), 0);
        let conditions = load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut LocalVariableData::default(),
        );
        assert!(conditions.is_ok());
        let expected = vec![GroundedCondition {
            elements_in_set_variable: Vec::new(),
            condition: Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            ))),
            ..Default::default()
        }];
        assert_eq!(conditions.unwrap(), expected);

        let condition = r"
condition: (= 0 e)
forall:
        - name: e
          object: s0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];
        let parameters = FxHashMap::default();

        let conditions = load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut LocalVariableData::default(),
        );
        assert!(conditions.is_ok());
        let expected = vec![
            GroundedCondition {
                elements_in_set_variable: vec![(0, 0)],
                condition: Condition::Constant(true),
                ..Default::default()
            },
            GroundedCondition {
                elements_in_set_variable: vec![(0, 1)],
                condition: Condition::Constant(false),
                ..Default::default()
            },
        ];
        assert_eq!(conditions.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob);
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d(String::from("b1"), vec![true, false]);
        assert!(result.is_ok());
        let result = registry.add_table_2d(
            String::from("b2"),
            vec![vec![false, true], vec![true, false]],
        );
        assert!(result.is_ok());

        let condition = r"
conddition: (is_in e0 s0)
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];
        let parameters = FxHashMap::default();

        let conditions = load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut LocalVariableData::default(),
        );
        assert!(conditions.is_err());

        let condition = r"
condition: (is 0 d)
forall:
        - name: e
          object: s0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut LocalVariableData::default(),
        );
        assert!(conditions.is_err());

        let condition = r"
condition: (is_in e s0)
forall:
        - object: p0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut LocalVariableData::default(),
        );
        assert!(conditions.is_err());

        let condition = r"
condition: (is_in e s0)
forall:
        - name: e
          object: null
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut LocalVariableData::default(),
        );
        assert!(conditions.is_err());

        let condition = r"
condition: (is_in e s0)
forall:
        - name: e
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut LocalVariableData::default(),
        );
        assert!(conditions.is_err());
    }
}
