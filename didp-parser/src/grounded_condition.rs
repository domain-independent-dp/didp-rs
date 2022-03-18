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

#[cfg(test)]
mod tests {
    use super::super::expression::*;
    use super::super::table;
    use super::*;
    use std::collections::HashMap;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec![String::from("object")];
        let mut name_to_object = HashMap::new();
        name_to_object.insert(String::from("object"), 0);
        let object_numbers = vec![2];

        let set_variable_names = vec![String::from("s0")];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert(String::from("s0"), 0);
        let set_variable_to_object = vec![0];

        let permutation_variable_names = vec![String::from("p0")];
        let mut name_to_permutation_variable = HashMap::new();
        name_to_permutation_variable.insert(String::from("p0"), 0);
        let permutation_variable_to_object = vec![0];

        let element_variable_names = vec![String::from("e0")];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert(String::from("e0"), 0);
        let element_variable_to_object = vec![0];

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            permutation_variable_names,
            name_to_permutation_variable,
            permutation_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            ..Default::default()
        }
    }

    fn generate_registry() -> table_registry::TableRegistry<variable::IntegerVariable> {
        let tables_1d = vec![table::Table1D::new(vec![true, false])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("b1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![
            vec![false, true],
            vec![true, false],
        ])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("b2"), 0);

        table_registry::TableRegistry {
            bool_tables: table_registry::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn load_from_yaml_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();

        let condition = r"
condition: (is_in e0 s0)
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(condition, &metadata, &registry);
        assert!(conditions.is_ok());
        let expected = vec![GroundedCondition {
            elements_in_set_variable: Vec::new(),
            elements_in_permutation_variable: Vec::new(),
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::SetVariable(0),
            )),
        }];
        assert_eq!(conditions.unwrap(), expected);

        let condition = r"
condition: (is 0 e)
parameters:
        - name: e
          object: s0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(condition, &metadata, &registry);
        assert!(conditions.is_ok());
        let expected = vec![
            GroundedCondition {
                elements_in_set_variable: vec![(0, 0)],
                elements_in_permutation_variable: Vec::new(),
                condition: Condition::Constant(true),
            },
            GroundedCondition {
                elements_in_set_variable: vec![(0, 1)],
                elements_in_permutation_variable: Vec::new(),
                condition: Condition::Constant(false),
            },
        ];
        assert_eq!(conditions.unwrap(), expected);

        let condition = r"
condition: (is_in e s0)
parameters:
        - name: e
          object: p0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(condition, &metadata, &registry);
        assert!(conditions.is_ok());
        let expected = vec![
            GroundedCondition {
                elements_in_set_variable: Vec::new(),
                elements_in_permutation_variable: vec![(0, 0)],
                condition: Condition::Set(SetCondition::IsIn(
                    ElementExpression::Constant(0),
                    SetExpression::SetVariable(0),
                )),
            },
            GroundedCondition {
                elements_in_set_variable: Vec::new(),
                elements_in_permutation_variable: vec![(0, 1)],
                condition: Condition::Set(SetCondition::IsIn(
                    ElementExpression::Constant(1),
                    SetExpression::SetVariable(0),
                )),
            },
        ];
        assert_eq!(conditions.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();

        let condition = r"
conddition: (is_in e0 s0)
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(condition, &metadata, &registry);
        assert!(conditions.is_err());

        let condition = r"
condition: (is 0 d)
parameters:
        - name: e
          object: s0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(condition, &metadata, &registry);
        assert!(conditions.is_err());

        let condition = r"
condition: (is_in e s0)
parameters:
        - object: p0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(condition, &metadata, &registry);
        assert!(conditions.is_err());

        let condition = r"
condition: (is_in e s0)
parameters:
        - name: e
          object: null
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(condition, &metadata, &registry);
        assert!(conditions.is_err());

        let condition = r"
condition: (is_in e s0)
parameters:
        - name: e
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = load_grounded_conditions_from_yaml(condition, &metadata, &registry);
        assert!(conditions.is_err());
    }
}
