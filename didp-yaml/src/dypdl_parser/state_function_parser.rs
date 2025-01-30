use super::parse_expression_from_yaml;
use crate::util;
use dypdl::{StateFunctions, StateMetadata, TableRegistry};
use rustc_hash::FxHashMap;
use std::error::Error;
use yaml_rust::Yaml;

/// Returns state functions loaded from YAML
///
/// # Errors
///
/// If the format is invalid.
pub fn load_state_functions_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    registry: &TableRegistry,
) -> Result<StateFunctions, Box<dyn Error>> {
    let array = util::get_array(value)?;
    let mut functions = StateFunctions::default();
    let parameters = FxHashMap::default();
    let mut reserved_names = metadata.get_name_set();
    reserved_names.extend(registry.get_name_set());

    for value in array {
        let map = util::get_map(value)?;
        let name = util::get_string_by_key(map, "name")?;
        let function_type = util::get_string_by_key(map, "type")?;
        let expression = util::get_yaml_by_key(map, "expression")?;

        if reserved_names.contains(&name) {
            return Err(format!("state function name `{}` is already used", name).into());
        }

        reserved_names.insert(name.clone());

        match &function_type[..] {
            "set" => {
                let expression = parse_expression_from_yaml::parse_set_from_yaml(
                    expression,
                    metadata,
                    &functions,
                    registry,
                    &parameters,
                )?;
                functions.add_set_function(name, expression)?;
            }
            "element" => {
                let expression = parse_expression_from_yaml::parse_element_from_yaml(
                    expression,
                    metadata,
                    &functions,
                    registry,
                    &parameters,
                )?;
                functions.add_element_function(name, expression)?;
            }
            "integer" => {
                let expression = parse_expression_from_yaml::parse_integer_from_yaml(
                    expression,
                    metadata,
                    &functions,
                    registry,
                    &parameters,
                )?;
                functions.add_integer_function(name, expression)?;
            }
            "continuous" => {
                let expression = parse_expression_from_yaml::parse_continuous_from_yaml(
                    expression,
                    metadata,
                    &functions,
                    registry,
                    &parameters,
                )?;
                functions.add_continuous_function(name, expression)?;
            }
            "bool" => {
                let expression = parse_expression_from_yaml::parse_condition_from_yaml(
                    expression,
                    metadata,
                    &functions,
                    registry,
                    &parameters,
                )?;
                functions.add_boolean_function(name, expression)?;
            }
            value => {
                return Err(format!("{:?} is not a function type", value).into());
            }
        }
    }

    Ok(functions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::prelude::*;
    use yaml_rust::YamlLoader;

    #[test]
    fn load_state_function_ok() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let sv = result.unwrap();
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());
        let ev = result.unwrap();

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let it = result.unwrap();
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());
        let ct = result.unwrap();

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)
    - name: sf5
      type: bool
      expression: (is_in sf1 sf2)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_ok());
        let functions = result.unwrap();

        let result = functions.get_element_function("sf1");
        assert!(result.is_ok());
        let sf1 = result.unwrap();
        assert_eq!(functions.element_functions[0], ev + 1);

        let result = functions.get_set_function("sf2");
        assert!(result.is_ok());
        let sf2 = result.unwrap();
        assert_eq!(functions.set_functions[0], sv.add(sf1.clone()));

        let result = functions.get_integer_function("sf3");
        assert!(result.is_ok());
        assert_eq!(functions.integer_functions[0], it.element(sf1.clone()));

        let result = functions.get_continuous_function("sf4");
        assert!(result.is_ok());
        assert_eq!(functions.continuous_functions[0], ct.element(sf1.clone()));

        let result = functions.get_boolean_function("sf5");
        assert!(result.is_ok());
        assert_eq!(functions.boolean_functions[0], sf2.contains(sf1));
    }

    #[test]
    fn load_state_function_yaml_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    name: sf1
    type: element
    expression: (+ ev 1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_map_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - - name: sf3
      - type: integer
      - expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_no_name_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_no_type_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_invalid_type_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: long
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_no_expression_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_variable_name_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sv
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_table_name_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: ct
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_invalid_element_expression_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (add sf1 sv)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_duplicate_element_function_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_invalid_set_expression_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (+ ev 1)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_duplicate_set_function_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_invalid_integer_expression_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (+ ev 1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_duplicate_integer_function_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_invalid_continuous_expression_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (+ ev 1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_duplicate_continuous_function_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_invalid_bool_expression_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)
    - name: sf5
      type: bool
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_duplicate_bool_function_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)
    - name: sf5
      type: bool
      expression: (is_in sf1 sf2)
    - name: sf5
      type: bool
      expression: (is_in sf1 sf2)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }

    #[test]
    fn load_state_function_order_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 4);
        assert!(result.is_ok());
        let object = result.unwrap();
        let result = metadata.add_set_variable("sv", object);
        assert!(result.is_ok());
        let result = metadata.add_element_variable("ev", object);
        assert!(result.is_ok());

        let mut registry = TableRegistry::default();
        let result = registry.add_table_1d("it", vec![3i32, 2i32, 1i32, 0i32]);
        assert!(result.is_ok());
        let result = registry.add_table_1d("ct", vec![0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_ok());

        let state_functions = r"
    - name: sf2
      type: set
      expression: (add sf1 sv)
    - name: sf1
      type: element
      expression: (+ ev 1)
    - name: sf3
      type: integer
      expression: (it sf1)
    - name: sf4
      type: continuous
      expression: (ct sf1)";

        let yaml = YamlLoader::load_from_str(state_functions);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_state_functions_from_yaml(yaml, &metadata, &registry);
        assert!(result.is_err());
    }
}
