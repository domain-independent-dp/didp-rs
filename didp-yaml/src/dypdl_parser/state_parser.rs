use crate::util;
use dypdl::prelude::*;
use dypdl::{ResourceVariables, SignatureVariables, State, StateMetadata};
use lazy_static::lazy_static;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use yaml_rust::yaml::Hash;

/// Returns a state loaded from YAML
///
/// # Errors
///
/// If the format is invalid.
pub fn load_state_from_yaml(
    value: &yaml_rust::Yaml,
    metadata: &StateMetadata,
) -> Result<State, Box<dyn std::error::Error>> {
    let value = util::get_map(value)?;
    let mut set_variables = Vec::with_capacity(metadata.set_variable_names.len());
    for name in &metadata.set_variable_names {
        let values = util::get_usize_array_by_key(value, name)?;
        let variable = metadata.get_set_variable(name)?;
        let object = metadata.get_object_type_of(variable)?;
        let capacity = metadata.get_number_of_objects(object)?;
        let mut set = Set::with_capacity(capacity);
        for v in values {
            if v >= capacity {
                return Err(util::YamlContentErr::new(format!(
                    "value `{v}` is out of range of a set variable `{name}`"
                ))
                .into());
            }
            set.insert(v);
        }
        set_variables.push(set);
    }
    let mut element_variables = Vec::with_capacity(metadata.element_variable_names.len());
    for name in &metadata.element_variable_names {
        let element = util::get_usize_by_key(value, name)?;
        element_variables.push(element);
    }
    let mut integer_variables = Vec::with_capacity(metadata.integer_variable_names.len());
    for name in &metadata.integer_variable_names {
        let value = util::get_numeric_by_key(value, name)?;
        integer_variables.push(value);
    }
    let mut continuous_variables = Vec::with_capacity(metadata.continuous_variable_names.len());
    for name in &metadata.continuous_variable_names {
        let value = util::get_numeric_by_key(value, name)?;
        continuous_variables.push(value);
    }
    let mut element_resource_variables =
        Vec::with_capacity(metadata.element_resource_variable_names.len());
    for name in &metadata.element_resource_variable_names {
        let element = util::get_usize_by_key(value, name)?;
        element_resource_variables.push(element);
    }
    let mut integer_resource_variables =
        Vec::with_capacity(metadata.integer_resource_variable_names.len());
    for name in &metadata.integer_resource_variable_names {
        let value = util::get_numeric_by_key(value, name)?;
        integer_resource_variables.push(value);
    }
    let mut continuous_resource_variables =
        Vec::with_capacity(metadata.continuous_resource_variable_names.len());
    for name in &metadata.continuous_resource_variable_names {
        let value = util::get_numeric_by_key(value, name)?;
        continuous_resource_variables.push(value);
    }
    Ok(State {
        signature_variables: SignatureVariables {
            set_variables,
            element_variables,
            integer_variables,
            continuous_variables,
        },
        resource_variables: ResourceVariables {
            element_variables: element_resource_variables,
            integer_variables: integer_resource_variables,
            continuous_variables: continuous_resource_variables,
        },
    })
}

pub fn ground_static_parameters_from_yaml(
    metadata: &StateMetadata,
    value: &yaml_rust::Yaml,
) -> Result<Vec<BTreeMap<String, usize>>, Box<dyn std::error::Error>> {
    let array = util::get_array(value)?;
    let mut parameters_array = Vec::with_capacity(array.len());
    parameters_array.push(BTreeMap::default());

    let mut reserved_names = FxHashSet::default();

    for value in array {
        let map = util::get_map(value)?;
        let name = util::get_string_by_key(map, "name")?;

        if let Some(name) = reserved_names.get(&name) {
            return Err(util::YamlContentErr::new(format!(
                "parameter name `{name}` is already used"
            ))
            .into());
        }

        reserved_names.insert(name.clone());

        let object = util::get_string_by_key(map, "object")?;
        let object = metadata.get_object_type(&object)?;
        let n = metadata.get_number_of_objects(object)?;
        let mut new_parameters_set = Vec::with_capacity(parameters_array.len() * n);

        for parameters in &parameters_array {
            for i in 0..n {
                let mut parameters = parameters.clone();
                parameters.insert(name.clone(), i);
                new_parameters_set.push(parameters);
            }
        }

        parameters_array = new_parameters_set;
    }

    Ok(parameters_array)
}

type GroundedParameterPair = (Vec<FxHashMap<String, usize>>, Vec<Vec<(usize, usize)>>);

pub fn ground_parameters_from_yaml(
    metadata: &StateMetadata,
    value: &yaml_rust::Yaml,
) -> Result<GroundedParameterPair, Box<dyn std::error::Error>> {
    let array = util::get_array(value)?;
    let mut parameters_array: Vec<FxHashMap<String, usize>> = Vec::with_capacity(array.len());
    parameters_array.push(FxHashMap::default());
    let mut elements_in_set_variable_array: Vec<Vec<(usize, usize)>> =
        Vec::with_capacity(array.len());
    elements_in_set_variable_array.push(vec![]);
    let mut reserved_names = FxHashSet::default();
    for value in array {
        let map = util::get_map(value)?;
        let name = util::get_string_by_key(map, "name")?;
        if let Some(name) = reserved_names.get(&name) {
            return Err(util::YamlContentErr::new(format!(
                "parameter name `{name}` is already used"
            ))
            .into());
        }
        reserved_names.insert(name.clone());
        let object = util::get_string_by_key(map, "object")?;
        let (n, set_index) = if let Ok(object) = metadata.get_object_type(&object) {
            (metadata.get_number_of_objects(object)?, None)
        } else if let Ok(v) = metadata.get_set_variable(&object) {
            let object = metadata.get_object_type_of(v)?;
            (metadata.get_number_of_objects(object)?, Some(v.id()))
        } else {
            return Err(util::YamlContentErr::new(format!(
                "no such object or set variable `{object}`"
            ))
            .into());
        };
        let mut new_parameteres_set = Vec::with_capacity(parameters_array.len() * n);
        let mut new_elements_in_set_variable_array =
            Vec::with_capacity(elements_in_set_variable_array.len() * n);
        for (parameters, elements_in_set_variable) in parameters_array
            .iter()
            .zip(elements_in_set_variable_array.iter())
        {
            for i in 0..n {
                let mut parameters = parameters.clone();
                parameters.insert(name.clone(), i);
                let mut elements_in_set_variable = elements_in_set_variable.clone();
                if let Some(j) = set_index {
                    elements_in_set_variable.push((j, i));
                }
                new_parameteres_set.push(parameters);
                new_elements_in_set_variable_array.push(elements_in_set_variable);
            }
        }
        parameters_array = new_parameteres_set;
        elements_in_set_variable_array = new_elements_in_set_variable_array;
    }

    Ok((parameters_array, elements_in_set_variable_array))
}

pub fn load_metadata_from_yaml(
    objects: &yaml_rust::Yaml,
    variables: &yaml_rust::Yaml,
    object_numbers_yaml: &yaml_rust::Yaml,
) -> Result<StateMetadata, Box<dyn std::error::Error>> {
    let mut metadata = StateMetadata::default();

    let object_names = util::get_string_array(objects)?;
    let object_numbers_yaml = util::get_map(object_numbers_yaml)?;
    for name in object_names.iter() {
        let number = util::get_usize_by_key(object_numbers_yaml, name)?;
        metadata.add_object_type(name.clone(), number)?;
    }

    let mut reserved_names = metadata.get_name_set();

    let variables = util::get_array(variables)?;
    for value in variables {
        let map = util::get_map(value)?;
        let name = util::get_string_by_key(map, "name")?;
        if let Some(name) = reserved_names.get(&name) {
            return Err(util::YamlContentErr::new(format!(
                "variable name `{name}` is already used"
            ))
            .into());
        }
        reserved_names.insert(name.clone());
        let variable_type = util::get_string_by_key(map, "type")?;
        match &variable_type[..] {
            "set" => {
                let object_name = util::get_string_by_key(map, "object")?;
                let ob = metadata.get_object_type(&object_name)?;
                metadata.add_set_variable(name, ob)?;
            }
            "element" => match get_less_is_better(map)? {
                Some(value) => {
                    let object_name = util::get_string_by_key(map, "object")?;
                    let ob = metadata.get_object_type(&object_name)?;
                    metadata.add_element_resource_variable(name, ob, value)?;
                }
                None => {
                    let object_name = util::get_string_by_key(map, "object")?;
                    let ob = metadata.get_object_type(&object_name)?;
                    metadata.add_element_variable(name, ob)?;
                }
            },
            "integer" => match get_less_is_better(map)? {
                Some(value) => {
                    metadata.add_integer_resource_variable(name, value)?;
                }
                None => {
                    metadata.add_integer_variable(name)?;
                }
            },
            "continuous" => match get_less_is_better(map)? {
                Some(value) => {
                    metadata.add_continuous_resource_variable(name, value)?;
                }
                None => {
                    metadata.add_continuous_variable(name)?;
                }
            },
            value => {
                return Err(util::YamlContentErr::new(format!(
                    "`{value:?}` is not a variable type"
                ))
                .into())
            }
        }
    }
    Ok(metadata)
}

fn get_less_is_better(map: &Hash) -> Result<Option<bool>, util::YamlContentErr> {
    lazy_static! {
        static ref KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("preference");
    }
    match map.get(&KEY) {
        Some(yaml_rust::Yaml::String(value)) if &value[..] == "greater" => Ok(Some(false)),
        Some(yaml_rust::Yaml::String(value)) if &value[..] == "less" => Ok(Some(true)),
        Some(value) => Err(util::YamlContentErr::new(format!(
            "expected `String(\"greater\")` or `String(\"less\")`, found `{value:?}`"
        ))),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_static_parameters_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 3);
        assert!(result.is_ok());
        let result = metadata.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());

        let mut map1 = BTreeMap::default();
        map1.insert(String::from("v0"), 0);
        map1.insert(String::from("v1"), 0);
        let mut map2 = BTreeMap::default();
        map2.insert(String::from("v0"), 0);
        map2.insert(String::from("v1"), 1);
        let mut map3 = BTreeMap::default();
        map3.insert(String::from("v0"), 1);
        map3.insert(String::from("v1"), 0);
        let mut map4 = BTreeMap::default();
        map4.insert(String::from("v0"), 1);
        map4.insert(String::from("v1"), 1);
        let mut map5 = BTreeMap::default();
        map5.insert(String::from("v0"), 2);
        map5.insert(String::from("v1"), 0);
        let mut map6 = BTreeMap::default();
        map6.insert(String::from("v0"), 2);
        map6.insert(String::from("v1"), 1);
        let expected_parameters = vec![map1, map2, map3, map4, map5, map6];

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: object
- name: v1
  object: small
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = ground_static_parameters_from_yaml(&metadata, yaml);
        assert!(result.is_ok());
        let parameters = result.unwrap();
        assert_eq!(parameters, expected_parameters);
    }

    #[test]
    fn ground_static_parameters_from_yaml_array_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 3);
        assert!(result.is_ok());
        let result = metadata.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
name: v0
object: object
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = ground_static_parameters_from_yaml(&metadata, yaml);
        assert!(result.is_err());
    }

    #[test]
    fn ground_static_parameters_from_yaml_no_name_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 3);
        assert!(result.is_ok());
        let result = metadata.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: object
- object: small
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = ground_static_parameters_from_yaml(&metadata, yaml);
        assert!(result.is_err());
    }

    #[test]
    fn ground_static_parameters_from_yaml_no_object_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 3);
        assert!(result.is_ok());
        let result = metadata.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: object
- name: v1
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = ground_static_parameters_from_yaml(&metadata, yaml);
        assert!(result.is_err());
    }

    #[test]
    fn ground_static_parameters_from_yaml_duplicate_name_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 3);
        assert!(result.is_ok());
        let result = metadata.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: object
- name: v0
  object: small
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = ground_static_parameters_from_yaml(&metadata, yaml);
        assert!(result.is_err());
    }

    #[test]
    fn state_metadata_load_from_yaml_err() {
        let objects = r"
- object
- object
";
        let variables = r"
- name: s0
  type: set 
  object: object
";
        let object_numbers = r"
object: 10
";
        let objects = yaml_rust::YamlLoader::load_from_str(objects);
        assert!(objects.is_ok());
        let objects = objects.unwrap();
        assert_eq!(objects.len(), 1);
        let objects = &objects[0];

        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];

        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let objects = r"
- object
- small
";
        let variables = r"
- name: object
  type: set 
  object: object
";
        let object_numbers = r"
object: 10
small: 3
";
        let objects = yaml_rust::YamlLoader::load_from_str(objects);
        assert!(objects.is_ok());
        let objects = objects.unwrap();
        assert_eq!(objects.len(), 1);
        let objects = &objects[0];

        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];

        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let objects = r"
- object
- small
";
        let variables = r"
- name: s0
  type: set 
  object: object
- name: s0
  type: numeric
";
        let object_numbers = r"
object: 10
small: 3
";
        let objects = yaml_rust::YamlLoader::load_from_str(objects);
        assert!(objects.is_ok());
        let objects = objects.unwrap();
        assert_eq!(objects.len(), 1);
        let objects = &objects[0];

        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];

        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let objects = r"
- object
- small
";
        let variables = r"
- name: s0
  type: null
  object: small 
";
        let object_numbers = r"
object: 10
small: 2
";
        let objects = yaml_rust::YamlLoader::load_from_str(objects);
        assert!(objects.is_ok());
        let objects = objects.unwrap();
        assert_eq!(objects.len(), 1);
        let objects = &objects[0];

        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];

        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- name: s0
  type: set
  object: null
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];
        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- object: object
  type: set
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];
        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- name: s0
  type: set
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];
        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- name: s0
  object: object
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];
        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let variables = r"
- name: s0
  type: set
  object: object
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let object_numbers = r"
object: 10
";
        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];
        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());

        let object_numbers = r"
object: 10
small: 2
";
        let object_numbers = yaml_rust::YamlLoader::load_from_str(object_numbers);
        assert!(object_numbers.is_ok());
        let object_numbers = object_numbers.unwrap();
        assert_eq!(object_numbers.len(), 1);
        let object_numbers = &object_numbers[0];

        let variables = r"
- name: s0
  type: set
  object: object
- name: n0
  type: integer
  preference: null
";
        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let metadata = load_metadata_from_yaml(objects, variables, object_numbers);
        assert!(metadata.is_err());
    }
}
