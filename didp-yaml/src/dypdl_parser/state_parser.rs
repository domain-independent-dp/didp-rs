use crate::util;
use dypdl::prelude::*;
use dypdl::{ResourceVariables, SignatureVariables, State, StateMetadata};
use lazy_static::lazy_static;
use rustc_hash::{FxHashMap, FxHashSet};

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
            set.insert(v);
        }
        set_variables.push(set);
    }
    let mut vector_variables = Vec::with_capacity(metadata.vector_variable_names.len());
    for name in &metadata.vector_variable_names {
        let vector = util::get_usize_array_by_key(value, name)?;
        vector_variables.push(vector);
    }
    let mut element_variables = Vec::with_capacity(metadata.element_variable_names.len());
    for (variable_id, name) in metadata.element_variable_names.iter().enumerate() {
        let element = util::get_usize_by_key(value, name)?;
        let object_id = metadata.element_variable_to_object[variable_id];
        let capacity = metadata.object_numbers[object_id];
        if element >= capacity {
            return Err(util::YamlContentErr::new(format!(
                "value `{}` for `{}` is out of the bound of `{}`, which has only {} objects",
                element, name, metadata.object_type_names[object_id], capacity,
            ))
            .into());
        }
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
    for (variable_id, name) in metadata.element_resource_variable_names.iter().enumerate() {
        let element = util::get_usize_by_key(value, name)?;
        let object_id = metadata.element_resource_variable_to_object[variable_id];
        let capacity = metadata.object_numbers[object_id];
        if element >= capacity {
            return Err(util::YamlContentErr::new(format!(
                "value `{}` for `{}` is out of the bound of `{}`, which has only {} objects",
                element, name, metadata.object_type_names[object_id], capacity,
            ))
            .into());
        }
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
            vector_variables,
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

type GroundedParameterTriplet = (
    Vec<FxHashMap<String, usize>>,
    Vec<Vec<(usize, usize)>>,
    Vec<Vec<(usize, usize)>>,
);

pub fn ground_parameters_from_yaml(
    metadata: &StateMetadata,
    value: &yaml_rust::Yaml,
) -> Result<GroundedParameterTriplet, Box<dyn std::error::Error>> {
    let array = util::get_array(value)?;
    let mut parameters_array: Vec<FxHashMap<String, usize>> = Vec::with_capacity(array.len());
    parameters_array.push(FxHashMap::default());
    let mut elements_in_set_variable_array: Vec<Vec<(usize, usize)>> =
        Vec::with_capacity(array.len());
    elements_in_set_variable_array.push(vec![]);
    let mut elements_in_vector_variable_array: Vec<Vec<(usize, usize)>> =
        Vec::with_capacity(array.len());
    elements_in_vector_variable_array.push(vec![]);
    let mut reserved_names = FxHashSet::default();
    for value in array {
        let map = util::get_map(value)?;
        let name = util::get_string_by_key(map, "name")?;
        if let Some(name) = reserved_names.get(&name) {
            return Err(util::YamlContentErr::new(format!(
                "parameter name `{}` is already used",
                name
            ))
            .into());
        }
        reserved_names.insert(name.clone());
        let object = util::get_string_by_key(map, "object")?;
        let (n, set_index, vector_index) = if let Ok(object) = metadata.get_object_type(&object) {
            (metadata.get_number_of_objects(object)?, None, None)
        } else if let Ok(v) = metadata.get_set_variable(&object) {
            let object = metadata.get_object_type_of(v)?;
            (metadata.get_number_of_objects(object)?, Some(v.id()), None)
        } else if let Ok(v) = metadata.get_vector_variable(&object) {
            let object = metadata.get_object_type_of(v)?;
            (metadata.get_number_of_objects(object)?, None, Some(v.id()))
        } else {
            return Err(util::YamlContentErr::new(format!(
                "no such object, set variable, or vector variable `{}`",
                object
            ))
            .into());
        };
        let mut new_parameteres_set = Vec::with_capacity(parameters_array.len() * n);
        let mut new_elements_in_set_variable_array =
            Vec::with_capacity(elements_in_set_variable_array.len() * n);
        let mut new_elements_in_vector_variable_array =
            Vec::with_capacity(elements_in_vector_variable_array.len() * n);
        for ((parameters, elements_in_set_variable), elements_in_vector_variable) in
            parameters_array
                .iter()
                .zip(elements_in_set_variable_array.iter())
                .zip(elements_in_vector_variable_array.iter())
        {
            for i in 0..n {
                let mut parameters = parameters.clone();
                parameters.insert(name.clone(), i);
                let mut elements_in_set_variable = elements_in_set_variable.clone();
                if let Some(j) = set_index {
                    elements_in_set_variable.push((j, i));
                }
                let mut elements_in_vector_variable = elements_in_vector_variable.clone();
                if let Some(j) = vector_index {
                    elements_in_vector_variable.push((j, i));
                }
                new_parameteres_set.push(parameters);
                new_elements_in_set_variable_array.push(elements_in_set_variable);
                new_elements_in_vector_variable_array.push(elements_in_vector_variable);
            }
        }
        parameters_array = new_parameteres_set;
        elements_in_set_variable_array = new_elements_in_set_variable_array;
        elements_in_vector_variable_array = new_elements_in_vector_variable_array;
    }

    Ok((
        parameters_array,
        elements_in_set_variable_array,
        elements_in_vector_variable_array,
    ))
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
                "variable name `{}` is already used",
                name
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
            "vector" => {
                let object_name = util::get_string_by_key(map, "object")?;
                let ob = metadata.get_object_type(&object_name)?;
                metadata.add_vector_variable(name, ob)?;
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
                    "`{:?}` is not a variable type",
                    value
                ))
                .into())
            }
        }
    }
    Ok(metadata)
}

fn get_less_is_better(
    map: &linked_hash_map::LinkedHashMap<yaml_rust::Yaml, yaml_rust::Yaml>,
) -> Result<Option<bool>, util::YamlContentErr> {
    lazy_static! {
        static ref KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("preference");
    }
    match map.get(&KEY) {
        Some(yaml_rust::Yaml::String(value)) if &value[..] == "greater" => Ok(Some(false)),
        Some(yaml_rust::Yaml::String(value)) if &value[..] == "less" => Ok(Some(true)),
        Some(value) => Err(util::YamlContentErr::new(format!(
            "expected `String(\"greater\")` or `String(\"less\")`, found `{:?}`",
            value
        ))),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_load_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 10);
        assert!(result.is_ok());
        let ob1 = result.unwrap();
        let result = metadata.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());
        let ob2 = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i1"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i2"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i3"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c0"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c1"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c2"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c3"));
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er0"), ob1, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er1"), ob1, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er2"), ob1, true);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er3"), ob2, false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir0"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir1"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir2"), true);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir3"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr0"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr1"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr2"), true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr3"), false);
        assert!(result.is_ok());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
s0: [0, 2]
s1: [0, 1]
s2: [0]
s3: []
p0: [0, 2]
p1: [0, 1]
p2: [0]
p3: []
e0: 0
e1: 1
e2: 2
e3: 0
i0: 0
i1: 1
i2: 2
i3: 3
c0: 0
c1: 1
c2: 2
c3: 3
er0: 0
er1: 1
er2: 2
er3: 0
ir0: 0
ir1: 1
ir2: 2
ir3: 3
cr0: 0
cr1: 1
cr2: 2
cr3: 3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let state = load_state_from_yaml(yaml, &metadata);
        let mut s0 = Set::with_capacity(10);
        s0.insert(0);
        s0.insert(2);
        let mut s1 = Set::with_capacity(10);
        s1.insert(0);
        s1.insert(1);
        let mut s2 = Set::with_capacity(10);
        s2.insert(0);
        let s3 = Set::with_capacity(2);
        let expected = State {
            signature_variables: SignatureVariables {
                set_variables: vec![s0, s1, s2, s3],
                vector_variables: vec![vec![0, 2], vec![0, 1], vec![0], vec![]],
                element_variables: vec![0, 1, 2, 0],
                integer_variables: vec![0, 1, 2, 3],
                continuous_variables: vec![0.0, 1.0, 2.0, 3.0],
            },
            resource_variables: ResourceVariables {
                element_variables: vec![0, 1, 2, 0],
                integer_variables: vec![0, 1, 2, 3],
                continuous_variables: vec![0.0, 1.0, 2.0, 3.0],
            },
        };
        assert_eq!(state.unwrap(), expected);
    }

    #[test]
    fn state_load_from_yaml_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 10);
        assert!(result.is_ok());
        let ob1 = result.unwrap();
        let result = metadata.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());
        let ob2 = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i1"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i2"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i3"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c0"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c1"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c2"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c3"));
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er0"), ob1, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er1"), ob1, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er2"), ob1, true);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er3"), ob2, false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir0"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir1"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir2"), true);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir3"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr0"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr1"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr2"), true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr3"), false);
        assert!(result.is_ok());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
s0: [0, 2]
s1: [0, 1]
s2: [0]
s3: []
p0: [0, 2]
p1: [0, 1]
p2: [0]
p3: []
e0: 0
e1: 1
e3: 0
er0: 0
er1: 1
er2: 1
er3: 0
i0: 0
i1: 1
i2: 2
i3: 3
c0: 0
c1: 1
c2: 2
c3: 3
ir0: 0
ir1: 1
ir2: 2
ir3: 3
cr0: 0
cr1: 1
cr2: 2
cr3: 3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let state = load_state_from_yaml(yaml, &metadata);
        assert!(state.is_err());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
s0: [0, 2]
s1: [0, 1]
s2: [0]
s3: []
p0: [0, 2]
p1: [0, 1]
p2: [0]
p3: []
e0: 0
e1: 1
e2: 1
e3: 3
er0: 0
er1: 1
er2: 1
er3: 0
i0: 0
i1: 1
i2: 2
i3: 3
c0: 0
c1: 1
c2: 2
c3: 3
ir0: 0
ir1: 1
ir2: 2
ir3: 3
cr0: 0
cr1: 1
cr2: 2
cr3: 3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let state = load_state_from_yaml(yaml, &metadata);
        assert!(state.is_err());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
s0: [0, 2]
s1: [0, 1]
s2: [0]
s3: []
p0: [0, 2]
p1: [0, 1]
p2: [0]
p3: []
e0: 0
e1: 1
e2: 1
e3: 0
er0: 0
er1: 1
er2: 1
er3: 3
i0: 0
i1: 1
i2: 2
i3: 3
c0: 0
c1: 1
c2: 2
c3: 3
ir0: 0
ir1: 1
ir2: 2
ir3: 3
cr0: 0
cr1: 1
cr2: 2
cr3: 3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let state = load_state_from_yaml(yaml, &metadata);
        assert!(state.is_err());
    }

    #[test]
    fn ground_parameters_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 10);
        assert!(result.is_ok());
        let ob1 = result.unwrap();
        let result = metadata.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());
        let ob2 = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i1"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i2"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i3"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c0"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c1"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c2"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c3"));
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er0"), ob1, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er1"), ob1, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er2"), ob1, true);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er3"), ob2, false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir0"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir1"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir2"), true);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir3"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr0"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr1"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr2"), true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr3"), false);
        assert!(result.is_ok());

        let mut map1 = FxHashMap::default();
        map1.insert(String::from("v0"), 0);
        map1.insert(String::from("v1"), 0);
        let mut map2 = FxHashMap::default();
        map2.insert(String::from("v0"), 0);
        map2.insert(String::from("v1"), 1);
        let mut map3 = FxHashMap::default();
        map3.insert(String::from("v0"), 1);
        map3.insert(String::from("v1"), 0);
        let mut map4 = FxHashMap::default();
        map4.insert(String::from("v0"), 1);
        map4.insert(String::from("v1"), 1);
        let expected_parameters = vec![map1, map2, map3, map4];

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: small
- name: v1
  object: s3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = ground_parameters_from_yaml(&metadata, yaml);
        assert!(result.is_ok());
        let (parameters, elements_in_set_variable_array, elements_in_vector_variable_array) =
            result.unwrap();
        let expected_elements_in_set_variable_array =
            vec![vec![(3, 0)], vec![(3, 1)], vec![(3, 0)], vec![(3, 1)]];
        let expected_elements_in_vector_variable_array = vec![vec![], vec![], vec![], vec![]];
        assert_eq!(parameters, expected_parameters);
        assert_eq!(
            elements_in_set_variable_array,
            expected_elements_in_set_variable_array
        );
        assert_eq!(
            elements_in_vector_variable_array,
            expected_elements_in_vector_variable_array
        );

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: s3
- name: v1
  object: p3
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = ground_parameters_from_yaml(&metadata, yaml);
        assert!(result.is_ok());
        let (parameters, elements_in_set_variable_array, elements_in_vector_variable_array) =
            result.unwrap();
        let expected_elements_in_set_variable_array =
            vec![vec![(3, 0)], vec![(3, 0)], vec![(3, 1)], vec![(3, 1)]];
        let expected_elements_in_vector_variable_array =
            vec![vec![(3, 0)], vec![(3, 1)], vec![(3, 0)], vec![(3, 1)]];
        assert_eq!(parameters, expected_parameters);
        assert_eq!(
            elements_in_set_variable_array,
            expected_elements_in_set_variable_array
        );
        assert_eq!(
            elements_in_vector_variable_array,
            expected_elements_in_vector_variable_array
        );
    }

    #[test]
    fn ground_parameters_from_yaml_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 10);
        assert!(result.is_ok());
        let ob1 = result.unwrap();
        let result = metadata.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());
        let ob2 = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_set_variable(String::from("s3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_vector_variable(String::from("p3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e0"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e1"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e2"), ob1);
        assert!(result.is_ok());
        let result = metadata.add_element_variable(String::from("e3"), ob2);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i1"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i2"));
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i3"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c0"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c1"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c2"));
        assert!(result.is_ok());
        let result = metadata.add_continuous_variable(String::from("c3"));
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er0"), ob1, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er1"), ob1, false);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er2"), ob1, true);
        assert!(result.is_ok());
        let result = metadata.add_element_resource_variable(String::from("er3"), ob2, false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir0"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir1"), false);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir2"), true);
        assert!(result.is_ok());
        let result = metadata.add_integer_resource_variable(String::from("ir3"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr0"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr1"), false);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr2"), true);
        assert!(result.is_ok());
        let result = metadata.add_continuous_resource_variable(String::from("cr3"), false);
        assert!(result.is_ok());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: small
- name: v1
  object: s5
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = ground_parameters_from_yaml(&metadata, yaml);
        assert!(result.is_err());

        let yaml = yaml_rust::YamlLoader::load_from_str(
            r"
- name: v0
  object: small
- name: v0
  object: s5
",
        );
        assert!(yaml.is_ok());
        let yaml = yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = ground_parameters_from_yaml(&metadata, yaml);
        assert!(result.is_err());
    }

    #[test]
    fn state_metadata_load_from_yaml_ok() {
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("n0"), 0);
        let expected = StateMetadata {
            integer_variable_names: vec![String::from("n0")],
            name_to_integer_variable,
            ..Default::default()
        };
        let objects = yaml_rust::Yaml::Array(Vec::new());
        let object_numbers = yaml_rust::Yaml::Hash(linked_hash_map::LinkedHashMap::default());
        let variables = r"
- name: n0
  type: integer
";

        let variables = yaml_rust::YamlLoader::load_from_str(variables);
        assert!(variables.is_ok());
        let variables = variables.unwrap();
        assert_eq!(variables.len(), 1);
        let variables = &variables[0];

        let metadata = load_metadata_from_yaml(&objects, variables, &object_numbers);
        assert!(metadata.is_ok());
        assert_eq!(metadata.unwrap(), expected);

        let mut expected = StateMetadata::default();
        let result = expected.add_object_type(String::from("object"), 10);
        assert!(result.is_ok());
        let ob1 = result.unwrap();
        let result = expected.add_object_type(String::from("small"), 2);
        assert!(result.is_ok());
        let ob2 = result.unwrap();
        let result = expected.add_set_variable(String::from("s0"), ob1);
        assert!(result.is_ok());
        let result = expected.add_set_variable(String::from("s1"), ob1);
        assert!(result.is_ok());
        let result = expected.add_set_variable(String::from("s2"), ob1);
        assert!(result.is_ok());
        let result = expected.add_set_variable(String::from("s3"), ob2);
        assert!(result.is_ok());
        let result = expected.add_vector_variable(String::from("p0"), ob1);
        assert!(result.is_ok());
        let result = expected.add_vector_variable(String::from("p1"), ob1);
        assert!(result.is_ok());
        let result = expected.add_vector_variable(String::from("p2"), ob1);
        assert!(result.is_ok());
        let result = expected.add_vector_variable(String::from("p3"), ob2);
        assert!(result.is_ok());
        let result = expected.add_element_variable(String::from("e0"), ob1);
        assert!(result.is_ok());
        let result = expected.add_element_variable(String::from("e1"), ob1);
        assert!(result.is_ok());
        let result = expected.add_element_variable(String::from("e2"), ob1);
        assert!(result.is_ok());
        let result = expected.add_element_variable(String::from("e3"), ob2);
        assert!(result.is_ok());
        let result = expected.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());
        let result = expected.add_integer_variable(String::from("i1"));
        assert!(result.is_ok());
        let result = expected.add_integer_variable(String::from("i2"));
        assert!(result.is_ok());
        let result = expected.add_integer_variable(String::from("i3"));
        assert!(result.is_ok());
        let result = expected.add_continuous_variable(String::from("c0"));
        assert!(result.is_ok());
        let result = expected.add_continuous_variable(String::from("c1"));
        assert!(result.is_ok());
        let result = expected.add_continuous_variable(String::from("c2"));
        assert!(result.is_ok());
        let result = expected.add_continuous_variable(String::from("c3"));
        assert!(result.is_ok());
        let result = expected.add_element_resource_variable(String::from("er0"), ob1, false);
        assert!(result.is_ok());
        let result = expected.add_element_resource_variable(String::from("er1"), ob1, false);
        assert!(result.is_ok());
        let result = expected.add_element_resource_variable(String::from("er2"), ob1, true);
        assert!(result.is_ok());
        let result = expected.add_element_resource_variable(String::from("er3"), ob2, false);
        assert!(result.is_ok());
        let result = expected.add_integer_resource_variable(String::from("ir0"), false);
        assert!(result.is_ok());
        let result = expected.add_integer_resource_variable(String::from("ir1"), false);
        assert!(result.is_ok());
        let result = expected.add_integer_resource_variable(String::from("ir2"), true);
        assert!(result.is_ok());
        let result = expected.add_integer_resource_variable(String::from("ir3"), false);
        assert!(result.is_ok());
        let result = expected.add_continuous_resource_variable(String::from("cr0"), false);
        assert!(result.is_ok());
        let result = expected.add_continuous_resource_variable(String::from("cr1"), false);
        assert!(result.is_ok());
        let result = expected.add_continuous_resource_variable(String::from("cr2"), true);
        assert!(result.is_ok());
        let result = expected.add_continuous_resource_variable(String::from("cr3"), false);
        assert!(result.is_ok());

        let objects = r"
- object
- small
";
        let variables = r"
- name: s0
  type: set
  object: object
- name: s1
  type: set
  object: object
- name: s2
  type: set
  object: object
- name: s3
  type: set
  object: small
- name: p0
  type: vector
  object: object
- name: p1
  type: vector 
  object: object
- name: p2
  type: vector
  object: object
- name: p3
  type: vector
  object: small
- name: e0
  type: element 
  object: object
- name: e1
  type: element
  object: object
- name: e2
  type: element
  object: object
- name: e3
  type: element
  object: small
- name: i0
  type: integer
- name: i1
  type: integer
- name: i2
  type: integer 
- name: i3
  type: integer
- name: c0
  type: continuous 
- name: c1
  type: continuous
- name: c2
  type: continuous
- name: c3
  type: continuous
- name: er0
  type: element 
  object: object
  preference: greater
- name: er1
  type: element
  object: object
  preference: greater
- name: er2
  type: element
  object: object
  preference: less
- name: er3
  type: element
  object: small
  preference: greater
- name: ir0
  type: integer
  preference: greater
- name: ir1
  type: integer
  preference: greater
- name: ir2
  type: integer 
  preference: less
- name: ir3
  type: integer
  preference: greater
- name: cr0
  type: continuous 
  preference: greater
- name: cr1
  type: continuous
  preference: greater
- name: cr2
  type: continuous
  preference: less
- name: cr3
  type: continuous
  preference: greater
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
        assert!(metadata.is_ok());
        assert_eq!(metadata.unwrap(), expected);
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
