use dypdl::State;

use super::ToYaml;
use yaml_rust::{yaml::Hash, Yaml};

pub fn state_to_yaml(
    state: &State,
    state_metadata: &dypdl::prelude::StateMetadata,
) -> Result<Yaml, Box<dyn std::error::Error>> {
    let mut hash = Hash::new();

    macro_rules! insert_variables {
        ($($name_field:ident).+, $($value_field:ident).+) => {
            for i in 0..state. $( $value_field ).+ .len(){
                hash.insert(Yaml::from_str(&state_metadata. $( $name_field ).+ [i]),
                            state. $( $value_field ).+ [i].to_yaml()?);
            }
        }
    }

    insert_variables!(
        integer_variable_names,
        signature_variables.integer_variables
    );
    insert_variables!(
        continuous_variable_names,
        signature_variables.continuous_variables
    );
    insert_variables!(
        element_variable_names,
        signature_variables.element_variables
    );
    insert_variables!(set_variable_names, signature_variables.set_variables);

    insert_variables!(
        integer_resource_variable_names,
        resource_variables.integer_variables
    );
    insert_variables!(
        continuous_resource_variable_names,
        resource_variables.continuous_variables
    );
    insert_variables!(
        element_resource_variable_names,
        resource_variables.element_variables
    );

    Ok(Yaml::Hash(hash))
}

#[cfg(test)]
mod tests {
    use std::vec;

    use dypdl::{ResourceVariables, Set, SignatureVariables};
    use rustc_hash::FxHashMap;
    use yaml_rust::{yaml::Hash, Yaml};

    use super::state_to_yaml;

    fn generate_metadata() -> dypdl::StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec!["s0".to_string(), "s1".to_string()];
        let mut name_to_set_variable = FxHashMap::default();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        let set_variable_to_object = vec![0, 0];

        let element_variable_names = vec!["e0".to_string(), "e1".to_string()];
        let mut name_to_element_variable = FxHashMap::default();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        let element_variable_to_object = vec![0, 0];

        let element_resource_variable_names = vec!["er0".to_string(), "er1".to_string()];
        let mut name_to_element_resource_variable = FxHashMap::default();
        name_to_element_resource_variable.insert("er0".to_string(), 0);
        name_to_element_resource_variable.insert("er1".to_string(), 1);
        let element_resource_variable_to_object = vec![0, 0];

        let integer_variable_names = vec!["i0".to_string(), "i1".to_string()];
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert("i0".to_string(), 0);
        name_to_integer_variable.insert("i1".to_string(), 1);

        let integer_resource_variable_names = vec!["ir0".to_string(), "ir1".to_string()];
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert("ir0".to_string(), 0);
        name_to_integer_resource_variable.insert("ir1".to_string(), 1);

        let continuous_variable_names = vec!["c0".to_string(), "c1".to_string()];
        let mut name_to_continuous_variable = FxHashMap::default();
        name_to_continuous_variable.insert("c0".to_string(), 0);
        name_to_continuous_variable.insert("c1".to_string(), 1);

        let continuous_resource_variable_names = vec!["cr0".to_string(), "cr1".to_string()];
        let mut name_to_continuous_resource_variable = FxHashMap::default();
        name_to_continuous_resource_variable.insert("cr0".to_string(), 0);
        name_to_continuous_resource_variable.insert("cr1".to_string(), 1);

        dypdl::StateMetadata {
            object_type_names: object_names,
            name_to_object_type: name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            vector_variable_names: vec![],
            name_to_vector_variable: FxHashMap::default(),
            vector_variable_to_object: vec![],
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            element_resource_variable_names,
            name_to_element_resource_variable,
            element_less_is_better: vec![false, true],
            element_resource_variable_to_object,
            integer_variable_names,
            name_to_integer_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, true],
            continuous_variable_names,
            name_to_continuous_variable,
            continuous_resource_variable_names,
            name_to_continuous_resource_variable,
            continuous_less_is_better: vec![false, true],
        }
    }

    fn generate_state() -> dypdl::State {
        dypdl::State {
            signature_variables: SignatureVariables {
                integer_variables: vec![1, 1],
                set_variables: vec![Set::with_capacity(10), Set::with_capacity(10)],
                element_variables: vec![9, 8],
                continuous_variables: vec![1.0, 1.1],
                ..Default::default()
            },
            resource_variables: ResourceVariables {
                integer_variables: vec![2, 2],
                element_variables: vec![4, 5],
                continuous_variables: vec![3.5, 4.5],
            },
        }
    }

    #[test]
    fn state_to_yaml_test() {
        let state_metadata = generate_metadata();
        let state = generate_state();

        let result = state_to_yaml(&state, &state_metadata);
        assert!(result.is_ok());

        let mut expected_yaml = Hash::new();
        expected_yaml.insert(Yaml::from_str("i0"), Yaml::Integer(1));
        expected_yaml.insert(Yaml::from_str("i1"), Yaml::Integer(1));
        expected_yaml.insert(Yaml::from_str("c0"), Yaml::Real((1.0).to_string()));
        expected_yaml.insert(Yaml::from_str("c1"), Yaml::Real((1.1).to_string()));
        expected_yaml.insert(Yaml::from_str("e0"), Yaml::Integer(9));
        expected_yaml.insert(Yaml::from_str("e1"), Yaml::Integer(8));
        expected_yaml.insert(Yaml::from_str("s0"), Yaml::Array(vec![]));
        expected_yaml.insert(Yaml::from_str("s1"), Yaml::Array(vec![]));

        expected_yaml.insert(Yaml::from_str("ir0"), Yaml::Integer(2));
        expected_yaml.insert(Yaml::from_str("ir1"), Yaml::Integer(2));
        expected_yaml.insert(Yaml::from_str("cr0"), Yaml::Real((3.5).to_string()));
        expected_yaml.insert(Yaml::from_str("cr1"), Yaml::Real((4.5).to_string()));
        expected_yaml.insert(Yaml::from_str("er0"), Yaml::Integer(4));
        expected_yaml.insert(Yaml::from_str("er1"), Yaml::Integer(5));

        assert_eq!(result.unwrap(), Yaml::Hash(expected_yaml));
    }
}
