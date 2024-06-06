use dypdl::StateMetadata;
use yaml_rust::{yaml::Hash, Yaml};

pub fn integer_variable_to_yaml(state_metadata: &StateMetadata, index: usize) -> Yaml {
    let mut variable_hash = Hash::new();

    variable_hash.insert(
        Yaml::from_str("name"),
        Yaml::String(state_metadata.integer_variable_names[index].clone()),
    );
    variable_hash.insert(Yaml::from_str("type"), Yaml::from_str("integer"));

    Yaml::Hash(variable_hash)
}

pub fn continuous_variable_to_yaml(state_metadata: &StateMetadata, index: usize) -> Yaml {
    let mut variable_hash = Hash::new();

    variable_hash.insert(
        Yaml::from_str("name"),
        Yaml::String(state_metadata.continuous_variable_names[index].clone()),
    );
    variable_hash.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));

    Yaml::Hash(variable_hash)
}

pub fn element_variable_to_yaml(state_metadata: &StateMetadata, index: usize) -> Yaml {
    let mut variable_hash = Hash::new();

    variable_hash.insert(
        Yaml::from_str("name"),
        Yaml::String(state_metadata.element_variable_names[index].clone()),
    );
    variable_hash.insert(Yaml::from_str("type"), Yaml::from_str("element"));
    variable_hash.insert(
        Yaml::from_str("object"),
        Yaml::String(
            state_metadata.object_type_names[state_metadata.element_variable_to_object[index]]
                .clone(),
        ),
    );

    Yaml::Hash(variable_hash)
}

pub fn set_variable_to_yaml(state_metadata: &StateMetadata, index: usize) -> Yaml {
    let mut variable_hash = Hash::new();

    variable_hash.insert(
        Yaml::from_str("name"),
        Yaml::String(state_metadata.set_variable_names[index].clone()),
    );
    variable_hash.insert(Yaml::from_str("type"), Yaml::from_str("set"));
    variable_hash.insert(
        Yaml::from_str("object"),
        Yaml::String(
            state_metadata.object_type_names[state_metadata.set_variable_to_object[index]].clone(),
        ),
    );

    Yaml::Hash(variable_hash)
}

pub fn integer_resource_variable_to_yaml(state_metadata: &StateMetadata, index: usize) -> Yaml {
    let mut variable_hash = Hash::new();

    variable_hash.insert(
        Yaml::from_str("name"),
        Yaml::String(state_metadata.integer_resource_variable_names[index].clone()),
    );
    variable_hash.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
    variable_hash.insert(
        Yaml::from_str("preference"),
        Yaml::from_str(if state_metadata.integer_less_is_better[index] {
            "less"
        } else {
            "greater"
        }),
    );

    Yaml::Hash(variable_hash)
}

pub fn continuous_resource_variable_to_yaml(state_metadata: &StateMetadata, index: usize) -> Yaml {
    let mut variable_hash = Hash::new();

    variable_hash.insert(
        Yaml::from_str("name"),
        Yaml::String(state_metadata.continuous_resource_variable_names[index].clone()),
    );
    variable_hash.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));
    variable_hash.insert(
        Yaml::from_str("preference"),
        Yaml::from_str(if state_metadata.continuous_less_is_better[index] {
            "less"
        } else {
            "greater"
        }),
    );

    Yaml::Hash(variable_hash)
}

pub fn element_resource_variable_to_yaml(state_metadata: &StateMetadata, index: usize) -> Yaml {
    let mut variable_hash = Hash::new();

    variable_hash.insert(
        Yaml::from_str("name"),
        Yaml::String(state_metadata.element_resource_variable_names[index].clone()),
    );
    variable_hash.insert(Yaml::from_str("type"), Yaml::from_str("element"));
    variable_hash.insert(
        Yaml::from_str("object"),
        Yaml::String(
            state_metadata.object_type_names
                [state_metadata.element_resource_variable_to_object[index]]
                .clone(),
        ),
    );
    variable_hash.insert(
        Yaml::from_str("preference"),
        Yaml::from_str(if state_metadata.element_less_is_better[index] {
            "less"
        } else {
            "greater"
        }),
    );

    Yaml::Hash(variable_hash)
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;
    use yaml_rust::{yaml::Hash, Yaml};

    use crate::dypdl_yaml_dumper::variable_to_yaml::*;

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

    #[test]
    fn integer_variable_to_yaml_ok() {
        let metadata = generate_metadata();
        let result = integer_variable_to_yaml(&metadata, 0);
        let mut expected_hash = Hash::new();
        expected_hash.insert(Yaml::from_str("name"), Yaml::from_str("i0"));
        expected_hash.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
        assert_eq!(result, Yaml::Hash(expected_hash));
    }

    #[test]
    fn continuous_variable_to_yaml_ok() {
        let metadata = generate_metadata();
        let result = continuous_variable_to_yaml(&metadata, 1);
        let mut expected_hash = Hash::new();
        expected_hash.insert(Yaml::from_str("name"), Yaml::from_str("c1"));
        expected_hash.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));
        assert_eq!(result, Yaml::Hash(expected_hash));
    }

    #[test]
    fn element_variable_to_yaml_ok() {
        let metadata = generate_metadata();
        let result = element_variable_to_yaml(&metadata, 0);
        let mut expected_hash = Hash::new();
        expected_hash.insert(Yaml::from_str("name"), Yaml::from_str("e0"));
        expected_hash.insert(Yaml::from_str("type"), Yaml::from_str("element"));
        expected_hash.insert(Yaml::from_str("object"), Yaml::from_str("object"));
        assert_eq!(result, Yaml::Hash(expected_hash));
    }

    #[test]
    fn set_variable_to_yaml_ok() {
        let metadata: StateMetadata = generate_metadata();
        let result = set_variable_to_yaml(&metadata, 1);
        let mut expected_hash = Hash::new();
        expected_hash.insert(Yaml::from_str("name"), Yaml::from_str("s1"));
        expected_hash.insert(Yaml::from_str("type"), Yaml::from_str("set"));
        expected_hash.insert(Yaml::from_str("object"), Yaml::from_str("object"));
        assert_eq!(result, Yaml::Hash(expected_hash));
    }

    #[test]
    fn integer_resource_variable_to_yaml_ok() {
        let metadata: StateMetadata = generate_metadata();
        let result = integer_resource_variable_to_yaml(&metadata, 1);
        let mut expected_hash = Hash::new();
        expected_hash.insert(Yaml::from_str("name"), Yaml::from_str("ir1"));
        expected_hash.insert(Yaml::from_str("type"), Yaml::from_str("integer"));
        expected_hash.insert(Yaml::from_str("preference"), Yaml::from_str("less"));
        assert_eq!(result, Yaml::Hash(expected_hash));
    }

    #[test]
    fn continuous_resource_variable_to_yaml_ok() {
        let metadata: StateMetadata = generate_metadata();
        let result = continuous_resource_variable_to_yaml(&metadata, 0);
        let mut expected_hash = Hash::new();
        expected_hash.insert(Yaml::from_str("name"), Yaml::from_str("cr0"));
        expected_hash.insert(Yaml::from_str("type"), Yaml::from_str("continuous"));
        expected_hash.insert(Yaml::from_str("preference"), Yaml::from_str("greater"));
        assert_eq!(result, Yaml::Hash(expected_hash));
    }

    #[test]
    fn element_resource_variable_to_yaml_ok() {
        let metadata: StateMetadata = generate_metadata();
        let result = element_resource_variable_to_yaml(&metadata, 1);
        let mut expected_hash = Hash::new();
        expected_hash.insert(Yaml::from_str("name"), Yaml::from_str("er1"));
        expected_hash.insert(Yaml::from_str("type"), Yaml::from_str("element"));
        expected_hash.insert(Yaml::from_str("object"), Yaml::from_str("object"));
        expected_hash.insert(Yaml::from_str("preference"), Yaml::from_str("less"));
        assert_eq!(result, Yaml::Hash(expected_hash));
    }
}
