use dypdl::expression::Condition;
use dypdl::{BaseCase, StateMetadata, TableRegistry};

use yaml_rust::yaml::Array;
use yaml_rust::{yaml::Hash, Yaml};

use super::ToYamlString;

pub fn base_case_to_yaml(
    bc: &BaseCase,
    state_metadata: &StateMetadata,
    table_registry: &TableRegistry,
) -> Result<Yaml, &'static str> {
    // The base case is a Hash if it has a cost, otherwise it is an Array.
    if let Some(cost) = &bc.cost {
        let mut condition_hash = Hash::new();
        let mut condition_array = Array::new();
        for condition in &bc.conditions {
            condition_array.push(Yaml::String(
                Condition::from(condition.clone())
                    .to_yaml_string(state_metadata, table_registry)?,
            ));
        }
        condition_hash.insert(Yaml::from_str("conditions"), Yaml::Array(condition_array));
        condition_hash.insert(
            Yaml::from_str("cost"),
            Yaml::String(cost.to_yaml_string(state_metadata, table_registry)?),
        );

        Ok(Yaml::Hash(condition_hash))
    } else {
        let mut condition_array = Array::new();
        for condition in &bc.conditions {
            condition_array.push(Yaml::String(
                Condition::from(condition.clone())
                    .to_yaml_string(state_metadata, table_registry)?,
            ));
        }
        Ok(Yaml::Array(condition_array))
    }
}

#[cfg(test)]
mod tests {
    use dypdl::{expression::Condition, Model, StateMetadata, TableRegistry};
    use yaml_rust::{yaml::Hash, Yaml};

    use super::base_case_to_yaml;

    #[test]
    fn base_case_with_cost_to_yaml_ok() {
        let mut model = Model::default();
        let result = model.add_base_case_with_cost(vec![Condition::Constant(true)], 1);
        assert!(result.is_ok(), "Cannot add the base case");

        let base_case = &model.base_cases[0];
        let dumped_yaml = base_case_to_yaml(
            base_case,
            &StateMetadata::default(),
            &TableRegistry::default(),
        );
        assert!(
            dumped_yaml.is_ok(),
            "Cannot construct Yaml for the base case"
        );

        let mut hash = Hash::new();
        hash.insert(
            Yaml::from_str("conditions"),
            Yaml::Array(vec![Yaml::String("true".to_owned())]),
        );
        hash.insert(Yaml::from_str("cost"), Yaml::String("1".to_owned()));

        let expected_yaml = Yaml::Hash(hash);
        assert_eq!(dumped_yaml.unwrap(), expected_yaml);
    }

    #[test]
    fn base_case_without_cost_to_yaml_ok() {
        let mut model = Model::default();
        let result =
            model.add_base_case(vec![Condition::Constant(true), Condition::Constant(false)]);
        assert!(result.is_ok(), "Cannot add the base case");

        let base_case = &model.base_cases[0];
        let dumped_yaml = base_case_to_yaml(
            base_case,
            &StateMetadata::default(),
            &TableRegistry::default(),
        );
        assert!(
            dumped_yaml.is_ok(),
            "Cannot construct Yaml for the base case"
        );

        let array = vec![
            Yaml::String("true".to_owned()),
            Yaml::String("false".to_owned()),
        ];

        let expected_yaml = Yaml::Array(array);
        assert_eq!(dumped_yaml.unwrap(), expected_yaml);
    }
}
