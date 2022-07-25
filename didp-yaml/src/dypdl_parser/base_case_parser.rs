use super::grounded_condition_parser::load_grounded_conditions_from_yaml;
use crate::util;
use dypdl::expression::Condition;
use dypdl::ModelErr;
use dypdl::{BaseCase, StateMetadata, TableRegistry};
use rustc_hash::FxHashMap;
use std::error::Error;

/// Returns a base case loaded from YAML.
///
/// # Errors
///
/// if the format is invalid.
pub fn load_base_case_from_yaml(
    value: &yaml_rust::Yaml,
    metadata: &StateMetadata,
    registry: &TableRegistry,
) -> Result<BaseCase, Box<dyn Error>> {
    let array = match value {
        yaml_rust::Yaml::Array(array) => array,
        _ => {
            return Err(
                util::YamlContentErr::new(format!("expected Array, found `{:?}`", value)).into(),
            )
        }
    };
    let parameters = FxHashMap::default();
    let mut conditions = Vec::new();
    for condition in array {
        let condition =
            load_grounded_conditions_from_yaml(condition, metadata, registry, &parameters)?;
        for c in condition {
            match c.condition {
                Condition::Constant(false)
                    if c.elements_in_set_variable.is_empty()
                        && c.elements_in_vector_variable.is_empty() =>
                {
                    return Err(
                        ModelErr::new(String::from("terminal condition never satisfied")).into(),
                    )
                }
                Condition::Constant(true) => {}
                _ => conditions.push(c),
            }
        }
    }
    Ok(BaseCase::new(conditions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::GroundedCondition;

    #[test]
    fn load_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let registry = TableRegistry::default();

        let expected = BaseCase::new(vec![GroundedCondition {
            condition: Condition::ComparisonI(
                ComparisonOperator::Ge,
                Box::new(IntegerExpression::Variable(0)),
                Box::new(IntegerExpression::Constant(0)),
            ),
            ..Default::default()
        }]);

        let base_case = yaml_rust::YamlLoader::load_from_str(r"[(>= i0 0)]");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case = load_base_case_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);

        let base_case = yaml_rust::YamlLoader::load_from_str(r"[(>= i0 0), (= i0 i0)]");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case = load_base_case_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);

        let base_case = yaml_rust::YamlLoader::load_from_str(
            r"[{ condition: (= e 1), forall: [ {name: e, object: s0} ] }]",
        );
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case = load_base_case_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_ok());
        let expected = BaseCase::new(vec![GroundedCondition {
            condition: Condition::Constant(false),
            elements_in_set_variable: vec![(0, 0)],
            ..Default::default()
        }]);
        assert_eq!(base_case.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_err() {
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let registry = TableRegistry::default();

        let base_case = yaml_rust::YamlLoader::load_from_str(r"(>= i0 0)");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case = load_base_case_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_err());

        let base_case = yaml_rust::YamlLoader::load_from_str(r"[(>= i0 0), (= i0 i0), (= 1 2)]");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case = load_base_case_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_err());
    }
}
