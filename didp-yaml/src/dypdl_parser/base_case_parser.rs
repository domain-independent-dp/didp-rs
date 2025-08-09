use super::grounded_condition_parser::load_grounded_conditions_from_yaml;
use super::load_state_from_yaml;
use super::parse_expression_from_yaml::{parse_continuous_from_yaml, parse_integer_from_yaml};
use crate::util;
use dypdl::prelude::*;
use dypdl::{BaseCase, GroundedCondition, ModelErr, TableRegistry};
use lazy_static::lazy_static;
use linked_hash_map::LinkedHashMap;
use rustc_hash::FxHashMap;
use std::error::Error;

fn load_conditions_from_array(
    array: &Vec<yaml_rust::Yaml>,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
) -> Result<Vec<GroundedCondition>, Box<dyn Error>> {
    let parameters = FxHashMap::default();
    let mut conditions = Vec::new();
    for condition in array {
        let condition = load_grounded_conditions_from_yaml(
            condition,
            metadata,
            functions,
            registry,
            &parameters,
        )?;
        for c in condition {
            match c.condition {
                Condition::Constant(false)
                    if c.elements_in_set_variable.is_empty()
                        && c.elements_in_vector_variable.is_empty() =>
                {
                    return Err(
                        ModelErr::new(String::from("a base case is never satisfied")).into(),
                    )
                }
                Condition::Constant(true) => {}
                _ => conditions.push(c),
            }
        }
    }
    Ok(conditions)
}

fn load_base_case_from_hash(
    map: &LinkedHashMap<yaml_rust::Yaml, yaml_rust::Yaml>,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    cost_type: &CostType,
) -> Result<BaseCase, Box<dyn Error>> {
    lazy_static! {
        static ref CONDITIONS_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("conditions");
        static ref COST_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("cost");
    }

    if let Some(array) = map.get(&CONDITIONS_KEY) {
        let array = util::get_array(array)?;
        let conditions = load_conditions_from_array(array, metadata, functions, registry)?;
        let parameters = FxHashMap::default();

        match map.get(&COST_KEY) {
            Some(cost) => match cost_type {
                CostType::Integer => {
                    let cost =
                        parse_integer_from_yaml(cost, metadata, functions, registry, &parameters)?;
                    Ok(BaseCase::with_cost(conditions, cost))
                }
                CostType::Continuous => {
                    let cost = parse_continuous_from_yaml(
                        cost,
                        metadata,
                        functions,
                        registry,
                        &parameters,
                    )?;
                    Ok(BaseCase::with_cost(conditions, cost.simplify(registry)))
                }
            },
            None => Ok(BaseCase::from(conditions)),
        }
    } else {
        Err(util::YamlContentErr::new(String::from("missing key conditions in a base case")).into())
    }
}

/// Returns a base case loaded from YAML.
///
/// # Errors
///
/// If the format is invalid.
pub fn load_base_case_from_yaml(
    value: &yaml_rust::Yaml,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    cost_type: &CostType,
) -> Result<BaseCase, Box<dyn Error>> {
    match value {
        yaml_rust::Yaml::Array(array) => {
            let conditions = load_conditions_from_array(array, metadata, functions, registry)?;
            Ok(BaseCase::from(conditions))
        }
        yaml_rust::Yaml::Hash(map) => {
            load_base_case_from_hash(map, metadata, functions, registry, cost_type)
        }
        _ => Err(util::YamlContentErr::new(format!("expected Array, found `{value:?}`")).into()),
    }
}

/// Returns a base state loaded from YAML.
///
/// # Errors
///
/// If the format is invalid.
pub fn load_base_state_from_yaml(
    value: &yaml_rust::Yaml,
    metadata: &StateMetadata,
    cost_type: &CostType,
) -> Result<(State, Option<CostExpression>), Box<dyn Error>> {
    if let Ok(state) = load_state_from_yaml(value, metadata) {
        Ok((state, None))
    } else {
        match value {
            yaml_rust::Yaml::Hash(map) => {
                lazy_static! {
                    static ref STATE_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("state");
                    static ref COST_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("cost");
                }

                if let Some(state) = map.get(&STATE_KEY) {
                    let state = load_state_from_yaml(state, metadata)?;

                    if let Some(cost) = map.get(&COST_KEY) {
                        match cost_type {
                            CostType::Integer => {
                                let cost = util::get_numeric(cost)?;
                                Ok((
                                    state,
                                    Some(CostExpression::from(IntegerExpression::Constant(cost))),
                                ))
                            }
                            CostType::Continuous => {
                                let cost = util::get_numeric(cost)?;
                                Ok((
                                    state,
                                    Some(CostExpression::from(ContinuousExpression::Constant(
                                        cost,
                                    ))),
                                ))
                            }
                        }
                    } else {
                        Ok((state, None))
                    }
                } else {
                    Err(util::YamlContentErr::new(String::from(
                        "missing key state in a base state",
                    ))
                    .into())
                }
            }
            _ => Err(util::YamlContentErr::new(format!(
                "expected Map for a base state, found `{value:?}`",
            ))
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::GroundedCondition;

    #[test]
    fn load_from_yaml_ok() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let expected = BaseCase::from(vec![GroundedCondition {
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
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_multiple_ok() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let expected = BaseCase::from(vec![GroundedCondition {
            condition: Condition::ComparisonI(
                ComparisonOperator::Ge,
                Box::new(IntegerExpression::Variable(0)),
                Box::new(IntegerExpression::Constant(0)),
            ),
            ..Default::default()
        }]);

        let base_case = yaml_rust::YamlLoader::load_from_str(r"[(>= i0 0), (= i0 i0)]");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_forall_ok() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let base_case = yaml_rust::YamlLoader::load_from_str(
            r"[{ condition: (= e 1), forall: [ {name: e, object: s0} ] }]",
        );
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_ok());
        let expected = BaseCase::from(vec![GroundedCondition {
            condition: Condition::Constant(false),
            elements_in_set_variable: vec![(0, 0)],
            ..Default::default()
        }]);
        assert_eq!(base_case.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_with_cost_none_ok() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let expected = BaseCase::from(vec![GroundedCondition {
            condition: Condition::ComparisonI(
                ComparisonOperator::Ge,
                Box::new(IntegerExpression::Variable(0)),
                Box::new(IntegerExpression::Constant(0)),
            ),
            ..Default::default()
        }]);

        let base_case =
            yaml_rust::YamlLoader::load_from_str(r"{ conditions: [(>= i0 0), (= i0 i0)] }");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_with_cost_integer_ok() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let expected = BaseCase::with_cost(
            vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            IntegerExpression::Variable(0),
        );

        let base_case = yaml_rust::YamlLoader::load_from_str(
            r"{ conditions: [(>= i0 0), (= i0 i0)], cost: i0 }",
        );
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_with_continuous_continuous_ok() {
        let cost_type = CostType::Continuous;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let expected = BaseCase::with_cost(
            vec![GroundedCondition {
                condition: Condition::ComparisonI(
                    ComparisonOperator::Ge,
                    Box::new(IntegerExpression::Variable(0)),
                    Box::new(IntegerExpression::Constant(0)),
                ),
                ..Default::default()
            }],
            ContinuousExpression::Constant(1.0),
        );

        let base_case = yaml_rust::YamlLoader::load_from_str(
            r"{ conditions: [(>= i0 0), (= i0 i0)], cost: 1.0 }",
        );
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let base_case = yaml_rust::YamlLoader::load_from_str(r"(>= i0 0)");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_err());
    }

    #[test]
    fn load_from_yaml_never_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let base_case = yaml_rust::YamlLoader::load_from_str(r"[(>= i0 0), (= i0 i0), (= 1 2)]");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_err());
    }

    #[test]
    fn load_from_yaml_no_condition_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let base_case = yaml_rust::YamlLoader::load_from_str(r"{ cost: i0 }");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_err());
    }

    #[test]
    fn load_from_yaml_with_integer_cost_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let base_case =
            yaml_rust::YamlLoader::load_from_str(r"{ conditions: [(>= i0 0)], cost: 1.0 }");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_err());
    }

    #[test]
    fn load_from_yaml_with_continuous_cost_err() {
        let cost_type = CostType::Continuous;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let base_case =
            yaml_rust::YamlLoader::load_from_str(r"{ conditions: [(>= i0 0)], cost: (+ i1 2.5) }");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_err());
    }

    #[test]
    fn load_from_yaml_no_array_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let base_case =
            yaml_rust::YamlLoader::load_from_str(r"{ conditions: (>= i0 0), cost: i0 }");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_err());
    }

    #[test]
    fn load_from_yaml_no_proper_conditions_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type(String::from("object"), 2);
        assert!(result.is_ok());
        let ob = result.unwrap();
        let result = metadata.add_set_variable(String::from("s0"), ob);
        assert!(result.is_ok());
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let functions = StateFunctions::default();

        let registry = TableRegistry::default();

        let base_case = yaml_rust::YamlLoader::load_from_str(
            r"{ conditions: [(>= i0 0), (>= i1 0)], cost: i0 }",
        );
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            load_base_case_from_yaml(&base_case[0], &metadata, &functions, &registry, &cost_type);
        assert!(base_case.is_err());
    }

    #[test]
    fn load_base_state_from_yaml_ok() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let base_state = yaml_rust::YamlLoader::load_from_str(r"{ i0: 0 }");
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = load_base_state_from_yaml(&base_state[0], &metadata, &cost_type);
        assert!(base_state.is_ok());
        assert_eq!(base_state.unwrap(), (state, None));
    }

    #[test]
    fn load_base_state_from_yaml_with_cost_int_ok() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let base_state = yaml_rust::YamlLoader::load_from_str(r"{ state: { i0: 0 }, cost: 1 }");
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = load_base_state_from_yaml(&base_state[0], &metadata, &cost_type);
        assert!(base_state.is_ok());
        assert_eq!(
            base_state.unwrap(),
            (
                state,
                Some(CostExpression::from(IntegerExpression::Constant(1)))
            )
        );
    }

    #[test]
    fn load_base_state_from_yaml_with_cost_continuous_ok() {
        let cost_type = CostType::Continuous;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let base_state = yaml_rust::YamlLoader::load_from_str(r"{ state: { i0: 0 }, cost: 1.5 }");
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = load_base_state_from_yaml(&base_state[0], &metadata, &cost_type);
        assert!(base_state.is_ok());
        assert_eq!(
            base_state.unwrap(),
            (
                state,
                Some(CostExpression::from(ContinuousExpression::Constant(1.5)))
            )
        );
    }

    #[test]
    fn load_base_state_from_yaml_without_cost_ok() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let state = State {
            signature_variables: SignatureVariables {
                integer_variables: vec![0],
                ..Default::default()
            },
            ..Default::default()
        };

        let base_state = yaml_rust::YamlLoader::load_from_str(r"{ state: { i0: 0 } }");
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = load_base_state_from_yaml(&base_state[0], &metadata, &cost_type);
        assert!(base_state.is_ok());
        assert_eq!(base_state.unwrap(), (state, None));
    }

    #[test]
    fn load_base_state_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let base_state = yaml_rust::YamlLoader::load_from_str(r"{ i1: 0 }");
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = load_base_state_from_yaml(&base_state[0], &metadata, &cost_type);
        assert!(base_state.is_err());
    }

    #[test]
    fn load_base_state_without_cost_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let base_state = yaml_rust::YamlLoader::load_from_str(r"{ state: { i1: 0 } }");
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = load_base_state_from_yaml(&base_state[0], &metadata, &cost_type);
        assert!(base_state.is_err());
    }

    #[test]
    fn load_base_state_no_state_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let base_state = yaml_rust::YamlLoader::load_from_str(r"{ cost: 1 }");
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = load_base_state_from_yaml(&base_state[0], &metadata, &cost_type);
        assert!(base_state.is_err());
    }

    #[test]
    fn load_base_state_wit_cost_integer_err() {
        let cost_type = CostType::Integer;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let base_state = yaml_rust::YamlLoader::load_from_str(r"{ state: { i1: 0 }, cost: 1.5 }");
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = load_base_state_from_yaml(&base_state[0], &metadata, &cost_type);
        assert!(base_state.is_err());
    }

    #[test]
    fn load_base_state_wit_cost_continuous_err() {
        let cost_type = CostType::Continuous;
        let mut metadata = StateMetadata::default();
        let result = metadata.add_integer_variable(String::from("i0"));
        assert!(result.is_ok());

        let base_state = yaml_rust::YamlLoader::load_from_str(r"{ state: { i1: 0 }, cost: i0 }");
        assert!(base_state.is_ok());
        let base_state = base_state.unwrap();
        assert_eq!(base_state.len(), 1);
        let base_state = load_base_state_from_yaml(&base_state[0], &metadata, &cost_type);
        assert!(base_state.is_err());
    }
}
