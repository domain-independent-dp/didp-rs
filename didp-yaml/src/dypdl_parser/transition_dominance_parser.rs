use crate::dypdl_parser::{self, expression_parser::ParseErr, state_parser, util};
use dypdl::prelude::Condition;
use dypdl::{StateFunctions, StateMetadata, TableRegistry, Transition, TransitionDominance};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::error::Error;
use yaml_rust::Yaml;

fn get_transition_index_from_full_name(
    name: &str,
    parameters: &BTreeMap<String, usize>,
    transitions: &[Transition],
) -> Option<usize> {
    for (id, t) in transitions.iter().enumerate() {
        if parameters.len() != t.parameter_names.len() {
            continue;
        }

        let mut full_name = name.to_string();

        for (name, value) in t.parameter_names.iter().zip(parameters.values()) {
            full_name += &format!(" {name}:{value}");
        }

        if t.get_full_name() == full_name {
            return Some(id);
        }
    }

    None
}

pub fn load_transition_dominance_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    forward_transitions: &[Transition],
    backward_transitions: &[Transition],
) -> Result<Vec<TransitionDominance>, Box<dyn Error>> {
    let array = util::get_array(value)?;
    let mut transition_dominance = Vec::with_capacity(array.len());

    for value in array {
        let map = util::get_map(value)?;
        let dominating = util::get_yaml_by_key(map, "dominating")?;
        let dominating = util::get_map(dominating)?;
        let dominating_name = util::get_string_by_key(dominating, "name")?;

        let dominating_parameters_array = match dominating.get(&Yaml::from_str("parameters")) {
            Some(value) => state_parser::ground_static_parameters_from_yaml(metadata, value)?,
            None => vec![BTreeMap::default()],
        };

        let dominated = util::get_yaml_by_key(map, "dominated")?;
        let dominated = util::get_map(dominated)?;
        let dominated_name = util::get_string_by_key(dominated, "name")?;

        let dominated_parameters_array = match dominated.get(&Yaml::from_str("parameters")) {
            Some(value) => state_parser::ground_static_parameters_from_yaml(metadata, value)?,
            None => vec![BTreeMap::default()],
        };

        let condition_array = if let Some(conditions) = map.get(&Yaml::from_str("conditions")) {
            util::get_array(conditions)?.clone()
        } else {
            Vec::default()
        };

        for dominating_parameters in &dominating_parameters_array {
            'dominated: for dominated_parameters in &dominated_parameters_array {
                let (dominating, transitions, backward) = if let Some(dominating) =
                    get_transition_index_from_full_name(
                        &dominating_name,
                        dominating_parameters,
                        forward_transitions,
                    ) {
                    (dominating, forward_transitions, false)
                } else if let Some(dominating) = get_transition_index_from_full_name(
                    &dominating_name,
                    dominating_parameters,
                    backward_transitions,
                ) {
                    (dominating, backward_transitions, true)
                } else {
                    return Err(Box::new(ParseErr::new(format!(
                        "dominating transition `{dominating_name}` not found in forward or backward transitions",
                    ))));
                };

                let dominated = if let Some(dominated) = get_transition_index_from_full_name(
                    &dominated_name,
                    dominated_parameters,
                    transitions,
                ) {
                    dominated
                } else {
                    return Err(Box::new(ParseErr::new(format!(
                        "dominated transition `{dominated_name}` not found",
                    ))));
                };

                if dominating == dominated {
                    continue;
                }

                let mut parameters = FxHashMap::default();

                for (key, value) in dominating_parameters.iter() {
                    parameters.insert(key.clone(), *value);
                }

                for (key, value) in dominated_parameters.iter() {
                    if parameters.insert(key.clone(), *value).is_some() {
                        return Err(Box::new(ParseErr::new(format!(
                            "parameter `{key}` is defined in both dominating and dominated",
                        ))));
                    }
                }

                let mut conditions = Vec::default();

                for value in &condition_array {
                    let grounded = dypdl_parser::load_grounded_conditions_from_yaml(
                        value,
                        metadata,
                        functions,
                        registry,
                        &parameters,
                    )?;

                    for c in grounded {
                        // Skip a condition that are always true
                        if c.condition == Condition::Constant(true) {
                            continue;
                        }

                        // Skip dominance if the condition is always false
                        if c.condition == Condition::Constant(false)
                            && c.elements_in_set_variable.is_empty()
                            && c.elements_in_vector_variable.is_empty()
                        {
                            continue 'dominated;
                        }

                        conditions.push(c);
                    }
                }

                transition_dominance.push(TransitionDominance {
                    dominating,
                    dominated,
                    backward,
                    conditions,
                })
            }
        }
    }

    Ok(transition_dominance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::prelude::ComparisonOperator;
    use yaml_rust::YamlLoader;

    #[test]
    fn test_load_transition_dominance_from_yaml_ok() {
        let metadata = StateMetadata::default();
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                ..Default::default()
            },
        ];
        let backward_transitions = vec![
            Transition {
                name: "transition3".to_string(),
                ..Default::default()
            },
            Transition {
                name: "transition4".to_string(),
                ..Default::default()
            },
        ];

        let yaml = r"
        - dominating:
            name: transition1
          dominated:
            name: transition2
        - dominating:
            name: transition4
          dominated:
            name: transition3
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_ok());
        let transition_dominance = result.unwrap();
        assert_eq!(transition_dominance.len(), 2);

        assert_eq!(
            transition_dominance[0],
            TransitionDominance {
                dominating: 0,
                dominated: 1,
                backward: false,
                conditions: Vec::new(),
            }
        );
        assert_eq!(
            transition_dominance[1],
            TransitionDominance {
                dominating: 1,
                dominated: 0,
                backward: true,
                conditions: Vec::new(),
            }
        );
    }

    #[test]
    fn test_load_transition_dominance_from_yaml_parameterized_ok() {
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 2);
        assert!(result.is_ok());
        let object_type = result.unwrap();
        let result = metadata.add_element_variable("v", object_type);
        assert!(result.is_ok());
        let v = result.unwrap();

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 1],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 1],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![0],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![1],
                ..Default::default()
            },
        ];
        let backward_transitions = vec![];

        let yaml = r"
        - dominating:
            name: transition1
            parameters:
              - name: a
                object: object
              - name: b
                object: object
          dominated:
            name: transition1
            parameters:
              - name: x
                object: object
              - name: y
                object: object
          conditions:
            - condition: (>= a c)
              forall:
                - name: c
                  object: object
            - (!= a b)
            - (!= x y)
            - (>= v a)
            - (>= v x)
            - condition: (>= v c)
              forall:
                - name: c
                  object: object
        - dominating:
            name: transition1
            parameters:
              - name: a
                object: object
              - name: b
                object: object
          dominated:
            name: transition2
            parameters:
              - name: x
                object: object
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_ok());
        let transition_dominance = result.unwrap();

        assert_eq!(
            transition_dominance[0],
            TransitionDominance {
                dominating: 2,
                dominated: 1,
                backward: false,
                conditions: vec![
                    Condition::comparison_e(ComparisonOperator::Ge, v, 1).into(),
                    Condition::comparison_e(ComparisonOperator::Ge, v, 0).into(),
                    Condition::comparison_e(ComparisonOperator::Ge, v, 0).into(),
                    Condition::comparison_e(ComparisonOperator::Ge, v, 1).into(),
                ]
            }
        );
        assert_eq!(
            transition_dominance[1],
            TransitionDominance {
                dominating: 0,
                dominated: 4,
                backward: false,
                conditions: Vec::new(),
            }
        );
        assert_eq!(
            transition_dominance[2],
            TransitionDominance {
                dominating: 0,
                dominated: 5,
                backward: false,
                conditions: Vec::new(),
            }
        );
        assert_eq!(
            transition_dominance[3],
            TransitionDominance {
                dominating: 1,
                dominated: 4,
                backward: false,
                conditions: Vec::new(),
            }
        );
        assert_eq!(
            transition_dominance[4],
            TransitionDominance {
                dominating: 1,
                dominated: 5,
                backward: false,
                conditions: Vec::new(),
            }
        );
        assert_eq!(
            transition_dominance[5],
            TransitionDominance {
                dominating: 2,
                dominated: 4,
                backward: false,
                conditions: Vec::new(),
            }
        );
        assert_eq!(
            transition_dominance[6],
            TransitionDominance {
                dominating: 2,
                dominated: 5,
                backward: false,
                conditions: Vec::new(),
            }
        );
        assert_eq!(
            transition_dominance[7],
            TransitionDominance {
                dominating: 3,
                dominated: 4,
                backward: false,
                conditions: Vec::new(),
            }
        );
        assert_eq!(
            transition_dominance[8],
            TransitionDominance {
                dominating: 3,
                dominated: 5,
                backward: false,
                conditions: Vec::new(),
            }
        );
    }

    #[test]
    fn test_load_transition_dominance_from_yaml_not_array_err() {
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 2);
        assert!(result.is_ok());

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 1],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 1],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![0],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![1],
                ..Default::default()
            },
        ];
        let backward_transitions = vec![];

        let yaml = r"
        dominating:
          name: transition1
          parameters:
            - name: a
              object: object
            - name: b
              object: object
        dominated:
          name: transition2
          parameters:
            - name: x
              object: object
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_load_transition_dominance_from_yaml_no_dominating_err() {
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 2);
        assert!(result.is_ok());

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 1],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 1],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![0],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![1],
                ..Default::default()
            },
        ];
        let backward_transitions = vec![];

        let yaml = r"
        - dominated:
            name: transition2
            parameters:
              - name: x
                object: object
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_load_transition_dominance_from_yaml_no_dominated_err() {
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 2);
        assert!(result.is_ok());

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 1],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 1],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![0],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![1],
                ..Default::default()
            },
        ];
        let backward_transitions = vec![];

        let yaml = r"
        - dominating:
            name: transition1
            parameters:
              - name: a
                object: object
              - name: b
                object: object
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_load_transition_dominance_from_yaml_dominated_not_found_err() {
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 2);
        assert!(result.is_ok());

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 1],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 1],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![0],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![1],
                ..Default::default()
            },
        ];
        let backward_transitions = vec![];

        let yaml = r"
        - dominating:
            name: transition1
            parameters:
              - name: a
                object: object
              - name: b
                object: object
          dominated:
            name: transition3
            parameters:
              - name: x
                object: object
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_load_transition_dominance_from_yaml_forward_backward_err() {
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 2);
        assert!(result.is_ok());

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 1],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 1],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![0],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![1],
                ..Default::default()
            },
        ];
        let backward_transitions = vec![Transition {
            name: "transition3".to_string(),
            ..Default::default()
        }];

        let yaml = r"
        - dominating:
            name: transition1
            parameters:
              - name: a
                object: object
              - name: b
                object: object
          dominated:
            name: transition3
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_load_transition_dominance_from_yaml_backward_forward_err() {
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 2);
        assert!(result.is_ok());

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 1],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 1],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![0],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![1],
                ..Default::default()
            },
        ];
        let backward_transitions = vec![Transition {
            name: "transition3".to_string(),
            ..Default::default()
        }];

        let yaml = r"
        - dominating:
            name: transition3
          dominated:
            name: transition1
            parameters:
              - name: a
                object: object
              - name: b
                object: object
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_load_transition_dominance_from_yaml_duplicate_parameter_err() {
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 2);
        assert!(result.is_ok());

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 1],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 1],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![0],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![1],
                ..Default::default()
            },
        ];
        let backward_transitions = vec![];

        let yaml = r"
        - dominating:
            name: transition1
            parameters:
              - name: a
                object: object
              - name: b
                object: object
          dominated:
            name: transition2
            parameters:
              - name: a
                object: object
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_load_transition_dominance_from_yaml_condition_err() {
        let registry = TableRegistry::default();
        let functions = StateFunctions::default();

        let mut metadata = StateMetadata::default();
        let result = metadata.add_object_type("object", 2);
        assert!(result.is_ok());

        let forward_transitions = vec![
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![0, 1],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 0],
                ..Default::default()
            },
            Transition {
                name: "transition1".to_string(),
                parameter_names: vec!["a".to_string(), "b".to_string()],
                parameter_values: vec![1, 1],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![0],
                ..Default::default()
            },
            Transition {
                name: "transition2".to_string(),
                parameter_names: vec!["a".to_string()],
                parameter_values: vec![1],
                ..Default::default()
            },
        ];
        let backward_transitions = vec![];

        let yaml = r"
        - dominating:
            name: transition1
            parameters:
              - name: a
                object: object
              - name: b
                object: object
        - dominated:
            name: transition2
            parameters:
              - name: x
                object: object
        - conditions:
          - (a != c)
        ";

        let yaml = YamlLoader::load_from_str(yaml);
        assert!(yaml.is_ok());
        let yaml = &yaml.unwrap();
        assert_eq!(yaml.len(), 1);
        let yaml = &yaml[0];
        let result = load_transition_dominance_from_yaml(
            yaml,
            &metadata,
            &functions,
            &registry,
            &forward_transitions,
            &backward_transitions,
        );
        assert!(result.is_err());
    }
}
