use crate::expression;
use crate::expression_parser;
use crate::state;
use crate::table_registry;
use crate::variable::Element;
use crate::yaml_util;
use lazy_static::lazy_static;
use std::collections;
use std::error::Error;
use yaml_rust::Yaml;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct GroundedCondition {
    pub elements_in_set_variable: Vec<(usize, usize)>,
    pub elements_in_vector_variable: Vec<(usize, usize)>,
    pub condition: expression::Condition,
}

impl GroundedCondition {
    pub fn is_satisfied(
        &self,
        state: &state::State,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
    ) -> Option<bool> {
        for (i, v) in &self.elements_in_set_variable {
            if !state.signature_variables.set_variables[*i].contains(*v) {
                return None;
            }
        }
        for (i, v) in &self.elements_in_vector_variable {
            if !state.signature_variables.vector_variables[*i].contains(v) {
                return None;
            }
        }
        Some(self.condition.eval(state, metadata, registry))
    }

    pub fn load_grounded_conditions_from_yaml(
        value: &Yaml,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
        parameters: &collections::HashMap<String, Element>,
    ) -> Result<Vec<GroundedCondition>, Box<dyn Error>> {
        lazy_static! {
            static ref CONDITION_KEY: Yaml = Yaml::from_str("condition");
            static ref FORALL_KEY: Yaml = Yaml::from_str("forall");
        }
        match value {
            Yaml::String(condition) => {
                let condition = expression_parser::parse_condition(
                    condition.clone(),
                    metadata,
                    registry,
                    &parameters,
                )?;
                Ok(vec![GroundedCondition {
                    condition: condition.simplify(registry),
                    elements_in_set_variable: vec![],
                    elements_in_vector_variable: vec![],
                }])
            }
            Yaml::Hash(map) => {
                let condition = yaml_util::get_string_by_key(map, "condition")?;
                match map.get(&Yaml::from_str("forall")) {
                    Some(forall) => {
                        let (
                            parameters_array,
                            elements_in_set_variable_array,
                            elements_in_vector_variable_array,
                        ) = metadata.ground_parameters_from_yaml(forall)?;
                        let mut conditions = Vec::with_capacity(parameters_array.len());
                        for ((forall, elements_in_set_variable), elements_in_vector_variable) in
                            parameters_array
                                .into_iter()
                                .zip(elements_in_set_variable_array.into_iter())
                                .zip(elements_in_vector_variable_array.into_iter())
                        {
                            let mut parameters = parameters.clone();
                            parameters.extend(forall);
                            let condition = expression_parser::parse_condition(
                                condition.clone(),
                                metadata,
                                registry,
                                &parameters,
                            )?;
                            conditions.push(GroundedCondition {
                                condition: condition.simplify(registry),
                                elements_in_set_variable,
                                elements_in_vector_variable,
                            });
                        }
                        Ok(conditions)
                    }
                    None => {
                        let condition = expression_parser::parse_condition(
                            condition,
                            metadata,
                            registry,
                            &parameters,
                        )?;
                        Ok(vec![GroundedCondition {
                            condition: condition.simplify(registry),
                            elements_in_set_variable: vec![],
                            elements_in_vector_variable: vec![],
                        }])
                    }
                }
            }
            _ => Err(yaml_util::YamlContentErr::new(format!(
                "expected String or Hash, found `{:?}`",
                value
            ))
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::expression::*;
    use super::super::table;
    use super::*;
    use crate::variable;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec![String::from("object")];
        let mut name_to_object = HashMap::new();
        name_to_object.insert(String::from("object"), 0);
        let object_numbers = vec![2];

        let set_variable_names = vec![String::from("s0")];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert(String::from("s0"), 0);
        let set_variable_to_object = vec![0];

        let vector_variable_names = vec![String::from("p0")];
        let mut name_to_vector_variable = HashMap::new();
        name_to_vector_variable.insert(String::from("p0"), 0);
        let vector_variable_to_object = vec![0];

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
            vector_variable_names,
            name_to_vector_variable,
            vector_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            ..Default::default()
        }
    }

    fn generate_registry() -> table_registry::TableRegistry {
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
    fn is_satisfied_test() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let mut s0 = variable::Set::with_capacity(2);
        s0.insert(0);
        let state = state::State {
            signature_variables: Rc::new(state::SignatureVariables {
                set_variables: vec![s0],
                vector_variables: vec![vec![1]],
                element_variables: vec![0],
                ..Default::default()
            }),
            ..Default::default()
        };

        let condition = GroundedCondition {
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::SetVariable(0),
            )),
            ..Default::default()
        };
        assert_eq!(
            condition.is_satisfied(&state, &metadata, &registry),
            Some(true)
        );

        let condition = GroundedCondition {
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::VectorVariable(0),
            )),
            ..Default::default()
        };
        assert_eq!(
            condition.is_satisfied(&state, &metadata, &registry),
            Some(false)
        );

        let condition = GroundedCondition {
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::VectorVariable(0),
            )),
            elements_in_set_variable: vec![(0, 0)],
            elements_in_vector_variable: vec![],
        };
        assert_eq!(
            condition.is_satisfied(&state, &metadata, &registry),
            Some(false)
        );

        let condition = GroundedCondition {
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Constant(1),
                SetExpression::VectorVariable(0),
            )),
            elements_in_set_variable: vec![(0, 1)],
            elements_in_vector_variable: vec![],
        };
        assert!(condition
            .is_satisfied(&state, &metadata, &registry)
            .is_none());

        let condition = GroundedCondition {
            condition: Condition::Set(SetCondition::Ne(
                ElementExpression::Constant(0),
                ElementExpression::Constant(0),
            )),
            elements_in_set_variable: vec![(0, 0), (0, 0)],
            elements_in_vector_variable: vec![],
        };
        assert_eq!(
            condition.is_satisfied(&state, &metadata, &registry),
            Some(false)
        );

        let condition = GroundedCondition {
            condition: Condition::Set(SetCondition::Ne(
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
            )),
            elements_in_set_variable: vec![(0, 1), (0, 0)],
            elements_in_vector_variable: vec![],
        };
        assert!(condition
            .is_satisfied(&state, &metadata, &registry)
            .is_none());

        let condition = GroundedCondition {
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
            )),
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![(0, 0)],
        };
        assert!(condition
            .is_satisfied(&state, &metadata, &registry)
            .is_none());

        let condition = GroundedCondition {
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Constant(1),
                SetExpression::SetVariable(0),
            )),
            elements_in_set_variable: vec![],
            elements_in_vector_variable: vec![(0, 1)],
        };
        assert_eq!(
            condition.is_satisfied(&state, &metadata, &registry),
            Some(false)
        );
    }

    #[test]
    fn load_from_yaml_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();

        let condition = r"
condition: (and (is_in e0 s0) true)
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];
        let parameters = collections::HashMap::new();

        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_ok());
        let expected = vec![GroundedCondition {
            elements_in_set_variable: Vec::new(),
            elements_in_vector_variable: Vec::new(),
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Variable(0),
                SetExpression::SetVariable(0),
            )),
        }];
        assert_eq!(conditions.unwrap(), expected);

        let condition = r"(is_in e0 s0)";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];
        let parameters = collections::HashMap::new();
        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_ok());
        assert_eq!(conditions.unwrap(), expected);

        let condition = r"(is_in a s0)";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];
        let mut parameters = collections::HashMap::new();
        parameters.insert(String::from("a"), 0);
        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_ok());
        let expected = vec![GroundedCondition {
            elements_in_set_variable: Vec::new(),
            elements_in_vector_variable: Vec::new(),
            condition: Condition::Set(SetCondition::IsIn(
                ElementExpression::Constant(0),
                SetExpression::SetVariable(0),
            )),
        }];
        assert_eq!(conditions.unwrap(), expected);

        let condition = r"
condition: (is 0 e)
forall:
        - name: e
          object: s0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];
        let parameters = collections::HashMap::new();

        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_ok());
        let expected = vec![
            GroundedCondition {
                elements_in_set_variable: vec![(0, 0)],
                elements_in_vector_variable: Vec::new(),
                condition: Condition::Constant(true),
            },
            GroundedCondition {
                elements_in_set_variable: vec![(0, 1)],
                elements_in_vector_variable: Vec::new(),
                condition: Condition::Constant(false),
            },
        ];
        assert_eq!(conditions.unwrap(), expected);

        let condition = r"
condition: (is_in e s0)
forall:
        - name: e
          object: p0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_ok());
        let expected = vec![
            GroundedCondition {
                elements_in_set_variable: Vec::new(),
                elements_in_vector_variable: vec![(0, 0)],
                condition: Condition::Set(SetCondition::IsIn(
                    ElementExpression::Constant(0),
                    SetExpression::SetVariable(0),
                )),
            },
            GroundedCondition {
                elements_in_set_variable: Vec::new(),
                elements_in_vector_variable: vec![(0, 1)],
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
        let parameters = collections::HashMap::new();

        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_err());

        let condition = r"
condition: (is 0 d)
forall:
        - name: e
          object: s0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_err());

        let condition = r"
condition: (is_in e s0)
forall:
        - object: p0
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_err());

        let condition = r"
condition: (is_in e s0)
forall:
        - name: e
          object: null
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_err());

        let condition = r"
condition: (is_in e s0)
forall:
        - name: e
";
        let condition = yaml_rust::YamlLoader::load_from_str(condition);
        assert!(condition.is_ok());
        let condition = condition.unwrap();
        assert_eq!(condition.len(), 1);
        let condition = &condition[0];

        let conditions = GroundedCondition::load_grounded_conditions_from_yaml(
            condition,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(conditions.is_err());
    }
}
