use crate::expression::{Condition, NumericExpression};
use crate::expression_parser::ParseNumericExpression;
use crate::grounded_condition::GroundedCondition;
use crate::state;
use crate::table_registry;
use crate::util;
use crate::variable::Numeric;
use crate::yaml_util;
use lazy_static::lazy_static;
use rustc_hash::FxHashMap;
use std::error::Error;
use std::fmt;
use std::str;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct BaseCase(Vec<GroundedCondition>);

impl<T: Numeric> BaseCase<T> {
    pub fn get_cost<U: state::DPState>(
        &self,
        state: &U,
        registry: &table_registry::TableRegistry,
    ) -> Option<T> {
        let is_satisfied = self
            .conditions
            .iter()
            .all(|x| x.is_satisfied(state, &registry).unwrap_or(true));
        if is_satisfied {
            Some(self.cost.eval(state, registry))
        } else {
            None
        }
    }

    pub fn is_satisfied<U: state::DPState>(
        &self,
        state: &U,
        registry: &table_registry::TableRegistry,
    ) -> bool {
        self.conditions
            .iter()
            .all(|x| x.is_satisfied(state, &registry).unwrap_or(true))
    }
}

impl<T: Numeric + ParseNumericExpression> BaseCase<T> {
    pub fn load_from_yaml(
        value: &yaml_rust::Yaml,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
    ) -> Result<BaseCase<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        lazy_static! {
            static ref COST_KEY: yaml_rust::Yaml = yaml_rust::Yaml::from_str("cost");
        }
        let (array, cost) = match value {
            yaml_rust::Yaml::Array(array) => (array, NumericExpression::Constant(T::zero())),
            yaml_rust::Yaml::Hash(map) => {
                let array = yaml_util::get_array_by_key(map, "conditions")?;
                let cost = match map.get(&COST_KEY) {
                    Some(cost) => {
                        let cost = yaml_util::get_string(cost)?;
                        let parameters = FxHashMap::default();
                        let cost = T::parse_expression(cost, metadata, registry, &parameters)?;
                        cost.simplify(registry)
                    }
                    _ => NumericExpression::Constant(T::zero()),
                };
                (array, cost)
            }
            _ => {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "expected Array or Hash, found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let parameters = FxHashMap::default();
        let mut conditions = Vec::new();
        for condition in array {
            let condition = GroundedCondition::load_grounded_conditions_from_yaml(
                &condition,
                &metadata,
                &registry,
                &parameters,
            )?;
            for c in condition {
                match c.condition {
                    Condition::Constant(false)
                        if c.elements_in_set_variable.is_empty()
                            && c.elements_in_vector_variable.is_empty() =>
                    {
                        return Err(util::ModelErr::new(String::from(
                            "terminal condition never satisfied",
                        ))
                        .into())
                    }
                    Condition::Constant(true) => {}
                    _ => conditions.push(c),
                }
            }
        }
        Ok(BaseCase { conditions, cost })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::variable;
    use rustc_hash::FxHashMap;

    fn generate_metadata() -> state::StateMetadata {
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("object"), 0);
        let mut name_to_set_variable = FxHashMap::default();
        name_to_set_variable.insert(String::from("s0"), 0);
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("i0"), 0);
        state::StateMetadata {
            object_names: vec![String::from("object")],
            name_to_object,
            object_numbers: vec![2],
            name_to_set_variable,
            set_variable_names: vec![String::from("s0")],
            set_variable_to_object: vec![0],
            integer_variable_names: vec![String::from("i0")],
            name_to_integer_variable,
            ..Default::default()
        }
    }

    fn generate_state() -> state::State {
        let mut s0 = variable::Set::with_capacity(2);
        s0.insert(0);
        s0.insert(1);
        state::State {
            signature_variables: state::SignatureVariables {
                set_variables: vec![s0],
                integer_variables: vec![1],
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn get_cost() {
        let state = generate_state();
        let registry = table_registry::TableRegistry::default();

        let base_case = BaseCase {
            conditions: vec![GroundedCondition {
                condition: Condition::Constant(true),
                ..Default::default()
            }],
            cost: NumericExpression::Constant(1),
        };
        assert_eq!(base_case.get_cost(&state, &registry), Some(1));

        let base_case = BaseCase {
            conditions: vec![
                GroundedCondition {
                    condition: Condition::Constant(true),
                    ..Default::default()
                },
                GroundedCondition {
                    condition: Condition::Constant(false),
                    ..Default::default()
                },
            ],
            cost: NumericExpression::Constant(1),
        };
        assert_eq!(base_case.get_cost(&state, &registry), None);
    }

    #[test]
    fn load_from_yaml_ok() {
        let metadata = generate_metadata();
        let registry = table_registry::TableRegistry::default();
        let expected = BaseCase {
            conditions: vec![GroundedCondition {
                condition: Condition::Comparison(Box::new(Comparison::ComparisonII(
                    ComparisonOperator::Ge,
                    NumericExpression::IntegerVariable(0),
                    NumericExpression::Constant(0),
                ))),
                ..Default::default()
            }],
            cost: NumericExpression::Constant(0),
        };

        let base_case = yaml_rust::YamlLoader::load_from_str(r"[(>= i0 0)]");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case = BaseCase::load_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);

        let base_case = yaml_rust::YamlLoader::load_from_str(r"conditions: [(>= i0 0), (= i0 i0)]");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case = BaseCase::load_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);

        let base_case = yaml_rust::YamlLoader::load_from_str(
            r"
conditions: [(>= i0 0)]
cost: '0'
",
        );
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case = BaseCase::load_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_ok());
        assert_eq!(base_case.unwrap(), expected);

        let base_case = yaml_rust::YamlLoader::load_from_str(
            r"conditions: [{ condition: (is e 1), forall: [ {name: e, object: s0} ] }]",
        );
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            BaseCase::<variable::Integer>::load_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_ok());
        let expected = BaseCase {
            conditions: vec![GroundedCondition {
                condition: Condition::Constant(false),
                elements_in_set_variable: vec![(0, 0)],
                ..Default::default()
            }],
            cost: NumericExpression::Constant(0),
        };
        assert_eq!(base_case.unwrap(), expected);
    }

    #[test]
    fn load_from_yaml_err() {
        let metadata = generate_metadata();
        let registry = table_registry::TableRegistry::default();

        let base_case = yaml_rust::YamlLoader::load_from_str(r"(>= i0 0)");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            BaseCase::<variable::Integer>::load_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_err());

        let base_case =
            yaml_rust::YamlLoader::load_from_str(r"conditions: [(>= i0 0), (= i0 i0), (= 1 2)]");
        assert!(base_case.is_ok());
        let base_case = base_case.unwrap();
        assert_eq!(base_case.len(), 1);
        let base_case =
            BaseCase::<variable::Integer>::load_from_yaml(&base_case[0], &metadata, &registry);
        assert!(base_case.is_err());
    }
}
