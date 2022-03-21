use crate::expression;
use crate::expression_parser;
use crate::grounded_condition::GroundedCondition;
use crate::state;
use crate::table_registry;
use crate::util;
use crate::variable::Numeric;
use crate::yaml_util;
use lazy_static::lazy_static;
use std::collections;
use std::error::Error;
use std::fmt;
use std::str;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct BaseCase<T: Numeric> {
    pub conditions: Vec<GroundedCondition>,
    pub cost: expression::NumericExpression<T>,
}

impl<T: Numeric> BaseCase<T> {
    pub fn get_cost(
        &self,
        state: &state::State,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
    ) -> Option<T> {
        let is_satisfied = self
            .conditions
            .iter()
            .all(|x| x.is_satisfied(state, &metadata, &registry).unwrap_or(true));
        if is_satisfied {
            Some(self.cost.eval(state, metadata, registry))
        } else {
            None
        }
    }

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
            yaml_rust::Yaml::Array(array) => {
                (array, expression::NumericExpression::Constant(T::zero()))
            }
            yaml_rust::Yaml::Hash(map) => {
                let array = yaml_util::get_array_by_key(map, "conditions")?;
                let cost = match map.get(&COST_KEY) {
                    Some(cost) => {
                        let cost = yaml_util::get_string(cost)?;
                        let parameters = collections::HashMap::new();
                        expression_parser::parse_numeric(cost, metadata, registry, &parameters)?
                    }
                    _ => expression::NumericExpression::Constant(T::zero()),
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
        let parameters = collections::HashMap::new();
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
                    expression::Condition::Constant(false) => {
                        return Err(util::ModelErr::new(String::from(
                            "terminal condition never satisfied",
                        ))
                        .into())
                    }
                    expression::Condition::Constant(true) => {}
                    _ => conditions.push(c),
                }
            }
        }
        Ok(BaseCase { conditions, cost })
    }
}
