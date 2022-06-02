use crate::expression;
use crate::expression_parser;
use crate::expression_parser::ParseNumericExpression;
use crate::state;
use crate::table_registry;
use crate::variable::{Continuous, Element, FromNumeric, Integer, Numeric, OrderedContinuous};
use crate::yaml_util;
use rustc_hash::FxHashMap;
use std::error;
use yaml_rust::Yaml;

pub trait ParesNumericExpressionFromYaml: ParseNumericExpression {
    fn parse_expression_from_yaml(
        value: &Yaml,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
        parameters: &FxHashMap<String, usize>,
    ) -> Result<expression::NumericExpression<Self>, Box<dyn error::Error>>
    where
        Self: Numeric;
}

impl ParesNumericExpressionFromYaml for Integer {
    fn parse_expression_from_yaml(
        value: &Yaml,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
        parameters: &FxHashMap<String, usize>,
    ) -> Result<expression::NumericExpression<Self>, Box<dyn error::Error>> {
        match value {
            yaml_rust::Yaml::String(value) => Ok(Integer::parse_expression(
                value.clone(),
                metadata,
                registry,
                parameters,
            )?),
            Yaml::Integer(value) => Ok(expression::NumericExpression::Constant(*value as Integer)),
            _ => Err(yaml_util::YamlContentErr::new(format!(
                "expected String or Integer, but is {:?}",
                value
            ))
            .into()),
        }
    }
}

impl ParesNumericExpressionFromYaml for Continuous {
    fn parse_expression_from_yaml(
        value: &Yaml,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
        parameters: &FxHashMap<String, usize>,
    ) -> Result<expression::NumericExpression<Self>, Box<dyn error::Error>> {
        match value {
            yaml_rust::Yaml::String(value) => Ok(Continuous::parse_expression(
                value.clone(),
                metadata,
                registry,
                parameters,
            )?),
            Yaml::Integer(value) => Ok(expression::NumericExpression::Constant(
                *value as Continuous,
            )),
            Yaml::Real(value) => match value.parse::<Continuous>() {
                Ok(value) => Ok(expression::NumericExpression::Constant(value)),
                Err(e) => Err(yaml_util::YamlContentErr::new(format!(
                    "could not parse {} as a number: {:?}",
                    value, e
                ))
                .into()),
            },
            _ => Err(yaml_util::YamlContentErr::new(format!(
                "expected String or Integer, but is {:?}",
                value
            ))
            .into()),
        }
    }
}

impl ParesNumericExpressionFromYaml for OrderedContinuous {
    fn parse_expression_from_yaml(
        value: &Yaml,
        metadata: &state::StateMetadata,
        registry: &table_registry::TableRegistry,
        parameters: &FxHashMap<String, usize>,
    ) -> Result<expression::NumericExpression<Self>, Box<dyn error::Error>> {
        match value {
            yaml_rust::Yaml::String(value) => Ok(OrderedContinuous::parse_expression(
                value.clone(),
                metadata,
                registry,
                parameters,
            )?),
            Yaml::Integer(value) => Ok(expression::NumericExpression::Constant(
                OrderedContinuous::from_continuous(*value as Continuous),
            )),
            Yaml::Real(value) => match value.parse::<OrderedContinuous>() {
                Ok(value) => Ok(expression::NumericExpression::Constant(value)),
                Err(e) => Err(yaml_util::YamlContentErr::new(format!(
                    "could not parse {} as a number: {:?}",
                    value, e
                ))
                .into()),
            },
            _ => Err(yaml_util::YamlContentErr::new(format!(
                "expected String or Integer, but is {:?}",
                value
            ))
            .into()),
        }
    }
}

pub fn parse_element_from_yaml(
    value: &Yaml,
    metadata: &state::StateMetadata,
    registry: &table_registry::TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::ElementExpression, Box<dyn error::Error>> {
    match value {
        yaml_rust::Yaml::String(value) => Ok(expression_parser::parse_element(
            value.clone(),
            metadata,
            registry,
            parameters,
        )?),
        Yaml::Integer(value) => Ok(expression::ElementExpression::Constant(*value as Element)),
        _ => Err(yaml_util::YamlContentErr::new(format!(
            "expected String or Integer, but is {:?}",
            value
        ))
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::OrderedFloat;

    #[test]
    fn integer_parse_expression_from_yaml_ok() {
        let metadata = state::StateMetadata::default();
        let registry = table_registry::TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1)"));
        let result = Integer::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::NumericExpression::NumericOperation(
                expression::NumericOperator::Add,
                Box::new(expression::NumericExpression::Cost),
                Box::new(expression::NumericExpression::Constant(1))
            )
        );

        let value = Yaml::Integer(1);
        let result = Integer::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expression::NumericExpression::Constant(1));
    }

    #[test]
    fn integer_parse_expression_from_yaml_err() {
        let metadata = state::StateMetadata::default();
        let registry = table_registry::TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result = Integer::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let value = Yaml::Real(String::from("1.2"));
        let result = Integer::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn continuous_parse_expression_from_yaml_ok() {
        let metadata = state::StateMetadata::default();
        let registry = table_registry::TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1)"));
        let result =
            Continuous::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::NumericExpression::NumericOperation(
                expression::NumericOperator::Add,
                Box::new(expression::NumericExpression::Cost),
                Box::new(expression::NumericExpression::Constant(1.0))
            )
        );

        let value = Yaml::Integer(1);
        let result =
            Continuous::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::NumericExpression::Constant(1.0)
        );

        let value = Yaml::Real(String::from("1.2"));
        let result =
            Continuous::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::NumericExpression::Constant(1.2)
        );
    }

    #[test]
    fn continuous_parse_expression_from_yaml_err() {
        let metadata = state::StateMetadata::default();
        let registry = table_registry::TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result =
            Continuous::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let value = Yaml::Real(String::from("a"));
        let result =
            Continuous::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let value = Yaml::Boolean(true);
        let result =
            Continuous::parse_expression_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn ordered_continuous_parse_expression_from_yaml_ok() {
        let metadata = state::StateMetadata::default();
        let registry = table_registry::TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1)"));
        let result = OrderedContinuous::parse_expression_from_yaml(
            &value,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::NumericExpression::NumericOperation(
                expression::NumericOperator::Add,
                Box::new(expression::NumericExpression::Cost),
                Box::new(expression::NumericExpression::Constant(OrderedFloat(1.0)))
            )
        );

        let value = Yaml::Integer(1);
        let result = OrderedContinuous::parse_expression_from_yaml(
            &value,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::NumericExpression::Constant(OrderedFloat(1.0))
        );

        let value = Yaml::Real(String::from("1.2"));
        let result = OrderedContinuous::parse_expression_from_yaml(
            &value,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::NumericExpression::Constant(OrderedFloat(1.2))
        );
    }

    #[test]
    fn oredered_continuous_parse_expression_from_yaml_err() {
        let metadata = state::StateMetadata::default();
        let registry = table_registry::TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result = OrderedContinuous::parse_expression_from_yaml(
            &value,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());

        let value = Yaml::Real(String::from("a"));
        let result = OrderedContinuous::parse_expression_from_yaml(
            &value,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());

        let value = Yaml::Boolean(true);
        let result = OrderedContinuous::parse_expression_from_yaml(
            &value,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_from_yaml_ok() {
        let metadata = state::StateMetadata::default();
        let registry = table_registry::TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("1"));
        let result = parse_element_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expression::ElementExpression::Constant(1));

        let value = Yaml::Integer(1);
        let result = parse_element_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expression::ElementExpression::Constant(1));
    }

    #[test]
    fn parse_element_from_yaml_err() {
        let metadata = state::StateMetadata::default();
        let registry = table_registry::TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result = parse_element_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let value = Yaml::Real(String::from("1.2"));
        let result = parse_element_from_yaml(&value, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
