use super::expression_parser;
use crate::util;
use dypdl::expression;
use dypdl::variable_type::{Continuous, Element, Integer};
use dypdl::{StateFunctions, StateMetadata, TableRegistry};
use rustc_hash::FxHashMap;
use std::error::Error;
use yaml_rust::Yaml;

pub fn parse_integer_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::IntegerExpression, Box<dyn Error>> {
    match value {
        Yaml::String(value) => Ok(expression_parser::parse_integer(
            value.clone(),
            metadata,
            functions,
            registry,
            parameters,
        )?),
        Yaml::Integer(value) => Ok(expression::IntegerExpression::Constant(*value as Integer)),
        _ => Err(util::YamlContentErr::new(format!(
            "expected String or Integer, but is {:?}",
            value
        ))
        .into()),
    }
}

pub fn parse_continuous_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::ContinuousExpression, Box<dyn Error>> {
    match value {
        Yaml::String(value) => Ok(expression_parser::parse_continuous(
            value.clone(),
            metadata,
            functions,
            registry,
            parameters,
        )?),
        Yaml::Integer(value) => Ok(expression::ContinuousExpression::Constant(
            *value as Continuous,
        )),
        Yaml::Real(value) => Ok(expression::ContinuousExpression::Constant(value.parse()?)),
        _ => Err(util::YamlContentErr::new(format!(
            "expected String, Integer, or Real, but is {:?}",
            value
        ))
        .into()),
    }
}

pub fn parse_element_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::ElementExpression, Box<dyn Error>> {
    match value {
        Yaml::String(value) => Ok(expression_parser::parse_element(
            value.clone(),
            metadata,
            functions,
            registry,
            parameters,
        )?),
        Yaml::Integer(value) => Ok(expression::ElementExpression::Constant(*value as Element)),
        _ => Err(util::YamlContentErr::new(format!(
            "expected String or Integer, but is {:?}",
            value
        ))
        .into()),
    }
}

pub fn parse_set_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::SetExpression, Box<dyn Error>> {
    match value {
        Yaml::String(value) => Ok(expression_parser::parse_set(
            value.clone(),
            metadata,
            functions,
            registry,
            parameters,
        )?),
        _ => Err(util::YamlContentErr::new(format!("expected String , but is {:?}", value)).into()),
    }
}

pub fn parse_condition_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::Condition, Box<dyn Error>> {
    match value {
        Yaml::String(value) => Ok(expression_parser::parse_condition(
            value.clone(),
            metadata,
            functions,
            registry,
            parameters,
        )?),
        Yaml::Boolean(value) => Ok(expression::Condition::Constant(*value)),
        _ => Err(util::YamlContentErr::new(format!(
            "expected String or Boolean, but is {:?}",
            value
        ))
        .into()),
    }
}

pub fn parse_vector_from_yaml(
    value: &Yaml,
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::VectorExpression, Box<dyn Error>> {
    match value {
        Yaml::String(value) => Ok(expression_parser::parse_vector(
            value.clone(),
            metadata,
            functions,
            registry,
            parameters,
        )?),
        value => {
            if let Ok(vector) = util::get_usize_array(value) {
                Ok(expression::VectorExpression::Reference(
                    expression::ReferenceExpression::Constant(vector),
                ))
            } else {
                Err(
                    util::YamlContentErr::new(format!("expected String , but is {:?}", value))
                        .into(),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::variable_type::Set;

    #[test]
    fn parse_integer_from_yaml_ok() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1)"));
        let result = parse_integer_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::IntegerExpression::BinaryOperation(
                expression::BinaryOperator::Add,
                Box::new(expression::IntegerExpression::Cost),
                Box::new(expression::IntegerExpression::Constant(1))
            )
        );

        let value = Yaml::Integer(1);
        let result = parse_integer_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expression::IntegerExpression::Constant(1));
    }

    #[test]
    fn parse_integer_from_yaml_err() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result = parse_integer_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let value = Yaml::Real(String::from("1.2"));
        let result = parse_integer_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_from_yaml_ok() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1)"));
        let result =
            parse_continuous_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::ContinuousExpression::BinaryOperation(
                expression::BinaryOperator::Add,
                Box::new(expression::ContinuousExpression::Cost),
                Box::new(expression::ContinuousExpression::Constant(1.0))
            )
        );

        let value = Yaml::Integer(1);
        let result =
            parse_continuous_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::ContinuousExpression::Constant(1.0)
        );

        let value = Yaml::Real(String::from("1.2"));
        let result =
            parse_continuous_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::ContinuousExpression::Constant(1.2)
        );
    }

    #[test]
    fn parse_continuous_from_yaml_err() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result =
            parse_continuous_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let value = Yaml::Real(String::from("a"));
        let result =
            parse_continuous_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let value = Yaml::Boolean(true);
        let result =
            parse_continuous_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_from_yaml_ok() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("1"));
        let result = parse_element_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expression::ElementExpression::Constant(1));

        let value = Yaml::Integer(1);
        let result = parse_element_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expression::ElementExpression::Constant(1));
    }

    #[test]
    fn parse_element_from_yaml_err() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(+ cost 1"));
        let result = parse_element_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let value = Yaml::Real(String::from("1.2"));
        let result = parse_element_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_from_yaml_ok() {
        let mut metadata = StateMetadata::default();
        let ob = metadata.add_object_type(String::from("something"), 3);
        assert!(ob.is_ok());
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(something 0 1)"));
        let result = parse_set_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(1);
        assert_eq!(
            result.unwrap(),
            expression::SetExpression::Reference(expression::ReferenceExpression::Constant(set))
        );
    }

    #[test]
    fn parse_set_from_yaml_err() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1)]);
        let result = parse_set_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_condition_from_yaml_ok() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(and true false)"));
        let result =
            parse_condition_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::Condition::And(
                Box::new(expression::Condition::Constant(true)),
                Box::new(expression::Condition::Constant(false))
            )
        );
    }

    #[test]
    fn parse_constant_condition_from_yaml_ok() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::Boolean(true);
        let result =
            parse_condition_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expression::Condition::Constant(true));
    }

    #[test]
    fn parse_condition_from_yaml_err() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::Integer(0);
        let result =
            parse_condition_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_vector_from_yaml_ok() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::String(String::from("(vector 0 1)"));
        let result = parse_vector_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::VectorExpression::Reference(expression::ReferenceExpression::Constant(
                vec![0, 1]
            ))
        );

        let value = Yaml::Array(vec![Yaml::Integer(0), Yaml::Integer(1)]);
        let result = parse_vector_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            expression::VectorExpression::Reference(expression::ReferenceExpression::Constant(
                vec![0, 1]
            ))
        );
    }

    #[test]
    fn parse_vector_from_yaml_err() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();

        let value = Yaml::Integer(0);
        let result = parse_vector_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let value = Yaml::Array(vec![Yaml::String(String::from("1")), Yaml::Integer(1)]);
        let result = parse_vector_from_yaml(&value, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }
}
