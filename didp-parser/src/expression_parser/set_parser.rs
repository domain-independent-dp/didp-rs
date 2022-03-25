use super::element_parser;
use super::reference_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{ReferenceExpression, SetElementOperator, SetExpression, SetOperator};
use crate::state::StateMetadata;
use crate::table_registry::TableRegistry;
use std::collections::HashMap;

pub fn parse_expression<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c HashMap<String, usize>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "!" => {
            let (expression, rest) = parse_expression(rest, metadata, registry, parameters)?;
            Ok((SetExpression::Complement(Box::new(expression)), rest))
        }
        "(" => {
            let (name, rest) = tokens
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            if let Some((expression, rest)) = element_parser::parse_table_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.set_tables,
            )? {
                Ok((
                    SetExpression::Reference(ReferenceExpression::Table(expression)),
                    rest,
                ))
            } else {
                let (expression, rest) =
                    parse_operation(name, rest, metadata, registry, parameters)?;
                Ok((expression, rest))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        _ => {
            let set = reference_parser::parse_atom(
                token,
                &registry.set_tables.name_to_constant,
                &metadata.name_to_set_variable,
            )?;
            Ok((SetExpression::Reference(set), rest))
        }
    }
}

fn parse_operation<'a, 'b, 'c>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c HashMap<String, usize>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    match name {
        "union" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((
                SetExpression::SetOperation(SetOperator::Union, Box::new(x), Box::new(y)),
                rest,
            ))
        }
        "difference" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((
                SetExpression::SetOperation(SetOperator::Difference, Box::new(x), Box::new(y)),
                rest,
            ))
        }
        "intersection" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((
                SetExpression::SetOperation(SetOperator::Intersection, Box::new(x), Box::new(y)),
                rest,
            ))
        }
        "add" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = element_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((
                SetExpression::SetElementOperation(SetElementOperator::Add, Box::new(x), y),
                rest,
            ))
        }
        "remove" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = element_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((
                SetExpression::SetElementOperation(SetElementOperator::Remove, Box::new(x), y),
                rest,
            ))
        }
        op => Err(ParseErr::new(format!("no such operator `{}`", op))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = HashMap::new();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            "s0".to_string(),
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
        ];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        name_to_set_variable.insert("s2".to_string(), 2);
        name_to_set_variable.insert("s3".to_string(), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let vector_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_vector_variable = HashMap::new();
        name_to_vector_variable.insert("p0".to_string(), 0);
        name_to_vector_variable.insert("p1".to_string(), 1);
        name_to_vector_variable.insert("p2".to_string(), 2);
        name_to_vector_variable.insert("p3".to_string(), 3);
        let vector_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
        ];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        name_to_element_variable.insert("e2".to_string(), 2);
        name_to_element_variable.insert("e3".to_string(), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let integer_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_integer_variable = HashMap::new();
        name_to_integer_variable.insert("n0".to_string(), 0);
        name_to_integer_variable.insert("n1".to_string(), 1);
        name_to_integer_variable.insert("n2".to_string(), 2);
        name_to_integer_variable.insert("n3".to_string(), 3);

        StateMetadata {
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
            integer_variable_names,
            name_to_integer_variable,
            ..Default::default()
        }
    }

    fn generate_parameters() -> HashMap<String, usize> {
        let mut parameters = HashMap::new();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    #[test]
    fn parse_expression_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, SetExpression::SetVariable(0)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["p0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, SetExpression::VectorVariable(0)));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_expression_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_complemnt_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["!", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, SetExpression::Complement(_)));
        if let SetExpression::Complement(s) = expression {
            assert!(matches!(*s, SetExpression::SetVariable(2)));
        }
        assert_eq!(rest, &tokens[2..]);
    }

    #[test]
    fn parse_complenent_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["!", "e2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["!", "n2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn pare_set_operation_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetOperation(SetOperator::Union, _, _)
        ));
        if let SetExpression::SetOperation(SetOperator::Union, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(*y, SetExpression::SetVariable(1)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "-", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetOperation(SetOperator::Difference, _, _)
        ));
        if let SetExpression::SetOperation(SetOperator::Difference, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(*y, SetExpression::SetVariable(1)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "&", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetOperation(SetOperator::Intersect, _, _)
        ));
        if let SetExpression::SetOperation(SetOperator::Intersect, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(*y, SetExpression::SetVariable(1)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "+", "s2", "e1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetElementOperation(SetElementOperator::Add, _, _)
        ));
        if let SetExpression::SetElementOperation(SetElementOperator::Add, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(y, ElementExpression::Variable(1)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "-", "s2", "1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetElementOperation(SetElementOperator::Remove, _, _)
        ));
        if let SetExpression::SetElementOperation(SetElementOperator::Remove, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(y, ElementExpression::Constant(1)));
        }
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn pare_set_operation_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "s2", "n1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "-", "s2", "n1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "&", "s2", "e1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "/", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::Reference(ReferenceExpression::Variable(1))
        ));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_atom_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e4", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
