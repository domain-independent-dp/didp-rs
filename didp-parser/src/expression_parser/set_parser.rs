use super::element_parser;
use super::reference_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{ReferenceExpression, SetElementOperator, SetExpression, SetOperator};
use crate::state::StateMetadata;
use crate::table_registry::TableRegistry;
use rustc_hash::FxHashMap;

pub fn parse_expression<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
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
            let (name, rest) = rest
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
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    match name {
        "union" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
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
            let (x, rest) =
                element_parser::parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((
                SetExpression::SetElementOperation(SetElementOperator::Add, x, Box::new(y)),
                rest,
            ))
        }
        "remove" => {
            let (x, rest) =
                element_parser::parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((
                SetExpression::SetElementOperation(SetElementOperator::Remove, x, Box::new(y)),
                rest,
            ))
        }
        op => Err(ParseErr::new(format!("no such operator `{}`", op))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::table::*;
    use crate::table_data::*;
    use crate::variable::*;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            "s0".to_string(),
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
        ];
        let mut name_to_set_variable = FxHashMap::default();
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
        let mut name_to_vector_variable = FxHashMap::default();
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
        let mut name_to_element_variable = FxHashMap::default();
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
        let mut name_to_integer_variable = FxHashMap::default();
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
    fn generate_registry() -> TableRegistry {
        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let default = Set::with_capacity(3);
        let tables_1d = vec![Table1D::new(vec![set, default.clone(), default])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("s1"), 0);
        TableRegistry {
            set_tables: TableData {
                tables_1d,
                name_to_table_1d,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn generate_parameters() -> FxHashMap<String, usize> {
        let mut parameters = FxHashMap::default();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    #[test]
    fn parse_expression_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reference(ReferenceExpression::Variable(0))
        );
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_expression_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_complemnt_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["!", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Complement(Box::new(SetExpression::Reference(
                ReferenceExpression::Variable(2)
            )))
        );
        assert_eq!(rest, &tokens[2..]);
    }

    #[test]
    fn parse_complenent_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["!", "e2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["!", "n2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn pare_set_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "union", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetOperation(
                SetOperator::Union,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1)))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "difference", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetOperation(
                SetOperator::Difference,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1)))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "intersection", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetOperation(
                SetOperator::Intersection,
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(1)))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "add", "e1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetElementOperation(
                SetElementOperator::Add,
                ElementExpression::Variable(1),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "remove", "1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::SetElementOperation(
                SetElementOperator::Remove,
                ElementExpression::Constant(1),
                Box::new(SetExpression::Reference(ReferenceExpression::Variable(2))),
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn pare_set_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "add", "n1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "remove", "n1", "s2", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "intersection", "s2", "e1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "/", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            SetExpression::Reference(ReferenceExpression::Variable(1))
        );
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_atom_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e4", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
