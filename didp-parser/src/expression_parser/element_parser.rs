use super::util;
use super::util::ParseErr;
use crate::expression::{ElementExpression, TableExpression};
use crate::state::StateMetadata;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::Element;
use std::collections::HashMap;

pub fn parse_expression<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c HashMap<String, usize>,
) -> Result<(ElementExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = tokens
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            if let Some((expression, rest)) = parse_table_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.element_tables,
            )? {
                Ok((ElementExpression::Table(Box::new(expression)), rest))
            } else {
                Err(ParseErr::new(format!("no such table `{}`", name)))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        _ => {
            let element = parse_atom(
                token,
                &registry.element_tables.name_to_constant,
                &metadata.name_to_element_variable,
                parameters,
            )?;
            Ok((element, rest))
        }
    }
}

type TableExpressionResult<'a, T> = Option<(TableExpression<T>, &'a [String])>;

pub fn parse_table_expression<'a, 'b, 'c, T: Clone>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c HashMap<String, usize>,
    tables: &'b TableData<T>,
) -> Result<TableExpressionResult<'a, T>, ParseErr> {
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableExpression::Table1D(*i, x), rest)))
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
        let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableExpression::Table2D(*i, x, y), rest)))
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
        let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
        let (z, rest) = parse_expression(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableExpression::Table3D(*i, x, y, z), rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let result = parse_table(*i, tokens, metadata, registry, parameters)?;
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

fn parse_table<'a, 'b, 'c, T: Clone>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c HashMap<String, usize>,
) -> Result<(TableExpression<T>, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((TableExpression::Table(i, args), rest));
        }
        let (expression, new_xs) = parse_expression(xs, metadata, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

fn parse_atom(
    token: &str,
    name_to_constant: &HashMap<String, Element>,
    name_to_variable: &HashMap<String, usize>,
    parameters: &HashMap<String, usize>,
) -> Result<ElementExpression, ParseErr> {
    if let Some(v) = parameters.get(token) {
        Ok(ElementExpression::Constant(*v))
    } else if let Some(v) = name_to_constant.get(token) {
        Ok(ElementExpression::Constant(*v))
    } else if let Some(i) = name_to_variable.get(token) {
        Ok(ElementExpression::Variable(*i))
    } else if token == "stage" {
        Ok(ElementExpression::Stage)
    } else {
        let v: Element = token.parse().map_err(|e| {
            ParseErr::new(format!("could not parse {} as a number: {:?}", token, e))
        })?;
        Ok(ElementExpression::Constant(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let tokens: Vec<String> = ["e1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Variable(1)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Constant(11)));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_expression_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["p1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["stage", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Stage));

        assert_eq!(rest, &tokens[1..]);
        let tokens: Vec<String> = ["e1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Variable(1)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Constant(11)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["param", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Constant(0)));
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
