use super::util;
use super::util::ParseErr;
use crate::expression::{
    ArgumentExpression, ElementExpression, SetElementOperator, SetExpression, SetOperator,
};
use crate::state;
use crate::variable;
use std::collections;

pub fn parse_set_expression<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (expression, rest) = parse_argument(tokens, metadata, parameters)?;
    match expression {
        ArgumentExpression::Set(expression) => Ok((expression, rest)),
        _ => Err(ParseErr::new(format!(
            "not a set expression: {:?}",
            expression
        ))),
    }
}

pub fn parse_element_expression<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(ElementExpression, &'a [String]), ParseErr> {
    let (expression, rest) = parse_argument(tokens, metadata, parameters)?;
    match expression {
        ArgumentExpression::Element(expression) => Ok((expression, rest)),
        _ => Err(ParseErr::new(format!(
            "not an element expression: {:?}",
            expression
        ))),
    }
}

pub fn parse_argument<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(ArgumentExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "!" => {
            let (expression, rest) = parse_set_expression(rest, metadata, parameters)?;
            Ok((
                ArgumentExpression::Set(SetExpression::Complement(Box::new(expression))),
                rest,
            ))
        }
        "(" => {
            let (expression, rest) = parse_operation(rest, metadata, parameters)?;
            Ok((ArgumentExpression::Set(expression), rest))
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        _ => {
            let argument = parse_atom(token, metadata, parameters)?;
            Ok((argument, rest))
        }
    }
}

fn parse_operation<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    let (x, rest) = parse_argument(rest, metadata, parameters)?;
    let (y, rest) = parse_argument(rest, metadata, parameters)?;
    let rest = util::parse_closing(rest)?;

    match &token[..] {
        "+" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                SetExpression::SetOperation(SetOperator::Union, Box::new(x), Box::new(y)),
                rest,
            )),
            (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => Ok((
                SetExpression::SetElementOperation(SetElementOperator::Add, Box::new(x), y),
                rest,
            )),
            args => Err(ParseErr::new(format!(
                "unexpected arguments for `+`: {:?}",
                args
            ))),
        },
        "-" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                SetExpression::SetOperation(SetOperator::Difference, Box::new(x), Box::new(y)),
                rest,
            )),
            (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => Ok((
                SetExpression::SetElementOperation(SetElementOperator::Remove, Box::new(x), y),
                rest,
            )),
            args => Err(ParseErr::new(format!(
                "unexpected arguments for `-`: {:?}",
                args
            ))),
        },
        "&" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                SetExpression::SetOperation(SetOperator::Intersect, Box::new(x), Box::new(y)),
                rest,
            )),
            args => Err(ParseErr::new(format!(
                "unexpected arguments for `*`: {:?}",
                args
            ))),
        },
        op => Err(ParseErr::new(format!("no such operator `{}`", op))),
    }
}

fn parse_atom(
    token: &str,
    metadata: &state::StateMetadata,
    parameters: &collections::HashMap<String, usize>,
) -> Result<ArgumentExpression, ParseErr> {
    if let Some(v) = parameters.get(token) {
        Ok(ArgumentExpression::Element(ElementExpression::Constant(*v)))
    } else if let Some(i) = metadata.name_to_element_variable.get(token) {
        Ok(ArgumentExpression::Element(ElementExpression::Variable(*i)))
    } else if let Some(i) = metadata.name_to_set_variable.get(token) {
        Ok(ArgumentExpression::Set(SetExpression::SetVariable(*i)))
    } else if let Some(i) = metadata.name_to_vector_variable.get(token) {
        Ok(ArgumentExpression::Set(SetExpression::VectorVariable(*i)))
    } else if token == "stage" {
        Ok(ArgumentExpression::Element(ElementExpression::Stage))
    } else {
        let v: variable::Element = token.parse().map_err(|e| {
            ParseErr::new(format!("could not parse {} as a number: {:?}", token, e))
        })?;
        Ok(ArgumentExpression::Element(ElementExpression::Constant(v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn generate_metadata() -> state::StateMetadata {
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
            integer_variable_names,
            name_to_integer_variable,
            ..Default::default()
        }
    }

    fn generate_parameters() -> collections::HashMap<String, usize> {
        let mut parameters = collections::HashMap::new();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    #[test]
    fn parse_set_expression_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, SetExpression::SetVariable(0)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["p0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, SetExpression::VectorVariable(0)));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_set_expression_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &metadata, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["!", "n2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &parameters);
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
        let result = parse_set_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "-", "s2", "n1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "&", "s2", "e1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "/", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_expression_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_element_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Variable(1)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_element_expression(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Constant(11)));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_element_expression_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_element_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["p1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_element_expression(&tokens, &metadata, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_argument_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["!", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Set(SetExpression::Complement(_))
        ));
        if let ArgumentExpression::Set(SetExpression::Complement(s)) = expression {
            assert!(matches!(*s, SetExpression::SetVariable(2)));
        }
        assert_eq!(rest, &tokens[2..]);

        let tokens: Vec<String> = ["(", "+", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Set(SetExpression::SetOperation(SetOperator::Union, _, _))
        ));
        if let ArgumentExpression::Set(SetExpression::SetOperation(SetOperator::Union, x, y)) =
            expression
        {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(*y, SetExpression::SetVariable(1)));
        }
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_argument_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [")", "(", "+", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &parameters);
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
        let result = parse_argument(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Element(ElementExpression::Stage)
        ));

        assert_eq!(rest, &tokens[1..]);
        let tokens: Vec<String> = ["e1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Element(ElementExpression::Variable(1))
        ));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Set(SetExpression::SetVariable(1))
        ));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["p1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Set(SetExpression::VectorVariable(1))
        ));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Element(ElementExpression::Constant(11))
        ));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["param", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Element(ElementExpression::Constant(0))
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
        let result = parse_argument(&tokens, &metadata, &parameters);
        assert!(result.is_err());
    }
}
