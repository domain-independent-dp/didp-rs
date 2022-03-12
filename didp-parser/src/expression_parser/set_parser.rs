use super::environment;
use super::ParseErr;
use crate::expression::{
    ArgumentExpression, ElementExpression, SetElementOperator, SetExpression, SetOperator,
};
use crate::variable;

pub fn parse_set_expression<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (expression, rest) = parse_argument(tokens, env)?;
    match expression {
        ArgumentExpression::Set(expression) => Ok((expression, rest)),
        _ => Err(ParseErr::Reason(format!(
            "not a set expression: {:?}",
            expression
        ))),
    }
}

pub fn parse_element_expression<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(ElementExpression, &'a [String]), ParseErr> {
    let (expression, rest) = parse_argument(tokens, env)?;
    match expression {
        ArgumentExpression::Element(expression) => Ok((expression, rest)),
        _ => Err(ParseErr::Reason(format!(
            "not an element expression: {:?}",
            expression
        ))),
    }
}

pub fn parse_argument<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(ArgumentExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "!" => {
            let (expression, rest) = parse_set_expression(rest, env)?;
            Ok((
                ArgumentExpression::Set(SetExpression::Complement(Box::new(expression))),
                rest,
            ))
        }
        "(" => {
            let (expression, rest) = parse_operation(rest, env)?;
            Ok((ArgumentExpression::Set(expression), rest))
        }
        ")" => Err(ParseErr::Reason("unexpected `)`".to_string())),
        _ => {
            let argument = parse_atom(token, env)?;
            Ok((argument, rest))
        }
    }
}

fn parse_operation<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    let (x, rest) = parse_argument(rest, env)?;
    let (y, rest) = parse_argument(rest, env)?;
    let rest = super::parse_closing(rest)?;

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
            args => Err(ParseErr::Reason(format!(
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
            args => Err(ParseErr::Reason(format!(
                "unexpected arguments for `-`: {:?}",
                args
            ))),
        },
        "*" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                SetExpression::SetOperation(SetOperator::Intersect, Box::new(x), Box::new(y)),
                rest,
            )),
            args => Err(ParseErr::Reason(format!(
                "unexpected arguments for `*`: {:?}",
                args
            ))),
        },
        op => Err(ParseErr::Reason(format!("no such operator: {}", op))),
    }
}

fn parse_atom<T: variable::Numeric>(
    token: &str,
    env: &environment::Environment<T>,
) -> Result<ArgumentExpression, ParseErr> {
    if token == "stage" {
        Ok(ArgumentExpression::Element(ElementExpression::Stage))
    } else if let Some(i) = env.element_variables.get(token) {
        Ok(ArgumentExpression::Element(ElementExpression::Variable(*i)))
    } else if let Some(i) = env.set_variables.get(token) {
        Ok(ArgumentExpression::Set(SetExpression::SetVariable(*i)))
    } else if let Some(i) = env.permutation_variables.get(token) {
        Ok(ArgumentExpression::Set(SetExpression::PermutationVariable(
            *i,
        )))
    } else {
        let n: variable::ElementVariable = token.parse().map_err(|e| {
            ParseErr::Reason(format!("could not parse {} as a number: {:?}", token, e))
        })?;
        Ok(ArgumentExpression::Element(ElementExpression::Constant(n)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn generate_env() -> environment::Environment<variable::IntegerVariable> {
        let mut set_variables = HashMap::new();
        set_variables.insert("s0".to_string(), 0);
        set_variables.insert("s1".to_string(), 1);
        set_variables.insert("s2".to_string(), 2);
        set_variables.insert("s3".to_string(), 3);

        let mut permutation_variables = HashMap::new();
        permutation_variables.insert("p0".to_string(), 0);
        permutation_variables.insert("p1".to_string(), 1);
        permutation_variables.insert("p2".to_string(), 2);
        permutation_variables.insert("p3".to_string(), 3);

        let mut element_variables = HashMap::new();
        element_variables.insert("e0".to_string(), 0);
        element_variables.insert("e1".to_string(), 1);
        element_variables.insert("e2".to_string(), 2);
        element_variables.insert("e3".to_string(), 3);

        let mut numeric_variables = HashMap::new();
        numeric_variables.insert("n0".to_string(), 0);
        numeric_variables.insert("n1".to_string(), 1);
        numeric_variables.insert("n2".to_string(), 2);
        numeric_variables.insert("n3".to_string(), 3);

        let mut resource_variables = HashMap::new();
        resource_variables.insert("r0".to_string(), 0);
        resource_variables.insert("r1".to_string(), 1);
        resource_variables.insert("r2".to_string(), 2);
        resource_variables.insert("r3".to_string(), 3);

        environment::Environment {
            set_variables,
            permutation_variables,
            element_variables,
            numeric_variables,
            resource_variables,
            functions_1d: HashMap::new(),
            functions_2d: HashMap::new(),
            functions_3d: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    #[test]
    fn parse_set_expression_ok() {
        let env = generate_env();

        let tokens: Vec<String> = ["s0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, SetExpression::SetVariable(0)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["p0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, SetExpression::PermutationVariable(0)));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_set_expression_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["e1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_complemnt_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["!", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &env);
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
        let env = generate_env();

        let tokens: Vec<String> = ["!", "e2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["!", "n2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn pare_set_operation_ok() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "+", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &env);
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
        let result = parse_set_expression(&tokens, &env);
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

        let tokens: Vec<String> = ["(", "*", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &env);
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
        let result = parse_set_expression(&tokens, &env);
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
        let result = parse_set_expression(&tokens, &env);
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
        let env = generate_env();

        let tokens: Vec<String> = ["(", "+", "s2", "n1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "-", "s2", "n1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "*", "s2", "e1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "/", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_expression_ok() {
        let env = generate_env();

        let tokens: Vec<String> = ["e1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_element_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Variable(1)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_element_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, ElementExpression::Constant(11)));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_element_expression_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["s1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_element_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["p1", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_element_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_argument_ok() {
        let env = generate_env();

        let tokens: Vec<String> = ["!", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &env);
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
        let result = parse_argument(&tokens, &env);
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
        let env = generate_env();

        let tokens: Vec<String> = [")", "(", "+", "s2", "s1", ")", "e0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let env = generate_env();

        let tokens: Vec<String> = ["stage", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &env);
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
        let result = parse_argument(&tokens, &env);
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
        let result = parse_argument(&tokens, &env);
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
        let result = parse_argument(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Set(SetExpression::PermutationVariable(1))
        ));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Element(ElementExpression::Constant(11))
        ));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_atom_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["e4", ")", "1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &env);
        assert!(result.is_err());
    }
}
