use crate::expression;
use crate::variable;
use std::fmt;
use std::str;

mod condition_parser;
mod environment;
mod function_parser;
mod numeric_parser;
mod set_parser;

pub use environment::Environment;

#[derive(Debug)]
pub enum ParseErr {
    Reason(String),
}

pub fn parse_numeric<T: variable::Numeric>(
    text: String,
    env: &environment::Environment<T>,
) -> Result<expression::NumericExpression<T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = tokenize(text);
    let (expression, rest) = numeric_parser::parse_expression(&tokens, env)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::Reason(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

pub fn parse_set<T: variable::Numeric>(
    text: String,
    env: &environment::Environment<T>,
) -> Result<expression::SetExpression, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = tokenize(text);
    let (expression, rest) = set_parser::parse_set_expression(&tokens, env)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::Reason(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

pub fn parse_element<T: variable::Numeric>(
    text: String,
    env: &environment::Environment<T>,
) -> Result<expression::ElementExpression, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = tokenize(text);
    let (expression, rest) = set_parser::parse_element_expression(&tokens, env)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::Reason(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

pub fn parse_condition<T: variable::Numeric>(
    text: String,
    env: &environment::Environment<T>,
) -> Result<expression::Condition<T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = tokenize(text);
    let (expression, rest) = condition_parser::parse_expression(&tokens, env)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::Reason(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

pub fn tokenize(text: String) -> Vec<String> {
    text.replace("(", " ( ")
        .replace(")", " ) ")
        .replace("|", " | ")
        .replace("!", " ! ")
        .split_whitespace()
        .map(|x| x.to_string())
        .collect()
}

fn parse_closing(tokens: &[String]) -> Result<&[String], ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    if token != ")" {
        Err(ParseErr::Reason(format!(
            "unexpected {}, expected `)`",
            token
        )))
    } else {
        Ok(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::numeric_function;
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

        let mut functions_1d = HashMap::new();
        let f1 = numeric_function::NumericFunction1D::new(Vec::new());
        functions_1d.insert("f1".to_string(), f1);
        let mut functions_2d = HashMap::new();
        let f2 = numeric_function::NumericFunction2D::new(Vec::new());
        functions_2d.insert("f2".to_string(), f2);
        let mut functions_3d = HashMap::new();
        let f3 = numeric_function::NumericFunction3D::new(Vec::new());
        functions_3d.insert("f3".to_string(), f3);
        let mut functions = HashMap::new();
        let f4 = numeric_function::NumericFunction::new(HashMap::new());
        functions.insert("f4".to_string(), f4);

        environment::Environment {
            set_variables,
            permutation_variables,
            element_variables,
            numeric_variables,
            resource_variables,
            functions_1d,
            functions_2d,
            functions_3d,
            functions,
        }
    }

    #[test]
    fn parse_numeric_ok() {
        let env = generate_env();
        let text = "(+ (- 5 (/ (f4 4 !s2 e0 3) (max (f2 2 e1) n0))) (* r1 (min 3 |(+ (* s0 (- s2 (+ s3 2))) (- s1 1))|)))".to_string();
        let result = parse_numeric(text, &env);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_numeric_err() {
        let env = generate_env();
        let text = "(+ cost 1))".to_string();
        let result = parse_numeric(text, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_ok() {
        let env = generate_env();
        let text = "(+ (* s0 (- s2 (+ s3 2))) (- s1 1))".to_string();
        let result = parse_set(text, &env);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_set_err() {
        let env = generate_env();
        let text = "s0)".to_string();
        let result = parse_set(text, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_ok() {
        let env = generate_env();
        let text = "e0".to_string();
        let result = parse_element(text, &env);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_element_err() {
        let env = generate_env();
        let text = "0)".to_string();
        let result = parse_element(text, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_condition_ok() {
        let env = generate_env();
        let text = "(not (and (and (and (> n0 2) (is_subset s0 p0)) (is_empty s0)) (or (< 1 n1) (is_in 2 s0))))"
            .to_string();
        let result = parse_condition(text, &env);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_condition_err() {
        let env = generate_env();
        let text = "(is_empty s[0]))".to_string();
        let result = parse_condition(text, &env);
        assert!(result.is_err());
    }

    #[test]
    fn tokenize_text() {
        let text = "(+ (- 5 (/ (f4 4 !s2 e0 3) (max (f2 2 e1) n0))) (* r1 (min 3 |(+ (* s0 (- s2 (+ s3 2))) (- s1 1))|)))".to_string();
        assert_eq!(
            tokenize(text),
            [
                "(", "+", "(", "-", "5", "(", "/", "(", "f4", "4", "!", "s2", "e0", "3", ")", "(",
                "max", "(", "f2", "2", "e1", ")", "n0", ")", ")", ")", "(", "*", "r1", "(", "min",
                "3", "|", "(", "+", "(", "*", "s0", "(", "-", "s2", "(", "+", "s3", "2", ")", ")",
                ")", "(", "-", "s1", "1", ")", ")", "|", ")", ")", ")",
            ]
        );
    }

    #[test]
    fn parse_closing_ok() {
        let tokens: Vec<String> = [")", "(", "+", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_closing(&tokens);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), &tokens[1..]);
    }

    #[test]
    fn parse_closing_err() {
        let tokens: Vec<String> = ["(", "+", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_closing(&tokens);
        assert!(result.is_err());
    }
}
