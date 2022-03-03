use crate::expression;
use crate::problem;
use crate::variable;
use std::fmt;
use std::str;

mod condition_parser;
mod function_parser;
mod numeric_parser;
mod set_parser;

#[derive(Debug)]
pub enum ParseErr {
    Reason(String),
}

pub fn parse_numeric<T: variable::Numeric>(
    text: String,
    problem: &problem::Problem<T>,
) -> Result<expression::NumericExpression<T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = tokenize(text);
    let (expression, rest) = numeric_parser::parse_expression(&tokens, problem)?;
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
    problem: &problem::Problem<T>,
) -> Result<expression::SetExpression, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = tokenize(text);
    let (expression, rest) = set_parser::parse_set_expression(&tokens, problem)?;
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
    problem: &problem::Problem<T>,
) -> Result<expression::ElementExpression, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = tokenize(text);
    let (expression, rest) = set_parser::parse_element_expression(&tokens, problem)?;
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
    problem: &problem::Problem<T>,
) -> Result<expression::Condition<T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = tokenize(text);
    let (expression, rest) = condition_parser::parse_expression(&tokens, problem)?;
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

    fn generate_problem() -> problem::Problem<variable::IntegerVariable> {
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

        problem::Problem {
            set_variable_to_max_size: vec![4],
            permutation_variable_to_max_length: vec![3],
            element_to_set: vec![0, 1],
            functions_1d,
            functions_2d,
            functions_3d,
            functions,
        }
    }

    #[test]
    fn parse_numeric_ok() {
        let problem = generate_problem();
        let text = "(+ (- 5 (/ (f4 4 !s[2] e[0] 3) (max (f2 2 e[1]) n[0]))) (* r[1] (min 3 |(+ (* s[0] (- s[2] (+ s[3] 2))) (- s[1] 1))|)))".to_string();
        let result = parse_numeric(text, &problem);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_numeric_err() {
        let problem = generate_problem();
        let text = "(+ g 1))".to_string();
        let result = parse_numeric(text, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_set_ok() {
        let problem = generate_problem();
        let text = "(+ (* s[0] (- s[2] (+ s[3] 2))) (- s[1] 1))".to_string();
        let result = parse_set(text, &problem);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_set_err() {
        let problem = generate_problem();
        let text = "s[0])".to_string();
        let result = parse_set(text, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_ok() {
        let problem = generate_problem();
        let text = "e[0]".to_string();
        let result = parse_element(text, &problem);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_element_err() {
        let problem = generate_problem();
        let text = "0)".to_string();
        let result = parse_element(text, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_condition_ok() {
        let problem = generate_problem();
        let text = "(not (and (and (and (> n[0] 2) (is_subset s[0] p[0])) (is_empty s[0])) (or (< 1 n[1]) (is_in 2 s[0]))))"
            .to_string();
        let result = parse_condition(text, &problem);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_condition_err() {
        let problem = generate_problem();
        let text = "(is_empty s[0]))".to_string();
        let result = parse_condition(text, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn tokenize_text() {
        let text = "(+ (- 5 (/ (f4 4 !s[2] e[0] 3) (max (f2 2 e[1]) n[0]))) (* r[1] (min 3 |(+ (* s[0] (- s[2] (+ s[3] 2))) (- s[1] 1))|)))".to_string();
        assert_eq!(
            tokenize(text),
            [
                "(", "+", "(", "-", "5", "(", "/", "(", "f4", "4", "!", "s[2]", "e[0]", "3", ")",
                "(", "max", "(", "f2", "2", "e[1]", ")", "n[0]", ")", ")", ")", "(", "*", "r[1]",
                "(", "min", "3", "|", "(", "+", "(", "*", "s[0]", "(", "-", "s[2]", "(", "+",
                "s[3]", "2", ")", ")", ")", "(", "-", "s[1]", "1", ")", ")", "|", ")", ")", ")",
            ]
        );
    }

    #[test]
    fn parse_closing_ok() {
        let tokens: Vec<String> = [")", "(", "+", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_closing(&tokens);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), &tokens[1..]);
    }

    #[test]
    fn parse_closing_err() {
        let tokens: Vec<String> = ["(", "+", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_closing(&tokens);
        assert!(result.is_err());
    }
}
