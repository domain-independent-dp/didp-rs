use crate::numeric_expression::*;
use crate::numeric_function;
use crate::problem;
use crate::variable;
use lazy_static::lazy_static;
use regex::Regex;
use std::fmt;
use std::str;

#[derive(Debug)]
pub enum ParseErr {
    Reason(String),
}

pub fn parse<T: variable::Numeric>(
    text: String,
    problem: &problem::Problem<T>,
) -> Result<NumericExpression<T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = tokenize(text);
    let (expression, _) = parse_tokens(&tokens, problem)?;
    Ok(expression)
}

pub fn tokenize(text: String) -> Vec<String> {
    text.replace("(", " ( ")
        .replace(")", " ) ")
        .replace("|", " | ")
        .split_whitespace()
        .map(|x| x.to_string())
        .collect()
}

pub fn parse_tokens<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "(" => parse_operation(rest, problem),
        ")" => Err(ParseErr::Reason("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, problem),
        _ => {
            let expression = parse_atom(token)?;
            Ok((expression, rest))
        }
    }
}

fn parse_operation<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (name, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    if let Some(f) = problem.functions_1d.get(name) {
        return parse_function_1d(f, tokens, problem);
    }
    if let Some(f) = problem.functions_2d.get(name) {
        return parse_function_2d(f, tokens, problem);
    }
    if let Some(f) = problem.functions_3d.get(name) {
        return parse_function_3d(f, tokens, problem);
    }
    if let Some(f) = problem.functions.get(name) {
        return parse_function(f, tokens, problem);
    }
    let (x, rest) = parse_tokens(rest, problem)?;
    let (y, rest) = parse_tokens(rest, problem)?;
    let rest = parse_closing(rest)?;
    match &name[..] {
        "+" => Ok((
            NumericExpression::NumericOperation(NumericOperator::Add, Box::new(x), Box::new(y)),
            rest,
        )),
        "-" => Ok((
            NumericExpression::NumericOperation(
                NumericOperator::Subtract,
                Box::new(x),
                Box::new(y),
            ),
            rest,
        )),
        "*" => Ok((
            NumericExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(x),
                Box::new(y),
            ),
            rest,
        )),
        "/" => Ok((
            NumericExpression::NumericOperation(NumericOperator::Divide, Box::new(x), Box::new(y)),
            rest,
        )),
        "min" => Ok((
            NumericExpression::NumericOperation(NumericOperator::Min, Box::new(x), Box::new(y)),
            rest,
        )),
        "max" => Ok((
            NumericExpression::NumericOperation(NumericOperator::Max, Box::new(x), Box::new(y)),
            rest,
        )),
        op => Err(ParseErr::Reason(format!("no such operator {}", op))),
    }
}

fn parse_function_1d<'a, 'b, T: variable::Numeric>(
    f: &'b numeric_function::NumericFunction1D<T>,
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (x, rest) = parse_argument(tokens, problem)?;
    let rest = parse_closing(rest)?;
    match x {
        ArgumentExpression::Element(x) => Ok((NumericExpression::Function1D(f, x), rest)),
        ArgumentExpression::Set(x) => Ok((NumericExpression::Function1DSum(f, x), rest)),
    }
}

fn parse_function_2d<'a, 'b, T: variable::Numeric>(
    f: &'b numeric_function::NumericFunction2D<T>,
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (x, rest) = parse_argument(tokens, problem)?;
    let (y, rest) = parse_argument(rest, problem)?;
    let rest = parse_closing(rest)?;
    match (x, y) {
        (ArgumentExpression::Element(x), ArgumentExpression::Element(y)) => {
            Ok((NumericExpression::Function2D(&f, x, y), rest))
        }
        (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => {
            Ok((NumericExpression::Function2DSum(&f, x, y), rest))
        }
        (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => {
            Ok((NumericExpression::Function2DSumX(&f, x, y), rest))
        }
        (ArgumentExpression::Element(x), ArgumentExpression::Set(y)) => {
            Ok((NumericExpression::Function2DSumY(&f, x, y), rest))
        }
    }
}

fn parse_function_3d<'a, 'b, T: variable::Numeric>(
    f: &'b numeric_function::NumericFunction3D<T>,
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (x, rest) = parse_argument(tokens, problem)?;
    let (y, rest) = parse_argument(rest, problem)?;
    let (z, rest) = parse_argument(rest, problem)?;
    let rest = parse_closing(rest)?;
    match (x, y, z) {
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Element(z),
        ) => Ok((NumericExpression::Function3D(&f, x, y, z), rest)),
        (ArgumentExpression::Set(x), ArgumentExpression::Set(y), ArgumentExpression::Set(z)) => {
            Ok((NumericExpression::Function3DSum(&f, x, y, z), rest))
        }
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Element(z),
        ) => Ok((NumericExpression::Function3DSumX(&f, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Element(z),
        ) => Ok((NumericExpression::Function3DSumY(&f, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Set(z),
        ) => Ok((NumericExpression::Function3DSumZ(&f, x, y, z), rest)),
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Element(z),
        ) => Ok((NumericExpression::Function3DSumXY(&f, x, y, z), rest)),
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Set(z),
        ) => Ok((NumericExpression::Function3DSumXZ(&f, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Set(z),
        ) => Ok((NumericExpression::Function3DSumYZ(&f, x, y, z), rest)),
    }
}

fn parse_function<'a, 'b, T: variable::Numeric>(
    f: &'b numeric_function::NumericFunction<T>,
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::Reason("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((NumericExpression::FunctionSum(f, args), rest));
        }
        let (expression, new_xs) = parse_argument(xs, problem)?;
        args.push(expression);
        xs = new_xs;
    }
}

fn parse_cardinality<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr> {
    let (expression, rest) = parse_argument(tokens, problem)?;
    let rest = parse_closing(rest)?;
    match expression {
        ArgumentExpression::Set(expression) => {
            Ok((NumericExpression::Cardinality(expression), rest))
        }
        _ => Err(ParseErr::Reason(
            "cardinality of not a set expression".to_string(),
        )),
    }
}

fn parse_atom<'a, 'b, T: variable::Numeric>(
    token: &'a str,
) -> Result<NumericExpression<'b, T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    lazy_static! {
        static ref NUMERIC: Regex = Regex::new(r"^n\[(\d+)\]$").unwrap();
        static ref RESOURCE: Regex = Regex::new(r"^r\[(\d+)\]$").unwrap();
    }

    if token == "g" {
        return Ok(NumericExpression::G);
    }

    if let Some(caps) = NUMERIC.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason("could not parse an index of a numeric variable".to_string())
        })?;
        return Ok(NumericExpression::Variable(i));
    }

    if let Some(caps) = RESOURCE.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason("could not parse an index of a resource variable".to_string())
        })?;
        return Ok(NumericExpression::ResourceVariable(i));
    }

    let n: T = token
        .parse()
        .map_err(|e| ParseErr::Reason(format!("could not parse {} as a number: {:?}", token, e)))?;
    Ok(NumericExpression::Number(n))
}

fn parse_argument<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(ArgumentExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "(" => parse_set_operation(rest, problem),
        ")" => Err(ParseErr::Reason("unexpected `)`".to_string())),
        _ => {
            let argument = parse_argument_atom(token)?;
            Ok((argument, rest))
        }
    }
}

fn parse_set_operation<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(ArgumentExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    let (x, rest) = parse_argument(rest, problem)?;
    let (y, rest) = parse_argument(rest, problem)?;
    let rest = parse_closing(rest)?;

    match &token[..] {
        "+" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                ArgumentExpression::Set(SetExpression::SetOperation(
                    SetOperator::Union,
                    Box::new(x),
                    Box::new(y),
                )),
                rest,
            )),
            (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => Ok((
                ArgumentExpression::Set(SetExpression::SetElementOperation(
                    SetElementOperator::Add,
                    Box::new(x),
                    y,
                )),
                rest,
            )),
            _ => Err(ParseErr::Reason("unexpected arguments for `+`".to_string())),
        },
        "-" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                ArgumentExpression::Set(SetExpression::SetOperation(
                    SetOperator::Difference,
                    Box::new(x),
                    Box::new(y),
                )),
                rest,
            )),
            (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => Ok((
                ArgumentExpression::Set(SetExpression::SetElementOperation(
                    SetElementOperator::Remove,
                    Box::new(x),
                    y,
                )),
                rest,
            )),
            _ => Err(ParseErr::Reason("unexpected arguments for `-`".to_string())),
        },
        "*" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                ArgumentExpression::Set(SetExpression::SetOperation(
                    SetOperator::Intersect,
                    Box::new(x),
                    Box::new(y),
                )),
                rest,
            )),
            _ => Err(ParseErr::Reason("unexpected arguments for `*`".to_string())),
        },
        op => Err(ParseErr::Reason(format!("no such operator: {}", op))),
    }
}

fn parse_argument_atom(token: &str) -> Result<ArgumentExpression, ParseErr> {
    lazy_static! {
        static ref ELEMENT: Regex = Regex::new(r"^e\[(\d+)\]$").unwrap();
        static ref SET: Regex = Regex::new(r"^s\[(\d+)\]$").unwrap();
        static ref PERMUTATION: Regex = Regex::new(r"^p\[(\d+)\]$").unwrap();
    }

    if let Some(caps) = ELEMENT.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason(format!(
                "could not parse an index of an element variable: {:?}",
                e
            ))
        })?;
        return Ok(ArgumentExpression::Element(
            ElementExpression::ElementVariable(i),
        ));
    }

    if let Some(caps) = SET.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason(format!(
                "could not parse an index of a set variable: {:?}",
                e
            ))
        })?;
        return Ok(ArgumentExpression::Set(SetExpression::SetVariable(i)));
    }

    if let Some(caps) = PERMUTATION.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason(format!(
                "could not parse an index of a set variable: {:?}",
                e
            ))
        })?;
        return Ok(ArgumentExpression::Set(SetExpression::SetVariable(i)));
    }

    let n: variable::ElementVariable = token
        .parse()
        .map_err(|e| ParseErr::Reason(format!("could not parse {} as a number: {:?}", token, e)))?;
    Ok(ArgumentExpression::Element(ElementExpression::Number(n)))
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
    use std::collections::HashMap;

    fn create_problem() -> problem::Problem<variable::IntegerVariable> {
        let mut functions_1d = HashMap::new();
        let f1 = numeric_function::NumericFunction1D::new(vec![0, 2, 4, 6]);
        functions_1d.insert("f1".to_string(), f1);

        let mut functions_2d = HashMap::new();
        let f2 = numeric_function::NumericFunction2D::new(vec![vec![0, 1], vec![2, 3]]);
        functions_2d.insert("f2".to_string(), f2);

        let mut functions_3d = HashMap::new();
        let f3 = numeric_function::NumericFunction3D::new(vec![vec![
            vec![0, 1, 2],
            vec![3, 4, 5],
            vec![6, 7, 8],
        ]]);
        functions_3d.insert("f3".to_string(), f3);

        let mut functions = HashMap::new();
        let mut map = HashMap::new();
        map.insert(vec![0, 0, 0, 0], 0);
        map.insert(vec![0, 0, 0, 1], 1);
        map.insert(vec![0, 0, 1, 0], 2);
        map.insert(vec![0, 0, 1, 1], 3);
        map.insert(vec![0, 1, 0, 0], 4);
        map.insert(vec![0, 1, 0, 1], 5);
        map.insert(vec![0, 1, 1, 0], 6);
        map.insert(vec![0, 1, 1, 1], 7);
        map.insert(vec![1, 0, 0, 0], 8);
        map.insert(vec![1, 0, 0, 1], 9);
        map.insert(vec![1, 0, 1, 0], 10);
        map.insert(vec![1, 0, 1, 1], 11);
        map.insert(vec![1, 1, 0, 0], 12);
        map.insert(vec![1, 1, 0, 1], 13);
        map.insert(vec![1, 1, 1, 0], 14);
        map.insert(vec![1, 1, 1, 1], 15);
        let f4 = numeric_function::NumericFunction::new(map);
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
    fn parse_ok() {}

    #[test]
    fn parse_err() {}

    #[test]
    fn tokenize_text() {
        let text = "(+ (- 5 (/ (f3 4 s[2] e[0] 3) (max (f4 2 e[1]) n[0]))) (* r[1] (min 3 |(+ (* s[0] (- s[2] (+ s[3] 2))) (- s[1] 1))|)))".to_string();
        assert_eq!(
            tokenize(text),
            [
                "(", "+", "(", "-", "5", "(", "/", "(", "f3", "4", "s[2]", "e[0]", "3", ")", "(",
                "max", "(", "f4", "2", "e[1]", ")", "n[0]", ")", ")", ")", "(", "*", "r[1]", "(",
                "min", "3", "|", "(", "+", "(", "*", "s[0]", "(", "-", "s[2]", "(", "+", "s[3]",
                "2", ")", ")", ")", "(", "-", "s[1]", "1", ")", ")", "|", ")", ")", ")",
            ]
        );
    }

    #[test]
    fn parse_token_ok() {}

    #[test]
    fn parse_token_err() {}

    #[test]
    fn parse_operation_ok() {}

    #[test]
    fn parse_operation_err() {}

    #[test]
    fn parse_function_1d_ok() {}

    #[test]
    fn parse_function_1d_err() {}

    #[test]
    fn parse_function_2d_ok() {}

    #[test]
    fn parse_function_2d_err() {}

    #[test]
    fn parse_function_3d_ok() {}

    #[test]
    fn parse_function_3d_err() {}

    #[test]
    fn parse_function_ok() {}

    #[test]
    fn parse_function_err() {}

    #[test]
    fn pare_cardinality_ok() {}

    #[test]
    fn pare_cardinality_err() {}

    #[test]
    fn parse_atom_ok() {}

    #[test]
    fn parse_atom_err() {}

    #[test]
    fn parse_argument_ok() {}

    #[test]
    fn parse_argument_err() {}

    #[test]
    fn pare_set_operation_ok() {}

    #[test]
    fn pare_set_operation_err() {}

    #[test]
    fn parse_argument_atom_ok() {}

    #[test]
    fn parse_argument_atom_err() {}

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
