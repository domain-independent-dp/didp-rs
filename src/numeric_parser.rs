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
    let (expression, rest) = parse_tokens(&tokens, problem)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::Reason(format!(
            "unexpected tokens at the end: {}",
            rest.join(" ")
        )))
    }
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
        return parse_function_1d(f, rest, problem);
    }
    if let Some(f) = problem.functions_2d.get(name) {
        return parse_function_2d(f, rest, problem);
    }
    if let Some(f) = problem.functions_3d.get(name) {
        return parse_function_3d(f, rest, problem);
    }
    if let Some(f) = problem.functions.get(name) {
        return parse_function(f, rest, problem);
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
    let (token, rest) = rest
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    if token != "|" {
        return Err(ParseErr::Reason(format!(
            "unexpected {}, expected `|`",
            token
        )));
    }
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
    Ok(NumericExpression::Constant(n))
}

fn parse_argument<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(ArgumentExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (expression, rest) = parse_set_operation(rest, problem)?;
            Ok((ArgumentExpression::Set(expression), rest))
        }
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
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    let (x, rest) = parse_argument(rest, problem)?;
    let (y, rest) = parse_argument(rest, problem)?;
    let rest = parse_closing(rest)?;

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
            _ => Err(ParseErr::Reason("unexpected arguments for `+`".to_string())),
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
            _ => Err(ParseErr::Reason("unexpected arguments for `-`".to_string())),
        },
        "*" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                SetExpression::SetOperation(SetOperator::Intersect, Box::new(x), Box::new(y)),
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
        return Ok(ArgumentExpression::Element(ElementExpression::Variable(i)));
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
        return Ok(ArgumentExpression::Set(SetExpression::PermutationVariable(
            i,
        )));
    }

    let n: variable::ElementVariable = token
        .parse()
        .map_err(|e| ParseErr::Reason(format!("could not parse {} as a number: {:?}", token, e)))?;
    Ok(ArgumentExpression::Element(ElementExpression::Constant(n)))
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
    fn parse_ok() {
        let problem = generate_problem();
        let text = "(+ (- 5 (/ (f4 4 s[2] e[0] 3) (max (f2 2 e[1]) n[0]))) (* r[1] (min 3 |(+ (* s[0] (- s[2] (+ s[3] 2))) (- s[1] 1))|)))".to_string();
        let result = parse(text, &problem);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_err() {
        let problem = generate_problem();
        let text = "(+ g 1))".to_string();
        let result = parse(text, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn tokenize_text() {
        let text = "(+ (- 5 (/ (f4 4 s[2] e[0] 3) (max (f2 2 e[1]) n[0]))) (* r[1] (min 3 |(+ (* s[0] (- s[2] (+ s[3] 2))) (- s[1] 1))|)))".to_string();
        assert_eq!(
            tokenize(text),
            [
                "(", "+", "(", "-", "5", "(", "/", "(", "f4", "4", "s[2]", "e[0]", "3", ")", "(",
                "max", "(", "f2", "2", "e[1]", ")", "n[0]", ")", ")", ")", "(", "*", "r[1]", "(",
                "min", "3", "|", "(", "+", "(", "*", "s[0]", "(", "-", "s[2]", "(", "+", "s[3]",
                "2", ")", ")", ")", "(", "-", "s[1]", "1", ")", ")", "|", ")", ")", ")",
            ]
        );
    }

    #[test]
    fn parse_token_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "+", "g", "1", ")", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_tokens(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Add, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Add, x, y) = expression {
            assert!(matches!(*x, NumericExpression::G));
            assert!(matches!(*y, NumericExpression::Constant(1)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["|", "s[0]", "|", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_tokens(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Cardinality(SetExpression::SetVariable(0))
        ));
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["r[0]", "2", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_tokens(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::ResourceVariable(0)));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_token_err() {
        let problem = generate_problem();

        let tokens = Vec::new();
        let result = parse_tokens(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "(", "+", "g", "1", ")", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_tokens(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_operation_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["f1", "e[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Function1D(_, _)));
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["f2", "0", "e[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Function2D(_, _, _)));
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f3", "0", "e[0]", "s[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function3DSumZ(_, _, _, _)
        ));
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["f4", "0", "e[0]", "s[0]", "p[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::FunctionSum(_, _)));
        assert_eq!(rest, &tokens[6..]);

        let tokens: Vec<String> = ["+", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Add, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Add, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["-", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Subtract, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Subtract, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["*", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Multiply, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Multiply, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["/", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Divide, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Divide, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["min", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Min, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Min, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["max", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Max, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Max, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_operation_err() {
        let problem = generate_problem();

        let tokens = Vec::new();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["+", "0", "n[0]", "n[1]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["+", "0", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["^", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_1d_ok() {
        let problem = generate_problem();
        let f = &problem.functions_1d["f1"];

        let tokens: Vec<String> = ["e[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_1d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Function1D(_, _)));
        if let NumericExpression::Function1D(g, x) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[2..]);

        let tokens: Vec<String> = ["s[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_1d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Function1DSum(_, _)));
        if let NumericExpression::Function1DSum(g, x) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
        }
        assert_eq!(rest, &tokens[2..]);
    }

    #[test]
    fn parse_function_1d_err() {
        let problem = generate_problem();
        let f = &problem.functions_1d["f1"];

        let tokens: Vec<String> = ["e[0]", "0", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_1d(f, &tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "n[0]", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_function_1d(f, &tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_2d_ok() {
        let problem = generate_problem();
        let f = &problem.functions_2d["f2"];

        let tokens: Vec<String> = ["0", "e[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_2d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Function2D(_, _, _)));
        if let NumericExpression::Function2D(g, x, y) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["s[0]", "p[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_2d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function2DSum(_, _, _)
        ));
        if let NumericExpression::Function2DSum(g, x, y) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["s[0]", "e[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_2d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function2DSumX(_, _, _)
        ));
        if let NumericExpression::Function2DSumX(g, x, y) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["0", "p[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_2d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function2DSumY(_, _, _)
        ));
        if let NumericExpression::Function2DSumY(g, x, y) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_function_2d_err() {
        let problem = generate_problem();
        let f = &problem.functions_2d["f2"];

        let tokens: Vec<String> = ["0", "e[0]", "p[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_2d(f, &tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_2d(f, &tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_3d_ok() {
        let problem = generate_problem();
        let f = &problem.functions_3d["f3"];

        let tokens: Vec<String> = ["0", "1", "e[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function3D(_, _, _, _)
        ));
        if let NumericExpression::Function3D(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s[0]", "s[1]", "p[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function3DSum(_, _, _, _)
        ));
        if let NumericExpression::Function3DSum(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s[0]", "1", "e[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function3DSumX(_, _, _, _)
        ));
        if let NumericExpression::Function3DSumX(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "s[1]", "e[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function3DSumY(_, _, _, _)
        ));
        if let NumericExpression::Function3DSumY(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "1", "p[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function3DSumZ(_, _, _, _)
        ));
        if let NumericExpression::Function3DSumZ(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s[0]", "s[1]", "e[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function3DSumXY(_, _, _, _)
        ));
        if let NumericExpression::Function3DSumXY(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s[0]", "1", "p[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function3DSumXZ(_, _, _, _)
        ));
        if let NumericExpression::Function3DSumXZ(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "s[1]", "p[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function3DSumYZ(_, _, _, _)
        ));
        if let NumericExpression::Function3DSumYZ(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_function_3d_err() {
        let problem = generate_problem();
        let f = &problem.functions_3d["f3"];

        let tokens: Vec<String> = ["0", "1", "e[0]", "2", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "1", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function_3d(f, &tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_ok() {
        let problem = generate_problem();
        let f = &problem.functions["f4"];
        let tokens: Vec<String> = ["s[2]", "1", "e[0]", "p[3]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function(f, &tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::FunctionSum(_, _)));
        if let NumericExpression::FunctionSum(g, args) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert_eq!(args.len(), 4);
            assert!(matches!(
                args[0],
                ArgumentExpression::Set(SetExpression::SetVariable(2))
            ));
            assert!(matches!(
                args[1],
                ArgumentExpression::Element(ElementExpression::Constant(1))
            ));
            assert!(matches!(
                args[2],
                ArgumentExpression::Element(ElementExpression::Variable(0))
            ));
            assert!(matches!(
                args[3],
                ArgumentExpression::Set(SetExpression::PermutationVariable(3))
            ));
        }
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_function_err() {
        let problem = generate_problem();
        let f = &problem.functions["f4"];

        let tokens: Vec<String> = ["s[2]", "1", "e[0]", "p[3]", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function(f, &tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["s[2]", "1", "e[0]", "p[3]", "n[0]"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_function(f, &tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn pare_cardinality_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["s[2]", "|", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_cardinality(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Cardinality(SetExpression::SetVariable(2))
        ));
        assert_eq!(rest, &tokens[2..]);
    }

    #[test]
    fn pare_cardinality_err() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["e[2]", "|", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_cardinality(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["s[2]", "s[0]", "|", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_cardinality(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let token = "g";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), NumericExpression::G));

        let token = "n[11]";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), NumericExpression::Variable(11)));

        let token = "r[11]";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            NumericExpression::ResourceVariable(11)
        ));

        let token = "11";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), NumericExpression::Constant(11)));
    }

    #[test]
    fn parse_atom_err() {
        let token = "h";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_err());

        let token = "e[1]";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_err());
    }

    #[test]
    fn parse_argument_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "+", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &problem);
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

        let tokens: Vec<String> = ["e[11]", "(", "+", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Element(ElementExpression::Variable(11))
        ));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_argument_err() {
        let problem = generate_problem();

        let tokens: Vec<String> = [")", "(", "+", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn pare_set_operation_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["+", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_operation(&tokens, &problem);
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
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["-", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_operation(&tokens, &problem);
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
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["*", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_operation(&tokens, &problem);
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
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["+", "s[2]", "e[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_operation(&tokens, &problem);
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
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["-", "s[2]", "1", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_operation(&tokens, &problem);
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
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn pare_set_operation_err() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["+", "s[2]", "n[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["-", "s[2]", "n[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["*", "s[2]", "e[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["/", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_set_operation(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_argument_atom_ok() {
        let token = "e[11]";
        let result = parse_argument_atom(token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ArgumentExpression::Element(ElementExpression::Variable(11))
        ));

        let token = "s[11]";
        let result = parse_argument_atom(token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ArgumentExpression::Set(SetExpression::SetVariable(11))
        ));

        let token = "p[11]";
        let result = parse_argument_atom(token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ArgumentExpression::Set(SetExpression::PermutationVariable(11))
        ));

        let token = "11";
        let result = parse_argument_atom(token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ArgumentExpression::Element(ElementExpression::Constant(11))
        ));
    }

    #[test]
    fn parse_argument_atom_err() {
        let token = "n[11]";
        let result = parse_argument_atom(token);
        assert!(result.is_err());
        let token = "s[11";
        let result = parse_argument_atom(token);
        assert!(result.is_err());
        let token = "ss[11]";
        let result = parse_argument_atom(token);
        assert!(result.is_err());
        let token = "e[11]]";
        let result = parse_argument_atom(token);
        assert!(result.is_err());
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
