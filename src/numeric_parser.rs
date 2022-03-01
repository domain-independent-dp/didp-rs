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

pub fn tokenize(text: String) -> Vec<String> {
    text.replace("(", " ( ")
        .replace(")", " ) ")
        .replace("|", " | ")
        .split_whitespace()
        .map(|x| x.to_string())
        .collect()
}

pub fn parse<'a, 'b, T: variable::Numeric>(
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
    let (x, rest) = parse(rest, problem)?;
    let (y, rest) = parse(rest, problem)?;
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

pub fn parse_closing(tokens: &[String]) -> Result<&[String], ParseErr> {
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
