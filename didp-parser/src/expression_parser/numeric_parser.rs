use super::element_parser;
use super::numeric_table_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{
    ArgumentExpression, NumericExpression, NumericOperator, NumericVectorExpression,
    VectorOrElementExpression,
};
use crate::state::StateMetadata;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::{Element, Integer, Numeric};
use rustc_hash::FxHashMap;
use std::fmt;
use std::str;

pub fn parse_expression<'a, T: Numeric>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(NumericExpression<T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    match parse_integer_expression(tokens, metadata, registry, parameters) {
        Ok(expression) => Ok(expression),
        Err(_) => parse_continuous_expression(tokens, metadata, registry, parameters),
    }
}

pub fn parse_integer_expression<'a, T: Numeric>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(NumericExpression<T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new(String::from("could not get token")))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new(String::from("could not get token")))?;
            if let Some((expression, rest)) = numeric_table_parser::parse_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.integer_tables,
            )? {
                Ok((NumericExpression::IntegerTable(expression), rest))
            } else if name == "length" {
                parse_length(rest, metadata, registry, parameters)
            } else if name == "last" {
                let (vector, rest) =
                    parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((NumericExpression::IntegerLast(Box::new(vector)), rest))
            } else if name == "at" {
                let (vector, rest) =
                    parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                let (i, rest) =
                    element_parser::parse_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((NumericExpression::IntegerAt(Box::new(vector), i), rest))
            } else if name == "reduce-sum" {
                let (vector, rest) =
                    parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((NumericExpression::IntegerReduceSum(Box::new(vector)), rest))
            } else if name == "reduce-product" {
                let (vector, rest) =
                    parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((
                    NumericExpression::IntegerReduceProduct(Box::new(vector)),
                    rest,
                ))
            } else {
                let (x, rest) = parse_integer_expression(rest, metadata, registry, parameters)?;
                let (y, rest) = parse_integer_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                let expression = parse_operation(name, x, y)?;
                Ok((expression, rest))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, metadata, registry, parameters),
        _ => {
            let expression = parse_integer_atom(token, metadata, registry)?;
            Ok((expression, rest))
        }
    }
}

pub fn parse_continuous_expression<'a, T: Numeric>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(NumericExpression<T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            if let Some((expression, rest)) = numeric_table_parser::parse_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.integer_tables,
            )? {
                Ok((NumericExpression::IntegerTable(expression), rest))
            } else if let Some((expression, rest)) = numeric_table_parser::parse_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.continuous_tables,
            )? {
                Ok((NumericExpression::ContinuousTable(expression), rest))
            } else if name == "length" {
                parse_length(rest, metadata, registry, parameters)
            } else if name == "last" {
                if let Ok((vector, rest)) =
                    parse_integer_vector_expression(rest, metadata, registry, parameters)
                {
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericExpression::IntegerLast(Box::new(vector)), rest))
                } else {
                    let (vector, rest) =
                        parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericExpression::ContinuousLast(Box::new(vector)), rest))
                }
            } else if name == "at" {
                if let Ok((vector, rest)) =
                    parse_integer_vector_expression(rest, metadata, registry, parameters)
                {
                    let (i, rest) =
                        element_parser::parse_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericExpression::IntegerAt(Box::new(vector), i), rest))
                } else {
                    let (vector, rest) =
                        parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                    let (i, rest) =
                        element_parser::parse_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericExpression::ContinuousAt(Box::new(vector), i), rest))
                }
            } else if name == "reduce-sum" {
                if let Ok((vector, rest)) =
                    parse_integer_vector_expression(rest, metadata, registry, parameters)
                {
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericExpression::IntegerReduceSum(Box::new(vector)), rest))
                } else {
                    let (vector, rest) =
                        parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((
                        NumericExpression::ContinuousReduceSum(Box::new(vector)),
                        rest,
                    ))
                }
            } else if name == "reduce-product" {
                if let Ok((vector, rest)) =
                    parse_integer_vector_expression(rest, metadata, registry, parameters)
                {
                    let rest = util::parse_closing(rest)?;
                    Ok((
                        NumericExpression::IntegerReduceProduct(Box::new(vector)),
                        rest,
                    ))
                } else {
                    let (vector, rest) =
                        parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((
                        NumericExpression::ContinuousReduceProduct(Box::new(vector)),
                        rest,
                    ))
                }
            } else {
                let (x, rest) = parse_continuous_expression(rest, metadata, registry, parameters)?;
                let (y, rest) = parse_continuous_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                let expression = parse_operation(name, x, y)?;
                Ok((expression, rest))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, metadata, registry, parameters),
        _ => {
            let expression = parse_atom(token, metadata, registry)?;
            Ok((expression, rest))
        }
    }
}

fn parse_operation<T: Numeric>(
    name: &str,
    x: NumericExpression<T>,
    y: NumericExpression<T>,
) -> Result<NumericExpression<T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    match &name[..] {
        "+" => Ok(NumericExpression::NumericOperation(
            NumericOperator::Add,
            Box::new(x),
            Box::new(y),
        )),
        "-" => Ok(NumericExpression::NumericOperation(
            NumericOperator::Subtract,
            Box::new(x),
            Box::new(y),
        )),
        "*" => Ok(NumericExpression::NumericOperation(
            NumericOperator::Multiply,
            Box::new(x),
            Box::new(y),
        )),
        "/" => Ok(NumericExpression::NumericOperation(
            NumericOperator::Divide,
            Box::new(x),
            Box::new(y),
        )),
        "min" => Ok(NumericExpression::NumericOperation(
            NumericOperator::Min,
            Box::new(x),
            Box::new(y),
        )),
        "max" => Ok(NumericExpression::NumericOperation(
            NumericOperator::Max,
            Box::new(x),
            Box::new(y),
        )),
        _ => Err(ParseErr::new(format!("no such operator `{}`", name))),
    }
}

fn parse_cardinality<'a, T: Numeric>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(NumericExpression<T>, &'a [String]), ParseErr> {
    let (expression, rest) =
        element_parser::parse_set_expression(tokens, metadata, registry, parameters)?;
    let (token, rest) = rest
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    if token != "|" {
        return Err(ParseErr::new(format!(
            "unexpected token: `{}`, expected `|`",
            token
        )));
    }
    Ok((NumericExpression::Cardinality(expression), rest))
}

fn parse_length<'a, T: Numeric>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(NumericExpression<T>, &'a [String]), ParseErr> {
    let (expression, rest) =
        element_parser::parse_vector_expression(tokens, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((NumericExpression::Length(expression), rest))
}

fn parse_integer_atom<T: Numeric>(
    token: &str,
    metadata: &StateMetadata,
    registry: &TableRegistry,
) -> Result<NumericExpression<T>, ParseErr> {
    if token == "cost" {
        Ok(NumericExpression::Cost)
    } else if let Some(v) = registry.integer_tables.name_to_constant.get(token) {
        Ok(NumericExpression::Constant(T::from(*v)))
    } else if let Some(i) = metadata.name_to_integer_variable.get(token) {
        Ok(NumericExpression::IntegerVariable(*i))
    } else if let Some(i) = metadata.name_to_integer_resource_variable.get(token) {
        Ok(NumericExpression::IntegerResourceVariable(*i))
    } else {
        let n: Integer = token.parse().map_err(|e| {
            ParseErr::new(format!("could not parse {} as a number: {:?}", token, e))
        })?;
        Ok(NumericExpression::Constant(T::from(n)))
    }
}

fn parse_atom<T: Numeric>(
    token: &str,
    metadata: &StateMetadata,
    registry: &TableRegistry,
) -> Result<NumericExpression<T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    if let Some(v) = registry.integer_tables.name_to_constant.get(token) {
        Ok(NumericExpression::Constant(T::from_integer(*v)))
    } else if let Some(v) = registry.continuous_tables.name_to_constant.get(token) {
        Ok(NumericExpression::Constant(T::from_continuous(*v)))
    } else if let Some(i) = metadata.name_to_integer_variable.get(token) {
        Ok(NumericExpression::IntegerVariable(*i))
    } else if let Some(i) = metadata.name_to_integer_resource_variable.get(token) {
        Ok(NumericExpression::IntegerResourceVariable(*i))
    } else if let Some(i) = metadata.name_to_continuous_variable.get(token) {
        Ok(NumericExpression::ContinuousVariable(*i))
    } else if let Some(i) = metadata.name_to_continuous_resource_variable.get(token) {
        Ok(NumericExpression::ContinuousResourceVariable(*i))
    } else if token == "cost" {
        Ok(NumericExpression::Cost)
    } else {
        let n: T = token.parse().map_err(|e| {
            ParseErr::new(format!("could not parse {} as a number: {:?}", token, e))
        })?;
        Ok(NumericExpression::Constant(n))
    }
}

fn parse_integer_vector_expression<'a, T: Numeric>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(NumericVectorExpression<T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            match &name[..] {
                "numeric-vector" => parse_integer_vector_constant(tokens, registry),
                "reverse" => {
                    let (vector, rest) =
                        parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericVectorExpression::Reverse(Box::new(vector)), rest))
                }
                "push" => {
                    let (v, rest) = parse_integer_expression(rest, metadata, registry, parameters)?;
                    let (vector, rest) =
                        parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericVectorExpression::Push(v, Box::new(vector)), rest))
                }
                "pop" => {
                    let (vector, rest) =
                        parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericVectorExpression::Pop(Box::new(vector)), rest))
                }
                "set" => {
                    let (v, rest) = parse_integer_expression(rest, metadata, registry, parameters)?;
                    let (vector, rest) =
                        parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                    let (i, rest) =
                        element_parser::parse_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericVectorExpression::Set(v, Box::new(vector), i), rest))
                }
                "table-vector" => {
                    let (name, rest) = rest
                        .split_first()
                        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
                    if let Some((expression, rest)) = parse_vector_table(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((expression, rest))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
                "sum-table-vector" => {
                    let (name, rest) = rest
                        .split_first()
                        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
                    if let Some((expression, rest)) = parse_vector_table_sum(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((expression, rest))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
                name => {
                    if let Ok((v, rest)) =
                        parse_integer_expression(rest, metadata, registry, parameters)
                    {
                        let (vector, rest) =
                            parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                        let rest = util::parse_closing(rest)?;
                        if let Some(expression) = parse_vector_numeric_operation(name, v, vector) {
                            Ok((expression, rest))
                        } else {
                            Err(ParseErr::new(format!(
                                "no such table or operation `{}`",
                                name
                            )))
                        }
                    } else {
                        let (x, rest) =
                            parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                        let (y, rest) =
                            parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                        let rest = util::parse_closing(rest)?;
                        if let Some(expression) = parse_vector_operation(name, x, y) {
                            Ok((expression, rest))
                        } else {
                            Err(ParseErr::new(format!(
                                "no such table or operation `{}`",
                                name
                            )))
                        }
                    }
                }
            }
        }
        _ => Err(ParseErr::new(format!("unexpected  token `{}`", token))),
    }
}

fn parse_integer_vector_constant<'a, T: Numeric>(
    tokens: &'a [String],
    registry: &TableRegistry,
) -> Result<(NumericVectorExpression<T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let mut result = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((NumericVectorExpression::Constant(result), rest));
        }
        let v = if let Some(v) = registry.integer_tables.name_to_constant.get(next_token) {
            T::from(*v)
        } else {
            let v: T = next_token.parse().map_err(|e| {
                ParseErr::new(format!(
                    "could not parse {} as a number: {:?}",
                    next_token, e
                ))
            })?;
            v
        };
        result.push(v);
        xs = rest;
    }
}

fn parse_continuous_vector_expression<'a, T: Numeric>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(NumericVectorExpression<T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            match &name[..] {
                "numeric-vector" => parse_continuous_vector_constant(tokens, registry),
                "reverse" => {
                    let (vector, rest) =
                        parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericVectorExpression::Reverse(Box::new(vector)), rest))
                }
                "push" => {
                    let (v, rest) =
                        parse_continuous_expression(rest, metadata, registry, parameters)?;
                    let (vector, rest) =
                        parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericVectorExpression::Push(v, Box::new(vector)), rest))
                }
                "pop" => {
                    let (vector, rest) =
                        parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericVectorExpression::Pop(Box::new(vector)), rest))
                }
                "set" => {
                    let (v, rest) =
                        parse_continuous_expression(rest, metadata, registry, parameters)?;
                    let (vector, rest) =
                        parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                    let (i, rest) =
                        element_parser::parse_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((NumericVectorExpression::Set(v, Box::new(vector), i), rest))
                }
                "table-vector" => {
                    let (name, rest) = rest
                        .split_first()
                        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
                    if let Some((expression, rest)) = parse_vector_table(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((expression, rest))
                    } else if let Some((expression, rest)) = parse_vector_table(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.continuous_tables,
                    )? {
                        Ok((expression, rest))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
                "sum-table-vector" => {
                    let (name, rest) = rest
                        .split_first()
                        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
                    if let Some((expression, rest)) = parse_vector_table_sum(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((expression, rest))
                    } else if let Some((expression, rest)) = parse_vector_table_sum(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.continuous_tables,
                    )? {
                        Ok((expression, rest))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
                name => {
                    if let Ok((v, rest)) =
                        parse_continuous_expression(rest, metadata, registry, parameters)
                    {
                        let (vector, rest) = parse_continuous_vector_expression(
                            rest, metadata, registry, parameters,
                        )?;
                        let rest = util::parse_closing(rest)?;
                        if let Some(expression) = parse_vector_numeric_operation(name, v, vector) {
                            Ok((expression, rest))
                        } else {
                            Err(ParseErr::new(format!(
                                "no such table or operation `{}`",
                                name
                            )))
                        }
                    } else {
                        let (x, rest) = parse_continuous_vector_expression(
                            rest, metadata, registry, parameters,
                        )?;
                        let (y, rest) = parse_continuous_vector_expression(
                            rest, metadata, registry, parameters,
                        )?;
                        let rest = util::parse_closing(rest)?;
                        if let Some(expression) = parse_vector_operation(name, x, y) {
                            Ok((expression, rest))
                        } else {
                            Err(ParseErr::new(format!(
                                "no such table or operation `{}`",
                                name
                            )))
                        }
                    }
                }
            }
        }
        _ => Err(ParseErr::new(format!("unexpected  token `{}`", token))),
    }
}

fn parse_continuous_vector_constant<'a, T: Numeric>(
    tokens: &'a [String],
    registry: &TableRegistry,
) -> Result<(NumericVectorExpression<T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let mut result = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((NumericVectorExpression::Constant(result), rest));
        }
        let v = if let Some(v) = registry.integer_tables.name_to_constant.get(next_token) {
            T::from(*v)
        } else if let Some(v) = registry.continuous_tables.name_to_constant.get(next_token) {
            T::from(*v)
        } else {
            let v: T = next_token.parse().map_err(|e| {
                ParseErr::new(format!(
                    "could not parse {} as a number: {:?}",
                    next_token, e
                ))
            })?;
            v
        };
        result.push(v);
        xs = rest;
    }
}

type NumericVectorTableReturnType<'a, T> = Option<(NumericVectorExpression<T>, &'a [String])>;

fn parse_vector_table<'a, T: Numeric, U: Numeric>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
    tables: &TableData<U>,
) -> Result<NumericVectorTableReturnType<'a, T>, ParseErr> {
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) =
            element_parser::parse_vector_expression(tokens, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((NumericVectorExpression::Table1D(*i, x), rest)))
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) = parse_vector_or_element(tokens, metadata, registry, parameters)?;
        let (y, rest) = parse_vector_or_element(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        match (x, y) {
            (VectorOrElementExpression::Vector(x), VectorOrElementExpression::Vector(y)) => {
                Ok(Some((NumericVectorExpression::Table2D(*i, x, y), rest)))
            }
            (VectorOrElementExpression::Vector(x), VectorOrElementExpression::Element(y)) => {
                Ok(Some((NumericVectorExpression::Table2DX(*i, x, y), rest)))
            }
            (VectorOrElementExpression::Element(x), VectorOrElementExpression::Vector(y)) => {
                Ok(Some((NumericVectorExpression::Table2DY(*i, x, y), rest)))
            }
            (x, y) => Err(ParseErr::new(format!(
                "arguments `{:?}` `{:?}` are invalid for `{}`",
                x, y, name
            ))),
        }
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (expression, rest) = parse_table(*i, tokens, metadata, registry, parameters)?;
        Ok(Some((expression, rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let (expression, rest) = parse_table(*i, tokens, metadata, registry, parameters)?;
        Ok(Some((expression, rest)))
    } else {
        Ok(None)
    }
}

fn parse_table<'a, T: Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(NumericVectorExpression<T>, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((NumericVectorExpression::Table(i, args), rest));
        }
        let (expression, new_xs) = parse_vector_or_element(xs, metadata, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

fn parse_vector_or_element<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(VectorOrElementExpression, &'a [String]), ParseErr> {
    if let Ok((element, rest)) =
        element_parser::parse_expression(tokens, metadata, registry, parameters)
    {
        Ok((VectorOrElementExpression::Element(element), rest))
    } else if let Ok((vector, rest)) =
        element_parser::parse_vector_expression(tokens, metadata, registry, parameters)
    {
        Ok((VectorOrElementExpression::Vector(vector), rest))
    } else {
        Err(ParseErr::new(format!(
            "could not parse tokens `{:?}`",
            tokens
        )))
    }
}

fn parse_vector_table_sum<'a, T: Numeric, U: Numeric>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
    tables: &TableData<U>,
) -> Result<NumericVectorTableReturnType<'a, T>, ParseErr> {
    if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) =
            numeric_table_parser::parse_argument(tokens, metadata, registry, parameters)?;
        let (y, rest) = numeric_table_parser::parse_argument(rest, metadata, registry, parameters)?;
        match (x, y) {
            (ArgumentExpression::Vector(x), ArgumentExpression::Set(y)) => {
                Ok(Some((NumericVectorExpression::Table2DXSum(*i, x, y), rest)))
            }
            (ArgumentExpression::Set(x), ArgumentExpression::Vector(y)) => {
                Ok(Some((NumericVectorExpression::Table2DYSum(*i, x, y), rest)))
            }
            (x, y) => Err(ParseErr::new(format!(
                "arguments `{:?}` `{:?}` are invalid for `{}`",
                x, y, name
            ))),
        }
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (expression, rest) = parse_table_sum(*i, tokens, metadata, registry, parameters)?;
        Ok(Some((expression, rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let (expression, rest) = parse_table_sum(*i, tokens, metadata, registry, parameters)?;
        Ok(Some((expression, rest)))
    } else {
        Ok(None)
    }
}

fn parse_table_sum<'a, T: Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(NumericVectorExpression<T>, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((NumericVectorExpression::TableSum(i, args), rest));
        }
        let (expression, new_xs) =
            numeric_table_parser::parse_argument(xs, metadata, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

fn parse_vector_numeric_operation<T: Numeric>(
    op: &str,
    v: NumericExpression<T>,
    vector: NumericVectorExpression<T>,
) -> Option<NumericVectorExpression<T>> {
    match op {
        "+" => Some(NumericVectorExpression::NumericOperation(
            NumericOperator::Add,
            v,
            Box::new(vector),
        )),
        "-" => Some(NumericVectorExpression::NumericOperation(
            NumericOperator::Subtract,
            v,
            Box::new(vector),
        )),
        "*" => Some(NumericVectorExpression::NumericOperation(
            NumericOperator::Multiply,
            v,
            Box::new(vector),
        )),
        "/" => Some(NumericVectorExpression::NumericOperation(
            NumericOperator::Divide,
            v,
            Box::new(vector),
        )),
        "max" => Some(NumericVectorExpression::NumericOperation(
            NumericOperator::Max,
            v,
            Box::new(vector),
        )),
        "min" => Some(NumericVectorExpression::NumericOperation(
            NumericOperator::Min,
            v,
            Box::new(vector),
        )),
        _ => None,
    }
}

fn parse_vector_operation<T: Numeric>(
    op: &str,
    x: NumericVectorExpression<T>,
    y: NumericVectorExpression<T>,
) -> Option<NumericVectorExpression<T>> {
    match op {
        "+" => Some(NumericVectorExpression::VectorOperation(
            NumericOperator::Add,
            Box::new(x),
            Box::new(y),
        )),
        "-" => Some(NumericVectorExpression::VectorOperation(
            NumericOperator::Subtract,
            Box::new(x),
            Box::new(y),
        )),
        "*" => Some(NumericVectorExpression::VectorOperation(
            NumericOperator::Multiply,
            Box::new(x),
            Box::new(y),
        )),
        "/" => Some(NumericVectorExpression::VectorOperation(
            NumericOperator::Divide,
            Box::new(x),
            Box::new(y),
        )),
        "max" => Some(NumericVectorExpression::VectorOperation(
            NumericOperator::Max,
            Box::new(x),
            Box::new(y),
        )),
        "min" => Some(NumericVectorExpression::VectorOperation(
            NumericOperator::Min,
            Box::new(x),
            Box::new(y),
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::table;
    use crate::table_data;
    use crate::variable::Continuous;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec![String::from("something")];
        let object_numbers = vec![10];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("something"), 0);

        let set_variable_names = vec![
            String::from("s0"),
            String::from("s1"),
            String::from("s2"),
            String::from("s3"),
        ];
        let mut name_to_set_variable = FxHashMap::default();
        name_to_set_variable.insert(String::from("s0"), 0);
        name_to_set_variable.insert(String::from("s1"), 1);
        name_to_set_variable.insert(String::from("s2"), 2);
        name_to_set_variable.insert(String::from("s3"), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let vector_variable_names = vec![
            String::from("v0"),
            String::from("v1"),
            String::from("v2"),
            String::from("v3"),
        ];
        let mut name_to_vector_variable = FxHashMap::default();
        name_to_vector_variable.insert(String::from("v0"), 0);
        name_to_vector_variable.insert(String::from("v1"), 1);
        name_to_vector_variable.insert(String::from("v2"), 2);
        name_to_vector_variable.insert(String::from("v3"), 3);
        let vector_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            String::from("e0"),
            String::from("e1"),
            String::from("e2"),
            String::from("e3"),
        ];
        let mut name_to_element_variable = FxHashMap::default();
        name_to_element_variable.insert(String::from("e0"), 0);
        name_to_element_variable.insert(String::from("e1"), 1);
        name_to_element_variable.insert(String::from("e2"), 2);
        name_to_element_variable.insert(String::from("e3"), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let integer_variable_names = vec![
            String::from("i0"),
            String::from("i1"),
            String::from("i2"),
            String::from("i3"),
        ];
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("i0"), 0);
        name_to_integer_variable.insert(String::from("i1"), 1);
        name_to_integer_variable.insert(String::from("i2"), 2);
        name_to_integer_variable.insert(String::from("i3"), 3);

        let integer_resource_variable_names = vec![
            String::from("ir0"),
            String::from("ir1"),
            String::from("ir2"),
            String::from("ir3"),
        ];
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert(String::from("ir0"), 0);
        name_to_integer_resource_variable.insert(String::from("ir1"), 1);
        name_to_integer_resource_variable.insert(String::from("ir2"), 2);
        name_to_integer_resource_variable.insert(String::from("ir3"), 3);

        let continuous_variable_names = vec![
            String::from("c0"),
            String::from("c1"),
            String::from("c2"),
            String::from("c3"),
        ];
        let mut name_to_continuous_variable = FxHashMap::default();
        name_to_continuous_variable.insert(String::from("c0"), 0);
        name_to_continuous_variable.insert(String::from("c1"), 1);
        name_to_continuous_variable.insert(String::from("c2"), 2);
        name_to_continuous_variable.insert(String::from("c3"), 3);

        let continuous_resource_variable_names = vec![
            String::from("cr0"),
            String::from("cr1"),
            String::from("cr2"),
            String::from("cr3"),
        ];
        let mut name_to_continuous_resource_variable = FxHashMap::default();
        name_to_continuous_resource_variable.insert(String::from("cr0"), 0);
        name_to_continuous_resource_variable.insert(String::from("cr1"), 1);
        name_to_continuous_resource_variable.insert(String::from("cr2"), 2);
        name_to_continuous_resource_variable.insert(String::from("cr3"), 3);

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
            continuous_variable_names,
            name_to_continuous_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true, false],
            continuous_resource_variable_names,
            name_to_continuous_resource_variable,
            continuous_less_is_better: vec![false, false, true, false],
        }
    }

    fn generate_parameters() -> FxHashMap<String, Element> {
        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("param"), 0);
        parameters
    }

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![table::Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let tables = vec![table::Table::new(FxHashMap::default(), 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        let integer_tables = table_data::TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("cf0"), 0.0);

        let tables_1d = vec![table::Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("cf1"), 0);

        let tables_2d = vec![table::Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("cf2"), 0);

        let tables_3d = vec![table::Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("cf3"), 0);

        let tables = vec![table::Table::new(FxHashMap::default(), 0.0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("cf4"), 0);

        let continuous_tables = table_data::TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        TableRegistry {
            integer_tables,
            continuous_tables,
            ..Default::default()
        }
    }

    #[test]
    fn parse_expression_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens = Vec::new();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "(", "+", "cost", "1", ")", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_expression_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens = Vec::new();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "cost", "n0", ")", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "cf1", "e0", ")", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "sum", "f4", "0", "e0", "s0", "p0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::IntegerTable(NumericTableExpression::TableSum(
                0,
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Variable(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Variable(0)
                    ))
                ]
            ))
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = ["(", "sum", "cf4", "0", "e0", "s0", "p0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        println!("{:?}", result);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::ContinuousTable(NumericTableExpression::TableSum(
                0,
                vec![
                    ArgumentExpression::Element(ElementExpression::Constant(0)),
                    ArgumentExpression::Element(ElementExpression::Variable(0)),
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Variable(0)
                    ))
                ]
            ))
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "f4", "0", "e0", "s0", "p0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Constant(0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "-", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Subtract,
                Box::new(NumericExpression::Constant(0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "*", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(NumericExpression::Constant(0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "/", "0.0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Divide,
                Box::new(NumericExpression::Constant(0.0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "min", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Min,
                Box::new(NumericExpression::Constant(0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "max", "0", "c0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Max,
                Box::new(NumericExpression::Constant(0)),
                Box::new(NumericExpression::ContinuousVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens = Vec::new();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", "i0", "i1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "^", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_cardinality_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "s2", "|", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Cardinality(SetExpression::Reference(
                ReferenceExpression::Variable(2)
            ))
        );
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_cardinality_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "e2", "|", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["|", "s2", "s0", "|", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["cost", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Cost);
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["f0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Constant(0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["cf0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Constant(0.0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["i1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::IntegerVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["c1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::ContinuousVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["ir1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::IntegerResourceVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["cr1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::ContinuousResourceVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Constant(11));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_atom_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["h", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["e1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
