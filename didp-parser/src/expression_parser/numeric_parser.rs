use super::element_parser;
use super::numeric_table_parser;
use super::table_vector_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{
    NumericExpression, NumericOperator, NumericVectorExpression, ReduceOperator,
};
use crate::state::StateMetadata;
use crate::table_registry::TableRegistry;
use crate::variable::{Continuous, Element, Integer, Numeric};
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
                Ok((NumericExpression::Last(Box::new(vector)), rest))
            } else if name == "at" {
                let (vector, rest) =
                    parse_integer_vector_expression(rest, metadata, registry, parameters)?;
                let (i, rest) =
                    element_parser::parse_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((NumericExpression::At(Box::new(vector), i), rest))
            } else if let Ok((vector, rest)) =
                parse_integer_vector_expression(rest, metadata, registry, parameters)
            {
                let rest = util::parse_closing(rest)?;
                Ok((parse_reduce(name, vector)?, rest))
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
                let (vector, rest) =
                    parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((NumericExpression::Last(Box::new(vector)), rest))
            } else if name == "at" {
                let (vector, rest) =
                    parse_continuous_vector_expression(rest, metadata, registry, parameters)?;
                let (i, rest) =
                    element_parser::parse_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((NumericExpression::At(Box::new(vector), i), rest))
            } else if let Ok((vector, rest)) =
                parse_continuous_vector_expression(rest, metadata, registry, parameters)
            {
                let rest = util::parse_closing(rest)?;
                Ok((parse_reduce(name, vector)?, rest))
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

fn parse_reduce<T: Numeric>(
    name: &str,
    vector: NumericVectorExpression<T>,
) -> Result<NumericExpression<T>, ParseErr> {
    match name {
        "reduce-sum" => Ok(NumericExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(vector),
        )),
        "reduce-product" => Ok(NumericExpression::Reduce(
            ReduceOperator::Product,
            Box::new(vector),
        )),
        "reduce-max" => Ok(NumericExpression::Reduce(
            ReduceOperator::Max,
            Box::new(vector),
        )),
        "reduce-min" => Ok(NumericExpression::Reduce(
            ReduceOperator::Min,
            Box::new(vector),
        )),
        _ => Err(ParseErr::new(format!(
            "no such reduction operator `{}`",
            name
        ))),
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
    match name {
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
    if let Some(v) = registry.integer_tables.name_to_constant.get(token) {
        Ok(NumericExpression::Constant(T::from(*v)))
    } else if let Some(i) = metadata.name_to_integer_variable.get(token) {
        Ok(NumericExpression::IntegerVariable(*i))
    } else if let Some(i) = metadata.name_to_integer_resource_variable.get(token) {
        Ok(NumericExpression::IntegerResourceVariable(*i))
    } else if token == "cost" {
        Ok(NumericExpression::Cost)
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
        let n: Continuous = token.parse().map_err(|e| {
            ParseErr::new(format!("could not parse {} as a number: {:?}", token, e))
        })?;
        Ok(NumericExpression::Constant(T::from(n)))
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
                "numeric-vector" => parse_integer_vector_constant(rest, registry),
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
                    if let Some((expression, rest)) = table_vector_parser::parse_expression(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((NumericVectorExpression::IntegerTable(expression), rest))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
                "table-sum-vector" => {
                    let (name, rest) = rest
                        .split_first()
                        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
                    if let Some((expression, rest)) = table_vector_parser::parse_sum_expression(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((NumericVectorExpression::IntegerTable(expression), rest))
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
            let v: Integer = next_token.parse().map_err(|e| {
                ParseErr::new(format!(
                    "could not parse {} as a number: {:?}",
                    next_token, e
                ))
            })?;
            T::from(v)
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
                "numeric-vector" => parse_continuous_vector_constant(rest, registry),
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
                    if let Some((expression, rest)) = table_vector_parser::parse_expression(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((NumericVectorExpression::IntegerTable(expression), rest))
                    } else if let Some((expression, rest)) = table_vector_parser::parse_expression(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.continuous_tables,
                    )? {
                        Ok((NumericVectorExpression::ContinuousTable(expression), rest))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
                "table-sum-vector" => {
                    let (name, rest) = rest
                        .split_first()
                        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
                    if let Some((expression, rest)) = table_vector_parser::parse_sum_expression(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((NumericVectorExpression::IntegerTable(expression), rest))
                    } else if let Some((expression, rest)) =
                        table_vector_parser::parse_sum_expression(
                            name,
                            rest,
                            metadata,
                            registry,
                            parameters,
                            &registry.continuous_tables,
                        )?
                    {
                        Ok((NumericVectorExpression::ContinuousTable(expression), rest))
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
            let v: Continuous = next_token.parse().map_err(|e| {
                ParseErr::new(format!(
                    "could not parse {} as a number: {:?}",
                    next_token, e
                ))
            })?;
            T::from(v)
        };
        result.push(v);
        xs = rest;
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
    fn parse_expression_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "f1", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::IntegerTable(NumericTableExpression::Table1D(
                0,
                ElementExpression::Constant(1)
            ))
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "+", "(", "cf1", "1", ")", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::ContinuousTable(
                    NumericTableExpression::Table1D(0, ElementExpression::Constant(1))
                )),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_integer_atom_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["cost", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Cost);
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["f0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Constant(0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["i1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::IntegerVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["ir1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::IntegerResourceVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Constant(11));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_integer_atom_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["c0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["cr0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["cf0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["1.2", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_atom_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["cost", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Cost);
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["f0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Constant(0.0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["cf0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Constant(0.0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["i1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::IntegerVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["c1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::ContinuousVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["ir1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
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
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Constant(11.0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["1.5", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericExpression::Constant(1.5));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_continuous_atom_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result =
            parse_continuous_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "sum", "f4", "0", "e0", "s0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
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
    }

    #[test]
    fn parse_integer_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "sum", "cf4", "0", "e0", "s0", "v0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "f4", "0", "e0", "s0", "v0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "sum", "f4", "0", "e0", "s0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
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

        let tokens: Vec<String> = ["(", "sum", "cf4", "0", "e0", "s0", "v0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
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
    fn parse_continuous_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "f4", "0", "e0", "s0", "v0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_integer_expression(&tokens, &metadata, &registry, &parameters);
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
        let result = parse_integer_expression(&tokens, &metadata, &registry, &parameters);
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
        let result = parse_integer_expression(&tokens, &metadata, &registry, &parameters);
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

        let tokens: Vec<String> = ["(", "/", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_integer_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Divide,
                Box::new(NumericExpression::Constant(0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "min", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_integer_expression(&tokens, &metadata, &registry, &parameters);
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

        let tokens: Vec<String> = ["(", "max", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_integer_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Max,
                Box::new(NumericExpression::Constant(0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_integer_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens = Vec::new();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", "i0", "i1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "^", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_operatin_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "0.0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Add,
                Box::new(NumericExpression::Constant(0.0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "-", "0.0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Subtract,
                Box::new(NumericExpression::Constant(0.0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "*", "0.0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Multiply,
                Box::new(NumericExpression::Constant(0.0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "/", "0.0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_expression(&tokens, &metadata, &registry, &parameters);
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

        let tokens: Vec<String> = ["(", "min", "0.0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Min,
                Box::new(NumericExpression::Constant(0.0)),
                Box::new(NumericExpression::IntegerVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "max", "0", "c0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::NumericOperation(
                NumericOperator::Max,
                Box::new(NumericExpression::Constant(0.0)),
                Box::new(NumericExpression::ContinuousVariable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_continuous_operatin_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens = Vec::new();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", "i0", "i1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "^", "0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_cardinality_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "s2", "|", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
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
    fn parse_integer_cardinality_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "e2", "|", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["|", "s2", "s0", "|", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_cardinality_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "s2", "|", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
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
    fn parse_continuous_cardinality_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "e2", "|", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["|", "s2", "s0", "|", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_length_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "length", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Length(VectorExpression::Reference(ReferenceExpression::Variable(
                0
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_integer_length_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "length", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_length_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "length", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Length(VectorExpression::Reference(ReferenceExpression::Variable(
                0
            )))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_continuous_length_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "length", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_last_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "last", "(", "numeric-vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Last(Box::new(NumericVectorExpression::Constant(vec![0, 1])))
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_integer_last_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "last", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_last_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = [
            "(",
            "last",
            "(",
            "numeric-vector",
            "0.0",
            "1.0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Last(Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0])))
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_continuous_last_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "last", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_at() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = [
            "(",
            "at",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "0",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::At(
                Box::new(NumericVectorExpression::Constant(vec![0, 1]),),
                ElementExpression::Constant(0),
            )
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_integer_at_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "at", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "at", "(", "numeric-vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "at", "(", "vector", "0", "1", ")", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_at() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = [
            "(",
            "at",
            "(",
            "numeric-vector",
            "0.0",
            "1.0",
            ")",
            "0",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::At(
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]),),
                ElementExpression::Constant(0),
            )
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_continuous_at_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "at", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "at", "(", "numeric-vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "at", "(", "vector", "0", "1", ")", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_reduce_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "reduce-sum",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Reduce(
                ReduceOperator::Sum,
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = [
            "(",
            "reduce-product",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Reduce(
                ReduceOperator::Product,
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = [
            "(",
            "reduce-max",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Reduce(
                ReduceOperator::Max,
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = [
            "(",
            "reduce-min",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Reduce(
                ReduceOperator::Min,
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_integer_reduce_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reduce-sum", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "reduce-product",
            "(",
            "vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "reduce-max", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "reduce-min", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "reduce-null",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_reduce_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "reduce-sum",
            "(",
            "numeric-vector",
            "0.0",
            "1.0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Reduce(
                ReduceOperator::Sum,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = [
            "(",
            "reduce-product",
            "(",
            "numeric-vector",
            "0.0",
            "1.0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Reduce(
                ReduceOperator::Product,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = [
            "(",
            "reduce-max",
            "(",
            "numeric-vector",
            "0.0",
            "1.0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Reduce(
                ReduceOperator::Max,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = [
            "(",
            "reduce-min",
            "(",
            "numeric-vector",
            "0.0",
            "1.0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericExpression::Reduce(
                ReduceOperator::Min,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_continuous_reduce_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reduce-sum", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "reduce-product",
            "(",
            "vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "reduce-max", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "reduce-min", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "reduce-null",
            "(",
            "numeric-vector",
            "0.0",
            "1.0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_continuous_expression::<Continuous>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_constant_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "numeric-vector", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericVectorExpression::Constant(vec![]));
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["(", "numeric-vector", "0", "1", "f0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericVectorExpression::Constant(vec![0, 1, 0]));
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_integer_vector_constant_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "numeric-vector", "1"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "numeric-vector", "0.0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "numeric-vector", "cf0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_constant_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "numeric-vector", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, NumericVectorExpression::Constant(vec![]));
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = [
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            "0.1",
            "cf0",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::Constant(vec![0.0, 1.0, 0.0, 0.1, 0.0])
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_continuous_vector_constant_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "numeric-vector", "1"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_reverse_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "reverse",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Constant(vec![
                0, 1, 0
            ])))
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_integer_vector_reverse_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reverse", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_reverse_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "reverse",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::Reverse(Box::new(NumericVectorExpression::Constant(vec![
                0.0, 1.0, 0.0
            ])))
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_continuous_vector_reverse_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reverse", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_push_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "push",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::Push(
                NumericExpression::Constant(1),
                Box::new(NumericVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);
    }

    #[test]
    fn parse_integer_vector_push_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "push",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "push", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "push", "1", "(", "vector", "0", "1", "f0", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "push",
            "1.0",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "push",
            "1",
            "(",
            "numeric-vector",
            "0.0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_push_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "push",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::Push(
                NumericExpression::Constant(1.0),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);
    }

    #[test]
    fn parse_continuous_vector_push_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "push",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "push", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "push", "1", "(", "vector", "0", "1", "f0", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_pop_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "pop",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::Pop(Box::new(NumericVectorExpression::Constant(vec![
                0, 1, 0
            ])))
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_integer_vector_pop_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "pop", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_pop_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "pop",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::Pop(Box::new(NumericVectorExpression::Constant(vec![
                0.0, 1.0, 0.0
            ])))
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_continuous_vector_pop_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "pop", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_set_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "set",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::Set(
                NumericExpression::Constant(1),
                Box::new(NumericVectorExpression::Constant(vec![0, 1, 0])),
                ElementExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_integer_vector_set_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "set",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "set", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "set", "1", "(", "vector", "0", "1", "f0", ")", "1", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "set",
            "1.0",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "set",
            "1",
            "(",
            "numeric-vector",
            "0.0",
            "1",
            "f0",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_set_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "set",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::Set(
                NumericExpression::Constant(1.0),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ElementExpression::Constant(1)
            )
        );
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_continuous_vector_set_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "set",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "set", "1", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "set", "1", "(", "vector", "0", "1", "f0", ")", "1", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_numeric_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Add,
                NumericExpression::Constant(1),
                Box::new(NumericVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Subtract,
                NumericExpression::Constant(1),
                Box::new(NumericVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Multiply,
                NumericExpression::Constant(1),
                Box::new(NumericVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Divide,
                NumericExpression::Constant(1),
                Box::new(NumericVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Max,
                NumericExpression::Constant(1),
                Box::new(NumericVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Min,
                NumericExpression::Constant(1),
                Box::new(NumericVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);
    }

    #[test]
    fn parse_integer_vector_numeric_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "-", "1", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "+",
            "1.0",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "+",
            "1",
            "(",
            "numeric-vector",
            "0.0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_numeric_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Add,
                NumericExpression::Constant(1.0),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Subtract,
                NumericExpression::Constant(1.0),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Multiply,
                NumericExpression::Constant(1.0),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Divide,
                NumericExpression::Constant(1.0),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Max,
                NumericExpression::Constant(1.0),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "1",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::NumericOperation(
                NumericOperator::Min,
                NumericExpression::Constant(1.0),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);
    }

    #[test]
    fn parse_continuous_vector_numeric_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "numeric-vector",
            "0",
            "1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "-", "1", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Add,
                Box::new(NumericVectorExpression::Constant(vec![0, 1])),
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Subtract,
                Box::new(NumericVectorExpression::Constant(vec![0, 1])),
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Multiply,
                Box::new(NumericVectorExpression::Constant(vec![0, 1])),
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Divide,
                Box::new(NumericVectorExpression::Constant(vec![0, 1])),
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Max,
                Box::new(NumericVectorExpression::Constant(vec![0, 1])),
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_integer_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Min,
                Box::new(NumericVectorExpression::Constant(vec![0, 1])),
                Box::new(NumericVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);
    }

    #[test]
    fn parse_integer_vector_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "numeric-vector",
            "1.0",
            "0",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "numeric-vector",
            "1",
            "0",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1.0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Add,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Subtract,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Multiply,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Divide,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Max,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            "(",
            "numeric-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::VectorOperation(
                NumericOperator::Min,
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(NumericVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);
    }

    #[test]
    fn parse_integer_vector_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "table-vector",
            "f3",
            "0",
            "(",
            "vector",
            "0",
            "1",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::IntegerTable(TableVectorExpression::Table(
                0,
                vec![
                    VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                    VectorOrElementExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                ]
            ))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-sum-vector",
            "f3",
            "s0",
            "(",
            "vector",
            "0",
            "1",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::IntegerTable(TableVectorExpression::TableSum(
                0,
                vec![
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(1)),
                ]
            ))
        );
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_integer_vector_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "table-vector", "cf1", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-vector", "cf2", "v0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-vector", "cf2", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-vector", "cf2", "0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-sum-vector", "cf2", "v0", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-sum-vector", "cf2", "s0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-vector", "cf2", "0", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-sum-vector", "cf2", "s0", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result =
            parse_integer_vector_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "table-vector",
            "f3",
            "0",
            "(",
            "vector",
            "0",
            "1",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::IntegerTable(TableVectorExpression::Table(
                0,
                vec![
                    VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                    VectorOrElementExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                ]
            ))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-sum-vector",
            "f3",
            "s0",
            "(",
            "vector",
            "0",
            "1",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::IntegerTable(TableVectorExpression::TableSum(
                0,
                vec![
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(1)),
                ]
            ))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-vector",
            "cf3",
            "0",
            "(",
            "vector",
            "0",
            "1",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::ContinuousTable(TableVectorExpression::Table(
                0,
                vec![
                    VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                    VectorOrElementExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                ]
            ))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-sum-vector",
            "cf3",
            "s0",
            "(",
            "vector",
            "0",
            "1",
            ")",
            "1",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_continuous_vector_expression::<Continuous>(
            &tokens,
            &metadata,
            &registry,
            &parameters,
        );
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericVectorExpression::ContinuousTable(TableVectorExpression::TableSum(
                0,
                vec![
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(1)),
                ]
            ))
        );
        assert_eq!(rest, &tokens[11..]);
    }
}
