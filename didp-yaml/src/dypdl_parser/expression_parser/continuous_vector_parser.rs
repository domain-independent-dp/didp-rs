use super::condition_parser;
use super::continuous_parser;
use super::element_parser;
use super::table_vector_parser;
use super::util;
use super::util::ParseErr;
use dypdl::expression::{
    BinaryOperator, CastOperator, ContinuousBinaryOperator, ContinuousExpression,
    ContinuousUnaryOperator, ContinuousVectorExpression, IntegerVectorExpression, ReduceOperator,
    UnaryOperator,
};
use dypdl::variable_type::{Continuous, Element};
use dypdl::{StateMetadata, TableRegistry};
use rustc_hash::FxHashMap;
use std::str;

pub fn parse_expression<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(ContinuousVectorExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            match &name[..] {
                "continuous-vector" => parse_continuous_vector_constant(rest, registry),
                "reverse" => {
                    let (vector, rest) = parse_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((ContinuousVectorExpression::Reverse(Box::new(vector)), rest))
                }
                "push" => {
                    let (v, rest) =
                        continuous_parser::parse_expression(rest, metadata, registry, parameters)?;
                    let (vector, rest) = parse_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((ContinuousVectorExpression::Push(v, Box::new(vector)), rest))
                }
                "pop" => {
                    let (vector, rest) = parse_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((ContinuousVectorExpression::Pop(Box::new(vector)), rest))
                }
                "set" => {
                    let (v, rest) =
                        continuous_parser::parse_expression(rest, metadata, registry, parameters)?;
                    let (vector, rest) = parse_expression(rest, metadata, registry, parameters)?;
                    let (i, rest) =
                        element_parser::parse_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((
                        ContinuousVectorExpression::Set(v, Box::new(vector), i),
                        rest,
                    ))
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
                        &registry.continuous_tables,
                    )? {
                        Ok((
                            ContinuousVectorExpression::Table(Box::new(expression)),
                            rest,
                        ))
                    } else if let Some((expression, rest)) = table_vector_parser::parse_expression(
                        name,
                        rest,
                        metadata,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((
                            ContinuousVectorExpression::FromInteger(Box::new(
                                IntegerVectorExpression::Table(Box::new(expression)),
                            )),
                            rest,
                        ))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
                "if" => {
                    let (condition, rest) =
                        condition_parser::parse_expression(rest, metadata, registry, parameters)?;
                    let (x, rest) = parse_expression(rest, metadata, registry, parameters)?;
                    let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((
                        ContinuousVectorExpression::If(
                            Box::new(condition),
                            Box::new(x),
                            Box::new(y),
                        ),
                        rest,
                    ))
                }
                name => {
                    let op = match name {
                        "table-sum-vector" => ReduceOperator::Sum,
                        "table-product-vector" => ReduceOperator::Product,
                        "table-max-vector" => ReduceOperator::Max,
                        "table-min-vector" => ReduceOperator::Min,
                        _ => {
                            return if let Ok(result) =
                                parse_vector_round(name, rest, metadata, registry, parameters)
                            {
                                Ok(result)
                            } else if let Ok(result) = parse_vector_unary_operation(
                                name, rest, metadata, registry, parameters,
                            ) {
                                Ok(result)
                            } else if let Ok((x, y, rest)) =
                                parse_continuous_and_vector(rest, metadata, registry, parameters)
                            {
                                Ok((parse_vector_binary_operation_x(name, x, y)?, rest))
                            } else if let Ok((x, y, rest)) =
                                parse_vector_and_continuous(rest, metadata, registry, parameters)
                            {
                                Ok((parse_vector_binary_operation_y(name, x, y)?, rest))
                            } else if let Ok((x, y, rest)) =
                                parse_vector_and_vector(rest, metadata, registry, parameters)
                            {
                                Ok((parse_vector_operation(name, x, y)?, rest))
                            } else {
                                Err(ParseErr::new(format!("could not parse `{:?}`", tokens)))
                            }
                        }
                    };
                    let (name, rest) = rest
                        .split_first()
                        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
                    if let Some((expression, rest)) = table_vector_parser::parse_reduce_expression(
                        name,
                        rest,
                        op.clone(),
                        metadata,
                        registry,
                        parameters,
                        &registry.continuous_tables,
                    )? {
                        Ok((
                            ContinuousVectorExpression::Table(Box::new(expression)),
                            rest,
                        ))
                    } else if let Some((expression, rest)) =
                        table_vector_parser::parse_reduce_expression(
                            name,
                            rest,
                            op,
                            metadata,
                            registry,
                            parameters,
                            &registry.integer_tables,
                        )?
                    {
                        Ok((
                            ContinuousVectorExpression::FromInteger(Box::new(
                                IntegerVectorExpression::Table(Box::new(expression)),
                            )),
                            rest,
                        ))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
            }
        }
        _ => Err(ParseErr::new(format!("unexpected  token `{}`", token))),
    }
}

fn parse_continuous_vector_constant<'a>(
    tokens: &'a [String],
    registry: &TableRegistry,
) -> Result<(ContinuousVectorExpression, &'a [String]), ParseErr> {
    let mut result = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((ContinuousVectorExpression::Constant(result), rest));
        }
        let v = if let Some(v) = registry.continuous_tables.name_to_constant.get(next_token) {
            *v
        } else if let Some(v) = registry.integer_tables.name_to_constant.get(next_token) {
            *v as Continuous
        } else {
            let v: Continuous = next_token.parse().map_err(|e| {
                ParseErr::new(format!(
                    "could not parse {} as a continuous atom: {:?}",
                    next_token, e
                ))
            })?;
            v
        };
        result.push(v);
        xs = rest;
    }
}

fn parse_vector_unary_operation<'a>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(ContinuousVectorExpression, &'a [String]), ParseErr> {
    let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    match name {
        "abs" => Ok((
            ContinuousVectorExpression::UnaryOperation(UnaryOperator::Abs, Box::new(x)),
            rest,
        )),
        "sqrt" => Ok((
            ContinuousVectorExpression::ContinuousUnaryOperation(
                ContinuousUnaryOperator::Sqrt,
                Box::new(x),
            ),
            rest,
        )),
        _ => Err(ParseErr::new(format!("no such unary operator `{}`", name))),
    }
}

fn parse_vector_round<'a>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(ContinuousVectorExpression, &'a [String]), ParseErr> {
    let op = match name {
        "ceil" => CastOperator::Ceil,
        "floor" => CastOperator::Floor,
        "round" => CastOperator::Round,
        _ => return Err(ParseErr::new(format!("no such unary operator `{}`", name))),
    };
    let (expression, rest) = parse_expression(tokens, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((
        ContinuousVectorExpression::Round(op, Box::new(expression)),
        rest,
    ))
}

fn parse_vector_and_continuous<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<
    (
        ContinuousVectorExpression,
        ContinuousExpression,
        &'a [String],
    ),
    ParseErr,
> {
    let (vector, rest) = parse_expression(tokens, metadata, registry, parameters)?;
    let (continuous, rest) =
        continuous_parser::parse_expression(rest, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((vector, continuous, rest))
}

fn parse_continuous_and_vector<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<
    (
        ContinuousExpression,
        ContinuousVectorExpression,
        &'a [String],
    ),
    ParseErr,
> {
    let (continuous, rest) =
        continuous_parser::parse_expression(tokens, metadata, registry, parameters)?;
    let (vector, rest) = parse_expression(rest, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((continuous, vector, rest))
}

fn parse_vector_and_vector<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<
    (
        ContinuousVectorExpression,
        ContinuousVectorExpression,
        &'a [String],
    ),
    ParseErr,
> {
    let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
    let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((x, y, rest))
}

fn parse_vector_binary_operation_x(
    op: &str,
    v: ContinuousExpression,
    vector: ContinuousVectorExpression,
) -> Result<ContinuousVectorExpression, ParseErr> {
    match op {
        "+" => Ok(ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            v,
            Box::new(vector),
        )),
        "-" => Ok(ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Sub,
            v,
            Box::new(vector),
        )),
        "*" => Ok(ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Mul,
            v,
            Box::new(vector),
        )),
        "/" => Ok(ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Div,
            v,
            Box::new(vector),
        )),
        "%" => Ok(ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Rem,
            v,
            Box::new(vector),
        )),
        "max" => Ok(ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Max,
            v,
            Box::new(vector),
        )),
        "min" => Ok(ContinuousVectorExpression::BinaryOperationX(
            BinaryOperator::Min,
            v,
            Box::new(vector),
        )),
        "pow" => Ok(ContinuousVectorExpression::ContinuousBinaryOperationX(
            ContinuousBinaryOperator::Pow,
            v,
            Box::new(vector),
        )),
        "log" => Ok(ContinuousVectorExpression::ContinuousBinaryOperationX(
            ContinuousBinaryOperator::Log,
            v,
            Box::new(vector),
        )),
        _ => Err(ParseErr::new(format!("no such binary operator `{}`", op))),
    }
}

fn parse_vector_binary_operation_y(
    op: &str,
    vector: ContinuousVectorExpression,
    v: ContinuousExpression,
) -> Result<ContinuousVectorExpression, ParseErr> {
    match op {
        "+" => Ok(ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(vector),
            v,
        )),
        "-" => Ok(ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Sub,
            Box::new(vector),
            v,
        )),
        "*" => Ok(ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Mul,
            Box::new(vector),
            v,
        )),
        "/" => Ok(ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Div,
            Box::new(vector),
            v,
        )),
        "%" => Ok(ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Rem,
            Box::new(vector),
            v,
        )),
        "max" => Ok(ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Max,
            Box::new(vector),
            v,
        )),
        "min" => Ok(ContinuousVectorExpression::BinaryOperationY(
            BinaryOperator::Min,
            Box::new(vector),
            v,
        )),
        "pow" => Ok(ContinuousVectorExpression::ContinuousBinaryOperationY(
            ContinuousBinaryOperator::Pow,
            Box::new(vector),
            v,
        )),
        "log" => Ok(ContinuousVectorExpression::ContinuousBinaryOperationY(
            ContinuousBinaryOperator::Log,
            Box::new(vector),
            v,
        )),
        _ => Err(ParseErr::new(format!("no such binary operator `{}`", op))),
    }
}

fn parse_vector_operation(
    op: &str,
    x: ContinuousVectorExpression,
    y: ContinuousVectorExpression,
) -> Result<ContinuousVectorExpression, ParseErr> {
    match op {
        "+" => Ok(ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(x),
            Box::new(y),
        )),
        "-" => Ok(ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Sub,
            Box::new(x),
            Box::new(y),
        )),
        "*" => Ok(ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Mul,
            Box::new(x),
            Box::new(y),
        )),
        "/" => Ok(ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Div,
            Box::new(x),
            Box::new(y),
        )),
        "%" => Ok(ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Rem,
            Box::new(x),
            Box::new(y),
        )),
        "max" => Ok(ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Max,
            Box::new(x),
            Box::new(y),
        )),
        "min" => Ok(ContinuousVectorExpression::VectorOperation(
            BinaryOperator::Min,
            Box::new(x),
            Box::new(y),
        )),
        "pow" => Ok(ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(x),
            Box::new(y),
        )),
        "log" => Ok(ContinuousVectorExpression::ContinuousVectorOperation(
            ContinuousBinaryOperator::Log,
            Box::new(x),
            Box::new(y),
        )),
        _ => Err(ParseErr::new(format!("no such binary operator `{}`", op))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::*;

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

        let element_resource_variable_names = vec![
            String::from("er0"),
            String::from("er1"),
            String::from("er2"),
            String::from("er3"),
        ];
        let mut name_to_element_resource_variable = FxHashMap::default();
        name_to_element_resource_variable.insert(String::from("er0"), 0);
        name_to_element_resource_variable.insert(String::from("er1"), 1);
        name_to_element_resource_variable.insert(String::from("er2"), 2);
        name_to_element_resource_variable.insert(String::from("er3"), 3);
        let element_resource_variable_to_object = vec![0, 0, 0, 0];

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
            object_type_names: object_names,
            name_to_object_type: name_to_object,
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
            element_resource_variable_names,
            name_to_element_resource_variable,
            element_resource_variable_to_object,
            element_less_is_better: vec![false, false, true, false],
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

        let tables_1d = vec![Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        let integer_tables = dypdl::TableData {
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

        let tables_1d = vec![Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("cf1"), 0);

        let tables_2d = vec![Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("cf2"), 0);

        let tables_3d = vec![Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("cf3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), 0.0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("cf4"), 0);

        let continuous_tables = dypdl::TableData {
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
    fn parse_continuous_vector_constant_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "continuous-vector", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousVectorExpression::Constant(vec![]));
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = [
            "(",
            "continuous-vector",
            "0",
            "1",
            "f0",
            "cf0",
            "1.5",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0, 0.0, 1.5])
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_continuous_vector_constant_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "continuous-vector", "1"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "continuous-vector", "cf1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Reverse(Box::new(ContinuousVectorExpression::Constant(
                vec![0.0, 1.0, 0.0]
            )))
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Push(
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
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
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "push", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "push", "1", "(", "vector", "0", "1", "f0", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Pop(Box::new(ContinuousVectorExpression::Constant(vec![
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Set(
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ElementExpression::Constant(1),
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
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "set", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "set", "1", "(", "vector", "0", "1", "f0", ")", "1", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "set",
            "e0",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_unary_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "abs",
            "(",
            "continuous-vector",
            "0",
            "-1.5",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, -1.5, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[9..]);

        let tokens: Vec<String> = [
            "(",
            "sqrt",
            "(",
            "continuous-vector",
            "0",
            "4.5",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::ContinuousUnaryOperation(
                ContinuousUnaryOperator::Sqrt,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 4.5, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_continuous_vector_unary_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "exp",
            "(",
            "continuous-vector",
            "0",
            "-1",
            "f0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_binary_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "1",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationX(
                BinaryOperator::Add,
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "1",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationX(
                BinaryOperator::Sub,
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "1",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationX(
                BinaryOperator::Mul,
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "1",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationX(
                BinaryOperator::Div,
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "%",
            "1",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationX(
                BinaryOperator::Rem,
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "1",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationX(
                BinaryOperator::Max,
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "1",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationX(
                BinaryOperator::Min,
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "pow",
            "1",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::ContinuousBinaryOperationX(
                ContinuousBinaryOperator::Pow,
                ContinuousExpression::Constant(1.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "log",
            "2",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::ContinuousBinaryOperationX(
                ContinuousBinaryOperator::Log,
                ContinuousExpression::Constant(2.0),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationY(
                BinaryOperator::Add,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ContinuousExpression::Constant(1.0),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationY(
                BinaryOperator::Sub,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ContinuousExpression::Constant(1.0),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationY(
                BinaryOperator::Mul,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ContinuousExpression::Constant(1.0),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationY(
                BinaryOperator::Div,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ContinuousExpression::Constant(1.0),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "%",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationY(
                BinaryOperator::Rem,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ContinuousExpression::Constant(1.0),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationY(
                BinaryOperator::Max,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ContinuousExpression::Constant(1.0),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::BinaryOperationY(
                BinaryOperator::Min,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ContinuousExpression::Constant(1.0),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "pow",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::ContinuousBinaryOperationY(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0, 0.0])),
                ContinuousExpression::Constant(1.0),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "log",
            "(",
            "continuous-vector",
            "2",
            "1",
            "3",
            ")",
            "2",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::ContinuousBinaryOperationY(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousVectorExpression::Constant(vec![2.0, 1.0, 3.0])),
                ContinuousExpression::Constant(2.0),
            )
        );
        assert_eq!(rest, &tokens[10..]);
    }

    #[test]
    fn parse_continuous_vector_binary_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "-", "1", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_vector_if_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "if",
            "true",
            "(",
            "continuous-vector",
            "0",
            ")",
            "(",
            "continuous-vector",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![1.0]))
            )
        );
        assert_eq!(rest, &tokens[12..]);
    }

    #[test]
    fn parse_continuous_vector_if_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "if",
            "0",
            "(",
            "continuous-vector",
            "0",
            ")",
            "(",
            "continuous-vector",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "if",
            "true",
            "(",
            "continuous-vector",
            "0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "if",
            "true",
            "true",
            "(",
            "continuous-vector",
            "0",
            ")",
            "(",
            "continuous-vector",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
            "continuous-vector",
            "0",
            "1",
            ")",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::VectorOperation(
                BinaryOperator::Add,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::VectorOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::VectorOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::VectorOperation(
                BinaryOperator::Div,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "%",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::VectorOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::VectorOperation(
                BinaryOperator::Max,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::VectorOperation(
                BinaryOperator::Min,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "pow",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            "(",
            "continuous-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::ContinuousVectorOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "log",
            "(",
            "continuous-vector",
            "2",
            "3",
            ")",
            "(",
            "continuous-vector",
            "2",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::ContinuousVectorOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousVectorExpression::Constant(vec![2.0, 3.0])),
                Box::new(ContinuousVectorExpression::Constant(vec![2.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[13..]);
    }

    #[test]
    fn parse_continuous_vector_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "continuous-vector",
            "1.0",
            "0",
            ")",
            "(",
            "integer-vector",
            "0",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "integer-vector",
            "1",
            "0",
            ")",
            "(",
            "continuous-vector",
            "0",
            "1.0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Table3D(
                0,
                VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                VectorOrElementExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1])
                )),
                VectorOrElementExpression::Element(ElementExpression::Constant(1)),
            )))
        );
        assert_eq!(rest, &tokens[11..]);

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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::FromInteger(Box::new(IntegerVectorExpression::Table(
                Box::new(TableVectorExpression::Table3D(
                    0,
                    VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                    VectorOrElementExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                ))
            )))
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Table3DReduce(
                ReduceOperator::Sum,
                0,
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1])
                )),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
            )))
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::FromInteger(Box::new(IntegerVectorExpression::Table(
                Box::new(TableVectorExpression::Table3DReduce(
                    ReduceOperator::Sum,
                    0,
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(1)),
                ))
            )))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-product-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Table3DReduce(
                ReduceOperator::Product,
                0,
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1])
                )),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
            )))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-product-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::FromInteger(Box::new(IntegerVectorExpression::Table(
                Box::new(TableVectorExpression::Table3DReduce(
                    ReduceOperator::Product,
                    0,
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(1)),
                ))
            )))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-max-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Table3DReduce(
                ReduceOperator::Max,
                0,
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1])
                )),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
            )))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-max-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::FromInteger(Box::new(IntegerVectorExpression::Table(
                Box::new(TableVectorExpression::Table3DReduce(
                    ReduceOperator::Max,
                    0,
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(1)),
                ))
            )))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-min-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::Table(Box::new(TableVectorExpression::Table3DReduce(
                ReduceOperator::Min,
                0,
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0))),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Constant(vec![0, 1])
                )),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
            )))
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = [
            "(",
            "table-min-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousVectorExpression::FromInteger(Box::new(IntegerVectorExpression::Table(
                Box::new(TableVectorExpression::Table3DReduce(
                    ReduceOperator::Min,
                    0,
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(1)),
                ))
            )))
        );
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_continuous_vector_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "table-vector", "ff", "v0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
