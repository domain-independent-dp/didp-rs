use super::condition_parser;
use super::continuous_vector_parser;
use super::element_parser;
use super::integer_parser;
use super::table_vector_parser;
use super::table_vector_parser::ModelData;
use super::util;
use super::util::ParseErr;
use dypdl::expression::{
    BinaryOperator, CastOperator, IntegerExpression, IntegerVectorExpression, ReduceOperator,
    UnaryOperator,
};
use dypdl::variable_type::{Element, Integer};
use dypdl::{StateFunctions, StateMetadata, TableRegistry};
use rustc_hash::FxHashMap;
use std::str;

pub fn parse_expression<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(IntegerVectorExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            match &name[..] {
                "integer-vector" => parse_integer_vector_constant(rest, registry),
                "reverse" => {
                    let (vector, rest) =
                        parse_expression(rest, metadata, functions, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((IntegerVectorExpression::Reverse(Box::new(vector)), rest))
                }
                "push" => {
                    let (v, rest) = integer_parser::parse_expression(
                        rest, metadata, functions, registry, parameters,
                    )?;
                    let (vector, rest) =
                        parse_expression(rest, metadata, functions, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((IntegerVectorExpression::Push(v, Box::new(vector)), rest))
                }
                "pop" => {
                    let (vector, rest) =
                        parse_expression(rest, metadata, functions, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((IntegerVectorExpression::Pop(Box::new(vector)), rest))
                }
                "set" => {
                    let (v, rest) = integer_parser::parse_expression(
                        rest, metadata, functions, registry, parameters,
                    )?;
                    let (vector, rest) =
                        parse_expression(rest, metadata, functions, registry, parameters)?;
                    let (i, rest) = element_parser::parse_expression(
                        rest, metadata, functions, registry, parameters,
                    )?;
                    let rest = util::parse_closing(rest)?;
                    Ok((IntegerVectorExpression::Set(v, Box::new(vector), i), rest))
                }
                "table-vector" => {
                    let (name, rest) = rest
                        .split_first()
                        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
                    if let Some((expression, rest)) = table_vector_parser::parse_expression(
                        name,
                        rest,
                        metadata,
                        functions,
                        registry,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((IntegerVectorExpression::Table(Box::new(expression)), rest))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
                "if" => {
                    let (condition, rest) = condition_parser::parse_expression(
                        rest, metadata, functions, registry, parameters,
                    )?;
                    let (x, rest) =
                        parse_expression(rest, metadata, functions, registry, parameters)?;
                    let (y, rest) =
                        parse_expression(rest, metadata, functions, registry, parameters)?;
                    let rest = util::parse_closing(rest)?;
                    Ok((
                        IntegerVectorExpression::If(Box::new(condition), Box::new(x), Box::new(y)),
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
                            return if let Ok(result) = parse_vector_from_continuous(
                                name, rest, metadata, functions, registry, parameters,
                            ) {
                                Ok(result)
                            } else if let Ok(result) = parse_vector_unary_operation(
                                name, rest, metadata, functions, registry, parameters,
                            ) {
                                Ok(result)
                            } else if let Ok((x, y, rest)) = parse_integer_and_vector(
                                rest, metadata, functions, registry, parameters,
                            ) {
                                Ok((parse_vector_binary_operation_x(name, x, y)?, rest))
                            } else if let Ok((x, y, rest)) = parse_vector_and_integer(
                                rest, metadata, functions, registry, parameters,
                            ) {
                                Ok((parse_vector_binary_operation_y(name, x, y)?, rest))
                            } else if let Ok((x, y, rest)) = parse_vector_and_vector(
                                rest, metadata, functions, registry, parameters,
                            ) {
                                Ok((parse_vector_operation(name, x, y)?, rest))
                            } else {
                                Err(ParseErr::new(format!("could not parse `{:?}`", tokens)))
                            }
                        }
                    };
                    let (name, rest) = rest
                        .split_first()
                        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
                    let model_data = ModelData {
                        metadata,
                        functions,
                        registry,
                    };
                    if let Some((expression, rest)) = table_vector_parser::parse_reduce_expression(
                        name,
                        rest,
                        op,
                        model_data,
                        parameters,
                        &registry.integer_tables,
                    )? {
                        Ok((IntegerVectorExpression::Table(Box::new(expression)), rest))
                    } else {
                        Err(ParseErr::new(format!("no such table `{}`", name)))
                    }
                }
            }
        }
        _ => Err(ParseErr::new(format!("unexpected token `{}`", token))),
    }
}

fn parse_integer_vector_constant<'a>(
    tokens: &'a [String],
    registry: &TableRegistry,
) -> Result<(IntegerVectorExpression, &'a [String]), ParseErr> {
    let mut result = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((IntegerVectorExpression::Constant(result), rest));
        }
        let v = if let Some(v) = registry.integer_tables.name_to_constant.get(next_token) {
            *v
        } else {
            let v: Integer = next_token.parse().map_err(|e| {
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

fn parse_vector_unary_operation<'a>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(IntegerVectorExpression, &'a [String]), ParseErr> {
    let (x, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    match name {
        "abs" => Ok((
            IntegerVectorExpression::UnaryOperation(UnaryOperator::Abs, Box::new(x)),
            rest,
        )),
        _ => Err(ParseErr::new(format!("no such unary operator `{}`", name))),
    }
}

fn parse_vector_from_continuous<'a>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(IntegerVectorExpression, &'a [String]), ParseErr> {
    let op = match name {
        "ceil" => CastOperator::Ceil,
        "floor" => CastOperator::Floor,
        "round" => CastOperator::Round,
        _ => return Err(ParseErr::new(format!("no such unary operator `{}`", name))),
    };
    let (expression, rest) = continuous_vector_parser::parse_expression(
        tokens, metadata, functions, registry, parameters,
    )?;
    let rest = util::parse_closing(rest)?;
    Ok((
        IntegerVectorExpression::FromContinuous(op, Box::new(expression)),
        rest,
    ))
}

fn parse_vector_and_integer<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(IntegerVectorExpression, IntegerExpression, &'a [String]), ParseErr> {
    let (vector, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
    let (integer, rest) =
        integer_parser::parse_expression(rest, metadata, functions, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((vector, integer, rest))
}

fn parse_integer_and_vector<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(IntegerExpression, IntegerVectorExpression, &'a [String]), ParseErr> {
    let (integer, rest) =
        integer_parser::parse_expression(tokens, metadata, functions, registry, parameters)?;
    let (vector, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((integer, vector, rest))
}

fn parse_vector_and_vector<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<
    (
        IntegerVectorExpression,
        IntegerVectorExpression,
        &'a [String],
    ),
    ParseErr,
> {
    let (x, rest) = parse_expression(tokens, metadata, functions, registry, parameters)?;
    let (y, rest) = parse_expression(rest, metadata, functions, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((x, y, rest))
}

fn parse_vector_binary_operation_x(
    op: &str,
    v: IntegerExpression,
    vector: IntegerVectorExpression,
) -> Result<IntegerVectorExpression, ParseErr> {
    match op {
        "+" => Ok(IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Add,
            v,
            Box::new(vector),
        )),
        "-" => Ok(IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Sub,
            v,
            Box::new(vector),
        )),
        "*" => Ok(IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Mul,
            v,
            Box::new(vector),
        )),
        "/" => Ok(IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Div,
            v,
            Box::new(vector),
        )),
        "%" => Ok(IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Rem,
            v,
            Box::new(vector),
        )),
        "max" => Ok(IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Max,
            v,
            Box::new(vector),
        )),
        "min" => Ok(IntegerVectorExpression::BinaryOperationX(
            BinaryOperator::Min,
            v,
            Box::new(vector),
        )),
        _ => Err(ParseErr::new(format!("no such binary operator `{}`", op))),
    }
}

fn parse_vector_binary_operation_y(
    op: &str,
    vector: IntegerVectorExpression,
    v: IntegerExpression,
) -> Result<IntegerVectorExpression, ParseErr> {
    match op {
        "+" => Ok(IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Add,
            Box::new(vector),
            v,
        )),
        "-" => Ok(IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Sub,
            Box::new(vector),
            v,
        )),
        "*" => Ok(IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Mul,
            Box::new(vector),
            v,
        )),
        "/" => Ok(IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Div,
            Box::new(vector),
            v,
        )),
        "%" => Ok(IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Rem,
            Box::new(vector),
            v,
        )),
        "max" => Ok(IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Max,
            Box::new(vector),
            v,
        )),
        "min" => Ok(IntegerVectorExpression::BinaryOperationY(
            BinaryOperator::Min,
            Box::new(vector),
            v,
        )),
        _ => Err(ParseErr::new(format!("no such binary operator `{}`", op))),
    }
}

fn parse_vector_operation(
    op: &str,
    x: IntegerVectorExpression,
    y: IntegerVectorExpression,
) -> Result<IntegerVectorExpression, ParseErr> {
    match op {
        "+" => Ok(IntegerVectorExpression::VectorOperation(
            BinaryOperator::Add,
            Box::new(x),
            Box::new(y),
        )),
        "-" => Ok(IntegerVectorExpression::VectorOperation(
            BinaryOperator::Sub,
            Box::new(x),
            Box::new(y),
        )),
        "*" => Ok(IntegerVectorExpression::VectorOperation(
            BinaryOperator::Mul,
            Box::new(x),
            Box::new(y),
        )),
        "/" => Ok(IntegerVectorExpression::VectorOperation(
            BinaryOperator::Div,
            Box::new(x),
            Box::new(y),
        )),
        "%" => Ok(IntegerVectorExpression::VectorOperation(
            BinaryOperator::Rem,
            Box::new(x),
            Box::new(y),
        )),
        "max" => Ok(IntegerVectorExpression::VectorOperation(
            BinaryOperator::Max,
            Box::new(x),
            Box::new(y),
        )),
        "min" => Ok(IntegerVectorExpression::VectorOperation(
            BinaryOperator::Min,
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
    fn parse_integer_vector_constant_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "integer-vector", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, IntegerVectorExpression::Constant(vec![]));
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["(", "integer-vector", "0", "1", "f0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, IntegerVectorExpression::Constant(vec![0, 1, 0]));
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_integer_vector_constant_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "integer-vector", "1"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "integer-vector", "0.0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "integer-vector", "cf0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_reverse_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "reverse",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::Reverse(Box::new(IntegerVectorExpression::Constant(vec![
                0, 1, 0
            ])))
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_integer_vector_reverse_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "reverse", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_push_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "push",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::Push(
                IntegerExpression::Constant(1),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);
    }

    #[test]
    fn parse_integer_vector_push_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "push",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "push", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "push", "1", "(", "vector", "0", "1", "f0", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "push",
            "1.0",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "push",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_pop_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "pop",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::Pop(Box::new(IntegerVectorExpression::Constant(vec![
                0, 1, 0
            ])))
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_integer_vector_pop_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "pop", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_set_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "set",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::Set(
                IntegerExpression::Constant(1),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0])),
                ElementExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_integer_vector_set_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "set",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "set", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "set", "1", "(", "vector", "0", "1", "f0", ")", "1", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "set",
            "1.0",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "set",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_unary_operation_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "abs",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(IntegerVectorExpression::Constant(vec![0, -1, 0]))
            )
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_integer_vector_unary_operation_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "sqrt",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_from_continuous_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "ceil",
            "(",
            "continuous-vector",
            "0.5",
            "-1.5",
            "cf0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::FromContinuous(
                CastOperator::Ceil,
                Box::new(ContinuousVectorExpression::Constant(vec![0.5, -1.5, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[9..]);

        let tokens: Vec<String> = [
            "(",
            "floor",
            "(",
            "continuous-vector",
            "0.5",
            "-1.5",
            "cf0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::FromContinuous(
                CastOperator::Floor,
                Box::new(ContinuousVectorExpression::Constant(vec![0.5, -1.5, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[9..]);

        let tokens: Vec<String> = [
            "(",
            "round",
            "(",
            "continuous-vector",
            "0.5",
            "-1.5",
            "cf0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::FromContinuous(
                CastOperator::Round,
                Box::new(ContinuousVectorExpression::Constant(vec![0.5, -1.5, 0.0]))
            )
        );
        assert_eq!(rest, &tokens[9..]);
    }

    #[test]
    fn parse_integer_vector_from_continuous_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "sqrt",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_binary_operation_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationX(
                BinaryOperator::Add,
                IntegerExpression::Constant(1),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationX(
                BinaryOperator::Sub,
                IntegerExpression::Constant(1),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationX(
                BinaryOperator::Mul,
                IntegerExpression::Constant(1),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationX(
                BinaryOperator::Div,
                IntegerExpression::Constant(1),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "%",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationX(
                BinaryOperator::Rem,
                IntegerExpression::Constant(1),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationX(
                BinaryOperator::Max,
                IntegerExpression::Constant(1),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationX(
                BinaryOperator::Min,
                IntegerExpression::Constant(1),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0]))
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationY(
                BinaryOperator::Add,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0])),
                IntegerExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationY(
                BinaryOperator::Sub,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0])),
                IntegerExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationY(
                BinaryOperator::Mul,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0])),
                IntegerExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationY(
                BinaryOperator::Div,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0])),
                IntegerExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "%",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationY(
                BinaryOperator::Rem,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0])),
                IntegerExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationY(
                BinaryOperator::Max,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0])),
                IntegerExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[10..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::BinaryOperationY(
                BinaryOperator::Min,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1, 0])),
                IntegerExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[10..]);
    }

    #[test]
    fn parse_integer_vector_binary_operation_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "-", "1", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "+",
            "1.0",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "+",
            "1",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "-", "1", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "integer-vector",
            "0",
            "1",
            "f0",
            ")",
            "1.0",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_if_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "if",
            "true",
            "(",
            "integer-vector",
            "0",
            ")",
            "(",
            "integer-vector",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(IntegerVectorExpression::Constant(vec![0])),
                Box::new(IntegerVectorExpression::Constant(vec![1]))
            )
        );
        assert_eq!(rest, &tokens[12..]);
    }

    #[test]
    fn parse_integer_vector_if_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "if",
            "0",
            "(",
            "integer-vector",
            "0",
            ")",
            "(",
            "integer-vector",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "true", "(", "integer-vector", "0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "if",
            "true",
            "true",
            "(",
            "integer-vector",
            "0",
            ")",
            "(",
            "integer-vector",
            "1",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_operation_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "integer-vector",
            "0",
            "1",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::VectorOperation(
                BinaryOperator::Add,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "-",
            "(",
            "integer-vector",
            "0",
            "1",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::VectorOperation(
                BinaryOperator::Sub,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "*",
            "(",
            "integer-vector",
            "0",
            "1",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::VectorOperation(
                BinaryOperator::Mul,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "/",
            "(",
            "integer-vector",
            "0",
            "1",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::VectorOperation(
                BinaryOperator::Div,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "%",
            "(",
            "integer-vector",
            "0",
            "1",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::VectorOperation(
                BinaryOperator::Rem,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "max",
            "(",
            "integer-vector",
            "0",
            "1",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::VectorOperation(
                BinaryOperator::Max,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);

        let tokens: Vec<String> = [
            "(",
            "min",
            "(",
            "integer-vector",
            "0",
            "1",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::VectorOperation(
                BinaryOperator::Min,
                Box::new(IntegerVectorExpression::Constant(vec![0, 1])),
                Box::new(IntegerVectorExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[13..]);
    }

    #[test]
    fn parse_integer_vector_operation_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [
            "(",
            "+",
            "(",
            "integer-vector",
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
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
            "integer-vector",
            "0",
            "1.0",
            ")",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_vector_table_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Table3D(
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Table3DReduce(
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Table3DReduce(
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Table3DReduce(
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
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            IntegerVectorExpression::Table(Box::new(TableVectorExpression::Table3DReduce(
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
    }

    #[test]
    fn parse_integer_vector_table_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "table-vector", "cf1", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-vector", "cf2", "v0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-vector", "cf2", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-vector", "cf2", "0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-vector", "cf2", "0", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-sum-vector", "cf2", "v0", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-sum-vector", "cf2", "s0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-sum-vector", "cf2", "s0", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-product-vector", "cf2", "v0", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-product-vector", "cf2", "s0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "table-product-vector",
            "cf2",
            "s0",
            "v0",
            "0",
            ")",
            ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-max-vector", "cf2", "v0", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-max-vector", "cf2", "s0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-max-vector", "cf2", "s0", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-min-vector", "cf2", "v0", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-min-vector", "cf2", "s0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "table-min-vector", "cf2", "s0", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &functions, &registry, &parameters);
        assert!(result.is_err());
    }
}
