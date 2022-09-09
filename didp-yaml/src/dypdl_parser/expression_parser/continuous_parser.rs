use super::condition_parser;
use super::continuous_vector_parser;
use super::element_parser;
use super::integer_parser;
use super::numeric_table_parser;
use super::util;
use super::util::ParseErr;
use dypdl::expression::{
    BinaryOperator, CastOperator, ContinuousBinaryOperator, ContinuousExpression,
    ContinuousUnaryOperator, ContinuousVectorExpression, IntegerExpression, ReduceOperator,
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
    parameters: &FxHashMap<String, Element>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
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
                &registry.continuous_tables,
            )? {
                Ok((ContinuousExpression::Table(Box::new(expression)), rest))
            } else if let Some((expression, rest)) = numeric_table_parser::parse_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.integer_tables,
            )? {
                Ok((
                    ContinuousExpression::FromInteger(Box::new(IntegerExpression::Table(
                        Box::new(expression),
                    ))),
                    rest,
                ))
            } else if name == "length" {
                parse_length(rest, metadata, registry, parameters)
            } else if name == "last" {
                let (vector, rest) = continuous_vector_parser::parse_expression(
                    rest, metadata, registry, parameters,
                )?;
                let rest = util::parse_closing(rest)?;
                Ok((ContinuousExpression::Last(Box::new(vector)), rest))
            } else if name == "at" {
                let (vector, rest) = continuous_vector_parser::parse_expression(
                    rest, metadata, registry, parameters,
                )?;
                let (i, rest) =
                    element_parser::parse_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((ContinuousExpression::At(Box::new(vector), i), rest))
            } else if let Ok((vector, rest)) =
                continuous_vector_parser::parse_expression(rest, metadata, registry, parameters)
            {
                let rest = util::parse_closing(rest)?;
                Ok((parse_reduce(name, vector)?, rest))
            } else if name == "if" {
                let (condition, rest) =
                    condition_parser::parse_expression(rest, metadata, registry, parameters)?;
                let (x, rest) = parse_expression(rest, metadata, registry, parameters)?;
                let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                Ok((
                    ContinuousExpression::If(Box::new(condition), Box::new(x), Box::new(y)),
                    rest,
                ))
            } else if name == "continuous" {
                parse_from_integer(rest, metadata, registry, parameters)
            } else if let Ok(result) = parse_round(name, rest, metadata, registry, parameters) {
                Ok(result)
            } else {
                let (x, rest) = parse_expression(rest, metadata, registry, parameters)?;
                let (expression, rest) =
                    if let Ok(expression) = parse_unary_operation(name, x.clone()) {
                        (expression, rest)
                    } else {
                        let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
                        (parse_binary_operation(name, x, y)?, rest)
                    };
                let rest = util::parse_closing(rest)?;
                Ok((expression, rest))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, metadata, registry, parameters),
        _ => {
            let expression = parse_continuous_atom(token, metadata, registry)?;
            Ok((expression, rest))
        }
    }
}

fn parse_reduce(
    name: &str,
    vector: ContinuousVectorExpression,
) -> Result<ContinuousExpression, ParseErr> {
    match name {
        "reduce-sum" => Ok(ContinuousExpression::Reduce(
            ReduceOperator::Sum,
            Box::new(vector),
        )),
        "reduce-product" => Ok(ContinuousExpression::Reduce(
            ReduceOperator::Product,
            Box::new(vector),
        )),
        "reduce-max" => Ok(ContinuousExpression::Reduce(
            ReduceOperator::Max,
            Box::new(vector),
        )),
        "reduce-min" => Ok(ContinuousExpression::Reduce(
            ReduceOperator::Min,
            Box::new(vector),
        )),
        _ => Err(ParseErr::new(format!(
            "no such reduction operator `{}`",
            name
        ))),
    }
}

fn parse_unary_operation(
    name: &str,
    x: ContinuousExpression,
) -> Result<ContinuousExpression, ParseErr> {
    match name {
        "abs" => Ok(ContinuousExpression::UnaryOperation(
            UnaryOperator::Abs,
            Box::new(x),
        )),
        "sqrt" => Ok(ContinuousExpression::ContinuousUnaryOperation(
            ContinuousUnaryOperator::Sqrt,
            Box::new(x),
        )),
        _ => Err(ParseErr::new(format!("no such unary operator `{}`", name))),
    }
}

fn parse_from_integer<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let (expression, rest) =
        integer_parser::parse_expression(tokens, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((
        ContinuousExpression::FromInteger(Box::new(expression)),
        rest,
    ))
}

fn parse_round<'a>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let op = match name {
        "ceil" => CastOperator::Ceil,
        "floor" => CastOperator::Floor,
        "round" => CastOperator::Round,
        "trunc" => CastOperator::Trunc,
        _ => return Err(ParseErr::new(format!("no such unary operator `{}`", name))),
    };
    let (expression, rest) = parse_expression(tokens, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((ContinuousExpression::Round(op, Box::new(expression)), rest))
}

fn parse_binary_operation(
    name: &str,
    x: ContinuousExpression,
    y: ContinuousExpression,
) -> Result<ContinuousExpression, ParseErr> {
    match name {
        "+" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Add,
            Box::new(x),
            Box::new(y),
        )),
        "-" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Sub,
            Box::new(x),
            Box::new(y),
        )),
        "*" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Mul,
            Box::new(x),
            Box::new(y),
        )),
        "/" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Div,
            Box::new(x),
            Box::new(y),
        )),
        "%" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Rem,
            Box::new(x),
            Box::new(y),
        )),
        "min" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Min,
            Box::new(x),
            Box::new(y),
        )),
        "max" => Ok(ContinuousExpression::BinaryOperation(
            BinaryOperator::Max,
            Box::new(x),
            Box::new(y),
        )),
        "pow" => Ok(ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Pow,
            Box::new(x),
            Box::new(y),
        )),
        "log" => Ok(ContinuousExpression::ContinuousBinaryOperation(
            ContinuousBinaryOperator::Log,
            Box::new(x),
            Box::new(y),
        )),
        _ => Err(ParseErr::new(format!("no such operator `{}`", name))),
    }
}

fn parse_cardinality<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
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
    Ok((ContinuousExpression::Cardinality(expression), rest))
}

fn parse_length<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
) -> Result<(ContinuousExpression, &'a [String]), ParseErr> {
    let (expression, rest) =
        element_parser::parse_vector_expression(tokens, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((ContinuousExpression::Length(expression), rest))
}

fn parse_continuous_atom(
    token: &str,
    metadata: &StateMetadata,
    registry: &TableRegistry,
) -> Result<ContinuousExpression, ParseErr> {
    if let Some(v) = registry.continuous_tables.name_to_constant.get(token) {
        Ok(ContinuousExpression::Constant(*v))
    } else if let Some(i) = metadata.name_to_continuous_variable.get(token) {
        Ok(ContinuousExpression::Variable(*i))
    } else if let Some(i) = metadata.name_to_continuous_resource_variable.get(token) {
        Ok(ContinuousExpression::ResourceVariable(*i))
    } else if let Some(v) = registry.integer_tables.name_to_constant.get(token) {
        Ok(ContinuousExpression::Constant(*v as Continuous))
    } else if let Some(i) = metadata.name_to_integer_variable.get(token) {
        Ok(ContinuousExpression::FromInteger(Box::new(
            IntegerExpression::Variable(*i),
        )))
    } else if let Some(i) = metadata.name_to_integer_resource_variable.get(token) {
        Ok(ContinuousExpression::FromInteger(Box::new(
            IntegerExpression::ResourceVariable(*i),
        )))
    } else if token == "cost" {
        Ok(ContinuousExpression::Cost)
    } else {
        let n: Continuous = token.parse().map_err(|e| {
            ParseErr::new(format!(
                "could not parse {} as a continuous atom: {:?}",
                token, e
            ))
        })?;
        Ok(ContinuousExpression::Constant(n))
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
    fn parse_continuous_atom_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["cost", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Cost);
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["cf0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Constant(0.0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["c1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Variable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["cr1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::ResourceVariable(1));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["f0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Constant(0.0));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["i1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Variable(1)))
        );
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["ir1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::ResourceVariable(1)))
        );
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11.5", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Constant(11.5));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(expression, ContinuousExpression::Constant(11.0));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_continuous_atom_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["e0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "sum", "cf4", "0", "e0", "s0", "v0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Table(Box::new(NumericTableExpression::TableReduce(
                ReduceOperator::Sum,
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
            )))
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_continuous_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "cf4", "0.0", "e0", "s0", "v0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_from_integer_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "continuous", "1", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::FromInteger(Box::new(IntegerExpression::Constant(1)),)
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_from_integer_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "continuous", "1.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_unary_operator_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "abs", "-4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::UnaryOperation(
                UnaryOperator::Abs,
                Box::new(ContinuousExpression::Constant(-4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "sqrt", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::ContinuousUnaryOperation(
                ContinuousUnaryOperator::Sqrt,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "floor", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Round(
                CastOperator::Floor,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "ceil", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Round(
                CastOperator::Ceil,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "round", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Round(
                CastOperator::Round,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["(", "trunc", "4.5", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Round(
                CastOperator::Trunc,
                Box::new(ContinuousExpression::Constant(4.5)),
            )
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_continuous_unary_operator_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "exp", "3.0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_if_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "true", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::If(
                Box::new(Condition::Constant(true)),
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[6..]);
    }

    #[test]
    fn parse_continuous_if_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "if", "true", "0.0", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "if", "0.0", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_binary_operation_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "+", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Add,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "-", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Sub,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "*", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "/", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Div,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "%", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Rem,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "min", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Min,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "max", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::BinaryOperation(
                BinaryOperator::Max,
                Box::new(ContinuousExpression::Constant(0.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "pow", "2.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Pow,
                Box::new(ContinuousExpression::Constant(2.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "log", "2.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::ContinuousBinaryOperation(
                ContinuousBinaryOperator::Log,
                Box::new(ContinuousExpression::Constant(2.0)),
                Box::new(ContinuousExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_continuous_binary_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens = Vec::new();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0.0", "c0", "c1", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0.0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "^", "0.0", "c0", ")", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_cardinality_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "s2", "|", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Cardinality(SetExpression::Reference(
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
        let tokens: Vec<String> = ["|", "e2", "|", "c0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["|", "s2", "s0", "|", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::Length(VectorExpression::Reference(
                ReferenceExpression::Variable(0)
            ))
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
            ContinuousExpression::Last(Box::new(ContinuousVectorExpression::Constant(vec![
                0.0, 1.0
            ])))
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
            "continuous-vector",
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ContinuousExpression::At(
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]),),
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "at", "(", "continuous-vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "at", "(", "vector", "0", "1", ")", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
            ContinuousExpression::Reduce(
                ReduceOperator::Sum,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = [
            "(",
            "reduce-product",
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
            ContinuousExpression::Reduce(
                ReduceOperator::Product,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = [
            "(",
            "reduce-max",
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
            ContinuousExpression::Reduce(
                ReduceOperator::Max,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
            )
        );
        assert_eq!(rest, &tokens[8..]);

        let tokens: Vec<String> = [
            "(",
            "reduce-min",
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
            ContinuousExpression::Reduce(
                ReduceOperator::Min,
                Box::new(ContinuousVectorExpression::Constant(vec![0.0, 1.0]))
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "reduce-max", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "reduce-min", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(",
            "reduce-null",
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
        assert!(result.is_err());
    }
}
