use super::numeric_table_parser;
use super::set_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{NumericExpression, NumericOperator};
use crate::state;
use crate::table_registry;
use crate::variable::{Integer, Numeric};
use std::collections;
use std::fmt;
use std::str;

pub fn parse_expression<'a, 'b, 'c, T: Numeric>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b table_registry::TableRegistry,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericExpression<T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    match parse_integer_expression(tokens, metadata, registry, parameters) {
        Ok(expression) => Ok(expression),
        Err(_) => parse_continuous_expression(tokens, metadata, registry, parameters),
    }
}

pub fn parse_integer_expression<'a, 'b, 'c, T: Numeric>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b table_registry::TableRegistry,
    parameters: &'c collections::HashMap<String, usize>,
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
                &registry.integer_tables,
                parameters,
            )? {
                Ok((NumericExpression::IntegerTable(expression), rest))
            } else {
                let (x, rest) = parse_integer_expression(rest, metadata, registry, parameters)?;
                let (y, rest) = parse_integer_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                let expression = parse_operation(name, x, y)?;
                Ok((expression, rest))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, metadata, parameters),
        _ => {
            let expression = parse_integer_atom(token, metadata, registry)?;
            Ok((expression, rest))
        }
    }
}

pub fn parse_continuous_expression<'a, 'b, 'c, T: Numeric>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b table_registry::TableRegistry,
    parameters: &'c collections::HashMap<String, usize>,
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
                &registry.integer_tables,
                parameters,
            )? {
                Ok((NumericExpression::IntegerTable(expression), rest))
            } else if let Some((expression, rest)) = numeric_table_parser::parse_expression(
                name,
                rest,
                metadata,
                &registry.continuous_tables,
                parameters,
            )? {
                Ok((NumericExpression::ContinuousTable(expression), rest))
            } else {
                let (x, rest) = parse_continuous_expression(rest, metadata, registry, parameters)?;
                let (y, rest) = parse_continuous_expression(rest, metadata, registry, parameters)?;
                let rest = util::parse_closing(rest)?;
                let expression = parse_operation(name, x, y)?;
                Ok((expression, rest))
            }
        }
        ")" => Err(ParseErr::new("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, metadata, parameters),
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

fn parse_cardinality<'a, 'b, 'c, T: Numeric>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericExpression<T>, &'a [String]), ParseErr> {
    let (expression, rest) = set_parser::parse_set_expression(tokens, metadata, parameters)?;
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

fn parse_integer_atom<T: Numeric>(
    token: &str,
    metadata: &state::StateMetadata,
    registry: &table_registry::TableRegistry,
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
    metadata: &state::StateMetadata,
    registry: &table_registry::TableRegistry,
) -> Result<NumericExpression<T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    if token == "cost" {
        Ok(NumericExpression::Cost)
    } else if let Some(v) = registry.integer_tables.name_to_constant.get(token) {
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
    } else {
        let n: T = token.parse().map_err(|e| {
            ParseErr::new(format!("could not parse {} as a number: {:?}", token, e))
        })?;
        Ok(NumericExpression::Constant(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::table;
    use std::collections::HashMap;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = HashMap::new();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            "s0".to_string(),
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
        ];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        name_to_set_variable.insert("s2".to_string(), 2);
        name_to_set_variable.insert("s3".to_string(), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let permutation_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_permutation_variable = HashMap::new();
        name_to_permutation_variable.insert("p0".to_string(), 0);
        name_to_permutation_variable.insert("p1".to_string(), 1);
        name_to_permutation_variable.insert("p2".to_string(), 2);
        name_to_permutation_variable.insert("p3".to_string(), 3);
        let permutation_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
        ];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        name_to_element_variable.insert("e2".to_string(), 2);
        name_to_element_variable.insert("e3".to_string(), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let integer_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_integer_variable = HashMap::new();
        name_to_integer_variable.insert("n0".to_string(), 0);
        name_to_integer_variable.insert("n1".to_string(), 1);
        name_to_integer_variable.insert("n2".to_string(), 2);
        name_to_integer_variable.insert("n3".to_string(), 3);

        let integer_resource_variable_names = vec![
            "r0".to_string(),
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ];
        let mut name_to_integer_resource_variable = HashMap::new();
        name_to_integer_resource_variable.insert("r0".to_string(), 0);
        name_to_integer_resource_variable.insert("r1".to_string(), 1);
        name_to_integer_resource_variable.insert("r2".to_string(), 2);
        name_to_integer_resource_variable.insert("r3".to_string(), 3);

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            permutation_variable_names,
            name_to_permutation_variable,
            permutation_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            integer_variable_names,
            name_to_integer_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true, false],
            ..Default::default()
        }
    }

    fn generate_parameters() -> collections::HashMap<String, usize> {
        let mut parameters = collections::HashMap::new();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    fn generate_registry() -> table_registry::TableRegistry {
        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![table::Table1D::new(Vec::new())];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(Vec::new())];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(Vec::new())];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("f3"), 0);

        let tables = vec![table::Table::new(HashMap::new(), 0)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("f4"), 0);

        table_registry::TableRegistry {
            integer_tables: table_registry::TableData {
                name_to_constant,
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                tables_3d,
                name_to_table_3d,
                tables,
                name_to_table,
            },
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
    fn parse_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "f4", "0", "e0", "s0", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::IntegerTable(NumericTableExpression::TableSum(_, _))
        ));
        if let NumericExpression::IntegerTable(NumericTableExpression::TableSum(i, args)) =
            expression
        {
            assert_eq!(i, 0);
            assert_eq!(args.len(), 4);
            assert_eq!(
                args[0],
                ArgumentExpression::Element(ElementExpression::Constant(0))
            );
            assert_eq!(
                args[1],
                ArgumentExpression::Element(ElementExpression::Variable(0))
            );
            assert_eq!(
                args[2],
                ArgumentExpression::Set(SetExpression::SetVariable(0))
            );
            assert_eq!(
                args[3],
                ArgumentExpression::Set(SetExpression::PermutationVariable(0))
            );
        }
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "f4", "0", "e0", "s0", "p0", "n0", ")", "n0", ")"]
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

        let tokens: Vec<String> = ["(", "+", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Add, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Add, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::IntegerVariable(0)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "-", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Subtract, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Subtract, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::IntegerVariable(0)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "*", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Multiply, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Multiply, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::IntegerVariable(0)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "/", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Divide, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Divide, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::IntegerVariable(0)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "min", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Min, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Min, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::IntegerVariable(0)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "max", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::NumericOperation(NumericOperator::Max, _, _)
        ));
        if let NumericExpression::NumericOperation(NumericOperator::Max, x, y) = expression {
            assert!(matches!(*x, NumericExpression::Constant(0)));
            assert!(matches!(*y, NumericExpression::IntegerVariable(0)));
        }
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

        let tokens: Vec<String> = ["(", "+", "0", "n0", "n1", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "^", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn pare_cardinality_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "s2", "|", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Cardinality(SetExpression::SetVariable(2))
        ));
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn pare_cardinality_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["|", "e2", "|", "n0", ")"]
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
        assert!(matches!(expression, NumericExpression::Cost));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["f0", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Constant(0)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["n1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::IntegerVariable(1)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["r1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::IntegerResourceVariable(1)
        ));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression::<Integer>(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Constant(11)));
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
