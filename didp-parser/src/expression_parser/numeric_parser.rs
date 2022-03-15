use super::function_parser;
use super::set_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{NumericExpression, NumericOperator};
use crate::function_registry;
use crate::state;
use crate::variable;
use std::collections;
use std::fmt;
use std::str;

pub fn parse_expression<'a, 'b, 'c, T: variable::Numeric>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b function_registry::FunctionRegistry<T>,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
            if let Some((expression, rest)) =
                function_parser::parse_expression(name, rest, metadata, registry, parameters)?
            {
                Ok((NumericExpression::Function(expression), rest))
            } else {
                parse_operation(name, rest, metadata, registry, parameters)
            }
        }
        ")" => Err(ParseErr::Reason("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, metadata, parameters),
        _ => {
            let expression = parse_atom(token, metadata)?;
            Ok((expression, rest))
        }
    }
}

fn parse_operation<'a, 'b, 'c, T: variable::Numeric>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b function_registry::FunctionRegistry<T>,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
    let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
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
        _ => Err(ParseErr::Reason(format!("no such operator: `{}`", name))),
    }
}

fn parse_cardinality<'a, 'b, 'c, T: variable::Numeric>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr> {
    let (expression, rest) = set_parser::parse_set_expression(tokens, metadata, parameters)?;
    let (token, rest) = rest
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    if token != "|" {
        return Err(ParseErr::Reason(format!(
            "unexpected token: `{}`, expected `|`",
            token
        )));
    }
    Ok((NumericExpression::Cardinality(expression), rest))
}

fn parse_atom<'a, 'b, T: variable::Numeric>(
    token: &'a str,
    metadata: &'b state::StateMetadata,
) -> Result<NumericExpression<'b, T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    if token == "cost" {
        Ok(NumericExpression::Cost)
    } else if let Some(i) = metadata.name_to_numeric_variable.get(token) {
        Ok(NumericExpression::Variable(*i))
    } else if let Some(i) = metadata.name_to_resource_variable.get(token) {
        Ok(NumericExpression::ResourceVariable(*i))
    } else {
        let n: T = token.parse().map_err(|e| {
            ParseErr::Reason(format!("could not parse {} as a number: {:?}", token, e))
        })?;
        Ok(NumericExpression::Constant(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::numeric_function;
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

        let numeric_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_numeric_variable = HashMap::new();
        name_to_numeric_variable.insert("n0".to_string(), 0);
        name_to_numeric_variable.insert("n1".to_string(), 1);
        name_to_numeric_variable.insert("n2".to_string(), 2);
        name_to_numeric_variable.insert("n3".to_string(), 3);

        let resource_variable_names = vec![
            "r0".to_string(),
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ];
        let mut name_to_resource_variable = HashMap::new();
        name_to_resource_variable.insert("r0".to_string(), 0);
        name_to_resource_variable.insert("r1".to_string(), 1);
        name_to_resource_variable.insert("r2".to_string(), 2);
        name_to_resource_variable.insert("r3".to_string(), 3);

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
            numeric_variable_names,
            name_to_numeric_variable,
            resource_variable_names,
            name_to_resource_variable,
            less_is_better: vec![false, false, true, false],
        }
    }

    fn generate_parameters() -> collections::HashMap<String, usize> {
        let mut parameters = collections::HashMap::new();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    fn generate_registry() -> function_registry::FunctionRegistry<variable::IntegerVariable> {
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
        let f4 = numeric_function::NumericFunction::new(HashMap::new(), 0);
        functions.insert("f4".to_string(), f4);

        function_registry::FunctionRegistry {
            functions_1d,
            functions_2d,
            functions_3d,
            functions,
        }
    }

    #[test]
    fn parse_expression_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens = Vec::new();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "(", "+", "cost", "1", ")", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let f = &registry.functions["f4"];
        let tokens: Vec<String> = ["(", "f4", "0", "e0", "s0", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Function(FunctionExpression::FunctionSum(_, _))
        ));
        if let NumericExpression::Function(FunctionExpression::FunctionSum(g, args)) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert_eq!(args.len(), 4);
            assert!(matches!(
                args[0],
                ArgumentExpression::Element(ElementExpression::Constant(0))
            ));
            assert!(matches!(
                args[1],
                ArgumentExpression::Element(ElementExpression::Variable(0))
            ));
            assert!(matches!(
                args[2],
                ArgumentExpression::Set(SetExpression::SetVariable(0))
            ));
            assert!(matches!(
                args[3],
                ArgumentExpression::Set(SetExpression::PermutationVariable(0))
            ));
        }
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_function_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "f4", "0", "e0", "s0", "p0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
            assert!(matches!(*y, NumericExpression::Variable(0)));
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
            assert!(matches!(*y, NumericExpression::Variable(0)));
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
            assert!(matches!(*y, NumericExpression::Variable(0)));
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
            assert!(matches!(*y, NumericExpression::Variable(0)));
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
            assert!(matches!(*y, NumericExpression::Variable(0)));
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
            assert!(matches!(*y, NumericExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_operation_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens = Vec::new();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", "n0", "n1", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "^", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
    fn parse_atom_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["cost", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Cost));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["n1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Variable(1)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["r1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::ResourceVariable(1)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
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
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["e1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
