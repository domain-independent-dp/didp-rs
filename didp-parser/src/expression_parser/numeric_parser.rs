use super::environment;
use super::function_parser;
use super::set_parser;
use super::ParseErr;
use crate::expression::{NumericExpression, NumericOperator};
use crate::variable;
use std::fmt;
use std::str;

pub fn parse_expression<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
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
            if let Some((expression, rest)) = function_parser::parse_expression(name, rest, env)? {
                Ok((NumericExpression::Function(expression), rest))
            } else {
                parse_operation(name, rest, env)
            }
        }
        ")" => Err(ParseErr::Reason("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, env),
        _ => {
            let expression = parse_atom(token, env)?;
            Ok((expression, rest))
        }
    }
}

fn parse_operation<'a, 'b, T: variable::Numeric>(
    name: &'a str,
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (x, rest) = parse_expression(tokens, env)?;
    let (y, rest) = parse_expression(rest, env)?;
    let rest = super::parse_closing(rest)?;
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

fn parse_cardinality<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr> {
    let (expression, rest) = set_parser::parse_set_expression(tokens, env)?;
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
    env: &'b environment::Environment<T>,
) -> Result<NumericExpression<'b, T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    if token == "cost" {
        Ok(NumericExpression::Cost)
    } else if let Some(i) = env.numeric_variables.get(token) {
        Ok(NumericExpression::Variable(*i))
    } else if let Some(i) = env.resource_variables.get(token) {
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

    fn generate_env() -> environment::Environment<variable::IntegerVariable> {
        let mut set_variables = HashMap::new();
        set_variables.insert("s0".to_string(), 0);
        set_variables.insert("s1".to_string(), 1);
        set_variables.insert("s2".to_string(), 2);
        set_variables.insert("s3".to_string(), 3);

        let mut permutation_variables = HashMap::new();
        permutation_variables.insert("p0".to_string(), 0);
        permutation_variables.insert("p1".to_string(), 1);
        permutation_variables.insert("p2".to_string(), 2);
        permutation_variables.insert("p3".to_string(), 3);

        let mut element_variables = HashMap::new();
        element_variables.insert("e0".to_string(), 0);
        element_variables.insert("e1".to_string(), 1);
        element_variables.insert("e2".to_string(), 2);
        element_variables.insert("e3".to_string(), 3);

        let mut numeric_variables = HashMap::new();
        numeric_variables.insert("n0".to_string(), 0);
        numeric_variables.insert("n1".to_string(), 1);
        numeric_variables.insert("n2".to_string(), 2);
        numeric_variables.insert("n3".to_string(), 3);

        let mut resource_variables = HashMap::new();
        resource_variables.insert("r0".to_string(), 0);
        resource_variables.insert("r1".to_string(), 1);
        resource_variables.insert("r2".to_string(), 2);
        resource_variables.insert("r3".to_string(), 3);

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

        environment::Environment {
            set_variables,
            permutation_variables,
            element_variables,
            numeric_variables,
            resource_variables,
            functions_1d,
            functions_2d,
            functions_3d,
            functions,
        }
    }

    #[test]
    fn parse_expression_err() {
        let env = generate_env();

        let tokens = Vec::new();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "(", "+", "cost", "1", ")", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_ok() {
        let env = generate_env();
        let f = &env.functions["f4"];
        let tokens: Vec<String> = ["(", "f4", "0", "e0", "s0", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
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
        let env = generate_env();
        let tokens: Vec<String> = ["(", "f4", "0", "e0", "s0", "p0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_operation_ok() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "+", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
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
        let result = parse_expression(&tokens, &env);
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
        let result = parse_expression(&tokens, &env);
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
        let result = parse_expression(&tokens, &env);
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
        let result = parse_expression(&tokens, &env);
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
        let result = parse_expression(&tokens, &env);
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
        let env = generate_env();

        let tokens = Vec::new();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", "n0", "n1", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "^", "0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn pare_cardinality_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["|", "s2", "|", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
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
        let env = generate_env();
        let tokens: Vec<String> = ["|", "e2", "|", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["|", "s2", "s0", "|", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let env = generate_env();

        let tokens: Vec<String> = ["cost", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Cost));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["n1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Variable(1)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["r1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::ResourceVariable(1)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Constant(11)));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_atom_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["h", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["e1", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }
}
