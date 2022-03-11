use super::function_parser;
use super::set_parser;
use super::ParseErr;
use crate::expression::{NumericExpression, NumericOperator};
use crate::problem;
use crate::variable;
use lazy_static::lazy_static;
use regex::Regex;
use std::fmt;
use std::str;

pub fn parse_expression<'a, 'b, T: variable::Numeric>(
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
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
            if let Some((expression, rest)) =
                function_parser::parse_expression(name, rest, problem)?
            {
                Ok((NumericExpression::Function(expression), rest))
            } else {
                parse_operation(name, rest, problem)
            }
        }
        ")" => Err(ParseErr::Reason("unexpected `)`".to_string())),
        "|" => parse_cardinality(rest, problem),
        _ => {
            let expression = parse_atom(token)?;
            Ok((expression, rest))
        }
    }
}

fn parse_operation<'a, 'b, T: variable::Numeric>(
    name: &'a str,
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (x, rest) = parse_expression(tokens, problem)?;
    let (y, rest) = parse_expression(rest, problem)?;
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
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr> {
    let (expression, rest) = set_parser::parse_set_expression(tokens, problem)?;
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
) -> Result<NumericExpression<'b, T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    lazy_static! {
        static ref NUMERIC: Regex = Regex::new(r"^n\[(\d+)\]$").unwrap();
        static ref RESOURCE: Regex = Regex::new(r"^r\[(\d+)\]$").unwrap();
    }

    if token == "cost" {
        return Ok(NumericExpression::Cost);
    }

    if let Some(caps) = NUMERIC.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason(format!(
                "could not parse an index of a numeric variable: {:?}",
                e
            ))
        })?;
        return Ok(NumericExpression::Variable(i));
    }

    if let Some(caps) = RESOURCE.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason(format!(
                "could not parse an index of a resource variable: {:?}",
                e
            ))
        })?;
        return Ok(NumericExpression::ResourceVariable(i));
    }

    let n: T = token
        .parse()
        .map_err(|e| ParseErr::Reason(format!("could not parse {} as a number: {:?}", token, e)))?;
    Ok(NumericExpression::Constant(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::numeric_function;
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
    fn parse_expression_err() {
        let problem = generate_problem();

        let tokens = Vec::new();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "(", "+", "g", "1", ")", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_ok() {
        let problem = generate_problem();
        let f = &problem.functions["f4"];
        let tokens: Vec<String> = ["(", "f4", "0", "e[0]", "s[0]", "p[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
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
        let problem = generate_problem();
        let tokens: Vec<String> = [
            "(", "f4", "0", "e[0]", "s[0]", "p[0]", "n[0]", ")", "n[0]", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_operation_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "+", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
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

        let tokens: Vec<String> = ["(", "-", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
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

        let tokens: Vec<String> = ["(", "*", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
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

        let tokens: Vec<String> = ["(", "/", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
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

        let tokens: Vec<String> = ["(", "min", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
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

        let tokens: Vec<String> = ["(", "max", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens = Vec::new();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", "n[0]", "n[1]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "0", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "^", "0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn pare_cardinality_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["|", "s[2]", "|", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
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
        let problem = generate_problem();
        let tokens: Vec<String> = ["|", "e[2]", "|", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["|", "s[2]", "s[0]", "|", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["cost", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Cost));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["n[11]", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Variable(11)));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["r[11]", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::ResourceVariable(11)
        ));
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["11", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::Constant(11)));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_atom_err() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["h", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["e[1]", "1", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_err());
    }
}
