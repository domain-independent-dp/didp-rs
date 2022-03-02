use super::function_parser;
use super::set_parser;
use crate::expression::numeric_expression::*;
use crate::expression::set_expression::*;
use crate::parser;
use crate::parser::ParseErr;
use crate::problem;
use crate::variable;
use lazy_static::lazy_static;
use regex::Regex;
use std::fmt;
use std::str;

pub fn parse<T: variable::Numeric>(
    text: String,
    problem: &problem::Problem<T>,
) -> Result<NumericExpression<T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tokens = parser::tokenize(text);
    let (expression, rest) = parse_expression(&tokens, problem)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::Reason(format!(
            "unexpected tokens at the end: {}",
            rest.join(" ")
        )))
    }
}

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
    let rest = parser::parse_closing(rest)?;
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

fn parse_cardinality<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(NumericExpression<'b, T>, &'a [String]), ParseErr> {
    let (expression, rest) = set_parser::parse_argument(tokens, problem)?;
    let (token, rest) = rest
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    if token != "|" {
        return Err(ParseErr::Reason(format!(
            "unexpected {}, expected `|`",
            token
        )));
    }
    match expression {
        ArgumentExpression::Set(expression) => {
            Ok((NumericExpression::Cardinality(expression), rest))
        }
        _ => Err(ParseErr::Reason(format!(
            "cardinality of not a set expression: {:?}",
            expression
        ))),
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
    fn parse_ok() {
        let problem = generate_problem();
        let text = "(+ (- 5 (/ (f4 4 !s[2] e[0] 3) (max (f2 2 e[1]) n[0]))) (* r[1] (min 3 |(+ (* s[0] (- s[2] (+ s[3] 2))) (- s[1] 1))|)))".to_string();
        let result = parse(text, &problem);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_err() {
        let problem = generate_problem();
        let text = "(+ g 1))".to_string();
        let result = parse(text, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_expression_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "+", "g", "1", ")", "2", ")"]
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
            assert!(matches!(*x, NumericExpression::G));
            assert!(matches!(*y, NumericExpression::Constant(1)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["|", "s[0]", "|", "2", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Cardinality(SetExpression::SetVariable(0))
        ));
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["r[0]", "2", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, NumericExpression::ResourceVariable(0)));
        assert_eq!(rest, &tokens[1..]);
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
    fn parse_operation_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation("+", &tokens, &problem);
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
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation("-", &tokens, &problem);
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
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation("*", &tokens, &problem);
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
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation("/", &tokens, &problem);
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
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation("min", &tokens, &problem);
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
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation("max", &tokens, &problem);
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
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_operation_err() {
        let problem = generate_problem();

        let tokens = Vec::new();
        let result = parse_operation("+", &tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "n[0]", "n[1]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation("+", &tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation("+", &tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "n[0]", ")", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation("^", &tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn pare_cardinality_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["s[2]", "|", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_cardinality(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            NumericExpression::Cardinality(SetExpression::SetVariable(2))
        ));
        assert_eq!(rest, &tokens[2..]);
    }

    #[test]
    fn pare_cardinality_err() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["e[2]", "|", "n[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_cardinality(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["s[2]", "s[0]", "|", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_cardinality(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let token = "g";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), NumericExpression::G));

        let token = "n[11]";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), NumericExpression::Variable(11)));

        let token = "r[11]";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            NumericExpression::ResourceVariable(11)
        ));

        let token = "11";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), NumericExpression::Constant(11)));
    }

    #[test]
    fn parse_atom_err() {
        let token = "h";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_err());

        let token = "e[1]";
        let result: Result<NumericExpression<variable::IntegerVariable>, ParseErr> =
            parse_atom(&token);
        assert!(result.is_err());
    }
}
