use super::environment;
use super::numeric_parser;
use super::set_parser;
use super::ParseErr;
use crate::expression::{ComparisonOperator, Condition, SetCondition};
use crate::variable;
use std::fmt;
use std::str;

pub fn parse_expression<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(Condition<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "(" => parse_operation(rest, env),
        _ => Err(ParseErr::Reason(format!("unexpected token: `{}`", token))),
    }
}

fn parse_operation<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(Condition<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (name, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &name[..] {
        "not" => {
            let (condition, rest) = parse_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Not(Box::new(condition)), rest))
        }
        "and" => {
            let (x, rest) = parse_expression(rest, env)?;
            let (y, rest) = parse_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::And(Box::new(x), Box::new(y)), rest))
        }
        "or" => {
            let (x, rest) = parse_expression(rest, env)?;
            let (y, rest) = parse_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Or(Box::new(x), Box::new(y)), rest))
        }
        "=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, env)?;
            let (y, rest) = numeric_parser::parse_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Eq, x, y), rest))
        }
        "!=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, env)?;
            let (y, rest) = numeric_parser::parse_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Ne, x, y), rest))
        }
        ">=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, env)?;
            let (y, rest) = numeric_parser::parse_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Ge, x, y), rest))
        }
        ">" => {
            let (x, rest) = numeric_parser::parse_expression(rest, env)?;
            let (y, rest) = numeric_parser::parse_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Gt, x, y), rest))
        }
        "<=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, env)?;
            let (y, rest) = numeric_parser::parse_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Le, x, y), rest))
        }
        "<" => {
            let (x, rest) = numeric_parser::parse_expression(rest, env)?;
            let (y, rest) = numeric_parser::parse_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Lt, x, y), rest))
        }
        "is_in" => {
            let (e, rest) = set_parser::parse_element_expression(rest, env)?;
            let (s, rest) = set_parser::parse_set_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Set(SetCondition::IsIn(e, s)), rest))
        }
        "is_subset" => {
            let (x, rest) = set_parser::parse_set_expression(rest, env)?;
            let (y, rest) = set_parser::parse_set_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Set(SetCondition::IsSubset(x, y)), rest))
        }
        "is_empty" => {
            let (s, rest) = set_parser::parse_set_expression(rest, env)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Set(SetCondition::IsEmpty(s)), rest))
        }
        _ => Err(ParseErr::Reason(format!("no such operator: `{}`", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
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

        environment::Environment {
            set_variables,
            permutation_variables,
            element_variables,
            numeric_variables,
            resource_variables,
            functions_1d: HashMap::new(),
            functions_2d: HashMap::new(),
            functions_3d: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    #[test]
    fn parse_err() {
        let env = generate_env();

        let tokens: Vec<String> = [")", "(", "is_in", "2", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "2", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_not_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", "not", "(", "is_empty", "s0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, Condition::Not(_)));
        if let Condition::Not(c) = expression {
            assert!(matches!(
                *c,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(0)))
            ));
        };
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_not_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "not", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "not", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "and", "(", "not", "s0", ")", "(", "is_empty", "s1", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_and_ok() {
        let env = generate_env();
        let tokens: Vec<String> = [
            "(", "and", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, Condition::And(_, _)));
        if let Condition::And(x, y) = expression {
            assert!(matches!(
                *x,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(0)))
            ));
            assert!(matches!(
                *y,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(1)))
            ));
        };
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_and_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "and", "(", "is_empty", "s0", ")", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "and", "(", "is_empty", "s0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "and", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", "(", "is_empty",
            "s2", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_or_ok() {
        let env = generate_env();
        let tokens: Vec<String> = [
            "(", "or", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, Condition::Or(_, _)));
        if let Condition::Or(x, y) = expression {
            assert!(matches!(
                *x,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(0)))
            ));
            assert!(matches!(
                *y,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(1)))
            ));
        };
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_or_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "or", "(", "is_empty", "s0", ")", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "or", "(", "is_empty", "s0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "or", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", "(", "is_empty",
            "s2", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_eq_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", "=", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Eq,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_eq_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "=", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_ne_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", "!=", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Ne,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_ne_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "!=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "!=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "!=", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_gt_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", ">", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Gt,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_gt_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", ">", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_ge_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", ">=", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Ge,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_ge_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", ">=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">=", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_lt_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", "<", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Lt,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_lt_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "<", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_le_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", "<=", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Le,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_le_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "<=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<=", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_in_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", "is_in", "2", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Set(SetCondition::IsIn(
                ElementExpression::Constant(2),
                SetExpression::SetVariable(0)
            ))
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_is_in_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "is_in", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_in", "0", "e1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_in", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_in", "0", "s1", "s2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_subset_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", "is_subset", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Set(SetCondition::IsSubset(
                SetExpression::SetVariable(0),
                SetExpression::SetVariable(1)
            ))
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_is_subset_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "is_subset", "e0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_subset", "s0", "e1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_subset", "s0", "s1", "s2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_empty_ok() {
        let env = generate_env();
        let tokens: Vec<String> = ["(", "is_empty", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &env);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(0)))
        ));
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_is_empty_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["(", "is_empty", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_empty", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &env);
        assert!(result.is_err());
    }
}
