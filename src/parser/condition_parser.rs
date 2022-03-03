use super::numeric_parser;
use super::set_parser;
use super::ParseErr;
use crate::expression::condition::{ComparisonOperator, Condition};
use crate::expression::set_condition::SetCondition;
use crate::problem;
use crate::variable;
use std::fmt;
use std::str;

pub fn parse_condition<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(Condition<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "(" => parse_operation(rest, problem),
        _ => Err(ParseErr::Reason(format!("unexpected token: `{}`", token))),
    }
}

fn parse_operation<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(Condition<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (name, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &name[..] {
        "not" => {
            let (condition, rest) = parse_condition(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Not(Box::new(condition)), rest))
        }
        "and" => {
            let (x, rest) = parse_condition(rest, problem)?;
            let (y, rest) = parse_condition(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::And(Box::new(x), Box::new(y)), rest))
        }
        "or" => {
            let (x, rest) = parse_condition(rest, problem)?;
            let (y, rest) = parse_condition(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Or(Box::new(x), Box::new(y)), rest))
        }
        "=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, problem)?;
            let (y, rest) = numeric_parser::parse_expression(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Eq, x, y), rest))
        }
        "!=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, problem)?;
            let (y, rest) = numeric_parser::parse_expression(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Ne, x, y), rest))
        }
        ">=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, problem)?;
            let (y, rest) = numeric_parser::parse_expression(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Ge, x, y), rest))
        }
        ">" => {
            let (x, rest) = numeric_parser::parse_expression(rest, problem)?;
            let (y, rest) = numeric_parser::parse_expression(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Gt, x, y), rest))
        }
        "<=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, problem)?;
            let (y, rest) = numeric_parser::parse_expression(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Le, x, y), rest))
        }
        "<" => {
            let (x, rest) = numeric_parser::parse_expression(rest, problem)?;
            let (y, rest) = numeric_parser::parse_expression(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Lt, x, y), rest))
        }
        "in" => {
            let (e, rest) = set_parser::parse_element(rest, problem)?;
            let (s, rest) = set_parser::parse_set(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Set(SetCondition::In(e, s)), rest))
        }
        "subset_of" => {
            let (x, rest) = set_parser::parse_set(rest, problem)?;
            let (y, rest) = set_parser::parse_set(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Set(SetCondition::SubsetOf(x, y)), rest))
        }
        "is_empty" => {
            let (s, rest) = set_parser::parse_set(rest, problem)?;
            let rest = super::parse_closing(rest)?;
            Ok((Condition::Set(SetCondition::IsEmpty(s)), rest))
        }
        _ => Err(ParseErr::Reason(format!("no such operator: `{}`", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::numeric_expression::*;
    use crate::expression::set_expression::*;
    use std::collections::HashMap;

    fn generate_problem() -> problem::Problem<variable::IntegerVariable> {
        problem::Problem {
            set_variable_to_max_size: vec![4],
            permutation_variable_to_max_length: vec![3],
            element_to_set: vec![0, 1],
            functions_1d: HashMap::new(),
            functions_2d: HashMap::new(),
            functions_3d: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    #[test]
    fn parse_err() {
        let problem = generate_problem();

        let tokens: Vec<String> = [")", "(", "in", "2", "s[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "2", "s[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_not_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", "not", "(", "is_empty", "s[0]", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "not", "s[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "not", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "and", "(", "not", "s[0]", ")", "(", "is_empty", "s[1]", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_and_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = [
            "(", "and", "(", "is_empty", "s[0]", ")", "(", "is_empty", "s[1]", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "and", "(", "is_empty", "s[0]", ")", "s[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "and", "(", "is_empty", "s[0]", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "and", "(", "is_empty", "s[0]", ")", "(", "is_empty", "s[1]", ")", "(",
            "is_empty", "s[2]", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_or_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = [
            "(", "or", "(", "is_empty", "s[0]", ")", "(", "is_empty", "s[1]", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "or", "(", "is_empty", "s[0]", ")", "s[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "or", "(", "is_empty", "s[0]", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "or", "(", "is_empty", "s[0]", ")", "(", "is_empty", "s[1]", ")", "(", "is_empty",
            "s[2]", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_eq_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", "=", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "=", "2", "e[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "=", "2", "n[0]", "n[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_ne_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", "!=", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "!=", "2", "e[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "!=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "!=", "2", "n[0]", "n[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_gt_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", ">", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", ">", "2", "e[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">", "2", "n[0]", "n[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_ge_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", ">=", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", ">=", "2", "e[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">=", "2", "n[0]", "n[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_lt_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", "<", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "<", "2", "e[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<", "2", "n[0]", "n[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_le_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", "<=", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "<=", "2", "e[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<=", "2", "n[0]", "n[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_in_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", "in", "2", "s[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Set(SetCondition::In(
                ElementExpression::Constant(2),
                SetExpression::SetVariable(0)
            ))
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_in_err() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "in", "s[0]", "s[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "in", "0", "e[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "in", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "in", "0", "s[1]", "s[2]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_subset_of_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", "subset_of", "s[0]", "s[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Set(SetCondition::SubsetOf(
                SetExpression::SetVariable(0),
                SetExpression::SetVariable(1)
            ))
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_subset_of_err() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "subset_of", "e[0]", "s[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "subset_of", "s[0]", "e[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "subset_of", "s[0]", "s[1]", "s[2]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_empty_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["(", "is_empty", "s[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_condition(&tokens, &problem);
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
        let problem = generate_problem();

        let tokens: Vec<String> = ["(", "is_empty", "e[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_empty", "s[0]", "s[1]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());
    }
}
