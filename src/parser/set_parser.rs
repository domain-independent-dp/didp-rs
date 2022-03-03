use super::ParseErr;
use crate::expression::set_expression::*;
use crate::problem;
use crate::variable;
use lazy_static::lazy_static;
use regex::Regex;

pub fn parse_set<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (expression, rest) = parse_argument(tokens, problem)?;
    match expression {
        ArgumentExpression::Set(expression) => Ok((expression, rest)),
        _ => Err(ParseErr::Reason(format!(
            "not a set expression: {:?}",
            expression
        ))),
    }
}

pub fn parse_element<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(ElementExpression, &'a [String]), ParseErr> {
    let (expression, rest) = parse_argument(tokens, problem)?;
    match expression {
        ArgumentExpression::Element(expression) => Ok((expression, rest)),
        _ => Err(ParseErr::Reason(format!(
            "not an element expression: {:?}",
            expression
        ))),
    }
}

pub fn parse_argument<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(ArgumentExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "!" => {
            let (expression, rest) = parse_complement(rest, problem)?;
            Ok((ArgumentExpression::Set(expression), rest))
        }
        "(" => {
            let (expression, rest) = parse_operation(rest, problem)?;
            Ok((ArgumentExpression::Set(expression), rest))
        }
        ")" => Err(ParseErr::Reason("unexpected `)`".to_string())),
        _ => {
            let argument = parse_atom(token)?;
            Ok((argument, rest))
        }
    }
}

fn parse_complement<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (expression, rest) = parse_set(tokens, problem)?;
    Ok((SetExpression::Complement(Box::new(expression)), rest))
}

fn parse_operation<'a, 'b, T: variable::Numeric>(
    tokens: &'a [String],
    problem: &'b problem::Problem<T>,
) -> Result<(SetExpression, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    let (x, rest) = parse_argument(rest, problem)?;
    let (y, rest) = parse_argument(rest, problem)?;
    let rest = super::parse_closing(rest)?;

    match &token[..] {
        "+" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                SetExpression::SetOperation(SetOperator::Union, Box::new(x), Box::new(y)),
                rest,
            )),
            (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => Ok((
                SetExpression::SetElementOperation(SetElementOperator::Add, Box::new(x), y),
                rest,
            )),
            args => Err(ParseErr::Reason(format!(
                "unexpected arguments for `+`: {:?}",
                args
            ))),
        },
        "-" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                SetExpression::SetOperation(SetOperator::Difference, Box::new(x), Box::new(y)),
                rest,
            )),
            (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => Ok((
                SetExpression::SetElementOperation(SetElementOperator::Remove, Box::new(x), y),
                rest,
            )),
            args => Err(ParseErr::Reason(format!(
                "unexpected arguments for `-`: {:?}",
                args
            ))),
        },
        "*" => match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok((
                SetExpression::SetOperation(SetOperator::Intersect, Box::new(x), Box::new(y)),
                rest,
            )),
            args => Err(ParseErr::Reason(format!(
                "unexpected arguments for `*`: {:?}",
                args
            ))),
        },
        op => Err(ParseErr::Reason(format!("no such operator: {}", op))),
    }
}

fn parse_atom(token: &str) -> Result<ArgumentExpression, ParseErr> {
    lazy_static! {
        static ref ELEMENT: Regex = Regex::new(r"^e\[(\d+)\]$").unwrap();
        static ref SET: Regex = Regex::new(r"^s\[(\d+)\]$").unwrap();
        static ref PERMUTATION: Regex = Regex::new(r"^p\[(\d+)\]$").unwrap();
    }

    if let Some(caps) = ELEMENT.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason(format!(
                "could not parse an index of an element variable: {:?}",
                e
            ))
        })?;
        return Ok(ArgumentExpression::Element(ElementExpression::Variable(i)));
    }

    if let Some(caps) = SET.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason(format!(
                "could not parse an index of a set variable: {:?}",
                e
            ))
        })?;
        return Ok(ArgumentExpression::Set(SetExpression::SetVariable(i)));
    }

    if let Some(caps) = PERMUTATION.captures(token) {
        let i: variable::ElementVariable = caps.get(1).unwrap().as_str().parse().map_err(|e| {
            ParseErr::Reason(format!(
                "could not parse an index of a set variable: {:?}",
                e
            ))
        })?;
        return Ok(ArgumentExpression::Set(SetExpression::PermutationVariable(
            i,
        )));
    }

    let n: variable::ElementVariable = token
        .parse()
        .map_err(|e| ParseErr::Reason(format!("could not parse {} as a number: {:?}", token, e)))?;
    Ok(ArgumentExpression::Element(ElementExpression::Constant(n)))
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn parse_argument_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["!", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Set(SetExpression::Complement(_))
        ));
        if let ArgumentExpression::Set(SetExpression::Complement(s)) = expression {
            assert!(matches!(*s, SetExpression::SetVariable(2)));
        }
        assert_eq!(rest, &tokens[2..]);

        let tokens: Vec<String> = ["(", "+", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Set(SetExpression::SetOperation(SetOperator::Union, _, _))
        ));
        if let ArgumentExpression::Set(SetExpression::SetOperation(SetOperator::Union, x, y)) =
            expression
        {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(*y, SetExpression::SetVariable(1)));
        }
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["e[11]", "(", "+", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            ArgumentExpression::Element(ElementExpression::Variable(11))
        ));
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_argument_err() {
        let problem = generate_problem();

        let tokens: Vec<String> = [")", "(", "+", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_complemnt_ok() {
        let problem = generate_problem();
        let tokens: Vec<String> = ["s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_complement(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, SetExpression::Complement(_)));
        if let SetExpression::Complement(s) = expression {
            assert!(matches!(*s, SetExpression::SetVariable(2)));
        }
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_complenent_err() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["e[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_complement(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["n[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_complement(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn pare_set_operation_ok() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["+", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetOperation(SetOperator::Union, _, _)
        ));
        if let SetExpression::SetOperation(SetOperator::Union, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(*y, SetExpression::SetVariable(1)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["-", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetOperation(SetOperator::Difference, _, _)
        ));
        if let SetExpression::SetOperation(SetOperator::Difference, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(*y, SetExpression::SetVariable(1)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["*", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetOperation(SetOperator::Intersect, _, _)
        ));
        if let SetExpression::SetOperation(SetOperator::Intersect, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(*y, SetExpression::SetVariable(1)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["+", "s[2]", "e[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetElementOperation(SetElementOperator::Add, _, _)
        ));
        if let SetExpression::SetElementOperation(SetElementOperator::Add, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(y, ElementExpression::Variable(1)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["-", "s[2]", "1", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            SetExpression::SetElementOperation(SetElementOperator::Remove, _, _)
        ));
        if let SetExpression::SetElementOperation(SetElementOperator::Remove, x, y) = expression {
            assert!(matches!(*x, SetExpression::SetVariable(2)));
            assert!(matches!(y, ElementExpression::Constant(1)));
        }
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn pare_set_operation_err() {
        let problem = generate_problem();

        let tokens: Vec<String> = ["+", "s[2]", "n[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["-", "s[2]", "n[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["*", "s[2]", "e[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());

        let tokens: Vec<String> = ["/", "s[2]", "s[1]", ")", "e[0]", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_operation(&tokens, &problem);
        assert!(result.is_err());
    }

    #[test]
    fn parse_atom_ok() {
        let token = "e[11]";
        let result = parse_atom(token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ArgumentExpression::Element(ElementExpression::Variable(11))
        ));

        let token = "s[11]";
        let result = parse_atom(token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ArgumentExpression::Set(SetExpression::SetVariable(11))
        ));

        let token = "p[11]";
        let result = parse_atom(token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ArgumentExpression::Set(SetExpression::PermutationVariable(11))
        ));

        let token = "11";
        let result = parse_atom(token);
        assert!(result.is_ok());
        assert!(matches!(
            result.unwrap(),
            ArgumentExpression::Element(ElementExpression::Constant(11))
        ));
    }

    #[test]
    fn parse_atom_err() {
        let token = "n[11]";
        let result = parse_atom(token);
        assert!(result.is_err());
        let token = "s[11";
        let result = parse_atom(token);
        assert!(result.is_err());
        let token = "ss[11]";
        let result = parse_atom(token);
        assert!(result.is_err());
        let token = "e[11]]";
        let result = parse_atom(token);
        assert!(result.is_err());
    }
}
