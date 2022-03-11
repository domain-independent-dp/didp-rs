use super::environment;
use super::set_parser;
use super::ParseErr;
use crate::expression::{ArgumentExpression, FunctionExpression};
use crate::numeric_function;
use crate::variable;
use std::fmt;
use std::str;

type FunctionParseResult<'a, 'b, T> = Option<(FunctionExpression<'b, T>, &'a [String])>;

pub fn parse_expression<'a, 'b, T: variable::Numeric>(
    name: &'a str,
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<FunctionParseResult<'a, 'b, T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    if let Some(f) = env.functions_1d.get(name) {
        let result = parse_function_1d(f, tokens, env)?;
        Ok(Some(result))
    } else if let Some(f) = env.functions_2d.get(name) {
        let result = parse_function_2d(f, tokens, env)?;
        Ok(Some(result))
    } else if let Some(f) = env.functions_3d.get(name) {
        let result = parse_function_3d(f, tokens, env)?;
        Ok(Some(result))
    } else if let Some(f) = env.functions.get(name) {
        let result = parse_function(f, tokens, env)?;
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

fn parse_function_1d<'a, 'b, T: variable::Numeric>(
    f: &'b numeric_function::NumericFunction1D<T>,
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(FunctionExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (x, rest) = set_parser::parse_argument(tokens, env)?;
    let rest = super::parse_closing(rest)?;
    match x {
        ArgumentExpression::Element(x) => Ok((FunctionExpression::Function1D(f, x), rest)),
        ArgumentExpression::Set(x) => Ok((FunctionExpression::Function1DSum(f, x), rest)),
    }
}

fn parse_function_2d<'a, 'b, T: variable::Numeric>(
    f: &'b numeric_function::NumericFunction2D<T>,
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(FunctionExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (x, rest) = set_parser::parse_argument(tokens, env)?;
    let (y, rest) = set_parser::parse_argument(rest, env)?;
    let rest = super::parse_closing(rest)?;
    match (x, y) {
        (ArgumentExpression::Element(x), ArgumentExpression::Element(y)) => {
            Ok((FunctionExpression::Function2D(&f, x, y), rest))
        }
        (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => {
            Ok((FunctionExpression::Function2DSum(&f, x, y), rest))
        }
        (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => {
            Ok((FunctionExpression::Function2DSumX(&f, x, y), rest))
        }
        (ArgumentExpression::Element(x), ArgumentExpression::Set(y)) => {
            Ok((FunctionExpression::Function2DSumY(&f, x, y), rest))
        }
    }
}

fn parse_function_3d<'a, 'b, T: variable::Numeric>(
    f: &'b numeric_function::NumericFunction3D<T>,
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(FunctionExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (x, rest) = set_parser::parse_argument(tokens, env)?;
    let (y, rest) = set_parser::parse_argument(rest, env)?;
    let (z, rest) = set_parser::parse_argument(rest, env)?;
    let rest = super::parse_closing(rest)?;
    match (x, y, z) {
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Element(z),
        ) => Ok((FunctionExpression::Function3D(&f, x, y, z), rest)),
        (ArgumentExpression::Set(x), ArgumentExpression::Set(y), ArgumentExpression::Set(z)) => {
            Ok((FunctionExpression::Function3DSum(&f, x, y, z), rest))
        }
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Element(z),
        ) => Ok((FunctionExpression::Function3DSumX(&f, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Element(z),
        ) => Ok((FunctionExpression::Function3DSumY(&f, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Set(z),
        ) => Ok((FunctionExpression::Function3DSumZ(&f, x, y, z), rest)),
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Element(z),
        ) => Ok((FunctionExpression::Function3DSumXY(&f, x, y, z), rest)),
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Set(z),
        ) => Ok((FunctionExpression::Function3DSumXZ(&f, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Set(z),
        ) => Ok((FunctionExpression::Function3DSumYZ(&f, x, y, z), rest)),
    }
}

fn parse_function<'a, 'b, T: variable::Numeric>(
    f: &'b numeric_function::NumericFunction<T>,
    tokens: &'a [String],
    env: &'b environment::Environment<T>,
) -> Result<(FunctionExpression<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::Reason("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((FunctionExpression::FunctionSum(f, args), rest));
        }
        let (expression, new_xs) = set_parser::parse_argument(xs, env)?;
        args.push(expression);
        xs = new_xs;
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
    fn parse_expression_ok() {
        let env = generate_env();

        let tokens: Vec<String> = ["n0", "1", ")", "n1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("max", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_function_1d_ok() {
        let env = generate_env();
        let f = &env.functions_1d["f1"];

        let tokens: Vec<String> = ["e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, FunctionExpression::Function1D(_, _)));
        if let FunctionExpression::Function1D(g, x) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[2..]);

        let tokens: Vec<String> = ["s0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function1DSum(_, _)
        ));
        if let FunctionExpression::Function1DSum(g, x) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
        }
        assert_eq!(rest, &tokens[2..]);
    }

    #[test]
    fn parse_function_1d_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["e0", "0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "n0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression("f1", &tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_2d_ok() {
        let env = generate_env();
        let f = &env.functions_2d["f2"];

        let tokens: Vec<String> = ["0", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function2D(_, _, _)
        ));
        if let FunctionExpression::Function2D(g, x, y) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["s0", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function2DSum(_, _, _)
        ));
        if let FunctionExpression::Function2DSum(g, x, y) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["s0", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function2DSumX(_, _, _)
        ));
        if let FunctionExpression::Function2DSumX(g, x, y) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["0", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function2DSumY(_, _, _)
        ));
        if let FunctionExpression::Function2DSumY(g, x, y) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_function_2d_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "e0", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_3d_ok() {
        let env = generate_env();
        let f = &env.functions_3d["f3"];

        let tokens: Vec<String> = ["0", "1", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3D(_, _, _, _)
        ));
        if let FunctionExpression::Function3D(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "s1", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSum(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSum(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "1", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumX(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumX(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "s1", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumY(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumY(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "1", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumZ(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumZ(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "s1", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumXY(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumXY(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "1", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumXZ(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumXZ(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "s1", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumYZ(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumYZ(g, x, y, z) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_function_3d_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["0", "1", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "1", "e0", "2", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "1", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &env);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_ok() {
        let env = generate_env();
        let f = &env.functions["f4"];
        let tokens: Vec<String> = ["s2", "1", "e0", "p3", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f4", &tokens, &env);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, FunctionExpression::FunctionSum(_, _)));
        if let FunctionExpression::FunctionSum(g, args) = expression {
            assert_eq!(g as *const _, f as *const _);
            assert_eq!(args.len(), 4);
            assert!(matches!(
                args[0],
                ArgumentExpression::Set(SetExpression::SetVariable(2))
            ));
            assert!(matches!(
                args[1],
                ArgumentExpression::Element(ElementExpression::Constant(1))
            ));
            assert!(matches!(
                args[2],
                ArgumentExpression::Element(ElementExpression::Variable(0))
            ));
            assert!(matches!(
                args[3],
                ArgumentExpression::Set(SetExpression::PermutationVariable(3))
            ));
        }
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_function_err() {
        let env = generate_env();

        let tokens: Vec<String> = ["s2", "1", "e0", "p3", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f4", &tokens, &env);
        assert!(result.is_err());

        let tokens: Vec<String> = ["s2", "1", "e0", "p3", "n0"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f4", &tokens, &env);
        assert!(result.is_err());
    }
}
