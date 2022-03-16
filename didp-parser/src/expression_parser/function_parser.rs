use super::set_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{ArgumentExpression, FunctionExpression};
use crate::function_registry;
use crate::state;
use crate::variable;
use std::collections;
use std::fmt;
use std::str;

pub fn parse_expression<'a, 'b, 'c, T: variable::Numeric>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b function_registry::FunctionRegistry<T>,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<Option<(FunctionExpression, &'a [String])>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    if let Some(i) = registry.name_to_function_1d.get(name) {
        let result = parse_function_1d(*i, tokens, metadata, parameters)?;
        Ok(Some(result))
    } else if let Some(i) = registry.name_to_function_2d.get(name) {
        let result = parse_function_2d(*i, tokens, metadata, parameters)?;
        Ok(Some(result))
    } else if let Some(i) = registry.name_to_function_3d.get(name) {
        let result = parse_function_3d(*i, tokens, metadata, parameters)?;
        Ok(Some(result))
    } else if let Some(i) = registry.name_to_function.get(name) {
        let result = parse_function(*i, tokens, metadata, parameters)?;
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

fn parse_function_1d<'a, 'b, 'c>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(FunctionExpression, &'a [String]), ParseErr> {
    let (x, rest) = set_parser::parse_argument(tokens, metadata, parameters)?;
    let rest = util::parse_closing(rest)?;
    match x {
        ArgumentExpression::Element(x) => Ok((FunctionExpression::Function1D(i, x), rest)),
        ArgumentExpression::Set(x) => Ok((FunctionExpression::Function1DSum(i, x), rest)),
    }
}

fn parse_function_2d<'a, 'b, 'c>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(FunctionExpression, &'a [String]), ParseErr> {
    let (x, rest) = set_parser::parse_argument(tokens, metadata, parameters)?;
    let (y, rest) = set_parser::parse_argument(rest, metadata, parameters)?;
    let rest = util::parse_closing(rest)?;
    match (x, y) {
        (ArgumentExpression::Element(x), ArgumentExpression::Element(y)) => {
            Ok((FunctionExpression::Function2D(i, x, y), rest))
        }
        (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => {
            Ok((FunctionExpression::Function2DSum(i, x, y), rest))
        }
        (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => {
            Ok((FunctionExpression::Function2DSumX(i, x, y), rest))
        }
        (ArgumentExpression::Element(x), ArgumentExpression::Set(y)) => {
            Ok((FunctionExpression::Function2DSumY(i, x, y), rest))
        }
    }
}

fn parse_function_3d<'a, 'b, 'c>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(FunctionExpression, &'a [String]), ParseErr> {
    let (x, rest) = set_parser::parse_argument(tokens, metadata, parameters)?;
    let (y, rest) = set_parser::parse_argument(rest, metadata, parameters)?;
    let (z, rest) = set_parser::parse_argument(rest, metadata, parameters)?;
    let rest = util::parse_closing(rest)?;
    match (x, y, z) {
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Element(z),
        ) => Ok((FunctionExpression::Function3D(i, x, y, z), rest)),
        (ArgumentExpression::Set(x), ArgumentExpression::Set(y), ArgumentExpression::Set(z)) => {
            Ok((FunctionExpression::Function3DSum(i, x, y, z), rest))
        }
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Element(z),
        ) => Ok((FunctionExpression::Function3DSumX(i, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Element(z),
        ) => Ok((FunctionExpression::Function3DSumY(i, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Set(z),
        ) => Ok((FunctionExpression::Function3DSumZ(i, x, y, z), rest)),
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Element(z),
        ) => Ok((FunctionExpression::Function3DSumXY(i, x, y, z), rest)),
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Set(z),
        ) => Ok((FunctionExpression::Function3DSumXZ(i, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Set(z),
        ) => Ok((FunctionExpression::Function3DSumYZ(i, x, y, z), rest)),
    }
}

fn parse_function<'a, 'b, 'c>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(FunctionExpression, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((FunctionExpression::FunctionSum(i, args), rest));
        }
        let (expression, new_xs) = set_parser::parse_argument(xs, metadata, parameters)?;
        args.push(expression);
        xs = new_xs;
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
        let functions_1d = vec![numeric_function::NumericFunction1D::new(Vec::new())];
        let mut name_to_function_1d = HashMap::new();
        name_to_function_1d.insert(String::from("f1"), 0);

        let functions_2d = vec![numeric_function::NumericFunction2D::new(Vec::new())];
        let mut name_to_function_2d = HashMap::new();
        name_to_function_2d.insert(String::from("f2"), 0);

        let functions_3d = vec![numeric_function::NumericFunction3D::new(Vec::new())];
        let mut name_to_function_3d = HashMap::new();
        name_to_function_3d.insert(String::from("f3"), 0);

        let functions = vec![numeric_function::NumericFunction::new(HashMap::new(), 0)];
        let mut name_to_function = HashMap::new();
        name_to_function.insert(String::from("f4"), 0);

        function_registry::FunctionRegistry {
            functions_1d,
            name_to_function_1d,
            functions_2d,
            name_to_function_2d,
            functions_3d,
            name_to_function_3d,
            functions,
            name_to_function,
        }
    }

    #[test]
    fn parse_expression_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["n0", "1", ")", "n1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("max", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_function_1d_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, FunctionExpression::Function1D(_, _)));
        if let FunctionExpression::Function1D(i, x) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[2..]);

        let tokens: Vec<String> = ["s0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function1DSum(_, _)
        ));
        if let FunctionExpression::Function1DSum(i, x) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, SetExpression::SetVariable(0)));
        }
        assert_eq!(rest, &tokens[2..]);
    }

    #[test]
    fn parse_function_1d_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["e0", "0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "n0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression("f1", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_2d_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["0", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function2D(_, _, _)
        ));
        if let FunctionExpression::Function2D(i, x, y) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["s0", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function2DSum(_, _, _)
        ));
        if let FunctionExpression::Function2DSum(i, x, y) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["s0", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function2DSumX(_, _, _)
        ));
        if let FunctionExpression::Function2DSumX(i, x, y) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["0", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function2DSumY(_, _, _)
        ));
        if let FunctionExpression::Function2DSumY(i, x, y) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_function_2d_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["0", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "e0", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_3d_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["0", "1", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3D(_, _, _, _)
        ));
        if let FunctionExpression::Function3D(i, x, y, z) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "s1", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSum(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSum(i, x, y, z) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "1", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumX(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumX(i, x, y, z) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "s1", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumY(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumY(i, x, y, z) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "1", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumZ(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumZ(i, x, y, z) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "s1", "e0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumXY(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumXY(i, x, y, z) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, ElementExpression::Variable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "1", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumXZ(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumXZ(i, x, y, z) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, SetExpression::SetVariable(0)));
            assert!(matches!(y, ElementExpression::Constant(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "s1", "p0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            FunctionExpression::Function3DSumYZ(_, _, _, _)
        ));
        if let FunctionExpression::Function3DSumYZ(i, x, y, z) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_function_3d_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["0", "1", "n0", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "1", "e0", "2", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "1", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_function_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();
        let tokens: Vec<String> = ["s2", "1", "e0", "p3", ")", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f4", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, FunctionExpression::FunctionSum(_, _)));
        if let FunctionExpression::FunctionSum(i, args) = expression {
            assert_eq!(i, 0);
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
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["s2", "1", "e0", "p3", "n0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f4", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["s2", "1", "e0", "p3", "n0"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f4", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
