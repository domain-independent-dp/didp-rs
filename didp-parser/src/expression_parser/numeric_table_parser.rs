use super::set_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{ArgumentExpression, NumericTableExpression};
use crate::state;
use crate::table_registry;
use crate::variable;
use std::collections;
use std::fmt;
use std::str;

pub fn parse_expression<'a, 'b, 'c, T: variable::Numeric>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b table_registry::TableRegistry<T>,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<Option<(NumericTableExpression, &'a [String])>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let tables = &registry.numeric_tables;
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let result = parse_table_1d(*i, tokens, metadata, parameters)?;
        Ok(Some(result))
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let result = parse_table_2d(*i, tokens, metadata, parameters)?;
        Ok(Some(result))
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let result = parse_table_3d(*i, tokens, metadata, parameters)?;
        Ok(Some(result))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let result = parse_table(*i, tokens, metadata, parameters)?;
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

fn parse_table_1d<'a, 'b, 'c>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericTableExpression, &'a [String]), ParseErr> {
    let (x, rest) = set_parser::parse_argument(tokens, metadata, parameters)?;
    let rest = util::parse_closing(rest)?;
    match x {
        ArgumentExpression::Element(x) => Ok((NumericTableExpression::Table1D(i, x), rest)),
        ArgumentExpression::Set(x) => Ok((NumericTableExpression::Table1DSum(i, x), rest)),
    }
}

fn parse_table_2d<'a, 'b, 'c>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericTableExpression, &'a [String]), ParseErr> {
    let (x, rest) = set_parser::parse_argument(tokens, metadata, parameters)?;
    let (y, rest) = set_parser::parse_argument(rest, metadata, parameters)?;
    let rest = util::parse_closing(rest)?;
    match (x, y) {
        (ArgumentExpression::Element(x), ArgumentExpression::Element(y)) => {
            Ok((NumericTableExpression::Table2D(i, x, y), rest))
        }
        (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => {
            Ok((NumericTableExpression::Table2DSum(i, x, y), rest))
        }
        (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => {
            Ok((NumericTableExpression::Table2DSumX(i, x, y), rest))
        }
        (ArgumentExpression::Element(x), ArgumentExpression::Set(y)) => {
            Ok((NumericTableExpression::Table2DSumY(i, x, y), rest))
        }
    }
}

fn parse_table_3d<'a, 'b, 'c>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericTableExpression, &'a [String]), ParseErr> {
    let (x, rest) = set_parser::parse_argument(tokens, metadata, parameters)?;
    let (y, rest) = set_parser::parse_argument(rest, metadata, parameters)?;
    let (z, rest) = set_parser::parse_argument(rest, metadata, parameters)?;
    let rest = util::parse_closing(rest)?;
    match (x, y, z) {
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Element(z),
        ) => Ok((NumericTableExpression::Table3D(i, x, y, z), rest)),
        (ArgumentExpression::Set(x), ArgumentExpression::Set(y), ArgumentExpression::Set(z)) => {
            Ok((NumericTableExpression::Table3DSum(i, x, y, z), rest))
        }
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Element(z),
        ) => Ok((NumericTableExpression::Table3DSumX(i, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Element(z),
        ) => Ok((NumericTableExpression::Table3DSumY(i, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Set(z),
        ) => Ok((NumericTableExpression::Table3DSumZ(i, x, y, z), rest)),
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Element(z),
        ) => Ok((NumericTableExpression::Table3DSumXY(i, x, y, z), rest)),
        (
            ArgumentExpression::Set(x),
            ArgumentExpression::Element(y),
            ArgumentExpression::Set(z),
        ) => Ok((NumericTableExpression::Table3DSumXZ(i, x, y, z), rest)),
        (
            ArgumentExpression::Element(x),
            ArgumentExpression::Set(y),
            ArgumentExpression::Set(z),
        ) => Ok((NumericTableExpression::Table3DSumYZ(i, x, y, z), rest)),
    }
}

fn parse_table<'a, 'b, 'c>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericTableExpression, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((NumericTableExpression::TableSum(i, args), rest));
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

    fn generate_registry() -> table_registry::TableRegistry<variable::IntegerVariable> {
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
            numeric_tables: table_registry::TableData {
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                tables_3d,
                name_to_table_3d,
                tables,
                name_to_table,
            },
            bool_tables: table_registry::TableData {
                tables_1d: Vec::new(),
                name_to_table_1d: HashMap::new(),
                tables_2d: Vec::new(),
                name_to_table_2d: HashMap::new(),
                tables_3d: Vec::new(),
                name_to_table_3d: HashMap::new(),
                tables: Vec::new(),
                name_to_table: HashMap::new(),
            },
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
    fn parse_table_1d_ok() {
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
        assert!(matches!(expression, NumericTableExpression::Table1D(_, _)));
        if let NumericTableExpression::Table1D(i, x) = expression {
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
            NumericTableExpression::Table1DSum(_, _)
        ));
        if let NumericTableExpression::Table1DSum(i, x) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, SetExpression::SetVariable(0)));
        }
        assert_eq!(rest, &tokens[2..]);

        let result = parse_expression("f0", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_1d_err() {
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
    fn parse_table_2d_ok() {
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
            NumericTableExpression::Table2D(_, _, _)
        ));
        if let NumericTableExpression::Table2D(i, x, y) = expression {
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
            NumericTableExpression::Table2DSum(_, _, _)
        ));
        if let NumericTableExpression::Table2DSum(i, x, y) = expression {
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
            NumericTableExpression::Table2DSumX(_, _, _)
        ));
        if let NumericTableExpression::Table2DSumX(i, x, y) = expression {
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
            NumericTableExpression::Table2DSumY(_, _, _)
        ));
        if let NumericTableExpression::Table2DSumY(i, x, y) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let result = parse_expression("f0", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_2d_err() {
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
    fn parse_table_3d_ok() {
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
            NumericTableExpression::Table3D(_, _, _, _)
        ));
        if let NumericTableExpression::Table3D(i, x, y, z) = expression {
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
            NumericTableExpression::Table3DSum(_, _, _, _)
        ));
        if let NumericTableExpression::Table3DSum(i, x, y, z) = expression {
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
            NumericTableExpression::Table3DSumX(_, _, _, _)
        ));
        if let NumericTableExpression::Table3DSumX(i, x, y, z) = expression {
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
            NumericTableExpression::Table3DSumY(_, _, _, _)
        ));
        if let NumericTableExpression::Table3DSumY(i, x, y, z) = expression {
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
            NumericTableExpression::Table3DSumZ(_, _, _, _)
        ));
        if let NumericTableExpression::Table3DSumZ(i, x, y, z) = expression {
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
            NumericTableExpression::Table3DSumXY(_, _, _, _)
        ));
        if let NumericTableExpression::Table3DSumXY(i, x, y, z) = expression {
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
            NumericTableExpression::Table3DSumXZ(_, _, _, _)
        ));
        if let NumericTableExpression::Table3DSumXZ(i, x, y, z) = expression {
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
            NumericTableExpression::Table3DSumYZ(_, _, _, _)
        ));
        if let NumericTableExpression::Table3DSumYZ(i, x, y, z) = expression {
            assert_eq!(i, 0);
            assert!(matches!(x, ElementExpression::Constant(0)));
            assert!(matches!(y, SetExpression::SetVariable(1)));
            assert!(matches!(z, SetExpression::PermutationVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let result = parse_expression("f0", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_3d_err() {
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
    fn parse_table_ok() {
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
        assert!(matches!(expression, NumericTableExpression::TableSum(_, _)));
        if let NumericTableExpression::TableSum(i, args) = expression {
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

        let result = parse_expression("f0", &tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_err() {
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
