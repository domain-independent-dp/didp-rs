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

type NumericTableParsingResult<'a, T> = Option<(NumericTableExpression<T>, &'a [String])>;

pub fn parse_expression<'a, 'b, 'c, T: variable::Numeric>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    tables: &'b table_registry::TableData<T>,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<NumericTableParsingResult<'a, T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
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

fn parse_table_1d<'a, 'b, 'c, T: variable::Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericTableExpression<T>, &'a [String]), ParseErr> {
    let (x, rest) = set_parser::parse_argument(tokens, metadata, parameters)?;
    let rest = util::parse_closing(rest)?;
    match x {
        ArgumentExpression::Element(x) => Ok((NumericTableExpression::Table1D(i, x), rest)),
        ArgumentExpression::Set(x) => Ok((NumericTableExpression::Table1DSum(i, x), rest)),
    }
}

fn parse_table_2d<'a, 'b, 'c, T: variable::Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericTableExpression<T>, &'a [String]), ParseErr> {
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

fn parse_table_3d<'a, 'b, 'c, T: variable::Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericTableExpression<T>, &'a [String]), ParseErr> {
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

fn parse_table<'a, 'b, 'c, T: variable::Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(NumericTableExpression<T>, &'a [String]), ParseErr> {
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
        let object_names = vec![String::from("object")];
        let object_numbers = vec![10];
        let mut name_to_object = HashMap::new();
        name_to_object.insert(String::from("object"), 0);

        let set_variable_names = vec![
            String::from("s0"),
            String::from("s1"),
            String::from("s2"),
            String::from("s3"),
        ];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert(String::from("s0"), 0);
        name_to_set_variable.insert(String::from("s1"), 1);
        name_to_set_variable.insert(String::from("s2"), 2);
        name_to_set_variable.insert(String::from("s3"), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let vector_variable_names = vec![
            String::from("p0"),
            String::from("p1"),
            String::from("p2"),
            String::from("p3"),
        ];
        let mut name_to_vector_variable = HashMap::new();
        name_to_vector_variable.insert(String::from("p0"), 0);
        name_to_vector_variable.insert(String::from("p1"), 1);
        name_to_vector_variable.insert(String::from("p2"), 2);
        name_to_vector_variable.insert(String::from("p3"), 3);
        let vector_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            String::from("e0"),
            String::from("e1"),
            String::from("e2"),
            String::from("e3"),
        ];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert(String::from("e0"), 0);
        name_to_element_variable.insert(String::from("e1"), 1);
        name_to_element_variable.insert(String::from("e2"), 2);
        name_to_element_variable.insert(String::from("e3"), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let integer_variable_names = vec![
            String::from("i0"),
            String::from("i1"),
            String::from("i2"),
            String::from("i3"),
        ];
        let mut name_to_integer_variable = HashMap::new();
        name_to_integer_variable.insert(String::from("i0"), 0);
        name_to_integer_variable.insert(String::from("i1"), 1);
        name_to_integer_variable.insert(String::from("i2"), 2);
        name_to_integer_variable.insert(String::from("i3"), 3);

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            vector_variable_names,
            name_to_vector_variable,
            vector_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            integer_variable_names,
            name_to_integer_variable,
            ..Default::default()
        }
    }

    fn generate_parameters() -> collections::HashMap<String, usize> {
        let mut parameters = collections::HashMap::new();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    fn generate_tables() -> table_registry::TableData<variable::Integer> {
        let mut name_to_constant = collections::HashMap::new();
        name_to_constant.insert(String::from("f0"), 0);

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

        table_registry::TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        }
    }

    #[test]
    fn parse_expression_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tables = generate_tables();

        let tokens: Vec<String> = ["i0", "1", ")", "i1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("max", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_1d_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tables = generate_tables();

        let tokens: Vec<String> = ["e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &tables, &parameters);
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

        let tokens: Vec<String> = ["s0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &tables, &parameters);
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

        let result = parse_expression("f0", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_1d_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tables = generate_tables();

        let tokens: Vec<String> = ["i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["e0", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "i0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression("f1", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_2d_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tables = generate_tables();

        let tokens: Vec<String> = ["0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
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

        let tokens: Vec<String> = ["s0", "p0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
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
            assert!(matches!(y, SetExpression::VectorVariable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["s0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
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

        let tokens: Vec<String> = ["0", "p0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
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
            assert!(matches!(y, SetExpression::VectorVariable(0)));
        }
        assert_eq!(rest, &tokens[3..]);

        let result = parse_expression("f0", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_2d_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tables = generate_tables();

        let tokens: Vec<String> = ["0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "e0", "p0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_3d_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tables = generate_tables();

        let tokens: Vec<String> = ["0", "1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
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

        let tokens: Vec<String> = ["s0", "s1", "p0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
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
            assert!(matches!(z, SetExpression::VectorVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
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

        let tokens: Vec<String> = ["0", "s1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
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

        let tokens: Vec<String> = ["0", "1", "p0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
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
            assert!(matches!(z, SetExpression::VectorVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["s0", "s1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
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

        let tokens: Vec<String> = ["s0", "1", "p0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
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
            assert!(matches!(z, SetExpression::VectorVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["0", "s1", "p0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
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
            assert!(matches!(z, SetExpression::VectorVariable(0)));
        }
        assert_eq!(rest, &tokens[4..]);

        let result = parse_expression("f0", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_3d_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tables = generate_tables();

        let tokens: Vec<String> = ["0", "1", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "1", "e0", "2", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tables = generate_tables();
        let tokens: Vec<String> = ["s2", "1", "e0", "p3", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f4", &tokens, &metadata, &tables, &parameters);
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
                ArgumentExpression::Set(SetExpression::VectorVariable(3))
            ));
        }
        assert_eq!(rest, &tokens[5..]);

        let result = parse_expression("f0", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let tables = generate_tables();

        let tokens: Vec<String> = ["s2", "1", "e0", "p3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f4", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["s2", "1", "e0", "p3", "i0"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression("f4", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());
    }
}
