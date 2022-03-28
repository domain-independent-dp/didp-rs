use super::element_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{ArgumentExpression, NumericTableExpression};
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::Numeric;
use crate::StateMetadata;
use rustc_hash::FxHashMap;
use std::fmt;
use std::str;

type NumericTableParsingResult<'a, T> = Option<(NumericTableExpression<T>, &'a [String])>;

pub fn parse_expression<'a, 'b, 'c, T: Numeric>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
    tables: &'b TableData<T>,
) -> Result<NumericTableParsingResult<'a, T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) = element_parser::parse_expression(tokens, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((NumericTableExpression::Table1D(*i, x), rest)))
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) = element_parser::parse_expression(tokens, metadata, registry, parameters)?;
        let (y, rest) = element_parser::parse_expression(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((NumericTableExpression::Table2D(*i, x, y), rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let result = parse_table(*i, tokens, metadata, registry, parameters)?;
        Ok(Some(result))
    } else if name == "sum" {
        let (name, rest) = tokens
            .split_first()
            .ok_or_else(|| ParseErr::new(String::from("could not get token")))?;
        parse_sum(name, rest, metadata, registry, parameters, tables)
    } else {
        Ok(None)
    }
}

fn parse_table<'a, 'b, 'c, T: Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(NumericTableExpression<T>, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((NumericTableExpression::Table(i, args), rest));
        }
        let (expression, new_xs) =
            element_parser::parse_expression(xs, metadata, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

fn parse_sum<'a, 'b, 'c, T: Numeric>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
    tables: &'b TableData<T>,
) -> Result<NumericTableParsingResult<'a, T>, ParseErr> {
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) = parse_argument(tokens, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        match x {
            ArgumentExpression::Set(x) => {
                Ok(Some((NumericTableExpression::Table1DSum(*i, x), rest)))
            }
            ArgumentExpression::Vector(x) => Ok(Some((
                NumericTableExpression::Table1DVectorSum(*i, x),
                rest,
            ))),
            _ => Err(ParseErr::new(format!(
                "argument `{:?}` is invalid for sum",
                name
            ))),
        }
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) = parse_argument(tokens, metadata, registry, parameters)?;
        let (y, rest) = parse_argument(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => {
                Ok(Some((NumericTableExpression::Table2DSum(*i, x, y), rest)))
            }
            (ArgumentExpression::Vector(x), ArgumentExpression::Vector(y)) => Ok(Some((
                NumericTableExpression::Table2DVectorSum(*i, x, y),
                rest,
            ))),
            (ArgumentExpression::Set(x), ArgumentExpression::Vector(y)) => Ok(Some((
                NumericTableExpression::Table2DSetVectorSum(*i, x, y),
                rest,
            ))),
            (ArgumentExpression::Vector(x), ArgumentExpression::Set(y)) => Ok(Some((
                NumericTableExpression::Table2DVectorSetSum(*i, x, y),
                rest,
            ))),
            (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => {
                Ok(Some((NumericTableExpression::Table2DSumX(*i, x, y), rest)))
            }
            (ArgumentExpression::Element(x), ArgumentExpression::Set(y)) => {
                Ok(Some((NumericTableExpression::Table2DSumY(*i, x, y), rest)))
            }
            (ArgumentExpression::Vector(x), ArgumentExpression::Element(y)) => Ok(Some((
                NumericTableExpression::Table2DVectorSumX(*i, x, y),
                rest,
            ))),
            (ArgumentExpression::Element(x), ArgumentExpression::Vector(y)) => Ok(Some((
                NumericTableExpression::Table2DVectorSumY(*i, x, y),
                rest,
            ))),
            (x, y) => Err(ParseErr::new(format!(
                "arguments `{:?}` `{:?}` are invalid for sum",
                x, y
            ))),
        }
    } else if let Some(i) = tables.name_to_table.get(name) {
        Ok(Some(parse_table_sum(
            *i, tokens, metadata, registry, parameters,
        )?))
    } else {
        Ok(None)
    }
}

fn parse_table_sum<'a, 'b, 'c, T: Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b StateMetadata,
    registry: &'b TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
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
        let (expression, new_xs) = parse_argument(xs, metadata, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

fn parse_argument<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(ArgumentExpression, &'a [String]), ParseErr> {
    if let Ok((element, rest)) =
        element_parser::parse_expression(tokens, metadata, registry, parameters)
    {
        Ok((ArgumentExpression::Element(element), rest))
    } else if let Ok((set, rest)) =
        element_parser::parse_set_expression(tokens, metadata, registry, parameters)
    {
        Ok((ArgumentExpression::Set(set), rest))
    } else if let Ok((vector, rest)) =
        element_parser::parse_vector_expression(tokens, metadata, registry, parameters)
    {
        Ok((ArgumentExpression::Vector(vector), rest))
    } else {
        Err(ParseErr::new(format!(
            "could not parse tokens `{:?}`",
            tokens
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::table;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec![String::from("object")];
        let object_numbers = vec![10];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("object"), 0);

        let set_variable_names = vec![
            String::from("s0"),
            String::from("s1"),
            String::from("s2"),
            String::from("s3"),
        ];
        let mut name_to_set_variable = FxHashMap::default();
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
        let mut name_to_vector_variable = FxHashMap::default();
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
        let mut name_to_element_variable = FxHashMap::default();
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
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("i0"), 0);
        name_to_integer_variable.insert(String::from("i1"), 1);
        name_to_integer_variable.insert(String::from("i2"), 2);
        name_to_integer_variable.insert(String::from("i3"), 3);

        StateMetadata {
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

    fn generate_parameters() -> FxHashMap<String, usize> {
        let mut parameters = FxHashMap::default();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), 0);

        let tables_1d = vec![table::Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let tables = vec![table::Table::new(FxHashMap::default(), 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        TableRegistry {
            integer_tables: TableData {
                name_to_constant,
                tables_1d,
                name_to_table_1d,
                tables_2d,
                name_to_table_2d,
                tables_3d,
                name_to_table_3d,
                tables,
                name_to_table,
            },
            ..Default::default()
        }
    }

    #[test]
    fn parse_expression_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["i0", "1", ")", "i1", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_1d_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f1",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericTableExpression::Table1D(0, ElementExpression::Variable(0)),
        );
        assert_eq!(rest, &tokens[2..]);

        let tokens: Vec<String> = ["f1", "s0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericTableExpression::Table1DSum(
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[3..]);

        let result = parse_expression(
            "f0",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_1d_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f1",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f1", "e0", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_2d_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f2",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericTableExpression::Table2D(
                0,
                ElementExpression::Constant(0),
                ElementExpression::Variable(0)
            )
        );
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["f2", "s0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericTableExpression::Table2DSum(
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "s0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericTableExpression::Table2DSumX(
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0)
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "0", "s0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericTableExpression::Table2DSumY(
                0,
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let result = parse_expression(
            "f0",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_2d_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["0", "i0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f2",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f2", "0", "e0", "p0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f2", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();
        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "p3", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            NumericTableExpression::TableSum(
                0,
                vec![
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(2)
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(1)),
                    ArgumentExpression::Element(ElementExpression::Variable(0)),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Variable(3)
                    )),
                ]
            )
        );
        assert_eq!(rest, &tokens[6..]);

        let result = parse_expression(
            "f0",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "p3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "p3", "i0"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }
}
