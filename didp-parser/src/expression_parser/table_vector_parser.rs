use super::element_parser;
use super::numeric_table_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{ArgumentExpression, TableVectorExpression, VectorOrElementExpression};
use crate::state::StateMetadata;
use crate::table_data::TableData;
use crate::table_registry::TableRegistry;
use crate::variable::{Element, Numeric};
use rustc_hash::FxHashMap;
use std::str;

type NumericVectorTableReturnType<'a, T> = Option<(TableVectorExpression<T>, &'a [String])>;

pub fn parse_expression<'a, T: Numeric>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
    tables: &TableData<T>,
) -> Result<NumericVectorTableReturnType<'a, T>, ParseErr> {
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) =
            element_parser::parse_vector_expression(tokens, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((TableVectorExpression::Table1D(*i, x), rest)))
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) = parse_vector_or_element(tokens, metadata, registry, parameters)?;
        let (y, rest) = parse_vector_or_element(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        match (x, y) {
            (VectorOrElementExpression::Vector(x), VectorOrElementExpression::Vector(y)) => {
                Ok(Some((TableVectorExpression::Table2D(*i, x, y), rest)))
            }
            (VectorOrElementExpression::Vector(x), VectorOrElementExpression::Element(y)) => {
                Ok(Some((TableVectorExpression::Table2DX(*i, x, y), rest)))
            }
            (VectorOrElementExpression::Element(x), VectorOrElementExpression::Vector(y)) => {
                Ok(Some((TableVectorExpression::Table2DY(*i, x, y), rest)))
            }
            (x, y) => Err(ParseErr::new(format!(
                "arguments `{:?}` `{:?}` are invalid for `{}`",
                x, y, name
            ))),
        }
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (expression, rest) = parse_table(*i, tokens, metadata, registry, parameters)?;
        Ok(Some((expression, rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let (expression, rest) = parse_table(*i, tokens, metadata, registry, parameters)?;
        Ok(Some((expression, rest)))
    } else {
        Ok(None)
    }
}

fn parse_table<'a, T: Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(TableVectorExpression<T>, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((TableVectorExpression::Table(i, args), rest));
        }
        let (expression, new_xs) = parse_vector_or_element(xs, metadata, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

fn parse_vector_or_element<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(VectorOrElementExpression, &'a [String]), ParseErr> {
    if let Ok((element, rest)) =
        element_parser::parse_expression(tokens, metadata, registry, parameters)
    {
        Ok((VectorOrElementExpression::Element(element), rest))
    } else if let Ok((vector, rest)) =
        element_parser::parse_vector_expression(tokens, metadata, registry, parameters)
    {
        Ok((VectorOrElementExpression::Vector(vector), rest))
    } else {
        Err(ParseErr::new(format!(
            "could not parse tokens `{:?}`",
            tokens
        )))
    }
}

pub fn parse_sum_expression<'a, T: Numeric>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, Element>,
    tables: &TableData<T>,
) -> Result<NumericVectorTableReturnType<'a, T>, ParseErr> {
    if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) =
            numeric_table_parser::parse_argument(tokens, metadata, registry, parameters)?;
        let (y, rest) = numeric_table_parser::parse_argument(rest, metadata, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        match (x, y) {
            (ArgumentExpression::Vector(x), ArgumentExpression::Set(y)) => {
                Ok(Some((TableVectorExpression::Table2DXSum(*i, x, y), rest)))
            }
            (ArgumentExpression::Set(x), ArgumentExpression::Vector(y)) => {
                Ok(Some((TableVectorExpression::Table2DYSum(*i, x, y), rest)))
            }
            (x, y) => Err(ParseErr::new(format!(
                "arguments `{:?}` `{:?}` are invalid for `{}`",
                x, y, name
            ))),
        }
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (expression, rest) = parse_table_sum(*i, tokens, metadata, registry, parameters)?;
        Ok(Some((expression, rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let (expression, rest) = parse_table_sum(*i, tokens, metadata, registry, parameters)?;
        Ok(Some((expression, rest)))
    } else {
        Ok(None)
    }
}

fn parse_table_sum<'a, T: Numeric>(
    i: usize,
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(TableVectorExpression<T>, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((TableVectorExpression::TableSum(i, args), rest));
        }
        let (expression, new_xs) =
            numeric_table_parser::parse_argument(xs, metadata, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::table;
    use crate::table_data;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec![String::from("something")];
        let object_numbers = vec![10];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert(String::from("something"), 0);

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
            String::from("v0"),
            String::from("v1"),
            String::from("v2"),
            String::from("v3"),
        ];
        let mut name_to_vector_variable = FxHashMap::default();
        name_to_vector_variable.insert(String::from("v0"), 0);
        name_to_vector_variable.insert(String::from("v1"), 1);
        name_to_vector_variable.insert(String::from("v2"), 2);
        name_to_vector_variable.insert(String::from("v3"), 3);
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

        let integer_resource_variable_names = vec![
            String::from("ir0"),
            String::from("ir1"),
            String::from("ir2"),
            String::from("ir3"),
        ];
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert(String::from("ir0"), 0);
        name_to_integer_resource_variable.insert(String::from("ir1"), 1);
        name_to_integer_resource_variable.insert(String::from("ir2"), 2);
        name_to_integer_resource_variable.insert(String::from("ir3"), 3);

        let continuous_variable_names = vec![
            String::from("c0"),
            String::from("c1"),
            String::from("c2"),
            String::from("c3"),
        ];
        let mut name_to_continuous_variable = FxHashMap::default();
        name_to_continuous_variable.insert(String::from("c0"), 0);
        name_to_continuous_variable.insert(String::from("c1"), 1);
        name_to_continuous_variable.insert(String::from("c2"), 2);
        name_to_continuous_variable.insert(String::from("c3"), 3);

        let continuous_resource_variable_names = vec![
            String::from("cr0"),
            String::from("cr1"),
            String::from("cr2"),
            String::from("cr3"),
        ];
        let mut name_to_continuous_resource_variable = FxHashMap::default();
        name_to_continuous_resource_variable.insert(String::from("cr0"), 0);
        name_to_continuous_resource_variable.insert(String::from("cr1"), 1);
        name_to_continuous_resource_variable.insert(String::from("cr2"), 2);
        name_to_continuous_resource_variable.insert(String::from("cr3"), 3);

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
            continuous_variable_names,
            name_to_continuous_variable,
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true, false],
            continuous_resource_variable_names,
            name_to_continuous_resource_variable,
            continuous_less_is_better: vec![false, false, true, false],
        }
    }

    fn generate_parameters() -> FxHashMap<String, Element> {
        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("param"), 0);
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

        let integer_tables = table_data::TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("cf0"), 0.0);

        let tables_1d = vec![table::Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("cf1"), 0);

        let tables_2d = vec![table::Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("cf2"), 0);

        let tables_3d = vec![table::Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("cf3"), 0);

        let tables = vec![table::Table::new(FxHashMap::default(), 0.0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("cf4"), 0);

        let continuous_tables = table_data::TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            tables_2d,
            name_to_table_2d,
            tables_3d,
            name_to_table_3d,
            tables,
            name_to_table,
        };

        TableRegistry {
            integer_tables,
            continuous_tables,
            ..Default::default()
        }
    }

    #[test]
    fn parse_integer_vector_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f5",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());

        let tokens: Vec<String> = ["(", "vector", "0", "1", ")", ")", ")"]
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
            TableVectorExpression::Table1D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[6..]);

        let tokens: Vec<String> = [
            "(", "vector", "0", "1", ")", "(", "vector", "0", "1", ")", ")", ")",
        ]
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
            TableVectorExpression::Table2D(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1]))
            )
        );
        assert_eq!(rest, &tokens[11..]);

        let tokens: Vec<String> = ["(", "vector", "0", "1", ")", "1", ")", ")"]
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
            TableVectorExpression::Table2DX(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                ElementExpression::Constant(1),
            )
        );
        assert_eq!(rest, &tokens[7..]);

        let tokens: Vec<String> = ["1", "(", "vector", "0", "1", ")", ")", ")"]
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
            TableVectorExpression::Table2DY(
                0,
                ElementExpression::Constant(1),
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            )
        );
        assert_eq!(rest, &tokens[7..]);

        let tokens: Vec<String> = ["0", "(", "vector", "0", "1", ")", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f3",
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
            TableVectorExpression::Table(
                0,
                vec![
                    VectorOrElementExpression::Element(ElementExpression::Constant(0)),
                    VectorOrElementExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    VectorOrElementExpression::Element(ElementExpression::Constant(1)),
                ]
            )
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_expression_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0", ")", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            "f1",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "1", ")", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            "f2",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["s0", "v0", ")", ")"]
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

        let tokens: Vec<String> = ["s0", "v0", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f3",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_sum_expression_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "vector", "0", "1", ")", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_sum_expression(
            "f6",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());

        let tokens: Vec<String> = ["(", "vector", "0", "1", ")", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_sum_expression(
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
            TableVectorExpression::Table2DXSum(
                0,
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
                SetExpression::Reference(ReferenceExpression::Variable(0)),
            )
        );
        assert_eq!(rest, &tokens[7..]);

        let tokens: Vec<String> = ["s0", "(", "vector", "0", "1", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_sum_expression(
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
            TableVectorExpression::Table2DYSum(
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                VectorExpression::Reference(ReferenceExpression::Constant(vec![0, 1])),
            )
        );
        assert_eq!(rest, &tokens[7..]);

        let tokens: Vec<String> = ["s0", "(", "vector", "0", "1", ")", "1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_sum_expression(
            "f3",
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
            TableVectorExpression::TableSum(
                0,
                vec![
                    ArgumentExpression::Set(SetExpression::Reference(
                        ReferenceExpression::Variable(0)
                    )),
                    ArgumentExpression::Vector(VectorExpression::Reference(
                        ReferenceExpression::Constant(vec![0, 1])
                    )),
                    ArgumentExpression::Element(ElementExpression::Constant(1)),
                ]
            )
        );
        assert_eq!(rest, &tokens[8..]);
    }

    #[test]
    fn parse_sum_expression_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0", "1", ")", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_sum_expression(
            "f2",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["e0", "v0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_sum_expression(
            "f2",
            &tokens,
            &metadata,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }
}
