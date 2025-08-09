use super::argument_parser::{parse_argument, parse_multiple_arguments};
use super::element_parser;
use super::util;
use super::util::ParseErr;
use dypdl::expression::{
    ArgumentExpression, ElementExpression, NumericTableExpression, ReduceOperator,
};
use dypdl::variable_type::Numeric;
use dypdl::{StateFunctions, StateMetadata, TableData, TableRegistry};
use rustc_hash::FxHashMap;
use std::fmt;
use std::str;

type NumericTableParsingResult<'a, T> = Option<(NumericTableExpression<T>, &'a [String])>;

pub fn parse_expression<'a, T: Numeric>(
    name: &str,
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
    tables: &TableData<T>,
) -> Result<NumericTableParsingResult<'a, T>, ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) =
            element_parser::parse_expression(tokens, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((NumericTableExpression::Table1D(*i, x), rest)))
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) =
            element_parser::parse_expression(tokens, metadata, functions, registry, parameters)?;
        let (y, rest) =
            element_parser::parse_expression(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((NumericTableExpression::Table2D(*i, x, y), rest)))
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (x, rest) =
            element_parser::parse_expression(tokens, metadata, functions, registry, parameters)?;
        let (y, rest) =
            element_parser::parse_expression(rest, metadata, functions, registry, parameters)?;
        let (z, rest) =
            element_parser::parse_expression(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((NumericTableExpression::Table3D(*i, x, y, z), rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let (args, rest) = parse_args(tokens, metadata, functions, registry, parameters)?;
        Ok(Some((NumericTableExpression::Table(*i, args), rest)))
    } else {
        let op = match name {
            "sum" => ReduceOperator::Sum,
            "product" => ReduceOperator::Product,
            "max" => ReduceOperator::Max,
            "min" => ReduceOperator::Min,
            _ => return Ok(None),
        };
        let (name, rest) = tokens
            .split_first()
            .ok_or_else(|| ParseErr::new(String::from("could not get token")))?;
        let model_data = ModelData {
            metadata,
            functions,
            registry,
        };
        parse_reduce(name, rest, op, model_data, parameters, tables)
    }
}

fn parse_args<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    functions: &StateFunctions,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(Vec<ElementExpression>, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((args, rest));
        }
        let (expression, new_xs) =
            element_parser::parse_expression(xs, metadata, functions, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

struct ModelData<'a> {
    metadata: &'a StateMetadata,
    functions: &'a StateFunctions,
    registry: &'a TableRegistry,
}

fn parse_reduce<'a, T: Numeric>(
    name: &str,
    tokens: &'a [String],
    op: ReduceOperator,
    model_data: ModelData,
    parameters: &FxHashMap<String, usize>,
    tables: &TableData<T>,
) -> Result<NumericTableParsingResult<'a, T>, ParseErr> {
    let metadata = model_data.metadata;
    let functions = model_data.functions;
    let registry = model_data.registry;

    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) = parse_argument(tokens, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        match x {
            ArgumentExpression::Set(x) => Ok(Some((
                NumericTableExpression::Table1DReduce(op, *i, x),
                rest,
            ))),
            ArgumentExpression::Vector(x) => Ok(Some((
                NumericTableExpression::Table1DVectorReduce(op, *i, x),
                rest,
            ))),
            _ => Err(ParseErr::new(format!(
                "argument `{name:?}` is invalid for sum",
            ))),
        }
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) = parse_argument(tokens, metadata, functions, registry, parameters)?;
        let (y, rest) = parse_argument(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        match (x, y) {
            (ArgumentExpression::Set(x), ArgumentExpression::Set(y)) => Ok(Some((
                NumericTableExpression::Table2DReduce(op, *i, x, y),
                rest,
            ))),
            (ArgumentExpression::Vector(x), ArgumentExpression::Vector(y)) => Ok(Some((
                NumericTableExpression::Table2DVectorReduce(op, *i, x, y),
                rest,
            ))),
            (ArgumentExpression::Set(x), ArgumentExpression::Vector(y)) => Ok(Some((
                NumericTableExpression::Table2DSetVectorReduce(op, *i, x, y),
                rest,
            ))),
            (ArgumentExpression::Vector(x), ArgumentExpression::Set(y)) => Ok(Some((
                NumericTableExpression::Table2DVectorSetReduce(op, *i, x, y),
                rest,
            ))),
            (ArgumentExpression::Set(x), ArgumentExpression::Element(y)) => Ok(Some((
                NumericTableExpression::Table2DReduceX(op, *i, x, y),
                rest,
            ))),
            (ArgumentExpression::Element(x), ArgumentExpression::Set(y)) => Ok(Some((
                NumericTableExpression::Table2DReduceY(op, *i, x, y),
                rest,
            ))),
            (ArgumentExpression::Vector(x), ArgumentExpression::Element(y)) => Ok(Some((
                NumericTableExpression::Table2DVectorReduceX(op, *i, x, y),
                rest,
            ))),
            (ArgumentExpression::Element(x), ArgumentExpression::Vector(y)) => Ok(Some((
                NumericTableExpression::Table2DVectorReduceY(op, *i, x, y),
                rest,
            ))),
            (x, y) => Err(ParseErr::new(format!(
                "arguments `{x:?}` `{y:?}` are invalid for sum",
            ))),
        }
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (x, rest) = parse_argument(tokens, metadata, functions, registry, parameters)?;
        let (y, rest) = parse_argument(rest, metadata, functions, registry, parameters)?;
        let (z, rest) = parse_argument(rest, metadata, functions, registry, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((
            NumericTableExpression::Table3DReduce(op, *i, x, y, z),
            rest,
        )))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let (args, rest) =
            parse_multiple_arguments(tokens, metadata, functions, registry, parameters)?;
        Ok(Some((
            NumericTableExpression::TableReduce(op, *i, args),
            rest,
        )))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::*;

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

        StateMetadata {
            object_type_names: object_names,
            name_to_object_type: name_to_object,
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

        let tables_1d = vec![Table1D::new(Vec::new())];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![Table2D::new(Vec::new())];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![Table3D::new(Vec::new())];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let tables = vec![Table::new(FxHashMap::default(), 0)];
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
        let functions = StateFunctions::default();
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
            &functions,
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
        let functions = StateFunctions::default();
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
            &functions,
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
    }

    #[test]
    fn parse_table_1d_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = [")", "i0", ")"].iter().map(|x| x.to_string()).collect();
        let result = parse_expression(
            "f1",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["s0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f1",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_1d_sum_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f1", "s0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table1DReduce(
                ReduceOperator::Sum,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["f1", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table1DVectorReduce(
                ReduceOperator::Sum,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_table_1d_sum_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f1", "s0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_1d_product_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f1", "s0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table1DReduce(
                ReduceOperator::Product,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["f1", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table1DVectorReduce(
                ReduceOperator::Product,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_table_1d_product_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f1", "s0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            "product",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_1d_max_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f1", "s0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table1DReduce(
                ReduceOperator::Max,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["f1", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table1DVectorReduce(
                ReduceOperator::Max,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_table_1d_max_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f1", "s0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            "max",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_1d_min_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f1", "s0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table1DReduce(
                ReduceOperator::Min,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[3..]);

        let tokens: Vec<String> = ["f1", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table1DVectorReduce(
                ReduceOperator::Min,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[3..]);
    }

    #[test]
    fn parse_table_1d_min_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f1", "s0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            "min",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_2d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
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
            &functions,
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
    }

    #[test]
    fn parse_table_2d_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
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
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_2d_sum_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f2", "s0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduce(
                ReduceOperator::Sum,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "v1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduce(
                ReduceOperator::Sum,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                VectorExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "s0", "v1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DSetVectorReduce(
                ReduceOperator::Sum,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                VectorExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorSetReduce(
                ReduceOperator::Sum,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
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
            &functions,
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
            NumericTableExpression::Table2DReduceX(
                ReduceOperator::Sum,
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
            &functions,
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
            NumericTableExpression::Table2DReduceY(
                ReduceOperator::Sum,
                0,
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduceX(
                ReduceOperator::Sum,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0)
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduceY(
                ReduceOperator::Sum,
                0,
                ElementExpression::Constant(0),
                VectorExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_table_2d_sum_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f2", "e0", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f2", "0", "e0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_2d_product_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f2", "s0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduce(
                ReduceOperator::Product,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "v1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduce(
                ReduceOperator::Product,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                VectorExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "s0", "v1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DSetVectorReduce(
                ReduceOperator::Product,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                VectorExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorSetReduce(
                ReduceOperator::Product,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "s0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduceX(
                ReduceOperator::Product,
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
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduceY(
                ReduceOperator::Product,
                0,
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduceX(
                ReduceOperator::Product,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0)
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduceY(
                ReduceOperator::Product,
                0,
                ElementExpression::Constant(0),
                VectorExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_table_2d_product_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f2", "e0", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f2", "0", "e0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            "product",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_2d_max_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f2", "s0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduce(
                ReduceOperator::Max,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "v1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduce(
                ReduceOperator::Max,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                VectorExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "s0", "v1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DSetVectorReduce(
                ReduceOperator::Max,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                VectorExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorSetReduce(
                ReduceOperator::Max,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "s0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduceX(
                ReduceOperator::Max,
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
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduceY(
                ReduceOperator::Max,
                0,
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduceX(
                ReduceOperator::Max,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0)
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduceY(
                ReduceOperator::Max,
                0,
                ElementExpression::Constant(0),
                VectorExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_table_2d_max_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f2", "e0", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f2", "0", "e0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            "max",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_2d_min_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f2", "s0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduce(
                ReduceOperator::Min,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "v1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduce(
                ReduceOperator::Min,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                VectorExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "s0", "v1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DSetVectorReduce(
                ReduceOperator::Min,
                0,
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                VectorExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "s1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorSetReduce(
                ReduceOperator::Min,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "s0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduceX(
                ReduceOperator::Min,
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
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DReduceY(
                ReduceOperator::Min,
                0,
                ElementExpression::Constant(0),
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "v0", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduceX(
                ReduceOperator::Min,
                0,
                VectorExpression::Reference(ReferenceExpression::Variable(0)),
                ElementExpression::Variable(0)
            )
        );
        assert_eq!(rest, &tokens[4..]);

        let tokens: Vec<String> = ["f2", "0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table2DVectorReduceY(
                ReduceOperator::Min,
                0,
                ElementExpression::Constant(0),
                VectorExpression::Reference(ReferenceExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_table_2d_min_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f2", "e0", "0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f2", "0", "e0", "v0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            "min",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_3d_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["0", "1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f3",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table3D(
                0,
                ElementExpression::Constant(0),
                ElementExpression::Constant(1),
                ElementExpression::Variable(0),
            )
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_table_3d_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["0", "1", "e0", "1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f3",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_3d_sum_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f3", "s2", "1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table3DReduce(
                ReduceOperator::Sum,
                0,
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(2))),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Element(ElementExpression::Variable(0)),
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn pares_table_3d_sum_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f3", "s2", "1", "e0", "1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_3d_product_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f3", "s2", "1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table3DReduce(
                ReduceOperator::Product,
                0,
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(2))),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Element(ElementExpression::Variable(0)),
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn pares_table_3d_product_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f3", "s2", "1", "e0", "1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_3d_max_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f3", "s2", "1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table3DReduce(
                ReduceOperator::Max,
                0,
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(2))),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Element(ElementExpression::Variable(0)),
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn pares_table_3d_max_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f3", "s2", "1", "e0", "1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_3d_min_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f3", "s2", "1", "e0", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table3DReduce(
                ReduceOperator::Min,
                0,
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(2))),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Element(ElementExpression::Variable(0)),
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn pares_table_3d_min_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f3", "s2", "1", "e0", "1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["0", "1", "e0", "e1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f4",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::Table(
                0,
                vec![
                    ElementExpression::Constant(0),
                    ElementExpression::Constant(1),
                    ElementExpression::Variable(0),
                    ElementExpression::Variable(1),
                ]
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_table_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();
        let tokens: Vec<String> = ["0", "1", "s0", "e1", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "f4",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_sum_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::TableReduce(
                ReduceOperator::Sum,
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
    }

    #[test]
    fn parse_table_sum_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", "i0"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "sum",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_product_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::TableReduce(
                ReduceOperator::Product,
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
    }

    #[test]
    fn parse_table_product_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", "i0"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "product",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_max_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::TableReduce(
                ReduceOperator::Max,
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
    }

    #[test]
    fn parse_table_max_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", "i0"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "max",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_min_ok() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
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
            NumericTableExpression::TableReduce(
                ReduceOperator::Min,
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
    }

    #[test]
    fn parse_table_min_err() {
        let metadata = generate_metadata();
        let functions = StateFunctions::default();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", "i0"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(
            "min",
            &tokens,
            &metadata,
            &functions,
            &registry,
            &parameters,
            &registry.integer_tables,
        );
        assert!(result.is_err());
    }
}
