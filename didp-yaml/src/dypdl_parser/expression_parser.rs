//! A module fto parse expressions from a string value.

use dypdl::expression;
use rustc_hash::FxHashMap;

mod argument_parser;
mod condition_parser;
mod continuous_parser;
mod continuous_vector_parser;
mod element_parser;
mod integer_parser;
mod integer_vector_parser;
mod numeric_table_parser;
mod table_vector_parser;
mod util;

pub use util::ParseErr;

/// Returns an integer expression parsed from a string value.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn parse_integer(
    text: String,
    metadata: &dypdl::StateMetadata,
    registry: &dypdl::TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::IntegerExpression, ParseErr> {
    let tokens = tokenize(text);
    let (expression, rest) =
        integer_parser::parse_expression(&tokens, metadata, registry, parameters)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::new(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

/// Returns a continuous expression parsed from a string value.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn parse_continuous(
    text: String,
    metadata: &dypdl::StateMetadata,
    registry: &dypdl::TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::ContinuousExpression, ParseErr> {
    let tokens = tokenize(text);
    let (expression, rest) =
        continuous_parser::parse_expression(&tokens, metadata, registry, parameters)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::new(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

/// Returns an element expression parsed from a string value.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn parse_element(
    text: String,
    metadata: &dypdl::StateMetadata,
    registry: &dypdl::TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::ElementExpression, ParseErr> {
    let tokens = tokenize(text);
    let (expression, rest) =
        element_parser::parse_expression(&tokens, metadata, registry, parameters)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::new(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

/// Returns a set expression parsed from a string value.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn parse_set(
    text: String,
    metadata: &dypdl::StateMetadata,
    registry: &dypdl::TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::SetExpression, ParseErr> {
    let tokens = tokenize(text);
    let (expression, rest) =
        element_parser::parse_set_expression(&tokens, metadata, registry, parameters)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::new(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

/// Returns a vector expression parsed from a string value.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn parse_vector(
    text: String,
    metadata: &dypdl::StateMetadata,
    registry: &dypdl::TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::VectorExpression, ParseErr> {
    let tokens = tokenize(text);
    let (expression, rest) =
        element_parser::parse_vector_expression(&tokens, metadata, registry, parameters)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::new(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

/// Returns a condition parsed from a string value.
///
/// `parameters` specify names and values of constants.
///
/// # Errors
///
/// If the format is invalid.
pub fn parse_condition(
    text: String,
    metadata: &dypdl::StateMetadata,
    registry: &dypdl::TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<expression::Condition, ParseErr> {
    let tokens = tokenize(text);
    let (expression, rest) =
        condition_parser::parse_expression(&tokens, metadata, registry, parameters)?;
    if rest.is_empty() {
        Ok(expression)
    } else {
        Err(ParseErr::new(format!(
            "unexpected tokens: `{}`",
            rest.join(" ")
        )))
    }
}

fn tokenize(text: String) -> Vec<String> {
    text.replace('(', " ( ")
        .replace(')', " ) ")
        .replace('|', " | ")
        .replace('~', " ~ ")
        .replace('{', " { ")
        .replace('}', " } ")
        .replace(':', " : ")
        .split_whitespace()
        .map(|x| x.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use dypdl::*;

    fn generate_metadata() -> dypdl::StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = FxHashMap::default();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            "s0".to_string(),
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
        ];
        let mut name_to_set_variable = FxHashMap::default();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        name_to_set_variable.insert("s2".to_string(), 2);
        name_to_set_variable.insert("s3".to_string(), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let vector_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_vector_variable = FxHashMap::default();
        name_to_vector_variable.insert("p0".to_string(), 0);
        name_to_vector_variable.insert("p1".to_string(), 1);
        name_to_vector_variable.insert("p2".to_string(), 2);
        name_to_vector_variable.insert("p3".to_string(), 3);
        let vector_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
        ];
        let mut name_to_element_variable = FxHashMap::default();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        name_to_element_variable.insert("e2".to_string(), 2);
        name_to_element_variable.insert("e3".to_string(), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let integer_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert("n0".to_string(), 0);
        name_to_integer_variable.insert("n1".to_string(), 1);
        name_to_integer_variable.insert("n2".to_string(), 2);
        name_to_integer_variable.insert("n3".to_string(), 3);

        let integer_resource_variable_names = vec![
            "r0".to_string(),
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ];
        let mut name_to_integer_resource_variable = FxHashMap::default();
        name_to_integer_resource_variable.insert("r0".to_string(), 0);
        name_to_integer_resource_variable.insert("r1".to_string(), 1);
        name_to_integer_resource_variable.insert("r2".to_string(), 2);
        name_to_integer_resource_variable.insert("r3".to_string(), 3);

        let continuous_variable_names = vec![
            "c0".to_string(),
            "c1".to_string(),
            "c2".to_string(),
            "c3".to_string(),
        ];
        let mut name_to_continuous_variable = FxHashMap::default();
        name_to_continuous_variable.insert("c0".to_string(), 0);
        name_to_continuous_variable.insert("c1".to_string(), 1);
        name_to_continuous_variable.insert("c2".to_string(), 2);
        name_to_continuous_variable.insert("c3".to_string(), 3);

        dypdl::StateMetadata {
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
            integer_resource_variable_names,
            name_to_integer_resource_variable,
            integer_less_is_better: vec![false, false, true, false],
            continuous_variable_names,
            name_to_continuous_variable,
            ..Default::default()
        }
    }

    fn generate_parameters() -> FxHashMap<String, usize> {
        let mut parameters = FxHashMap::default();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    fn generate_registry() -> dypdl::TableRegistry {
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

        dypdl::TableRegistry {
            integer_tables: dypdl::TableData {
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
    fn parse_set_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "(union (intersection s0 (difference s2 (add 2 s3))) (remove 1 s1))".to_string();
        let result = parse_set(text, &metadata, &registry, &parameters);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_set_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "s0)".to_string();
        let result = parse_set(text, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_integer_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "n0".to_string();
        let result = parse_integer(text, &metadata, &registry, &parameters);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_integer_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "n0)".to_string();
        let result = parse_integer(text, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_continuous_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "c0".to_string();
        let result = parse_continuous(text, &metadata, &registry, &parameters);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_continuous_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "c0)".to_string();
        let result = parse_continuous(text, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_element_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "e0".to_string();
        let result = parse_element(text, &metadata, &registry, &parameters);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_element_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "e0)".to_string();
        let result = parse_element(text, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_condition_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "(not (and (and (and (> n0 2) (is_subset s0 s1)) (is_empty s0)) (or (< 1 n1) (is_in 2 s0))))"
            .to_string();
        let result = parse_condition(text, &metadata, &registry, &parameters);
        assert!(result.is_ok());
    }

    #[test]
    fn parse_condition_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let text = "(is_empty s[0]))".to_string();
        let result = parse_condition(text, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn tokenize_text() {
        let text = "(+ (- 5 (/ (sum f4 4 ~s2 e0 3) (max (f2 2 e1) n0))) (* r1 (min 3 |(union (intersection s0 (difference s2 (add 2 s3))) (remove 1 s1))|)))".to_string();
        assert_eq!(
            tokenize(text),
            [
                "(",
                "+",
                "(",
                "-",
                "5",
                "(",
                "/",
                "(",
                "sum",
                "f4",
                "4",
                "~",
                "s2",
                "e0",
                "3",
                ")",
                "(",
                "max",
                "(",
                "f2",
                "2",
                "e1",
                ")",
                "n0",
                ")",
                ")",
                ")",
                "(",
                "*",
                "r1",
                "(",
                "min",
                "3",
                "|",
                "(",
                "union",
                "(",
                "intersection",
                "s0",
                "(",
                "difference",
                "s2",
                "(",
                "add",
                "2",
                "s3",
                ")",
                ")",
                ")",
                "(",
                "remove",
                "1",
                "s1",
                ")",
                ")",
                "|",
                ")",
                ")",
                ")",
            ]
        );
    }
}
