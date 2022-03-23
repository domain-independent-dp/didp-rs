use super::set_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::BoolTableExpression;
use crate::state;
use crate::table_data;
use std::collections;

pub fn parse_expression<'a, 'b, 'c>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    tables: &'b table_data::TableData<bool>,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<Option<(BoolTableExpression, &'a [String])>, ParseErr> {
    if let Some(i) = tables.name_to_table_1d.get(name) {
        let (x, rest) = set_parser::parse_element_expression(tokens, metadata, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((BoolTableExpression::Table1D(*i, x), rest)))
    } else if let Some(i) = tables.name_to_table_2d.get(name) {
        let (x, rest) = set_parser::parse_element_expression(tokens, metadata, parameters)?;
        let (y, rest) = set_parser::parse_element_expression(rest, metadata, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((BoolTableExpression::Table2D(*i, x, y), rest)))
    } else if let Some(i) = tables.name_to_table_3d.get(name) {
        let (x, rest) = set_parser::parse_element_expression(tokens, metadata, parameters)?;
        let (y, rest) = set_parser::parse_element_expression(rest, metadata, parameters)?;
        let (z, rest) = set_parser::parse_element_expression(rest, metadata, parameters)?;
        let rest = util::parse_closing(rest)?;
        Ok(Some((BoolTableExpression::Table3D(*i, x, y, z), rest)))
    } else if let Some(i) = tables.name_to_table.get(name) {
        let result = parse_table(*i, tokens, metadata, parameters)?;
        Ok(Some(result))
    } else {
        Ok(None)
    }
}

fn parse_table<'a, 'b, 'c>(
    i: usize,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(BoolTableExpression, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((BoolTableExpression::Table(i, args), rest));
        }
        let (expression, new_xs) = set_parser::parse_element_expression(xs, metadata, parameters)?;
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

        let integer_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_integer_variable = HashMap::new();
        name_to_integer_variable.insert("n0".to_string(), 0);
        name_to_integer_variable.insert("n1".to_string(), 1);
        name_to_integer_variable.insert("n2".to_string(), 2);
        name_to_integer_variable.insert("n3".to_string(), 3);

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
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

    fn generate_tables() -> table_data::TableData<bool> {
        let mut name_to_constant = HashMap::new();
        name_to_constant.insert(String::from("f0"), true);

        let tables_1d = vec![table::Table1D::new(vec![true, false])];
        let mut name_to_table_1d = HashMap::new();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![vec![true, false]])];
        let mut name_to_table_2d = HashMap::new();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![vec![vec![true, false]]])];
        let mut name_to_table_3d = HashMap::new();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = HashMap::new();
        let key = vec![0, 0, 0, 0];
        map.insert(key, true);
        let key = vec![0, 0, 0, 1];
        map.insert(key, false);
        let tables = vec![table::Table::new(map, false)];
        let mut name_to_table = HashMap::new();
        name_to_table.insert(String::from("f4"), 0);

        table_data::TableData {
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
    fn parse_table_1d_ok() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, _) = result.unwrap();
        assert!(matches!(
            expression,
            BoolTableExpression::Table1D(0, ElementExpression::Constant(0))
        ));

        let result = parse_expression("f0", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_1d_err() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["n0", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "e1", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f1", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [")", "n0", ")"].iter().map(|x| String::from(*x)).collect();
        let result = parse_expression("f1", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_2d_ok() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0", "e1", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, _) = result.unwrap();
        assert!(matches!(
            expression,
            BoolTableExpression::Table2D(
                0,
                ElementExpression::Constant(0),
                ElementExpression::Variable(1)
            )
        ));

        let result = parse_expression("f0", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_2d_err() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["n0", "e1", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "e1", "e3", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_3d_ok() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0", "e1", "e2", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, _) = result.unwrap();
        assert!(matches!(
            expression,
            BoolTableExpression::Table3D(
                0,
                ElementExpression::Constant(0),
                ElementExpression::Variable(1),
                ElementExpression::Variable(2)
            )
        ));

        let result = parse_expression("f0", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_3d_err() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["n0", "e1", "e2", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "e1", "e3", "e4", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "e1", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f3", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_table_ok() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["0", "e1", "e2", "e3", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f4", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_some());
        let (expression, _) = result.unwrap();
        assert!(matches!(expression, BoolTableExpression::Table(0, _)));
        if let BoolTableExpression::Table(i, args) = expression {
            assert_eq!(i, 0);
            assert_eq!(args.len(), 4);
            assert!(matches!(args[0], ElementExpression::Constant(0)));
            assert!(matches!(args[1], ElementExpression::Variable(1)));
            assert!(matches!(args[2], ElementExpression::Variable(2)));
            assert!(matches!(args[3], ElementExpression::Variable(3)));
        }

        let result = parse_expression("f0", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn parse_table_err() {
        let metadata = generate_metadata();
        let tables = generate_tables();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["n0", "e1", "e2", "e3", ")", "n0", ")"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["0", "e1", "e3", "e4"]
            .iter()
            .map(|x| String::from(*x))
            .collect();
        let result = parse_expression("f2", &tokens, &metadata, &tables, &parameters);
        assert!(result.is_err());
    }
}
