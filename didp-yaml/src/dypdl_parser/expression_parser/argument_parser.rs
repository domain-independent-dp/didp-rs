use super::element_parser;
use super::util::ParseErr;
use dypdl::expression::ArgumentExpression;
use dypdl::{StateMetadata, TableRegistry};
use rustc_hash::FxHashMap;

pub fn parse_argument<'a>(
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

pub fn parse_multiple_arguments<'a>(
    tokens: &'a [String],
    metadata: &StateMetadata,
    registry: &TableRegistry,
    parameters: &FxHashMap<String, usize>,
) -> Result<(Vec<ArgumentExpression>, &'a [String]), ParseErr> {
    let mut args = Vec::new();
    let mut xs = tokens;
    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;
        if next_token == ")" {
            return Ok((args, rest));
        }
        let (expression, new_xs) = parse_argument(xs, metadata, registry, parameters)?;
        args.push(expression);
        xs = new_xs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::variable_type::*;
    use dypdl::TableData;

    fn generate_metadata() -> StateMetadata {
        let object_names = vec![String::from("something")];
        let object_numbers = vec![3];
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

        let element_resource_variable_names = vec![
            String::from("er0"),
            String::from("er1"),
            String::from("er2"),
            String::from("er3"),
        ];
        let mut name_to_element_resource_variable = FxHashMap::default();
        name_to_element_resource_variable.insert(String::from("er0"), 0);
        name_to_element_resource_variable.insert(String::from("er1"), 1);
        name_to_element_resource_variable.insert(String::from("er2"), 2);
        name_to_element_resource_variable.insert(String::from("er3"), 3);
        let element_resource_variable_to_object = vec![0, 0, 0, 0];

        let integer_variable_names = vec![
            String::from("n0"),
            String::from("n1"),
            String::from("n2"),
            String::from("n3"),
        ];
        let mut name_to_integer_variable = FxHashMap::default();
        name_to_integer_variable.insert(String::from("n0"), 0);
        name_to_integer_variable.insert(String::from("n1"), 1);
        name_to_integer_variable.insert(String::from("n2"), 2);
        name_to_integer_variable.insert(String::from("n3"), 3);

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
            element_resource_variable_names,
            name_to_element_resource_variable,
            element_resource_variable_to_object,
            element_less_is_better: vec![false, false, true, false],
            ..Default::default()
        }
    }

    fn generate_registry() -> TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("et0"), 1);

        let tables_1d = vec![dypdl::Table1D::new(vec![1, 0])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("et1"), 0);

        let tables_2d = vec![dypdl::Table2D::new(vec![vec![1, 0]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("et2"), 0);

        let tables_3d = vec![dypdl::Table3D::new(vec![vec![vec![1, 0]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("et3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, 1);
        let key = vec![0, 0, 0, 1];
        map.insert(key, 0);
        let tables = vec![dypdl::Table::new(map, 0)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("et4"), 0);

        let element_tables = TableData {
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

        let mut set = Set::with_capacity(3);
        set.insert(0);
        set.insert(2);
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("st0"), set.clone());
        let default = Set::with_capacity(3);
        let tables_1d = vec![dypdl::Table1D::new(vec![set, default.clone(), default])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("st1"), 0);

        let set_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            ..Default::default()
        };

        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("vt0"), vec![0, 1]);
        let tables_1d = vec![dypdl::Table1D::new(vec![vec![0, 1]])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("vt1"), 0);

        let vector_tables = TableData {
            name_to_constant,
            tables_1d,
            name_to_table_1d,
            ..Default::default()
        };

        TableRegistry {
            element_tables,
            set_tables,
            vector_tables,
            ..Default::default()
        }
    }

    fn generate_parameters() -> FxHashMap<String, usize> {
        let mut parameters = FxHashMap::default();
        parameters.insert(String::from("param"), 0);
        parameters
    }

    #[test]
    fn parse_argument_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["e0", "v3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ArgumentExpression::Element(ElementExpression::Variable(0))
        );
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["s0", "v3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(0)))
        );
        assert_eq!(rest, &tokens[1..]);

        let tokens: Vec<String> = ["v0", "v3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            ArgumentExpression::Vector(VectorExpression::Reference(ReferenceExpression::Variable(
                0
            )))
        );
        assert_eq!(rest, &tokens[1..]);
    }

    #[test]
    fn parse_argument_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["n0", "v3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_argument(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_multiple_arguments_ok() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["s2", "1", "e0", "v3", ")", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_multiple_arguments(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (result, rest) = result.unwrap();
        assert_eq!(
            result,
            vec![
                ArgumentExpression::Set(SetExpression::Reference(ReferenceExpression::Variable(2))),
                ArgumentExpression::Element(ElementExpression::Constant(1)),
                ArgumentExpression::Element(ElementExpression::Variable(0)),
                ArgumentExpression::Vector(VectorExpression::Reference(
                    ReferenceExpression::Variable(3)
                )),
            ]
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_multiple_arguments_argument_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["s2", "1", "e0", "v3", "i0", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_multiple_arguments(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_multiple_arguments_no_closing_err() {
        let metadata = generate_metadata();
        let parameters = generate_parameters();
        let registry = generate_registry();

        let tokens: Vec<String> = ["f4", "s2", "1", "e0", "v3", "i0"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_multiple_arguments(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
