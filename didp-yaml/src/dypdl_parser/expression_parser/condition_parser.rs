use super::continuous_parser;
use super::element_parser;
use super::integer_parser;
use super::util;
use super::util::ParseErr;
use dypdl::expression::{
    ComparisonOperator, Condition, ContinuousExpression, ElementExpression, IntegerExpression,
    SetCondition,
};
use rustc_hash::FxHashMap;

pub fn parse_expression<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b dypdl::StateMetadata,
    registry: &'b dypdl::TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(Condition, &'a [String]), ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    match &token[..] {
        "(" => {
            let (name, rest) = rest
                .split_first()
                .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
            if let Some((expression, rest)) = element_parser::parse_table_expression(
                name,
                rest,
                metadata,
                registry,
                parameters,
                &registry.bool_tables,
            )? {
                Ok((Condition::Table(Box::new(expression)), rest))
            } else {
                parse_operation(name, rest, metadata, registry, parameters)
            }
        }
        "true" => Ok((Condition::Constant(true), rest)),
        "false" => Ok((Condition::Constant(false), rest)),
        key => {
            if let Some(value) = registry.bool_tables.name_to_constant.get(key) {
                Ok((Condition::Constant(*value), rest))
            } else {
                Err(ParseErr::new(format!("unexpected token: `{}`", token)))
            }
        }
    }
}

fn parse_operation<'a, 'b, 'c>(
    name: &'a str,
    tokens: &'a [String],
    metadata: &'b dypdl::StateMetadata,
    registry: &'b dypdl::TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(Condition, &'a [String]), ParseErr> {
    match name {
        "not" => {
            let (condition, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Not(Box::new(condition)), rest))
        }
        "and" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::And(Box::new(x), Box::new(y)), rest))
        }
        "or" => {
            let (x, rest) = parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Or(Box::new(x), Box::new(y)), rest))
        }
        "is_in" => {
            let (element, rest) =
                element_parser::parse_expression(tokens, metadata, registry, parameters)?;
            let (set, rest) =
                element_parser::parse_set_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((
                Condition::Set(Box::new(SetCondition::IsIn(element, set))),
                rest,
            ))
        }
        "is_subset" => {
            let (x, rest) =
                element_parser::parse_set_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) =
                element_parser::parse_set_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Set(Box::new(SetCondition::IsSubset(x, y))), rest))
        }
        "is_empty" => {
            let (set, rest) =
                element_parser::parse_set_expression(tokens, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Set(Box::new(SetCondition::IsEmpty(set))), rest))
        }
        _ => parse_comparison(name, tokens, metadata, registry, parameters),
    }
}

fn parse_comparison<'a, 'b, 'c>(
    operator: &'a str,
    tokens: &'a [String],
    metadata: &'b dypdl::StateMetadata,
    registry: &'b dypdl::TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(Condition, &'a [String]), ParseErr> {
    let operator = match operator {
        "=" => ComparisonOperator::Eq,
        "!=" => ComparisonOperator::Ne,
        ">=" => ComparisonOperator::Ge,
        ">" => ComparisonOperator::Gt,
        "<=" => ComparisonOperator::Le,
        "<" => ComparisonOperator::Lt,
        _ => return Err(ParseErr::new(format!("no such operator `{}`", operator))),
    };
    if let Ok((x, y, rest)) = parse_ii(tokens, metadata, registry, parameters) {
        Ok((
            Condition::ComparisonI(operator, Box::new(x), Box::new(y)),
            rest,
        ))
    } else if let Ok((x, y, rest)) = parse_cc(tokens, metadata, registry, parameters) {
        Ok((
            Condition::ComparisonC(operator, Box::new(x), Box::new(y)),
            rest,
        ))
    } else if let Ok((x, y, rest)) = parse_ee(tokens, metadata, registry, parameters) {
        Ok((
            Condition::ComparisonE(operator, Box::new(x), Box::new(y)),
            rest,
        ))
    } else {
        Err(ParseErr::new(format!(
            "could not parse `{:?}` as a comparison",
            tokens
        )))
    }
}

fn parse_ii<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b dypdl::StateMetadata,
    registry: &'b dypdl::TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(IntegerExpression, IntegerExpression, &'a [String]), ParseErr> {
    let (x, rest) = integer_parser::parse_expression(tokens, metadata, registry, parameters)?;
    let (y, rest) = integer_parser::parse_expression(rest, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((x, y, rest))
}

fn parse_cc<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b dypdl::StateMetadata,
    registry: &'b dypdl::TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(ContinuousExpression, ContinuousExpression, &'a [String]), ParseErr> {
    let (x, rest) = continuous_parser::parse_expression(tokens, metadata, registry, parameters)?;
    let (y, rest) = continuous_parser::parse_expression(rest, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((x, y, rest))
}

fn parse_ee<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b dypdl::StateMetadata,
    registry: &'b dypdl::TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(ElementExpression, ElementExpression, &'a [String]), ParseErr> {
    let (x, rest) = element_parser::parse_expression(tokens, metadata, registry, parameters)?;
    let (y, rest) = element_parser::parse_expression(rest, metadata, registry, parameters)?;
    let rest = util::parse_closing(rest)?;
    Ok((x, y, rest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;

    fn generate_metadata() -> dypdl::StateMetadata {
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

        dypdl::StateMetadata {
            object_type_names: object_names,
            name_to_object_type: name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            integer_variable_names,
            name_to_integer_variable,
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
        name_to_constant.insert(String::from("f0"), true);

        let tables_1d = vec![dypdl::Table1D::new(vec![true, false])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![dypdl::Table2D::new(vec![vec![true, false]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![dypdl::Table3D::new(vec![vec![vec![true, false]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, true);
        let key = vec![0, 0, 0, 1];
        map.insert(key, false);
        let tables = vec![dypdl::Table::new(map, false)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        dypdl::TableRegistry {
            bool_tables: dypdl::TableData {
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
    fn parse_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [")", "(", "is_in", "2", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "2", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn pare_constant_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["f0", ")"].iter().map(|x| String::from(*x)).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, _) = result.unwrap();
        assert_eq!(expression, Condition::Constant(true));

        let tokens: Vec<String> = ["true", ")"].iter().map(|x| String::from(*x)).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, _) = result.unwrap();
        assert_eq!(expression, Condition::Constant(true));

        let tokens: Vec<String> = ["false", ")"].iter().map(|x| String::from(*x)).collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, _) = result.unwrap();
        assert_eq!(expression, Condition::Constant(false));
    }

    #[test]
    fn parse_table_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = [
            "(", "f4", "0", "e0", "e1", "e2", ")", "(", "is", "0", "e0", ")", ")",
        ]
        .iter()
        .map(|x| String::from(*x))
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Table(Box::new(TableExpression::Table(
                0,
                vec![
                    ElementExpression::Constant(0),
                    ElementExpression::Variable(0),
                    ElementExpression::Variable(1),
                    ElementExpression::Variable(2),
                ]
            )))
        );
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_table_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = [
            "(", "f4", "0", "e0", "e1", "e2", "i0", ")", "(", "is", "0", "e0", ")", ")",
        ]
        .iter()
        .map(|x| String::from(*x))
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_not_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "not", "(", "is_empty", "s0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Not(Box::new(Condition::Set(Box::new(SetCondition::IsEmpty(
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )))))
        );
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_not_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "not", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "not", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "and", "(", "not", "s0", ")", "(", "is_empty", "s1", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_and_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = [
            "(", "and", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::And(
                Box::new(Condition::Set(Box::new(SetCondition::IsEmpty(
                    SetExpression::Reference(ReferenceExpression::Variable(0))
                )))),
                Box::new(Condition::Set(Box::new(SetCondition::IsEmpty(
                    SetExpression::Reference(ReferenceExpression::Variable(1))
                ))))
            )
        );
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_and_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "and", "(", "is_empty", "s0", ")", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "and", "(", "is_empty", "s0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "and", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", "(", "is_empty",
            "s2", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_or_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = [
            "(", "or", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Or(
                Box::new(Condition::Set(Box::new(SetCondition::IsEmpty(
                    SetExpression::Reference(ReferenceExpression::Variable(0))
                )))),
                Box::new(Condition::Set(Box::new(SetCondition::IsEmpty(
                    SetExpression::Reference(ReferenceExpression::Variable(1))
                ))))
            )
        );
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_or_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "or", "(", "is_empty", "s0", ")", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "or", "(", "is_empty", "s0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "or", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", "(", "is_empty",
            "s2", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_lt_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "<", "2", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonI(
                ComparisonOperator::Lt,
                Box::new(IntegerExpression::Constant(2)),
                Box::new(IntegerExpression::Variable(0))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_is_in_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "is_in", "2", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Set(Box::new(SetCondition::IsIn(
                ElementExpression::Constant(2),
                SetExpression::Reference(ReferenceExpression::Variable(0))
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_is_in_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "is_in", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_in", "0", "e1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_in", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_in", "0", "s1", "s2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_subset_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "is_subset", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Set(Box::new(SetCondition::IsSubset(
                SetExpression::Reference(ReferenceExpression::Variable(0)),
                SetExpression::Reference(ReferenceExpression::Variable(1))
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_is_subset_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "is_subset", "e0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_subset", "s0", "e1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_subset", "s0", "s1", "s2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_empty_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "is_empty", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Set(Box::new(SetCondition::IsEmpty(SetExpression::Reference(
                ReferenceExpression::Variable(0)
            ))))
        );
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_is_empty_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "is_empty", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_empty", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_comparison_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "=", "1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonI(
                ComparisonOperator::Eq,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "!=", "1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonI(
                ComparisonOperator::Ne,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">=", "1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonI(
                ComparisonOperator::Ge,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">", "1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonI(
                ComparisonOperator::Gt,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<=", "1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonI(
                ComparisonOperator::Le,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<", "1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonI(
                ComparisonOperator::Lt,
                Box::new(IntegerExpression::Constant(1)),
                Box::new(IntegerExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "=", "1.5", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonC(
                ComparisonOperator::Eq,
                Box::new(ContinuousExpression::Constant(1.5)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "!=", "1.5", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonC(
                ComparisonOperator::Ne,
                Box::new(ContinuousExpression::Constant(1.5)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">=", "1.5", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonC(
                ComparisonOperator::Ge,
                Box::new(ContinuousExpression::Constant(1.5)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">", "1.5", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonC(
                ComparisonOperator::Gt,
                Box::new(ContinuousExpression::Constant(1.5)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<=", "1.5", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonC(
                ComparisonOperator::Le,
                Box::new(ContinuousExpression::Constant(1.5)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<", "1.5", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonC(
                ComparisonOperator::Lt,
                Box::new(ContinuousExpression::Constant(1.5)),
                Box::new(ContinuousExpression::Constant(2.0))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "=", "e1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonE(
                ComparisonOperator::Eq,
                Box::new(ElementExpression::Variable(1)),
                Box::new(ElementExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "!=", "e1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonE(
                ComparisonOperator::Ne,
                Box::new(ElementExpression::Variable(1)),
                Box::new(ElementExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">=", "e1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonE(
                ComparisonOperator::Ge,
                Box::new(ElementExpression::Variable(1)),
                Box::new(ElementExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">", "e1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonE(
                ComparisonOperator::Gt,
                Box::new(ElementExpression::Variable(1)),
                Box::new(ElementExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<=", "e1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonE(
                ComparisonOperator::Le,
                Box::new(ElementExpression::Variable(1)),
                Box::new(ElementExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<", "e1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::ComparisonE(
                ComparisonOperator::Lt,
                Box::new(ElementExpression::Variable(1)),
                Box::new(ElementExpression::Constant(2))
            )
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_comparison_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "==", "1", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "==", "1.5", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "==", "e0", "1.5", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
