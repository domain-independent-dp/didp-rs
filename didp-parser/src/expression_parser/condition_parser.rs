use super::element_parser;
use super::numeric_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{Comparison, ComparisonOperator, Condition, SetCondition};
use crate::state;
use crate::table_registry;
use crate::variable;
use rustc_hash::FxHashMap;

pub fn parse_expression<'a, 'b, 'c>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b table_registry::TableRegistry,
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
                Ok((Condition::Table(expression), rest))
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
    metadata: &'b state::StateMetadata,
    registry: &'b table_registry::TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(Condition, &'a [String]), ParseErr> {
    match &name[..] {
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
        "is" => {
            let (x, rest) =
                element_parser::parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = element_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Set(Box::new(SetCondition::Eq(x, y))), rest))
        }
        "is_not" => {
            let (x, rest) =
                element_parser::parse_expression(tokens, metadata, registry, parameters)?;
            let (y, rest) = element_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Set(Box::new(SetCondition::Ne(x, y))), rest))
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
        _ => parse_comparison(name, tokens, metadata, registry, parameters)
            .map(|(condition, rest)| (Condition::Comparison(Box::new(condition)), rest)),
    }
}

fn parse_comparison<'a, 'b, 'c>(
    operator: &'a str,
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b table_registry::TableRegistry,
    parameters: &'c FxHashMap<String, usize>,
) -> Result<(Comparison, &'a [String]), ParseErr> {
    let operator = match operator {
        "=" => ComparisonOperator::Eq,
        "!=" => ComparisonOperator::Ne,
        ">=" => ComparisonOperator::Ge,
        ">" => ComparisonOperator::Gt,
        "<=" => ComparisonOperator::Le,
        "<" => ComparisonOperator::Lt,
        _ => return Err(ParseErr::new(format!("no such operator `{}`", operator))),
    };

    if let Ok((x, rest)) = numeric_parser::parse_integer_expression::<variable::Integer>(
        tokens, metadata, registry, parameters,
    ) {
        if let Ok((y, rest)) = numeric_parser::parse_integer_expression::<variable::Integer>(
            rest, metadata, registry, parameters,
        ) {
            let rest = util::parse_closing(rest)?;
            Ok((Comparison::ComparisonII(operator, x, y), rest))
        } else {
            let (y, rest) = numeric_parser::parse_continuous_expression::<variable::Continuous>(
                rest, metadata, registry, parameters,
            )?;
            let rest = util::parse_closing(rest)?;
            Ok((Comparison::ComparisonIC(operator, x, y), rest))
        }
    } else {
        let (x, rest) = numeric_parser::parse_continuous_expression::<variable::Continuous>(
            tokens, metadata, registry, parameters,
        )?;
        if let Ok((y, rest)) = numeric_parser::parse_integer_expression::<variable::Integer>(
            rest, metadata, registry, parameters,
        ) {
            let rest = util::parse_closing(rest)?;
            Ok((Comparison::ComparisonCI(operator, x, y), rest))
        } else {
            let (y, rest) = numeric_parser::parse_continuous_expression::<variable::Continuous>(
                rest, metadata, registry, parameters,
            )?;
            let rest = util::parse_closing(rest)?;
            Ok((Comparison::ComparisonCC(operator, x, y), rest))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::table;
    use crate::table_data;

    fn generate_metadata() -> state::StateMetadata {
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

        state::StateMetadata {
            object_names,
            name_to_object,
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

    fn generate_registry() -> table_registry::TableRegistry {
        let mut name_to_constant = FxHashMap::default();
        name_to_constant.insert(String::from("f0"), true);

        let tables_1d = vec![table::Table1D::new(vec![true, false])];
        let mut name_to_table_1d = FxHashMap::default();
        name_to_table_1d.insert(String::from("f1"), 0);

        let tables_2d = vec![table::Table2D::new(vec![vec![true, false]])];
        let mut name_to_table_2d = FxHashMap::default();
        name_to_table_2d.insert(String::from("f2"), 0);

        let tables_3d = vec![table::Table3D::new(vec![vec![vec![true, false]]])];
        let mut name_to_table_3d = FxHashMap::default();
        name_to_table_3d.insert(String::from("f3"), 0);

        let mut map = FxHashMap::default();
        let key = vec![0, 0, 0, 0];
        map.insert(key, true);
        let key = vec![0, 0, 0, 1];
        map.insert(key, false);
        let tables = vec![table::Table::new(map, false)];
        let mut name_to_table = FxHashMap::default();
        name_to_table.insert(String::from("f4"), 0);

        table_registry::TableRegistry {
            bool_tables: table_data::TableData {
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
            Condition::Table(TableExpression::Table(
                0,
                vec![
                    ElementExpression::Constant(0),
                    ElementExpression::Variable(0),
                    ElementExpression::Variable(1),
                    ElementExpression::Variable(2),
                ]
            ))
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
    fn parse_eq_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "=", "2", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonIC(
                ComparisonOperator::Eq,
                NumericExpression::Constant(2),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "=", "2.0", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCI(
                ComparisonOperator::Eq,
                NumericExpression::Constant(2.0),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "=", "2.0", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCC(
                ComparisonOperator::Eq,
                NumericExpression::Constant(2.0),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "=", "2", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Eq,
                NumericExpression::Constant(2),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_eq_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "=", "2", "i0", "i1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_ne_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "!=", "2", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Ne,
                NumericExpression::Constant(2),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "!=", "2", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonIC(
                ComparisonOperator::Ne,
                NumericExpression::Constant(2),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "!=", "2.0", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCI(
                ComparisonOperator::Ne,
                NumericExpression::Constant(2.0),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "!=", "2.0", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCC(
                ComparisonOperator::Ne,
                NumericExpression::Constant(2.0),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_ne_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "!=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "!=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "!=", "2", "i0", "i1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_gt_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", ">", "2", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Gt,
                NumericExpression::Constant(2),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">", "2", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonIC(
                ComparisonOperator::Gt,
                NumericExpression::Constant(2),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">", "2.0", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCI(
                ComparisonOperator::Gt,
                NumericExpression::Constant(2.0),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">", "2.0", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCC(
                ComparisonOperator::Gt,
                NumericExpression::Constant(2.0),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_gt_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", ">", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">", "2", "i0", "i1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_ge_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", ">=", "2", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Ge,
                NumericExpression::Constant(2),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">=", "2", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonIC(
                ComparisonOperator::Ge,
                NumericExpression::Constant(2),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">=", "2.0", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCI(
                ComparisonOperator::Ge,
                NumericExpression::Constant(2.0),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", ">=", "2.0", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCC(
                ComparisonOperator::Ge,
                NumericExpression::Constant(2.0),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_ge_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", ">=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">=", "2", "i0", "i1", ")", ")"]
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
            Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Lt,
                NumericExpression::Constant(2),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<", "2", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonIC(
                ComparisonOperator::Lt,
                NumericExpression::Constant(2),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<", "2.0", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCI(
                ComparisonOperator::Lt,
                NumericExpression::Constant(2.0),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<", "2.0", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCC(
                ComparisonOperator::Lt,
                NumericExpression::Constant(2.0),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_lt_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "<", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<", "2", "i0", "i1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_le_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "<=", "2", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonII(
                ComparisonOperator::Le,
                NumericExpression::Constant(2),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<=", "2", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonIC(
                ComparisonOperator::Le,
                NumericExpression::Constant(2),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<=", "2.0", "i0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCI(
                ComparisonOperator::Le,
                NumericExpression::Constant(2.0),
                NumericExpression::IntegerVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);

        let tokens: Vec<String> = ["(", "<=", "2.0", "c0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Comparison(Box::new(Comparison::ComparisonCC(
                ComparisonOperator::Le,
                NumericExpression::Constant(2.0),
                NumericExpression::ContinuousVariable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_le_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "<=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<=", "2", "i0", "i1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "is", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Set(Box::new(SetCondition::Eq(
                ElementExpression::Constant(2),
                ElementExpression::Variable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_is_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "is", "e0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is", "i0", "e1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is", "0", "e1", "e2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_not_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "is_not", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert_eq!(
            expression,
            Condition::Set(Box::new(SetCondition::Ne(
                ElementExpression::Constant(2),
                ElementExpression::Variable(0)
            )))
        );
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_is_not_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "is_not", "e0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_not", "i0", "e1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_not", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_not", "0", "e1", "e2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
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
}
