use crate::dypdl_parser::expression_parser;
use crate::dypdl_parser::expression_parser::ModelData;
use crate::util;
use dypdl::CostExpression;
use rustc_hash::FxHashMap;
use std::error::Error;

/// `model` is mutable because a custom cost expression may contain a `reduce`/`filter`/
/// `filter` binder, which registers a fresh local variable in `model` as it is parsed.
pub fn load_custom_cost_expressions(
    model: &mut dypdl::Model,
    backward: bool,
    custom_cost_type: &dypdl::CostType,
    cost_expressions: &FxHashMap<String, String>,
) -> Result<(Vec<CostExpression>, Vec<CostExpression>), Box<dyn Error>> {
    let original_forced_transitions = if backward {
        &model.backward_forced_transitions
    } else {
        &model.forward_forced_transitions
    };
    let mut forced_custom_costs = Vec::with_capacity(original_forced_transitions.len());
    let mut parameters = FxHashMap::default();
    for t in original_forced_transitions {
        for (name, value) in t.parameter_names.iter().zip(t.parameter_values.iter()) {
            parameters.insert(name.clone(), *value);
        }
        let custom_cost = if let Some(expression) = cost_expressions.get(&t.name) {
            let mut model_data = ModelData {
                metadata: &model.state_metadata,
                functions: &model.state_functions,
                registry: &model.table_registry,
                parameters: &parameters,
                local_variable_data: &mut model.local_variable_data,
            };
            match custom_cost_type {
                dypdl::CostType::Integer => CostExpression::Integer(
                    expression_parser::parse_integer(expression.clone(), &mut model_data)?
                        .simplify(&model.table_registry),
                ),
                dypdl::CostType::Continuous => CostExpression::Continuous(
                    expression_parser::parse_continuous(expression.clone(), &mut model_data)?
                        .simplify(&model.table_registry),
                ),
            }
        } else {
            return Err(util::YamlContentErr::new(format!(
                "expression for `{name}` is undefined",
                name = t.name
            ))
            .into());
        };
        forced_custom_costs.push(custom_cost);
        parameters.clear();
    }
    let original_transitions = if backward {
        &model.backward_transitions
    } else {
        &model.forward_transitions
    };
    let mut custom_costs = Vec::with_capacity(original_transitions.len());
    for t in original_transitions {
        for (name, value) in t.parameter_names.iter().zip(t.parameter_values.iter()) {
            parameters.insert(name.clone(), *value);
        }
        let custom_cost = if let Some(expression) = cost_expressions.get(&t.name) {
            let mut model_data = ModelData {
                metadata: &model.state_metadata,
                functions: &model.state_functions,
                registry: &model.table_registry,
                parameters: &parameters,
                local_variable_data: &mut model.local_variable_data,
            };
            match custom_cost_type {
                dypdl::CostType::Integer => CostExpression::Integer(
                    expression_parser::parse_integer(expression.clone(), &mut model_data)?
                        .simplify(&model.table_registry),
                ),
                dypdl::CostType::Continuous => CostExpression::Continuous(
                    expression_parser::parse_continuous(expression.clone(), &mut model_data)?
                        .simplify(&model.table_registry),
                ),
            }
        } else {
            return Err(util::YamlContentErr::new(format!(
                "expression for `{name}` is undefined",
                name = t.name
            ))
            .into());
        };
        custom_costs.push(custom_cost);
        parameters.clear();
    }
    Ok((custom_costs, forced_custom_costs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dypdl::expression::*;
    use dypdl::{CostExpression, Transition};

    fn generate_model() -> dypdl::Model {
        dypdl::Model {
            forward_transitions: vec![Transition {
                name: String::from("forward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(1)),
                )),
                ..Default::default()
            }],
            forward_forced_transitions: vec![Transition {
                name: String::from("forward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(2)),
                )),
                ..Default::default()
            }],
            backward_transitions: vec![Transition {
                name: String::from("backward"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(3)),
                )),
                ..Default::default()
            }],
            backward_forced_transitions: vec![Transition {
                name: String::from("backward_forced"),
                cost: CostExpression::Integer(IntegerExpression::BinaryOperation(
                    BinaryOperator::Add,
                    Box::new(IntegerExpression::Cost),
                    Box::new(IntegerExpression::Constant(4)),
                )),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn load_custom_cost_expressions_forward_ok() {
        let mut model = generate_model();

        let mut cost_expressions = FxHashMap::default();
        cost_expressions.insert(String::from("forward"), String::from("(* cost 1)"));
        cost_expressions.insert(String::from("forward_forced"), String::from("(* cost 2)"));
        let custom_cost_type = dypdl::CostType::Integer;

        let result =
            load_custom_cost_expressions(&mut model, false, &custom_cost_type, &cost_expressions);
        assert!(result.is_ok());
        let (custom_costs, forced_custom_costs) = result.unwrap();
        assert_eq!(
            custom_costs,
            vec![CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(1)),
            )),]
        );
        assert_eq!(
            forced_custom_costs,
            vec![CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(2)),
            )),]
        );
    }

    #[test]
    fn load_custom_cost_expressions_backward_ok() {
        let mut model = generate_model();

        let mut cost_expressions = FxHashMap::default();
        cost_expressions.insert(String::from("backward"), String::from("(* cost 1)"));
        cost_expressions.insert(String::from("backward_forced"), String::from("(* cost 2)"));
        let custom_cost_type = dypdl::CostType::Integer;

        let result =
            load_custom_cost_expressions(&mut model, true, &custom_cost_type, &cost_expressions);
        assert!(result.is_ok());
        let (custom_costs, forced_custom_costs) = result.unwrap();
        assert_eq!(
            custom_costs,
            vec![CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(1)),
            )),]
        );
        assert_eq!(
            forced_custom_costs,
            vec![CostExpression::Integer(IntegerExpression::BinaryOperation(
                BinaryOperator::Mul,
                Box::new(IntegerExpression::Cost),
                Box::new(IntegerExpression::Constant(2)),
            )),]
        );
    }

    #[test]
    fn load_custom_cost_expressions_undefined_err() {
        let mut model = generate_model();

        let mut cost_expressions = FxHashMap::default();
        cost_expressions.insert(String::from("backward"), String::from("(* cost 3)"));
        cost_expressions.insert(String::from("backward_forced"), String::from("(* cost 4)"));
        let custom_cost_type = dypdl::CostType::Integer;

        let result =
            load_custom_cost_expressions(&mut model, false, &custom_cost_type, &cost_expressions);
        assert!(result.is_err());
    }

    #[test]
    fn load_custom_cost_expressions_parse_err() {
        let mut model = generate_model();

        let mut cost_expressions = FxHashMap::default();
        cost_expressions.insert(String::from("forward"), String::from("(* cost 1)"));
        cost_expressions.insert(String::from("forward_forced"), String::from("(^ cost 2)"));
        let custom_cost_type = dypdl::CostType::Integer;

        let result =
            load_custom_cost_expressions(&mut model, false, &custom_cost_type, &cost_expressions);
        assert!(result.is_err());
    }
}
