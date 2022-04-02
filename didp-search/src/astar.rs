use crate::forward_bfs;
use crate::util;
use didp_parser::expression;
use didp_parser::expression_parser;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

pub fn astar<T: variable::Numeric + Ord + fmt::Display>(
    model: &didp_parser::Model<T>,
    config: &yaml_rust::Yaml,
) -> Result<util::Solution<T>, Box<dyn Error>>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let map = match config {
        yaml_rust::Yaml::Hash(map) => map,
        _ => {
            return Err(
                util::ConfigErr::new(format!("expected Hash, but found `{:?}`", config)).into(),
            )
        }
    };
    let parameters = FxHashMap::default();
    let h_expression = match map.get(&yaml_rust::Yaml::from_str("h")) {
        Some(yaml_rust::Yaml::String(string)) => expression_parser::parse_numeric(
            string.clone(),
            &model.state_metadata,
            &model.table_registry,
            &parameters,
        )?,
        None => expression::NumericExpression::Constant(T::zero()),
        value => {
            return Err(
                util::ConfigErr::new(format!("expected String, but found `{:?}`", value)).into(),
            )
        }
    };
    let h_function = |state: &didp_parser::State, model: &didp_parser::Model<T>| {
        Some(h_expression.eval(state, &model.table_registry))
    };
    let registry_capacity = match map.get(&yaml_rust::Yaml::from_str("registry_capacity")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
        None => Some(1000000),
        value => {
            return Err(
                util::ConfigErr::new(format!("expected Integer, but found `{:?}`", value)).into(),
            )
        }
    };
    let ub = match map.get(&yaml_rust::Yaml::from_str("ub")) {
        Some(yaml_rust::Yaml::Integer(value)) => Some(T::from_integer(*value as variable::Integer)),
        Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().map_err(|e| {
            util::ConfigErr::new(format!("could not parse {} as a number: {:?}", value, e))
        })?),
        None => None,
        value => {
            return Err(
                util::ConfigErr::new(format!("expected Integer, but found `{:?}`", value)).into(),
            )
        }
    };
    match map.get(&yaml_rust::Yaml::from_str("f")) {
        Some(yaml_rust::Yaml::String(string)) => match &string[..] {
            "+" => {
                let f_function = |g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| g + h;
                Ok(forward_bfs::forward_bfs(
                    model,
                    h_function,
                    f_function,
                    ub,
                    registry_capacity,
                ))
            }
            "max" => {
                let f_function =
                    |g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| cmp::max(g, h);
                Ok(forward_bfs::forward_bfs(
                    model,
                    h_function,
                    f_function,
                    ub,
                    registry_capacity,
                ))
            }
            "min" => {
                let f_function =
                    |g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| cmp::min(g, h);
                Ok(forward_bfs::forward_bfs(
                    model,
                    h_function,
                    f_function,
                    ub,
                    registry_capacity,
                ))
            }
            op => Err(
                util::ConfigErr::new(format!("unexpected operator for f function `{}`", op)).into(),
            ),
        },
        None => {
            let f_function = |g, h, _: &didp_parser::State, _: &didp_parser::Model<T>| g + h;
            Ok(forward_bfs::forward_bfs(
                model,
                h_function,
                f_function,
                ub,
                registry_capacity,
            ))
        }
        value => {
            Err(util::ConfigErr::new(format!("expected String, but found `{:?}`", value)).into())
        }
    }
}
