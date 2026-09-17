use super::{LoadedSolution, SolutionCost, SolutionError};
use dypdl::variable_type::{Element, Integer};
use dypdl::{CostType, Model, Transition};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use yaml_rust::{yaml::Hash, Yaml, YamlLoader};

fn format_error(path: &str, message: impl Into<String>) -> SolutionError {
    SolutionError::Format {
        path: path.to_owned(),
        message: message.into(),
    }
}

fn get_map<'a>(yaml: &'a Yaml, path: &str) -> Result<&'a Hash, SolutionError> {
    yaml.as_hash()
        .ok_or_else(|| format_error(path, "expected a mapping"))
}

fn check_keys(map: &Hash, allowed: &[&str], path: &str) -> Result<(), SolutionError> {
    for key in map.keys() {
        match key.as_str() {
            Some(key) if allowed.contains(&key) => (),
            Some(key) => return Err(format_error(&format!("{path}.{key}"), "unknown field")),
            None => return Err(format_error(path, "field names must be strings")),
        }
    }
    Ok(())
}

/// Loads a candidate solution from a single YAML document using an existing model.
///
/// The document must contain a `transitions` sequence. Each entry contains a
/// `name` and an optional `parameters` mapping. Names and complete parameter maps
/// must uniquely identify a forward or forward-forced transition in the model.
/// The optional `cost` field is a declared objective, not a computed one.
/// Unknown fields, ambiguous transitions, and non-finite costs are rejected.
/// This function does not check feasibility; call [`super::validate_solution`].
pub fn load_solution_from_yaml(
    model: &Model,
    yaml: &Yaml,
) -> Result<LoadedSolution, SolutionError> {
    let map = get_map(yaml, "solution")?;
    check_keys(map, &["cost", "transitions"], "solution")?;
    let cost = match map.get(&Yaml::String("cost".to_owned())) {
        None | Some(Yaml::Null) => None,
        Some(value) => Some(load_cost(value, model.cost_type)?),
    };
    let entries = map
        .get(&Yaml::String("transitions".to_owned()))
        .ok_or_else(|| format_error("transitions", "missing required field"))?
        .as_vec()
        .ok_or_else(|| format_error("transitions", "expected a sequence"))?;

    // The raw name plus an order-independent parameter map is the identity.
    // Do not use get_full_name(): its text is not an unambiguous serialization.
    let mut index = FxHashMap::<_, Vec<&Transition>>::default();
    for transition in model
        .forward_transitions
        .iter()
        .chain(&model.forward_forced_transitions)
    {
        if transition.parameter_names.len() != transition.parameter_values.len() {
            return Err(format_error(
                "model.transitions",
                format!(
                    "transition `{}` has inconsistent parameter names and values",
                    transition.name
                ),
            ));
        }
        let parameters: BTreeMap<_, _> = transition
            .parameter_names
            .iter()
            .cloned()
            .zip(transition.parameter_values.iter().copied())
            .collect();
        if parameters.len() != transition.parameter_names.len() {
            return Err(format_error(
                "model.transitions",
                format!(
                    "transition `{}` has duplicate parameter names",
                    transition.name
                ),
            ));
        }
        index
            .entry((transition.name.clone(), parameters))
            .or_default()
            .push(transition);
    }

    let mut transitions = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let path = format!("transitions[{i}]");
        let entry = get_map(entry, &path)?;
        check_keys(entry, &["name", "parameters"], &path)?;
        let name = entry
            .get(&Yaml::String("name".to_owned()))
            .and_then(Yaml::as_str)
            .ok_or_else(|| {
                format_error(
                    &format!("{path}.name"),
                    "expected a transition name (string)",
                )
            })?;
        let mut parameters = BTreeMap::new();
        if let Some(value) = entry.get(&Yaml::String("parameters".to_owned())) {
            let parameter_path = format!("{path}.parameters");
            for (key, value) in get_map(value, &parameter_path)? {
                let key = key.as_str().ok_or_else(|| {
                    format_error(&parameter_path, "parameter names must be strings")
                })?;
                let value_path = format!("{parameter_path}.{key}");
                let value = value
                    .as_i64()
                    .and_then(|value| Element::try_from(value).ok())
                    .ok_or_else(|| {
                        format_error(&value_path, "expected a non-negative element index")
                    })?;
                parameters.insert(key.to_owned(), value);
            }
        }
        let key = (name.to_owned(), parameters);
        let candidates = index.get(&key).ok_or_else(|| {
            format_error(&path, format!("no forward transition named `{name}` with parameters {:?}; check the name and all parameter names and values", key.1))
        })?;
        if candidates.len() != 1 {
            return Err(format_error(&path, format!("ambiguous transition `{name}` with parameters {:?}: {} model transitions match", key.1, candidates.len())));
        }
        transitions.push(candidates[0].clone());
    }
    Ok(LoadedSolution { cost, transitions })
}

fn load_cost(yaml: &Yaml, cost_type: CostType) -> Result<SolutionCost, SolutionError> {
    match cost_type {
        CostType::Integer => yaml
            .as_i64()
            .and_then(|value| Integer::try_from(value).ok())
            .map(SolutionCost::Integer)
            .ok_or_else(|| {
                format_error(
                    "cost",
                    "expected a 32-bit integer for this integer-cost model",
                )
            }),
        CostType::Continuous => {
            let value = match yaml {
                Yaml::Integer(value) => Some(*value as f64),
                Yaml::Real(_) => yaml.as_f64(),
                _ => None,
            }
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                format_error(
                    "cost",
                    "expected a finite number for this continuous-cost model",
                )
            })?;
            Ok(SolutionCost::Continuous(value))
        }
    }
}

/// Loads a candidate solution from YAML text. Exactly one document is required.
pub fn load_solution_from_str(
    model: &Model,
    source: &str,
) -> Result<LoadedSolution, SolutionError> {
    let documents = YamlLoader::load_from_str(source)
        .map_err(|error| format_error("solution", format!("invalid YAML: {error}")))?;
    if documents.len() != 1 {
        return Err(format_error(
            "solution",
            format!(
                "expected exactly one YAML document, found {}",
                documents.len()
            ),
        ));
    }
    load_solution_from_yaml(model, &documents[0])
}

/// Loads a candidate solution from a YAML file using an existing model.
pub fn load_solution_from_file(
    model: &Model,
    path: impl AsRef<Path>,
) -> Result<LoadedSolution, SolutionError> {
    let path = path.as_ref();
    let source = fs::read_to_string(path).map_err(|source| SolutionError::Io {
        path: path.to_owned(),
        source,
    })?;
    load_solution_from_str(model, &source)
}
