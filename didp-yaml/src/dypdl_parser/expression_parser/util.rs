use dypdl::variable_type::Element;
use dypdl::{LocalVariableData, StateFunctions, StateMetadata, TableRegistry};
use rustc_hash::FxHashMap;
use std::error;
use std::fmt;

/// Bundles the pieces of context that stay fixed for the whole of one expression parse:
/// read-only model data, the grounding parameter substitutions, and a mutable handle to the
/// model's local variable registry (grown on the fly by `bind_local_variable` whenever a
/// `reduce`/`filter` binder is parsed).
pub struct ModelData<'a> {
    /// Names, types, and indices of state variables and object types.
    pub metadata: &'a StateMetadata,
    /// Registered state functions.
    pub functions: &'a StateFunctions,
    /// Registered tables and dictionaries.
    pub registry: &'a TableRegistry,
    /// Values substituted for grounded parameters.
    pub parameters: &'a FxHashMap<String, Element>,
    /// Local variables registered while parsing higher-order expressions.
    pub local_variable_data: &'a mut LocalVariableData,
}

/// Error representing that an expression could not be parsed.
#[derive(Debug)]
pub struct ParseErr(String);

impl ParseErr {
    /// Returns a new `ParseErr` with the given message.
    pub fn new(message: String) -> ParseErr {
        ParseErr(format!("Error in parsing expression: {message}"))
    }
}

impl fmt::Display for ParseErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl error::Error for ParseErr {}

/// Runs a speculative parse attempt `f`. If `f` fails, restores `model_data.local_variable_data`
/// to its state before the attempt, so that local variables bound while parsing a discarded
/// alternative (e.g. trying to parse tokens as one expression type before falling back to
/// another) are not left behind as orphaned entries.
pub fn try_parse<'a, T>(
    model_data: &mut ModelData<'a>,
    f: impl FnOnce(&mut ModelData<'a>) -> Result<T, ParseErr>,
) -> Result<T, ParseErr> {
    let local_variable_data_backup = model_data.local_variable_data.clone();
    let result = f(model_data);

    if result.is_err() {
        *model_data.local_variable_data = local_variable_data_backup;
    }

    result
}

pub fn get_next_token_and_rest(tokens: &[String]) -> Result<(&String, &[String]), ParseErr> {
    let (item, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    Ok((item, rest))
}

pub fn parse_closing(tokens: &[String]) -> Result<&[String], ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::new("could not get token".to_string()))?;
    if token != ")" {
        Err(ParseErr::new(format!("unexpected {token}, expected `)`")))
    } else {
        Ok(rest)
    }
}

pub fn parse_parameterized_state_function_name<'a>(
    name: &str,
    tokens: &'a [String],
    parameters: &FxHashMap<String, usize>,
) -> Result<(String, &'a [String]), ParseErr> {
    let mut name = name.to_string();
    let mut xs = tokens;

    loop {
        let (next_token, rest) = xs
            .split_first()
            .ok_or_else(|| ParseErr::new("could not find closing `)`".to_string()))?;

        if next_token == ")" {
            return Ok((name, rest));
        }

        if let Some(v) = parameters.get(next_token) {
            name += &format!("_{v}");
        } else {
            name += &format!("_{next_token}");
        }

        xs = rest;
    }
}

/// Registers a fresh local variable in `model_data.local_variable_data` and binds `name` to it,
/// scoped to whatever is parsed with the returned map. The set/table expression being reduced
/// or filtered must be parsed with the *original* `local_variables` map (it cannot see the new
/// binding), while the body/condition must be parsed with the returned map.
///
/// The registered name is a synthetic `__local_var_{n}`, not `name` itself: `name` is only a
/// parse-time token used to resolve later occurrences back to this id, so the same surface name
/// can be reused across unrelated or nested binders without colliding in the model's local
/// variable registry.
pub fn bind_local_variable(
    model_data: &mut ModelData,
    name: &str,
    local_variables: &FxHashMap<String, usize>,
) -> (usize, FxHashMap<String, usize>) {
    let n = model_data.local_variable_data.number_of_variables();
    let local_variable = model_data
        .local_variable_data
        .add(format!("__local_var_{n}"))
        .expect("generated local variable name must be unique");

    let mut local_variables = local_variables.clone();
    local_variables.insert(name.to_string(), local_variable.id());

    (local_variable.id(), local_variables)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_next_token_and_rest_ok() {
        let tokens: Vec<String> = [")", "(", "+", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = get_next_token_and_rest(&tokens);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), (&tokens[0], &tokens[1..]));
    }

    #[test]
    fn get_next_token_and_rest_err() {
        let tokens: Vec<String> = [].iter().map(|x: &&str| x.to_string()).collect();
        let result = get_next_token_and_rest(&tokens);
        assert!(result.is_err());
    }

    #[test]
    fn parse_closing_ok() {
        let tokens: Vec<String> = [")", "(", "+", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_closing(&tokens);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), &tokens[1..]);
    }

    #[test]
    fn parse_closing_err() {
        let tokens: Vec<String> = ["(", "+", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_closing(&tokens);
        assert!(result.is_err());
    }

    #[test]
    fn parse_parameterized_state_function_name_ok() {
        let name = "state_function";
        let tokens: Vec<_> = ["a", "0", "b", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 1), ("b".to_string(), 2)]);
        let result = parse_parameterized_state_function_name(name, &tokens, &parameters);
        assert!(result.is_ok());
        let (name, rest) = result.unwrap();
        assert_eq!(name, "state_function_1_0_2");
        assert_eq!(rest, &[")", ")"]);
    }

    #[test]
    fn parse_parameterized_state_function_name_err() {
        let name = "state_function";
        let tokens: Vec<_> = ["a", "0", "b"].iter().map(|x| x.to_string()).collect();
        let parameters = FxHashMap::from_iter(vec![("a".to_string(), 1), ("b".to_string(), 2)]);
        let result = parse_parameterized_state_function_name(name, &tokens, &parameters);
        assert!(result.is_err());
    }

    fn generate_model_data<'a>(
        metadata: &'a StateMetadata,
        functions: &'a StateFunctions,
        registry: &'a TableRegistry,
        parameters: &'a FxHashMap<String, Element>,
        local_variable_data: &'a mut LocalVariableData,
    ) -> ModelData<'a> {
        ModelData {
            metadata,
            functions,
            registry,
            parameters,
            local_variable_data,
        }
    }

    #[test]
    fn bind_local_variable_first_binding() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();
        let mut local_variable_data = LocalVariableData::default();
        let mut model_data = generate_model_data(
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut local_variable_data,
        );

        let local_variables = FxHashMap::default();
        let (id, local_variables) = bind_local_variable(&mut model_data, "i", &local_variables);

        assert_eq!(id, 0);
        assert_eq!(
            local_variables,
            FxHashMap::from_iter([("i".to_string(), 0)])
        );
        assert_eq!(model_data.local_variable_data.number_of_variables(), 1);
        assert_eq!(
            model_data
                .local_variable_data
                .get("__local_var_0")
                .unwrap()
                .id(),
            0
        );
    }

    #[test]
    fn bind_local_variable_preserves_outer_bindings() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();
        let mut local_variable_data = LocalVariableData::default();
        let mut model_data = generate_model_data(
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut local_variable_data,
        );

        let outer_local_variables = FxHashMap::from_iter([("j".to_string(), 5)]);
        let (id, inner_local_variables) =
            bind_local_variable(&mut model_data, "i", &outer_local_variables);

        assert_eq!(id, 0);
        assert_eq!(
            inner_local_variables,
            FxHashMap::from_iter([("j".to_string(), 5), ("i".to_string(), 0)])
        );
        // The map passed in is not mutated; only the returned copy gains the new binding.
        assert_eq!(
            outer_local_variables,
            FxHashMap::from_iter([("j".to_string(), 5)])
        );
    }

    #[test]
    fn bind_local_variable_sequential_calls_produce_unique_ids() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();
        let mut local_variable_data = LocalVariableData::default();
        let mut model_data = generate_model_data(
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut local_variable_data,
        );

        let (first_id, local_variables) =
            bind_local_variable(&mut model_data, "i", &FxHashMap::default());
        let (second_id, local_variables) =
            bind_local_variable(&mut model_data, "j", &local_variables);

        assert_eq!(first_id, 0);
        assert_eq!(second_id, 1);
        assert_eq!(
            local_variables,
            FxHashMap::from_iter([("i".to_string(), 0), ("j".to_string(), 1)])
        );
        assert_eq!(model_data.local_variable_data.number_of_variables(), 2);
        assert_eq!(
            model_data
                .local_variable_data
                .get("__local_var_1")
                .unwrap()
                .id(),
            1
        );
    }

    #[test]
    fn bind_local_variable_shadowing_reuses_surface_name() {
        let metadata = StateMetadata::default();
        let functions = StateFunctions::default();
        let registry = TableRegistry::default();
        let parameters = FxHashMap::default();
        let mut local_variable_data = LocalVariableData::default();
        let mut model_data = generate_model_data(
            &metadata,
            &functions,
            &registry,
            &parameters,
            &mut local_variable_data,
        );

        let (outer_id, outer_local_variables) =
            bind_local_variable(&mut model_data, "i", &FxHashMap::default());
        let (inner_id, inner_local_variables) =
            bind_local_variable(&mut model_data, "i", &outer_local_variables);

        assert_eq!(outer_id, 0);
        assert_eq!(inner_id, 1);
        assert_ne!(outer_id, inner_id);
        // The inner scope's map shadows "i" with the new id.
        assert_eq!(
            inner_local_variables,
            FxHashMap::from_iter([("i".to_string(), 1)])
        );
        // The outer scope's own map is untouched by the nested binder.
        assert_eq!(
            outer_local_variables,
            FxHashMap::from_iter([("i".to_string(), 0)])
        );
        // Both generated local variables remain registered in the model.
        assert_eq!(model_data.local_variable_data.number_of_variables(), 2);
    }
}
