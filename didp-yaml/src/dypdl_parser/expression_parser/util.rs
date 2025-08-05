use rustc_hash::FxHashMap;
use std::error;
use std::fmt;

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
}
