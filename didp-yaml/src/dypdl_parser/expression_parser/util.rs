use std::error;
use std::fmt;

/// Error representing that an expression could not be parsed.
#[derive(Debug)]
pub struct ParseErr(String);

impl ParseErr {
    /// Returns a new `ParseErr` with the given message.
    pub fn new(message: String) -> ParseErr {
        ParseErr(format!("Error in parsing expression: {}", message))
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
        Err(ParseErr::new(format!("unexpected {}, expected `)`", token)))
    } else {
        Ok(rest)
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
}
