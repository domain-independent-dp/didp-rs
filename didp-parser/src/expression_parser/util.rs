#[derive(Debug)]
pub enum ParseErr {
    Reason(String),
}

pub fn parse_closing(tokens: &[String]) -> Result<&[String], ParseErr> {
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    if token != ")" {
        Err(ParseErr::Reason(format!(
            "unexpected {}, expected `)`",
            token
        )))
    } else {
        Ok(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
