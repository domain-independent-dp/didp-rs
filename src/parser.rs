pub mod condition_parser;
pub mod function_parser;
pub mod numeric_parser;
pub mod set_parser;

#[derive(Debug)]
pub enum ParseErr {
    Reason(String),
}

pub fn tokenize(text: String) -> Vec<String> {
    text.replace("(", " ( ")
        .replace(")", " ) ")
        .replace("|", " | ")
        .replace("!", " ! ")
        .split_whitespace()
        .map(|x| x.to_string())
        .collect()
}

fn parse_closing(tokens: &[String]) -> Result<&[String], ParseErr> {
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
    fn tokenize_text() {
        let text = "(+ (- 5 (/ (f4 4 !s[2] e[0] 3) (max (f2 2 e[1]) n[0]))) (* r[1] (min 3 |(+ (* s[0] (- s[2] (+ s[3] 2))) (- s[1] 1))|)))".to_string();
        assert_eq!(
            tokenize(text),
            [
                "(", "+", "(", "-", "5", "(", "/", "(", "f4", "4", "!", "s[2]", "e[0]", "3", ")",
                "(", "max", "(", "f2", "2", "e[1]", ")", "n[0]", ")", ")", ")", "(", "*", "r[1]",
                "(", "min", "3", "|", "(", "+", "(", "*", "s[0]", "(", "-", "s[2]", "(", "+",
                "s[3]", "2", ")", ")", ")", "(", "-", "s[1]", "1", ")", ")", "|", ")", ")", ")",
            ]
        );
    }

    #[test]
    fn parse_closing_ok() {
        let tokens: Vec<String> = [")", "(", "+", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_closing(&tokens);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), &tokens[1..]);
    }

    #[test]
    fn parse_closing_err() {
        let tokens: Vec<String> = ["(", "+", "2", "n[0]", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_closing(&tokens);
        assert!(result.is_err());
    }
}
