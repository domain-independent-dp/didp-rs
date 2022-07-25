use std::error::Error;
use std::fmt;

/// A struct representing an error in modeling.
#[derive(Debug)]
pub struct ModelErr(String);

impl ModelErr {
    pub fn new(message: String) -> ModelErr {
        ModelErr(format!("Error in problem definiton: {}", message))
    }
}

impl fmt::Display for ModelErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ModelErr {}
