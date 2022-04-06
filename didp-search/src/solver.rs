use didp_parser::variable;
use std::error::Error;
use std::fmt;

pub type Solution<T> = Option<(T, Vec<didp_parser::Transition<T>>)>;

#[derive(Debug, Clone)]
pub struct ConfigErr(String);

impl ConfigErr {
    pub fn new(message: String) -> ConfigErr {
        ConfigErr(format!("Error in config: {}", message))
    }
}

impl fmt::Display for ConfigErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ConfigErr {}

pub trait Solver<T: variable::Numeric> {
    fn set_ub(&mut self, _: Option<T>) {}

    fn solve(&mut self, model: &didp_parser::Model<T>) -> Solution<T>;
}
