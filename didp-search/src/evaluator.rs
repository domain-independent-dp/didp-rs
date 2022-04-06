use didp_parser::variable;

pub trait Evaluator<T: variable::Numeric> {
    fn eval(&self, state: &didp_parser::State, model: &didp_parser::Model<T>) -> Option<T>;
}
