use didp_parser::variable;

pub trait Evaluator<T: variable::Numeric> {
    fn eval<U: didp_parser::DPState>(&self, state: &U, model: &didp_parser::Model<T>) -> Option<T>;
}
