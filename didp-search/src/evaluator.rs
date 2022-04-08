use didp_parser::variable;

pub trait Evaluator<T: variable::Numeric> {
    fn eval<U: variable::Numeric, S: didp_parser::DPState>(
        &self,
        state: &S,
        model: &didp_parser::Model<U>,
    ) -> Option<T>;
}
