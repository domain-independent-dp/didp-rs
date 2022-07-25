use dypdl::variable_type;

/// A trait representing an evaluator which takes a state and returns a numeric value.
pub trait Evaluator {
    /// Returns the evaluation result.
    fn eval<T: variable_type::Numeric + Ord, S: dypdl::DPState>(
        &self,
        state: &S,
        model: &dypdl::Model,
    ) -> Option<T>;
}
