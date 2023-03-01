use dypdl::variable_type::Numeric;

/// Node prioritized by g- and f-values.
pub trait PrioritizedNode<T: Numeric> {
    /// Returns the g-value.
    fn g(&self) -> T;

    /// Returns the f-value.
    fn f(&self) -> T;
}
