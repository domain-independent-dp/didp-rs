use dypdl::variable_type::Numeric;

/// A trait representing a node prioritized by g- and f-values.
pub trait PrioritizedNode<T: Numeric> {
    /// Returns the g-value.
    fn g(&self) -> T;

    /// Returns the f-value.
    fn f(&self) -> T;
}
