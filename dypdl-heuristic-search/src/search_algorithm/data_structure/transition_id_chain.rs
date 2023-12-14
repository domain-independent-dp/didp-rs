/// Trait to get a sequence of ids.
pub trait GetTransitionIds {
    /// Returns ids up to this node.
    fn ids(&self) -> Vec<(bool, usize)>;

    /// Returns the last id.
    fn last(&self) -> Option<(bool, usize)>;
}
